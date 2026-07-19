//! Runtime-filter candidate analysis and five-phase kernel emission.

use super::*;
use crate::egir::soac::filter as filter_soac;

impl KernelPlanBuilder<'_, '_> {
    pub(super) fn lower_parallel_filter(
        &mut self,
        body: crate::egir::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: BoundFilter,
    ) -> error::Result<()> {
        self.lower_bound_filter(body, kernel, recipe)
    }

    fn lower_bound_filter(
        &mut self,
        filter_entry: crate::egir::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: BoundFilter,
    ) -> error::Result<()> {
        let family = FilterKernelFamilyBuilder::new(self, filter_entry, recipe).build()?;
        family.install(kernel, &mut self.schedule)
    }
}

struct FilterKernelFamily {
    domain: schedule::KernelDomain,
    work: filter_soac::WorkBuffers,
    storage: filter_soac::RuntimeStorage<SemanticResourceRef>,
    flags: crate::egir::program::PlannedEntry,
    scan: crate::egir::program::PlannedEntry,
    combine: crate::egir::program::PlannedEntry,
    apply_offsets: crate::egir::program::PlannedEntry,
    scatter: crate::egir::program::PlannedEntry,
    scan_workgroup_width: u32,
    scan_groups: u32,
}

struct FilterKernelFamilyBuilder<'lowering, 'resources, 'effects> {
    lowering: &'lowering mut KernelPlanBuilder<'resources, 'effects>,
    entry: crate::egir::program::PlannedEntry,
    candidate: FilterCandidate,
    work: filter_soac::WorkBuffers,
    elem_ty: Type<TypeName>,
}

impl<'lowering, 'resources, 'effects> FilterKernelFamilyBuilder<'lowering, 'resources, 'effects> {
    fn new(
        lowering: &'lowering mut KernelPlanBuilder<'resources, 'effects>,
        entry: crate::egir::program::PlannedEntry,
        recipe: BoundFilter,
    ) -> Self {
        Self {
            lowering,
            entry,
            candidate: recipe.candidate,
            work: recipe.work,
            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
        }
    }

    fn build(mut self) -> error::Result<FilterKernelFamily> {
        let domain = schedule::domain_from_space(&self.candidate.space)
            .unwrap_or(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let flags = self.build_flags()?;
        let mut scan = self.build_scan()?;
        let (combine, apply_offsets) = self.build_scan_tail(&mut scan)?;
        let scatter = self.build_scatter()?;
        Ok(FilterKernelFamily {
            domain,
            work: self.work,
            storage: self.candidate.storage.runtime(),
            flags,
            scan,
            combine,
            apply_offsets,
            scatter,
            scan_workgroup_width: self.lowering.policy.reduce_phase1_width,
            scan_groups: self.lowering.policy.filter_scan_groups,
        })
    }

    fn build_flags(&self) -> error::Result<crate::egir::program::PlannedEntry> {
        use crate::interface::StorageRole;

        let mut storage = self
            .entry
            .resource_declarations
            .iter()
            .filter(|declaration| declaration.role == StorageRole::Input)
            .cloned()
            .collect::<Vec<_>>();
        storage.push(self.declaration(self.work.flags, StorageRole::Output)?);
        let spec = ProjectionSpec::unit(
            format!("{}_filter_flags", self.entry.name),
            self.entry.execution_model.clone(),
            storage,
        );
        Ok(project_kernel_body(&self.entry, spec)?)
    }

    fn build_scan(&self) -> error::Result<crate::egir::program::PlannedEntry> {
        use crate::interface::StorageRole;

        let storage = [
            (self.work.flags, StorageRole::Input),
            (self.work.offsets, StorageRole::Output),
            (self.work.block_sums, StorageRole::Output),
        ]
        .into_iter()
        .map(|(resource, role)| self.declaration(resource, role))
        .collect::<Result<Vec<_>, _>>()?;
        let spec = ProjectionSpec::unit(
            format!("{}_filter_scan", self.entry.name),
            ExecutionModel::Compute {
                local_size: (self.lowering.policy.reduce_phase1_width, 1, 1),
            },
            storage,
        );
        Ok(project_kernel_body(&self.entry, spec)?)
    }

    fn build_scan_tail(
        &mut self,
        scan: &mut crate::egir::program::PlannedEntry,
    ) -> error::Result<(
        crate::egir::program::PlannedEntry,
        crate::egir::program::PlannedEntry,
    )> {
        let zero = graph_ops::intern_u32(&mut scan.graph, 0, None);
        let add_name = format!("{}_filter_scan_add", self.entry.name);
        let add_fn = synthesize_u32_add_function(add_name.clone(), self.entry.span);
        self.lowering.schedule.define_callable(add_fn);
        let scan_scratch = ScanScratch {
            block_sums: self.work.block_sums.0,
            block_offsets: self.work.block_offsets.0,
        };
        let combine = ScanPhase2Spec {
            entry_name: scan.name.clone(),
            operator: add_name.clone(),
            elem_ty: self.elem_ty.clone(),
            source_graph: &scan.graph,
            neutral: zero,
            scratch: scan_scratch,
            total_out: Some(self.candidate.storage.length.0),
        };
        let mut combine = combine.build(self.lowering.effect_ids).map_err(|error| {
            format!(
                "failed to synthesize filter scan for `{}`: {error}",
                self.entry.name
            )
        })?;
        apply_manifest_resource_sizes(&mut combine, &self.lowering.resources)?;
        let swap_wrapper_name = format!("{}_filter_scan_add_offsets", self.entry.name);
        let swap_wrapper =
            synthesize_swap_wrapper(swap_wrapper_name, add_name, self.elem_ty.clone(), self.entry.span);
        let swap_region = self.lowering.schedule.define_callable(swap_wrapper);
        let apply_offsets = ScanPhase3Spec {
            entry_name: scan.name.clone(),
            swap_region,
            elem_ty: self.elem_ty.clone(),
            output_resource: self.work.offsets.0,
            block_offsets: self.work.block_offsets.0,
            width: self.lowering.policy.reduce_phase1_width,
        };
        let mut apply_offsets = apply_offsets.build(self.lowering.effect_ids)?;
        apply_manifest_resource_sizes(&mut apply_offsets, &self.lowering.resources)?;
        Ok((combine, apply_offsets))
    }

    fn build_scatter(&self) -> error::Result<crate::egir::program::PlannedEntry> {
        use crate::interface::StorageRole;

        let mut resources = self.entry.resource_declarations.clone();
        for declaration in &mut resources {
            if declaration.resource == self.candidate.storage.length {
                declaration.role = StorageRole::Input;
            }
        }
        resources.push(self.declaration(self.work.flags, StorageRole::Input)?);
        resources.push(self.declaration(self.work.offsets, StorageRole::Input)?);
        resources.push(self.declaration(self.work.block_offsets, StorageRole::Input)?);
        let spec = ProjectionSpec::preserving_interface(&self.entry, resources);
        Ok(project_kernel_body(&self.entry, spec)?)
    }

    fn declaration(
        &self,
        resource: SemanticResourceRef,
        role: crate::interface::StorageRole,
    ) -> error::Result<SemanticResourceDecl> {
        filter_resource_declaration(&self.lowering.resources, resource, role, &self.elem_ty)
    }
}

impl FilterKernelFamily {
    fn install(self, kernel: schedule::KernelId, schedule: &mut schedule::KernelPlan) -> error::Result<()> {
        use schedule::KernelDomain;

        let FilterKernelFamily {
            domain,
            work,
            storage,
            flags,
            scan,
            combine,
            apply_offsets,
            scatter,
            scan_workgroup_width,
            scan_groups,
        } = self;
        let scatter = schedule::KernelRecipeSpec::filter(
            scatter,
            schedule::KernelKind::FilterScatter,
            filter_soac::ParallelStage::Scatter,
            filter_soac::ParallelConfig {
                buffers: work,
                scan_workgroup_width,
            },
            storage,
        );
        let flags = schedule::PhaseSpec::filter(
            flags,
            schedule::DomainSelection::Explicit(domain.clone()),
            schedule::KernelKind::FilterFlags,
            filter_soac::ParallelStage::Flags,
            filter_soac::ParallelConfig {
                buffers: work,
                scan_workgroup_width,
            },
            storage,
        );
        // The scan runs a fixed worker grid so each worker scans a large chunk;
        // flags and scatter remain one-thread-per-input-element.
        let scan = schedule::PhaseSpec::filter(
            scan,
            schedule::DomainSelection::Explicit(KernelDomain::Fixed {
                x: scan_groups,
                y: 1,
                z: 1,
            }),
            schedule::KernelKind::FilterScan,
            filter_soac::ParallelStage::Scan,
            filter_soac::ParallelConfig {
                buffers: work,
                scan_workgroup_width,
            },
            storage,
        );
        let combine = schedule::PhaseSpec::new(
            combine,
            schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            schedule::KernelKind::FilterCombine,
        );
        let apply_offsets = schedule::PhaseSpec::new(
            apply_offsets,
            schedule::DomainSelection::Explicit(domain),
            schedule::KernelKind::FilterScan,
        );
        schedule.install_chain(
            kernel,
            schedule::KernelChainSpec::new(scatter).with_before(vec![flags, scan, combine, apply_offsets]),
        )?;
        Ok(())
    }
}

fn filter_resource_declaration(
    resources: &model::ResourceIndex<'_>,
    reference: SemanticResourceRef,
    role: crate::interface::StorageRole,
    elem_ty: &Type<TypeName>,
) -> error::Result<SemanticResourceDecl> {
    let resource = reference.0;
    let logical = resources.get(resource)?;
    Ok(SemanticResourceDecl {
        resource: reference,
        role,
        elem_ty: elem_ty.clone(),
        size: logical.size.clone(),
    })
}

#[derive(Clone)]
/// Complete graph-local runtime-filter recipe, consumed before entry mutation.
pub(super) struct FilterCandidate {
    pub semantic_id: SemanticOpId,
    pub space: SegSpace,
    storage: StoredFilterStorage,
}

#[derive(Clone, Copy)]
struct StoredFilterStorage {
    scratch: SemanticResourceRef,
    length: SemanticResourceRef,
}

impl StoredFilterStorage {
    fn runtime(self) -> filter_soac::RuntimeStorage<SemanticResourceRef> {
        filter_soac::RuntimeStorage {
            scratch: self.scratch,
            length: filter_soac::RuntimeLength::Stored(self.length),
        }
    }
}

pub(super) struct BoundFilter {
    candidate: FilterCandidate,
    work: filter_soac::WorkBuffers,
}

pub(super) fn analyze_filter_candidates(
    entry: &SemanticEntry,
) -> error::Result<Vec<(SemanticOpId, RecipeSelection<FilterCandidate>)>> {
    let mut analysis = Vec::new();
    for effect in entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects) {
        let SideEffectKind::Soac(SoacEffect(
            semantic_id,
            Soac::Filter(filter_soac::Op {
                state:
                    filter_soac::SemanticState {
                        space,
                        storage: filter_soac::Output::Runtime { scratch, length },
                    },
                ..
            }),
        )) = &effect.kind
        else {
            continue;
        };
        let semantic_id = *semantic_id;
        let selection = match length {
            filter_soac::RuntimeLength::Stored(len_out) => RecipeSelection::Parallel(FilterCandidate {
                semantic_id,
                space: space.clone(),
                storage: StoredFilterStorage {
                    scratch: *scratch,
                    length: *len_out,
                },
            }),
            filter_soac::RuntimeLength::ViewOnly => {
                RecipeSelection::Serial(FallbackReason::UnsupportedDestination)
            }
        };
        analysis.push((semantic_id, selection));
    }
    if analysis.len() > 1 {
        for (_, selection) in &mut analysis {
            *selection = RecipeSelection::Serial(FallbackReason::UnsupportedOperationShape);
        }
    }
    Ok(analysis)
}

impl BoundFilter {
    pub(super) fn bind(
        candidate: FilterCandidate,
        resources: &model::ResourceIndex<'_>,
    ) -> error::Result<Self> {
        let owner = candidate.semantic_id;
        let resource_id = |kind, slot| {
            resources.exactly_one_at(owner, kind, slot).map(|resource| SemanticResourceRef(resource.id()))
        };
        let work = filter_soac::WorkBuffers {
            flags: resource_id(CompilerResourceKind::FilterFlags, 0)?,
            offsets: resource_id(CompilerResourceKind::FilterOffsets, 1)?,
            block_sums: resource_id(CompilerResourceKind::FilterScanBlockSums, 2)?,
            block_offsets: resource_id(CompilerResourceKind::FilterScanBlockOffsets, 3)?,
        };
        Ok(Self { candidate, work })
    }
}
