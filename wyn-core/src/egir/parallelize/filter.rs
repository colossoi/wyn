//! Runtime-filter candidate analysis and five-phase kernel emission.

use super::model::{FILTER_SCAN_GROUPS, REDUCE_PHASE1_WIDTH};
use super::*;
use crate::egir;
use crate::egir::soac::filter as filter_soac;
use crate::interface;

impl KernelPlanBuilder<'_> {
    pub(super) fn lower_parallel_filter(
        &mut self,
        body: egir::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: BoundFilter,
        output_projection: Option<Vec<usize>>,
    ) -> error::Result<()> {
        let family = FilterKernelFamilyBuilder::new(self, body, recipe).build()?;
        family.install(kernel, &mut self.schedule, output_projection)
    }
}

struct FilterKernelFamily {
    domain: schedule::KernelDomain,
    work: filter_soac::WorkBuffers,
    storage: filter_soac::RuntimeStorage<SemanticResourceRef>,
    flags: BuiltPhase,
    scan: BuiltPhase,
    combine: BuiltPhase,
    apply_offsets: BuiltPhase,
    scatter: BuiltPhase,
    scan_grid: FilterScanGrid,
}

struct FilterKernelFamilyBuilder<'lowering, 'effects> {
    lowering: &'lowering mut KernelPlanBuilder<'effects>,
    entry: egir::program::PlannedEntry,
    candidate: FilterCandidate,
    work: filter_soac::WorkBuffers,
    elem_ty: Type<TypeName>,
}

impl<'lowering, 'effects> FilterKernelFamilyBuilder<'lowering, 'effects> {
    fn new(
        lowering: &'lowering mut KernelPlanBuilder<'effects>,
        entry: egir::program::PlannedEntry,
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
            scan_grid: self.candidate.scan_grid,
        })
    }

    fn build_flags(&mut self) -> error::Result<BuiltPhase> {
        use crate::interface::StorageRole;

        let mut storage = self
            .entry
            .resource_declarations
            .iter()
            .filter(|declaration| declaration.role.reads())
            .cloned()
            .map(|mut declaration| {
                declaration.role = StorageRole::Input;
                declaration
            })
            .collect::<Vec<_>>();
        storage.push(self.declaration(self.work.flags, StorageRole::Output));
        let name = format!("{}_filter_flags", self.entry.name);
        let id = self.lowering.identities.alloc_entry(name.clone());
        let spec = ProjectionSpec::unit(name, self.entry.execution_model.clone(), storage);
        Ok(BuiltPhase::from_declarations(project_kernel_body(
            &self.entry,
            id,
            spec,
        )?))
    }

    fn build_scan(&mut self) -> error::Result<BuiltPhase> {
        use crate::interface::StorageRole;

        let storage = [
            (self.work.flags, StorageRole::Input),
            (self.work.offsets, StorageRole::Output),
            (self.work.block_sums, StorageRole::Output),
        ]
        .into_iter()
        .map(|(resource, role)| self.declaration(resource, role))
        .collect();
        let name = format!("{}_filter_scan", self.entry.name);
        let id = self.lowering.identities.alloc_entry(name.clone());
        let spec = ProjectionSpec::unit(
            name,
            ExecutionModel::Compute {
                local_size: self.candidate.scan_grid.local_size(),
            },
            storage,
        );
        Ok(BuiltPhase::from_declarations(project_kernel_body(
            &self.entry,
            id,
            spec,
        )?))
    }

    fn build_scan_tail(&mut self, scan: &mut BuiltPhase) -> error::Result<(BuiltPhase, BuiltPhase)> {
        let zero = graph_ops::intern_u32(&mut scan.body.graph, 0, None);
        let add_name = format!("{}_filter_scan_add", self.entry.name);
        let span = self.entry.span;
        let add_region = self.lowering.define_callable(add_name, |region, name| {
            synthesize_u32_add_function(region, name, span)
        })?;
        let add_function = self.lowering.callable(add_region).clone();
        let scan_scratch = ScanScratch {
            block_sums: self.work.block_sums.0,
            block_offsets: self.work.block_offsets.0,
        };
        let combine = ScanPhase2Spec {
            entry_name: scan.body.name.clone(),
            operator: &add_function,
            elem_ty: self.elem_ty.clone(),
            source_graph: &scan.body.graph,
            operator_captures: &[],
            capture_inputs: &[],
            neutral: zero,
            scratch: scan_scratch,
            total_out: Some(self.candidate.storage.length.0),
            reduction_output: None,
        };
        let combine = combine
            .build(
                &mut self.lowering.identities,
                self.lowering.semantic_ids,
                self.lowering.effect_ids,
            )
            .map_err(|error| {
                format!(
                    "failed to synthesize filter scan for `{}`: {error}",
                    self.entry.name
                )
            })?;
        let swap_wrapper_name = format!("{}_filter_scan_add_offsets", self.entry.name);
        let elem_ty = self.elem_ty.clone();
        let swap_region = self.lowering.define_callable(swap_wrapper_name, |region, name| {
            synthesize_swap_wrapper(region, name, &add_function, elem_ty, Vec::new(), span)
        })?;
        let apply_offsets = ScanPhase3Spec {
            entry_name: scan.body.name.clone(),
            swap_region,
            elem_ty: self.elem_ty.clone(),
            source_graph: &scan.body.graph,
            operator_captures: Vec::new(),
            capture_inputs: Vec::new(),
            output_resource: self.work.offsets.0,
            block_offsets: self.work.block_offsets.0,
            width: self.candidate.scan_grid.workgroup_width(),
            post: None,
        };
        let apply_offsets = apply_offsets.build(
            &mut self.lowering.identities,
            self.lowering.semantic_ids,
            self.lowering.effect_ids,
        )?;
        Ok((combine, apply_offsets))
    }

    fn build_scatter(&self) -> error::Result<BuiltPhase> {
        use crate::interface::StorageRole;

        let mut resources = self.entry.resource_declarations.clone();
        for declaration in &mut resources {
            if declaration.resource == self.candidate.storage.length {
                declaration.role = StorageRole::Input;
            }
        }
        resources.push(self.declaration(self.work.flags, StorageRole::Input));
        resources.push(self.declaration(self.work.offsets, StorageRole::Input));
        resources.push(self.declaration(self.work.block_offsets, StorageRole::Input));
        let spec = ProjectionSpec::preserving_interface(&self.entry, resources);
        Ok(BuiltPhase::from_declarations(project_kernel_body(
            &self.entry,
            self.entry.id,
            spec,
        )?))
    }

    fn declaration(
        &self,
        resource: SemanticResourceRef,
        role: interface::StorageRole,
    ) -> SemanticResourceDecl {
        SemanticResourceDecl { resource, role }
    }
}

impl FilterKernelFamily {
    fn install(
        self,
        kernel: schedule::KernelId,
        schedule: &mut schedule::KernelPlan,
        output_projection: Option<Vec<usize>>,
    ) -> error::Result<()> {
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
            scan_grid,
        } = self;
        let scan_workgroup_width = scan_grid.workgroup_width();
        let scan_dispatch = schedule::KernelDispatch::explicit(scan_grid.domain());
        let scatter = scatter
            .filter(
                schedule::KernelDispatch::inferred(domain.clone()),
                filter_soac::ParallelStage::Scatter,
                filter_soac::ParallelConfig {
                    buffers: work,
                    scan_workgroup_width,
                },
                storage,
            )
            .with_output_projection(output_projection);
        let flags = flags.filter(
            schedule::KernelDispatch::explicit(domain.clone()),
            filter_soac::ParallelStage::Flags,
            filter_soac::ParallelConfig {
                buffers: work,
                scan_workgroup_width,
            },
            storage,
        );
        // The scan runs a fixed worker grid so each worker scans a large chunk;
        // flags and scatter remain one-thread-per-input-element.
        let scan = scan.filter(
            scan_dispatch.clone(),
            filter_soac::ParallelStage::Scan,
            filter_soac::ParallelConfig {
                buffers: work,
                scan_workgroup_width,
            },
            storage,
        );
        let combine = combine.compute(
            schedule::KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            "filter_combine",
        );
        let apply_offsets = apply_offsets.compute(scan_dispatch, "filter_apply_offsets");
        schedule.replace_chain(
            kernel,
            vec![flags, scan, combine, apply_offsets],
            scatter,
            Vec::new(),
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct FilterScanGrid {
    workgroup_width: u32,
    workgroups_x: u32,
}

impl FilterScanGrid {
    fn workgroup_width(self) -> u32 {
        self.workgroup_width
    }

    fn worker_count(self) -> u32 {
        self.workgroup_width * self.workgroups_x
    }

    fn local_size(self) -> (u32, u32, u32) {
        (self.workgroup_width, 1, 1)
    }

    fn domain(self) -> schedule::KernelDomain {
        schedule::KernelDomain::Fixed {
            x: self.workgroups_x,
            y: 1,
            z: 1,
        }
    }
}

#[derive(Clone)]
/// Complete graph-local runtime-filter recipe, consumed before entry mutation.
pub(super) struct FilterCandidate {
    pub semantic_id: SemanticOpId,
    pub space: SegSpace,
    storage: StoredFilterStorage,
    scan_grid: FilterScanGrid,
}

impl FilterCandidate {
    pub(super) fn scan_worker_count(&self) -> u32 {
        self.scan_grid.worker_count()
    }
}

#[derive(Clone, Copy)]
struct StoredFilterStorage {
    data: SemanticResourceRef,
    length: SemanticResourceRef,
}

impl StoredFilterStorage {
    fn runtime(self) -> filter_soac::RuntimeStorage<SemanticResourceRef> {
        filter_soac::RuntimeStorage {
            data: self.data,
            length: self.length,
        }
    }
}

pub(super) struct BoundFilter {
    candidate: FilterCandidate,
    work: filter_soac::WorkBuffers,
}

pub(super) fn analyze_filter_candidate(
    entry: &egir::program::AllocatedEntry,
    site: SideEffectSite,
) -> Option<CandidateSelection<FilterCandidate>> {
    let SideEffectKind::Soac(SoacEffect(
        semantic_id,
        Soac::Filter(filter_soac::Op {
            state:
                filter_soac::SemanticState {
                    space,
                    output: filter_soac::Output::Runtime(runtime),
                    ..
                },
            ..
        }),
    )) = &entry.graph.skeleton.effect(site).kind
    else {
        return None;
    };
    Some(match (runtime.backing, runtime.length) {
        (filter_soac::RuntimeBacking::Bound(data), filter_soac::RuntimeLength::Stored(length)) => {
            CandidateSelection::Selected(FilterCandidate {
                semantic_id: *semantic_id,
                space: space.clone(),
                storage: StoredFilterStorage { data, length },
                scan_grid: FilterScanGrid {
                    workgroup_width: REDUCE_PHASE1_WIDTH,
                    workgroups_x: FILTER_SCAN_GROUPS,
                },
            })
        }
        _ => CandidateSelection::Fallback,
    })
}

impl BoundFilter {
    pub(super) fn bind(candidate: FilterCandidate, resources: &super::planning::ScratchBindings) -> Self {
        let owner = candidate.semantic_id;
        let resource_id = |kind, slot| SemanticResourceRef(resources.id(owner, kind, slot));
        let work = filter_soac::WorkBuffers {
            flags: resource_id(CompilerResourceKind::FilterFlags, 0),
            offsets: resource_id(CompilerResourceKind::FilterOffsets, 1),
            block_sums: resource_id(CompilerResourceKind::FilterScanBlockSums, 2),
            block_offsets: resource_id(CompilerResourceKind::FilterScanBlockOffsets, 3),
        };
        Self { candidate, work }
    }
}
