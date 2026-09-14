//! Runtime-filter candidate analysis and five-phase kernel emission.

use super::model::{FILTER_SCAN_GROUPS, REDUCE_PHASE1_WIDTH};
use super::planning::{OperationRef, ScratchRef};
use super::*;
use crate::egir;
use crate::egir::soac::filter as filter_soac;
use crate::egir::soac::SegmentedMetadata;
use crate::egir::types::SegExtent;
use crate::interface;

impl KernelPlanBuilder<'_> {
    pub(super) fn lower_parallel_filter(
        &mut self,
        body: egir::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: FilterRecipe<ResourceId>,
        output_projection: Option<Vec<usize>>,
    ) -> ParallelizeResult<schedule::PreparedRecipe> {
        let family = FilterKernelFamilyBuilder::new(self, body, recipe)?.build()?;
        let ids = [
            self.schedule.allocate_kernel(),
            self.schedule.allocate_kernel(),
            self.schedule.allocate_kernel(),
            self.schedule.allocate_kernel(),
        ];
        family.prepare(kernel, ids, output_projection)
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
    candidate: FilterContext,
    work: filter_soac::WorkBuffers,
    elem_ty: Type<TypeName>,
}

impl<'lowering, 'effects> FilterKernelFamilyBuilder<'lowering, 'effects> {
    fn new(
        lowering: &'lowering mut KernelPlanBuilder<'effects>,
        entry: egir::program::PlannedEntry,
        recipe: FilterRecipe<ResourceId>,
    ) -> ParallelizeResult<Self> {
        let op = recipe.operation.filter(&entry)?;
        let filter_soac::Output::Runtime(runtime) = &op.state.output else {
            return Err("filter recipe has no runtime output".into());
        };
        let (filter_soac::RuntimeBacking::Bound(data), filter_soac::RuntimeLength::Stored(length)) =
            (&runtime.backing, &runtime.length)
        else {
            return Err("filter recipe has no resident output".into());
        };
        let candidate = FilterContext {
            space: op.state.segment.space.clone(),
            storage: filter_soac::RuntimeStorage {
                data: *data,
                length: *length,
            },
            scan_grid: FilterScanGrid {
                workgroup_width: REDUCE_PHASE1_WIDTH,
                workgroups_x: FILTER_SCAN_GROUPS,
            },
        };
        let work = recipe.work;
        Ok(Self {
            lowering,
            entry,
            candidate,
            work: filter_soac::WorkBuffers {
                flags: SemanticResourceRef(work.flags),
                offsets: SemanticResourceRef(work.offsets),
                block_sums: SemanticResourceRef(work.block_sums),
                block_offsets: SemanticResourceRef(work.block_offsets),
            },
            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
        })
    }

    fn build(mut self) -> ParallelizeResult<FilterKernelFamily> {
        let domain = schedule::domain_from_space(&self.candidate.space)
            .unwrap_or(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let flags = self.build_flags()?;
        let mut scan = self.build_scan()?;
        let (combine, apply_offsets) = self.build_scan_tail(&mut scan)?;
        let scatter = self.build_scatter()?;
        Ok(FilterKernelFamily {
            domain,
            work: self.work,
            storage: self.candidate.storage,
            flags,
            scan,
            combine,
            apply_offsets,
            scatter,
            scan_grid: self.candidate.scan_grid,
        })
    }

    fn build_flags(&mut self) -> ParallelizeResult<BuiltPhase> {
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

    fn build_scan(&mut self) -> ParallelizeResult<BuiltPhase> {
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

    fn build_scan_tail(&mut self, scan: &mut BuiltPhase) -> ParallelizeResult<(BuiltPhase, BuiltPhase)> {
        let zero = graph_ops::intern_u32(&mut scan.body.graph, 0, None);
        let add_name = format!("{}_filter_scan_add", self.entry.name);
        let span = self.entry.span;
        let add_region = self.lowering.define_callable(add_name, |region, name| {
            synthesize_u32_add_function(region, name, span)
        })?;
        let add_function = self.lowering.callable(add_region)?.clone();
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

    fn build_scatter(&self) -> ParallelizeResult<BuiltPhase> {
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
    fn prepare(
        self,
        kernel: schedule::KernelId,
        ids: [schedule::KernelId; 4],
        output_projection: Option<Vec<usize>>,
    ) -> ParallelizeResult<schedule::PreparedRecipe> {
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
        Ok(schedule::PreparedRecipe::sequence(
            vec![
                (ids[0], flags),
                (ids[1], scan),
                (ids[2], combine),
                (ids[3], apply_offsets),
                (kernel, scatter),
            ],
            kernel,
        )?)
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

struct FilterContext {
    space: SegSpace,
    storage: filter_soac::RuntimeStorage<SemanticResourceRef>,
    scan_grid: FilterScanGrid,
}

#[derive(Debug)]
pub struct FilterRecipe<R> {
    pub operation: OperationRef,
    pub work: filter_soac::WorkBuffers<R>,
}

pub(super) fn construct_filter_recipe(
    entry: &egir::program::AllocatedEntry,
    site: SideEffectSite,
) -> Option<FilterRecipe<ScratchRef>> {
    let SideEffectKind::Soac(SoacEffect(
        semantic_id,
        Soac::Filter(filter_soac::Op {
            state:
                filter_soac::SemanticState {
                    segment: SegmentedMetadata { space, .. },
                    output: filter_soac::Output::Runtime(runtime),
                    ..
                },
            ..
        }),
    )) = &entry.graph.skeleton.effect(site).kind
    else {
        return None;
    };
    let (filter_soac::RuntimeBacking::Bound(_), filter_soac::RuntimeLength::Stored(_)) =
        (runtime.backing, runtime.length)
    else {
        return None;
    };
    let element_size = match space.dims() {
        [SegExtent::Fixed(count)] => egir::program::LogicalSize::FixedBytes(u64::from(*count) * 4),
        [SegExtent::ResourceLength {
            resource, elem_bytes, ..
        }] => egir::program::LogicalSize::LikeResource {
            resource: resource.0,
            elem_bytes: 4,
            src_elem_bytes: *elem_bytes,
        },
        _ => egir::program::LogicalSize::SameAsDispatch { elem_bytes: 4 },
    };
    let workers =
        egir::program::LogicalSize::FixedBytes(u64::from(REDUCE_PHASE1_WIDTH * FILTER_SCAN_GROUPS) * 4);
    let scratch = |kind, slot, size| {
        ScratchRef::new(
            *semantic_id,
            kind,
            slot,
            Type::Constructed(TypeName::UInt(32), vec![]),
            size,
        )
    };
    Some(FilterRecipe {
        operation: OperationRef {
            site,
            owner: *semantic_id,
        },
        work: filter_soac::WorkBuffers {
            flags: scratch(CompilerResourceKind::FilterFlags, 0, element_size.clone()),
            offsets: scratch(CompilerResourceKind::FilterOffsets, 1, element_size),
            block_sums: scratch(CompilerResourceKind::FilterScanBlockSums, 2, workers.clone()),
            block_offsets: scratch(CompilerResourceKind::FilterScanBlockOffsets, 3, workers),
        },
    })
}
