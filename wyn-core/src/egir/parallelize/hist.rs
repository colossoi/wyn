//! Atomic lowering eligibility for canonical histograms.

use crate::builtins;
use crate::egir;
use crate::flow;
use crate::pipeline_descriptor;
use polytype::Type;

use crate::ast::TypeName;
use crate::egir::allocation::ResourcesAllocated;
use crate::egir::program::SemanticOpId;
use crate::egir::soac::hist;
use crate::egir::types::{
    PureOp, SegResourceAccess, SegSpace, Semantic, SkeletonTerminator, ValueId, ValueKind,
};
use crate::op::BinaryOperator;
use crate::ssa::types::{AtomicOp, ConstantValue};
use crate::types::TypeExt;
use crate::ResourceId;
use std::collections::HashSet;

use super::planning::LocatedHist;

/// A histogram proven to be expressible as one native atomic update per
/// operation and input element.
pub(super) enum HistCandidate {
    Atomic(AtomicCandidate),
    Bucket(BucketCandidate),
}

pub(super) struct AtomicCandidate {
    pub owner: SemanticOpId,
    pub space: SegSpace,
    pub operations: Vec<hist::AtomicUpdate>,
}

pub(super) struct BucketCandidate {
    pub site: egir::types::SideEffectSite,
    pub owner: SemanticOpId,
    pub space: SegSpace,
    pub bucket_count: u32,
    pub destination: ResourceId,
    pub input_resources: Vec<ResourceId>,
    pub counts: ResourceId,
    pub overflow: ResourceId,
}

pub(super) fn analyze_hist_candidate(
    program: &ResourcesAllocated,
    entry: &egir::program::PlannedEntry,
    located: LocatedHist<'_>,
) -> Option<HistCandidate> {
    let graph = &entry.graph;
    let hist::SemanticState::Segmented(space) = &located.op.state else {
        return None;
    };
    if let [operation] = located.op.form.operations.as_slice() {
        if let hist::Update::BucketInsert { counts, overflow, .. } = operation.update {
            let bucket_count = u32::try_from(constant_i32(graph, operation.shape[0])?).ok()?;
            if bucket_count == 0 || operation.shape.len() != 1 || operation.destinations.len() != 1 {
                return None;
            }
            let resources_for = |node| {
                let closure = egir::graph_ops::value_producer_closure(graph, [node]);
                entry.resources_referenced_by_nodes(graph, closure.nodes)
            };
            let resource_for = |node| {
                let resources = resources_for(node);
                let mut resources = resources.into_iter();
                let resource = resources.next()?;
                resources.next().is_none().then_some(resource)
            };
            let effect = graph.skeleton.effect(located.site);
            let captures = located.op.capture_nodes();
            let input_resources = effect
                .operands
                .iter()
                .filter_map(|operand| operand.value())
                .chain(captures.iter().copied())
                .flat_map(resources_for)
                .collect();
            return Some(HistCandidate::Bucket(BucketCandidate {
                site: located.site,
                owner: located.owner,
                space: space.clone(),
                bucket_count,
                destination: resource_for(operation.destinations[0].value())?,
                input_resources,
                counts: resource_for(counts.value())?,
                overflow: resource_for(overflow.value())?,
            }));
        }
    }
    let operations = located
        .op
        .form
        .operations
        .iter()
        .map(|operation| analyze_operation(program, graph, operation))
        .collect::<Option<Vec<_>>>()?;
    Some(HistCandidate::Atomic(AtomicCandidate {
        owner: located.owner,
        space: space.clone(),
        operations,
    }))
}

fn analyze_operation(
    program: &ResourcesAllocated,
    graph: &egir::types::EGraph<Semantic>,
    operation: &hist::HistOp,
) -> Option<hist::AtomicUpdate> {
    // The race factor is a contention estimate. Until replicated histograms
    // are available, atomic recipes are selected only for bounded contention;
    // high or dynamic estimates retain the serial fallback.
    let race_factor = constant_i32(graph, operation.race_factor)?;
    if !(1..=32).contains(&race_factor) {
        return None;
    }

    let hist::Update::Reduce { operator, .. } = &operation.update else {
        return None;
    };
    let [value_type] = operator.result_types.as_slice() else {
        return None;
    };
    let signed = match value_type {
        Type::Constructed(TypeName::Int(32), args) if args.is_empty() => true,
        Type::Constructed(TypeName::UInt(32), args) if args.is_empty() => false,
        _ => return None,
    };
    let body = operator.seg_body()?;
    let function = program.region(body.region)?;
    let direct = recognize_direct_atomic(function, signed);
    Some(direct.map_or(hist::AtomicUpdate::CompareExchange, hist::AtomicUpdate::Direct))
}

fn recognize_direct_atomic(function: &egir::program::Func<Semantic>, signed: bool) -> Option<AtomicOp> {
    if function.graph.skeleton.blocks.len() != 1 || function.graph.has_ordered_effects() {
        return None;
    }
    let SkeletonTerminator::Return(Some(result)) =
        &function.graph.skeleton.blocks[function.graph.skeleton.entry].term
    else {
        return None;
    };
    let result = result.single_value()?;
    let ValueKind::Pure { op, operands } = &function.graph.nodes[result].kind else {
        return None;
    };
    if !matches_parameter_pair(&function.graph, operands) {
        return None;
    }
    match op {
        PureOp::BinOp(BinaryOperator::Add) => Some(AtomicOp::Add),
        PureOp::BinOp(BinaryOperator::BitwiseAnd) => Some(AtomicOp::And),
        PureOp::BinOp(BinaryOperator::BitwiseOr) => Some(AtomicOp::Or),
        PureOp::BinOp(BinaryOperator::BitwiseXor) => Some(AtomicOp::Xor),
        PureOp::Intrinsic { id, .. } if *id == builtins::catalog().known().min => {
            Some(if signed { AtomicOp::SignedMin } else { AtomicOp::UnsignedMin })
        }
        PureOp::Intrinsic { id, .. } if *id == builtins::catalog().known().max => {
            Some(if signed { AtomicOp::SignedMax } else { AtomicOp::UnsignedMax })
        }
        _ => None,
    }
}
fn matches_parameter_pair(graph: &egir::types::EGraph<Semantic>, operands: &[ValueId]) -> bool {
    let [left, right] = operands else {
        return false;
    };
    matches!(
        (&graph.nodes[*left].kind, &graph.nodes[*right].kind),
        (
            ValueKind::FuncParam { parameter: left },
            ValueKind::FuncParam { parameter: right }
        ) if (left.index() == 0 && right.index() == 1)
            || (left.index() == 1 && right.index() == 0)
    )
}

fn constant_i32(graph: &egir::types::EGraph<Semantic>, node: ValueId) -> Option<i32> {
    match &graph.nodes[node].kind {
        ValueKind::Constant(ConstantValue::I32(value)) => Some(*value),
        ValueKind::Pure {
            op: PureOp::Int(value),
            operands,
        } if operands.is_empty() => value.parse().ok(),
        _ => None,
    }
}

const MAX_WORKGROUPS_PER_AXIS: u64 = 65_535;

#[cfg(test)]
fn checked_axis_workgroups(extents: &[u32], local_size: u32) -> Option<u32> {
    if local_size == 0 {
        return None;
    }
    let items = extents.iter().try_fold(1u64, |product, extent| product.checked_mul(u64::from(*extent)))?;
    let local_size = u64::from(local_size);
    let workgroups = items.checked_add(local_size - 1)?.checked_div(local_size)?;
    (workgroups <= MAX_WORKGROUPS_PER_AXIS).then(|| workgroups as u32)
}

#[derive(Clone, Copy)]
enum AxisDispatch {
    Direct(u32),
    GridStride {
        workgroups: u32,
        items: u32,
    },
}

fn plan_axis(extents: &[u32], local_size: u32) -> Option<AxisDispatch> {
    if local_size == 0 {
        return None;
    }
    let items = extents.iter().try_fold(1u64, |product, extent| product.checked_mul(u64::from(*extent)))?;
    let local = u64::from(local_size);
    let workgroups = items.checked_add(local - 1)?.checked_div(local)?;
    if workgroups <= MAX_WORKGROUPS_PER_AXIS {
        return Some(AxisDispatch::Direct(workgroups as u32));
    }
    let items = u32::try_from(items).ok()?;
    if items > i32::MAX as u32 {
        return None;
    }
    let physical_items = MAX_WORKGROUPS_PER_AXIS.checked_mul(local)?;
    Some(AxisDispatch::GridStride {
        workgroups: MAX_WORKGROUPS_PER_AXIS as u32,
        items: u32::try_from(physical_items).ok()?,
    })
}

fn fixed_array_extent(ty: &Type<TypeName>) -> Option<u32> {
    if let Some(components) = egir::types::as_soa_tuple(ty) {
        return components.first().and_then(|component| fixed_array_extent(component));
    }
    match ty.array_size()? {
        Type::Constructed(TypeName::Size(size), _) => u32::try_from(*size).ok(),
        _ => None,
    }
}

fn fixed_seg_extent(graph: &egir::types::EGraph<Semantic>, extent: &egir::types::SegExtent) -> Option<u32> {
    match extent {
        egir::types::SegExtent::Fixed(count) => Some(*count),
        egir::types::SegExtent::Value(node) => constant_i32(graph, *node)
            .and_then(|value| u32::try_from(value).ok())
            .or_else(|| fixed_array_extent(&graph.nodes[*node].ty)),
        egir::types::SegExtent::ResourceLength { view, .. } => {
            fixed_array_extent(&graph.nodes[view.value()].ty)
        }
        egir::types::SegExtent::PushConstant { .. } => None,
    }
}

/// Map a fixed row-major logical domain onto WebGPU's three dispatch axes.
/// Logical dimensions remain contiguous within each physical axis. The
/// innermost dimension is kept on x, while all legal divisions of the outer
/// prefix between y and z are considered with checked wide arithmetic.
fn bucket_dispatch_topology(
    graph: &egir::types::EGraph<Semantic>,
    space: &SegSpace,
    local_size: (u32, u32, u32),
) -> Result<Option<(hist::DispatchTopology, super::schedule::KernelDomain)>, String> {
    let dimensions =
        space.dims().iter().map(|extent| fixed_seg_extent(graph, extent)).collect::<Option<Vec<_>>>();
    let Some(dimensions) = dimensions else {
        return Ok(None);
    };
    let (local_x, local_y, local_z) = local_size;
    if dimensions.is_empty() || local_x == 0 || local_y == 0 || local_z == 0 {
        return Err("bucket_scatter requires a nonempty domain and nonzero compute local sizes".into());
    }
    let rank = dimensions.len();
    let x_start = rank - 1;
    let preferred_z_end = x_start.saturating_sub(1);
    let z_ends = std::iter::once(preferred_z_end)
        .chain((0..=x_start).filter(|candidate| *candidate != preferred_z_end));
    let mut grid_stride = None;
    for z_end in z_ends {
        let Some(x) = plan_axis(&dimensions[x_start..], local_x) else {
            continue;
        };
        let Some(y) = plan_axis(&dimensions[z_end..x_start], local_y) else {
            continue;
        };
        let Some(z) = plan_axis(&dimensions[..z_end], local_z) else {
            continue;
        };
        let mut topology = hist::DispatchTopology {
            axes: [
                hist::DispatchAxis {
                    start: x_start,
                    end: rank,
                },
                hist::DispatchAxis {
                    start: z_end,
                    end: x_start,
                },
                hist::DispatchAxis { start: 0, end: z_end },
            ],
            grid_stride: None,
        };
        let axes = [x, y, z];
        let strided = axes
            .iter()
            .enumerate()
            .filter_map(|(axis, dispatch)| match dispatch {
                AxisDispatch::GridStride { items, .. } => Some((axis, *items)),
                AxisDispatch::Direct(_) => None,
            })
            .collect::<Vec<_>>();
        if strided.len() > 1 {
            continue;
        }
        let workgroups = axes.map(|dispatch| match dispatch {
            AxisDispatch::Direct(workgroups) | AxisDispatch::GridStride { workgroups, .. } => workgroups,
        });
        let domain = super::schedule::KernelDomain::Fixed {
            x: workgroups[0],
            y: workgroups[1],
            z: workgroups[2],
        };
        if let Some((axis, items)) = strided.first().copied() {
            topology.grid_stride = Some(hist::GridStride { axis, items });
            if grid_stride.is_none() {
                grid_stride = Some((topology, domain));
            }
        } else {
            return Ok(Some((topology, domain)));
        }
    }

    if let Some(topology) = grid_stride {
        return Ok(Some(topology));
    }

    Err(format!(
        "bucket_scatter domain {dimensions:?} cannot be partitioned or strip-mined across WebGPU's x/y/z dispatch limits"
    ))
}

impl super::KernelPlanBuilder<'_, '_> {
    pub(super) fn lower_parallel_bucket(
        &mut self,
        body: egir::program::PlannedEntry,
        kernel: super::schedule::KernelId,
        candidate: BucketCandidate,
        output_projection: Option<Vec<usize>>,
    ) -> super::error::Result<()> {
        use super::{project_kernel_body, project_single_effect_body, BuiltPhase, ProjectionSpec};
        use crate::ResourceAccess;

        let local_size = match &body.execution_model {
            flow::ExecutionModel::Compute { local_size } => *local_size,
            _ => (1, 1, 1),
        };
        let fixed_dispatch = bucket_dispatch_topology(&body.graph, &candidate.space, local_size)
            .map_err(super::error::ParallelizeError::Invalid)?;
        let (insert_domain, insert_topology) = if let Some((topology, domain)) = fixed_dispatch {
            (domain, Some(topology))
        } else if candidate.space.dims().len() == 1 {
            let domain = super::schedule::domain_from_space(&candidate.space).ok_or_else(|| {
                super::error::ParallelizeError::Invalid(
                    "bucket_scatter dynamic rank-one domain is not host-dispatchable".into(),
                )
            })?;
            (domain, None)
        } else {
            return Err(super::error::ParallelizeError::Invalid(
                "bucket_scatter resource-derived multidimensional dispatch is not representable in the current descriptor"
                    .into(),
            ));
        };
        let declarations = body.resource_declarations.clone();

        let init_name = format!("{}_bucket_init", body.name);
        let init_id = self.identities.alloc_entry(init_name.clone());
        let init_body = project_single_effect_body(
            &body,
            init_id,
            candidate.site,
            ProjectionSpec::unit(init_name, body.execution_model.clone(), declarations.clone()),
        )?;
        let init = BuiltPhase::new(
            init_body,
            vec![
                SegResourceAccess::<ResourceId> {
                    resource: candidate.counts,
                    access: ResourceAccess::Write,
                },
                SegResourceAccess::<ResourceId> {
                    resource: candidate.overflow,
                    access: ResourceAccess::Write,
                },
            ],
        )
        .bucket(
            super::schedule::KernelDispatch::explicit(super::schedule::KernelDomain::Elements(
                pipeline_descriptor::DispatchLen::Fixed {
                    count: candidate.bucket_count,
                },
            )),
            candidate.owner,
            hist::ParallelStage::Init,
            None,
        );

        let insert_name = format!("{}_bucket_insert", body.name);
        let insert_id = self.identities.alloc_entry(insert_name.clone());
        let insert_body = project_single_effect_body(
            &body,
            insert_id,
            candidate.site,
            ProjectionSpec::unit(insert_name, body.execution_model.clone(), declarations.clone()),
        )?;
        let mut insert_resources = vec![
            SegResourceAccess::<ResourceId> {
                resource: candidate.destination,
                access: ResourceAccess::Write,
            },
            SegResourceAccess::<ResourceId> {
                resource: candidate.counts,
                access: ResourceAccess::ReadWrite,
            },
            SegResourceAccess::<ResourceId> {
                resource: candidate.overflow,
                access: ResourceAccess::Write,
            },
        ];
        insert_resources.extend(
            candidate.input_resources.iter().map(|resource| SegResourceAccess::<ResourceId> {
                resource: *resource,
                access: ResourceAccess::Read,
            }),
        );
        let insert = BuiltPhase::new(
            insert_body,
            super::merge_scheduled_resources(&[], &insert_resources),
        )
        .bucket(
            super::schedule::KernelDispatch::inferred(insert_domain),
            candidate.owner,
            hist::ParallelStage::Insert,
            insert_topology,
        );

        let finish_body = project_kernel_body(
            &body,
            body.id,
            ProjectionSpec::preserving_interface(&body, declarations),
        )?;
        let published_outputs = finish_body
            .outputs
            .iter()
            .filter_map(|output| output.resource.map(|resource| resource.0))
            .collect::<HashSet<_>>();
        let mut finish_resources = vec![SegResourceAccess::<ResourceId> {
            resource: candidate.overflow,
            access: ResourceAccess::Read,
        }];
        finish_resources.extend(
            published_outputs
                .into_iter()
                .filter(|resource| *resource != candidate.destination && *resource != candidate.counts)
                .map(|resource| SegResourceAccess::<ResourceId> {
                    resource,
                    access: ResourceAccess::Write,
                }),
        );
        let finish = BuiltPhase::new(
            finish_body,
            super::merge_scheduled_resources(&[], &finish_resources),
        )
        .bucket(
            super::schedule::KernelDispatch::explicit(super::schedule::KernelDomain::Fixed {
                x: 1,
                y: 1,
                z: 1,
            }),
            candidate.owner,
            hist::ParallelStage::Finish,
            None,
        )
        .with_output_projection(output_projection);

        self.schedule.replace_chain(kernel, vec![init, insert], finish, Vec::new())?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod dispatch_tests;
