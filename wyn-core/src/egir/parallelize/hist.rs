//! Atomic lowering eligibility for canonical histograms.

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::allocation::ResourcesAllocated;
use crate::egir::program::SemanticOpId;
use crate::egir::soac::hist;
use crate::egir::types::{ENode, NodeId, PureOp, SegSpace, Semantic, SkeletonTerminator};
use crate::op::BinaryOperator;
use crate::ssa::types::{AtomicOp, ConstantValue};
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
    pub site: crate::egir::types::SideEffectSite,
    pub owner: SemanticOpId,
    pub space: SegSpace,
    pub bucket_count: u32,
    pub destination: crate::ResourceId,
    pub input_resources: Vec<crate::ResourceId>,
    pub counts: crate::ResourceId,
    pub overflow: crate::ResourceId,
}

pub(super) fn analyze_hist_candidate(
    program: &ResourcesAllocated,
    entry: &crate::egir::program::PlannedEntry,
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
                let closure = crate::egir::graph_ops::value_producer_closure(graph, [node]);
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
                .operand_nodes
                .iter()
                .chain(captures.iter())
                .flat_map(|node| resources_for(*node))
                .collect();
            return Some(HistCandidate::Bucket(BucketCandidate {
                site: located.site,
                owner: located.owner,
                space: space.clone(),
                bucket_count,
                destination: resource_for(operation.destinations[0])?,
                input_resources,
                counts: resource_for(counts)?,
                overflow: resource_for(overflow)?,
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
    graph: &crate::egir::types::EGraph<Semantic>,
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

fn recognize_direct_atomic(
    function: &crate::egir::program::SemanticFunc,
    signed: bool,
) -> Option<AtomicOp> {
    if function.graph.skeleton.blocks.len() != 1
        || function.graph.skeleton.blocks.values().any(|block| !block.side_effects.is_empty())
    {
        return None;
    }
    let SkeletonTerminator::Return(Some(result)) =
        function.graph.skeleton.blocks[function.graph.skeleton.entry].term
    else {
        return None;
    };
    let ENode::Pure { op, operands } = &function.graph.nodes[result].kind else {
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
        PureOp::Intrinsic { id, .. } if *id == crate::builtins::catalog().known().min => {
            Some(if signed { AtomicOp::SignedMin } else { AtomicOp::UnsignedMin })
        }
        PureOp::Intrinsic { id, .. } if *id == crate::builtins::catalog().known().max => {
            Some(if signed { AtomicOp::SignedMax } else { AtomicOp::UnsignedMax })
        }
        _ => None,
    }
}
fn matches_parameter_pair(graph: &crate::egir::types::EGraph<Semantic>, operands: &[NodeId]) -> bool {
    let [left, right] = operands else {
        return false;
    };
    matches!(
        (&graph.nodes[*left].kind, &graph.nodes[*right].kind),
        (ENode::FuncParam { index: 0 }, ENode::FuncParam { index: 1 })
            | (ENode::FuncParam { index: 1 }, ENode::FuncParam { index: 0 })
    )
}

fn constant_i32(graph: &crate::egir::types::EGraph<Semantic>, node: NodeId) -> Option<i32> {
    match &graph.nodes[node].kind {
        ENode::Constant(ConstantValue::I32(value)) => Some(*value),
        ENode::Pure {
            op: PureOp::Int(value),
            operands,
        } if operands.is_empty() => value.parse().ok(),
        _ => None,
    }
}

/// Map a fixed ranked row-major domain onto WebGPU's three dispatch axes.
/// The innermost item dimension uses x, the next dimension uses y, and any
/// remaining outer prefix is flattened onto z.
fn tiled_bucket_domain(
    space: &SegSpace,
    local_size: (u32, u32, u32),
) -> Option<super::schedule::KernelDomain> {
    const MAX_WORKGROUPS_PER_AXIS: u32 = 65_535;
    let dimensions = space
        .dims()
        .iter()
        .map(|extent| match extent {
            crate::egir::types::SegExtent::Fixed(count) => Some(*count),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let (local_x, local_y, local_z) = local_size;
    if dimensions.len() < 2 || local_x == 0 || local_y == 0 || local_z == 0 {
        return None;
    }
    let x_items = *dimensions.last()?;
    let y = dimensions[dimensions.len() - 2];
    let z = dimensions[..dimensions.len() - 2]
        .iter()
        .try_fold(1u32, |product, count| product.checked_mul(*count))?;
    let workgroups = |items: u32, local: u32| items.checked_add(local - 1)?.checked_div(local);
    let x = workgroups(x_items, local_x)?;
    let y = workgroups(y, local_y)?;
    let z = workgroups(z, local_z)?;
    (x <= MAX_WORKGROUPS_PER_AXIS && y <= MAX_WORKGROUPS_PER_AXIS && z <= MAX_WORKGROUPS_PER_AXIS)
        .then_some(super::schedule::KernelDomain::Fixed { x, y, z })
}

impl super::KernelPlanBuilder<'_, '_> {
    pub(super) fn lower_parallel_bucket(
        &mut self,
        body: crate::egir::program::PlannedEntry,
        kernel: super::schedule::KernelId,
        candidate: BucketCandidate,
        output_projection: Option<Vec<usize>>,
    ) -> super::error::Result<()> {
        use super::{project_kernel_body, project_single_effect_body, BuiltPhase, ProjectionSpec};
        use crate::ResourceAccess;

        let local_size = match &body.execution_model {
            crate::flow::ExecutionModel::Compute { local_size } => *local_size,
            _ => (1, 1, 1),
        };
        let tiled_domain = tiled_bucket_domain(&candidate.space, local_size);
        let insert_domain = tiled_domain
            .clone()
            .or_else(|| super::schedule::domain_from_space(&candidate.space))
            .unwrap_or(super::schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let insert_stage = if tiled_domain.is_some() {
            hist::ParallelStage::InsertTiled
        } else {
            hist::ParallelStage::Insert
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
                super::schedule::ScheduledResource {
                    resource: candidate.counts,
                    access: ResourceAccess::Write,
                },
                super::schedule::ScheduledResource {
                    resource: candidate.overflow,
                    access: ResourceAccess::Write,
                },
            ],
        )
        .bucket(
            super::schedule::KernelDispatch::explicit(super::schedule::KernelDomain::Elements(
                crate::pipeline_descriptor::DispatchLen::Fixed {
                    count: candidate.bucket_count,
                },
            )),
            candidate.owner,
            hist::ParallelStage::Init,
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
            super::schedule::ScheduledResource {
                resource: candidate.destination,
                access: ResourceAccess::Write,
            },
            super::schedule::ScheduledResource {
                resource: candidate.counts,
                access: ResourceAccess::ReadWrite,
            },
            super::schedule::ScheduledResource {
                resource: candidate.overflow,
                access: ResourceAccess::Write,
            },
        ];
        insert_resources.extend(candidate.input_resources.iter().map(|resource| {
            super::schedule::ScheduledResource {
                resource: *resource,
                access: ResourceAccess::Read,
            }
        }));
        let insert = BuiltPhase::new(
            insert_body,
            super::merge_scheduled_resources(&[], &insert_resources),
        )
        .bucket(
            super::schedule::KernelDispatch::inferred(insert_domain),
            candidate.owner,
            insert_stage,
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
        let mut finish_resources = vec![super::schedule::ScheduledResource {
            resource: candidate.overflow,
            access: ResourceAccess::Read,
        }];
        finish_resources.extend(
            published_outputs
                .into_iter()
                .filter(|resource| *resource != candidate.destination && *resource != candidate.counts)
                .map(|resource| super::schedule::ScheduledResource {
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
        )
        .with_output_projection(output_projection);

        self.schedule.replace_chain(kernel, vec![init, insert], finish, Vec::new())?;
        Ok(())
    }
}
