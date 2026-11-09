//! Expand scheduled phases straight into SSA instructions and structured loops.

use super::{error, Body, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::egglog::query::Query;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

mod contribution;
mod coordinates;
mod loops;
mod maps;
mod screma;
mod shapes;
mod updates;
use screma::screma;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_width: u32,
) -> Result<(), OptimizeError> {
    if let Some(stage) = body.compiler.facts.constructor("Stage", (phase_owner, phase)) {
        if let Some(shape) = Query(&body.compiler.program.graph).row("ShapePhase", |row| row[0] == stage)? {
            return shapes::emit(body, scope, &shape[1..]);
        }
    }
    if matches!(phase, "loop_enter" | "loop_test" | "loop_step" | "loop_exit") {
        return loops::emit(body, scope, phase_owner, phase, phase_width);
    }
    if matches!(phase, "scalar" | "view") {
        let Some(context) = body.compiler.facts.dispatch_context(phase_owner) else {
            return Err(error("scalar kernel has no selected context"));
        };
        body.context = context;
    }
    if phase == "view" {
        let source = body.compiler.facts.source(phase_owner).ok_or_else(|| error("view source missing"))?;
        let value = body.value(scope, source)?;
        let length = body.length(value.clone())?;
        if body.compiler.facts.loops(source).is_some() {
            // Published views start at zero even when the loop finishes with a slice.
            let storage = body
                .compiler
                .facts
                .lookup("ArrayStorage", (source,))
                .ok_or_else(|| error("loop storage missing"))?;
            let (binding, element, _) =
                body.compiler.bindings.buffer(storage)?.ok_or_else(|| error("loop allocation missing"))?;
            let element = body.compiler.facts.physical_type(element, true)?;
            let output = body.op(
                OpTag::StorageView(PureViewSource::Storage(binding)),
                vec![Body::number(0), length.clone()],
                Body::view_type(&element, crate::types::buffer_tag(binding)),
            )?;
            body.copy_array(output, value, length.clone(), &[])?;
        }
        let slot = body
            .compiler
            .facts
            .constructor("ViewLength", (source,))
            .ok_or_else(|| error("view length slot missing"))?;
        let slot = body.resource(scope, slot, 2)?;
        return body.store(slot, Body::number(0), length);
    }
    if phase == "scalar" {
        let operations = body.compiler.facts.scalar_group(phase_owner);
        for operation in operations {
            let Some(source) = body.compiler.facts.source(operation) else {
                return Err(error("scalar dispatch source missing"));
            };
            let value = body.value(scope, source)?;
            if let Some(output) = body.slot(scope, operation, "scalar", 0, 2)? {
                let zero = body.literal(0, &types::i32())?;
                body.store(output, zero, value)?;
            }
        }
        return Ok(());
    }
    let Some(plan) = body.compiler.facts.group(phase_owner) else {
        return Err(error("kernel has no fusion plan"));
    };
    if matches!(phase, "ordered" | "local") {
        for prerequisite in body.compiler.facts.set("LocalPrerequisites", (phase_owner,)) {
            body.value(scope, prerequisite)?;
        }
    }
    let results = body.compiler.facts.array_outputs(plan)?;
    match phase {
        "elements" => maps::emit(body, scope, phase_owner, phase, phase_width, plan, &results),
        "initialize" => updates::initialize(body, scope, phase_owner, phase, phase_width),
        "ordered" | "local" => {
            updates::initialize(body, scope, phase_owner, phase, phase_width)?;
            if body.compiler.facts.accumulators(plan)?.is_empty() {
                maps::emit(body, scope, phase_owner, phase, phase_width, plan, &results)
            } else {
                screma(body, scope, phase_owner, phase, phase_width, plan, &results)
            }
        }
        "chunks" | "combine" | "offsets" => {
            screma(body, scope, phase_owner, phase, phase_width, plan, &results)
        }
        _ => Err(error(format!("scheduled {} kernel is not implemented", phase))),
    }
}

fn iteration(
    body: &mut Body<'_, '_, '_>,
    phase: &str,
    width: u32,
) -> Result<(Typed, Typed), OptimizeError> {
    match phase {
        "ordered" => Ok((Body::number(0), Body::number(1))),
        "local" => Ok((body.local_id()?, Body::number(width))),
        _ => invocation(body, width),
    }
}

pub(super) fn invocation(body: &mut Body<'_, '_, '_>, width: u32) -> Result<(Typed, Typed), OptimizeError> {
    let (group, groups) = workgroup(body)?;
    let width = Body::number(width);
    let start = body.binary(BinaryOperator::Multiply, group, width.clone())?;
    let lane = body.local_id()?;
    let start = body.binary(BinaryOperator::Add, start, lane)?;
    let step = body.binary(BinaryOperator::Multiply, groups, width)?;
    Ok((start, step))
}

fn workgroup(body: &mut Body<'_, '_, '_>) -> Result<(Typed, Typed), OptimizeError> {
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let coordinates = body.op(
        OpTag::Intrinsic {
            id: catalog().known().workgroup_id,
            overload_idx: 0,
        },
        vec![],
        types::vec(3, uint.clone()),
    )?;
    let mut start = body.field(coordinates.clone(), 0)?;
    if let Some((x, y, z)) = body.grid {
        for (axis, stride) in [(1, u64::from(x)), (2, u64::from(x) * u64::from(y))] {
            let coordinate = body.field(coordinates.clone(), axis)?;
            let stride = body.literal((stride) as i64, &uint)?;
            let offset = body.binary(BinaryOperator::Multiply, coordinate, stride)?;
            start = body.binary(BinaryOperator::Add, start, offset)?;
        }
        let groups = body.literal((u64::from(x) * u64::from(y) * u64::from(z)) as i64, &uint)?;
        return Ok((start, groups));
    }
    let step = body.op(
        OpTag::Intrinsic {
            id: catalog().known().num_workgroups,
            overload_idx: 0,
        },
        vec![],
        uint,
    )?;
    Ok((start, step))
}

pub(super) fn element(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    plan: Option<Value>,
    source: Value,
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
) -> Result<Typed, OptimizeError> {
    if indices.is_empty() {
        return body.value(scope, source);
    }
    if let Some(value) = cache.get(&source) {
        return Ok(value.clone());
    }
    let mapping = plan.zip(body.compiler.facts.operation(source)).is_some_and(|(plan, op)| {
        body.compiler.facts.contains("PlanMember", (plan, op))
            && body
                .compiler
                .facts
                .lookup("PlanMapping", (op,))
                .is_some_and(|flag| body.compiler.program.graph.value_to_base::<bool>(flag))
    });
    let traversal = if mapping {
        Some(
            Query(&body.compiler.program.graph)
                .required("ElementTraversal", (source, indices.len() as i64))?,
        )
    } else if Query(&body.compiler.program.graph).flag("ExprImportDispatchedSource", (source,))? {
        None
    } else {
        body.compiler.facts.lookup("StreamTraversal", (source, indices.len() as i64))
    };
    let value = body.invocation_local(|body| {
        if let Some(traversal) = traversal {
            emit_traversal(
                body,
                scope,
                plan,
                traversal,
                indices,
                cache,
                &|body, scope, leaf| leaf.value(body, scope),
            )
        } else {
            let mut value = body.value(scope, source)?;
            for index in indices {
                value = body.index(value, index.clone())?;
            }
            Ok(value)
        }
    })?;
    cache.insert(source, value.clone());
    Ok(value)
}

enum TraversalLeaf {
    Source(Value),
    Value(Typed),
    Result {
        source: Value,
        owner: Value,
        port: Value,
    },
}

impl TraversalLeaf {
    fn value(self, body: &mut Body<'_, '_, '_>, scope: Value) -> Result<Typed, OptimizeError> {
        match self {
            Self::Source(source) | Self::Result { source, .. } => body.value(scope, source),
            Self::Value(value) => Ok(value),
        }
    }
    fn length(self, body: &mut Body<'_, '_, '_>, scope: Value) -> Result<Typed, OptimizeError> {
        match self {
            Self::Result { source, owner, .. } => {
                let plan = Query(&body.compiler.program.graph).required("SelectedPlan", (owner,))?;
                if Query(&body.compiler.program.graph).flag("RoutedArray", (plan, source))? {
                    let Some(route) = body.compiler.facts.record("PlanOutputRoute", "OutputAt", (plan,))
                    else {
                        return Err(error("routed array has no output coordinate map"));
                    };
                    screma(body, scope, owner, "measure", 1, plan, &[])?;
                    let Some(count) = body.values.get(&route[0]) else {
                        return Err(error("fold did not produce its result extent"));
                    };
                    Ok(count.clone())
                } else {
                    body.source_length(scope, source)
                }
            }
            Self::Source(source) => body.source_length(scope, source),
            Self::Value(value) => body.length(value),
        }
    }
}

fn emit_traversal(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    plan: Option<Value>,
    traversal: Value,
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
    emit: &dyn Fn(&mut Body<'_, '_, '_>, Value, TraversalLeaf) -> Result<Typed, OptimizeError>,
) -> Result<Typed, OptimizeError> {
    if let Some(fields) = body.compiler.facts.enode("TraversalElement", traversal) {
        if body.compiler.facts.integer(fields[1]) != indices.len() as i64 {
            return Err(error("element traversal rank differs from its domain"));
        }
        let value = element(body, scope, plan, fields[0], indices, cache)?;
        return emit(body, scope, TraversalLeaf::Value(value));
    }
    if let Some(fields) = body.compiler.facts.enode("TraversalField", traversal) {
        let field = body.compiler.facts.integer(fields[0]) as usize;
        return emit_traversal(
            body,
            scope,
            plan,
            fields[1],
            indices,
            cache,
            &|body, scope, leaf| {
                let value = leaf.value(body, scope)?;
                let value = body.field(value, field)?;
                emit(body, scope, TraversalLeaf::Value(value))
            },
        );
    }
    if let Some(fields) = body.compiler.facts.enode("TraversalReindex", traversal) {
        let formulas = body.compiler.facts.vector(fields[1])?;
        if formulas.len() > indices.len() {
            return Err(error("coordinate transform exceeds traversal rank"));
        }
        let arguments = body.tuple(indices.to_vec())?;
        let mut mapped = indices.to_vec();
        for (index, formula) in formulas.into_iter().enumerate() {
            mapped[index] = body.formula(scope, formula, &arguments, &[])?;
        }
        return emit_traversal(
            body,
            scope,
            plan,
            fields[0],
            &mapped,
            &mut LookupMap::default(),
            emit,
        );
    }
    if let Some(fields) = body.compiler.facts.enode("TraversalIndex", traversal) {
        if body.compiler.facts.integer(fields[1]) != indices.len() as i64 {
            return Err(error("indexed traversal rank differs from its domain"));
        }
        let mut value = body.value(scope, fields[0])?;
        for index in indices {
            value = body.index(value, index.clone())?;
        }
        return emit(body, scope, TraversalLeaf::Value(value));
    }
    if let Some(fields) = body.compiler.facts.enode("TraversalChoice", traversal) {
        let condition = body.value(scope, fields[0])?;
        return body.branch(
            scope,
            condition,
            |body| {
                body.in_region(scope, fields[1], vec![], |body, _| {
                    emit_traversal(
                        body,
                        fields[1],
                        plan,
                        fields[2],
                        indices,
                        &mut cache.clone(),
                        emit,
                    )
                })
            },
            |body| {
                body.in_region(scope, fields[3], vec![], |body, _| {
                    emit_traversal(
                        body,
                        fields[3],
                        plan,
                        fields[4],
                        indices,
                        &mut cache.clone(),
                        emit,
                    )
                })
            },
            Some((fields[1], fields[3])),
        );
    }
    let (owner, region, arguments, next, remaining) =
        if let Some(fields) = body.compiler.facts.enode("TraversalFor", traversal) {
            let Some((index, remaining)) = indices.split_first() else {
                return Err(error("traversal rank exceeds the selected domain"));
            };
            let mut arguments = Vec::new();
            // Cached leaves and input rows are evaluated at different coordinate ranks.
            let mut input_cache = LookupMap::default();
            let cache = if indices.len() == 1 { cache } else { &mut input_cache };
            for input in body.compiler.facts.vector(fields[3])? {
                arguments.push(element(
                    body,
                    scope,
                    Some(fields[2]),
                    input,
                    std::slice::from_ref(index),
                    cache,
                )?);
            }
            (fields[0], fields[1], arguments, fields[4], remaining)
        } else if let Some(fields) = body.compiler.facts.enode("TraversalCall", traversal) {
            let mut arguments = Vec::new();
            for argument in body.compiler.facts.vector(fields[3])? {
                arguments.push(body.value(scope, argument)?);
            }
            (fields[1], fields[2], arguments, fields[4], indices)
        } else if indices.is_empty() {
            let leaf = if let Some(fields) = body.compiler.facts.enode("TraversalValue", traversal) {
                TraversalLeaf::Source(fields[0])
            } else if let Some(fields) = body.compiler.facts.enode("TraversalResult", traversal) {
                TraversalLeaf::Result {
                    source: fields[0],
                    owner: fields[1],
                    port: fields[3],
                }
            } else {
                return Err(error("invalid traversal leaf"));
            };
            return emit(body, scope, leaf);
        } else {
            return Err(error("traversal rank is less than the selected domain"));
        };
    body.in_region(owner, region, arguments, |body, _| {
        emit_traversal(
            body,
            region,
            plan,
            next,
            remaining,
            &mut LookupMap::default(),
            emit,
        )
    })
}
pub(super) fn write_arrays(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    plan: Value,
    results: &[(i64, Value)],
    indices: &[Typed],
    destinations: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let query = Query(&body.compiler.program.graph);
    let totals = query.required("PlanTotals", (plan,))?;
    let totals = body.compiler.facts.integer(query.required("ResultCount", (totals,))?);
    for (slot, source) in results {
        let output = body.output_target(scope, operation, "output", *slot - totals)?;
        let mappings = body.compiler.facts.set("MappedOutputs", (operation, *slot));
        if output.is_some() || !mappings.is_empty() {
            let value = element(body, scope, Some(plan), *source, indices, cache)?;
            for destination in mappings {
                body.store_mapped(scope, destination, destinations, value.clone())?;
            }
            if let Some(output) = output {
                output.store(body, destinations, value)?;
            }
        }
    }
    Ok(())
}
