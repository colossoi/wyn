//! Read-only source queries shared by fact import and final body construction.
use crate::egglog::data::{
    is_slice, length_source, value_source, Array, ExprId, ExprKind, Ir, OperationId, OperationKind,
    ScremaForm,
};
use crate::egglog::dependencies::{Analysis, References};
use crate::egglog::timing::span;
use crate::egglog::visit::{Operand, OperandRole};
use crate::egglog::{OptimizeError, SCHEMA};
use egglog_engine::EGraph;
use sink::{InputSite, Kind, Operation, Sink};
use std::collections::{BTreeMap, BTreeSet};

mod egglog;
mod facts;
mod sink;
mod summary;
pub(in crate::egglog) use egglog::Egglog;

/// Load declarations once and insert all source facts in one native update.
pub(in crate::egglog) fn import(data: &Ir) -> Result<EGraph, OptimizeError> {
    let _timing = span("egglog fusion import");
    let mut graph = EGraph::default();
    {
        let _schema = span("egglog fusion import / schema");
        graph.parse_and_run_program(Some("fusion-schema.egg".into()), SCHEMA)?;
    }
    {
        let _facts = span("egglog fusion import / facts");
        graph.update(|state| Ok(emit(data, &mut Egglog::new(state)?)))??;
    }
    Ok(graph)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::egglog) enum Role {
    Input,
    Other,
    Argument,
    Length,
}
impl From<OperandRole> for Role {
    fn from(role: OperandRole) -> Self {
        match role {
            OperandRole::Input => Self::Input,
            OperandRole::Capture | OperandRole::Neutral => Self::Other,
            OperandRole::Argument => Self::Argument,
        }
    }
}

/// Read the source once and send facts directly to the selected backend.
pub(in crate::egglog) fn emit(data: &Ir, sink: &mut impl Sink) -> Result<(), OptimizeError> {
    let mut analysis = Analysis::new(data);
    facts::scalar_regions(data, &analysis.dependencies, sink)?;
    let mut schedules = analysis.dependencies.schedules(data)?;
    schedules.retain(|_, ops| {
        ops.iter().any(|id| {
            matches!(
                data.operations[*id].kind,
                OperationKind::Screma { .. } | OperationKind::Filter { .. }
            )
        })
    });
    let included = |op| {
        schedules.contains_key(&data.operations[op].region) && analysis.dependencies.live.contains(&op)
    };
    facts::operations(data, &analysis.dependencies, &schedules, sink)?;
    for (consumer, producer) in analysis.dependencies.dependencies() {
        if included(producer) && included(consumer) {
            sink.dependency(producer, consumer)?;
        }
    }
    for (gate, operations) in analysis.dependencies.effects.gates() {
        for &op in operations {
            if included(op) {
                sink.effect_member(gate, op)?;
            }
        }
    }
    for (op, gate) in analysis.dependencies.effects.waits() {
        if included(op) {
            sink.effect_wait(op, gate)?;
        }
    }
    let visitor = &mut analysis.visitor;
    for &consumer in &analysis.dependencies.live {
        let kind = &data.operations[consumer].kind;
        let mut values = BTreeMap::<Role, References>::new();
        let mut region_uses = BTreeSet::new();
        match kind {
            OperationKind::Call { function, args } if length_source(data, kind).is_some() => {
                values.insert(Role::Argument, visitor.expression(*function));
                values.insert(Role::Length, visitor.expressions(args));
            }
            _ => kind.for_each_operand(&mut |operand| match operand {
                Operand::Value(role, e) => {
                    let refs = visitor.expression(e);
                    values.entry(role.into()).or_default().extend(&refs, &mut visitor.sets);
                }
                Operand::Region(r) => {
                    region_uses.insert(r);
                }
            }),
        }
        for (role, refs) in values {
            let deps = refs.dependencies(&analysis.external, &mut visitor.sets);
            for p in visitor.sets.iter(deps) {
                let producer = OperationId::from(p);
                if included(producer) {
                    facts::usage(data, producer, consumer, role, included(consumer), sink)?;
                }
            }
        }
        for region in region_uses {
            for p in visitor.sets.iter(analysis.external[&region]) {
                let producer = OperationId::from(p);
                if included(producer) {
                    facts::usage(data, producer, consumer, Role::Other, included(consumer), sink)?;
                }
            }
        }
    }
    for region in analysis.active {
        let deps = analysis.results[&region].dependencies(&analysis.external, &mut visitor.sets);
        for p in visitor.sets.iter(deps) {
            let producer = OperationId::from(p);
            if included(producer) {
                sink.observed(producer)?;
            }
        }
    }
    Ok(())
}
pub(super) fn counts(form: &ScremaForm) -> (usize, usize) {
    (
        form.scans.iter().map(|s| s.neutral.len()).sum(),
        form.reductions.iter().map(|r| r.neutral.len()).sum(),
    )
}
/// Decode a single producer output and its slices without constructing a tuple tree.
fn produced_input(
    data: &Ir,
    array: &Array,
    producer: OperationId,
) -> Option<(usize, Vec<(ExprId, ExprId)>)> {
    let Array::Value(id) = array else { return None };
    match &data.expressions[value_source(data, *id)].kind {
        ExprKind::OperationResult(op) if *op == producer => Some((0, vec![])),
        ExprKind::Coerce(v) => produced_input(data, &Array::Value(*v), producer),
        ExprKind::Array(a) => produced_input(data, a, producer),
        ExprKind::Project { tuple, index } if matches!(data.expressions[*tuple].kind, ExprKind::OperationResult(op) if op == producer) => {
            Some((*index, vec![]))
        }
        ExprKind::PureApp { function, args } if is_slice(data, *function) => {
            let [base, start, end] = args.as_slice() else {
                return None;
            };
            let (slot, mut slices) = produced_input(data, &Array::Value(*base), producer)?;
            slices.push((*start, *end));
            Some((slot, slices))
        }
        _ => None,
    }
}

/// Visit leaves directly; slices of whole tuples remain opaque inputs.
fn input_leaves(data: &Ir, array: &Array, visit: &mut impl FnMut(&Array)) {
    match array {
        Array::Zip(arrays) => {
            for a in arrays {
                input_leaves(data, a, visit);
            }
        }
        Array::Value(id) => match &data.expressions[value_source(data, *id)].kind {
            ExprKind::Coerce(v) => input_leaves(data, &Array::Value(*v), visit),
            ExprKind::Array(a) => input_leaves(data, a, visit),
            _ => visit(array),
        },
        _ => visit(array),
    }
}

pub(super) fn routes(data: &Ir, producer: OperationId, consumer: OperationId) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    if input_slices(data, producer, consumer).is_none() {
        return out;
    }
    let mut valid = true;
    for a in inputs(&data.operations[consumer].kind) {
        input_leaves(data, a, &mut |a| {
            if let Some((slot, _)) = produced_input(data, a, producer) {
                out.insert(slot);
            } else {
                let mut refs = BTreeSet::new();
                array_references(data, a, &mut refs);
                // Partial routing would leave a self-dependency after fusion.
                valid &= !refs.iter().any(|v| matches!(data.expressions[*v].kind, ExprKind::OperationResult(op) if op == producer));
            }
        });
    }
    if !valid {
        out.clear();
    }
    out
}

pub(super) fn input_slices(
    data: &Ir,
    producer: OperationId,
    consumer: OperationId,
) -> Option<Vec<(ExprId, ExprId)>> {
    let mut selected = None;
    let mut valid = true;
    for a in inputs(&data.operations[consumer].kind) {
        input_leaves(data, a, &mut |a| {
            if let Some((_, slices)) = produced_input(data, a, producer) {
                if selected.as_ref().is_some_and(|old| *old != slices) {
                    valid = false;
                }
                selected = Some(slices);
            }
        });
    }
    valid.then(|| selected.unwrap_or_default())
}
pub(super) fn inputs(kind: &OperationKind) -> Vec<&Array> {
    match kind {
        OperationKind::Screma { inputs, .. }
        | OperationKind::Scatter { inputs, .. }
        | OperationKind::BucketScatter { inputs, .. } => inputs.iter().collect(),
        OperationKind::Filter { inputs, .. } => inputs.iter().collect(),
        OperationKind::ReduceByIndex { inputs, .. } => inputs.iter().collect(),
        _ => vec![],
    }
}

/// Read an element-demand edge. Egglog checks profitability and complete uses.
pub(super) fn indexed_demand(
    data: &Ir,
    producer: OperationId,
    consumer: OperationId,
) -> Option<(usize, Vec<usize>)> {
    if data.operations[producer].region != data.operations[consumer].region {
        return None;
    }
    let OperationKind::Index { array, .. } = data.operations[consumer].kind else {
        return None;
    };
    projection(data, array, producer)
}
// TLC may represent an array of tuples as projected component arrays. The
// first projection selects the operation result; the rest select its element.
fn projection(data: &Ir, value: ExprId, producer: OperationId) -> Option<(usize, Vec<usize>)> {
    fn walk(data: &Ir, value: ExprId, producer: OperationId, path: &mut Vec<usize>) -> Option<()> {
        match data.expressions[value].kind {
            ExprKind::Coerce(v) => walk(data, v, producer, path),
            ExprKind::Project { tuple, index } => {
                walk(data, tuple, producer, path)?;
                path.push(index);
                Some(())
            }
            ExprKind::OperationResult(op) if op == producer => Some(()),
            _ => None,
        }
    }
    let mut path = vec![];
    walk(data, value, producer, &mut path)?;
    let (&slot, path) = path.split_first()?;
    Some((slot, path.to_vec()))
}
pub(super) fn references(data: &Ir, v: ExprId, out: &mut BTreeSet<ExprId>) {
    let v = value_source(data, v);
    match &data.expressions[v].kind {
        ExprKind::Parameter(_) | ExprKind::OperationResult(_) => {
            out.insert(v);
        }
        ExprKind::Project { tuple, .. } | ExprKind::Coerce(tuple) => references(data, *tuple, out),
        ExprKind::Tuple(vs) | ExprKind::Vector(vs) | ExprKind::Closure { captures: vs, .. } => {
            for &v in vs {
                references(data, v, out)
            }
        }
        ExprKind::PureApp { function, args } => {
            references(data, *function, out);
            for &v in args {
                references(data, v, out)
            }
        }
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => {
            for v in [condition, then_value, else_value] {
                references(data, *v, out)
            }
        }
        ExprKind::Array(a) => array_references(data, a, out),
        _ => {}
    }
}
fn array_references(data: &Ir, a: &Array, out: &mut BTreeSet<ExprId>) {
    match a {
        Array::Value(v) => references(data, *v, out),
        Array::Literal(vs) => {
            for &v in vs {
                references(data, v, out)
            }
        }
        Array::Zip(xs) => {
            for a in xs {
                array_references(data, a, out)
            }
        }
        Array::Range { start, len, step } => {
            references(data, *start, out);
            references(data, *len, out);
            if let Some(s) = step {
                references(data, *s, out);
            }
        }
    }
}
