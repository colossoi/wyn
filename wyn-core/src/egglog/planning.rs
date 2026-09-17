//! Import structural facts for relational scheduling. This adapter never decides
//! which values to materialize, which resources to allocate, or which phase writes
//! an output. Those decisions belong to the .egg rules.
use super::data::body_signature;
use super::visit::{Operand, OperandRole};
use super::OptimizeError;
use crate::egglog::data::{
    is_slice, Array, DispatchId, ExprData, ExprId, ExprKind, OperationKind, OutputData, TypeData, TypeId,
};
use crate::egglog::dependencies::{safe_body, Dependencies};
use crate::egglog::{Program, Scheduled};
use crate::interface::EntryKind;
use crate::ssa::layout::type_byte_size;
use crate::types::{bool_type, canonical_storage_buffer_ty, Type, TypeExt, TypeName};
use egglog_engine::ast::Literal;
use egglog_engine::{EGraph, Term};
use std::collections::{BTreeMap, BTreeSet, HashMap};

mod read;
pub(super) use read::{read, Readout};

pub(super) const RULES: &str = concat!(
    include_str!("planning.egg"),
    "\n",
    include_str!("schedule.egg"),
    "\n",
    include_str!("residency.egg"),
    "\n",
    include_str!("allocation.egg"),
    "\n",
    include_str!("dispatch.egg"),
    "\n",
);
pub(super) const KEYS: &str = "(datatype ExprKey (ExprId i64))\n(datatype TypeKey (TypeId i64))\n";
pub(super) const RUN: &str =
    "(run-schedule (seq (saturate (run structure)) (saturate (run classify)) (saturate (seq (run residency) (run schedule) (run allocation) (run dispatch)))))";

/// Assign the derived order to emitted launch IDs. Kernel emission contributes
/// identities only; it cannot add its own dependency decisions.
pub(super) fn read_dispatch_order(
    graph: &EGraph,
    data: &mut Program<Scheduled>,
) -> Result<(), OptimizeError> {
    let (rows, _, dag) = graph.function_to_dag("DispatchDependency", usize::MAX, false)?;
    for row in rows {
        let Term::App(_, args) = dag.get(row) else {
            return Err(invalid_order());
        };
        let [before, after] = args.as_slice() else {
            return Err(invalid_order());
        };
        let (Term::Lit(Literal::Int(before)), Term::Lit(Literal::Int(after))) =
            (dag.get(*before), dag.get(*after))
        else {
            return Err(invalid_order());
        };
        let before = DispatchId::from(u32::try_from(*before).map_err(|_| invalid_order())?);
        let after = DispatchId::from(u32::try_from(*after).map_err(|_| invalid_order())?);
        if data.state.dispatches.get(before).is_none() {
            return Err(invalid_order());
        }
        let Some(dispatch) = data.state.dispatches.get_mut(after) else {
            return Err(invalid_order());
        };
        dispatch.dependencies.insert(before);
    }
    Ok(())
}

fn invalid_order() -> OptimizeError {
    OptimizeError::Output("invalid relational dispatch order".into())
}

fn expr(e: ExprId) -> String {
    format!("(ExprId {})", e.as_u32())
}
fn ty(t: TypeId) -> String {
    format!("(TypeId {})", t.as_u32())
}

pub(super) fn facts(
    data: &Program<Scheduled>,
    summary: &Dependencies,
    count_type: TypeId,
    out: &mut String,
) {
    out.push_str(&format!("(CounterType {})\n", ty(count_type)));
    let mut values = BTreeSet::new();
    for (&id, output) in &data.state.outputs {
        values.insert(output.expression);
        if output.scalar {
            out.push_str(&format!(
                "(ReturnScalar {} {} {})\n",
                id.as_u32(),
                expr(output.expression),
                ty(data.expressions[output.expression].ty)
            ));
        } else {
            out.push_str(&format!(
                "(ReturnArray {} {})\n",
                id.as_u32(),
                expr(output.expression)
            ));
        }
    }
    let mut regions = BTreeSet::new();
    let symbols: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    for (&id, entry) in &data.entries {
        regions.insert(data.definitions[entry.definition].body);
        if entry.declaration.entry_kind != EntryKind::Compute {
            continue;
        }
        out.push_str(&format!(
            "(HostRoot {} {})\n",
            id.as_u32(),
            data.definitions[entry.definition].body.egglog()
        ));
    }
    for &id in &summary.live {
        let op = &data.operations[id];
        regions.insert(op.region);
        let key = id.egglog();
        let collective = matches!(
            op.kind,
            OperationKind::Screma { .. }
                | OperationKind::Filter { .. }
                | OperationKind::Scatter { .. }
                | OperationKind::BucketScatter { .. }
                | OperationKind::ReduceByIndex { .. }
        );
        out.push_str(&format!(
            "(set (ContainsCollective {key}) {collective})\n(set (ScalarBoundary {key}) false)\n"
        ));
        out.push_str(&format!("(Site {key} {})\n", op.region.egglog()));
        for r in op.kind.structured_regions() {
            regions.insert(r);
            out.push_str(&format!("(Enters {key} {})\n", r.egglog()));
        }
        if let OperationKind::Loop { header, body, .. } = &op.kind {
            out.push_str(&format!(
                "(Repeated {key} {} {})\n",
                header.egglog(),
                body.egglog()
            ));
        }
        let called = match &op.kind {
            OperationKind::Call { function, .. } => match &data.expressions[*function].kind {
                ExprKind::Lambda(r) => Some(*r),
                ExprKind::Global(s) | ExprKind::Closure { code: s, .. } => symbols.get(s).copied(),
                _ => None,
            },
            OperationKind::EvalGlobal(s) => symbols.get(s).copied(),
            _ => None,
        };
        if let Some(r) = called {
            regions.insert(r);
            out.push_str(&format!("(Enters {key} {})\n", r.egglog()));
        }
        op.kind.for_each_operand(&mut |operand| {
            if let Operand::Value(role, e) = operand {
                let role = if matches!(role, OperandRole::Input) { "input" } else { "environment" };
                values.insert(e);
                out.push_str(&format!("(Operand {key} \"{role}\" {})\n", expr(e)));
            }
        });
        let inputs = match &op.kind {
            OperationKind::Screma { form, inputs, .. } => {
                out.push_str(&format!(
                    "(CollectiveShape {key} {} {} {})\n",
                    form.scans.len(),
                    form.reductions.len(),
                    summary.discardable.contains(&id)
                ));
                let scans = form.scans.iter().flat_map(|s| &s.neutral);
                let totals = form.reductions.iter().flat_map(|r| &r.neutral);
                let scan_count = scans.clone().count();
                let total_count = totals.clone().count();
                out.push_str(&format!("(TotalCount {key} {total_count})\n"));
                for (i, e) in scans.enumerate() {
                    let t = ty(data.expressions[*e].ty);
                    out.push_str(&format!(
                        "(Accumulator {key} {i} {t})\n(ScanComponent {key} {i} {t})\n"
                    ));
                }
                for (i, e) in totals.enumerate() {
                    let t = ty(data.expressions[*e].ty);
                    out.push_str(&format!(
                        "(Accumulator {key} {} {t})\n(TotalResult {key} {i} {t})\n",
                        scan_count + i
                    ));
                }
                for (i, t) in body_signature(&form.post).1.into_iter().enumerate() {
                    out.push_str(&format!("(ArrayResult {key} {} {})\n", total_count + i, ty(t)));
                }
                for (i, t) in
                    body_signature(&form.pre).1.into_iter().skip(scan_count + total_count).enumerate()
                {
                    out.push_str(&format!("(MappedComponent {key} {i} {})\n", ty(t)));
                }
                Some(inputs)
            }
            OperationKind::Filter {
                map, body, inputs, ..
            } => {
                let safe = safe_body(map, &summary.safe_regions) && safe_body(body, &summary.safe_regions);
                out.push_str(&format!("(FilterShape {key} {safe})\n"));
                if let Some(t) = body_signature(map).1.first() {
                    out.push_str(&format!("(FilterResult {key} {})\n", ty(*t)));
                }
                Some(inputs)
            }
            OperationKind::Scatter {
                destination, inputs, ..
            }
            | OperationKind::ReduceByIndex {
                destination, inputs, ..
            }
            | OperationKind::BucketScatter {
                destination, inputs, ..
            } => {
                out.push_str(&format!("(IndexedWrite {key})\n"));
                out.push_str(&format!("(UpdatedResult {key} 0 {})\n", expr(destination.value)));
                if matches!(op.kind, OperationKind::BucketScatter { .. }) {
                    out.push_str(&format!("(BucketResult {key} {})\n", expr(destination.value)));
                    // Its ranked iteration space is still a source payload. The
                    // selected serial phase has one invocation regardless of rank.
                    None
                } else {
                    Some(inputs)
                }
            }
            _ => {
                out.push_str(&format!("(ScalarSite {key})\n"));
                if data.types[op.ty].ty == bool_type()
                    || type_byte_size(&data.types[op.ty].ty).is_some_and(|n| n > 0)
                {
                    out.push_str(&format!("(ScalarCandidate {key} {})\n", ty(op.ty)));
                }
                None
            }
        };
        if let Some(inputs) = inputs {
            out.push_str(&format!(
                "(InputDomain {key} {})\n",
                inputs.first().map(|a| extent(a)).unwrap_or("(Fixed 0)".into())
            ));
        }
    }
    for r in regions {
        for (i, &e) in data.regions[r].results.iter().enumerate() {
            values.insert(e);
            out.push_str(&format!("(ExitValue {} {i} {})\n", r.egglog(), expr(e)));
        }
    }
    for (after, before) in summary.dependencies() {
        out.push_str(&format!(
            "(SourceDependency {} {})\n",
            after.egglog(),
            before.egglog()
        ));
    }
    for (gate, inputs) in summary.effects.gates() {
        for op in inputs {
            out.push_str(&format!("(EffectInput {gate} {})\n", op.egglog()));
        }
    }
    for (op, gate) in summary.effects.waits() {
        out.push_str(&format!("(EffectWait {} {gate})\n", op.egglog()));
    }

    // Each expression is visited once. Only structural edges and view metadata
    // enter the planner: no arithmetic AST or per-stage Rust dependency closure.
    let mut pending: Vec<_> = values.iter().copied().collect();
    while let Some(e) = pending.pop() {
        let value = &data.expressions[e];
        let key = expr(e);
        out.push_str(&format!("(SourceType {key} {})\n", ty(value.ty)));
        let mut generic_children = false;
        match &value.kind {
            ExprKind::Parameter(p) => {
                out.push_str(&format!(
                    "(SourceParameter {key} {})\n(set (HasInputFields {key}) false)\n",
                    data.parameters[*p].region.egglog()
                ));
            }
            ExprKind::OperationResult(op) => {
                if matches!(
                    data.operations[*op].kind,
                    OperationKind::Screma { .. } | OperationKind::BucketScatter { .. }
                ) {
                    out.push_str(&format!("(ResultTuple {key} {})\n", op.egglog()));
                } else {
                    out.push_str(&format!("(DirectResult {key} {} 0)\n", op.egglog()));
                }
            }
            ExprKind::Project { tuple, index } => {
                out.push_str(&format!("(Projection {key} {} {index})\n", expr(*tuple)));
            }
            ExprKind::Tuple(fields) | ExprKind::Vector(fields) => {
                for (i, &field) in fields.iter().enumerate() {
                    out.push_str(&format!("(FieldValue {key} {i} {})\n", expr(field)));
                }
                generic_children = true;
            }
            ExprKind::Coerce(inner) | ExprKind::Array(Array::Value(inner)) => {
                out.push_str(&format!("(ForwardValue {key} {})\n", expr(*inner)));
            }
            ExprKind::PureApp { function, args } if is_slice(data, *function) && args.len() == 3 => {
                out.push_str(&format!(
                    "(SliceView {key} {} {} {})\n",
                    expr(args[0]),
                    expr(args[1]),
                    expr(args[2])
                ));
            }
            _ => generic_children = true,
        }
        if generic_children && !matches!(value.kind, ExprKind::Tuple(_) | ExprKind::Vector(_)) {
            out.push_str(&format!("(ComputedValue {key})\n"));
        }
        for child in value.kind.children() {
            if generic_children {
                out.push_str(&format!("(ChildValue {key} {})\n", expr(child)));
            }
            if values.insert(child) {
                pending.push(child);
            }
        }
    }
}

fn extent(array: &Array) -> String {
    match array {
        Array::Value(e) => format!("(Length {})", expr(*e)),
        Array::Zip(xs) => xs.first().map(extent).unwrap_or("(Fixed 0)".into()),
        Array::Literal(xs) => format!("(Fixed {})", xs.len()),
        Array::Range { len, .. } => format!("(Scalar {})", expr(*len)),
    }
}

/// Expose source result slots once; tuple projection is structural import, not
/// an allocation decision. Use the same global identities for existing values.
pub(super) fn outputs(data: &mut Program<Scheduled>, out: &mut String) {
    let mut types: HashMap<_, _> = data.types.iter().map(|(&id, t)| (t.ty.clone(), id)).collect();
    let mut expressions: HashMap<_, _> = data.expressions.iter().map(|(&id, e)| (e.clone(), id)).collect();
    let entries: Vec<_> =
        data.entries.iter().map(|(&id, e)| (id, data.definitions[e.definition].body)).collect();
    for (entry, region) in entries {
        if data.entries[entry].declaration.entry_kind != EntryKind::Compute {
            continue;
        }
        for e in data.regions[region].results.clone() {
            let mut ty = &data.types[data.expressions[e].ty].ty;
            while let Type::Constructed(TypeName::Existential(_), fields) = ty {
                ty = &fields[0];
            }
            let fields = match ty {
                Type::Constructed(TypeName::Unit | TypeName::SideEffect | TypeName::StorageTexture, _) => {
                    continue
                }
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => Some(fields.clone()),
                _ => None,
            };
            let values = if let Some(fields) = fields {
                fields
                    .into_iter()
                    .enumerate()
                    .map(|(index, ty)| {
                        if let ExprKind::Tuple(values) = &data.expressions[e].kind {
                            return values[index];
                        }
                        let ty = *types
                            .entry(ty.clone())
                            .or_insert_with(|| data.ir.types.alloc(TypeData { ty }));
                        let value = ExprData {
                            ty,
                            kind: ExprKind::Project { tuple: e, index },
                        };
                        *expressions
                            .entry(value.clone())
                            .or_insert_with(|| data.ir.expressions.alloc(value))
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![e]
            };
            for (index, expression) in values.into_iter().enumerate() {
                let scalar = !canonical_storage_buffer_ty(&data.types[data.expressions[expression].ty].ty)
                    .is_array();
                data.state.outputs.alloc(OutputData {
                    entry,
                    index,
                    expression,
                    buffer: None,
                    scalar,
                });
            }
        }
    }
    let entry_regions: BTreeSet<_> =
        data.entries.values().map(|e| data.definitions[e.definition].body).collect();
    let parameters: Vec<_> = data
        .expressions
        .iter()
        .filter_map(|(&id, e)| {
            let ExprKind::Parameter(p) = e.kind else {
                return None;
            };
            let Type::Constructed(TypeName::Tuple(_), ts) = &data.types[e.ty].ty else {
                return None;
            };
            (ts.iter().all(TypeExt::is_array) && entry_regions.contains(&data.parameters[p].region))
                .then(|| (id, ts.clone()))
        })
        .collect();
    for (tuple, ts) in parameters {
        let ExprKind::Parameter(parameter) = data.expressions[tuple].kind else {
            unreachable!("input tuple {tuple:?} is not a parameter");
        };
        let region = data.parameters[parameter].region;
        out.push_str(&format!(
            "(set (HasInputFields {}) true)\n(SourceType {} {})\n",
            expr(tuple),
            expr(tuple),
            ty(data.expressions[tuple].ty)
        ));
        for (index, field_type) in ts.into_iter().enumerate() {
            let field_ty = *types
                .entry(field_type.clone())
                .or_insert_with(|| data.ir.types.alloc(TypeData { ty: field_type }));
            let value = ExprData {
                ty: field_ty,
                kind: ExprKind::Project { tuple, index },
            };
            let field =
                *expressions.entry(value.clone()).or_insert_with(|| data.ir.expressions.alloc(value));
            out.push_str(&format!(
                "(FieldValue {} {index} {})\n(ChildValue {} {})\n(ParameterValue {} {})\n(SourceType {} {})\n(Projection {} {} {index})\n",
                expr(tuple), expr(field), expr(tuple), expr(field), expr(field), region.egglog(), expr(field), ty(field_ty), expr(field), expr(tuple),
            ));
        }
    }
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
