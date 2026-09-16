//! Import structural facts for relational scheduling. This adapter never decides
//! which values to materialize, which resources to allocate, or which phase writes
//! an output. Those decisions belong to the .egg rules.

use super::data::*;
use super::fusion::{is_slice, signature};
use super::visit::{Operand, OperandRole};
use super::{expressions, snapshot, timing, OptimizeError};
use egglog_engine::{
    ast::{Command, Literal},
    EGraph, Term,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

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
const KEYS: &str = "(datatype ExprKey (ExprId i64))\n(datatype TypeKey (TypeId i64))\n";
pub(super) const RUN: &str =
    "(run-schedule (saturate (seq (run residency) (run schedule) (run allocation) (run dispatch))))";

/// Return replayable planning commands and the same evaluated graph used by the
/// recipe reader. ExprKey/TypeKey are shared with the separately loaded expression
/// layer in --egg-out; no scalar syntax is needed to run the planner itself.
pub(super) fn analyze(
    data: &mut AssociatedData,
    summary: &snapshot::Snapshot,
) -> Result<(Vec<Command>, EGraph), OptimizeError> {
    let _timing = timing::span("relational planning");
    let input_fields = outputs(data);
    let count_type = super::fusion::ty(
        data,
        crate::types::Type::Constructed(crate::types::TypeName::UInt(32), vec![]),
    );
    let source = timing::time("import planning facts", || {
        facts(data, summary, count_type, &input_fields)
    });
    let mut commands = expressions::parse(RULES)?;
    commands.extend(expressions::parse(&source)?);
    commands.extend(expressions::parse(RUN)?);
    let mut graph = EGraph::default();
    graph.parse_and_run_program(Some("ids.egg".into()), include_str!("ids.egg"))?;
    graph.parse_and_run_program(None, KEYS)?;
    timing::time("derive stages and storage", || {
        graph.run_program(commands.clone())
    })?;
    Ok((commands, graph))
}

/// Assign the derived order to emitted launch IDs. Kernel emission contributes
/// identities only; it cannot add its own dependency decisions.
pub(super) fn dispatch_order(
    graph: &mut EGraph,
    launches: &str,
    data: &mut AssociatedData,
) -> Result<Vec<Command>, OptimizeError> {
    let _timing = timing::span("read dispatch order");
    let mut commands = expressions::parse(launches)?;
    commands.extend(expressions::parse("(run-schedule (saturate (run dispatch)))")?);
    graph.run_program(commands.clone())?;
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
        if data.dispatches.get(before).is_none() {
            return Err(invalid_order());
        }
        let Some(dispatch) = data.dispatches.get_mut(after) else {
            return Err(invalid_order());
        };
        dispatch.dependencies.insert(before);
    }
    Ok(commands)
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

fn facts(
    data: &AssociatedData,
    summary: &snapshot::Snapshot,
    count_type: TypeId,
    input_fields: &BTreeMap<ExprId, Vec<ExprId>>,
) -> String {
    let mut out = String::new();
    writeln!(out, "(CounterType {})", ty(count_type)).unwrap();
    let mut values = BTreeSet::new();
    for (&parent, fields) in input_fields {
        let ExprKind::Parameter(p) = data.expressions[parent].kind else {
            unreachable!()
        };
        for (i, &field) in fields.iter().enumerate() {
            values.insert(field);
            writeln!(
                out,
                "(FieldValue {} {i} {})\n(ChildValue {} {})\n(ParameterValue {} {})",
                expr(parent),
                expr(field),
                expr(parent),
                expr(field),
                expr(field),
                data.parameters[p].region.egglog()
            )
            .unwrap();
        }
    }
    for (&id, output) in &data.outputs {
        values.insert(output.expression);
        if output.scalar {
            writeln!(
                out,
                "(ReturnScalar {} {} {})",
                id.as_u32(),
                expr(output.expression),
                ty(data.expressions[output.expression].ty)
            )
            .unwrap();
        } else {
            writeln!(out, "(ReturnArray {} {})", id.as_u32(), expr(output.expression)).unwrap();
        }
    }
    let mut regions = BTreeSet::new();
    let symbols: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    for (&id, entry) in &data.entries {
        regions.insert(data.definitions[entry.definition].body);
        writeln!(
            out,
            "(HostRoot {} {})",
            id.as_u32(),
            data.definitions[entry.definition].body.egglog()
        )
        .unwrap();
    }
    for &id in &summary.live {
        let op = &data.operations[id];
        regions.insert(op.region);
        let key = id.egglog();
        writeln!(out, "(Site {key} {})", op.region.egglog()).unwrap();
        for r in op.kind.structured_regions() {
            regions.insert(r);
            writeln!(out, "(Enters {key} {})", r.egglog()).unwrap();
        }
        if let OperationKind::Loop { header, body, .. } = &op.kind {
            writeln!(out, "(Repeated {key} {} {})", header.egglog(), body.egglog()).unwrap();
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
            writeln!(out, "(Enters {key} {})", r.egglog()).unwrap();
        }
        op.kind.for_each_operand(&mut |operand| {
            if let Operand::Value(role, e) = operand {
                let role = if matches!(role, OperandRole::Input) { "input" } else { "environment" };
                values.insert(e);
                writeln!(out, "(Operand {key} \"{role}\" {})", expr(e)).unwrap();
            }
        });
        let inputs = match &op.kind {
            OperationKind::Screma { form, inputs, .. } => {
                writeln!(
                    out,
                    "(CollectiveShape {key} {} {} {})",
                    form.scans.len(),
                    form.reductions.len(),
                    summary.discardable.contains(&id)
                )
                .unwrap();
                let scans: Vec<_> = form.scans.iter().flat_map(|s| s.neutral.iter()).copied().collect();
                let totals: Vec<_> =
                    form.reductions.iter().flat_map(|r| r.neutral.iter()).copied().collect();
                writeln!(out, "(TotalCount {key} {})", totals.len()).unwrap();
                for (i, e) in scans.iter().chain(&totals).enumerate() {
                    writeln!(out, "(Accumulator {key} {i} {})", ty(data.expressions[*e].ty)).unwrap();
                }
                for (i, e) in scans.iter().enumerate() {
                    writeln!(out, "(ScanComponent {key} {i} {})", ty(data.expressions[*e].ty)).unwrap();
                }
                for (i, e) in totals.iter().enumerate() {
                    writeln!(out, "(TotalResult {key} {i} {})", ty(data.expressions[*e].ty)).unwrap();
                }
                for (i, t) in signature(&form.post).1.into_iter().enumerate() {
                    writeln!(out, "(ArrayResult {key} {} {})", totals.len() + i, ty(t)).unwrap();
                }
                for (i, t) in
                    signature(&form.pre).1.into_iter().skip(scans.len() + totals.len()).enumerate()
                {
                    writeln!(out, "(MappedComponent {key} {i} {})", ty(t)).unwrap();
                }
                Some(inputs)
            }
            OperationKind::Filter {
                map, body, inputs, ..
            } => {
                let safe = snapshot::safe_body(map, &summary.safe_regions)
                    && snapshot::safe_body(body, &summary.safe_regions);
                writeln!(out, "(FilterShape {key} {safe})").unwrap();
                if let Some(t) = signature(map).1.first() {
                    writeln!(out, "(FilterResult {key} {})", ty(*t)).unwrap();
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
                writeln!(out, "(IndexedWrite {key})").unwrap();
                writeln!(out, "(UpdatedResult {key} 0 {})", expr(destination.value)).unwrap();
                if matches!(op.kind, OperationKind::BucketScatter { .. }) {
                    writeln!(out, "(BucketResult {key} {})", expr(destination.value)).unwrap();
                    // Its ranked iteration space is still a source payload. The
                    // selected serial phase has one invocation regardless of rank.
                    None
                } else {
                    Some(inputs)
                }
            }
            _ => {
                writeln!(out, "(ScalarSite {key})").unwrap();
                None
            }
        };
        if let Some(inputs) = inputs {
            writeln!(
                out,
                "(InputDomain {key} {})",
                inputs.first().map(|a| extent(a)).unwrap_or("(Fixed 0)".into())
            )
            .unwrap();
        }
    }
    for r in regions {
        for (i, &e) in data.regions[r].results.iter().enumerate() {
            values.insert(e);
            writeln!(out, "(ExitValue {} {i} {})", r.egglog(), expr(e)).unwrap();
        }
    }
    for (after, before) in summary.dependencies() {
        writeln!(out, "(SourceDependency {} {})", after.egglog(), before.egglog()).unwrap();
    }
    for (gate, inputs) in summary.effects.gates() {
        for op in inputs {
            writeln!(out, "(EffectInput {gate} {})", op.egglog()).unwrap();
        }
    }
    for (op, gate) in summary.effects.waits() {
        writeln!(out, "(EffectWait {} {gate})", op.egglog()).unwrap();
    }

    // Each expression is visited once. Only structural edges and view metadata
    // enter the planner: no arithmetic AST or per-stage Rust dependency closure.
    let mut pending: Vec<_> = values.iter().copied().collect();
    while let Some(e) = pending.pop() {
        let value = &data.expressions[e];
        let key = expr(e);
        writeln!(out, "(SourceType {key} {})", ty(value.ty)).unwrap();
        let mut generic_children = false;
        match &value.kind {
            ExprKind::Parameter(p) => {
                if input_fields.contains_key(&e) {
                    continue;
                }
                writeln!(
                    out,
                    "(ParameterValue {key} {})",
                    data.parameters[*p].region.egglog()
                )
                .unwrap();
            }
            ExprKind::OperationResult(op) => {
                if matches!(
                    data.operations[*op].kind,
                    OperationKind::Screma { .. } | OperationKind::BucketScatter { .. }
                ) {
                    writeln!(out, "(ResultTuple {key} {})", op.egglog()).unwrap();
                } else {
                    writeln!(out, "(DirectResult {key} {} 0)", op.egglog()).unwrap();
                }
            }
            ExprKind::Project { tuple, index } => {
                writeln!(out, "(Projection {key} {} {index})", expr(*tuple)).unwrap();
            }
            ExprKind::Tuple(fields) | ExprKind::Vector(fields) => {
                for (i, &field) in fields.iter().enumerate() {
                    writeln!(out, "(FieldValue {key} {i} {})", expr(field)).unwrap();
                }
                generic_children = true;
            }
            ExprKind::Coerce(inner) | ExprKind::Array(Array::Value(inner)) => {
                writeln!(out, "(ForwardValue {key} {})", expr(*inner)).unwrap();
            }
            ExprKind::PureApp { function, args } if is_slice(data, *function) && args.len() == 3 => {
                writeln!(
                    out,
                    "(SliceView {key} {} {} {})",
                    expr(args[0]),
                    expr(args[1]),
                    expr(args[2])
                )
                .unwrap();
            }
            _ => generic_children = true,
        }
        if generic_children && !matches!(value.kind, ExprKind::Tuple(_) | ExprKind::Vector(_)) {
            writeln!(out, "(ComputedValue {key})").unwrap();
        }
        for child in value.kind.children() {
            if generic_children {
                writeln!(out, "(ChildValue {key} {})", expr(child)).unwrap();
            }
            if values.insert(child) {
                pending.push(child);
            }
        }
    }
    out
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
fn outputs(data: &mut AssociatedData) -> BTreeMap<ExprId, Vec<ExprId>> {
    use crate::types::{Type, TypeExt, TypeName};
    let mut types: std::collections::HashMap<_, _> =
        data.types.iter().map(|(&id, t)| (t.ty.clone(), id)).collect();
    let mut expressions: std::collections::HashMap<_, _> =
        data.expressions.iter().map(|(&id, e)| (e.clone(), id)).collect();
    let entries: Vec<_> =
        data.entries.iter().map(|(&id, e)| (id, data.definitions[e.definition].body)).collect();
    for (entry, region) in entries {
        for e in data.regions[region].results.clone() {
            let mut ty = &data.types[data.expressions[e].ty].ty;
            while let Type::Constructed(TypeName::Existential(_), fields) = ty {
                ty = &fields[0];
            }
            let fields = match ty {
                Type::Constructed(TypeName::Unit | TypeName::SideEffect, _) => continue,
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
                        let ty =
                            *types.entry(ty.clone()).or_insert_with(|| data.types.alloc(TypeData { ty }));
                        let value = ExprData {
                            ty,
                            kind: ExprKind::Project { tuple: e, index },
                        };
                        *expressions.entry(value.clone()).or_insert_with(|| data.expressions.alloc(value))
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![e]
            };
            for (index, expression) in values.into_iter().enumerate() {
                let scalar = !crate::types::canonical_storage_buffer_ty(
                    &data.types[data.expressions[expression].ty].ty,
                )
                .is_array();
                data.outputs.alloc(OutputData {
                    entry,
                    index,
                    expression,
                    buffer: None,
                    scalar,
                });
            }
        }
    }
    let mut fields = BTreeMap::new();
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
        let values = ts
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                let ty = *types.entry(ty.clone()).or_insert_with(|| data.types.alloc(TypeData { ty }));
                let value = ExprData {
                    ty,
                    kind: ExprKind::Project { tuple, index },
                };
                *expressions.entry(value.clone()).or_insert_with(|| data.expressions.alloc(value))
            })
            .collect();
        fields.insert(tuple, values);
    }
    fields
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
