//! Import structural facts for relational scheduling. This adapter never decides
//! which values to materialize, which resources to allocate, or which phase writes
//! an output. Those decisions belong to the .egg rules.
use super::data::body_signature;
use super::visit::{Operand, OperandRole};
use crate::egglog::data::{
    is_slice, Array, ExprData, ExprKind, OperationKind, OutputData, TypeData, TypeId,
};
use crate::egglog::dependencies::{safe_body, Dependencies};
use crate::egglog::{Program, Scheduled};
use crate::interface::EntryKind;
use crate::ssa::layout::type_byte_size;
use crate::types::{bool_type, canonical_storage_buffer_ty, Type, TypeExt, TypeName};
use egglog_engine::{Error, FullState, Value, Write};
use std::collections::{BTreeMap, BTreeSet, HashMap};

mod read;
pub(super) use read::{read, Readout, Recipe};

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
    "(run-schedule (seq (saturate (run structure)) (saturate (run classify)) (saturate (seq (run residency) (run schedule) (run allocation) (run dispatch))) (run readout)))";

pub(super) fn facts(
    data: &Program<Scheduled>,
    summary: &Dependencies,
    count_type: TypeId,
    sink: &mut FullState<'_, '_>,
) -> Result<(), Error> {
    let count_type = sink.add("TypeId", i64::from(count_type.as_u32()))?;
    sink.add("CounterType", count_type)?;
    let mut values = BTreeSet::new();
    for (&id, output) in &data.state.outputs {
        values.insert(output.expression);
        let e = sink.add("ExprId", i64::from(output.expression.as_u32()))?;
        if output.scalar {
            let ty = sink.add(
                "TypeId",
                i64::from(data.expressions[output.expression].ty.as_u32()),
            )?;
            sink.add("ReturnScalar", (i64::from(id.as_u32()), e, ty))?;
        } else {
            sink.add("ReturnArray", (i64::from(id.as_u32()), e))?;
        }
    }
    let mut regions = BTreeSet::new();
    let symbols: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    for (&id, entry) in &data.entries {
        let region = data.definitions[entry.definition].body;
        regions.insert(region);
        if entry.declaration.entry_kind == EntryKind::Compute {
            let region = sink.add("RegionId", i64::from(region.as_u32()))?;
            sink.add("HostRoot", (i64::from(id.as_u32()), region))?;
        }
    }
    for &id in &summary.live {
        let op = &data.operations[id];
        regions.insert(op.region);
        let key = sink.add("OperationId", i64::from(id.as_u32()))?;
        let collective = matches!(
            op.kind,
            OperationKind::Screma { .. }
                | OperationKind::Filter { .. }
                | OperationKind::Scatter { .. }
                | OperationKind::BucketScatter { .. }
                | OperationKind::ReduceByIndex { .. }
        );
        sink.set("ContainsCollective", key, collective)?;
        sink.set("ScalarBoundary", key, false)?;
        let region = sink.add("RegionId", i64::from(op.region.as_u32()))?;
        sink.add("Site", (key, region))?;
        for r in op.kind.structured_regions() {
            regions.insert(r);
            let r = sink.add("RegionId", i64::from(r.as_u32()))?;
            sink.add("Enters", (key, r))?;
        }
        if let OperationKind::Loop { header, body, .. } = &op.kind {
            let header = sink.add("RegionId", i64::from(header.as_u32()))?;
            let body = sink.add("RegionId", i64::from(body.as_u32()))?;
            sink.add("Repeated", (key, header, body))?;
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
            let r = sink.add("RegionId", i64::from(r.as_u32()))?;
            sink.add("Enters", (key, r))?;
        }
        let mut result: Result<(), Error> = Ok(());
        op.kind.for_each_operand(&mut |operand| {
            if result.is_err() {
                return;
            }
            if let Operand::Value(role, e) = operand {
                let role = if matches!(role, OperandRole::Input) { "input" } else { "environment" };
                values.insert(e);
                result = (|| {
                    let e = sink.add("ExprId", i64::from(e.as_u32()))?;
                    sink.add("Operand", (key, role, e))?;
                    Ok(())
                })();
            }
        });
        result?;
        let inputs = match &op.kind {
            OperationKind::Screma { form, inputs, .. } => {
                sink.add(
                    "CollectiveShape",
                    (
                        key,
                        form.scans.len() as i64,
                        form.reductions.len() as i64,
                        summary.discardable.contains(&id),
                    ),
                )?;
                let scans = form.scans.iter().flat_map(|s| &s.neutral);
                let totals = form.reductions.iter().flat_map(|r| &r.neutral);
                let scan_count = scans.clone().count();
                let total_count = totals.clone().count();
                sink.add("TotalCount", (key, total_count as i64))?;
                for (i, e) in scans.enumerate() {
                    let t = sink.add("TypeId", i64::from(data.expressions[*e].ty.as_u32()))?;
                    sink.add("Accumulator", (key, i as i64, t))?;
                    sink.add("ScanComponent", (key, i as i64, t))?;
                }
                for (i, e) in totals.enumerate() {
                    let t = sink.add("TypeId", i64::from(data.expressions[*e].ty.as_u32()))?;
                    sink.add("Accumulator", (key, (scan_count + i) as i64, t))?;
                    sink.add("TotalResult", (key, i as i64, t))?;
                }
                for (i, t) in body_signature(&form.post).1.into_iter().enumerate() {
                    let t = sink.add("TypeId", i64::from(t.as_u32()))?;
                    sink.add("ArrayResult", (key, (total_count + i) as i64, t))?;
                }
                for (i, t) in
                    body_signature(&form.pre).1.into_iter().skip(scan_count + total_count).enumerate()
                {
                    let t = sink.add("TypeId", i64::from(t.as_u32()))?;
                    sink.add("MappedComponent", (key, i as i64, t))?;
                }
                Some(inputs)
            }
            OperationKind::Filter {
                map, body, inputs, ..
            } => {
                let safe = safe_body(map, &summary.safe_regions) && safe_body(body, &summary.safe_regions);
                sink.add("FilterShape", (key, safe))?;
                if let Some(t) = body_signature(map).1.first() {
                    let t = sink.add("TypeId", i64::from(t.as_u32()))?;
                    sink.add("FilterResult", (key, t))?;
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
                sink.add("IndexedWrite", key)?;
                let destination = sink.add("ExprId", i64::from(destination.value.as_u32()))?;
                sink.add("UpdatedResult", (key, 0i64, destination))?;
                if matches!(op.kind, OperationKind::BucketScatter { .. }) {
                    sink.add("BucketResult", (key, destination))?;
                    // Ranked iteration is a source payload; the serial phase
                    // has one invocation regardless of rank.
                    None
                } else {
                    Some(inputs)
                }
            }
            _ => {
                sink.add("ScalarSite", key)?;
                if data.types[op.ty].ty == bool_type()
                    || type_byte_size(&data.types[op.ty].ty).is_some_and(|n| n > 0)
                {
                    let t = sink.add("TypeId", i64::from(op.ty.as_u32()))?;
                    sink.add("ScalarCandidate", (key, t))?;
                }
                None
            }
        };
        if let Some(inputs) = inputs {
            let n = extent(inputs.first(), sink)?;
            sink.add("InputDomain", (key, n))?;
        }
    }
    for r in regions {
        let region = sink.add("RegionId", i64::from(r.as_u32()))?;
        for (i, &e) in data.regions[r].results.iter().enumerate() {
            values.insert(e);
            let e = sink.add("ExprId", i64::from(e.as_u32()))?;
            sink.add("ExitValue", (region, i as i64, e))?;
        }
    }
    for (after, before) in summary.dependencies() {
        let after = sink.add("OperationId", i64::from(after.as_u32()))?;
        let before = sink.add("OperationId", i64::from(before.as_u32()))?;
        sink.add("SourceDependency", (after, before))?;
    }
    for (gate, inputs) in summary.effects.gates() {
        for op in inputs {
            let op = sink.add("OperationId", i64::from(op.as_u32()))?;
            sink.add("EffectInput", (gate as i64, op))?;
        }
    }
    for (op, gate) in summary.effects.waits() {
        let op = sink.add("OperationId", i64::from(op.as_u32()))?;
        sink.add("EffectWait", (op, gate as i64))?;
    }

    // Visit each expression once, importing only structural edges and views.
    let mut pending: Vec<_> = values.iter().copied().collect();
    while let Some(e) = pending.pop() {
        let value = &data.expressions[e];
        let key = sink.add("ExprId", i64::from(e.as_u32()))?;
        let ty = sink.add("TypeId", i64::from(value.ty.as_u32()))?;
        sink.add("SourceType", (key, ty))?;
        let mut generic_children = false;
        match &value.kind {
            ExprKind::Parameter(p) => {
                let region = sink.add("RegionId", i64::from(data.parameters[*p].region.as_u32()))?;
                sink.add("SourceParameter", (key, region))?;
                sink.set("HasInputFields", key, false)?;
            }
            ExprKind::OperationResult(op) => {
                let operation = sink.add("OperationId", i64::from(op.as_u32()))?;
                if matches!(
                    data.operations[*op].kind,
                    OperationKind::Screma { .. } | OperationKind::BucketScatter { .. }
                ) {
                    sink.add("ResultTuple", (key, operation))?;
                } else {
                    sink.add("DirectResult", (key, operation, 0i64))?;
                }
            }
            ExprKind::Project { tuple, index } => {
                let tuple = sink.add("ExprId", i64::from(tuple.as_u32()))?;
                sink.add("Projection", (key, tuple, *index as i64))?;
            }
            ExprKind::Tuple(fields) | ExprKind::Vector(fields) => {
                for (i, field) in fields.iter().enumerate() {
                    let field = sink.add("ExprId", i64::from(field.as_u32()))?;
                    sink.add("FieldValue", (key, i as i64, field))?;
                }
                generic_children = true;
            }
            ExprKind::Coerce(inner) | ExprKind::Array(Array::Value(inner)) => {
                let inner = sink.add("ExprId", i64::from(inner.as_u32()))?;
                sink.add("ForwardValue", (key, inner))?;
            }
            ExprKind::PureApp { function, args } if is_slice(data, *function) && args.len() == 3 => {
                let array = sink.add("ExprId", i64::from(args[0].as_u32()))?;
                let start = sink.add("ExprId", i64::from(args[1].as_u32()))?;
                let len = sink.add("ExprId", i64::from(args[2].as_u32()))?;
                sink.add("SliceView", (key, array, start, len))?;
            }
            _ => generic_children = true,
        }
        if generic_children && !matches!(value.kind, ExprKind::Tuple(_) | ExprKind::Vector(_)) {
            sink.add("ComputedValue", key)?;
        }
        for child in value.kind.children() {
            if generic_children {
                let child = sink.add("ExprId", i64::from(child.as_u32()))?;
                sink.add("ChildValue", (key, child))?;
            }
            if values.insert(child) {
                pending.push(child);
            }
        }
    }
    Ok(())
}

fn extent(array: Option<&Array>, sink: &mut FullState<'_, '_>) -> Result<Value, Error> {
    match array {
        Some(Array::Value(e)) => {
            let e = sink.add("ExprId", i64::from(e.as_u32()))?;
            sink.add("Length", e)
        }
        Some(Array::Zip(xs)) => extent(xs.first(), sink),
        Some(Array::Literal(xs)) => sink.add("Fixed", xs.len() as i64),
        Some(Array::Range { len, .. }) => {
            let len = sink.add("ExprId", i64::from(len.as_u32()))?;
            sink.add("Scalar", len)
        }
        None => sink.add("Fixed", 0i64),
    }
}

/// Expose source result slots once; tuple projection is structural import, not
/// an allocation decision. Use the same global identities for existing values.
pub(super) fn outputs(data: &mut Program<Scheduled>, sink: &mut FullState<'_, '_>) -> Result<(), Error> {
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
        let tuple_key = sink.add("ExprId", i64::from(tuple.as_u32()))?;
        let tuple_type = sink.add("TypeId", i64::from(data.expressions[tuple].ty.as_u32()))?;
        sink.set("HasInputFields", tuple_key, true)?;
        sink.add("SourceType", (tuple_key, tuple_type))?;
        let region = sink.add("RegionId", i64::from(region.as_u32()))?;
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
            let field = sink.add("ExprId", i64::from(field.as_u32()))?;
            let field_ty = sink.add("TypeId", i64::from(field_ty.as_u32()))?;
            sink.add("FieldValue", (tuple_key, index as i64, field))?;
            sink.add("ChildValue", (tuple_key, field))?;
            sink.add("ParameterValue", (field, region))?;
            sink.add("SourceType", (field, field_ty))?;
            sink.add("Projection", (field, tuple_key, index as i64))?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
