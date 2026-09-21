//! Import structural facts for relational scheduling. This adapter never decides
//! which values to materialize, which resources to allocate, or which phase writes
//! an output. Those decisions belong to the .egg rules.
use super::data::body_signature;
use super::visit::{Operand, OperandRole};
use crate::builtins::{by_id, Purity};
use crate::egglog::data::{
    is_slice, length_source, Array, ExprData, ExprKind, OperationKind, OutputData, SoacBody, TypeData,
    TypeId,
};
use crate::egglog::dependencies::Dependencies;
use crate::egglog::{Program, Scheduled};
use crate::interface::EntryKind;
use crate::ssa::layout::type_byte_size;
use crate::types::{
    bool_type, canonical_storage_buffer_ty, is_array_variant_bounded, strip_existentials, Type, TypeExt,
    TypeName,
};
use crate::PipelineTopologyPolicy;
use egglog_engine::{Error, FullState, RawValues, Value, Write};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(super) mod abi;
mod host_sizes;
mod read;
pub(super) use read::{number, read, rows, Readout, Recipe};

pub(super) const RULES: &str = concat!(
    include_str!("planning.egg"),
    "\n",
    include_str!("schedule.egg"),
    "\n",
    include_str!("residency.egg"),
    "\n",
    include_str!("allocation.egg"),
    "\n",
    include_str!("reuse.egg"),
    "\n",
    include_str!("dispatch.egg"),
    "\n",
);
pub(super) const KEYS: &str = "(datatype ExprKey (ExprId i64))\n(datatype TypeKey (TypeId i64))\n";
pub(super) const RUN: &str = r#"(run-schedule (seq
    (saturate (run structure))
    (saturate (run classify))
    (saturate (seq (run residency) (run schedule) (run allocation) (run dispatch)))
    (saturate (run reuse)) (run reuse-blockers)
    (run reuse-owned-input) (run reuse-input) (saturate (run reuse-output))
    (run select-reuse) (run select-allocation)
    (saturate (seq (run residency) (run allocation) (run dispatch)))
    (run materialize-outputs) (run allocation) (run select-allocation)
    (saturate (run allocation))
    (saturate (run abi)) (saturate (run abi-final))
    (run readout)))"#;

pub(super) fn facts(
    data: &Program<Scheduled>,
    summary: &Dependencies,
    count_type: TypeId,
    topology: PipelineTopologyPolicy,
    sink: &mut FullState<'_, '_>,
) -> Result<(), Error> {
    let count_type = sink.add("TypeId", i64::from(count_type.as_u32()))?;
    sink.add("CounterType", count_type)?;
    let mut values = BTreeSet::new();
    for (&id, output) in &data.state.outputs {
        values.insert(output.expression);
        let e = sink.add("ExprId", i64::from(output.expression.as_u32()))?;
        if !canonical_storage_buffer_ty(&data.types[data.expressions[output.expression].ty].ty).is_array() {
            let ty = sink.add(
                "TypeId",
                i64::from(data.expressions[output.expression].ty.as_u32()),
            )?;
            sink.add("ReturnScalar", (i64::from(id.as_u32()), ty))?;
        } else {
            sink.add("ReturnArray", (i64::from(id.as_u32()), e))?;
        }
    }
    let mut regions = BTreeSet::new();
    let symbols: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    for (&id, entry) in &data.entries {
        let region = data.definitions[entry.definition].body;
        regions.insert(region);
        let r = sink.add("RegionId", i64::from(region.as_u32()))?;
        let compute = entry.declaration.entry_kind == EntryKind::Compute;
        sink.add("SourceEntry", (i64::from(id.as_u32()), r, compute))?;
        if let Some(grid) = &entry.declaration.compute_dispatch {
            sink.add(
                "SourceGrid",
                (
                    i64::from(id.as_u32()),
                    i64::from(grid.x),
                    i64::from(grid.y),
                    i64::from(grid.z),
                ),
            )?;
        }
        let root = sink.add("EntryRoot", i64::from(id.as_u32()))?;
        for &e in &data.regions[region].results {
            values.insert(e);
            let e = sink.add("AbiExpr", i64::from(e.as_u32()))?;
            sink.add("AbiRootNeed", (root, e))?;
        }
        if !compute || topology == PipelineTopologyPolicy::AuthoredOnly {
            sink.add("DeviceRegion", r)?;
        }
        if compute && topology == PipelineTopologyPolicy::AllowGenerated {
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
        if let Some(array) = length_source(data, &op.kind) {
            let result = sink.add("Result", (key, 0i64))?;
            let result = sink.add("AbiResource", result)?;
            let array = sink.add("AbiExpr", i64::from(array.as_u32()))?;
            let length = sink.add("AbiLength", array)?;
            sink.add("AbiAlias", (result, length))?;
        }
        if let OperationKind::If {
            then_region,
            else_region,
            ..
        } = &op.kind
        {
            let result = sink.add("Result", (key, 0i64))?;
            let result = sink.add("AbiResource", result)?;
            let yes = sink.add(
                "AbiExpr",
                i64::from(data.regions[*then_region].results[0].as_u32()),
            )?;
            let no = sink.add(
                "AbiExpr",
                i64::from(data.regions[*else_region].results[0].as_u32()),
            )?;
            sink.add("AbiChoice", (result, yes, no))?;
        }
        for r in op.kind.structured_regions() {
            regions.insert(r);
            let r = sink.add("RegionId", i64::from(r.as_u32()))?;
            sink.add("Enters", (key, r))?;
        }
        match &op.kind {
            OperationKind::If { .. } | OperationKind::Loop { .. } => {
                sink.add("SourceControl", key)?;
            }
            OperationKind::Call { .. } | OperationKind::EvalGlobal(_) => {
                sink.add("SourceCall", key)?;
                sink.add("HostEvaluation", key)?;
            }
            OperationKind::Index { .. } => {
                sink.add("HostEvaluation", key)?;
            }
            _ => {}
        }
        for body in op.kind.callbacks() {
            if let SoacBody::Apply { region, .. } = body {
                regions.insert(*region);
                let region = sink.add("RegionId", i64::from(region.as_u32()))?;
                sink.add("Callback", (key, region))?;
                sink.add("DeviceRegion", region)?;
            }
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
        let effectful = match &op.kind {
            OperationKind::Scatter { .. }
            | OperationKind::BucketScatter { .. }
            | OperationKind::ReduceByIndex { .. } => true,
            OperationKind::Call { function, .. } if called.is_none() => {
                !matches!(data.expressions[*function].kind, ExprKind::Builtin(id)
                    if by_id(data.builtins[id].builtin).raw.purity == Purity::Pure)
            }
            OperationKind::EvalGlobal(_) => called.is_none(),
            _ => false,
        };
        if effectful {
            sink.add("ParallelEffect", key)?;
        }
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
            OperationKind::Screma {
                form,
                inputs,
                reuse_inputs,
            } => {
                sink.add(
                    "CollectiveShape",
                    (key, !form.scans.is_empty(), !form.reductions.is_empty()),
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
                    if let Some(Some(input)) = reuse_inputs.get(total_count + i) {
                        if let Some(Array::Value(e)) = inputs.get(*input) {
                            if data.types[data.expressions[*e].ty].ty.elem_type() == Some(&data.types[t].ty)
                            {
                                let e = sink.add("ExprId", i64::from(e.as_u32()))?;
                                sink.add("ReusePermission", (key, (total_count + i) as i64, e))?;
                            }
                        }
                    }
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
            OperationKind::Filter { map, inputs, .. } => {
                sink.add("FilterShape", key)?;
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
                if let OperationKind::ReduceByIndex { body, .. } = &op.kind {
                    let safe = matches!(
                        data.types[destination.elem_ty].ty,
                        Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _)
                    );
                    let update = sink.add(reducer_operator(data, body), RawValues(vec![]))?;
                    sink.add("IndexedReducer", (key, safe, update))?;
                } else if let OperationKind::BucketScatter { shape, .. } = &op.kind {
                    let shape = &data.bucket_shapes[*shape];
                    let mut axes = vec![None; usize::from(shape.domain_rank)];
                    for (input, mapping) in inputs.iter().zip(&shape.input_dimensions) {
                        for (axis, &domain_axis) in mapping.iter().enumerate() {
                            axes[usize::from(domain_axis)] = if axis == 0 {
                                Some(extent(Some(input), sink)?)
                            } else if let Some(n) = inner_dimension(input, axis, data) {
                                Some(sink.add("Fixed", n as i64)?)
                            } else {
                                None
                            };
                        }
                    }
                    let known = axes.iter().all(Option::is_some);
                    sink.add("BucketShape", (key, known))?;
                    if known {
                        let mut n = sink.add("Fixed", 1i64)?;
                        for axis in axes.into_iter().flatten() {
                            n = sink.add("Product", (n, axis))?;
                        }
                        sink.add("InputDomain", (key, n))?;
                    }
                } else {
                    sink.add("IndexedWrite", key)?;
                }
                let destination_expr = sink.add("ExprId", i64::from(destination.value.as_u32()))?;
                if matches!(op.kind, OperationKind::Scatter { initialize: true, .. }) {
                    let ty = sink.add("TypeId", i64::from(destination.elem_ty.as_u32()))?;
                    let n = sink.add("Length", destination_expr)?;
                    sink.add("InitializedResult", (key, 0i64, ty, n))?;
                } else {
                    sink.add("UpdatedResult", (key, 0i64, destination_expr))?;
                }
                if matches!(op.kind, OperationKind::BucketScatter { .. }) {
                    sink.add("BucketResult", (key, destination_expr))?;
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
        for &e in &data.regions[r].results {
            values.insert(e);
            let e = sink.add("ExprId", i64::from(e.as_u32()))?;
            sink.add("ExitValue", (region, e))?;
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
        // Source size and interface leaves supplement the structural facts.
        let abi_value = sink.add("AbiExpr", i64::from(e.as_u32()))?;
        if crate::types::as_soa_tuple(strip_existentials(&data.types[value.ty].ty)).is_some() {
            // A logical tuple array has the length of its first component.
            // Import that relationship even when its fields are literals or
            // parameters, rather than results with a planned LiveLength.
            let first = sink.add("AbiField", (abi_value, 0i64))?;
            let length = sink.add("AbiLength", first)?;
            sink.add("AbiArrayLength", (abi_value, length))?;
        }
        if let Some(Type::Constructed(TypeName::Size(n), _)) =
            strip_existentials(&data.types[value.ty].ty).array_size()
        {
            let n = sink.add("AbiNumber", *n as i64)?;
            let relation = if strip_existentials(&data.types[value.ty].ty)
                .array_variant()
                .is_some_and(is_array_variant_bounded)
            {
                "AbiArrayBound"
            } else {
                "AbiArrayLength"
            };
            sink.add(relation, (abi_value, n))?;
        }
        match &value.kind {
            ExprKind::Parameter(p) => {
                let p = sink.add("AbiParameter", i64::from(p.as_u32()))?;
                sink.add("AbiAlias", (abi_value, p))?;
            }
            ExprKind::Int(n) => {
                if let Some(n) = n.parse::<i64>().ok().filter(|&n| n >= 0) {
                    let n = sink.add("AbiNumber", n)?;
                    sink.add("AbiAlias", (abi_value, n))?;
                }
            }
            ExprKind::PureApp { function, args } if args.len() == 2 => {
                if let ExprKind::BinOp(op) = &data.expressions[*function].kind {
                    let operation = match op.as_str() {
                        "+" => Some("add"),
                        "-" => Some("sub"),
                        "*" => Some("mul"),
                        _ => None,
                    };
                    let signed = match &data.types[value.ty].ty {
                        Type::Constructed(TypeName::Int(32), _) => Some(true),
                        Type::Constructed(TypeName::UInt(32), _) => Some(false),
                        _ => None,
                    };
                    if let (Some(operation), Some(signed)) = (operation, signed) {
                        let left = sink.add("AbiExpr", i64::from(args[0].as_u32()))?;
                        let right = sink.add("AbiExpr", i64::from(args[1].as_u32()))?;
                        let operation = format!("{}_{operation}", if signed { "i32" } else { "u32" });
                        let result = sink.add("AbiBinary", (operation, left, right))?;
                        sink.add("AbiAlias", (abi_value, result))?;
                    }
                }
            }
            ExprKind::Array(array) => {
                let n = extent(Some(array), sink)?;
                let n = sink.add("AbiExtent", n)?;
                sink.add("AbiArrayLength", (abi_value, n))?;
            }
            _ => {}
        }
        let mut generic_children = false;
        match &value.kind {
            ExprKind::Parameter(_) => {
                sink.add("SourceParameter", key)?;
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

fn reducer_operator(data: &Program<Scheduled>, body: &SoacBody) -> &'static str {
    let SoacBody::Apply { region, captures, .. } = body else {
        return "AtomicCas";
    };
    let region = &data.regions[*region];
    let [result] = region.results.as_slice() else {
        return "AtomicCas";
    };
    let ExprKind::PureApp { function, args } = &data.expressions[*result].kind else {
        return "AtomicCas";
    };
    let [left, right] = args.as_slice() else {
        return "AtomicCas";
    };
    let (ExprKind::Parameter(a), ExprKind::Parameter(b)) =
        (&data.expressions[*left].kind, &data.expressions[*right].kind)
    else {
        return "AtomicCas";
    };
    if !captures.is_empty()
        || region.parameters.len() != 2
        || a == b
        || !region.parameters.contains(a)
        || !region.parameters.contains(b)
    {
        return "AtomicCas";
    }
    match &data.expressions[*function].kind {
        ExprKind::BinOp(op) if op == "+" => "AtomicAdd",
        ExprKind::BinOp(op) if op == "&" => "AtomicAnd",
        ExprKind::BinOp(op) if op == "|" => "AtomicOr",
        ExprKind::BinOp(op) if op == "^" => "AtomicXor",
        _ => "AtomicCas",
    }
}

pub(super) fn extent(array: Option<&Array>, sink: &mut FullState<'_, '_>) -> Result<Value, Error> {
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

fn inner_dimension(array: &Array, axis: usize, data: &Program<Scheduled>) -> Option<usize> {
    let mut ty = match array {
        Array::Value(e) => &data.types[data.expressions[*e].ty].ty,
        Array::Zip(inputs) => return inner_dimension(inputs.first()?, axis, data),
        Array::Literal(items) => {
            return inner_dimension(&Array::Value(*items.first()?), axis - 1, data);
        }
        Array::Range { .. } => return None,
    };
    let mut axis = axis;
    loop {
        if let Type::Constructed(TypeName::Tuple(_), fields) = ty {
            ty = fields.first()?;
            continue;
        }
        let dims = ty.array_dims()?;
        if axis < dims.len() {
            let Type::Constructed(TypeName::Size(n), _) = dims[axis] else {
                return None;
            };
            return Some(n);
        }
        axis -= dims.len();
        ty = ty.elem_type()?;
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
                let ty = canonical_storage_buffer_ty(&data.types[data.expressions[expression].ty].ty);
                let output = data.state.outputs.alloc(OutputData {
                    entry,
                    index,
                    expression,
                    buffer: None,
                    copy: false,
                });
                sink.add(
                    "OutputOwner",
                    (i64::from(output.as_u32()), i64::from(entry.as_u32())),
                )?;
                if ty.is_array() {
                    if let Some(element) = ty.elem_type() {
                        let element = *types
                            .entry(element.clone())
                            .or_insert_with(|| data.ir.types.alloc(TypeData { ty: element.clone() }));
                        let element = sink.add("TypeId", i64::from(element.as_u32()))?;
                        let expression = sink.add("ExprId", i64::from(expression.as_u32()))?;
                        let length = sink.add("Length", expression)?;
                        sink.add(
                            "ReturnedArrayLayout",
                            (i64::from(output.as_u32()), element, length),
                        )?;
                    }
                }
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
        let tuple_key = sink.add("ExprId", i64::from(tuple.as_u32()))?;
        sink.set("HasInputFields", tuple_key, true)?;
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
            sink.add("FieldValue", (tuple_key, index as i64, field))?;
            sink.add("ChildValue", (tuple_key, field))?;
            sink.add("ParameterValue", field)?;
            sink.add("Projection", (field, tuple_key, index as i64))?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
