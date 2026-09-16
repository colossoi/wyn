//! Sidecar algebra for fusion decisions. Egglog sees ports and legality facts;
//! scalar bodies remain parameterized regions or compositions of opaque bodies.

use super::data::*;
use crate::types::{self, TypeExt};
use std::collections::{BTreeMap, BTreeSet};

mod envelope;
mod filter;
mod indexed;
mod project;
mod rewrite;
mod slices;
pub(super) use envelope::{envelope, memory_compatible};
pub(super) use filter::{length_source, masked};
pub(super) use indexed::{demands, indexed};

pub(super) fn ty(data: &mut AssociatedData, value: types::Type) -> TypeId {
    if let Some((&id, _)) = data.types.iter().find(|(_, t)| t.ty == value) {
        return id;
    }
    data.types.alloc(TypeData { ty: value })
}
pub(super) fn expr(data: &mut AssociatedData, ty: TypeId, kind: ExprKind) -> ExprId {
    let value = ExprData { ty, kind };
    if let Some((&id, _)) = data.expressions.iter().find(|(_, e)| **e == value) {
        return id;
    }
    data.expressions.alloc(value)
}
pub(super) fn signature(body: &SoacBody) -> (Vec<TypeId>, Vec<TypeId>) {
    match body {
        SoacBody::Apply {
            parameters, results, ..
        } => (parameters.clone(), results.clone()),
        SoacBody::Identity(ts) => (ts.clone(), ts.clone()),
        SoacBody::Route { parameters, indices } => (
            parameters.clone(),
            indices.iter().map(|&i| parameters[i]).collect(),
        ),
        SoacBody::Compose { first, then } => (signature(first).0, signature(then).1),
        SoacBody::Parallel { left, right } => {
            let (p, mut r) = signature(left);
            r.extend(signature(right).1);
            (p, r)
        }
    }
}
fn compose(first: SoacBody, then: SoacBody) -> SoacBody {
    if matches!(first, SoacBody::Identity(_)) {
        return then;
    }
    if matches!(then, SoacBody::Identity(_)) {
        return first;
    }
    SoacBody::Compose {
        first: Box::new(first),
        then: Box::new(then),
    }
}
fn route(types: &[TypeId], indices: Vec<usize>) -> SoacBody {
    if indices == (0..types.len()).collect::<Vec<_>>() {
        SoacBody::Identity(types.to_vec())
    } else {
        SoacBody::Route {
            parameters: types.to_vec(),
            indices,
        }
    }
}

/// A small wiring builder: invoke a body once, retain its results, then route
/// arbitrary inputs/results. It never inspects scalar expression syntax.
struct Wiring {
    body: SoacBody,
    types: Vec<TypeId>,
}
impl Wiring {
    fn new(types: Vec<TypeId>) -> Self {
        Self {
            body: SoacBody::Identity(types.clone()),
            types,
        }
    }
    fn call(&mut self, body: SoacBody, args: Vec<usize>) -> Vec<usize> {
        let results = signature(&body).1;
        let start = self.types.len();
        let call = compose(route(&self.types, args), body);
        self.body = compose(
            self.body.clone(),
            SoacBody::Parallel {
                left: Box::new(SoacBody::Identity(self.types.clone())),
                right: Box::new(call),
            },
        );
        self.types.extend(&results);
        (start..self.types.len()).collect()
    }
    fn finish(self, results: Vec<usize>) -> SoacBody {
        compose(self.body, route(&self.types, results))
    }
}

#[derive(Clone, Debug)]
enum Input {
    External(Array),
    Produced(usize, Vec<(ExprId, ExprId)>),
    Tuple(Vec<Input>),
}
fn input(data: &AssociatedData, array: &Array, producer: Option<OperationId>) -> Input {
    match array {
        Array::Zip(arrays) => Input::Tuple(arrays.iter().map(|a| input(data, a, producer)).collect()),
        Array::Value(id) => match &data.expressions[*id].kind {
            ExprKind::OperationResult(op) if Some(*op) == producer => Input::Produced(0, vec![]),
            ExprKind::Coerce(v) => input(data, &Array::Value(*v), producer),
            ExprKind::Array(a) => input(data, a, producer),
            ExprKind::Project { tuple, index } if matches!(data.expressions[*tuple].kind, ExprKind::OperationResult(op) if Some(op) == producer) => {
                Input::Produced(*index, vec![])
            }
            ExprKind::PureApp { function, args } if slices::is_slice(data, *function) => {
                if let [base, start, end] = args.as_slice() {
                    if let Input::Produced(slot, mut transforms) =
                        input(data, &Array::Value(*base), producer)
                    {
                        transforms.push((*start, *end));
                        return Input::Produced(slot, transforms);
                    }
                }
                Input::External(array.clone())
            }
            _ => Input::External(array.clone()),
        },
        _ => Input::External(array.clone()),
    }
}
fn flatten(tree: &Input, arrays: &mut Vec<Array>) {
    match tree {
        Input::External(a) => {
            if !arrays.contains(a) {
                arrays.push(a.clone());
            }
        }
        Input::Tuple(xs) => {
            for x in xs {
                flatten(x, arrays);
            }
        }
        Input::Produced(..) => {}
    }
}
fn element(data: &mut AssociatedData, array: &Array) -> Option<TypeId> {
    match array {
        Array::Value(id) => {
            let array_ty = types::canonical_storage_buffer_ty(&data.types[data.expressions[*id].ty].ty);
            Some(ty(data, array_ty.elem_type()?.clone()))
        }
        Array::Literal(xs) => Some(data.expressions[*xs.first()?].ty),
        Array::Range { start, .. } => Some(data.expressions[*start].ty),
        Array::Zip(xs) => {
            let ts = xs.iter().map(|x| element(data, x)).collect::<Option<Vec<_>>>()?;
            Some(ty(
                data,
                types::tuple(ts.iter().map(|t| data.types[*t].ty.clone()).collect()),
            ))
        }
    }
}
fn wire_input(
    data: &mut AssociatedData,
    region: RegionId,
    wiring: &mut Wiring,
    tree: &Input,
    arrays: &[Array],
    produced: &[usize],
) -> Option<usize> {
    wire_input_at(data, region, wiring, tree, arrays, produced, 0)
}
fn wire_input_at(
    data: &mut AssociatedData,
    region: RegionId,
    wiring: &mut Wiring,
    tree: &Input,
    arrays: &[Array],
    produced: &[usize],
    base: usize,
) -> Option<usize> {
    match tree {
        Input::External(a) => arrays.iter().position(|x| x == a).map(|i| i + base),
        Input::Produced(i, _) => produced.get(*i).copied(),
        Input::Tuple(xs) => {
            let args = xs
                .iter()
                .map(|x| wire_input_at(data, region, wiring, x, arrays, produced, base))
                .collect::<Option<Vec<_>>>()?;
            let types = args.iter().map(|&i| wiring.types[i]).collect();
            let body = project::tuple(data, region, types);
            Some(wiring.call(body, args)[0])
        }
    }
}
fn counts(form: &ScremaForm) -> (usize, usize) {
    (
        form.scans.iter().map(|s| s.neutral.len()).sum(),
        form.reductions.iter().map(|r| r.neutral.len()).sum(),
    )
}
fn result_types(data: &mut AssociatedData, op: OperationId) -> Vec<TypeId> {
    // Operation results are always a tuple, even for a single logical result.
    let types::Type::Constructed(types::TypeName::Tuple(_), fields) =
        data.types[data.operations[op].ty].ty.clone()
    else {
        return vec![];
    };
    fields.into_iter().map(|t| ty(data, t)).collect()
}
pub(super) fn routes(
    data: &AssociatedData,
    producer: OperationId,
    consumer: OperationId,
) -> BTreeSet<usize> {
    fn visit(data: &AssociatedData, producer: OperationId, i: Input, out: &mut BTreeSet<usize>) -> bool {
        match i {
            Input::Produced(i, _) => {
                out.insert(i);
                true
            }
            Input::Tuple(xs) => xs.into_iter().all(|x| visit(data, producer, x, out)),
            Input::External(a) => {
                let mut refs = BTreeSet::new();
                project::array_references(data, &a, &mut refs);
                !refs.iter().any(|v| matches!(data.expressions[*v].kind, ExprKind::OperationResult(op) if op == producer))
            }
        }
    }
    let mut out = BTreeSet::new();
    if input_slices(data, producer, consumer).is_none() {
        return out;
    }
    for a in inputs(&data.operations[consumer].kind) {
        // A partially routed input would leave a self-dependency after fusion.
        if !visit(data, producer, input(data, a, Some(producer)), &mut out) {
            return BTreeSet::new();
        }
    }
    out
}
pub(super) fn input_slices(
    data: &AssociatedData,
    producer: OperationId,
    consumer: OperationId,
) -> Option<Vec<(ExprId, ExprId)>> {
    let mut selected = None;
    fn visit(tree: Input, selected: &mut Option<Vec<(ExprId, ExprId)>>) -> Option<()> {
        match tree {
            Input::Produced(_, transforms) => {
                if selected.as_ref().is_some_and(|old| *old != transforms) {
                    return None;
                }
                *selected = Some(transforms);
            }
            Input::Tuple(xs) => {
                for x in xs {
                    visit(x, selected)?;
                }
            }
            _ => {}
        }
        Some(())
    }
    for a in inputs(&data.operations[consumer].kind) {
        visit(input(data, a, Some(producer)), &mut selected)?;
    }
    Some(selected.unwrap_or_default())
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

/// Prove equal iteration domains without assuming unrelated runtime lengths are equal.
pub(super) fn same_domain(data: &AssociatedData, a: OperationId, b: OperationId) -> bool {
    fn domain(data: &AssociatedData, a: &Array) -> Array {
        match a {
            Array::Zip(xs) if !xs.is_empty() => domain(data, &xs[0]),
            Array::Value(id) => match &data.expressions[*id].kind {
                ExprKind::Coerce(v) => domain(data, &Array::Value(*v)),
                ExprKind::Array(a) => domain(data, a),
                ExprKind::Project { tuple, .. } => {
                    if let ExprKind::OperationResult(op) = data.expressions[*tuple].kind {
                        if let OperationKind::Screma { inputs, .. } = &data.operations[op].kind {
                            if let Some(first) = inputs.first() {
                                return domain(data, first);
                            }
                        }
                        a.clone()
                    } else {
                        a.clone()
                    }
                }
                _ => a.clone(),
            },
            _ => a.clone(),
        }
    }
    let aa = inputs(&data.operations[a].kind);
    let bb = inputs(&data.operations[b].kind);
    let (Some(a), Some(b)) = (aa.first(), bb.first()) else {
        return false;
    };
    let (a, b) = (domain(data, a), domain(data, b));
    if a == b {
        return true;
    }
    fn fixed(data: &AssociatedData, a: &Array) -> Option<u64> {
        match a {
            Array::Literal(xs) => Some(xs.len() as u64),
            Array::Value(id) => match data.types[data.expressions[*id].ty].ty.array_size() {
                Some(types::Type::Constructed(types::TypeName::Size(n), _)) => Some(*n as u64),
                _ => None,
            },
            _ => None,
        }
    }
    fixed(data, &a).is_some_and(|n| Some(n) == fixed(data, &b))
}

/// Construct one combined Screma and the mapping from each old result slot to
/// its new slot. Producer outputs with observers survive the contraction.
pub(super) fn scremas(
    data: &mut AssociatedData,
    producer: OperationId,
    consumer: OperationId,
    horizontal: bool,
    retain: bool,
) -> Option<()> {
    let OperationKind::Screma {
        form: a,
        inputs: mut ai,
        ..
    } = data.operations[producer].kind.clone()
    else {
        return None;
    };
    let OperationKind::Screma {
        form: b, inputs: bi, ..
    } = data.operations[consumer].kind.clone()
    else {
        return None;
    };
    let region = data.operations[consumer].region;
    let (sa, ra) = counts(&a);
    let (sb, rb) = counts(&b);
    if !horizontal {
        let transforms = input_slices(data, producer, consumer)?;
        if !transforms.is_empty() {
            if retain || sa + ra != 0 {
                return None;
            }
            ai = ai.iter().map(|a| slices::apply(data, a, &transforms)).collect::<Option<_>>()?;
        }
    }
    let at = signature(&a.pre).1;
    let bt = signature(&b.pre).1;
    let ap = signature(&a.post).1;
    let bp = signature(&b.post).1;
    let trees_a: Vec<_> = ai.iter().map(|i| input(data, i, None)).collect();
    let trees_b: Vec<_> = bi.iter().map(|i| input(data, i, (!horizontal).then_some(producer))).collect();
    let mut arrays = vec![];
    for t in trees_a.iter().chain(&trees_b) {
        flatten(t, &mut arrays);
    }
    let parameters = arrays.iter().map(|a| element(data, a)).collect::<Option<Vec<_>>>()?;
    let mut pre = Wiring::new(parameters.clone());
    let args = trees_a
        .iter()
        .map(|t| wire_input(data, region, &mut pre, t, &arrays, &[]))
        .collect::<Option<Vec<_>>>()?;
    let av = pre.call(a.pre.clone(), args);
    let mut origins = vec![];
    let (pre, post) = if horizontal {
        let args = trees_b
            .iter()
            .map(|t| wire_input(data, region, &mut pre, t, &arrays, &[]))
            .collect::<Option<Vec<_>>>()?;
        let bv = pre.call(b.pre.clone(), args);
        let order = av[..sa]
            .iter()
            .chain(&bv[..sb])
            .chain(&av[sa..sa + ra])
            .chain(&bv[sb..sb + rb])
            .chain(&av[sa + ra..])
            .chain(&bv[sb + rb..])
            .copied()
            .collect();
        let params =
            at[..sa].iter().chain(&bt[..sb]).chain(&at[sa + ra..]).chain(&bt[sb + rb..]).copied().collect();
        let mut post = Wiring::new(params);
        let aa = (0..sa).chain(sa + sb..sa + sb + at.len() - sa - ra).collect();
        let pa = post.call(a.post.clone(), aa);
        let ba = (sa..sa + sb)
            .chain(sa + sb + at.len() - sa - ra..sa + sb + at.len() - sa - ra + bt.len() - sb - rb)
            .collect();
        let pb = post.call(b.post.clone(), ba);
        origins.extend((0..ra).map(|i| (producer, i)));
        origins.extend((0..rb).map(|i| (consumer, i)));
        origins.extend((0..ap.len()).map(|i| (producer, ra + i)));
        origins.extend((0..bp.len()).map(|i| (consumer, rb + i)));
        (pre.finish(order), post.finish(pa.into_iter().chain(pb).collect()))
    } else if sa == 0 {
        // Reduction-bearing producers use the same path as ordinary maps.
        let produced = pre.call(a.post.clone(), av[ra..].to_vec());
        let mut slots = vec![usize::MAX; ra];
        slots.extend(&produced);
        let args = trees_b
            .iter()
            .map(|t| wire_input(data, region, &mut pre, t, &arrays, &slots))
            .collect::<Option<Vec<_>>>()?;
        if args.contains(&usize::MAX) {
            return None;
        }
        let bv = pre.call(b.pre.clone(), args);
        let kept = if retain { produced } else { vec![] };
        let order = bv[..sb]
            .iter()
            .chain(&av[..ra])
            .chain(&bv[sb..sb + rb])
            .chain(&bv[sb + rb..])
            .chain(&kept)
            .copied()
            .collect();
        let params =
            signature(&b.post).0.into_iter().chain(if retain { ap.clone() } else { vec![] }).collect();
        let mut post = Wiring::new(params);
        let cb = post.call(b.post.clone(), (0..signature(&b.post).0.len()).collect());
        let keep =
            (signature(&b.post).0.len()..signature(&b.post).0.len() + kept.len()).collect::<Vec<_>>();
        origins.extend((0..ra).map(|i| (producer, i)));
        origins.extend((0..rb + bp.len()).map(|i| (consumer, i)));
        if retain {
            origins.extend((0..ap.len()).map(|i| (producer, ra + i)));
        }
        (
            pre.finish(order),
            post.finish(cb.into_iter().chain(keep).collect()),
        )
    } else {
        return project::across_barrier(data, producer, consumer, retain);
    };
    let form = ScremaForm {
        pre,
        scans: if horizontal { a.scans.into_iter().chain(b.scans).collect() } else { b.scans },
        reductions: a.reductions.into_iter().chain(b.reductions).collect(),
        post,
    };
    install(data, producer, consumer, form, arrays, origins)
}

fn install(
    data: &mut AssociatedData,
    producer: OperationId,
    consumer: OperationId,
    form: ScremaForm,
    inputs: Vec<Array>,
    origins: Vec<(OperationId, usize)>,
) -> Option<()> {
    let old_expressions: Vec<_> = data.expressions.iter().map(|(&id, e)| (id, e.clone())).collect();
    let a = result_types(data, producer);
    let b = result_types(data, consumer);
    let fields = origins
        .iter()
        .map(|&(op, i)| if op == producer { a.get(i).copied() } else { b.get(i).copied() })
        .collect::<Option<Vec<_>>>()?;
    let result_ty = ty(
        data,
        types::tuple(fields.iter().map(|t| data.types[*t].ty.clone()).collect()),
    );
    let result = expr(data, result_ty, ExprKind::OperationResult(consumer));
    let mut substitutions = BTreeMap::new();
    for (old, ts) in [(producer, a), (consumer, b)] {
        let mut values = vec![];
        for (i, t) in ts.iter().enumerate() {
            let Some(slot) = origins.iter().position(|&x| x == (old, i)) else {
                continue;
            };
            values.push(expr(
                data,
                *t,
                ExprKind::Project {
                    tuple: result,
                    index: slot,
                },
            ));
        }
        if values.len() != ts.len() {
            continue;
        }
        let old_ty = data.operations[old].ty;
        let replacement = expr(data, old_ty, ExprKind::Tuple(values));
        for (id, e) in &old_expressions {
            if e.kind == ExprKind::OperationResult(old) && *id != result {
                substitutions.insert(*id, replacement);
            }
        }
    }
    // Whole-result substitution alone cannot represent an absorbed array slot.
    // Rewrite every surviving projection directly, including the consumer's old
    // result expression when its ID was reused by the interner.
    for (id, e) in old_expressions {
        if let ExprKind::Project { tuple, index } = e.kind {
            if let ExprKind::OperationResult(op) = data.expressions[tuple].kind {
                if let Some(slot) = origins.iter().position(|&x| x == (op, index)) {
                    let replacement = expr(
                        data,
                        e.ty,
                        ExprKind::Project {
                            tuple: result,
                            index: slot,
                        },
                    );
                    if replacement != id {
                        substitutions.insert(id, replacement);
                    }
                }
            }
        }
    }
    data.operations[consumer].ty = result_ty;
    data.operations[consumer].kind = OperationKind::Screma {
        form,
        inputs,
        ownership: vec![types::SoacOwnership::Fresh; origins.len()],
    };
    data.regions[data.operations[producer].region].members.remove(&producer);
    rewrite::all(data, &substitutions);
    Some(())
}
