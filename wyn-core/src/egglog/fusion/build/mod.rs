//! Materialize the completed plan. Construction helpers stay inside this module.
use super::analysis::{counts, input_slices, references, routes};
use super::plan::Step;
use crate::egglog::data::{
    body_signature as signature, intern_expr as expr, intern_type as ty, is_slice, value_source, Array,
    AssociatedData, ExprId, ExprKind, OperationId, OperationKind, RegionId, ScremaForm, SoacBody, TypeId,
};
use crate::egglog::{rewrite, OptimizeError};
use crate::types::{self, TypeExt};
use std::collections::{BTreeMap, BTreeSet};

mod body;
mod envelope;
mod filter;
mod indexed;
mod slices;
use body::{finish, invoke, region};
use envelope::envelope;
use filter::masked;
use indexed::indexed;

#[derive(Clone, Debug)]
enum Input {
    External(Array),
    Produced(usize, Vec<(ExprId, ExprId)>),
    Tuple(Vec<Input>),
}
fn input(data: &AssociatedData, array: &Array, producer: Option<OperationId>) -> Input {
    match array {
        Array::Zip(arrays) => Input::Tuple(arrays.iter().map(|a| input(data, a, producer)).collect()),
        Array::Value(id) => match &data.expressions[value_source(data, *id)].kind {
            ExprKind::OperationResult(op) if Some(*op) == producer => Input::Produced(0, vec![]),
            ExprKind::Coerce(v) => input(data, &Array::Value(*v), producer),
            ExprKind::Array(a) => input(data, a, producer),
            ExprKind::Project { tuple, index } if matches!(data.expressions[*tuple].kind, ExprKind::OperationResult(op) if Some(op) == producer) => {
                Input::Produced(*index, vec![])
            }
            ExprKind::PureApp { function, args } if is_slice(data, *function) => {
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

pub(super) fn apply_step(
    data: &mut AssociatedData,
    Step {
        family,
        region,
        producer,
        consumer,
        retained,
        lengths,
        demands,
    }: Step,
) -> Result<(), OptimizeError> {
    let valid = data
        .regions
        .get(region)
        .is_some_and(|scope| scope.members.contains(&producer) && scope.members.contains(&consumer));
    if !valid || (producer == consumer && family != 4 && family != 5) {
        return Err(OptimizeError::Extraction(
            "fusion candidate does not belong to the selected graph".into(),
        ));
    }
    match family {
        0 | 1 => scremas(data, producer, consumer, family == 1, retained),
        2 => envelope(data, producer, consumer),
        3 | 5 => masked(data, producer, consumer, &lengths),
        4 => indexed(data, producer, &demands),
        _ => None,
    }.ok_or_else(|| OptimizeError::Extraction(format!(
        "fusion body composition disagrees with its legality facts: family {family}, {producer:?} -> {consumer:?}"
    )))
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
    let value = element_type(data, array)?;
    Some(ty(data, value))
}
fn element_type(data: &AssociatedData, array: &Array) -> Option<types::Type> {
    match array {
        Array::Value(id) => {
            let array_ty = types::canonical_storage_buffer_ty(&data.types[data.expressions[*id].ty].ty);
            array_ty.elem_type().cloned()
        }
        Array::Literal(xs) => Some(data.types[data.expressions[*xs.first()?].ty].ty.clone()),
        Array::Range { start, .. } => Some(data.types[data.expressions[*start].ty].ty.clone()),
        Array::Zip(xs) => Some(types::tuple(
            xs.iter().map(|x| element_type(data, x)).collect::<Option<_>>()?,
        )),
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
            let body = body::tuple(data, region, types);
            Some(wiring.call(body, args)[0])
        }
    }
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
/// Construct one combined Screma and the mapping from each old result slot to
/// its new slot. Producer outputs with observers survive the contraction.
fn scremas(
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
        return across_barrier(data, producer, consumer, retain);
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
fn input_value(
    data: &mut AssociatedData,
    tree: &Input,
    arrays: &[Array],
    external: &[ExprId],
    produced: &[ExprId],
) -> Option<ExprId> {
    match tree {
        Input::External(a) => external.get(arrays.iter().position(|x| x == a)?).copied(),
        Input::Produced(i, _) => produced.get(*i).copied(),
        Input::Tuple(xs) => {
            let vs = xs
                .iter()
                .map(|x| input_value(data, x, arrays, external, produced))
                .collect::<Option<Vec<_>>>()?;
            let ts = vs.iter().map(|v| data.types[data.expressions[*v].ty].ty.clone()).collect();
            let t = ty(data, types::tuple(ts));
            Some(expr(data, t, ExprKind::Tuple(vs)))
        }
    }
}
fn across_barrier(
    data: &mut AssociatedData,
    producer: OperationId,
    consumer: OperationId,
    retain: bool,
) -> Option<()> {
    let OperationKind::Screma {
        form: a, inputs: ai, ..
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
    let parent = data.operations[consumer].region;
    let (sa, ra) = counts(&a);
    let (sb, rb) = counts(&b);
    let at = signature(&a.pre).1;
    let bt = signature(&b.pre).1;
    let trees_a: Vec<_> = ai.iter().map(|i| input(data, i, None)).collect();
    let trees_b: Vec<_> = bi.iter().map(|i| input(data, i, Some(producer))).collect();
    let mut arrays = vec![];
    for t in trees_a.iter().chain(&trees_b) {
        flatten(t, &mut arrays);
    }
    let params = arrays.iter().map(|a| element(data, a)).collect::<Option<Vec<_>>>()?;
    let mut pre = Wiring::new(params.clone());
    let aa = trees_a
        .iter()
        .map(|t| wire_input(data, parent, &mut pre, t, &arrays, &[]))
        .collect::<Option<Vec<_>>>()?;
    let av = pre.call(a.pre.clone(), aa);
    let collective = if sb + rb == 0 {
        vec![]
    } else {
        let projection_params: Vec<_> = at[sa + ra..].iter().chain(&params).copied().collect();
        let (r, args) = region(data, parent, &projection_params);
        let (_, holes) = region(data, parent, &at[..sa]);
        let args_a = holes.iter().chain(&args[..at.len() - sa - ra]).copied().collect();
        let produced = invoke(data, &a.post, args_a)?;
        if ra > 0 && routes(data, producer, consumer).iter().any(|&r| r < ra) {
            return None;
        }
        // Reduction result slots are never selected by a streamed input.
        let mut outputs = vec![args[0]; ra];
        outputs.extend(produced);
        let external = &args[at.len() - sa - ra..];
        let args_b = trees_b
            .iter()
            .map(|t| input_value(data, t, &arrays, external, &outputs))
            .collect::<Option<Vec<_>>>()?;
        let values = invoke(data, &b.pre, args_b)?;
        let selected = values[..sb + rb].to_vec();
        let mut refs = BTreeSet::new();
        for &v in &selected {
            references(data, v, &mut refs);
        }
        if holes.iter().any(|h| refs.contains(h)) {
            return None;
        }
        let body = finish(data, r, projection_params, selected);
        pre.call(
            body,
            av[sa + ra..].iter().copied().chain(0..params.len()).collect(),
        )
    };
    let pre_outputs = av[..sa]
        .iter()
        .chain(&collective[..sb])
        .chain(&av[sa..sa + ra])
        .chain(&collective[sb..])
        .chain(&av[sa + ra..])
        .copied()
        .chain(0..params.len())
        .collect();
    let post_params: Vec<_> =
        at[..sa].iter().chain(&bt[..sb]).chain(&at[sa + ra..]).chain(&params).copied().collect();
    let mut post = Wiring::new(post_params);
    let produced = post.call(
        a.post.clone(),
        (0..sa).chain(sa + sb..sa + sb + at.len() - sa - ra).collect(),
    );
    let mut outputs = vec![usize::MAX; ra];
    outputs.extend(&produced);
    let base = sa + sb + at.len() - sa - ra;
    let args = trees_b
        .iter()
        .map(|t| wire_input_at(data, parent, &mut post, t, &arrays, &outputs, base))
        .collect::<Option<Vec<_>>>()?;
    if args.contains(&usize::MAX) {
        return None;
    }
    let middle = post.call(b.pre.clone(), args);
    let results = post.call(
        b.post.clone(),
        (sa..sa + sb).chain(middle[sb + rb..].iter().copied()).collect(),
    );
    let mut origins: Vec<_> = (0..ra)
        .map(|i| (producer, i))
        .chain((0..rb + signature(&b.post).1.len()).map(|i| (consumer, i)))
        .collect();
    if retain {
        origins.extend((0..produced.len()).map(|i| (producer, ra + i)));
    }
    let form = ScremaForm {
        pre: pre.finish(pre_outputs),
        scans: a.scans.into_iter().chain(b.scans).collect(),
        reductions: a.reductions.into_iter().chain(b.reductions).collect(),
        post: post.finish(results.into_iter().chain(if retain { produced } else { vec![] }).collect()),
    };
    install(data, producer, consumer, form, arrays, origins)
}
