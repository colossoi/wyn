//! Project pure scalar results in the sidecar. Missing scan values are represented
//! by distinct parameters and must disappear from the selected pre-barrier slice.
use super::*;

pub(super) fn tuple(data: &mut AssociatedData, parent: RegionId, parameters: Vec<TypeId>) -> SoacBody {
    let (region, args) = region(data, parent, &parameters);
    let t = ty(
        data,
        types::tuple(parameters.iter().map(|t| data.types[*t].ty.clone()).collect()),
    );
    let value = expr(data, t, ExprKind::Tuple(args));
    finish(data, region, parameters, vec![value])
}
pub(super) fn region(
    data: &mut AssociatedData,
    parent: RegionId,
    types: &[TypeId],
) -> (RegionId, Vec<ExprId>) {
    let id = data.regions.alloc(RegionData {
        definition: data.regions[parent].definition,
        parent: Some(parent),
        parameters: vec![],
        members: Default::default(),
        results: vec![],
    });
    // Diagnostic name only; identity is the newly allocated ParameterId.
    let symbol = data.definitions[data.regions[parent].definition].symbol;
    let mut args = vec![];
    for &ty in types {
        let p = data.parameters.alloc(ParameterData {
            region: id,
            symbol,
            ty,
        });
        data.regions[id].parameters.push(p);
        args.push(expr(data, ty, ExprKind::Parameter(p)));
    }
    (id, args)
}
pub(super) fn finish(
    data: &mut AssociatedData,
    region: RegionId,
    parameters: Vec<TypeId>,
    values: Vec<ExprId>,
) -> SoacBody {
    let results = values.iter().map(|v| data.expressions[*v].ty).collect();
    let mut leaves = BTreeSet::new();
    for &v in &values {
        references(data, v, &mut leaves);
    }
    for &op in &data.regions[region].members {
        match &data.operations[op].kind {
            OperationKind::Call { args, .. } => {
                for &v in args {
                    references(data, v, &mut leaves);
                }
            }
            OperationKind::Index { array, index } => {
                references(data, *array, &mut leaves);
                references(data, *index, &mut leaves);
            }
            _ => {}
        }
    }
    let captures: Vec<_> = leaves
        .into_iter()
        .filter(|v| match data.expressions[*v].kind {
            ExprKind::Parameter(p) => data.parameters[p].region != region,
            ExprKind::OperationResult(op) => data.operations[op].region != region,
            _ => false,
        })
        .collect();
    let mut replacements = BTreeMap::new();
    let symbol = data.definitions[data.regions[region].definition].symbol;
    for &v in &captures {
        let ty = data.expressions[v].ty;
        let p = data.parameters.alloc(ParameterData { region, symbol, ty });
        data.regions[region].parameters.push(p);
        let e = expr(data, ty, ExprKind::Parameter(p));
        replacements.insert(v, e);
    }
    for op in data.regions[region].members.clone() {
        let mut kind = data.operations[op].kind.clone();
        rewrite::operation(data, &mut kind, &mut replacements);
        data.operations[op].kind = kind;
    }
    data.regions[region].results =
        values.into_iter().map(|v| rewrite::value(data, v, &mut replacements)).collect();
    SoacBody::Apply {
        region,
        parameters,
        results,
        captures,
    }
}

pub(super) fn references(data: &AssociatedData, v: ExprId, out: &mut BTreeSet<ExprId>) {
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
pub(super) fn array_references(data: &AssociatedData, a: &Array, out: &mut BTreeSet<ExprId>) {
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

/// Materialize opaque body calls into a new region without inspecting their
/// scalar implementations. Used when projection is unnecessary.
pub(super) fn call(
    data: &mut AssociatedData,
    region: RegionId,
    body: &SoacBody,
    args: Vec<ExprId>,
) -> Option<Vec<ExprId>> {
    match body {
        SoacBody::Identity(_) => Some(args),
        SoacBody::Route { indices, .. } => indices.iter().map(|&i| args.get(i).copied()).collect(),
        SoacBody::Compose { first, then } => {
            let vs = call(data, region, first, args)?;
            call(data, region, then, vs)
        }
        SoacBody::Parallel { left, right } => {
            let mut vs = call(data, region, left, args.clone())?;
            vs.extend(call(data, region, right, args)?);
            Some(vs)
        }
        SoacBody::Apply {
            region: target,
            captures,
            results,
            ..
        } => {
            let args: Vec<_> = args.into_iter().chain(captures.iter().copied()).collect();
            let t = match results.as_slice() {
                [t] => *t,
                _ => ty(
                    data,
                    types::tuple(results.iter().map(|t| data.types[*t].ty.clone()).collect()),
                ),
            };
            let fty = args.iter().rev().fold(data.types[t].ty.clone(), |r, a| {
                types::function(data.types[data.expressions[*a].ty].ty.clone(), r)
            });
            let fty = ty(data, fty);
            let f = expr(data, fty, ExprKind::Lambda(*target));
            let v = operation(data, region, t, OperationKind::Call { function: f, args });
            Some(if results.len() == 1 {
                vec![v]
            } else {
                results
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| expr(data, t, ExprKind::Project { tuple: v, index: i }))
                    .collect()
            })
        }
    }
}
pub(super) fn operation(
    data: &mut AssociatedData,
    region: RegionId,
    ty: TypeId,
    kind: OperationKind,
) -> ExprId {
    let id = data.operations.alloc(OperationData {
        region,
        ty,
        kind,
        source_position: data.regions[region].members.len(),
        span: crate::ast::Span::generated(),
    });
    data.regions[region].members.insert(id);
    expr(data, ty, ExprKind::OperationResult(id))
}
fn field(data: &mut AssociatedData, v: ExprId, i: usize, t: TypeId) -> ExprId {
    match data.expressions[v].kind.clone() {
        ExprKind::Tuple(vs) => vs[i],
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => {
            let a = field(data, then_value, i, t);
            let b = field(data, else_value, i, t);
            expr(
                data,
                t,
                ExprKind::If {
                    condition,
                    then_value: a,
                    else_value: b,
                },
            )
        }
        _ => expr(data, t, ExprKind::Project { tuple: v, index: i }),
    }
}
fn scalar(data: &mut AssociatedData, id: ExprId, map: &mut BTreeMap<ExprId, ExprId>) -> Option<ExprId> {
    if let Some(&v) = map.get(&id) {
        return Some(v);
    }
    let ExprData { ty: t, kind } = data.expressions[id].clone();
    let result = match kind {
        ExprKind::Project { tuple, index } => {
            let v = scalar(data, tuple, map)?;
            field(data, v, index, t)
        }
        ExprKind::Tuple(vs) | ExprKind::Vector(vs) => {
            let vector = matches!(data.expressions[id].kind, ExprKind::Vector(_));
            let vs = vs.into_iter().map(|v| scalar(data, v, map)).collect::<Option<Vec<_>>>()?;
            expr(
                data,
                t,
                if vector { ExprKind::Vector(vs) } else { ExprKind::Tuple(vs) },
            )
        }
        ExprKind::PureApp { function, args } => {
            let args = args.into_iter().map(|v| scalar(data, v, map)).collect::<Option<Vec<_>>>()?;
            expr(data, t, ExprKind::PureApp { function, args })
        }
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => {
            let c = scalar(data, condition, map)?;
            let a = scalar(data, then_value, map)?;
            let b = scalar(data, else_value, map)?;
            expr(
                data,
                t,
                ExprKind::If {
                    condition: c,
                    then_value: a,
                    else_value: b,
                },
            )
        }
        ExprKind::Coerce(v) => {
            let v = scalar(data, v, map)?;
            expr(data, t, ExprKind::Coerce(v))
        }
        ExprKind::OperationResult(_)
        | ExprKind::Lambda(_)
        | ExprKind::Closure { .. }
        | ExprKind::Array(_) => return None,
        _ => id,
    };
    map.insert(id, result);
    Some(result)
}
pub(super) fn invoke(data: &mut AssociatedData, body: &SoacBody, args: Vec<ExprId>) -> Option<Vec<ExprId>> {
    match body {
        SoacBody::Identity(_) => Some(args),
        SoacBody::Route { indices, .. } => indices.iter().map(|&i| args.get(i).copied()).collect(),
        SoacBody::Compose { first, then } => {
            let vs = invoke(data, first, args)?;
            invoke(data, then, vs)
        }
        SoacBody::Parallel { left, right } => {
            let mut vs = invoke(data, left, args.clone())?;
            vs.extend(invoke(data, right, args)?);
            Some(vs)
        }
        SoacBody::Apply { region, captures, .. } => {
            let record = data.regions[*region].clone();
            if !record.members.is_empty() {
                return None;
            }
            let args: Vec<_> = args.into_iter().chain(captures.iter().copied()).collect();
            if args.len() != record.parameters.len() {
                return None;
            }
            let substitutions: BTreeMap<_, _> = record.parameters.into_iter().zip(args).collect();
            let mut memo = data
                .expressions
                .iter()
                .filter_map(|(&id, e)| {
                    if let ExprKind::Parameter(p) = e.kind {
                        substitutions.get(&p).map(|&v| (id, v))
                    } else {
                        None
                    }
                })
                .collect();
            record.results.into_iter().map(|v| scalar(data, v, &mut memo)).collect()
        }
    }
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
pub(super) fn across_barrier(
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
