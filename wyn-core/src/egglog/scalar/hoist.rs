//! Binding availability is relative to an invocation, never to an ExprId alone.
use super::*;
use crate::builtins::lowering::{BuiltinLowering, PrimOp};

mod specialize;

pub(super) fn run(data: &mut AssociatedData) -> Result<(), OptimizeError> {
    let _timing = timing::span("hoisting");
    data.placements = Default::default();
    // Move inner callback computations first. A new outer capture expression
    // can then itself be invariant in an enclosing SOAC or explicit loop.
    loop {
        let graph = analyze(data)?;
        let (rows, _, dag) = graph.function_to_dag("HoistCapture", usize::MAX, false)?;
        let mut choices: BTreeMap<(OperationId, RegionId), BTreeSet<ExprId>> = BTreeMap::new();
        for row in rows {
            let a = extract::app(&dag, row, "HoistCapture", 3)?;
            choices
                .entry((
                    extract::key(&dag, a[0], "OperationId")?,
                    extract::key(&dag, a[1], "RegionId")?,
                ))
                .or_default()
                .insert(extract::key(&dag, a[2], "ExprId")?);
        }
        let Some(((op, region), values)) = choices.into_iter().next() else {
            break;
        };
        timing::time("specialize SOAC captures", || {
            specialize::apply(data, op, region, &ordered(data, &values))
        })?;
    }
    let graph = analyze(data)?;
    for (relation, operation) in [("HoistBefore", true), ("HoistSelect", false)] {
        let (rows, _, dag) = graph.function_to_dag(relation, usize::MAX, false)?;
        for row in rows {
            let a = extract::app(&dag, row, relation, 2)?;
            let before = if operation {
                PlacementSite::Operation(extract::key(&dag, a[0], "OperationId")?)
            } else {
                PlacementSite::Expression(extract::key(&dag, a[0], "ExprId")?)
            };
            place(data, before, extract::key(&dag, a[1], "ExprId")?);
        }
    }
    let (_, live) = expressions::program(data, &[])?;
    let (rows, _, dag) = graph.function_to_dag("Execution", usize::MAX, false)?;
    let operations = rows
        .into_iter()
        .map(|row| {
            let a = extract::app(&dag, row, "Execution", 3)?;
            extract::key(&dag, a[1], "OperationId")
        })
        .collect::<Result<BTreeSet<OperationId>, _>>()?;
    timing::time("prune placements", || prune(data, &live, &operations));
    Ok(())
}

pub(super) fn place(data: &mut AssociatedData, before: PlacementSite, expression: ExprId) {
    let value = PlacementData { before, expression };
    if !data.placements.values().any(|p| *p == value) {
        data.placements.alloc(value);
    }
}

pub(super) fn ordered(data: &AssociatedData, selected: &BTreeSet<ExprId>) -> Vec<ExprId> {
    fn visit(
        data: &AssociatedData,
        id: ExprId,
        selected: &BTreeSet<ExprId>,
        seen: &mut BTreeSet<ExprId>,
        out: &mut Vec<ExprId>,
    ) {
        if !seen.insert(id) {
            return;
        }
        for x in children(data, id) {
            visit(data, x, selected, seen, out);
        }
        if selected.contains(&id) {
            out.push(id);
        }
    }
    let mut out = vec![];
    let mut seen = BTreeSet::new();
    for &id in selected {
        visit(data, id, selected, &mut seen, &mut out);
    }
    out
}

fn analyze(data: &AssociatedData) -> Result<EGraph, OptimizeError> {
    let _timing = timing::span("analyze placements");
    let (program, live) = expressions::program(data, &[])?;
    graph(program, include_str!("../hoist.egg"), &facts(data, &live)?)
}

pub(super) fn facts(data: &AssociatedData, live: &BTreeSet<ExprId>) -> Result<String, OptimizeError> {
    let _timing = timing::span("derive availability facts");
    let mut facts = String::new();
    for &id in live {
        let e = name(id);
        if matches!(
            data.expressions[id].kind,
            ExprKind::Parameter(_) | ExprKind::OperationResult(_)
        ) {
            facts.push_str(&format!("(Leaf {e})\n"));
        } else {
            let args = children(data, id).into_iter().map(name).collect::<Vec<_>>().join(" ");
            facts.push_str(&format!("(LocalInputs {e} (vec-of {args}))\n"));
            if total_node(data, id) {
                facts.push_str(&format!("(TotalNode {e})\n"));
            }
            if matches!(
                data.expressions[id].kind,
                ExprKind::PureApp { .. }
                    | ExprKind::Coerce(_)
                    | ExprKind::Project { .. }
                    | ExprKind::Vector(_)
            ) {
                facts.push_str(&format!("(Computation {e})\n"));
            }
        }
    }
    for (op, env) in availability(data)? {
        let available = live
            .iter()
            .filter(|&&id| match data.expressions[id].kind {
                ExprKind::Parameter(p) => env.parameters.contains(&p),
                ExprKind::OperationResult(op) => env.operations.contains(&op),
                _ => false,
            })
            .map(|&id| name(id))
            .collect::<Vec<_>>()
            .join(" ");
        facts.push_str(&format!(
            "(AvailableBefore {} (set-of {available}))\n",
            op.egglog()
        ));
    }
    let mut total = BTreeMap::new();
    let mut captures_by_site: BTreeMap<(OperationId, RegionId), BTreeSet<ParameterId>> = BTreeMap::new();
    for (&op, record) in &data.operations {
        let mut bodies = vec![];
        callbacks(&record.kind, &mut bodies);
        for body in bodies {
            let SoacBody::Apply {
                region,
                parameters,
                captures,
                ..
            } = body
            else {
                continue;
            };
            let formals = &data.regions[*region].parameters;
            if formals.len() != parameters.len() + captures.len() {
                return Err(error("SOAC capture arity"));
            }
            let safe: BTreeSet<_> = formals[parameters.len()..]
                .iter()
                .zip(captures)
                .filter_map(|(&p, &v)| total_value(data, v, &mut total).then_some(p))
                .collect();
            // A fused operation can invoke the same region with different
            // captures/roles. Its specialization must be safe at every use.
            captures_by_site
                .entry((op, *region))
                .and_modify(|known| known.retain(|p| safe.contains(p)))
                .or_insert(safe);
        }
    }
    for ((op, region), safe) in captures_by_site {
        let available = live
            .iter()
            .filter(|&&id| matches!(data.expressions[id].kind, ExprKind::Parameter(p) if safe.contains(&p)))
            .map(|&id| name(id))
            .collect::<Vec<_>>()
            .join(" ");
        facts.push_str(&format!(
            "(CaptureParameters {} {} (set-of {available}))\n",
            op.egglog(),
            region.egglog()
        ));
    }
    facts.push_str("(run-schedule (saturate (run hoist)))\n");
    Ok(facts)
}

// A whitelist is intentional: catalog purity alone does not prove that an
// expression is safe on paths that previously did not evaluate it.
fn total_node(data: &AssociatedData, id: ExprId) -> bool {
    match &data.expressions[id].kind {
        ExprKind::Int(_)
        | ExprKind::FloatBits(_)
        | ExprKind::Bool(_)
        | ExprKind::Unit
        | ExprKind::Parameter(_)
        | ExprKind::OperationResult(_)
        | ExprKind::BinOp(_)
        | ExprKind::UnOp(_)
        | ExprKind::Builtin(_)
        | ExprKind::Tuple(_)
        | ExprKind::Vector(_) => true,
        ExprKind::Project { tuple, .. } => matches!(
            data.types[data.expressions[*tuple].ty].ty,
            crate::types::Type::Constructed(crate::types::TypeName::Tuple(_), _)
        ),
        ExprKind::Coerce(x) => data.expressions[*x].ty == data.expressions[id].ty,
        ExprKind::PureApp { function, .. } => match &data.expressions[*function].kind {
            ExprKind::BinOp(op) => matches!(
                op.as_str(),
                "+" | "-" | "*" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&" | "|" | "^" | "&&" | "||"
            ),
            ExprKind::UnOp(_) => true,
            _ => matches!(
                fold::lowering(data, *function),
                Some(BuiltinLowering::PrimOp(
                    PrimOp::Bitcast
                        | PrimOp::SIToFP
                        | PrimOp::UIToFP
                        | PrimOp::SConvert
                        | PrimOp::UConvert
                        | PrimOp::FPConvert
                        | PrimOp::GlslExt(4 | 5 | 8 | 9)
                ))
            ),
        },
        _ => false,
    }
}

fn total_value(data: &AssociatedData, id: ExprId, memo: &mut BTreeMap<ExprId, bool>) -> bool {
    if let Some(&v) = memo.get(&id) {
        return v;
    }
    let total = total_node(data, id) && children(data, id).into_iter().all(|v| total_value(data, v, memo));
    memo.insert(id, total);
    total
}

#[derive(Clone, Default)]
struct Environment {
    parameters: BTreeSet<ParameterId>,
    operations: BTreeSet<OperationId>,
}
fn availability(data: &AssociatedData) -> Result<BTreeMap<OperationId, Environment>, OptimizeError> {
    fn region(
        data: &AssociatedData,
        id: RegionId,
        mut env: Environment,
        schedules: &BTreeMap<RegionId, Vec<OperationId>>,
        seen: &mut BTreeSet<RegionId>,
        out: &mut BTreeMap<OperationId, Environment>,
    ) -> Environment {
        if !seen.insert(id) {
            return env;
        }
        env.parameters.extend(&data.regions[id].parameters);
        for &op in schedules.get(&id).into_iter().flatten() {
            out.insert(op, env.clone());
            match data.operations[op].kind {
                OperationKind::If {
                    then_region,
                    else_region,
                    ..
                } => {
                    region(data, then_region, env.clone(), schedules, seen, out);
                    region(data, else_region, env.clone(), schedules, seen, out);
                }
                OperationKind::Loop { header, body, .. } => {
                    let inner = region(data, header, env.clone(), schedules, seen, out);
                    region(data, body, inner, schedules, seen, out);
                }
                _ => {}
            }
            env.operations.insert(op);
        }
        env
    }
    let schedules = super::super::snapshot::analyze(data).schedules(data)?;
    let mut out = BTreeMap::new();
    let mut seen = BTreeSet::new();
    // Parent IDs precede children in imported and specialized regions.
    for id in data.regions.ids() {
        region(data, id, Environment::default(), &schedules, &mut seen, &mut out);
    }
    Ok(out)
}

pub(super) fn callbacks<'a>(kind: &'a OperationKind, out: &mut Vec<&'a SoacBody>) {
    fn body<'a>(b: &'a SoacBody, out: &mut Vec<&'a SoacBody>) {
        match b {
            SoacBody::Apply { .. } => out.push(b),
            SoacBody::Compose { first, then } => {
                body(first, out);
                body(then, out);
            }
            SoacBody::Parallel { left, right } => {
                body(left, out);
                body(right, out);
            }
            _ => {}
        }
    }
    match kind {
        OperationKind::Screma { form, .. } => {
            body(&form.pre, out);
            body(&form.post, out);
            for scan in &form.scans {
                body(&scan.operator, out);
            }
            for reduce in &form.reductions {
                body(&reduce.operator, out);
            }
        }
        OperationKind::Filter { map, body: b, .. } | OperationKind::ReduceByIndex { map, body: b, .. } => {
            body(map, out);
            body(b, out);
        }
        OperationKind::Scatter { body: b, .. } | OperationKind::BucketScatter { body: b, .. } => {
            body(b, out)
        }
        _ => {}
    }
}

fn prune(data: &mut AssociatedData, live: &BTreeSet<ExprId>, operations: &BTreeSet<OperationId>) {
    // A dominating outer placement covers nested loop/branch sites. Keep sibling
    // sites separate: their runtime evaluations can see different outer iterations.
    let old: Vec<_> = data.placements.values().cloned().collect();
    data.placements = Default::default();
    for p in &old {
        if !live.contains(&p.expression)
            || !match p.before {
                PlacementSite::Operation(op) => operations.contains(&op),
                PlacementSite::Expression(e) => live.contains(&e),
            }
        {
            continue;
        }
        let redundant = old.iter().any(|outer| {
            if outer == p || outer.expression != p.expression {
                return false;
            }
            let (PlacementSite::Operation(a), PlacementSite::Operation(b)) = (outer.before, p.before)
            else {
                return false;
            };
            let roots: Vec<_> = match data.operations[a].kind {
                OperationKind::Loop { header, body, .. } => vec![header, body],
                OperationKind::If {
                    then_region,
                    else_region,
                    ..
                } => vec![then_region, else_region],
                _ => vec![],
            };
            let mut region = Some(data.operations[b].region);
            while let Some(r) = region {
                if roots.contains(&r) {
                    return true;
                }
                region = data.regions[r].parent;
            }
            false
        });
        if !redundant {
            data.placements.alloc(p.clone());
        }
    }
}

pub(super) fn output(data: &AssociatedData) -> String {
    data.placements
        .values()
        .map(|p| match p.before {
            PlacementSite::Operation(op) => format!(
                "(EvaluateBefore {} (ExprId {}))\n",
                op.egglog(),
                p.expression.as_u32()
            ),
            PlacementSite::Expression(e) => format!(
                "(EvaluateBeforeSelect (ExprId {}) (ExprId {}))\n",
                e.as_u32(),
                p.expression.as_u32()
            ),
        })
        .collect()
}

pub(in crate::egglog) fn at(data: &AssociatedData, site: PlacementSite) -> Vec<ExprId> {
    ordered(
        data,
        &data.placements.values().filter(|p| p.before == site).map(|p| p.expression).collect(),
    )
}
