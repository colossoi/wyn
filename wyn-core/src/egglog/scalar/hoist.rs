//! Binding availability is relative to an invocation, never to an ExprId alone.
use super::*;
use crate::builtins::lowering::{BuiltinLowering, PrimOp};

mod analysis;
mod captures;
mod placements;
use placements::Placements;
mod bounds;
mod specialize;
mod uses;

pub(super) fn run(data: &mut AssociatedData) -> Result<(), OptimizeError> {
    let _timing = timing::span("hoisting");
    data.placements = Default::default();
    let mut placements = Placements::default();
    timing::time("specialize SOAC captures", || {
        captures::run(data, &mut placements)
    })?;
    let mut analysis = analysis::Analysis::new(data)?;
    timing::time("select loop and branch placements", || {
        analysis.place(data, &mut placements)
    });
    timing::time("prune placements", || {
        placements.finish(data, &analysis.uses, &analysis.control)
    });
    Ok(())
}

pub(super) fn ordered(data: &AssociatedData, selected: &BTreeSet<ExprId>) -> Vec<ExprId> {
    wyn_graph::dag_postorder(
        selected.iter().copied(),
        |_| false,
        |e, out| out.extend(data.expressions[e].kind.children()),
    )
    .into_iter()
    .filter(|e| selected.contains(e))
    .collect()
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

pub(in crate::egglog) fn index(data: &AssociatedData) -> BTreeMap<PlacementSite, Vec<ExprId>> {
    let mut sites: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
    let mut values = BTreeSet::new();
    for p in data.placements.values() {
        sites.entry(p.before).or_default().insert(p.expression);
        values.insert(p.expression);
    }
    // Shared expressions may be placed at many invocation sites. Order their
    // combined DAG once, then sort each site's selected values by that rank.
    let ranks: BTreeMap<_, _> =
        ordered(data, &values).into_iter().enumerate().map(|(i, e)| (e, i)).collect();
    sites
        .into_iter()
        .map(|(site, values)| {
            let mut values: Vec<_> = values.into_iter().collect();
            values.sort_unstable_by_key(|e| ranks[e]);
            (site, values)
        })
        .collect()
}
