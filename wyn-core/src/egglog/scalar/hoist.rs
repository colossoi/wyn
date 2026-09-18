//! Binding availability is relative to an invocation, never to an ExprId alone.
use super::fold::lowering;
use super::OptimizeError;
use crate::egglog::data::{ExprId, ExprKind, Ir, PlacementData, PlacementId, PlacementSite};
use crate::egglog::timing::span;
use crate::types::{Type, TypeName};
use placements::Placements;
use std::collections::{BTreeMap, BTreeSet};
use wyn_base::IdArena;

mod analysis;
mod bounds;
mod captures;
mod placements;
mod specialize;
mod uses;

pub(super) fn run(data: &mut Ir) -> Result<IdArena<PlacementId, PlacementData>, OptimizeError> {
    let _timing = span("egglog hoisting");
    let mut placements = Placements::default();
    captures::run(data, &mut placements)?;
    let mut analysis = analysis::Analysis::new(data)?;
    analysis.place(data, &mut placements);
    Ok(placements.finish(data, &analysis.uses, &analysis.control))
}

pub(super) fn ordered(data: &Ir, selected: &BTreeSet<ExprId>) -> Vec<ExprId> {
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
fn total_node(data: &Ir, id: ExprId) -> bool {
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
            Type::Constructed(TypeName::Tuple(_), _)
        ),
        ExprKind::Coerce(x) => data.expressions[*x].ty == data.expressions[id].ty,
        ExprKind::PureApp { function, .. } => match &data.expressions[*function].kind {
            ExprKind::BinOp(op) => matches!(
                op.as_str(),
                "+" | "-" | "*" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&" | "|" | "^" | "&&" | "||"
            ),
            ExprKind::UnOp(_) => true,
            _ => lowering(data, *function).is_some_and(|lowering| lowering.is_speculatable()),
        },
        _ => false,
    }
}

pub(in crate::egglog) fn index(
    data: &Ir,
    placements: &IdArena<PlacementId, PlacementData>,
) -> BTreeMap<PlacementSite, Vec<ExprId>> {
    let mut sites: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
    let mut values = BTreeSet::new();
    for p in placements.values() {
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

#[cfg(test)]
#[path = "hoist_tests.rs"]
mod tests;
