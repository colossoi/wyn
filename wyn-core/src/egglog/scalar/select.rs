//! Form eager value choices before equality saturation. Every newly evaluated
//! arm dependency must be cheap and total, with parameters/literals as leaves.
//! Execution results remain opaque: their availability requires placement proofs.

use super::{fold, hoist};
use crate::builtins::{catalog, lowering::BuiltinLowering, select};
use crate::egglog::data::{intern_expr, intern_type, BuiltinData, ExprId, ExprKind, Ir, TypeId};
use crate::egglog::rewrite::{self, Rewriter};
use crate::types::{self, TypeExt};
use egglog_engine::{Error, FullState, Read, Write};
use std::collections::{BTreeMap, BTreeSet};
use wyn_graph::dag_postorder;

#[cfg(test)]
#[path = "select_tests.rs"]
mod tests;

pub(in crate::egglog) fn form(data: &mut Ir) {
    let order = order(data);
    let mut rewrite = Rewriter::new(data);
    let mut replacements = BTreeMap::new();
    let mut functions = BTreeMap::new();
    for id in order {
        let original = data.expressions[id].clone();
        let mut kind = original.kind.clone();
        kind.for_each_child_mut(&mut |child| {
            if let Some(&replacement) = replacements.get(child) {
                *child = replacement;
            }
        });
        if let ExprKind::If {
            condition,
            then_value,
            else_value,
        } = kind
        {
            let ty = &data.types[original.ty].ty;
            if select::supported_type(ty)
                && data.types[data.expressions[condition].ty].ty == types::bool_type()
                && data.expressions[then_value].ty == original.ty
                && data.expressions[else_value].ty == original.ty
                && cheap_arms(data, [then_value, else_value])
            {
                let function = *functions.entry(original.ty).or_insert_with(|| function(data, original.ty));
                kind = ExprKind::PureApp {
                    function,
                    args: vec![else_value, then_value, condition],
                };
            }
        }
        if kind != original.kind {
            let value = rewrite.intern(data, original.ty, kind);
            replacements.insert(id, value);
        }
    }
    rewrite::all(data, &replacements);
}

fn order(data: &Ir) -> Vec<ExprId> {
    dag_postorder(
        data.expressions.iter().map(|(&id, _)| id),
        |_| false,
        |id, out| out.extend(data.expressions[id].kind.children()),
    )
}

fn operands(kind: &ExprKind) -> Vec<ExprId> {
    match kind {
        ExprKind::PureApp { args, .. } => args.clone(),
        _ => kind.children(),
    }
}

fn cheap_arms(data: &Ir, roots: [ExprId; 2]) -> bool {
    let mut pending = Vec::from(roots);
    let mut seen = BTreeSet::new();
    let mut cost = 0;
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(added) = node_cost(data, id) else {
            return false;
        };
        cost += added;
        if cost > select::ADDED_WORK_LIMIT {
            return false;
        }
        pending.extend(operands(&data.expressions[id].kind));
    }
    true
}

fn node_cost(data: &Ir, id: ExprId) -> Option<usize> {
    let expression = &data.expressions[id];
    let ty = &data.types[expression.ty].ty;
    if !select::supported_type(ty) {
        return None;
    }
    match &expression.kind {
        ExprKind::Parameter(_)
        | ExprKind::Int(_)
        | ExprKind::FloatBits(_)
        | ExprKind::Bool(_)
        | ExprKind::Vector(_) => return Some(0),
        ExprKind::Project { tuple, index } => {
            let ty = &data.types[data.expressions[*tuple].ty].ty;
            if !ty.vec_size().is_some_and(|size| *index < size) {
                return None;
            }
        }
        ExprKind::PureApp { function, .. } if hoist::total_node(data, id) => {
            match &data.expressions[*function].kind {
                ExprKind::BinOp(_) | ExprKind::UnOp(_) => {}
                _ => {
                    let BuiltinLowering::PrimOp(prim) = fold::lowering(data, *function)? else {
                        return None;
                    };
                    if !select::cheap_primop(prim) {
                        return None;
                    }
                }
            }
        }
        _ => return None,
    }
    Some(
        operands(&expression.kind)
            .iter()
            .map(|&arg| data.types[data.expressions[arg].ty].ty.vec_size().unwrap_or(1))
            .fold(ty.vec_size().unwrap_or(1), usize::max),
    )
}

fn function(data: &mut Ir, ty: TypeId) -> ExprId {
    let known = catalog().known().select;
    let builtin =
        data.builtins.iter().find_map(|(&id, b)| (b.builtin == known && b.overload_idx == 0).then_some(id));
    let builtin = builtin.unwrap_or_else(|| {
        data.builtins.alloc(BuiltinData {
            builtin: known,
            overload_idx: 0,
        })
    });
    let value = data.types[ty].ty.clone();
    let signature = intern_type(
        data,
        types::function(
            value.clone(),
            types::function(value.clone(), types::function(types::bool_type(), value)),
        ),
    );
    intern_expr(data, signature, ExprKind::Builtin(builtin))
}

pub(super) fn facts(data: &mut Ir, mut sink: FullState<'_, '_>) -> Result<(), Error> {
    let bool_ty = intern_type(data, types::bool_type());
    let not_ty = intern_type(data, types::function(types::bool_type(), types::bool_type()));
    let not_ft = sink.add("TypeId", (i64::from(not_ty.as_u32()),))?;
    let not_node = sink.add("UnOp", ("!",))?;
    let not = sink.add("Typed", (not_ft, not_node))?;
    sink.add("BooleanNot", (not,))?;
    // Boolean select identities also apply in programs with no authored bool result.
    let bool_t = sink.add("TypeId", (i64::from(bool_ty.as_u32()),))?;
    sink.add("ScalarType", (bool_t, "bool"))?;
    // These are unconditional, whole-DAG proofs, never facts inherited from a
    // particular use's branch. Required evaluations cannot disappear in EqSat.
    let mut total = BTreeSet::new();
    for id in order(data) {
        if node_cost(data, id).is_some()
            && operands(&data.expressions[id].kind).iter().all(|child| total.contains(child))
        {
            total.insert(id);
            if let Some(value) = sink.lookup("SourceExpression", (i64::from(id.as_u32()),))? {
                sink.add("TotalSelectValue", (value,))?;
            }
        }
    }
    let types = data
        .types
        .iter()
        .filter_map(|(&id, t)| select::supported_type(&t.ty).then_some(id))
        .collect::<Vec<_>>();
    for ty in types {
        let f = function(data, ty);
        let ExprKind::Builtin(builtin) = data.expressions[f].kind else {
            continue;
        };
        let b = sink.add("BuiltinId", (i64::from(builtin.as_u32()),))?;
        let node = sink.add("Builtin", (b,))?;
        let ft = sink.add("TypeId", (i64::from(data.expressions[f].ty.as_u32()),))?;
        let f = sink.add("Typed", (ft, node))?;
        let t = sink.add("TypeId", (i64::from(ty.as_u32()),))?;
        sink.add("SelectFunction", (t, f))?;
    }
    Ok(())
}
