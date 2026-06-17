//! Push array-valued conditionals into pointwise producers.
//!
//! Compute entries need a top-level producer shape for EGIR-side
//! pointwise parallelization. An expression like
//! `if c then map(f, xs) else map(g, xs)` is still pointwise, but if it
//! reaches output normalization in that shape the parallelizer sees a
//! branch that produces an array instead of a producer. This pass rewrites
//! the conservative cases to one `Map` whose lambda contains the branch.

use super::VarRef;
use super::{ArrayExpr, Def, Lambda, Program, SoacBody, SoacDestination, SoacOp};
use super::{Term, TermIdSource, TermKind};
use crate::SymbolId;
use crate::ast::{Span, TypeName};
use crate::builtins::{BuiltinId, catalog};
use crate::types::TypeExt;
use polytype::Type;
use std::collections::HashSet;

pub fn run(program: Program) -> Program {
    let mut term_ids = TermIdSource::new();
    let defs = program
        .defs
        .into_iter()
        .map(|def| Def {
            body: rewrite_term(def.body, &mut term_ids),
            ..def
        })
        .collect();

    Program { defs, ..program }
}

fn rewrite_term(term: Term, term_ids: &mut TermIdSource) -> Term {
    let term = term.map_children(&mut |child| rewrite_term(child, term_ids));
    rewrite_if_over_map(term, term_ids)
}

#[derive(Clone)]
struct PrefixLet {
    name: SymbolId,
    name_ty: Type<TypeName>,
    rhs: Term,
    span: Span,
}

struct MapBranch {
    prefix: Vec<PrefixLet>,
    lam: SoacBody,
    inputs: Vec<ArrayExpr>,
}

fn rewrite_if_over_map(term: Term, term_ids: &mut TermIdSource) -> Term {
    let TermKind::If {
        cond,
        then_branch,
        else_branch,
    } = term.kind
    else {
        return term;
    };

    let then_term = *then_branch;
    let else_term = *else_branch;

    let Some(then_map) = extract_map_branch(then_term.clone()) else {
        return Term {
            kind: TermKind::If {
                cond,
                then_branch: Box::new(then_term),
                else_branch: Box::new(else_term),
            },
            ..term
        };
    };
    let Some(else_map) = extract_map_branch(else_term.clone()) else {
        return Term {
            kind: TermKind::If {
                cond,
                then_branch: Box::new(then_term),
                else_branch: Box::new(else_term),
            },
            ..term
        };
    };

    let original_ty = term.ty.clone();
    let original_span = term.span;
    if !can_fuse_if_maps(&cond, &then_map, &else_map, &original_ty) {
        return Term {
            id: term.id,
            ty: original_ty.clone(),
            span: original_span,
            kind: TermKind::If {
                cond,
                then_branch: Box::new(then_term),
                else_branch: Box::new(else_term),
            },
        };
    }

    build_fused_map_if(*cond, then_map, else_map, original_ty, original_span, term_ids)
}

fn extract_map_branch(term: Term) -> Option<MapBranch> {
    match term.kind {
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let span = term.span;
            let mut branch = extract_map_branch(*body)?;
            branch.prefix.insert(
                0,
                PrefixLet {
                    name,
                    name_ty,
                    rhs: *rhs,
                    span,
                },
            );
            Some(branch)
        }
        TermKind::Soac(SoacOp::Map {
            lam,
            inputs,
            destination: _,
        }) => Some(MapBranch {
            prefix: Vec::new(),
            lam,
            inputs,
        }),
        _ => None,
    }
}

fn can_fuse_if_maps(
    cond: &Term,
    then_map: &MapBranch,
    else_map: &MapBranch,
    result_ty: &Type<TypeName>,
) -> bool {
    if then_map.inputs.is_empty() || else_map.inputs.is_empty() {
        return false;
    }
    if !then_map.lam.captures.is_empty() || !else_map.lam.captures.is_empty() {
        return false;
    }
    if then_map.lam.lam.ret_ty != else_map.lam.lam.ret_ty {
        return false;
    }
    if then_map.lam.lam.params.len() != then_map.inputs.len()
        || else_map.lam.lam.params.len() != else_map.inputs.len()
    {
        return false;
    }
    branches_have_same_domain(then_map, else_map, result_ty)
        && prefixes_can_be_hoisted(cond, then_map, else_map)
}

fn build_fused_map_if(
    cond: Term,
    then_map: MapBranch,
    else_map: MapBranch,
    result_ty: Type<TypeName>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let mut inputs = Vec::with_capacity(then_map.inputs.len() + else_map.inputs.len());
    inputs.extend(then_map.inputs);
    inputs.extend(else_map.inputs);

    let then_lambda = then_map.lam.lam;
    let else_lambda = else_map.lam.lam;
    let mut params = Vec::with_capacity(then_lambda.params.len() + else_lambda.params.len());
    params.extend(then_lambda.params);
    params.extend(else_lambda.params);

    let elem_ty = then_lambda.ret_ty.clone();
    let body = Term {
        id: term_ids.next_id(),
        ty: elem_ty.clone(),
        span,
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: then_lambda.body,
            else_branch: else_lambda.body,
        },
    };
    let mut result = Term {
        id: term_ids.next_id(),
        ty: result_ty,
        span,
        kind: TermKind::Soac(SoacOp::Map {
            lam: SoacBody {
                lam: Lambda {
                    params,
                    body: Box::new(body),
                    ret_ty: elem_ty,
                },
                captures: Vec::new(),
            },
            inputs,
            destination: SoacDestination::Fresh,
        }),
    };

    for prefix in then_map.prefix.into_iter().chain(else_map.prefix).rev() {
        result = Term {
            id: term_ids.next_id(),
            ty: result.ty.clone(),
            span: prefix.span,
            kind: TermKind::Let {
                name: prefix.name,
                name_ty: prefix.name_ty,
                rhs: Box::new(prefix.rhs),
                body: Box::new(result),
            },
        };
    }
    result
}

fn branches_have_same_domain(
    then_map: &MapBranch,
    else_map: &MapBranch,
    result_ty: &Type<TypeName>,
) -> bool {
    if static_array_dims_match(result_ty) {
        return true;
    }

    let Some(then_len) = branch_len_key(then_map) else {
        return false;
    };
    let Some(else_len) = branch_len_key(else_map) else {
        return false;
    };
    then_len == else_len
}

fn static_array_dims_match(result_ty: &Type<TypeName>) -> bool {
    let Some(dim) = result_ty.array_dim(0) else {
        return false;
    };
    matches!(
        dim,
        Type::Constructed(TypeName::Size(_), _) | Type::Constructed(TypeName::SizeVar(_), _)
    )
}

fn branch_len_key(branch: &MapBranch) -> Option<LenKey> {
    let env: Vec<(SymbolId, &Term)> =
        branch.prefix.iter().map(|prefix| (prefix.name, &prefix.rhs)).collect();
    array_expr_len_key(branch.inputs.first()?, &env, &mut HashSet::new())
}

fn prefixes_can_be_hoisted(cond: &Term, then_map: &MapBranch, else_map: &MapBranch) -> bool {
    let mut names = HashSet::new();
    for prefix in then_map.prefix.iter().chain(&else_map.prefix) {
        if !names.insert(prefix.name) {
            return false;
        }
    }
    if names.is_empty() {
        return true;
    }

    let mut cond_refs = HashSet::new();
    collect_symbol_refs(cond, &mut cond_refs);
    if names.iter().any(|name| cond_refs.contains(name)) {
        return false;
    }

    let mut then_refs = HashSet::new();
    for input in &then_map.inputs {
        collect_symbol_refs_array(input, &mut then_refs);
    }
    collect_symbol_refs_soac_body(&then_map.lam, &mut then_refs);

    let mut else_refs = HashSet::new();
    for input in &else_map.inputs {
        collect_symbol_refs_array(input, &mut else_refs);
    }
    collect_symbol_refs_soac_body(&else_map.lam, &mut else_refs);

    let then_names: HashSet<_> = then_map.prefix.iter().map(|prefix| prefix.name).collect();
    let else_names: HashSet<_> = else_map.prefix.iter().map(|prefix| prefix.name).collect();
    if then_names.iter().any(|name| else_refs.contains(name)) {
        return false;
    }
    if else_names.iter().any(|name| then_refs.contains(name)) {
        return false;
    }
    true
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LenKey {
    Zero,
    Int(String),
    Symbol(SymbolId),
    Builtin(BuiltinId),
    BinOp(String),
    App(Box<LenKey>, Vec<LenKey>),
    ArrayLen(Box<LenKey>),
    Coerce(Box<LenKey>),
    Tuple(Vec<LenKey>),
    Proj(Box<LenKey>, usize),
}

fn array_expr_len_key(
    ae: &ArrayExpr,
    env: &[(SymbolId, &Term)],
    resolving: &mut HashSet<SymbolId>,
) -> Option<LenKey> {
    match ae {
        ArrayExpr::Ref(term) => term_array_len_key(term, env, resolving),
        ArrayExpr::Range { len, .. } => len_key(len, env, resolving),
        ArrayExpr::StorageView(crate::tlc::StorageView { len, .. }) => len_key(len, env, resolving),
        ArrayExpr::Literal(items) => Some(LenKey::Int(items.len().to_string())),
        ArrayExpr::Zip(items) => items.first().and_then(|item| array_expr_len_key(item, env, resolving)),
        ArrayExpr::Soac(_) => None,
    }
}

fn term_array_len_key(
    term: &Term,
    env: &[(SymbolId, &Term)],
    resolving: &mut HashSet<SymbolId>,
) -> Option<LenKey> {
    if let Some(bound) = extract_iota_bound(term) {
        return len_key(&bound, env, resolving);
    }

    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => {
            if !resolving.insert(*sym) {
                return None;
            }
            let out = env
                .iter()
                .rev()
                .find(|(name, _)| name == sym)
                .and_then(|(_, rhs)| term_array_len_key(rhs, env, resolving))
                .or_else(|| Some(LenKey::ArrayLen(Box::new(LenKey::Symbol(*sym)))));
            resolving.remove(sym);
            out
        }
        TermKind::ArrayExpr(ae) => array_expr_len_key(ae, env, resolving),
        TermKind::App { func, args } if is_builtin(func, catalog().known().slice) && args.len() == 3 => {
            len_key_sub(&args[2], &args[1], env, resolving)
        }
        TermKind::Let { name, rhs, body, .. } => {
            let mut extended = env.to_vec();
            extended.push((*name, rhs));
            term_array_len_key(body, &extended, resolving)
        }
        TermKind::Coerce { inner, .. } => term_array_len_key(inner, env, resolving),
        _ => None,
    }
}

fn len_key(term: &Term, env: &[(SymbolId, &Term)], resolving: &mut HashSet<SymbolId>) -> Option<LenKey> {
    match &term.kind {
        TermKind::IntLit(s) if s == "0" => Some(LenKey::Zero),
        TermKind::IntLit(s) => Some(LenKey::Int(s.clone())),
        TermKind::BoolLit(v) => Some(LenKey::Int((*v as i32).to_string())),
        TermKind::Var(VarRef::Symbol(sym)) => {
            if resolving.contains(sym) {
                return None;
            }
            if let Some((_, rhs)) = env.iter().rev().find(|(name, _)| name == sym) {
                resolving.insert(*sym);
                let out = len_key(rhs, env, resolving);
                resolving.remove(sym);
                out
            } else {
                Some(LenKey::Symbol(*sym))
            }
        }
        TermKind::Var(VarRef::Builtin { id, .. }) => Some(LenKey::Builtin(*id)),
        TermKind::BinOp(op) => Some(LenKey::BinOp(op.op.clone())),
        TermKind::App { func, args } => {
            let f = len_key(func, env, resolving)?;
            let args: Option<Vec<_>> = args.iter().map(|arg| len_key(arg, env, resolving)).collect();
            let args = args?;
            if let LenKey::BinOp(op) = &f {
                if args.len() == 2 {
                    if op == "-" && is_zero_len_key(&args[1]) {
                        return Some(args[0].clone());
                    }
                    if op == "+" && is_zero_len_key(&args[1]) {
                        return Some(args[0].clone());
                    }
                    if op == "+" && is_zero_len_key(&args[0]) {
                        return Some(args[1].clone());
                    }
                }
            }
            Some(LenKey::App(Box::new(f), args))
        }
        TermKind::Let { name, rhs, body, .. } => {
            let mut extended = env.to_vec();
            extended.push((*name, rhs));
            len_key(body, &extended, resolving)
        }
        TermKind::Coerce { inner, .. } => Some(LenKey::Coerce(Box::new(len_key(inner, env, resolving)?))),
        TermKind::Tuple(items) => {
            let items: Option<Vec<_>> = items.iter().map(|item| len_key(item, env, resolving)).collect();
            Some(LenKey::Tuple(items?))
        }
        TermKind::TupleProj { tuple, idx } => {
            Some(LenKey::Proj(Box::new(len_key(tuple, env, resolving)?), *idx))
        }
        _ => None,
    }
}

fn len_key_sub(
    lhs: &Term,
    rhs: &Term,
    env: &[(SymbolId, &Term)],
    resolving: &mut HashSet<SymbolId>,
) -> Option<LenKey> {
    let lhs = len_key(lhs, env, resolving)?;
    let rhs = len_key(rhs, env, resolving)?;
    if is_zero_len_key(&rhs) {
        return Some(lhs);
    }
    Some(LenKey::App(
        Box::new(LenKey::BinOp("-".to_string())),
        vec![lhs, rhs],
    ))
}

fn is_zero_len_key(key: &LenKey) -> bool {
    match key {
        LenKey::Zero => true,
        LenKey::Int(s) => s == "0",
        _ => false,
    }
}

fn extract_iota_bound(term: &Term) -> Option<Term> {
    let TermKind::Let { name, rhs, body, .. } = &term.kind else {
        return None;
    };
    let TermKind::ArrayExpr(ArrayExpr::Range { start, len, .. }) = &body.kind else {
        return None;
    };
    let TermKind::App { func, args } = &len.kind else {
        return None;
    };
    let [arg, zero] = args.as_slice() else {
        return None;
    };
    let TermKind::BinOp(op) = &func.kind else {
        return None;
    };
    let arg_is_name = matches!(&arg.kind, TermKind::Var(VarRef::Symbol(sym)) if sym == name);
    (op.op == "-" && is_zero_term(start) && is_zero_term(zero) && arg_is_name).then(|| (**rhs).clone())
}

fn is_builtin(term: &Term, id: BuiltinId) -> bool {
    matches!(&term.kind, TermKind::Var(VarRef::Builtin { id: builtin, .. }) if *builtin == id)
}

fn is_zero_term(term: &Term) -> bool {
    matches!(&term.kind, TermKind::IntLit(s) if s == "0")
}

fn collect_symbol_refs(term: &Term, out: &mut HashSet<SymbolId>) {
    if let TermKind::Var(VarRef::Symbol(sym)) = &term.kind {
        out.insert(*sym);
    }
    collect_symbol_refs_in_soac_places(term, out);
    term.for_each_child(&mut |child| collect_symbol_refs(child, out));
}

fn collect_symbol_refs_soac_body(body: &SoacBody, out: &mut HashSet<SymbolId>) {
    collect_symbol_refs(&body.lam.body, out);
    for (_, _, expr) in &body.captures {
        collect_symbol_refs(expr, out);
    }
}

fn collect_symbol_refs_array(ae: &ArrayExpr, out: &mut HashSet<SymbolId>) {
    match ae {
        ArrayExpr::Ref(term) => collect_symbol_refs(term, out),
        ArrayExpr::Zip(items) => {
            for item in items {
                collect_symbol_refs_array(item, out);
            }
        }
        ArrayExpr::Soac(soac) => collect_symbol_refs_soac(soac, out),
        ArrayExpr::Literal(items) => {
            for item in items {
                collect_symbol_refs(item, out);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            collect_symbol_refs(start, out);
            collect_symbol_refs(len, out);
            if let Some(step) = step {
                collect_symbol_refs(step, out);
            }
        }
        ArrayExpr::StorageView(crate::tlc::StorageView { offset, len, .. }) => {
            collect_symbol_refs(offset, out);
            collect_symbol_refs(len, out);
        }
    }
}

fn collect_symbol_refs_soac(soac: &SoacOp, out: &mut HashSet<SymbolId>) {
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            collect_symbol_refs_soac_body(lam, out);
            for input in inputs {
                collect_symbol_refs_array(input, out);
            }
        }
        SoacOp::Reduce { op, ne, input } => {
            collect_symbol_refs_soac_body(op, out);
            collect_symbol_refs(ne, out);
            collect_symbol_refs_array(input, out);
        }
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
        } => {
            collect_symbol_refs_soac_body(op, out);
            collect_symbol_refs_soac_body(reduce_op, out);
            collect_symbol_refs(ne, out);
            for input in inputs {
                collect_symbol_refs_array(input, out);
            }
        }
        SoacOp::Screma {
            map_lams,
            accumulators,
            inputs,
        } => {
            for lam in map_lams {
                collect_symbol_refs_soac_body(lam, out);
            }
            for acc in accumulators {
                collect_symbol_refs_soac_body(&acc.step_lam, out);
                collect_symbol_refs_soac_body(&acc.reduce_op, out);
                collect_symbol_refs(&acc.ne, out);
            }
            for input in inputs {
                collect_symbol_refs_array(input, out);
            }
        }
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            ..
        } => {
            collect_symbol_refs_soac_body(op, out);
            collect_symbol_refs_soac_body(reduce_op, out);
            collect_symbol_refs(ne, out);
            collect_symbol_refs_array(input, out);
        }
        SoacOp::Filter { pred, input, .. } => {
            collect_symbol_refs_soac_body(pred, out);
            collect_symbol_refs_array(input, out);
        }
        SoacOp::Scatter { dest, lam, inputs } => {
            collect_symbol_refs_place(dest, out);
            collect_symbol_refs_soac_body(lam, out);
            for input in inputs {
                collect_symbol_refs_array(input, out);
            }
        }
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
        } => {
            collect_symbol_refs_place(dest, out);
            collect_symbol_refs_soac_body(op, out);
            collect_symbol_refs(ne, out);
            collect_symbol_refs_array(indices, out);
            collect_symbol_refs_array(values, out);
        }
    }
}

fn collect_symbol_refs_in_soac_places(term: &Term, out: &mut HashSet<SymbolId>) {
    if let TermKind::Soac(soac) = &term.kind {
        match soac {
            SoacOp::Scatter { dest, .. } | SoacOp::ReduceByIndex { dest, .. } => {
                collect_symbol_refs_place(dest, out);
            }
            _ => {}
        }
    }
}

fn collect_symbol_refs_place(place: &super::Place, out: &mut HashSet<SymbolId>) {
    match place {
        super::Place::BufferSlice { base, offset, .. } => {
            collect_symbol_refs(base, out);
            collect_symbol_refs(offset, out);
        }
        super::Place::LocalArray { id, .. } => {
            out.insert(*id);
        }
    }
}
