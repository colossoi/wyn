//! Early SOAC normalization pass for TLC.
//!
//! When a Map has multiple inputs (from an absorbed zip) but the lambda takes
//! a single tuple parameter, rewrites the lambda to take N separate parameters.
//!
//! The rewrite is purely semantic: every use of the old tuple param `p` in the
//! body is replaced with a tuple construction `(p0, p1, ..., pN)`. Downstream
//! passes (partial eval, project folding) simplify projections on that
//! reconstructed tuple into direct references to the individual params.
//!
//! Runs before fusion and defunctionalization.

use crate::SymbolTable;
use crate::ast::TypeName;
use polytype::Type;

use super::{ArrayExpr, Def, Lambda, Program, SoacOp, Term, TermIdSource, TermKind};

/// Normalize SOACs in a TLC program.
pub fn normalize_soacs(program: Program) -> Program {
    let mut symbols = program.symbols;
    let mut term_ids = TermIdSource::new();

    let defs = program
        .defs
        .into_iter()
        .map(|def| {
            let body = normalize_term(def.body, &mut symbols, &mut term_ids);
            Def { body, ..def }
        })
        .collect();

    Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols,
    }
}

fn normalize_term(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    let term = term.map_children(&mut |child| normalize_term(child, symbols, term_ids));

    if let TermKind::Soac(SoacOp::Map { .. }) = &term.kind {
        let (id, ty, span) = (term.id, term.ty, term.span);
        match term.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                let normalized = normalize_map(lam, inputs, symbols, term_ids);
                Term {
                    id,
                    ty,
                    span,
                    kind: TermKind::Soac(normalized),
                }
            }
            _ => unreachable!(),
        }
    } else {
        term
    }
}

/// Find or create the `_w_tuple` symbol for tuple construction.
fn get_tuple_sym(symbols: &mut SymbolTable) -> crate::SymbolId {
    // Reuse existing if available.
    for (id, name) in symbols.iter() {
        if name == "_w_tuple" {
            return *id;
        }
    }
    symbols.alloc("_w_tuple".to_string())
}

/// If the Map has multiple inputs but a single tuple-typed lambda param,
/// split the param into N separate params and substitute.
fn normalize_map(
    lam: Lambda,
    inputs: Vec<ArrayExpr>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> SoacOp {
    if inputs.len() <= 1 || lam.params.len() != 1 {
        return SoacOp::Map { lam, inputs };
    }

    let (old_param, ref param_ty) = lam.params[0];

    // Must be a concrete tuple type matching the input count.
    let flat_types = match param_ty {
        Type::Constructed(TypeName::Tuple(_), types) if !types.is_empty() => flatten_tuple_types(types),
        _ => return SoacOp::Map { lam, inputs },
    };

    if flat_types.len() != inputs.len() || has_type_variables(param_ty) {
        return SoacOp::Map { lam, inputs };
    }

    // Create N fresh params.
    let new_params: Vec<(crate::SymbolId, Type<TypeName>)> = flat_types
        .iter()
        .enumerate()
        .map(|(i, ty)| (symbols.alloc(format!("_sn_{}", i)), ty.clone()))
        .collect();

    // Substitute: every `Var(old_param)` → `_w_tuple(Var(p0), ..., Var(pN))`
    // reconstructed with the original tuple type. Downstream simplification
    // (partial eval / project folding) will reduce proj(tuple(...), i) → pi.
    let tuple_sym = get_tuple_sym(symbols);
    let span = lam.body.span;
    let rewritten_body = substitute_param(
        *lam.body,
        old_param,
        &new_params,
        param_ty,
        tuple_sym,
        term_ids,
        span,
    );

    SoacOp::Map {
        lam: Lambda {
            params: new_params,
            body: Box::new(rewritten_body),
            ret_ty: lam.ret_ty,
            captures: lam.captures,
        },
        inputs,
    }
}

/// Replace every occurrence of `Var(old_sym)` with a tuple reconstruction
/// from the new params. Respects shadowing.
fn substitute_param(
    term: Term,
    old_sym: crate::SymbolId,
    new_params: &[(crate::SymbolId, Type<TypeName>)],
    tuple_ty: &Type<TypeName>,
    tuple_sym: crate::SymbolId,
    term_ids: &mut TermIdSource,
    span: crate::ast::Span,
) -> Term {
    if let TermKind::Var(sym) = &term.kind {
        if *sym == old_sym {
            return build_tuple_reconstruction(new_params, tuple_ty, tuple_sym, term_ids, span);
        }
    }

    // Stop at shadowing.
    match &term.kind {
        TermKind::Let { name, .. } if *name == old_sym => return term,
        TermKind::Lambda(lam) if lam.params.iter().any(|(s, _)| *s == old_sym) => return term,
        _ => {}
    }

    term.map_children(&mut |child| {
        substitute_param(child, old_sym, new_params, tuple_ty, tuple_sym, term_ids, span)
    })
}

/// Build `Tuple(Var(p0), Var(p1), ..., Var(pN))` matching the original tuple type.
///
/// For nested tuple types like `((A, B), C)` with flat params `[p0, p1, p2]`,
/// builds `Tuple(Tuple(Var(p0), Var(p1)), Var(p2))` to match the nesting.
fn build_tuple_reconstruction(
    new_params: &[(crate::SymbolId, Type<TypeName>)],
    tuple_ty: &Type<TypeName>,
    tuple_sym: crate::SymbolId,
    term_ids: &mut TermIdSource,
    span: crate::ast::Span,
) -> Term {
    match tuple_ty {
        Type::Constructed(TypeName::Tuple(n), component_types) if !component_types.is_empty() => {
            let mut offset = 0;
            let mut elements = Vec::with_capacity(*n);
            for comp_ty in component_types {
                let count = flat_type_count(comp_ty);
                let sub_params = &new_params[offset..offset + count];
                let elem = build_tuple_reconstruction(sub_params, comp_ty, tuple_sym, term_ids, span);
                elements.push(elem);
                offset += count;
            }
            let func = Term {
                id: term_ids.next_id(),
                ty: tuple_ty.clone(),
                span,
                kind: TermKind::Var(tuple_sym),
            };
            Term {
                id: term_ids.next_id(),
                ty: tuple_ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(func),
                    args: elements,
                },
            }
        }
        _ => {
            // Leaf — single param.
            assert_eq!(new_params.len(), 1);
            let (sym, ty) = &new_params[0];
            Term {
                id: term_ids.next_id(),
                ty: ty.clone(),
                span,
                kind: TermKind::Var(*sym),
            }
        }
    }
}

/// Count how many flat (non-tuple) types a type expands to.
fn flat_type_count(ty: &Type<TypeName>) -> usize {
    match ty {
        Type::Constructed(TypeName::Tuple(_), children) if !children.is_empty() => {
            children.iter().map(flat_type_count).sum()
        }
        _ => 1,
    }
}

/// Recursively flatten nested tuple types: ((A, B), C) → [A, B, C]
fn flatten_tuple_types(types: &[Type<TypeName>]) -> Vec<Type<TypeName>> {
    let mut flat = Vec::new();
    for ty in types {
        match ty {
            Type::Constructed(TypeName::Tuple(_), children) if !children.is_empty() => {
                flat.extend(flatten_tuple_types(children));
            }
            _ => flat.push(ty.clone()),
        }
    }
    flat
}

fn has_type_variables(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(_, args) => args.iter().any(has_type_variables),
    }
}
