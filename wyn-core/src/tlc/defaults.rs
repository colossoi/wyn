//! Default-valued TLC term construction for `--fill-holes`.
//!
//! Given a Wyn type, produce a canonical zero/empty TLC `Term` of
//! that type — numeric `0`, `false`, `()`, tuple/vec/array filled
//! componentwise. Types that have no sensible zero-value (function
//! arrows, unresolved type variables, user-defined records / sums,
//! storage-backed arrays) push an error into the `Transformer`'s
//! `fill_hole_errors` accumulator and return a throwaway i32
//! placeholder. The caller (`to_tlc`) checks the accumulator at the
//! end and turns any accumulated errors into a
//! `CompilerError::TypeHole`.
//!
//! Keeping this off of `Transformer`'s main impl block keeps
//! `tlc/mod.rs` focused on AST → TLC rewriting.
use super::{Term, TermKind, Transformer};
use crate::ast::{Span, TypeName};
use crate::err_type_hole;
use crate::types::TypeScheme;
use polytype::Type;

/// Replace every free type variable in `ty` with a position-appropriate
/// default: `i32` in ordinary type positions, `SizePlaceholder` in array
/// size positions, and `ArrayVariantComposite` in array variant positions.
/// Using a plain `i32` for every free var corrupts Array types — the size
/// / variant axes carry kinds that `i32` cannot satisfy, and downstream
/// passes (storage-binding detection, monomorphization, SoA) misbehave
/// silently. `bound` is the stack of ∀-quantified variable IDs in scope;
/// they remain as variables.
fn default_free_vars_in_type(ty: &mut Type<TypeName>, bound: &[usize]) {
    match ty {
        Type::Variable(v) if !bound.contains(v) => {
            *ty = Type::Constructed(TypeName::Int(32), vec![]);
        }
        Type::Variable(_) => {}
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            default_free_vars_in_type(&mut args[0], bound);
            default_free_vars_in_array_size(&mut args[1], bound);
            default_free_vars_in_array_variant(&mut args[2], bound);
        }
        Type::Constructed(_, args) => {
            for a in args {
                default_free_vars_in_type(a, bound);
            }
        }
    }
}

/// Free variable sitting in an Array size position → `SizePlaceholder`,
/// matching the convention used by parsing + entry-point storage-binding
/// detection. Bound vars pass through unchanged.
fn default_free_vars_in_array_size(ty: &mut Type<TypeName>, bound: &[usize]) {
    if let Type::Variable(v) = ty {
        if !bound.contains(v) {
            *ty = Type::Constructed(TypeName::SizePlaceholder, vec![]);
        }
    }
}

/// Free variable in an Array variant position → `ArrayVariantComposite`,
/// matching SoA's convention for unresolved variants.
fn default_free_vars_in_array_variant(ty: &mut Type<TypeName>, bound: &[usize]) {
    if let Type::Variable(v) = ty {
        if !bound.contains(v) {
            *ty = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
        }
    }
}

fn default_free_vars_in_scheme_inner(scheme: &mut TypeScheme, bound: &mut Vec<usize>) {
    match scheme {
        TypeScheme::Monotype(ty) => default_free_vars_in_type(ty, bound),
        TypeScheme::Polytype { variable, body } => {
            bound.push(*variable);
            default_free_vars_in_scheme_inner(body, bound);
            bound.pop();
        }
    }
}

/// Rewrite each `TypeScheme` in place so that any free type variable
/// (one not bound by an enclosing ∀) becomes `i32`. Called from
/// `to_tlc` under `--fill-holes` to give the transformer a fully
/// ground `TypeTable` — without this pass, type-inference variables
/// that survived unsolved (e.g. `let x = ??? in ...` where no use
/// pins x's type) surface as `Type::Variable` during
/// `default_term_for_type` and would otherwise push a
/// "unresolved type variable" fill-hole error.
pub fn default_free_vars_in_table<'a>(schemes: impl IntoIterator<Item = &'a mut TypeScheme>) {
    let mut bound = Vec::new();
    for scheme in schemes {
        default_free_vars_in_scheme_inner(scheme, &mut bound);
        debug_assert!(bound.is_empty());
    }
}

/// Extract a `usize` size from a type position that should be a
/// size literal (e.g. `args[1]` of `Vec` or `args[1]` of `Array`).
/// Returns `None` for size variables or skolems, which aren't
/// resolved to literals at type-check time.
pub(super) fn type_size_literal(ty: &Type<TypeName>) -> Option<usize> {
    match ty {
        Type::Constructed(TypeName::Size(n), _) => Some(*n),
        _ => None,
    }
}

/// Build a default-valued TLC term of the given type.
pub(super) fn default_term_for_type(tr: &mut Transformer<'_>, ty: &Type<TypeName>, span: Span) -> Term {
    match ty {
        Type::Constructed(TypeName::Int(_), args) if args.is_empty() => {
            tr.mk_term(ty.clone(), span, TermKind::IntLit("0".into()))
        }
        Type::Constructed(TypeName::UInt(_), args) if args.is_empty() => {
            tr.mk_term(ty.clone(), span, TermKind::IntLit("0".into()))
        }
        Type::Constructed(TypeName::Float(_), args) if args.is_empty() => {
            tr.mk_term(ty.clone(), span, TermKind::FloatLit(0.0))
        }
        Type::Constructed(TypeName::Bool, args) if args.is_empty() => {
            tr.mk_term(ty.clone(), span, TermKind::BoolLit(false))
        }
        Type::Constructed(TypeName::Unit, _) => tr.build_call("_w_unit", &[], ty.clone(), span),
        Type::Constructed(TypeName::Tuple(_), args) => {
            let elems: Vec<Term> = args.iter().map(|a| default_term_for_type(tr, a, span)).collect();
            tr.mk_tuple(elems, ty.clone(), span)
        }
        Type::Constructed(TypeName::Vec, args) if args.len() == 2 => {
            // Vec[elem, size] — args[0] is elem, args[1] is size.
            let elem_ty = &args[0];
            match type_size_literal(&args[1]) {
                Some(n) => {
                    let elem = default_term_for_type(tr, elem_ty, span);
                    let elems: Vec<Term> = (0..n).map(|_| elem.clone()).collect();
                    tr.mk_vec_lit(elems, ty.clone(), span)
                }
                None => hole_fill_error(tr, span, ty, "vector size is not a literal"),
            }
        }
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            // Array[elem, size, variant]. Only Composite arrays can be
            // default-filled — View/Virtual need a buffer binding or
            // a range, which `--fill-holes` can't synthesize.
            let elem_ty = &args[0];
            let size = type_size_literal(&args[1]);
            let is_composite = matches!(&args[2], Type::Constructed(TypeName::ArrayVariantComposite, _));
            match (is_composite, size) {
                (true, Some(n)) => {
                    let elem = default_term_for_type(tr, elem_ty, span);
                    let elems: Vec<Term> = (0..n).map(|_| elem.clone()).collect();
                    tr.mk_array_lit(elems, ty.clone(), span)
                }
                (true, None) => hole_fill_error(tr, span, ty, "array size is not a literal"),
                (false, _) => hole_fill_error(
                    tr,
                    span,
                    ty,
                    "only Composite array variants can be default-filled \
                     (View/Virtual need a buffer binding or range)",
                ),
            }
        }
        Type::Variable(_) => hole_fill_error(
            tr,
            span,
            ty,
            "hole has an unresolved type variable — add a type annotation \
             or fill the hole manually",
        ),
        Type::Constructed(TypeName::Arrow, _) => {
            hole_fill_error(tr, span, ty, "cannot synthesize a default function value")
        }
        _ => hole_fill_error(tr, span, ty, "no default value available for this type"),
    }
}

/// Record a fill-holes failure and return a throwaway placeholder
/// term of type i32. The placeholder is never observed — `to_tlc`
/// surfaces the accumulated errors before any consumer reads the
/// transformed program.
fn hole_fill_error(tr: &mut Transformer<'_>, span: Span, ty: &Type<TypeName>, reason: &str) -> Term {
    tr.fill_hole_errors.push(err_type_hole!(
        "--fill-holes: at {}:{}: {} (type: {:?})",
        span.start_line,
        span.start_col,
        reason,
        ty
    ));
    tr.mk_term(
        Type::Constructed(TypeName::Int(32), vec![]),
        span,
        TermKind::IntLit("0".into()),
    )
}
