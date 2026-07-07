//! Normalise compute entries into explicit per-slot stores.
//!
//! Before this pass: a compute entry's body is a Term whose tail
//! expression "produces" the entry's return value. Single-output entries
//! return a single Term; tuple-output entries return a `Tuple(N)`. The
//! downstream stack then decomposes this — `analyze_entry` walks the
//! tail looking for a parallelisable SOAC, `egir::realize_outputs`
//! flattens the tuple and per-slot retargets producer SOACs.
//!
//! After this pass: a compute entry's body produces no value
//! (`SideEffect`-typed). Its tail is a chain of
//! `TermKind::OutputSlotStore { slot_index, value, .. }` terms, one
//! per declared output, wired together via `Let { name: _ }`
//! sequencing and capped with a `SideEffect`-typed terminator.
//! Single-output and multi-output cases share one structural shape.
//! After running, `def.ty == def.body.ty` for every normalised entry.
//!
//! The pass runs after `apply_ownership` (so `SoacDestination` is
//! already accurate) and before `lift_gathers` / `defunctionalize` /
//! `parallelize_soacs` (so the simpler IR shape feeds those passes).
//!
//! Tail shapes handled: a `TermKind::Tuple(operands)` matching
//! `outputs.len()` → N slots; any other tail with a single-output
//! entry → 1 slot. Multi-output entries whose tail is a single value
//! of tuple type (e.g. a `reduce` returning `(u32, [4]u32)`) flow
//! through unchanged — `egir::realize_outputs` decomposes them via
//! `Project` over the `Return(Some(result))` terminator.
//!
//! `SoacDestination::OutputView` association with bindings stays at
//! EGIR time (`from_tlc::build_entry_outputs` allocates `BindingRef`s
//! per output index). `OutputSlotStore` carries `slot_index: usize`,
//! and EGIR maps it to the slot's binding.

use super::{Def, DefMeta, Lambda, Program, Term, TermIdSource, TermKind};
use crate::ast::TypeName;
use crate::SymbolTable;
use polytype::Type;

#[cfg(test)]
#[path = "normalize_outputs_tests.rs"]
mod normalize_outputs_tests;

#[derive(Debug)]
pub enum NormalizeError {
    /// The tuple-tail operand count doesn't match the declared output
    /// count.
    SlotCountMismatch {
        entry: String,
        outputs: usize,
        operands: usize,
    },
}

impl std::fmt::Display for NormalizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NormalizeError::SlotCountMismatch {
                entry,
                outputs,
                operands,
            } => write!(
                f,
                "normalize_outputs: entry `{entry}` declares {outputs} outputs but its tail \
                 Tuple has {operands} operands"
            ),
        }
    }
}

impl std::error::Error for NormalizeError {}

pub fn run(mut program: Program) -> Result<Program, NormalizeError> {
    let mut term_ids = TermIdSource::new();
    let entry_indices: Vec<usize> = program
        .defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match &d.meta {
            DefMeta::EntryPoint(decl) if decl.entry_type.is_compute() => Some(i),
            _ => None,
        })
        .collect();

    let Program { defs, symbols, .. } = &mut program;
    for idx in entry_indices {
        normalize_entry(&mut defs[idx], &mut term_ids, symbols)?;
    }
    Ok(program)
}

/// The tuple arity reachable by peeling existential / uniqueness markers off a
/// type, or `None` if it isn't a (wrapped) tuple. Mirrors how those markers are
/// seen through elsewhere — a `?k. (A, B)` or `*(A, B)` entry output is N
/// separate bindings.
fn tuple_arity_through_markers(ty: &Type<TypeName>) -> Option<usize> {
    match ty {
        Type::Constructed(TypeName::Existential(_), args) if args.len() == 1 => {
            tuple_arity_through_markers(&args[0])
        }
        Type::Constructed(TypeName::Tuple(n), _) => Some(*n),
        _ => None,
    }
}

fn normalize_entry(
    def: &mut Def,
    term_ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> Result<(), NormalizeError> {
    let DefMeta::EntryPoint(decl) = &def.meta else {
        unreachable!("filtered to EntryPoint")
    };
    let entry_name = decl.name.clone();
    // A single declared output that is an existential (and/or unique) wrapper
    // over a tuple denotes one binding *per tuple component* — the quantifier /
    // uniqueness marker is seen through, the same way uniqueness markers are
    // elsewhere, with the single `?k.` binder shared across the components.
    // `?k. (A, B)` is two output slots, not one.
    let n_outputs = if decl.outputs.len() == 1 {
        tuple_arity_through_markers(&decl.outputs[0].ty).unwrap_or(1)
    } else {
        decl.outputs.len()
    };

    // Peel the outer Lambda (entry params) and any tail Let-chain; the
    // tail expression is what we decompose. We rebuild the same
    // Lambda/Let frame around the new `SideEffect`-typed body so all
    // intermediate bindings stay in scope for the slot writes.
    let body_span = def.body.span;
    let body = std::mem::replace(
        &mut def.body,
        // placeholder; overwritten below
        Term {
            id: term_ids.next_id(),
            ty: Type::Constructed(TypeName::Unit, vec![]),
            span: body_span,
            kind: TermKind::UnitLit,
        },
    );

    let new_body = rewrite_body(body, &entry_name, n_outputs, term_ids, symbols)?;
    // Sync `def.ty` with the rewritten body so `def.ty == def.body.ty`.
    // The body's outer Lambda now ends in `SideEffect`; the def's
    // signature follows. The original declared per-slot output shape
    // still lives on `decl.outputs[i].ty + attribute` — EGIR
    // conversion reads it from there.
    def.ty = new_body.ty.clone();
    def.body = new_body;

    Ok(())
}

/// Walk through outer `Lambda` / `Let` layers, recurse on the body, and
/// rebuild the same frame around the rewritten tail.
fn rewrite_body(
    term: Term,
    entry_name: &str,
    n_outputs: usize,
    term_ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> Result<Term, NormalizeError> {
    match term.kind {
        TermKind::Lambda(Lambda {
            params,
            body,
            ret_ty: _,
        }) => {
            let new_body = rewrite_body(*body, entry_name, n_outputs, term_ids, symbols)?;
            // Derive the rebuilt Lambda's arrow ty + ret_ty from
            // `new_body.ty`. For fully-normalised bodies that's
            // `SideEffect`; for the multi-output non-Tuple fallthrough
            // in `emit_slot_writes` (which preserves the tail's tuple
            // type) it's the original tuple. Forcing one or the other
            // would claim a return type the body doesn't produce.
            let body_ty = new_body.ty.clone();
            let mut arrow_ty = body_ty.clone();
            for (_, pt) in params.iter().rev() {
                arrow_ty = Type::Constructed(TypeName::Arrow, vec![pt.clone(), arrow_ty]);
            }
            Ok(Term {
                id: term_ids.next_id(),
                ty: arrow_ty,
                span: term.span,
                kind: TermKind::Lambda(Lambda {
                    params,
                    body: Box::new(new_body),
                    ret_ty: body_ty,
                }),
            })
        }
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_body = rewrite_body(*body, entry_name, n_outputs, term_ids, symbols)?;
            let body_ty = new_body.ty.clone();
            Ok(Term {
                id: term_ids.next_id(),
                ty: body_ty,
                span: term.span,
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs,
                    body: Box::new(new_body),
                },
            })
        }
        // Tail position — decompose.
        _ => emit_slot_writes(term, entry_name, n_outputs, term_ids, symbols),
    }
}

/// `tail` is the entry's tail expression. Decompose into a chain of
/// `OutputSlotStore` terms wired by `Let { name: _ }` sequencing.
fn emit_slot_writes(
    tail: Term,
    entry_name: &str,
    n_outputs: usize,
    term_ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> Result<Term, NormalizeError> {
    // If the entry has zero outputs (unit return), there are no slot
    // writes to emit — but the tail may be a side-effectful expression
    // (e.g. a storage-image update tail) whose value sinks to unit
    // checking. Return it unchanged so the side effect survives. The
    // body's tail still has unit type, so the enclosing Lambda's
    // `ret_ty: Unit` rewrite stays consistent.
    if n_outputs == 0 {
        return Ok(tail);
    }

    // Decompose `tail` into per-slot value terms.
    let slot_values: Vec<Term> = match tail.kind {
        TermKind::Tuple(operands) if operands.len() == n_outputs => operands,
        TermKind::Tuple(operands) => {
            return Err(NormalizeError::SlotCountMismatch {
                entry: entry_name.to_string(),
                outputs: n_outputs,
                operands: operands.len(),
            });
        }
        _ if n_outputs == 1 => vec![tail.clone()],
        // Multi-output entries whose tail is a *single* value of tuple
        // type (e.g. a reduce returning `(u32, [4]u32)` or a function
        // call returning a tuple). Phase 1A leaves these unnormalised:
        // `egir::realize_outputs` still handles them via the existing
        // `Project` decomposition over a `Return(Some(result))`
        // terminator. The non-normalised tail flows through unchanged.
        _ => {
            return Ok(Term {
                id: term_ids.next_id(),
                ty: tail.ty.clone(),
                span: tail.span,
                kind: tail.kind,
            });
        }
    };

    // Build the chain: store_N → store_{N-1} → … → store_0 → <terminator>.
    // Innermost is a `SideEffect`-typed terminator; each outer wraps
    // the previous in `Let { name: _, rhs: store, body: inner }`.
    let mut chain = side_effect_terminator(term_ids, tail.span);
    for (i, value) in slot_values.into_iter().enumerate().rev() {
        let store = make_store(i, value, term_ids);
        chain = sequence(store, chain, term_ids, symbols);
    }
    Ok(chain)
}

fn make_store(slot_index: usize, value: Term, term_ids: &mut TermIdSource) -> Term {
    let span = value.span;
    Term {
        id: term_ids.next_id(),
        ty: side_effect_ty(),
        span,
        kind: TermKind::OutputSlotStore {
            slot_index,
            value: Box::new(value),
        },
    }
}

/// `let _seq: SideEffect = first in rest` — a sequencing operator. The
/// sequencing-let's `name_ty` is `SideEffect` to be honest about what
/// `first` is: a side-effecting operation (`OutputSlotStore`) that
/// produces no value, only writes to the bound output buffer.
fn sequence(first: Term, rest: Term, term_ids: &mut TermIdSource, symbols: &mut SymbolTable) -> Term {
    let span = first.span;
    let name = symbols.alloc("_seq".to_string());
    Term {
        id: term_ids.next_id(),
        ty: side_effect_ty(),
        span,
        kind: TermKind::Let {
            name,
            name_ty: side_effect_ty(),
            rhs: Box::new(first),
            body: Box::new(rest),
        },
    }
}

/// `SideEffect`-typed terminator for the slot-store chain. Uses the
/// `UnitLit` term kind for now (the chain needs to end in *something*);
/// the wrapping `Term.ty` overrides to `SideEffect` so the chain's
/// type-level story is consistent end-to-end.
fn side_effect_terminator(term_ids: &mut TermIdSource, span: crate::ast::Span) -> Term {
    Term {
        id: term_ids.next_id(),
        ty: side_effect_ty(),
        span,
        kind: TermKind::UnitLit,
    }
}

fn side_effect_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::SideEffect, vec![])
}
