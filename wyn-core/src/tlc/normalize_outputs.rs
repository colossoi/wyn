//! Normalise compute entries into explicit per-slot stores.
//!
//! Before this pass: a compute entry's body is a Term whose tail
//! expression "produces" the entry's return value. Single-output entries
//! return a single Term; tuple-output entries return a `Tuple(N)`. The
//! downstream stack then decomposes this — `analyze_entry` walks the
//! tail looking for a parallelisable SOAC, `egir::assign_outputs`
//! flattens the tuple and per-slot retargets producer SOACs.
//!
//! After this pass: a compute entry's body is *unit-producing*. Its
//! tail is a chain of `TermKind::OutputSlotStore { slot_index, value, .. }`
//! terms, one per declared output, wired together via `Let { name: _ }`
//! sequencing and terminated by `UnitLit`. Single-output and
//! multi-output cases share one structural shape.
//!
//! The pass runs after `apply_ownership` (so `SoacDestination` is
//! already accurate) and before `lift_gathers` / `defunctionalize` /
//! `parallelize_soacs` (so the simpler IR shape feeds those passes).
//!
//! Phase 1 scope:
//!   * Tail shapes handled: a `TermKind::Tuple(operands)` matching
//!     `outputs.len()` → N slots; any other tail with a single-output
//!     entry → 1 slot.
//!   * Tail shapes deferred: `If` / `Loop` / function calls returning a
//!     tuple — error out with `NormalizeError::UnsupportedTail` and
//!     defer until a workload needs them.
//!
//! `SoacDestination::OutputView` association with bindings stays at
//! EGIR time (`from_tlc::build_entry_outputs` allocates `BindingRef`s
//! per output index). `OutputSlotStore` carries `slot_index: usize`,
//! and EGIR maps it to the slot's binding.

use super::{Def, DefMeta, Lambda, Program, Term, TermIdSource, TermKind};
use crate::SymbolTable;
use crate::ast::TypeName;
use polytype::Type;

#[cfg(test)]
#[path = "normalize_outputs_tests.rs"]
mod normalize_outputs_tests;

#[derive(Debug)]
pub enum NormalizeError {
    /// The entry tail doesn't match a shape this pass can decompose.
    UnsupportedTail {
        entry: String,
        shape: &'static str,
    },
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
            NormalizeError::UnsupportedTail { entry, shape } => {
                write!(
                    f,
                    "normalize_outputs: entry `{entry}` tail has shape `{shape}` which Phase 1 \
                     doesn't yet decompose into output slots; rewrite to a Tuple(...) of slot \
                     producers or report a bug"
                )
            }
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

fn normalize_entry(
    def: &mut Def,
    term_ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> Result<(), NormalizeError> {
    let DefMeta::EntryPoint(decl) = &def.meta else {
        unreachable!("filtered to EntryPoint")
    };
    let entry_name = decl.name.clone();
    let n_outputs = decl.outputs.len();

    // Peel the outer Lambda (entry params) and any tail Let-chain; the
    // tail expression is what we decompose. We rebuild the same
    // Lambda/Let frame around the new unit-producing body so all
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
    def.body = new_body;

    // Intentionally leave `def.ty` unchanged. The outer Lambda's `ret_ty`
    // is overwritten to Unit in `rewrite_body` so the body's
    // Term-level types are consistent (the chain root is `UnitLit`).
    // But the def's *signature* still names the original output type so
    // EGIR conversion can build `EntryOutput`s from it.

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
            // `new_body.ty`. For fully-normalised bodies that's Unit;
            // for the multi-output non-Tuple fallthrough in
            // `emit_slot_writes` (which preserves the tail's tuple
            // type) it's the original tuple. Forcing Unit here would
            // claim a return type the body doesn't actually produce.
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
    // (e.g. `image_store(img, xy, c)`) whose value is unit by type
    // checking. Return it unchanged so the side effect survives. The
    // body's tail still has unit type, so the enclosing Lambda's
    // `ret_ty: Unit` rewrite stays consistent.
    if n_outputs == 0 {
        return Ok(tail);
    }

    // Decompose `tail` into per-slot sources.
    let slot_sources: Vec<(Term, Type<TypeName>)> = match tail.kind {
        TermKind::Tuple(operands) if operands.len() == n_outputs => {
            operands.into_iter().map(|t| (t.clone(), t.ty)).collect()
        }
        TermKind::Tuple(operands) => {
            return Err(NormalizeError::SlotCountMismatch {
                entry: entry_name.to_string(),
                outputs: n_outputs,
                operands: operands.len(),
            });
        }
        _ if n_outputs == 1 => vec![(tail.clone(), tail.ty)],
        // Multi-output entries whose tail is a *single* value of tuple
        // type (e.g. a reduce returning `(u32, [4]u32)` or a function
        // call returning a tuple). Phase 1A leaves these unnormalised:
        // `egir::assign_outputs` still handles them via the existing
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

    // Build the chain: store_N → store_{N-1} → … → store_0 → UnitLit.
    // We construct from the inside out: the innermost is UnitLit; each
    // outer wraps the previous in `Let { name: _, rhs: store, body: inner }`.
    let mut chain = unit_term(term_ids, tail.span);
    for (i, (value, value_ty)) in slot_sources.into_iter().enumerate().rev() {
        let store = make_store(i, value, value_ty, term_ids);
        chain = sequence(store, chain, term_ids, symbols);
    }
    Ok(chain)
}

fn make_store(
    slot_index: usize,
    value: Term,
    value_ty: Type<TypeName>,
    term_ids: &mut TermIdSource,
) -> Term {
    let span = value.span;
    Term {
        id: term_ids.next_id(),
        ty: Type::Constructed(TypeName::Unit, vec![]),
        span,
        kind: TermKind::OutputSlotStore {
            slot_index,
            value: Box::new(value),
            value_ty,
        },
    }
}

/// `let _seq: () = first in rest` — a sequencing operator.
fn sequence(first: Term, rest: Term, term_ids: &mut TermIdSource, symbols: &mut SymbolTable) -> Term {
    let span = first.span;
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let name = symbols.alloc("_seq".to_string());
    Term {
        id: term_ids.next_id(),
        ty: unit_ty.clone(),
        span,
        kind: TermKind::Let {
            name,
            name_ty: unit_ty,
            rhs: Box::new(first),
            body: Box::new(rest),
        },
    }
}

fn unit_term(term_ids: &mut TermIdSource, span: crate::ast::Span) -> Term {
    Term {
        id: term_ids.next_id(),
        ty: Type::Constructed(TypeName::Unit, vec![]),
        span,
        kind: TermKind::UnitLit,
    }
}

fn tail_shape_name(kind: &TermKind) -> &'static str {
    match kind {
        TermKind::Var(_) => "Var",
        TermKind::BinOp(_) => "BinOp",
        TermKind::UnOp(_) => "UnOp",
        TermKind::Lambda(_) => "Lambda",
        TermKind::App { .. } => "App",
        TermKind::Let { .. } => "Let",
        TermKind::IntLit(_) => "IntLit",
        TermKind::FloatLit(_) => "FloatLit",
        TermKind::BoolLit(_) => "BoolLit",
        TermKind::UnitLit => "UnitLit",
        TermKind::Coerce { .. } => "Coerce",
        TermKind::Extern(_) => "Extern",
        TermKind::If { .. } => "If",
        TermKind::Loop { .. } => "Loop",
        TermKind::Soac(_) => "Soac",
        TermKind::ArrayExpr(_) => "ArrayExpr",
        TermKind::Tuple(_) => "Tuple",
        TermKind::TupleProj { .. } => "TupleProj",
        TermKind::Index { .. } => "Index",
        TermKind::VecLit(_) => "VecLit",
        TermKind::OutputSlotStore { .. } => "OutputSlotStore",
    }
}
