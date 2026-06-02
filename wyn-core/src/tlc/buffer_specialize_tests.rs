//! Tests for the buffer-specialization pass.
//!
//! Currently parked: a single repro for the known gap in
//! `rewrite_specialized_body`'s alias handling — see
//! `bare_var_view_alias_via_if_not_resolved`.

use super::BufferSpecializer;
use crate::ast::{Span, TypeName};
use crate::tlc::{Term, TermId, TermIdSource, TermKind, VarRef};
use crate::{BindingRef, SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashMap;

// ---------- type helpers ----------

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

/// `[]i32` view: `Array[i32, ?_size, View]`.
fn view_i32_ty() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty(),
            Type::Variable(999),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    )
}

// ---------- builder ----------

struct B {
    symbols: SymbolTable,
    ids: TermIdSource,
}

impl B {
    fn new() -> Self {
        B {
            symbols: SymbolTable::new(),
            ids: TermIdSource::new(),
        }
    }

    fn sym(&mut self, name: &str) -> SymbolId {
        self.symbols.alloc(name.to_string())
    }

    fn id(&mut self) -> TermId {
        self.ids.next_id()
    }

    fn term(&mut self, kind: TermKind, ty: Type<TypeName>) -> Term {
        Term {
            id: self.id(),
            ty,
            span: Span::new(0, 0, 0, 0),
            kind,
        }
    }

    fn var(&mut self, sym: SymbolId, ty: Type<TypeName>) -> Term {
        self.term(TermKind::Var(VarRef::Symbol(sym)), ty)
    }
}

/// Construct a `BufferSpecializer` with empty state. Test-local only;
/// fields are private but `#[path]`-included tests are child modules of
/// `buffer_specialize` and so see them.
fn fresh_specializer(symbols: SymbolTable, term_ids: TermIdSource) -> BufferSpecializer {
    BufferSpecializer {
        symbols,
        term_ids,
        buffer_map: HashMap::new(),
        specializations: HashMap::new(),
        new_defs: Vec::new(),
        def_map: HashMap::new(),
    }
}

/// Walk a term and collect every `(SymbolId)` referenced by a bare
/// `Var(Symbol)` — used to detect surviving alias references in the
/// rewritten body.
fn collect_var_refs(term: &Term, out: &mut Vec<SymbolId>) {
    if let TermKind::Var(VarRef::Symbol(s)) = &term.kind {
        out.push(*s);
    }
    term.for_each_child(&mut |c| collect_var_refs(c, out));
}

/// Parked: `let alias = (if cond then view_param else view_param) in
/// length(alias)`. The Let arm's `try_resolve_view_expr` only matches
/// bare `Var(view_param)`, slice intrinsics, and a few App shapes —
/// not `If`. So the alias never gets added to `view_params`, and the
/// downstream `length(alias)` reaches the Var arm of
/// `rewrite_specialized_body` and falls through unchanged. After the
/// pass, the rewritten body still contains a bare `Var(alias)` of view
/// type that no downstream pass tracks.
///
/// No production source exercises this today (conway's view-param uses
/// all flow through user-function calls, which `specialize_call`
/// handles by discarding the rewritten arg). Keeping this as a
/// `#[ignore]`'d repro so the gap is documented and discoverable; drop
/// the ignore once `try_resolve_view_expr` learns to descend through
/// `If` (or `view_params` is extended on any let RHS that's view-typed
/// regardless of shape).
#[test]
#[ignore = "parked: view-param alias via If RHS leaves a bare Var(alias) in the body"]
fn bare_var_view_alias_via_if_not_resolved() {
    let mut b = B::new();
    let board = b.sym("board");
    let alias = b.sym("alias");
    let offset = b.sym("board_offset");
    let len = b.sym("board_len");

    // view_params pre-seeded with `board` (as `process_entry_point`
    // would after reading the entry's param_bindings).
    let mut view_params: HashMap<SymbolId, (SymbolId, SymbolId, BindingRef, Type<TypeName>)> =
        HashMap::new();
    view_params.insert(board, (offset, len, BindingRef::new(0, 0), i32_ty()));

    // Build `let alias = (if true then Var(board) else Var(board)) in
    // _w_intrinsic_length(alias)`.
    let cond = b.term(TermKind::BoolLit(true), bool_ty());
    let then_b = b.var(board, view_i32_ty());
    let else_b = b.var(board, view_i32_ty());
    let if_term = b.term(
        TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_b),
            else_branch: Box::new(else_b),
        },
        view_i32_ty(),
    );
    // _w_intrinsic_length call: App(Var(_w_intrinsic_length), [Var(alias)]).
    // The Var arm we want to exercise fires in the App-args recursion
    // when `alias` reaches it without being in view_params.
    let length_sym = b.sym("_w_intrinsic_length");
    let length_func = b.var(length_sym, u32_ty()); // ty approximate; not load-bearing
    let alias_ref = b.var(alias, view_i32_ty());
    let length_call = b.term(
        TermKind::App {
            func: Box::new(length_func),
            args: vec![alias_ref],
        },
        u32_ty(),
    );
    let let_term = b.term(
        TermKind::Let {
            name: alias,
            name_ty: view_i32_ty(),
            rhs: Box::new(if_term),
            body: Box::new(length_call),
        },
        u32_ty(),
    );

    let mut spec = fresh_specializer(b.symbols, b.ids);
    let rewritten = spec.rewrite_specialized_body(&let_term, &view_params);

    // The rewritten body must NOT contain a bare `Var(alias)` — that
    // would mean the alias survived as an untracked view-typed local,
    // which downstream lowering can't represent.
    let mut refs = Vec::new();
    collect_var_refs(&rewritten, &mut refs);
    assert!(
        !refs.contains(&alias),
        "Var(alias) survived in rewritten body; the Let arm failed to extend view_params \
         for an If-shaped view-yielding RHS."
    );
}
