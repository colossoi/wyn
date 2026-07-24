//! Normalize nested SOAC expressions into flat let chains.
//!
//! This is a source-shape pass, not a fusion pass: EGIR still makes every
//! producer/consumer decision. The flat form ensures TLC-to-EGIR conversion
//! emits each SOAC as an explicit side effect and preserves the semantic value
//! edges that EGIR needs.

use super::data::Empty;
use super::{
    wrap_let_bindings, LetBinding, Program, RewriteDecision, Term, TermId, TermIdSource, TermKind,
    TermRewriter, VarRef,
};
use crate::SymbolTable;

#[derive(Debug, Clone, Copy, Default)]
pub struct SoacsAnfNormalized;

impl super::Stage for SoacsAnfNormalized {
    type Family = super::monomorphize::Monomorphic;
    type GlobalContext = super::context::RewriteGlobal;
}

pub fn run(
    mut program: Program<super::stage::ConditionalProducersCanonicalized>,
) -> Program<SoacsAnfNormalized> {
    let mut rewriter = SoacAnfRewriter {
        symbols: &mut program.symbols,
        term_ids: &mut program.term_ids,
    };
    for def in &mut program.defs {
        rewriter.rewrite_tracked(&mut def.body);
    }

    let program: Program<SoacsAnfNormalized> = program.into_stage();
    debug_assert!(
        verify_flattened(&program).is_ok(),
        "SOAC ANF normalization left a nested let rhs"
    );
    program
}

struct SoacAnfRewriter<'a> {
    symbols: &'a mut SymbolTable,
    term_ids: &'a mut TermIdSource,
}

impl TermRewriter<Empty, Empty> for SoacAnfRewriter<'_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node_before_children(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        let mut changed = false;
        while flatten_nested_let(term, self.term_ids) {
            changed = true;
        }
        if changed {
            RewriteDecision::Changed
        } else {
            RewriteDecision::Unchanged
        }
    }

    fn rewrite_node(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        let mut changed = false;
        if let Some(replacement) = hoist_soac_arguments(term, self.symbols, self.term_ids) {
            *term = replacement;
            changed = true;
        }
        while flatten_nested_let(term, self.term_ids) {
            changed = true;
        }
        if changed {
            RewriteDecision::Changed
        } else {
            RewriteDecision::Unchanged
        }
    }
}

fn hoist_soac_arguments(
    term: &Term<Empty, Empty>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term<Empty, Empty>> {
    let TermKind::App { func, args } = &term.kind else {
        return None;
    };
    if !args.iter().any(|arg| matches!(arg.kind, TermKind::Soac(_))) {
        return None;
    }

    let mut new_args = Vec::with_capacity(args.len());
    let mut bindings = Vec::new();
    for arg in args {
        if matches!(arg.kind, TermKind::Soac(_)) {
            let name = symbols.alloc("_anf".to_string());
            let ty = arg.ty.clone();
            bindings.push(LetBinding {
                name,
                name_ty: ty.clone(),
                rhs: arg.clone(),
                span: term.span,
            });
            new_args.push(Term {
                id: term_ids.next_id(),
                ty,
                span: term.span,
                kind: TermKind::Var(VarRef::Symbol(name)),
            });
        } else {
            new_args.push(arg.clone());
        }
    }

    let app = Term {
        id: term_ids.next_id(),
        ty: term.ty.clone(),
        span: term.span,
        kind: TermKind::App {
            func: func.clone(),
            args: new_args,
        },
    };
    Some(wrap_let_bindings(bindings, app, term_ids))
}

fn flatten_nested_let(term: &mut Term<Empty, Empty>, term_ids: &mut TermIdSource) -> bool {
    let is_nested = matches!(
        &term.kind,
        TermKind::Let { rhs, .. } if matches!(rhs.kind, TermKind::Let { .. })
    );
    if !is_nested {
        return false;
    }

    let outer_kind = std::mem::replace(&mut term.kind, TermKind::UnitLit);
    let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = outer_kind
    else {
        unreachable!("checked outer let shape");
    };

    let mut bindings = Vec::new();
    let mut inner = *rhs;
    while matches!(&inner.kind, TermKind::Let { .. }) {
        let inner_kind = std::mem::replace(&mut inner.kind, TermKind::UnitLit);
        let TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } = inner_kind
        else {
            unreachable!("checked inner let shape");
        };
        bindings.push(LetBinding {
            name,
            name_ty,
            rhs: *rhs,
            span: inner.span,
        });
        inner = *body;
    }

    let mut result = Term {
        id: term_ids.next_id(),
        ty: term.ty.clone(),
        span: term.span,
        kind: TermKind::Let {
            name,
            name_ty,
            rhs: Box::new(inner),
            body,
        },
    };
    let binding_count = bindings.len();
    for (index, binding) in bindings.into_iter().rev().enumerate() {
        let id = if index + 1 == binding_count { term.id } else { term_ids.next_id() };
        result = Term {
            id,
            ty: term.ty.clone(),
            span: binding.span,
            kind: TermKind::Let {
                name: binding.name,
                name_ty: binding.name_ty,
                rhs: Box::new(binding.rhs),
                body: Box::new(result),
            },
        };
    }
    *term = result;
    true
}

fn verify_flattened(program: &Program<SoacsAnfNormalized>) -> Result<(), ()> {
    fn walk(term: &Term<Empty, Empty>) -> Result<(), ()> {
        if matches!(&term.kind, TermKind::Let { rhs, .. } if matches!(rhs.kind, TermKind::Let { .. })) {
            return Err(());
        }
        let mut result = Ok(());
        term.for_each_child(&mut |child| {
            if result.is_ok() {
                result = walk(child);
            }
        });
        result
    }
    program.defs.iter().try_for_each(|def| walk(&def.body))
}
