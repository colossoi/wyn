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
    program: Program<super::stage::ConditionalProducersCanonicalized>,
) -> Program<SoacsAnfNormalized> {
    let Program {
        defs,
        mut symbols,
        def_syms,
        mut term_ids,
        global_context,
    } = program;
    let mut rewriter = SoacAnfRewriter {
        symbols: &mut symbols,
        term_ids: &mut term_ids,
    };
    let defs = defs
        .into_iter()
        .map(|def| super::Def {
            body: def.body.rewrite_owned(&mut rewriter),
            ..def
        })
        .collect();
    let program = Program::from_parts(defs, symbols, def_syms, term_ids, global_context);
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

    fn rewrite_owned_node_before_children(
        &mut self,
        mut term: Term<Empty, Empty>,
    ) -> (Term<Empty, Empty>, RewriteDecision) {
        let mut changed = false;
        loop {
            let (flattened, flattened_once) = flatten_nested_let(term, self.term_ids);
            term = flattened;
            if !flattened_once {
                break;
            }
            changed = true;
        }
        let decision = if changed { RewriteDecision::Changed } else { RewriteDecision::Unchanged };
        (term, decision)
    }

    fn rewrite_owned_node(&mut self, term: Term<Empty, Empty>) -> (Term<Empty, Empty>, RewriteDecision) {
        let mut changed = false;
        let (mut term, hoisted) = hoist_soac_arguments(term, self.symbols, self.term_ids);
        changed |= hoisted;
        loop {
            let (flattened, flattened_once) = flatten_nested_let(term, self.term_ids);
            term = flattened;
            if !flattened_once {
                break;
            }
            changed = true;
        }
        let decision = if changed { RewriteDecision::Changed } else { RewriteDecision::Unchanged };
        (term, decision)
    }
}

fn hoist_soac_arguments(
    term: Term<Empty, Empty>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> (Term<Empty, Empty>, bool) {
    let is_hoistable = matches!(
        &term.kind,
        TermKind::App { args, .. }
            if args.iter().any(|arg| matches!(&arg.kind, TermKind::Soac(_)))
    );
    if !is_hoistable {
        return (term, false);
    }

    let Term {
        id: _,
        ty,
        span,
        kind,
    } = term;
    let TermKind::App { func, args } = kind else {
        unreachable!("checked application shape");
    };
    let mut new_args = Vec::with_capacity(args.len());
    let mut bindings = Vec::new();
    for arg in args {
        if matches!(&arg.kind, TermKind::Soac(_)) {
            let name = symbols.alloc("_anf".to_string());
            let ty = arg.ty.clone();
            bindings.push(LetBinding {
                name,
                name_ty: ty.clone(),
                rhs: arg,
                span,
            });
            new_args.push(Term {
                id: term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Var(VarRef::Symbol(name)),
            });
        } else {
            new_args.push(arg);
        }
    }

    let app = Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::App { func, args: new_args },
    };
    (wrap_let_bindings(bindings, app, term_ids), true)
}

fn flatten_nested_let(term: Term<Empty, Empty>, term_ids: &mut TermIdSource) -> (Term<Empty, Empty>, bool) {
    let is_nested = matches!(
        &term.kind,
        TermKind::Let { rhs, .. } if matches!(rhs.kind, TermKind::Let { .. })
    );
    if !is_nested {
        return (term, false);
    }

    let Term { id, ty, span, kind } = term;
    let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = kind
    else {
        unreachable!("checked outer let shape");
    };

    let mut bindings = Vec::new();
    let mut inner = *rhs;
    while matches!(&inner.kind, TermKind::Let { .. }) {
        let Term {
            id: _,
            ty: _,
            span: inner_span,
            kind: inner_kind,
        } = inner;
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
            span: inner_span,
        });
        inner = *body;
    }

    let mut result = Term {
        id: term_ids.next_id(),
        ty: ty.clone(),
        span,
        kind: TermKind::Let {
            name,
            name_ty,
            rhs: Box::new(inner),
            body,
        },
    };
    let binding_count = bindings.len();
    for (index, binding) in bindings.into_iter().rev().enumerate() {
        let binding_id = if index + 1 == binding_count { id } else { term_ids.next_id() };
        result = Term {
            id: binding_id,
            ty: ty.clone(),
            span: binding.span,
            kind: TermKind::Let {
                name: binding.name,
                name_ty: binding.name_ty,
                rhs: Box::new(binding.rhs),
                body: Box::new(result),
            },
        };
    }
    (result, true)
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
