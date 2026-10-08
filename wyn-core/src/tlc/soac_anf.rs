//! Normalize nested SOAC expressions into flat let chains.
//!
//! This is a source-shape pass, not a fusion pass: egglog still makes every
//! producer/consumer decision. The flat form ensures TLC-to-egglog conversion
//! emits each SOAC as an explicit side effect and preserves the semantic value
//! edges that egglog needs.

use super::bindings::flatten_nested_let;
use super::data::Empty;
use super::{Bindings, Program, RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter};
use crate::SymbolTable;

#[derive(Debug, Clone, Copy)]
pub enum SoacsAnfNormalizedTag {}
pub type SoacsAnfNormalized =
    super::Program<SoacsAnfNormalizedTag, super::monomorphize::Monomorphic, super::context::RewriteGlobal>;

pub fn normalize_soacs_to_anf(program: super::stage::SoaNormalized) -> SoacsAnfNormalized {
    let Program {
        defs,
        mut symbols,
        mut term_ids,
        global_context,
        state: _,
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
    let program = Program::from_parts(defs, symbols, term_ids, global_context);
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

    fn rewrite_owned_node(&mut self, term: Term<Empty, Empty>) -> (Term<Empty, Empty>, RewriteDecision) {
        let (term, hoisted) = hoist_soac_arguments(term, self.symbols, self.term_ids);
        let (term, flattened) = flatten_nested_let(term, self.term_ids);
        let changed = hoisted || flattened;
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
    let mut bindings = Bindings::new();
    for arg in args {
        if matches!(&arg.kind, TermKind::Soac(_)) {
            new_args.push(bindings.name(arg, "_anf", symbols, term_ids));
        } else {
            new_args.push(arg);
        }
    }

    let app = Term::fresh(term_ids, ty, span, TermKind::App { func, args: new_args });
    (bindings.finish(app, term_ids), true)
}

fn verify_flattened(program: &SoacsAnfNormalized) -> Result<(), ()> {
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
