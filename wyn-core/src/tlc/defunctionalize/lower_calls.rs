//! Lower explicit closure applications to direct calls.
//!
//! This pass needs no separate analysis or patch table. Once closure data is
//! stored on the callable term, lowering is a local bottom-up rewrite:
//! `closure(code, captures)(args)` becomes `code(args, captures)`.

use super::{ClosureConverted, Defunctionalized};
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::{Program, RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter, VarRef};
use crate::{LookupMap, SymbolId, SymbolTable};

#[derive(Debug)]
pub enum ClosureCallsLowerError {
    IndirectCall {
        def: SymbolId,
        func_kind: &'static str,
    },
    ArityMismatch {
        def: SymbolId,
        target: SymbolId,
        expected: usize,
        actual: usize,
    },
}

pub fn verify_closure_calls_lowered<S>(program: &Program<S>) -> Result<(), ClosureCallsLowerError>
where
    S: crate::tlc::Stage<Family = ClosureConverted>,
{
    let arities: LookupMap<SymbolId, usize> =
        program.defs.iter().map(|def| (def.name, def.arity)).collect();
    for def in &program.defs {
        verify_term(&def.body, def.name, &arities, &program.symbols)?;
    }
    Ok(())
}

fn verify_term(
    term: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    def: SymbolId,
    arities: &LookupMap<SymbolId, usize>,
    symbols: &SymbolTable,
) -> Result<(), ClosureCallsLowerError> {
    if let TermKind::App { func, args } = &term.kind {
        if !is_static_func(&func.kind) {
            return Err(ClosureCallsLowerError::IndirectCall {
                def,
                func_kind: discriminant_name(&func.kind),
            });
        }
        if let TermKind::Var(VarRef::Symbol(target)) = &func.kind {
            let expected = arities
                .get(target)
                .copied()
                .or_else(|| symbols.get(*target).and_then(|name| crate::builtins::intrinsic_arity(name)));
            if let Some(expected) = expected {
                if expected != args.len() {
                    return Err(ClosureCallsLowerError::ArityMismatch {
                        def,
                        target: *target,
                        expected,
                        actual: args.len(),
                    });
                }
            }
        }
    }

    let mut result = Ok(());
    term.for_each_child(&mut |child| {
        if result.is_ok() {
            result = verify_term(child, def, arities, symbols);
        }
    });
    result
}

fn is_static_func(kind: &TermKind<ExplicitClosurePayload, ExplicitCapturesPayload>) -> bool {
    matches!(
        kind,
        TermKind::Var(_) | TermKind::BinOp(_) | TermKind::UnOp(_) | TermKind::Extern(_)
    )
}

fn discriminant_name(kind: &TermKind<ExplicitClosurePayload, ExplicitCapturesPayload>) -> &'static str {
    match kind {
        TermKind::Var(_) => "Var",
        TermKind::BinOp(_) => "BinOp",
        TermKind::UnOp(_) => "UnOp",
        TermKind::Lambda(_) => "Lambda",
        TermKind::Closure(_) => "Closure",
        TermKind::App { .. } => "App",
        TermKind::Let { .. } => "Let",
        TermKind::IntLit(_) => "IntLit",
        TermKind::FloatLit(_) => "FloatLit",
        TermKind::BoolLit(_) => "BoolLit",
        TermKind::Extern(_) => "Extern",
        TermKind::If { .. } => "If",
        TermKind::Loop { .. } => "Loop",
        TermKind::Soac(_) => "Soac",
        TermKind::ArrayExpr(_) => "ArrayExpr",
        TermKind::Tuple(_) => "Tuple",
        TermKind::TupleProj { .. } => "TupleProj",
        TermKind::Index { .. } => "Index",
        TermKind::VecLit(_) => "VecLit",
        TermKind::UnitLit => "UnitLit",
        TermKind::Coerce { .. } => "Coerce",
    }
}

struct CallLowerer<'a> {
    term_ids: &'a mut TermIdSource,
}

impl TermRewriter<ExplicitClosurePayload, ExplicitCapturesPayload> for CallLowerer<'_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(
        &mut self,
        term: &mut Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> RewriteDecision {
        let TermKind::App { func, args } = &mut term.kind else {
            return RewriteDecision::Unchanged;
        };
        let TermKind::Closure(closure) = &mut func.kind else {
            return RewriteDecision::Unchanged;
        };
        if args.len() != closure.param_count {
            return RewriteDecision::Unchanged;
        }

        let code = closure.code;
        args.append(&mut closure.captures);
        func.kind = TermKind::Var(VarRef::Symbol(code));
        func.id = self.term_ids.next_id();
        RewriteDecision::Changed
    }
}

pub(super) fn run(program: &mut Program<Defunctionalized>) {
    let mut lowerer = CallLowerer {
        term_ids: &mut program.term_ids,
    };
    for def in &mut program.defs {
        lowerer.rewrite_tracked(&mut def.body);
    }
    verify_closure_calls_lowered(program)
        .unwrap_or_else(|error| panic!("closure-call lowering verifier failed: {error:?}"));
}

#[cfg(test)]
#[path = "lower_calls_tests.rs"]
mod lower_calls_tests;
