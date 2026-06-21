#![cfg(test)]

use super::run;
use super::VarRef;
use crate::ast::{Span, TypeName};
use crate::builtins::by_id;
use crate::tlc::{Def, DefMeta, Program, Term, TermId, TermIdSource, TermKind};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashMap;

/// Test helper that manages symbol table and term ID generation.
struct TestBuilder {
    symbols: SymbolTable,
    ids: TermIdSource,
}

impl TestBuilder {
    fn new() -> Self {
        TestBuilder {
            symbols: SymbolTable::new(),
            ids: TermIdSource::new(),
        }
    }

    fn sym(&mut self, name: &str) -> SymbolId {
        self.symbols.alloc(name.to_string())
    }

    fn next_id(&mut self) -> TermId {
        self.ids.next_id()
    }

    fn span(&self) -> Span {
        Span::dummy()
    }

    fn finish(self) -> SymbolTable {
        self.symbols
    }
}

#[test]
fn test_specialize_sign_f32() {
    let mut b = TestBuilder::new();
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);

    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    // Build: sign(x) where x: f32. After NameResolution covers every
    // catalog ref, real-program calls arrive at `specialize` as
    // `VarRef::Builtin`; the pass no longer touches `VarRef::Symbol`
    // (those are user bindings that may shadow catalog names).
    let x_var = Term {
        id: b.next_id(),
        ty: f32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };

    let sign_id = crate::builtins::catalog().known().sign;
    let sign_var = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![f32_ty.clone(), f32_ty.clone()]),
        span: b.span(),
        kind: TermKind::Var(VarRef::Builtin {
            id: sign_id,
            overload_idx: 0,
        }),
    };

    let sign_call = Term {
        id: b.next_id(),
        ty: f32_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(sign_var),
            args: vec![x_var],
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![Def {
            name: test_sym,
            ty: f32_ty.clone(),
            body: sign_call,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let specialized = run(program);

    // Check that sign became f32.sign
    match &specialized.defs[0].body.kind {
        TermKind::App { func, .. } => {
            let name = match &func.kind {
                TermKind::Var(VarRef::Symbol(sym)) => {
                    specialized.symbols.get(*sym).expect("BUG: symbol not in table").clone()
                }
                TermKind::Var(VarRef::Builtin { id, .. }) => by_id(*id).raw.surface_name.to_string(),
                _ => panic!("Expected Var, got {:?}", func.kind),
            };
            assert_eq!(name, "f32.sign");
        }
        _ => panic!("Expected App"),
    }
}

#[test]
fn test_specialize_min_i32() {
    let mut b = TestBuilder::new();
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let a_sym = b.sym("a");
    let b_sym = b.sym("b");
    let test_sym = b.sym("test");

    // Build: min(a, b) where a, b: i32. Catalog refs arrive as
    // `VarRef::Builtin` post-NameResolution.
    let a_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(a_sym)),
    };

    let b_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(b_sym)),
    };

    let partial_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), i32_ty.clone()]);
    let func_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), partial_ty.clone()]);

    let min_id = crate::builtins::catalog().known().min;
    let min_var = Term {
        id: b.next_id(),
        ty: func_ty,
        span: b.span(),
        kind: TermKind::Var(VarRef::Builtin {
            id: min_id,
            overload_idx: 0,
        }),
    };

    let min_a_b = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(min_var),
            args: vec![a_var, b_var],
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![Def {
            name: test_sym,
            ty: i32_ty.clone(),
            body: min_a_b,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let specialized = run(program);

    // Check that min became i32.min in the application
    match &specialized.defs[0].body.kind {
        TermKind::App { func, args } => {
            let name = match &func.kind {
                TermKind::Var(VarRef::Symbol(sym)) => {
                    specialized.symbols.get(*sym).expect("BUG: symbol not in table").clone()
                }
                TermKind::Var(VarRef::Builtin { id, .. }) => by_id(*id).raw.surface_name.to_string(),
                _ => panic!("Expected Var, got {:?}", func.kind),
            };
            assert_eq!(name, "i32.min");
            assert_eq!(args.len(), 2, "Expected 2 args in flattened App");
        }
        _ => panic!("Expected App"),
    }
}
