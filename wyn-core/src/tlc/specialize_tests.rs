#![cfg(test)]

use crate::ast::{Span, TypeName};
use crate::tlc::specialize::specialize;
use crate::tlc::{Def, DefMeta, Program, Term, TermIdSource, TermKind};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

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

    fn next_id(&mut self) -> super::TermId {
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
    let sign_sym = b.sym("sign");
    let test_sym = b.sym("test");

    // Build: sign(x) where x: f32
    let x_var = Term {
        id: b.next_id(),
        ty: f32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(x_sym),
    };

    let sign_var = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![f32_ty.clone(), f32_ty.clone()]),
        span: b.span(),
        kind: TermKind::Var(sign_sym),
    };

    let sign_call = Term {
        id: b.next_id(),
        ty: f32_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(sign_var),
            arg: Box::new(x_var),
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
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let specialized = specialize(program);

    // Check that sign became f32.sign
    match &specialized.defs[0].body.kind {
        TermKind::App { func, .. } => match &func.kind {
            TermKind::Var(sym) => {
                let name = specialized.symbols.get(*sym).expect("BUG: symbol not in table");
                assert_eq!(name, "f32.sign");
            }
            _ => panic!("Expected Var, got {:?}", func.kind),
        },
        _ => panic!("Expected App"),
    }
}

#[test]
fn test_specialize_min_i32() {
    let mut b = TestBuilder::new();
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let a_sym = b.sym("a");
    let b_sym = b.sym("b");
    let min_sym = b.sym("min");
    let test_sym = b.sym("test");

    // Build: min(a, b) where a, b: i32
    // In curried form: App(App(Var("min"), a), b)
    let a_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(a_sym),
    };

    let b_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(b_sym),
    };

    let partial_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), i32_ty.clone()]);
    let func_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), partial_ty.clone()]);

    let min_var = Term {
        id: b.next_id(),
        ty: func_ty,
        span: b.span(),
        kind: TermKind::Var(min_sym),
    };

    let min_a = Term {
        id: b.next_id(),
        ty: partial_ty,
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(min_var),
            arg: Box::new(a_var),
        },
    };

    let min_a_b = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(min_a),
            arg: Box::new(b_var),
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
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let specialized = specialize(program);

    // Check that min became i32.min in the inner application
    match &specialized.defs[0].body.kind {
        TermKind::App { func, .. } => match &func.kind {
            TermKind::App { func: inner_func, .. } => match &inner_func.kind {
                TermKind::Var(sym) => {
                    let name = specialized.symbols.get(*sym).expect("BUG: symbol not in table");
                    assert_eq!(name, "i32.min");
                }
                _ => panic!("Expected Var, got {:?}", inner_func.kind),
            },
            _ => panic!("Expected inner App, got {:?}", func.kind),
        },
        _ => panic!("Expected App"),
    }
}
