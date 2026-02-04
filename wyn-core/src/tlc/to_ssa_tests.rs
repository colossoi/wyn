#![cfg(test)]

use crate::ast::{BinaryOp, Span, TypeName};
use crate::tlc::to_ssa::*;
use crate::tlc::{Def as TlcDef, DefMeta, Program as TlcProgram, Term, TermIdSource, TermKind};
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

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
fn test_convert_simple_function() {
    let mut b = TestBuilder::new();
    let span = b.span();

    let x_sym = b.sym("x");
    let y_sym = b.sym("y");
    let add_sym = b.sym("add");

    // def add(x, y) = x + y
    let x_var = Term {
        id: b.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::Var(x_sym),
    };

    let y_var = Term {
        id: b.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::Var(y_sym),
    };

    let binop_ty = Type::Constructed(
        TypeName::Arrow,
        vec![
            i32_ty(),
            Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
        ],
    );
    let binop_term = Term {
        id: b.next_id(),
        ty: binop_ty,
        span,
        kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
    };

    let binop_x = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
        span,
        kind: TermKind::App {
            func: Box::new(binop_term),
            arg: Box::new(x_var),
        },
    };

    let add_body = Term {
        id: b.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(binop_x),
            arg: Box::new(y_var),
        },
    };

    let lam_y = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
        span,
        kind: TermKind::Lam {
            param: y_sym,
            param_ty: i32_ty(),
            body: Box::new(add_body),
        },
    };

    let lam_x = Term {
        id: b.next_id(),
        ty: Type::Constructed(
            TypeName::Arrow,
            vec![
                i32_ty(),
                Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
            ],
        ),
        span,
        kind: TermKind::Lam {
            param: x_sym,
            param_ty: i32_ty(),
            body: Box::new(lam_y),
        },
    };

    let symbols = b.finish();

    let program = TlcProgram {
        defs: vec![TlcDef {
            name: add_sym,
            ty: lam_x.ty.clone(),
            body: lam_x,
            meta: DefMeta::Function,
            arity: 2,
        }],
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let ssa_program = convert_program(&program).unwrap();

    assert_eq!(ssa_program.functions.len(), 1);
    assert_eq!(ssa_program.functions[0].name, "add");
    assert_eq!(ssa_program.functions[0].body.params.len(), 2);
}
