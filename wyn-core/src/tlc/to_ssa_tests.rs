#![cfg(test)]

use crate::ast::{BinaryOp, Span, TypeName};
use crate::tlc::to_ssa::*;
use crate::tlc::{Def as TlcDef, DefMeta, Program as TlcProgram, Term, TermIdSource, TermKind};
use polytype::Type;

fn make_span(line: usize, col: usize) -> Span {
    Span {
        start_line: line,
        start_col: col,
        end_line: line,
        end_col: col + 1,
    }
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
fn test_convert_simple_function() {
    let mut ids = TermIdSource::new();
    let span = make_span(1, 1);

    // def add(x, y) = x + y
    let x_var = Term {
        id: ids.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::Var("x".to_string()),
    };

    let y_var = Term {
        id: ids.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::Var("y".to_string()),
    };

    let binop_ty = Type::Constructed(
        TypeName::Arrow,
        vec![
            i32_ty(),
            Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
        ],
    );
    let binop_term = Term {
        id: ids.next_id(),
        ty: binop_ty,
        span,
        kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
    };

    let binop_x = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
        span,
        kind: TermKind::App {
            func: Box::new(binop_term),
            arg: Box::new(x_var),
        },
    };

    let add_body = Term {
        id: ids.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(binop_x),
            arg: Box::new(y_var),
        },
    };

    let lam_y = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
        span,
        kind: TermKind::Lam {
            param: "y".to_string(),
            param_ty: i32_ty(),
            body: Box::new(add_body),
        },
    };

    let lam_x = Term {
        id: ids.next_id(),
        ty: Type::Constructed(
            TypeName::Arrow,
            vec![
                i32_ty(),
                Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
            ],
        ),
        span,
        kind: TermKind::Lam {
            param: "x".to_string(),
            param_ty: i32_ty(),
            body: Box::new(lam_y),
        },
    };

    let program = TlcProgram {
        defs: vec![TlcDef {
            name: "add".to_string(),
            ty: lam_x.ty.clone(),
            body: lam_x,
            meta: DefMeta::Function,
            arity: 2,
        }],
        uniforms: vec![],
        storage: vec![],
    };

    let ssa_program = convert_program(&program).unwrap();

    assert_eq!(ssa_program.functions.len(), 1);
    assert_eq!(ssa_program.functions[0].name, "add");
    assert_eq!(ssa_program.functions[0].body.params.len(), 2);
}
