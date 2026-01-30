#[cfg(test)]
mod tests {
    use crate::ast::{BinaryOp, Span, TypeName};
    use crate::mir::Def as MirDef;
    use crate::tlc::to_mir::TlcToMir;
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

    /// Unit test for TlcToMir::transform.
    ///
    /// This directly constructs TLC terms and transforms them to MIR,
    /// testing the TLC→MIR lowering in isolation without running the full pipeline.
    #[test]
    fn test_transform_simple_function() {
        // def add(x, y) = x + y
        // In TLC (after lifting): def add = |x| |y| (+) x y
        let mut ids = TermIdSource::new();
        let span = make_span(1, 1);

        let x_var = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span,
            kind: TermKind::Var("x".to_string()),
        };

        let y_var = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span,
            kind: TermKind::Var("y".to_string()),
        };

        // Build: (+) x y as App(App(BinOp(+), x), y)
        let int_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let partial_ty = Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), int_ty.clone()]);
        let binop_ty = Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), partial_ty.clone()]);

        let binop_x = Term {
            id: ids.next_id(),
            ty: partial_ty,
            span,
            kind: TermKind::App {
                func: Box::new(Term {
                    id: ids.next_id(),
                    ty: binop_ty,
                    span,
                    kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
                }),
                arg: Box::new(x_var),
            },
        };

        let add_body = Term {
            id: ids.next_id(),
            ty: int_ty,
            span,
            kind: TermKind::App {
                func: Box::new(binop_x),
                arg: Box::new(y_var),
            },
        };

        // |y| body
        let lam_y = Term {
            id: ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![
                    Type::Constructed(TypeName::Int(32), vec![]),
                    Type::Constructed(TypeName::Int(32), vec![]),
                ],
            ),
            span,
            kind: TermKind::Lam {
                param: "y".to_string(),
                param_ty: Type::Constructed(TypeName::Int(32), vec![]),
                body: Box::new(add_body),
            },
        };

        // |x| |y| body
        let lam_x = Term {
            id: ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![
                    Type::Constructed(TypeName::Int(32), vec![]),
                    Type::Constructed(
                        TypeName::Arrow,
                        vec![
                            Type::Constructed(TypeName::Int(32), vec![]),
                            Type::Constructed(TypeName::Int(32), vec![]),
                        ],
                    ),
                ],
            ),
            span,
            kind: TermKind::Lam {
                param: "x".to_string(),
                param_ty: Type::Constructed(TypeName::Int(32), vec![]),
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

        let mir = TlcToMir::transform(&program, &std::collections::HashMap::new());

        assert_eq!(mir.defs.len(), 1);
        match &mir.defs[0] {
            MirDef::Function { name, params, .. } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("Expected Function"),
        }
    }
}
