#[cfg(test)]
mod tests {
    use crate::ast::{BinaryOp, Span, TypeName};
    use crate::mir::{self, Def as MirDef};
    use crate::tlc::to_mir::TlcToMir;
    use crate::tlc::{
        Def as TlcDef, DefMeta, FunctionName, Program as TlcProgram, Term, TermIdSource, TermKind,
    };
    use crate::{Compiler, build_builtins};
    use polytype::Type;

    /// Run full MIR pipeline (but not SPIR-V lowering) on source code.
    /// Returns the final MIR after all passes.
    fn full_mir_pipeline(input: &str) -> mir::Program {
        let mut frontend = crate::cached_frontend();
        let parsed = Compiler::parse(input, &mut frontend.node_counter).expect("Parsing failed");
        let alias_checked = parsed
            .desugar(&mut frontend.node_counter)
            .expect("Desugaring failed")
            .resolve(&frontend.module_manager)
            .expect("Name resolution failed")
            .fold_ast_constants()
            .type_check(&frontend.module_manager, &mut frontend.schemes)
            .expect("Type checking failed")
            .alias_check()
            .expect("Alias checking failed");

        let builtins = build_builtins(&alias_checked.ast, &frontend.module_manager);
        alias_checked
            .to_tlc(builtins, &frontend.schemes, &frontend.module_manager)
            .skip_partial_eval()
            .lift()
            .to_mir()
            .hoist_materializations()
            .normalize()
            .monomorphize()
            .expect("Monomorphization failed")
            .skip_folding()
            .filter_reachable()
            .lift_bindings()
            .mir
    }

    fn make_span(line: usize, col: usize) -> Span {
        Span {
            start_line: line,
            start_col: col,
            end_line: line,
            end_col: col + 1,
        }
    }

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
        let binop_x = Term {
            id: ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![
                    Type::Constructed(TypeName::Int(32), vec![]),
                    Type::Constructed(TypeName::Int(32), vec![]),
                ],
            ),
            span,
            kind: TermKind::App {
                func: Box::new(FunctionName::BinOp(BinaryOp { op: "+".to_string() })),
                arg: Box::new(x_var),
            },
        };

        let add_body = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(binop_x))),
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

    /// Run MIR pipeline up to (but not including) reachability filtering.
    /// This lets us see what MIR was generated before dead code elimination.
    #[allow(dead_code)]
    fn mir_before_reachability(input: &str) -> mir::Program {
        let mut frontend = crate::cached_frontend();
        let parsed = Compiler::parse(input, &mut frontend.node_counter).expect("Parsing failed");
        let alias_checked = parsed
            .desugar(&mut frontend.node_counter)
            .expect("Desugaring failed")
            .resolve(&frontend.module_manager)
            .expect("Name resolution failed")
            .fold_ast_constants()
            .type_check(&frontend.module_manager, &mut frontend.schemes)
            .expect("Type checking failed")
            .alias_check()
            .expect("Alias checking failed");

        let builtins = build_builtins(&alias_checked.ast, &frontend.module_manager);
        alias_checked
            .to_tlc(builtins, &frontend.schemes, &frontend.module_manager)
            .skip_partial_eval()
            .lift()
            .to_mir()
            .hoist_materializations()
            .normalize()
            .monomorphize()
            .expect("Monomorphization failed")
            .skip_folding()
            .mir
    }

    /// Integration test: full pipeline from source to MIR with `map` intrinsic.
    /// This test is for debugging the map/prelude challenge.
    #[test]
    fn test_map_intrinsic_full_pipeline() {
        let source = r#"
def myfunc(x: i32, y: i32, arr: [4]i32) [4]i32 =
  map(|e| e + x + y, arr)

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
  let result = myfunc(1, 2, [1, 2, 3, 4]) in
  @[f32.i32(result[0]), 0.0, 0.0, 1.0]
"#;

        let mir = mir_before_reachability(source);

        // Print MIR
        eprintln!("\n=== FINAL MIR ===\n{}\n=================\n", mir);
    }
}
