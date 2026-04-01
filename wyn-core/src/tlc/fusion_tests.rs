    use crate::tlc::fusion::*;
    use crate::ast::Span;

    fn dummy_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn mk_term(kind: TermKind, ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
        Term {
            id: term_ids.next_id(),
            ty,
            span: dummy_span(),
            kind,
        }
    }

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn f32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Float(32), vec![])
    }

    fn array_ty(elem: Type<TypeName>) -> Type<TypeName> {
        Type::Constructed(
            TypeName::Array,
            vec![
                elem,
                Type::Variable(0), // size
                Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            ],
        )
    }

    /// Build a simple map: `map(lam, [input])`
    fn mk_map(lam: Lambda, input: Term, result_ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
        mk_term(
            TermKind::Soac(SoacOp::Map {
                lam,
                inputs: vec![ArrayExpr::Ref(Box::new(input))],
            }),
            result_ty,
            term_ids,
        )
    }

    /// Build a lambda with one parameter
    fn mk_lambda1(param: SymbolId, param_ty: Type<TypeName>, body: Term, ret_ty: Type<TypeName>) -> Lambda {
        Lambda {
            params: vec![(param, param_ty)],
            body: Box::new(body),
            ret_ty,
            captures: vec![],
        }
    }

    // -------------------------------------------------------------------------
    // Test: simple map(g, map(f, a)) → map(g∘f, a)
    // -------------------------------------------------------------------------
    #[test]
    fn test_simple_map_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        // Symbols
        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string()); // f's param
        let y_sym = symbols.alloc("y".to_string()); // g's param

        // f: i32 → i32 (identity-like, just returns x)
        let f_body = mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids);
        let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());

        // g: i32 → i32 (identity-like, just returns y)
        let g_body = mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids);
        let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());

        // a: [i32]
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);

        // Producer: map(f, a)
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        // Consumer: map(g, b)
        let b_ref = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(g, b_ref, array_ty(i32_ty()), &mut term_ids);

        // let b = map(f, a) in map(g, b)
        let program_body = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // The result should be a single Map (no Let binding)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                // Input should be 'a' (the original array)
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => match &t.kind {
                        TermKind::Var(s) => assert_eq!(*s, a_sym),
                        other => panic!("Expected Var(a), got {:?}", other),
                    },
                    other => panic!("Expected Ref, got {:?}", other),
                }

                // Lambda should have f's param (x)
                assert_eq!(lam.params.len(), 1);
                assert_eq!(lam.params[0].0, x_sym);

                // Body should be: let _fused = x in _fused
                // (g's body is y, substituted to _fused; f's body is x)
                match &lam.body.kind {
                    TermKind::Let { rhs, body, .. } => {
                        // rhs is f's body (Var(x))
                        assert!(matches!(&rhs.kind, TermKind::Var(s) if *s == x_sym));
                        // body should be Var(_fused) — the fresh symbol
                        assert!(matches!(&body.kind, TermKind::Var(_)));
                    }
                    other => panic!("Expected Let (composed body), got {:?}", other),
                }
            }
            other => panic!("Expected fused Soac(Map), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: chain of three maps fused
    // -------------------------------------------------------------------------
    #[test]
    fn test_chain_of_three_maps() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let c_sym = symbols.alloc("c".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());
        let z_sym = symbols.alloc("z".to_string());

        // f, g, h: i32 → i32
        let f = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let h = mk_lambda1(
            z_sym,
            i32_ty(),
            mk_term(TermKind::Var(z_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        let b_ref = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let middle = mk_map(g, b_ref, array_ty(i32_ty()), &mut term_ids);

        let c_ref = mk_term(TermKind::Var(c_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(h, c_ref, array_ty(i32_ty()), &mut term_ids);

        // let b = map(f, a) in let c = map(g, b) in map(h, c)
        let inner_let = mk_term(
            TermKind::Let {
                name: c_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(middle),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let outer_let = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(inner_let),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer_let,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // Should be a single Map with a's input (all three fused)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { inputs, lam }) => {
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Lambda param should be f's original param (x)
                assert_eq!(lam.params[0].0, x_sym);
            }
            other => panic!("Expected fully fused Map, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: multi-use intermediate (no fusion)
    // -------------------------------------------------------------------------
    #[test]
    fn test_multi_use_no_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());

        let f = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        // Consumer uses b
        let b_ref1 = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(g, b_ref1, array_ty(i32_ty()), &mut term_ids);

        // Second use of b (in an App, making count_uses == 2)
        let b_ref2 = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);

        // Body: App(consumer, b) — artificial but creates two uses
        let body = mk_term(
            TermKind::App {
                func: Box::new(consumer),
                arg: Box::new(b_ref2),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program_body = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(body),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let result = fuse_maps(program);

        // Should still be a Let (no fusion because b is used twice)
        assert!(matches!(&result.defs[0].body.kind, TermKind::Let { .. }));
    }

    // -------------------------------------------------------------------------
    // Test: zip-fused producer: map(g, map(f, zip(a,b)))
    // -------------------------------------------------------------------------
    #[test]
    fn test_zip_fused_producer() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let intermediate_sym = symbols.alloc("inter".to_string());
        let x1_sym = symbols.alloc("x1".to_string());
        let x2_sym = symbols.alloc("x2".to_string());
        let y_sym = symbols.alloc("y".to_string());

        let _tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), f32_ty()]);

        // f: (i32, f32) → i32 — takes two params (zip-fused)
        let f = Lambda {
            params: vec![(x1_sym, i32_ty()), (x2_sym, f32_ty())],
            body: Box::new(mk_term(TermKind::Var(x1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };

        // Producer: map(f, [a, b]) with zip-fused inputs
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let b = mk_term(TermKind::Var(b_sym), array_ty(f32_ty()), &mut term_ids);
        let producer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![ArrayExpr::Ref(Box::new(a)), ArrayExpr::Ref(Box::new(b))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        // g: i32 → i32
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        // Consumer: map(g, inter) with single input
        let inter_ref = mk_term(TermKind::Var(intermediate_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(g, inter_ref, array_ty(i32_ty()), &mut term_ids);

        // let inter = map(f, [a, b]) in map(g, inter)
        let program_body = mk_term(
            TermKind::Let {
                name: intermediate_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // Should be a Map with [a, b] inputs (producer's multi-inputs preserved)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 2);
                // Lambda should have f's params (x1, x2)
                assert_eq!(lam.params.len(), 2);
                assert_eq!(lam.params[0].0, x1_sym);
                assert_eq!(lam.params[1].0, x2_sym);
            }
            other => panic!("Expected fused Map with 2 inputs, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: consumer with multiple inputs (no fusion)
    // -------------------------------------------------------------------------
    #[test]
    fn test_consumer_multi_input_no_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let other_sym = symbols.alloc("other".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y1_sym = symbols.alloc("y1".to_string());
        let y2_sym = symbols.alloc("y2".to_string());

        let f = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        // g: (i32, i32) → i32, takes two inputs
        let g = Lambda {
            params: vec![(y1_sym, i32_ty()), (y2_sym, i32_ty())],
            body: Box::new(mk_term(TermKind::Var(y1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };

        // Consumer: map(g, [b, other]) — b plus another array
        let b_ref = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let other = mk_term(TermKind::Var(other_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: g,
                inputs: vec![ArrayExpr::Ref(Box::new(b_ref)), ArrayExpr::Ref(Box::new(other))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program_body = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let result = fuse_maps(program);

        // Should NOT fuse — consumer has multiple inputs
        assert!(matches!(&result.defs[0].body.kind, TermKind::Let { .. }));
    }

    // -------------------------------------------------------------------------
    // Test: inline map(f, map(g, a)) — no Let binding
    // -------------------------------------------------------------------------
    #[test]
    fn test_inline_map_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());

        // Inner: map(g, a) where g(x) = x
        let g = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let inner_map = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

        // Outer: map(f, inner_map) where f(y) = y — inner_map is inline (Ref)
        let f = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![ArrayExpr::Ref(Box::new(inner_map))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // Should be a single Map with a's input, param x (inner g's param)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Lambda should have g's param (x), not f's param (y)
                assert_eq!(lam.params.len(), 1);
                assert_eq!(lam.params[0].0, x_sym);
            }
            other => panic!("Expected fused Map, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: inline chain map(f, map(g, map(h, a))) — no Let bindings
    // -------------------------------------------------------------------------
    #[test]
    fn test_inline_chain_of_three() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());
        let z_sym = symbols.alloc("z".to_string());

        // h(x) = x
        let h = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let inner = mk_map(h, a, array_ty(i32_ty()), &mut term_ids);

        // g(y) = y
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let middle = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: g,
                inputs: vec![ArrayExpr::Ref(Box::new(inner))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        // f(z) = z
        let f = mk_lambda1(
            z_sym,
            i32_ty(),
            mk_term(TermKind::Var(z_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![ArrayExpr::Ref(Box::new(middle))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // All three fused into one Map over a
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Innermost param (h's param x) should be the lambda param
                assert_eq!(lam.params.len(), 1);
                assert_eq!(lam.params[0].0, x_sym);
            }
            other => panic!("Expected fully fused Map, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: zip-fused consumer — map(f, zip(map(g, a), b))
    // One input is an inline nested map, the other is not
    // -------------------------------------------------------------------------
    #[test]
    fn test_zip_fused_consumer_inline() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y1_sym = symbols.alloc("y1".to_string());
        let y2_sym = symbols.alloc("y2".to_string());

        // Inner: map(g, a) where g(x) = x
        let g = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let inner_map = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

        // Outer: map(f, zip(map(g, a), b)) — zip already absorbed into multi-input
        // f(y1, y2) = y1  (takes two params from zip)
        let f = Lambda {
            params: vec![(y1_sym, i32_ty()), (y2_sym, f32_ty())],
            body: Box::new(mk_term(TermKind::Var(y1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };
        let b = mk_term(TermKind::Var(b_sym), array_ty(f32_ty()), &mut term_ids);
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![
                    ArrayExpr::Ref(Box::new(inner_map)), // will be fused
                    ArrayExpr::Ref(Box::new(b)),         // stays as-is
                ],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // The inner map should be fused: y1's slot replaced by g's param x,
        // input[0] is now Ref(a), input[1] is Ref(b)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 2);
                // First input: a (was map(g, a), now fused)
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Second input: b (unchanged)
                match &inputs[1] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == b_sym)),
                    other => panic!("Expected Ref(b), got {:?}", other),
                }
                // Params: x (from g), y2 (from f — kept)
                assert_eq!(lam.params.len(), 2);
                assert_eq!(lam.params[0].0, x_sym);
                assert_eq!(lam.params[1].0, y2_sym);
            }
            other => panic!("Expected fused Map with 2 inputs, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: map-zip-map — map(f, zip(map(g, a), map(h, b)))
    // Both zip inputs are inline maps, both should be fused
    // -------------------------------------------------------------------------
    #[test]
    fn test_map_zip_map() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string()); // g's param
        let w_sym = symbols.alloc("w".to_string()); // h's param
        let y1_sym = symbols.alloc("y1".to_string()); // f's param 1
        let y2_sym = symbols.alloc("y2".to_string()); // f's param 2

        // g(x) = x : i32 → i32
        let g = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let map_g_a = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

        // h(w) = w : f32 → f32
        let h = mk_lambda1(
            w_sym,
            f32_ty(),
            mk_term(TermKind::Var(w_sym), f32_ty(), &mut term_ids),
            f32_ty(),
        );
        let b = mk_term(TermKind::Var(b_sym), array_ty(f32_ty()), &mut term_ids);
        let map_h_b = mk_map(h, b, array_ty(f32_ty()), &mut term_ids);

        // f(y1: i32, y2: f32) = y1
        let f = Lambda {
            params: vec![(y1_sym, i32_ty()), (y2_sym, f32_ty())],
            body: Box::new(mk_term(TermKind::Var(y1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };

        // map(f, zip(map(g, a), map(h, b))) — zip absorbed
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![
                    ArrayExpr::Ref(Box::new(map_g_a)),
                    ArrayExpr::Ref(Box::new(map_h_b)),
                ],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                // Both intermediates eliminated: inputs are [a, b]
                assert_eq!(inputs.len(), 2);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                match &inputs[1] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == b_sym)),
                    other => panic!("Expected Ref(b), got {:?}", other),
                }
                // Params: x (from g), w (from h)
                assert_eq!(lam.params.len(), 2);
                assert_eq!(lam.params[0].0, x_sym);
                assert_eq!(lam.params[1].0, w_sym);
            }
            other => panic!("Expected fused Map, got {:?}", other),
        }
    }

