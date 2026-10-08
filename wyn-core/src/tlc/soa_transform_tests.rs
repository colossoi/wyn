use crate::tlc::soa::*;
use crate::types;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn size_ty(n: usize) -> Type<TypeName> {
    Type::Constructed(TypeName::Size(n), vec![])
}

fn composite_variant() -> Type<TypeName> {
    Type::Constructed(TypeName::ArrayVariantComposite, vec![])
}

fn array_ty(elem: Type<TypeName>, size: usize) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![elem, composite_variant(), size_ty(size), types::no_buffer()],
    )
}

fn tuple_ty(args: Vec<Type<TypeName>>) -> Type<TypeName> {
    Type::Constructed(TypeName::Tuple(args.len()), args)
}

#[test]
fn test_soa_type_scalar() {
    assert_eq!(soa_type(&i32_ty()), i32_ty());
    assert_eq!(soa_type(&f32_ty()), f32_ty());
}

#[test]
fn test_soa_type_plain_array() {
    let arr = array_ty(f32_ty(), 4);
    assert_eq!(soa_type(&arr), arr);
}

#[test]
fn test_soa_type_array_of_tuple() {
    // [4](i32, f32) → ([4]i32, [4]f32)
    let arr = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
    let expected = tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]);
    assert_eq!(soa_type(&arr), expected);
}

#[test]
fn test_soa_type_nested_array() {
    // [n][m](A,B) → ([n][m]A, [n][m]B)
    let inner = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 3);
    let outer = array_ty(inner, 5);
    let result = soa_type(&outer);

    // First soa_type on outer: elem is [3](i32,f32), which soa_type transforms to ([3]i32, [3]f32)
    // That's a tuple, so the outer array distributes:
    // ([5]([3]i32), [5]([3]f32)) — but wait, [5]([3]i32) is [5][3]i32 which is fine (no tuple elem)
    // Actually: soa_type([5][3](i32,f32)):
    //   elem = [3](i32,f32), soa_type → ([3]i32, [3]f32), which is Tuple
    //   So outer distributes: ([5]([3]i32), [5]([3]f32))
    //   But [5]([3]i32) has elem = ([3]i32) which is NOT a tuple, so it stays.
    // Actually ([3]i32) is Array, not Tuple. So the elem of the outer after soa
    // is ([3]i32, [3]f32) which IS a tuple. So we get:
    // (Array[([3]i32), composite, 5], Array[([3]f32), composite, 5])
    // = ([5][3]i32 via nesting... no, it's [5]([3]i32))
    // Hmm, [5](something) where something = Tuple. So distribute:
    // The soa_type of the inner element is ([3]i32, [3]f32).
    // Distributing array over this tuple: ([5]([3]i32), [5]([3]f32))
    // These are arrays whose elements are arrays (not tuples), so no further transformation.
    let expected = tuple_ty(vec![
        array_ty(array_ty(i32_ty(), 3), 5),
        array_ty(array_ty(f32_ty(), 3), 5),
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_soa_type_standalone_tuple() {
    // (A, [n](B,C)) → (A, ([n]B, [n]C))
    let inner = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
    let standalone = tuple_ty(vec![f32_ty(), inner]);
    let result = soa_type(&standalone);
    let expected = tuple_ty(vec![
        f32_ty(),
        tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]),
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_soa_type_arrow() {
    // ([4](i32,f32)) -> i32  →  ([4]i32, [4]f32) -> i32
    let param = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
    let arrow = Type::Constructed(TypeName::Arrow, vec![param, i32_ty()]);
    let result = soa_type(&arrow);
    let expected = Type::Constructed(
        TypeName::Arrow,
        vec![
            tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]),
            i32_ty(),
        ],
    );
    assert_eq!(result, expected);
}

#[test]
fn test_soa_type_is_idempotent() {
    let original = Type::Constructed(
        TypeName::Arrow,
        vec![
            array_ty(
                tuple_ty(vec![i32_ty(), array_ty(tuple_ty(vec![f32_ty(), i32_ty()]), 3)]),
                5,
            ),
            tuple_ty(vec![f32_ty(), array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4)]),
        ],
    );
    let normalized = soa_type(&original);
    assert_eq!(soa_type(&normalized), normalized);
}

fn value(ids: &mut TermIdSource, ty: Type<TypeName>, kind: TermKind) -> Term {
    Term::fresh(ids, ty, Span::generated(), kind)
}

fn call(ids: &mut TermIdSource, reference: VarRef, args: Vec<Term>, ty: Type<TypeName>) -> Term {
    let func_ty = tlc::curried_function_type(args.iter().map(|arg| &arg.ty), &ty);
    let func = value(ids, func_ty, TermKind::Var(reference));
    value(
        ids,
        ty,
        TermKind::App {
            func: Box::new(func),
            args,
        },
    )
}

fn builtin(id: crate::builtins::BuiltinId) -> VarRef {
    VarRef::Builtin { id, overload_idx: 0 }
}

/// Check the result immediately after the combined normalization pass.
/// In particular, projections and calls must already agree
/// with their operands, and duplicated uses must have distinct term IDs.
fn lower_checked(term: Term, ids: &mut TermIdSource, symbols: &mut SymbolTable) -> Term {
    fn check(term: &Term, ids: &mut std::collections::HashSet<TermId>) {
        assert!(ids.insert(term.id), "duplicate term ID: {:?}", term.id);
        assert_eq!(term.ty, soa_type(&term.ty));
        match &term.kind {
            TermKind::Tuple(fields) => {
                assert_eq!(
                    term.ty,
                    tuple_ty(fields.iter().map(|field| field.ty.clone()).collect())
                );
            }
            TermKind::TupleProj { tuple, idx } => {
                let Type::Constructed(TypeName::Tuple(_), fields) = &tuple.ty else {
                    panic!("projection of non-tuple")
                };
                assert_eq!(term.ty, fields[*idx]);
            }
            TermKind::Index { array, .. } => assert_eq!(Some(&term.ty), array.ty.elem_type()),
            TermKind::Let {
                name_ty, rhs, body, ..
            } => {
                assert!(!matches!(rhs.kind, TermKind::Let { .. }));
                assert_eq!(*name_ty, rhs.ty);
                assert_eq!(term.ty, body.ty);
            }
            TermKind::App { func, args } => {
                assert!(args.iter().all(|arg| !matches!(arg.kind, TermKind::Soac(_))));
                assert_eq!(
                    func.ty,
                    tlc::curried_function_type(args.iter().map(|arg| &arg.ty), &term.ty)
                );
            }
            TermKind::ArrayExpr(ArrayExpr::Literal(elements)) => {
                let elem_ty = term.ty.elem_type().expect("literal is an array");
                for element in elements {
                    assert_eq!(&element.ty, elem_ty);
                }
            }
            TermKind::ArrayExpr(ArrayExpr::Zip(_)) => panic!("standalone zip survived"),
            TermKind::Lambda(lam) => {
                assert_eq!(lam.ret_ty, lam.body.ty);
                assert_eq!(
                    term.ty,
                    tlc::curried_function_type(lam.params.iter().map(|(_, ty)| ty), &lam.ret_ty)
                );
            }
            TermKind::Loop {
                loop_var_ty,
                init,
                init_bindings,
                body,
                ..
            } => {
                assert_eq!(*loop_var_ty, init.ty);
                assert_eq!(term.ty, body.ty);
                for (_, ty, rhs) in init_bindings {
                    assert_eq!(*ty, rhs.ty);
                }
            }
            _ => {}
        }
        term.for_each_child(&mut |child| check(child, ids));
    }
    let result = SoaTransformer {
        term_ids: ids,
        symbols,
    }
    .rewrite_owned(term);
    check(&result, &mut std::collections::HashSet::new());
    result
}

fn count_calls(term: &Term, reference: VarRef) -> usize {
    let mut count = usize::from(
        matches!(&term.kind, TermKind::App { func, .. } if matches!(&func.kind, TermKind::Var(r) if *r == reference)),
    );
    term.for_each_child(&mut |child| count += count_calls(child, reference));
    count
}

#[test]
fn indexing_nested_local_layouts_constructs_typed_projections_and_evaluates_operands_once() {
    for elem_ty in [
        tuple_ty(vec![i32_ty(), tuple_ty(vec![f32_ty(), i32_ty()])]),
        array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 3),
    ] {
        let mut ids = TermIdSource::new();
        let mut symbols = SymbolTable::new();
        let producer = VarRef::Symbol(symbols.alloc("produce".into()));
        let offset = VarRef::Symbol(symbols.alloc("offset".into()));
        let array = call(&mut ids, producer, vec![], array_ty(elem_ty.clone(), 2));
        let index = call(&mut ids, offset, vec![], i32_ty());
        let term = value(
            &mut ids,
            elem_ty.clone(),
            TermKind::Index {
                array: Box::new(array),
                index: Box::new(index),
            },
        );
        let result = lower_checked(term, &mut ids, &mut symbols);
        assert_eq!(result.ty, soa_type(&elem_ty));
        assert_eq!(count_calls(&result, producer), 1);
        assert_eq!(count_calls(&result, offset), 1);
    }
}

#[test]
fn local_updates_and_lengths_follow_the_complete_target_layout() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let elem_ty = tuple_ty(vec![tuple_ty(vec![i32_ty(), f32_ty()]), i32_ty()]);
    let array_ty = array_ty(elem_ty.clone(), 2);
    let source = VarRef::Symbol(symbols.alloc("source".into()));
    let update = VarRef::Symbol(symbols.alloc("update".into()));
    let offset = VarRef::Symbol(symbols.alloc("offset".into()));
    let array = call(&mut ids, source, vec![], array_ty.clone());
    let index = call(&mut ids, offset, vec![], i32_ty());
    let element = call(&mut ids, update, vec![], elem_ty);
    let update_term = call(
        &mut ids,
        builtin(catalog().known().array_with),
        vec![array, index, element],
        array_ty,
    );
    let term = call(
        &mut ids,
        builtin(catalog().known().length),
        vec![update_term],
        i32_ty(),
    );
    let result = lower_checked(term, &mut ids, &mut symbols);
    assert_eq!(count_calls(&result, builtin(catalog().known().array_with)), 3);
    assert_eq!(count_calls(&result, builtin(catalog().known().length)), 1);
    for reference in [source, update, offset] {
        assert_eq!(count_calls(&result, reference), 1);
    }
}

#[test]
fn literal_distribution_handles_empty_and_nested_elements_without_repeating_calls() {
    for len in [0, 2] {
        let mut ids = TermIdSource::new();
        let mut symbols = SymbolTable::new();
        let elem_ty = tuple_ty(vec![i32_ty(), tuple_ty(vec![f32_ty(), i32_ty()])]);
        let producer = VarRef::Symbol(symbols.alloc("element".into()));
        let elements = (0..len).map(|_| call(&mut ids, producer, vec![], elem_ty.clone())).collect();
        let ty = array_ty(elem_ty, len);
        let term = value(
            &mut ids,
            ty.clone(),
            TermKind::ArrayExpr(ArrayExpr::Literal(elements)),
        );
        let result = lower_checked(term, &mut ids, &mut symbols);
        assert_eq!(result.ty, soa_type(&ty));
        assert_eq!(count_calls(&result, producer), len);
    }
}

#[test]
fn standalone_zip_components_use_their_own_array_types() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let a = value(&mut ids, i32_ty(), TermKind::IntLit("1".into()));
    let b = value(&mut ids, f32_ty(), TermKind::FloatLit(2.0));
    let ty = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 1);
    let term = value(
        &mut ids,
        ty.clone(),
        TermKind::ArrayExpr(ArrayExpr::Zip(vec![
            ArrayExpr::Literal(vec![a]),
            ArrayExpr::Literal(vec![b]),
        ])),
    );
    let result = lower_checked(term, &mut ids, &mut symbols);
    assert_eq!(result.ty, soa_type(&ty));
}

#[test]
fn storage_backed_tuple_arrays_keep_direct_indexing_and_updates() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let elem_ty = tuple_ty(vec![i32_ty(), f32_ty()]);
    let storage_ty = types::view_array_with_size(
        &elem_ty,
        size_ty(4),
        types::buffer_tag(crate::BindingRef::new(0, 0)),
    );
    let source = VarRef::Symbol(symbols.alloc("storage".into()));
    let array = value(&mut ids, storage_ty.clone(), TermKind::Var(source));
    let index = value(&mut ids, i32_ty(), TermKind::IntLit("0".into()));
    let term = value(
        &mut ids,
        elem_ty.clone(),
        TermKind::Index {
            array: Box::new(array),
            index: Box::new(index),
        },
    );
    let result = lower_checked(term, &mut ids, &mut symbols);
    let TermKind::Index { array, .. } = result.kind else {
        panic!("storage indexing was distributed")
    };
    assert_eq!(array.ty, storage_ty);
    let array = value(&mut ids, storage_ty.clone(), TermKind::Var(source));
    let index = value(&mut ids, i32_ty(), TermKind::IntLit("0".into()));
    let element = call(
        &mut ids,
        VarRef::Symbol(symbols.alloc("element".into())),
        vec![],
        elem_ty,
    );
    let term = call(
        &mut ids,
        builtin(catalog().known().array_with),
        vec![array, index, element],
        storage_ty.clone(),
    );
    let result = lower_checked(term, &mut ids, &mut symbols);
    assert_eq!(result.ty, storage_ty);
    assert!(matches!(result.kind, TermKind::App { .. }));
    assert_eq!(count_calls(&result, builtin(catalog().known().array_with)), 1);
}

#[test]
fn uninitialized_arrays_keep_builtin_identity_in_every_component() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let ty = array_ty(tuple_ty(vec![i32_ty(), tuple_ty(vec![f32_ty(), i32_ty()])]), 4);
    for as_call in [false, true] {
        let term = if as_call {
            call(&mut ids, builtin(catalog().known().uninit), vec![], ty.clone())
        } else {
            value(
                &mut ids,
                ty.clone(),
                TermKind::Var(builtin(catalog().known().uninit)),
            )
        };
        let result = lower_checked(term, &mut ids, &mut symbols);
        assert_eq!(result.ty, soa_type(&ty));
        fn leaves(term: &Term) -> usize {
            match &term.kind {
                TermKind::Tuple(fields) => fields.iter().map(leaves).sum(),
                TermKind::Var(reference) if *reference == builtin(catalog().known().uninit) => 1,
                _ => panic!("uninit must stay a typed builtin"),
            }
        }
        assert_eq!(leaves(&result), 3);
    }
}

#[test]
fn lambda_and_loop_binders_are_lowered_with_their_uses() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let param = symbols.alloc("param".into());
    let acc = symbols.alloc("acc".into());
    let first = symbols.alloc("first".into());
    let elem_ty = tuple_ty(vec![i32_ty(), f32_ty()]);
    let ty = array_ty(elem_ty.clone(), 2);
    let init = value(&mut ids, ty.clone(), TermKind::Var(VarRef::Symbol(param)));
    let body = value(&mut ids, ty.clone(), TermKind::Var(VarRef::Symbol(acc)));
    let array = value(&mut ids, ty.clone(), TermKind::Var(VarRef::Symbol(acc)));
    let index = value(&mut ids, i32_ty(), TermKind::IntLit("0".into()));
    let extract = value(
        &mut ids,
        elem_ty.clone(),
        TermKind::Index {
            array: Box::new(array),
            index: Box::new(index),
        },
    );
    let cond = value(
        &mut ids,
        Type::Constructed(TypeName::Bool, vec![]),
        TermKind::BoolLit(false),
    );
    let loop_body = value(
        &mut ids,
        ty.clone(),
        TermKind::Loop {
            loop_var: acc,
            loop_var_ty: ty.clone(),
            init: Box::new(init),
            init_bindings: vec![(first, elem_ty, extract)],
            kind: tlc::LoopKind::While { cond: Box::new(cond) },
            body: Box::new(body),
        },
    );
    let lambda = tlc::rebuild_nested_lam(&[(param, ty.clone())], loop_body, Span::generated(), &mut ids);
    let result = lower_checked(lambda, &mut ids, &mut symbols);
    let TermKind::Lambda(lam) = result.kind else {
        panic!("lambda")
    };
    assert_eq!(lam.params[0].1, soa_type(&ty));
    assert!(matches!(lam.body.kind, TermKind::Loop { .. }));
}

#[test]
fn map_input_and_callback_metadata_follow_nested_array_lowering() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let input = symbols.alloc("input".into());
    let param = symbols.alloc("row".into());
    let row_ty = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 3);
    let input_ty = array_ty(row_ty.clone(), 2);
    let row = value(&mut ids, row_ty.clone(), TermKind::Var(VarRef::Symbol(param)));
    let body = call(&mut ids, builtin(catalog().known().length), vec![row], i32_ty());
    let term = value(
        &mut ids,
        array_ty(i32_ty(), 2),
        TermKind::Soac(tlc::SoacOp::Map {
            lam: tlc::SoacBody {
                lam: tlc::Lambda {
                    params: vec![(param, row_ty.clone())],
                    body: Box::new(body),
                    ret_ty: i32_ty(),
                },
                data: (),
            },
            inputs: vec![ArrayExpr::Var(VarRef::Symbol(input), input_ty.clone())],
            destination: tlc::SoacOwnership::Fresh,
        }),
    );
    let result = lower_checked(term.clone(), &mut ids, &mut symbols);
    let TermKind::Soac(tlc::SoacOp::Map { lam, inputs, .. }) = result.kind else {
        panic!("map")
    };
    assert_eq!(lam.lam.params[0].1, soa_type(&row_ty));
    assert_eq!(lam.lam.ret_ty, lam.lam.body.ty);
    assert_eq!(inputs[0].array_type(), soa_type(&input_ty));

    // A consumer of the map must stay inside its lambda, with the original
    // let flattened after hoisting and all generated bindings correctly typed.
    let consume = VarRef::Symbol(symbols.alloc("consume".into()));
    let rhs = call(&mut ids, consume, vec![term], i32_ty());
    let name = symbols.alloc("result".into());
    let body = value(&mut ids, i32_ty(), TermKind::Var(VarRef::Symbol(name)));
    let bound = value(
        &mut ids,
        i32_ty(),
        TermKind::Let {
            name,
            name_ty: i32_ty(),
            rhs: Box::new(rhs),
            body: Box::new(body),
        },
    );
    let lambda = tlc::rebuild_nested_lam(&[(input, input_ty)], bound, Span::generated(), &mut ids);
    let result = lower_checked(lambda, &mut ids, &mut symbols);
    let TermKind::Lambda(lam) = result.kind else {
        panic!("lambda")
    };
    let TermKind::Let { rhs, body, .. } = lam.body.kind else {
        panic!("map binding")
    };
    assert!(matches!(rhs.kind, TermKind::Soac(_)));
    let TermKind::Let { rhs, .. } = body.kind else {
        panic!("consumer binding")
    };
    assert!(matches!(rhs.kind, TermKind::App { .. }));
    assert_eq!(count_calls(&rhs, consume), 1);
}
