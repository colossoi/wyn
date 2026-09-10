use super::*;
use crate::egir::graph_ops::{alloca, intern_storage_view};
use crate::egir::types::{EffectToken, FuncParam, Physical};
use crate::BindingRef;
use wyn_base::IdSource;

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
fn physical_identity_preserves_one_tuple_result_and_multiple_logical_results() {
    let mut graph = EGraph::<Physical>::new();
    let block = graph.skeleton.entry;
    let mut effects = IdSource::<EffectToken>::new();
    let first = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_type(), None);
    let second = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_type(), None);
    let pair_type = result_type(&[i32_type(), i32_type()]);
    let pair = graph.intern_pure(
        PureOp::Tuple(2),
        smallvec![first, second],
        pair_type.clone(),
        None,
    );

    for (types, arguments, count) in [
        (vec![], vec![], 0),
        (vec![pair_type], vec![pair], 1),
        (vec![i32_type(), i32_type()], vec![first, second], 2),
    ] {
        let results = emit_physical_call(
            &mut graph,
            block,
            &PhysicalCallables::new(),
            &Lambda::identity(types.clone()),
            arguments,
            None,
            &mut effects,
        )
        .unwrap();
        assert_eq!(results.len(), count);
        assert_eq!(
            results.iter().map(|result| result.ty().clone()).collect::<Vec<_>>(),
            types
        );
        if count == 1 {
            assert!(results[0].is_product());
            assert_eq!(results[0].values(), vec![first, second]);
        }
    }
    assert!(graph.calls().is_empty());
    assert!(graph.skeleton.blocks[block].side_effects.is_empty());
}

#[test]
fn physical_call_appends_value_view_and_place_captures_in_order() {
    let mut graph = EGraph::<Physical>::new();
    let block = graph.skeleton.entry;
    let mut effects = IdSource::<EffectToken>::new();
    let input = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_type(), None);
    let capture = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_type(), None);
    let view = intern_storage_view(&mut graph, BindingRef::new(0, 0), i32_type(), None);
    let place = alloca(&mut graph, i32_type(), &mut effects, None).append_to(&mut graph.skeleton, block);
    let captures = vec![
        OperandRef::Value(capture),
        graph.operand_ref(view),
        OperandRef::Place(place),
    ];
    assert!(matches!(captures[1], OperandRef::View(_)));
    let params = Parameters::from_ordered([
        FuncParam::value("input".into(), i32_type()),
        FuncParam::value("value_capture".into(), i32_type()),
        callable_parameter::<BindingRef, WynLanguage>(
            "view_capture".into(),
            graph.value(view).ty().clone(),
        ),
        FuncParam::place("place_capture".into(), graph.place(place).ty().clone()),
    ]);
    let mut body = EGraph::<Physical>::new();
    let parameters = params
        .iter_with_ids()
        .map(|(id, parameter)| body.add_parameter(id, parameter.representation()))
        .collect::<Vec<_>>();
    let result_abi = by_value_function_result::<WynLanguage>(i32_type());
    let result =
        egir::graph_ops::bind_by_value_result(&mut body, &result_abi, parameters[0].value().unwrap());
    body.skeleton.blocks[body.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    let region = FunctionId::from_index(0);
    let callee = Func::<Physical>::new(
        region,
        "captures".into(),
        Span::generated(),
        None,
        params,
        result_abi,
        CallEffects::Pure,
        body,
    );
    let callables = [(region, callee)].into_iter().collect();
    let lambda = Lambda::region(
        SegBody::new(region, captures.clone()),
        vec![i32_type()],
        vec![i32_type()],
    );

    let results = emit_physical_call(
        &mut graph,
        block,
        &callables,
        &lambda,
        vec![input],
        None,
        &mut effects,
    )
    .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].ty(), &i32_type());
    let calls = graph.calls().values().collect::<Vec<_>>();
    let [call] = calls.as_slice() else {
        panic!("expected one lambda call")
    };
    assert_eq!(call.callee(), region);
    assert_eq!(
        call.arguments().collect::<Vec<_>>(),
        [vec![OperandRef::Value(input)], captures].concat()
    );
}

#[test]
fn physical_adapter_rejects_invalid_identity_arity_and_missing_regions() {
    let mut graph = EGraph::<Physical>::new();
    let block = graph.skeleton.entry;
    let mut effects = IdSource::<EffectToken>::new();
    let input = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_type(), None);
    let mut invalid_identity = Lambda::identity(vec![i32_type()]);
    invalid_identity.result_types.clear();
    let missing_region = Lambda::region(
        SegBody::new(FunctionId::from_index(0), vec![]),
        vec![i32_type()],
        vec![i32_type()],
    );
    for (lambda, arguments, expected) in [
        (invalid_identity, vec![input], "identity lambda has signature"),
        (Lambda::identity(vec![i32_type()]), vec![], "requires 1 arguments"),
        (missing_region, vec![input], "callable boundary"),
    ] {
        let error = emit_physical_call(
            &mut graph,
            block,
            &PhysicalCallables::new(),
            &lambda,
            arguments,
            None,
            &mut effects,
        )
        .unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
    assert!(graph.calls().is_empty());
    assert!(graph.skeleton.blocks[block].side_effects.is_empty());
}

#[test]
fn filter_validation_rejects_identity_lambdas_with_mismatched_signatures() {
    use crate::egir::soac::filter;
    use crate::egir::types::SoacInputType;

    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let body = filter::Body {
        inputs: vec![SoacInputType::array(crate::types::view_array_of(
            &i32_type(),
            crate::types::no_buffer(),
        ))],
        map: Lambda::identity(vec![i32_type()]),
        predicate: Lambda::region(
            SegBody::new(FunctionId::from_index(0), vec![]),
            vec![i32_type()],
            vec![bool_type.clone()],
        ),
    };
    body.validate().unwrap();
    let mut invalid_map = body.clone();
    invalid_map.map.result_types = vec![bool_type.clone()];
    invalid_map.predicate.parameter_types = vec![bool_type];
    let error = invalid_map.validate().unwrap_err();
    assert!(error.contains("Filter map identity lambda"), "{error}");

    let mut invalid_predicate = body;
    invalid_predicate.predicate.body = LambdaBody::Identity;
    let error = invalid_predicate.validate().unwrap_err();
    assert!(error.contains("Filter predicate identity lambda"), "{error}");
}
