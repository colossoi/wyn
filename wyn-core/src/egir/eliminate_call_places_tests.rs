use super::*;
use crate::ast::{Span, TypeName};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, FuncParam, OperandRef, Parameters,
    PlaceAccess, PlaceRegion, PlaceType, SkeletonTerminator, WynLanguage,
};
use crate::ssa::types::ConstantValue;
use crate::{BindingRef, FunctionId};
use polytype::Type;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

#[test]
fn leaves_effectful_value_only_calls_for_ssa_elaboration() {
    let ty = u32_ty();
    let region = FunctionId::from_index(0);
    let params: Parameters<BindingRef, Type<TypeName>> = [callable_parameter::<BindingRef, WynLanguage>(
        "value".into(),
        ty.clone(),
    )]
    .into_iter()
    .collect();
    let parameter = params.ids().next().unwrap();
    let mut callee_graph = EGraph::<Physical>::new();
    let value = callee_graph.add_test_value_parameter(parameter, ty.clone());
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(callee_graph.value_result(value)));
    let callee = Func::<Physical>::new(
        region,
        "effectful_value_only".into(),
        Span::dummy(),
        None,
        params.clone(),
        by_value_function_result::<WynLanguage>(ty.clone()),
        CallEffects::General,
        callee_graph,
    );
    let callees = [(region, callee)].into_iter().collect();

    let mut caller = EGraph::<Physical>::new();
    let caller_params: Parameters<BindingRef, Type<TypeName>> = [callable_parameter::<
        BindingRef,
        WynLanguage,
    >("value".into(), ty.clone())]
    .into_iter()
    .collect();
    let argument = caller.add_test_value_parameter(caller_params.ids().next().unwrap(), ty.clone());
    let (_, result) = caller
        .emit_call(
            caller.skeleton.entry,
            region,
            &params,
            &by_value_function_result::<WynLanguage>(ty),
            [OperandRef::Value(argument)],
            CallEffects::General,
            None,
            None,
        )
        .unwrap();
    caller.skeleton.blocks[caller.skeleton.entry].term = SkeletonTerminator::Return(Some(result));

    let stats = eliminate_body(&mut caller, &callees, &mut IdSource::new()).unwrap();

    assert_eq!(stats.calls, 0);
    assert_eq!(caller.side_effect_index().calls().count(), 1);

    let boundaries = [(
        region,
        (
            params,
            by_value_function_result::<WynLanguage>(u32_ty()),
            CallEffects::General,
        ),
    )]
    .into_iter()
    .collect();
    verify_graph_calls(&caller, "caller", &boundaries).unwrap();
}

#[test]
fn eliminates_internal_calls_with_place_arguments() {
    let ty = u32_ty();
    let region = FunctionId::from_index(0);
    let place_ty = PlaceType {
        pointee: ty.clone(),
        region: PlaceRegion::Parametric,
        access: PlaceAccess::ReadOnly,
    };
    let params = Parameters::from_ordered([FuncParam::place("source".into(), place_ty.clone())]);
    let parameter = params.ids().next().unwrap();
    let mut callee_graph = EGraph::<Physical>::new();
    callee_graph.add_parameter(parameter, params.get(parameter).unwrap().representation());
    let value = callee_graph.intern_constant(ConstantValue::U32(7), ty.clone());
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(callee_graph.value_result(value)));
    let callee = Func::<Physical>::new(
        region,
        "place_reader".into(),
        Span::dummy(),
        None,
        params.clone(),
        by_value_function_result::<WynLanguage>(ty.clone()),
        CallEffects::General,
        callee_graph,
    );
    let callees = [(region, callee)].into_iter().collect();

    let mut caller = EGraph::<Physical>::new();
    let place = caller.add_parameter(parameter, &crate::egir::types::OperandType::Place(place_ty));
    let (_, result) = caller
        .emit_call(
            caller.skeleton.entry,
            region,
            &params,
            &by_value_function_result::<WynLanguage>(ty),
            [place],
            CallEffects::General,
            None,
            None,
        )
        .unwrap();
    caller.skeleton.blocks[caller.skeleton.entry].term = SkeletonTerminator::Return(Some(result));

    let stats = eliminate_body(&mut caller, &callees, &mut IdSource::new()).unwrap();

    assert_eq!(stats.calls, 1);
    assert_eq!(caller.side_effect_index().calls().count(), 0);
}

#[test]
fn verifier_rejects_calls_without_a_stable_callee_boundary() {
    let ty = u32_ty();
    let region = FunctionId::from_index(0);
    let params: Parameters<BindingRef, Type<TypeName>> = [callable_parameter::<BindingRef, WynLanguage>(
        "value".into(),
        ty.clone(),
    )]
    .into_iter()
    .collect();
    let mut graph = EGraph::<Physical>::new();
    let argument = graph.add_test_value_parameter(params.ids().next().unwrap(), ty.clone());
    graph
        .emit_call(
            graph.skeleton.entry,
            region,
            &params,
            &by_value_function_result::<WynLanguage>(ty),
            [OperandRef::Value(argument)],
            CallEffects::Pure,
            None,
            None,
        )
        .unwrap();

    let error = verify_graph_calls(&graph, "caller", &LookupMap::new()).unwrap_err();

    assert!(error.contains("without a stable callable ABI"), "{error}");
}
