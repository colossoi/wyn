use super::*;
use crate::ast::TypeName;
use crate::egir::types::{EffectOp, EffectToken, OperandRef, SkeletonTerminator};
use crate::op;
use crate::ssa::types::ConstantValue;
use polytype::Type;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn tuple_ty(fields: Vec<Type<TypeName>>) -> Type<TypeName> {
    Type::Constructed(TypeName::Tuple(fields.len()), fields)
}

fn fixed_array_ty(element: Type<TypeName>, length: usize) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            element,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(length), vec![]),
            types::no_buffer(),
        ],
    )
}

fn alloca_place(graph: &mut EGraph<Physical>, ty: Type<TypeName>) -> PlaceId {
    graph.add_alloca_place(
        PlaceType {
            pointee: ty,
            region: PlaceRegion::Function,
            access: PlaceAccess::ReadWrite,
        },
        None,
    )
}

#[test]
fn physical_result_rebinding_replaces_leaves_and_folds_exposed_projections() {
    let mut graph = EGraph::<Physical>::new();
    let scalar = u32_ty();
    let pair = tuple_ty(vec![scalar.clone(), scalar.clone()]);
    let old_root = graph.alloc_side_effect_result(pair.clone());
    let old = bind_by_value_result(
        &mut graph,
        &crate::egir::types::by_value_function_result::<WynLanguage>(pair.clone()),
        old_root,
    );
    let left = graph.intern_constant(ConstantValue::U32(11), scalar.clone());
    let right = graph.intern_constant(ConstantValue::U32(29), scalar.clone());
    let replacement_root = graph.intern_pure(PureOp::Tuple(2), smallvec![left, right], pair.clone(), None);
    let replacement = bind_by_value_result(
        &mut graph,
        &crate::egir::types::by_value_function_result::<WynLanguage>(pair),
        replacement_root,
    );
    let old_values = old.values();

    rebind_physical_result(&mut graph, &old, &replacement).unwrap();

    assert_eq!(graph.canonical_value(old_values[0]), left);
    assert_eq!(graph.canonical_value(old_values[1]), right);
}

#[test]
fn indexed_result_emission_supports_array_of_products_and_product_of_arrays() {
    let mut graph = EGraph::<Physical>::new();
    let block = graph.skeleton.entry;
    let scalar = u32_ty();
    let pair = tuple_ty(vec![scalar.clone(), scalar.clone()]);
    let left = graph.intern_constant(ConstantValue::U32(3), scalar.clone());
    let right = graph.intern_constant(ConstantValue::U32(5), scalar.clone());
    let produced = ResultBinding::product(
        pair.clone(),
        [
            ResultBinding::destination(scalar.clone(), ResultDestination::ReturnValue(left)),
            ResultBinding::destination(scalar.clone(), ResultDestination::ReturnValue(right)),
        ],
    );
    let index = graph.intern_constant(
        ConstantValue::I32(1),
        Type::Constructed(TypeName::Int(32), vec![]),
    );
    let mut effect_ids = IdSource::new();

    let aos_ty = fixed_array_ty(pair.clone(), 4);
    let aos_place = alloca_place(&mut graph, aos_ty.clone());
    let aos = ResultBinding::destination(
        aos_ty,
        ResultDestination::Place(PlaceDestination::Fixed(aos_place)),
    );
    emit_result_to_indexed_destination(&mut graph, block, &produced, &aos, index, &mut effect_ids).unwrap();

    let left_array = fixed_array_ty(scalar.clone(), 4);
    let right_array = fixed_array_ty(scalar.clone(), 4);
    let left_place = alloca_place(&mut graph, left_array.clone());
    let right_place = alloca_place(&mut graph, right_array.clone());
    let poa = ResultBinding::product(
        tuple_ty(vec![left_array.clone(), right_array.clone()]),
        [
            ResultBinding::destination(
                left_array,
                ResultDestination::Place(PlaceDestination::Fixed(left_place)),
            ),
            ResultBinding::destination(
                right_array,
                ResultDestination::Place(PlaceDestination::Bounded {
                    storage: right_place,
                    length: right_place,
                }),
            ),
        ],
    );
    emit_result_to_indexed_destination(&mut graph, block, &produced, &poa, index, &mut effect_ids).unwrap();

    assert_eq!(graph.skeleton.blocks[block].side_effects.len(), 4);
    assert!(graph.skeleton.blocks[block]
        .side_effects
        .iter()
        .all(|effect| matches!(effect.kind(), SideEffectKind::Effect(EffectOp::Store { .. }))));
}

#[test]
fn value_producer_closure_crosses_effects_block_params_and_loop_cycles() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let ty = u32_ty();
    let source = graph.intern_constant(ConstantValue::U32(0), ty.clone());
    let produced = graph.alloc_side_effect_result(ty.clone());
    let produced_binding = graph.value_result(produced);
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Op {
            tag: PureOp::Materialize,
        }),
        operands: smallvec![OperandRef::Value(source)],
        result: Some(produced_binding),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
    let entry_args = graph.admit_flow_values([produced]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: entry_args,
    };

    let current = graph.add_block_param(header, ty.clone());
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let next = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![current, one],
        ty.clone(),
        None,
    );
    let cond = graph.intern_constant(
        ConstantValue::Bool(true),
        Type::Constructed(TypeName::Bool, vec![]),
    );

    let merged = graph.add_block_param(exit, ty.clone());
    let next_args = graph.admit_flow_values([next]);
    let current_args = graph.admit_flow_values([current]);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: header,
        then_args: next_args,
        else_target: exit,
        else_args: current_args,
    };
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(Some(graph.value_result(merged)));
    let tail = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![merged, one],
        ty,
        None,
    );

    let closure = value_producer_closure(&graph, [tail]);

    assert_eq!(
        closure.effects,
        HashSet::from([SideEffectSite {
            block: entry,
            index: 0,
        }])
    );
    for expected in [tail, merged, current, next, one, cond, produced, source] {
        assert!(
            closure.nodes.contains(&expected),
            "producer closure omitted {expected:?}"
        );
    }

    let uses = ValueUseIndex::build(&graph);
    let pure = uses.pure_observers(source);
    assert_eq!(
        pure.effect_sites().collect::<HashSet<_>>(),
        HashSet::from([SideEffectSite {
            block: entry,
            index: 0,
        }])
    );
    assert!(pure.terminator_blocks().next().is_none());

    let flowing = uses.value_observers(source);
    assert_eq!(
        flowing.effect_sites().collect::<HashSet<_>>(),
        HashSet::from([SideEffectSite {
            block: entry,
            index: 0,
        }])
    );
    assert_eq!(
        flowing.terminator_blocks().collect::<HashSet<_>>(),
        HashSet::from([entry, header, exit])
    );
    assert!(uses.pure_reaches(current, next));
    assert!(!uses.pure_reaches(source, produced));
}
