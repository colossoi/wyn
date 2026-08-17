use crate::ast::TypeName;
use crate::LookupMap;
use polytype::Type;
use smallvec::smallvec;

use super::super::graph_ops::{detached_alloca, detached_store, emit_result_to_place};
use super::super::ir::PlaceOp;
use super::super::program::{PhysicalEGraph, PhysicalFunc, PhysicalResourceRef};
use super::super::types::{
    destination_passing_function_result, CallEffects, CallSiteId, EffectOp, EffectToken, FuncParam,
    Language, OperandRef, OperandType, ParameterId, PlaceAccess, PlaceDestination, PlaceRegion, PlaceType,
    PureOp, ResultDestination, SideEffectKind, SkeletonTerminator, ValueId, ValueKind, WynLanguage,
};

#[derive(Clone)]
struct CallableBoundary {
    params: Vec<FuncParam<super::super::program::PhysicalResourceRef, Type<TypeName>>>,
    result: super::super::types::FunctionResult<Type<TypeName>>,
    effects: CallEffects,
}

pub(super) fn resolve(
    mut program: super::super::parallelize::Planned,
) -> Result<super::super::parallelize::Planned, String> {
    for function in &mut program.functions {
        resolve_place_parameters(&mut function.params, &mut function.graph)?;
    }
    for entry in &mut program.entry_points {
        resolve_place_parameters(&mut entry.params, &mut entry.graph)?;
    }
    for function in &mut program.functions {
        resolve_function(function, &mut program.global_context.effect_ids)?;
    }
    let boundaries = program
        .functions
        .iter()
        .map(|function| {
            (
                function.region,
                CallableBoundary {
                    params: function.params().to_vec(),
                    result: function.result().clone(),
                    effects: function.effects(),
                },
            )
        })
        .collect::<LookupMap<_, _>>();
    for function in &mut program.functions {
        resolve_calls(
            &mut function.graph,
            &boundaries,
            &mut program.global_context.effect_ids,
        )?;
    }
    for entry in &mut program.entry_points {
        resolve_calls(
            &mut entry.graph,
            &boundaries,
            &mut program.global_context.effect_ids,
        )?;
    }
    for constant in &mut program.constants {
        resolve_calls(
            &mut constant.graph,
            &boundaries,
            &mut program.global_context.effect_ids,
        )?;
    }
    Ok(program)
}

fn resolve_place_parameters(
    params: &mut [FuncParam<PhysicalResourceRef, Type<TypeName>>],
    graph: &mut PhysicalEGraph,
) -> Result<(), String> {
    for (index, parameter) in params.iter_mut().enumerate() {
        let OperandType::Value(ty) = parameter.representation() else {
            continue;
        };
        if !WynLanguage::is_materialized_aggregate(ty) {
            continue;
        }
        let ty = ty.clone();
        let parameter_id = ParameterId::new(index);
        let source = graph
            .nodes
            .iter()
            .find_map(|(value, definition)| {
                matches!(
                    definition.kind(),
                    ValueKind::FuncParam { parameter } if *parameter == parameter_id
                )
                .then_some(value)
            })
            .ok_or_else(|| format!("physical parameter {index} has no graph binding"))?;
        let place_ty = PlaceType {
            pointee: ty.clone(),
            region: PlaceRegion::Parametric,
            access: PlaceAccess::ReadOnly,
        };
        let place = graph.add_place_parameter(parameter_id, place_ty.clone());
        let view_ty = crate::types::view_array_of(&ty, crate::types::no_buffer());
        let view = graph.add_place_view(place, view_ty, graph.nodes[source].span()).value();
        graph.replace_value_references(source, view);
        if !graph.remove_func_param(source) {
            return Err(format!("physical parameter {index} is not a value parameter"));
        }
        *parameter.representation_mut() = OperandType::Place(place_ty);
        super::super::graph_ops::normalize_place_backed_value_consumers(graph, view);
    }
    Ok(())
}

fn resolve_function(
    function: &mut PhysicalFunc,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let mut params = std::mem::take(&mut function.params);
    let result =
        destination_passing_function_result::<_, WynLanguage>(function.result().ty().clone(), &mut params);
    let destination_parameters = result
        .destination_leaves()
        .into_iter()
        .filter_map(|leaf| match leaf.single_destination() {
            Some((_, ResultDestination::Place(PlaceDestination::Fixed(parameter)))) => Some(*parameter),
            Some((_, ResultDestination::Place(PlaceDestination::Bounded { .. }))) => {
                unreachable!("bounded callable results require an explicit length route")
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if destination_parameters.is_empty() {
        function.params = params;
        function.result = result;
        return Ok(());
    }

    let mut parameter_places = LookupMap::new();
    for parameter in destination_parameters {
        let representation = params[parameter.index()].representation();
        let place = function
            .graph
            .add_parameter(parameter, representation)
            .place()
            .expect("destination parameter has a place representation");
        parameter_places.insert(parameter, place);
    }

    let returning_blocks = function
        .graph
        .skeleton
        .blocks
        .iter()
        .filter_map(|(block, contents)| {
            matches!(contents.term, SkeletonTerminator::Return(Some(_))).then_some(block)
        })
        .collect::<Vec<_>>();
    let mut redirected_allocas = Vec::new();
    for block in returning_blocks {
        let SkeletonTerminator::Return(Some(binding)) = function.graph.skeleton.blocks[block].term.clone()
        else {
            unreachable!()
        };
        let old_leaves = binding.destination_leaves();
        let new_leaves = result.destination_leaves();
        if old_leaves.len() != new_leaves.len() {
            return Err(format!(
                "function `{}` returns {} logical leaves for an ABI with {} leaves",
                function.name,
                old_leaves.len(),
                new_leaves.len()
            ));
        }
        let mut return_values = LookupMap::new();
        let mut tail = block;
        for (source, route) in old_leaves.iter().zip(&new_leaves) {
            if source.ty() != route.ty() {
                return Err(format!(
                    "function `{}` changes result leaf type from {:?} to {:?}",
                    function.name,
                    source.ty(),
                    route.ty()
                ));
            }
            match route.single_destination() {
                Some((_, ResultDestination::ReturnValue(slot))) => {
                    let value = source.single_value().ok_or_else(|| {
                        format!(
                            "function `{}` routes a scalar result through a place",
                            function.name
                        )
                    })?;
                    return_values.insert(*slot, value);
                }
                Some((_, ResultDestination::Place(PlaceDestination::Fixed(parameter)))) => {
                    let destination = parameter_places[parameter];
                    let source_place = source.places().first().copied();
                    if let Some(source_place) = source_place.filter(|source| {
                        matches!(function.graph.place(*source).op(), PlaceOp::AllocaResult)
                    }) {
                        function.graph.replace_place_references(source_place, destination);
                        redirected_allocas.push(source_place);
                    } else {
                        tail = emit_result_to_place(
                            &mut function.graph,
                            tail,
                            source,
                            destination,
                            effect_ids,
                            Some(function.span),
                        )?;
                    }
                }
                Some((_, ResultDestination::Place(PlaceDestination::Bounded { .. }))) => {
                    unreachable!("bounded callable results require an explicit length route")
                }
                None => unreachable!("a result leaf has one destination"),
            }
        }
        let physical_return = result.bind(
            |slot, _| return_values[&slot],
            |parameter| parameter_places[&parameter],
        );
        function.graph.skeleton.blocks[tail].term = SkeletonTerminator::Return(Some(physical_return));
    }
    for (_, block) in &mut function.graph.skeleton.blocks {
        block.side_effects.retain(|effect| {
            !matches!(
                effect.kind(),
                SideEffectKind::Effect(EffectOp::Alloca { result })
                    if redirected_allocas.contains(result)
            )
        });
    }
    function.params = params;
    function.result = result;
    function.effects = match function.effects {
        CallEffects::Pure | CallEffects::DestinationWrite => CallEffects::DestinationWrite,
        CallEffects::General => CallEffects::General,
    };
    Ok(())
}

fn resolve_calls(
    graph: &mut PhysicalEGraph,
    boundaries: &LookupMap<crate::FunctionId, CallableBoundary>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let calls = graph.calls().keys().collect::<Vec<_>>();
    for site in calls {
        let callee = graph.call(site).callee();
        let Some(boundary) = boundaries.get(&callee) else {
            continue;
        };
        if call_arguments_match(graph, graph.call(site).arguments(), &boundary.params)
            && graph.call(site).result().destination_count() == boundary.result.destination_count()
            && graph.call(site).result().values().len()
                == boundary
                    .result
                    .destination_leaves()
                    .iter()
                    .filter(|leaf| {
                        matches!(
                            leaf.single_destination(),
                            Some((_, ResultDestination::ReturnValue(_)))
                        )
                    })
                    .count()
        {
            continue;
        }
        resolve_call(graph, site, boundary, effect_ids)?;
    }
    Ok(())
}

fn call_arguments_match(
    graph: &PhysicalEGraph,
    arguments: &[OperandRef],
    params: &[FuncParam<PhysicalResourceRef, Type<TypeName>>],
) -> bool {
    arguments.len() == params.len()
        && arguments.iter().zip(params).all(|(argument, parameter)| {
            match (argument, parameter.representation()) {
                (OperandRef::Value(value), OperandType::Value(ty)) => {
                    graph.nodes.get(*value).is_some_and(|value| value.ty() == ty)
                }
                (OperandRef::View(view), OperandType::View(ty)) => {
                    graph.nodes.get(view.value()).is_some_and(|value| value.ty() == &ty.array)
                }
                (OperandRef::Place(place), OperandType::Place(ty)) => {
                    graph.places().get(*place).is_some_and(|place| {
                        place.ty().pointee == ty.pointee && ty.access.accepts(place.ty().access)
                    })
                }
                _ => false,
            }
        })
}

fn resolve_call(
    graph: &mut PhysicalEGraph,
    site: CallSiteId,
    boundary: &CallableBoundary,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let old = graph.call(site).clone();
    let old_leaves = old.result().destination_leaves();
    let new_leaves = boundary.result.destination_leaves();
    if old_leaves.len() != new_leaves.len() {
        return Err(format!(
            "call to {:?} binds {} logical leaves for an ABI with {} leaves",
            old.callee(),
            old_leaves.len(),
            new_leaves.len()
        ));
    }
    let anchor = graph
        .side_effect_index()
        .call_site(site)
        .ok_or_else(|| format!("call {site:?} has no explicit skeleton site"))?;
    let mut allocations = Vec::new();
    let mut replacements = Vec::new();
    let mut return_values = LookupMap::new();
    let mut destination_arguments = LookupMap::new();
    let mut arguments = Vec::with_capacity(boundary.params.len());
    for (index, argument) in old.arguments().iter().copied().enumerate() {
        let parameter = boundary.params.get(index).ok_or_else(|| {
            format!(
                "call to {:?} supplies more arguments than its physical boundary",
                old.callee()
            )
        })?;
        let argument = match parameter.representation() {
            OperandType::Place(place_ty) => {
                let place =
                    if let Some(place) = addressable_argument_place(graph, argument, &place_ty.pointee) {
                        place
                    } else {
                        let value = argument.value().ok_or_else(|| {
                            format!(
                                "call to {:?} argument {index} cannot be materialized as {:?}",
                                old.callee(),
                                place_ty.pointee
                            )
                        })?;
                        let value = graph.canonical_value(value);
                        if graph.nodes[value].ty() != &place_ty.pointee {
                            return Err(format!(
                                "call to {:?} argument {index} has type {:?}, expected place pointee {:?}",
                                old.callee(),
                                graph.nodes[value].ty(),
                                place_ty.pointee
                            ));
                        }
                        let span = graph.nodes[value].span();
                        let (place, allocation) =
                            detached_alloca(graph, place_ty.pointee.clone(), effect_ids, span);
                        allocations.push(allocation);
                        allocations.push(detached_store(place, value, effect_ids, span));
                        place
                    };
                OperandRef::Place(place)
            }
            OperandType::Value(_) | OperandType::View(_) => graph.canonical_operand(argument),
        };
        arguments.push(argument);
    }
    for (source, route) in old_leaves.iter().zip(&new_leaves) {
        match route.single_destination() {
            Some((_, ResultDestination::ReturnValue(slot))) => {
                let value = source.single_value().ok_or_else(|| {
                    format!(
                        "call to {:?} routes a scalar result through a place",
                        old.callee()
                    )
                })?;
                return_values.insert(*slot, value);
            }
            Some((ty, ResultDestination::Place(PlaceDestination::Fixed(parameter)))) => {
                let destination = match source.single_destination() {
                    Some((_, ResultDestination::Place(PlaceDestination::Fixed(place)))) => *place,
                    Some((_, ResultDestination::Place(PlaceDestination::Bounded { storage, .. }))) => {
                        *storage
                    }
                    Some((_, ResultDestination::ReturnValue(value))) => {
                        let value = graph.canonical_value(*value);
                        match graph.nodes[value].kind() {
                            ValueKind::PlaceView { place } => *place,
                            _ => {
                                let (place, effect) =
                                    detached_alloca(graph, ty.clone(), effect_ids, graph.nodes[value].span);
                                allocations.push(effect);
                                let view_ty = crate::types::view_array_of(ty, crate::types::no_buffer());
                                let view =
                                    graph.add_place_view(place, view_ty, graph.nodes[value].span).value();
                                replacements.push((value, view));
                                place
                            }
                        }
                    }
                    None => unreachable!("a result leaf has one destination"),
                };
                destination_arguments.insert(*parameter, destination);
            }
            Some((_, ResultDestination::Place(PlaceDestination::Bounded { .. }))) => {
                unreachable!("bounded callable results require an explicit length route")
            }
            None => unreachable!("a result leaf has one destination"),
        }
    }
    if !allocations.is_empty() {
        graph.skeleton.blocks[anchor.block].side_effects.splice(anchor.index..anchor.index, allocations);
    }
    for (old, new) in replacements {
        graph.replace_value_references(old, new);
        graph.install_aliases([(old, new)]);
    }
    for parameter in arguments.len()..boundary.params.len() {
        let parameter = ParameterId::new(parameter);
        let place = destination_arguments.get(&parameter).ok_or_else(|| {
            format!(
                "call to {:?} has no binding for destination parameter {}",
                old.callee(),
                parameter.index()
            )
        })?;
        arguments.push(OperandRef::Place(*place));
    }
    let result = boundary.result.bind(
        |slot, _| return_values[&slot],
        |parameter| destination_arguments[&parameter],
    );
    graph
        .calls
        .get_mut(site)
        .expect("call site remains live while its ABI is rewritten")
        .replace_boundary(arguments, result, boundary.effects);
    Ok(())
}

fn addressable_argument_place(
    graph: &mut PhysicalEGraph,
    argument: OperandRef,
    pointee: &Type<TypeName>,
) -> Option<super::super::types::PlaceId> {
    match graph.canonical_operand(argument) {
        OperandRef::Place(place) => Some(place),
        OperandRef::View(view) => addressable_value_place(graph, view.value(), pointee),
        OperandRef::Value(value) => addressable_value_place(graph, value, pointee),
    }
}

fn addressable_value_place(
    graph: &mut PhysicalEGraph,
    value: ValueId,
    pointee: &Type<TypeName>,
) -> Option<super::super::types::PlaceId> {
    let value = graph.canonical_value(value);
    match graph.nodes[value].kind().clone() {
        ValueKind::PlaceView { place } => Some(place),
        ValueKind::Pure {
            op: PureOp::Index,
            operands,
        } if operands.len() == 2 => {
            let base = graph.canonical_value(operands[0]);
            let index = graph.canonical_value(operands[1]);
            let span = graph.nodes[value].span();
            if let OperandRef::View(view) = graph.operand_ref(base) {
                Some(graph.add_view_index_place(view, index, pointee.clone(), span))
            } else {
                let base_ty = graph.nodes[base].ty().clone();
                let base = addressable_value_place(graph, base, &base_ty)?;
                Some(graph.add_index_place(base, index, pointee.clone(), span))
            }
        }
        ValueKind::Pure {
            op: PureOp::Project { index },
            operands,
        } if operands.len() == 1 => {
            let base = graph.canonical_value(operands[0]);
            let base_ty = graph.nodes[base].ty().clone();
            let base = addressable_value_place(graph, base, &base_ty)?;
            let coordinate = graph.intern_pure(
                PureOp::Int(index.to_string()),
                smallvec![],
                Type::Constructed(TypeName::Int(32), vec![]),
                graph.nodes[value].span(),
            );
            Some(graph.add_index_place(base, coordinate, pointee.clone(), graph.nodes[value].span()))
        }
        _ => None,
    }
}
