use crate::ast::TypeName;
use crate::flow;
use crate::types;
use crate::types::TypeExt;
use crate::FunctionId;
use crate::{BindingRef, LookupMap};
use polytype::Type;
use wyn_base::IdSource;

use super::super::graph_ops::{adapt_physical_call_argument, detached_alloca, emit_result_to_place};
use super::super::ir::PlaceOp;
use super::super::program::Func;
use super::super::types::{
    by_value_function_result, destination_passing_function_result, CallEffects, CallSiteId, EGraph,
    EffectOp, EffectToken, FuncParam, FunctionResult, Language, OperandRef, OperandType, Parameters,
    Physical, PlaceAccess, PlaceDestination, PlaceRegion, ResultBinding, ResultDestination, SideEffect,
    SideEffectKind, SkeletonTerminator, ValueId, ValueKind, ViewType, WynLanguage,
};

pub(super) fn resolve(
    mut program: super::super::parallelize::Planned,
) -> Result<super::super::parallelize::Planned, String> {
    for entry in &mut program.entry_points {
        let parameter_resources = entry
            .parameter_inputs
            .iter()
            .map(|inputs| {
                inputs.iter().find_map(|input| entry.inputs.get(input.0).and_then(|input| input.resource))
            })
            .collect::<Vec<_>>();
        resolve_entry_parameter_representations(
            &mut entry.params,
            &mut entry.graph,
            &parameter_resources,
            &mut program.global_context.effect_ids,
        )?;
    }
    for function in &mut program.functions {
        resolve_function_parameters(function)?;
    }
    for function in &mut program.functions {
        resolve_function(function, &mut program.global_context.effect_ids)?;
    }
    let mut boundaries = program
        .functions
        .iter()
        .map(|function| {
            (
                function.region,
                (
                    function.params().clone(),
                    function.result().clone(),
                    function.effects(),
                ),
            )
        })
        .collect::<LookupMap<_, _>>();
    for declaration in &program.externs {
        boundaries.insert(
            declaration.id,
            (
                Parameters::from_ordered(
                    declaration.params.iter().map(|(ty, name)| FuncParam::value(name.clone(), ty.clone())),
                ),
                by_value_function_result::<WynLanguage>(declaration.return_ty.clone()),
                CallEffects::Pure,
            ),
        );
    }
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

enum CallResultRouting {
    Allocate,
    Mapped {
        destinations: ResultBinding<Type<TypeName>>,
        lane: ValueId,
    },
    Existing(ResultBinding<Type<TypeName>>),
}

type BoundCall = (
    Vec<OperandRef>,
    Option<ResultBinding<Type<TypeName>>>,
    Vec<SideEffect<Physical>>,
);

pub(super) fn emit_call(
    graph: &mut EGraph<Physical>,
    block: flow::BlockId,
    callee: &Func<Physical>,
    arguments: impl IntoIterator<Item = OperandRef>,
    mapped_destinations: Option<(&[ResultBinding<Type<TypeName>>], ValueId)>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<ResultBinding<Type<TypeName>>, String> {
    let routing = match mapped_destinations {
        Some((destinations, lane)) => {
            let destinations = if callee.result().is_product() {
                ResultBinding::product(callee.result().ty().clone(), destinations.iter().cloned())
            } else {
                let [destination] = destinations else {
                    return Err(format!(
                        "call to {:?} has one logical result but {} mapped destinations",
                        callee.region,
                        destinations.len()
                    ));
                };
                destination.clone()
            };
            CallResultRouting::Mapped { destinations, lane }
        }
        None => CallResultRouting::Allocate,
    };
    let (arguments, result, prelude) = bind_call_boundary(
        graph,
        callee.region,
        callee.params(),
        callee.result(),
        arguments,
        routing,
        effect_ids,
    )?;
    debug_assert!(result.is_none());
    graph.skeleton.blocks[block].side_effects.extend(prelude);
    graph
        .emit_call(
            block,
            callee.region,
            callee.params(),
            callee.result(),
            arguments,
            callee.effects(),
            None,
            None,
        )
        .map(|(_, result)| result)
}

fn bind_call_boundary(
    graph: &mut EGraph<Physical>,
    callee: FunctionId,
    parameters: &Parameters<BindingRef, Type<TypeName>>,
    function_result: &FunctionResult<Type<TypeName>>,
    arguments: impl IntoIterator<Item = OperandRef>,
    routing: CallResultRouting,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<BoundCall, String> {
    let destination_parameters = function_result.destination_parameters();
    let ordinary_parameters = parameters
        .iter_with_ids()
        .enumerate()
        .filter(|(_, (id, _))| !destination_parameters.contains(id))
        .collect::<Vec<_>>();
    let ordinary_arguments = flatten_call_arguments(graph, arguments)?;
    if ordinary_arguments.len() != ordinary_parameters.len() {
        return Err(format!(
            "call to {callee:?} supplies {} ordinary arguments for {} parameters",
            ordinary_arguments.len(),
            ordinary_parameters.len()
        ));
    }

    let mut physical_arguments = vec![None; parameters.len()];
    let mut prelude = Vec::new();
    for (argument, (index, (_, parameter))) in ordinary_arguments.into_iter().zip(ordinary_parameters) {
        let (argument, effects) =
            adapt_physical_call_argument(graph, argument, parameter, callee, index, effect_ids)?;
        prelude.extend(effects);
        physical_arguments[index] = Some(argument);
    }

    let routed_leaves = match &routing {
        CallResultRouting::Allocate => {
            function_result.destination_leaves().into_iter().map(|route| (route, None)).collect::<Vec<_>>()
        }
        CallResultRouting::Mapped { destinations, .. } | CallResultRouting::Existing(destinations) => {
            let routes = function_result.destination_leaves();
            let sources = destinations.destination_leaves();
            if routes.len() != sources.len() {
                return Err(format!(
                    "call to {callee:?} binds {} results to {} physical result leaves",
                    sources.len(),
                    routes.len()
                ));
            }
            routes.into_iter().zip(sources).map(|(route, source)| (route, Some(source))).collect()
        }
    };

    let preserve_existing_result = matches!(&routing, CallResultRouting::Existing(_));
    let mut return_values = LookupMap::new();
    let mut destination_arguments = LookupMap::new();
    for (route, source) in routed_leaves {
        let (ty, destination) = route.single_destination().expect("physical result leaf has one route");
        let parameter = match destination {
            ResultDestination::ReturnValue(slot) => {
                if preserve_existing_result {
                    let value = source.as_ref().and_then(ResultBinding::single_value).ok_or_else(|| {
                        format!("call to {callee:?} routes a scalar result through a place")
                    })?;
                    return_values.insert(*slot, value);
                }
                continue;
            }
            ResultDestination::Place(PlaceDestination::Fixed(parameter)) => *parameter,
            ResultDestination::Place(PlaceDestination::Bounded { .. }) => {
                return Err(format!(
                    "call to {callee:?} has a bounded result without an explicit length route"
                ))
            }
        };

        let place = match (&routing, source.as_ref()) {
            (CallResultRouting::Allocate, None) => {
                let (place, effect) = detached_alloca(graph, ty.clone(), effect_ids, None);
                prelude.push(effect);
                place
            }
            (CallResultRouting::Mapped { lane, .. }, Some(source)) => {
                let (array_ty, destination) = source
                    .single_destination()
                    .ok_or_else(|| "mapped result leaf has no destination".to_owned())?;
                let element_ty = types::array_elem(array_ty)
                    .ok_or_else(|| "mapped result destination is not an array".to_owned())?;
                if element_ty != ty {
                    return Err(format!(
                        "call to {callee:?} maps result type {ty:?} into array element {element_ty:?}"
                    ));
                }
                match destination {
                    ResultDestination::ReturnValue(view) => {
                        graph.add_view_index_place(graph.view_id(*view), *lane, ty.clone(), None)
                    }
                    ResultDestination::Place(PlaceDestination::Fixed(place)) => {
                        graph.add_index_place(*place, *lane, ty.clone(), None)
                    }
                    ResultDestination::Place(PlaceDestination::Bounded { storage, .. }) => {
                        graph.add_index_place(*storage, *lane, ty.clone(), None)
                    }
                }
            }
            (CallResultRouting::Existing(_), Some(source)) => {
                let (_, destination) =
                    source.single_destination().expect("existing result leaf has one route");
                match destination {
                    ResultDestination::Place(PlaceDestination::Fixed(place)) => *place,
                    ResultDestination::Place(PlaceDestination::Bounded { storage, .. }) => *storage,
                    ResultDestination::ReturnValue(value) => {
                        let value = graph.canonical_value(*value);
                        if let Some(place) =
                            super::super::graph_ops::addressable_value_place(graph, value, ty)
                        {
                            place
                        } else {
                            let span = graph.nodes[value].span();
                            let (place, effect) = detached_alloca(graph, ty.clone(), effect_ids, span);
                            prelude.push(effect);
                            let view_ty = types::view_array_of(ty, types::no_buffer());
                            let view = graph.add_place_view(place, view_ty, span).value();
                            graph.replace_value_references(value, view);
                            graph.install_aliases([(value, view)]);
                            place
                        }
                    }
                }
            }
            _ => unreachable!("result routing supplies the required source shape"),
        };
        destination_arguments.insert(parameter, place);
        let position = parameters
            .abi_position(parameter)
            .ok_or_else(|| format!("call to {callee:?} names an undeclared destination parameter"))?;
        physical_arguments[position] = Some(OperandRef::Place(place));
    }

    let arguments = physical_arguments
        .into_iter()
        .enumerate()
        .map(|(index, argument)| {
            argument.ok_or_else(|| format!("call to {callee:?} has no argument for parameter {index}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = preserve_existing_result.then(|| {
        function_result.bind(
            |slot, _| return_values[&slot],
            |parameter| destination_arguments[&parameter],
        )
    });
    Ok((arguments, result, prelude))
}

fn resolve_entry_parameter_representations(
    params: &mut Parameters<BindingRef, Type<TypeName>>,
    graph: &mut EGraph<Physical>,
    parameter_resources: &[Option<BindingRef>],
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let parameter_ids = params.ids().collect::<Vec<_>>();
    for (index, parameter_id) in parameter_ids.into_iter().enumerate() {
        let OperandType::Value(ty) = params.get(parameter_id).unwrap().representation() else {
            continue;
        };
        let ty = ty.clone();
        let resource = parameter_resources.get(index).copied().flatten();
        let representation = physical_parameter_representation(&ty, resource);
        if representation == *params.get(parameter_id).unwrap().representation() {
            continue;
        }
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
        if let OperandType::Place(place_ty) = &representation {
            let place = graph.add_place_parameter(parameter_id, place_ty.clone());
            let span = graph.nodes[source].span();
            let result = super::super::graph_ops::bind_result_from_place(
                graph,
                &super::super::types::by_value_function_result::<WynLanguage>(ty.clone()),
                place,
                graph.skeleton.entry,
                effect_ids,
                span,
            )?;
            super::super::graph_ops::rebind_result_projection_references(graph, source, &result)?;
            let reference = super::super::graph_ops::pack_result_references(graph, &result)?;
            graph.replace_value_references(source, reference);
            graph.remove_func_param(source);
        } else {
            super::super::graph_ops::retype_projection_tree(graph, source, representation.ty());
        }
        *params.get_mut(parameter_id).unwrap().representation_mut() = representation;
    }
    synchronize_soac_input_types(graph);
    Ok(())
}

fn resolve_function_parameters(function: &mut Func<Physical>) -> Result<(), String> {
    let old_params = function.params.drain_ordered();
    let mut old_parameter_nodes = Vec::new();
    for (old_index, (old_parameter, parameter)) in old_params.into_iter().enumerate() {
        let logical_ty = parameter.ty().clone();
        let logical_abi = super::super::types::by_value_function_result::<WynLanguage>(logical_ty.clone());
        let physical_ty = super::super::graph_ops::place_reference_type(&logical_ty);
        let source = function
            .graph
            .nodes
            .iter()
            .find_map(|(value, definition)| {
                matches!(
                    definition.kind(),
                    ValueKind::FuncParam { parameter } if *parameter == old_parameter
                )
                .then_some(value)
            })
            .ok_or_else(|| {
                format!(
                    "function `{}` parameter {old_index} has no value binding",
                    function.name
                )
            })?;
        old_parameter_nodes.push(source);
        super::super::graph_ops::retype_projection_tree(&mut function.graph, source, &physical_ty);

        let mut group = Vec::new();
        for (path, leaf) in logical_abi.destination_leaves_with_paths() {
            let name = path.iter().fold(parameter.name().to_owned(), |name, index| {
                format!("{name}_{index}")
            });
            if WynLanguage::is_materialized_aggregate(leaf.ty()) || WynLanguage::is_view(leaf.ty()) {
                let id = function.params.push(FuncParam::place(
                    name,
                    super::super::types::PlaceType {
                        pointee: materialized_array_type(leaf.ty()),
                        region: PlaceRegion::Parametric,
                        access: PlaceAccess::ReadOnly,
                    },
                ));
                group.push(id);
            } else {
                let id = function.params.push(FuncParam::value(name, leaf.ty().clone()));
                group.push(id);
            }
        }
        let abi = super::super::types::by_value_function_result::<WynLanguage>(physical_ty);
        let leaves = abi.destination_leaves();
        if leaves.len() != group.len() {
            return Err(format!(
                "function `{}` parameter {old_index} has {} logical leaves but {} physical parameters",
                function.name,
                leaves.len(),
                group.len()
            ));
        }
        let values = group
            .iter()
            .copied()
            .zip(&leaves)
            .map(|(parameter_id, leaf)| {
                let parameter = function.params.get(parameter_id).unwrap();
                let operand = function.graph.add_parameter(parameter_id, parameter.representation());
                match operand {
                    OperandRef::Value(value) => value,
                    OperandRef::View(view) => view.value(),
                    OperandRef::Place(place) => {
                        function.graph.add_place_view(place, leaf.ty().clone(), None).value()
                    }
                }
            })
            .collect::<Vec<_>>();
        let binding = abi.bind(
            |slot, _| values[slot.index()],
            |_| unreachable!("parameter products bind place leaves through view values"),
        );
        super::super::graph_ops::rebind_result_projection_references(
            &mut function.graph,
            source,
            &binding,
        )?;
    }
    super::super::graph_ops::fold_exposed_projections(&mut function.graph);
    for parameter in old_parameter_nodes {
        function.graph.remove_func_param(parameter);
    }
    synchronize_soac_input_types(&mut function.graph);
    Ok(())
}

fn materialized_array_type(ty: &Type<TypeName>) -> Type<TypeName> {
    let Type::Constructed(TypeName::Array, args) = ty else {
        return ty.clone();
    };
    let mut args = args.clone();
    args[1] = types::array_variant_composite();
    *args.last_mut().expect("array has a buffer argument") = types::no_buffer();
    Type::Constructed(TypeName::Array, args)
}

fn physical_parameter_representation(
    ty: &Type<TypeName>,
    resource: Option<BindingRef>,
) -> OperandType<BindingRef, Type<TypeName>> {
    if WynLanguage::is_materialized_aggregate(ty) {
        return match resource {
            Some(resource) => OperandType::View(ViewType {
                array: types::view_array_of(ty, types::buffer_tag(resource)),
                region: PlaceRegion::Resource(resource),
                access: PlaceAccess::ReadOnly,
            }),
            None => OperandType::Place(super::super::types::PlaceType {
                pointee: materialized_array_type(ty),
                region: PlaceRegion::Parametric,
                access: PlaceAccess::ReadOnly,
            }),
        };
    }
    let logical_abi = super::super::types::by_value_function_result::<WynLanguage>(ty.clone());
    if logical_abi.destination_leaves().iter().any(|leaf| {
        (WynLanguage::is_materialized_aggregate(leaf.ty()) || WynLanguage::is_view(leaf.ty()))
            && !matches!(
                leaf.ty().array_buffer(),
                Some(Type::Constructed(TypeName::Buffer(_), _))
            )
    }) {
        return OperandType::Place(super::super::types::PlaceType {
            pointee: materialize_product_leaves(ty),
            region: PlaceRegion::Parametric,
            access: PlaceAccess::ReadOnly,
        });
    }
    OperandType::Value(super::super::graph_ops::place_reference_type(ty))
}

fn materialize_product_leaves(ty: &Type<TypeName>) -> Type<TypeName> {
    if WynLanguage::is_materialized_aggregate(ty) || WynLanguage::is_view(ty) {
        return materialized_array_type(ty);
    }
    match ty {
        Type::Constructed(name, fields) if WynLanguage::product_fields(ty).is_some() => Type::Constructed(
            name.clone(),
            fields.iter().map(materialize_product_leaves).collect(),
        ),
        _ => ty.clone(),
    }
}

fn flatten_call_arguments(
    graph: &mut EGraph<Physical>,
    arguments: impl IntoIterator<Item = OperandRef>,
) -> Result<Vec<OperandRef>, String> {
    let mut flattened = Vec::new();
    for argument in arguments {
        let argument = graph.canonical_operand(argument);
        let Some(value) = argument.value() else {
            flattened.push(argument);
            continue;
        };
        let ty = graph.nodes[value].ty().clone();
        if WynLanguage::product_fields(&ty).is_none() {
            flattened.push(argument);
            continue;
        }
        let abi = super::super::types::by_value_function_result::<WynLanguage>(ty);
        let binding = super::super::graph_ops::bind_by_value_result(graph, &abi, value);
        flattened.extend(binding.values().into_iter().map(|value| graph.operand_ref(value)));
    }
    Ok(flattened)
}

fn synchronize_soac_input_types(graph: &mut EGraph<Physical>) {
    let types = graph
        .nodes
        .iter()
        .map(|(value, definition)| (value, definition.ty().clone()))
        .collect::<LookupMap<_, _>>();
    for (_, block) in &mut graph.skeleton.blocks {
        for effect in &mut block.side_effects {
            let SideEffectKind::Soac(super::super::types::SoacEffect(_, soac)) = &mut effect.kind else {
                continue;
            };
            for (input, operand) in soac.input_types_mut().iter_mut().zip(&effect.operands) {
                if let Some(ty) = operand.value().and_then(|value| types.get(&value)) {
                    input.array = ty.clone();
                }
            }
        }
    }
}

fn resolve_function(
    function: &mut Func<Physical>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let mut params = std::mem::take(&mut function.params);
    let result =
        destination_passing_function_result::<_, WynLanguage>(function.result().ty().clone(), &mut params);
    let destination_parameters = result.destination_parameters();
    if destination_parameters.is_empty() {
        function.params = params;
        function.result = result;
        return Ok(());
    }

    let mut parameter_places = LookupMap::new();
    for parameter in destination_parameters {
        let representation = params
            .get(parameter)
            .expect("destination parameter belongs to the function boundary")
            .representation();
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
    graph: &mut EGraph<Physical>,
    boundaries: &LookupMap<
        FunctionId,
        (
            Parameters<BindingRef, Type<TypeName>>,
            super::super::types::FunctionResult<Type<TypeName>>,
            CallEffects,
        ),
    >,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let calls = graph.side_effect_index().calls().map(|(call, _)| call).collect::<Vec<_>>();
    for site in calls {
        let callee = graph.call(site).callee();
        let Some(boundary) = boundaries.get(&callee) else {
            continue;
        };
        resolve_call(graph, site, boundary, effect_ids)?;
    }
    Ok(())
}

fn resolve_call(
    graph: &mut EGraph<Physical>,
    site: CallSiteId,
    boundary: &(
        Parameters<BindingRef, Type<TypeName>>,
        super::super::types::FunctionResult<Type<TypeName>>,
        CallEffects,
    ),
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let old = graph.call(site).clone();
    let anchor = graph
        .side_effect_index()
        .call_site(site)
        .ok_or_else(|| format!("call {site:?} has no explicit skeleton site"))?;
    let (arguments, result, prelude) = bind_call_boundary(
        graph,
        old.callee(),
        &boundary.0,
        &boundary.1,
        old.arguments().copied(),
        CallResultRouting::Existing(old.result().clone()),
        effect_ids,
    )?;
    if !prelude.is_empty() {
        graph.skeleton.blocks[anchor.block].side_effects.splice(anchor.index..anchor.index, prelude);
    }
    graph
        .calls
        .get_mut(site)
        .expect("call site remains live while its ABI is rewritten")
        .replace_boundary(
            boundary.0.ids().zip(arguments).collect(),
            result.expect("existing call preserves its logical result bindings"),
            boundary.2,
        );
    Ok(())
}
