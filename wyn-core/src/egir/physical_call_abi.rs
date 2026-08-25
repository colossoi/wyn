use crate::ast::TypeName;
use crate::flow;
use crate::types;
use crate::FunctionId;
use crate::{BindingRef, LookupMap, StableMap};
use polytype::Type;
use wyn_base::IdSource;

use super::graph_ops::{adapt_physical_call_argument, detached_alloca, emit_result_to_place};
use super::ir::PlaceOp;
use super::program::{ConstantDef, Func, PhysicalEntry};
use super::types::{
    by_value_function_result, destination_passing_function_result, CallEffects, CallSiteId, EGraph,
    EffectOp, EffectToken, FuncParam, FunctionResult, Language, OperandRef, Parameters, Physical,
    PlaceAccess, PlaceDestination, PlaceRegion, ResultBinding, ResultDestination, SideEffect,
    SideEffectKind, SkeletonTerminator, ValueId, ValueKind, WynLanguage,
};

pub(crate) type CallableBoundary = (
    Parameters<BindingRef, Type<TypeName>>,
    FunctionResult<Type<TypeName>>,
    CallEffects,
);

/// Consume a physical function whose body types have been lowered and return
/// it only after its complete callable boundary and every return have been
/// converted to the physical ABI.
pub(crate) fn physicalize_function_boundary(
    mut function: Func<Physical, WynLanguage, super::ir::PhysicalizingCallableAbi>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<Func<Physical>, String> {
    physicalize_function_parameters(&mut function)?;
    physicalize_function_results(&mut function, effect_ids)?;
    let Func {
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
        abi: _,
    } = function;
    Ok(Func::new(
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
    ))
}

pub(crate) fn callable_boundaries(
    functions: &[Func<Physical>],
    externs: &[crate::types::ExternDecl<Type<TypeName>>],
) -> LookupMap<FunctionId, CallableBoundary> {
    let mut boundaries = functions
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
    for declaration in externs {
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
    boundaries
}

/// Reconcile every existing call only after all internal callable boundaries
/// have reached their final physical representation. This ordering supports
/// recursive and mutually recursive functions without traversal dependence.
pub(crate) fn reconcile_program_calls(
    functions: &mut [Func<Physical>],
    entries: &mut [PhysicalEntry],
    constants: &mut [ConstantDef<Physical>],
    externs: &[crate::types::ExternDecl<Type<TypeName>>],
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let boundaries = callable_boundaries(functions, externs);
    for function in functions {
        reconcile_calls(&mut function.graph, &boundaries, effect_ids)?;
    }
    for entry in entries {
        reconcile_calls(&mut entry.graph, &boundaries, effect_ids)?;
    }
    for constant in constants {
        reconcile_calls(&mut constant.graph, &boundaries, effect_ids)?;
    }
    Ok(())
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

pub(crate) fn emit_call(
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
                        if let Some(place) = super::graph_ops::addressable_value_place(graph, value, ty) {
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

fn physicalize_function_parameters(
    function: &mut Func<Physical, WynLanguage, super::ir::PhysicalizingCallableAbi>,
) -> Result<(), String> {
    let old_params = function.params.drain_ordered();
    for (old_index, (old_parameter, parameter)) in old_params.into_iter().enumerate() {
        let logical_ty = parameter.ty().clone();
        let logical_abi = super::types::by_value_function_result::<WynLanguage>(logical_ty.clone());
        let physical_ty = super::graph_ops::place_reference_type(&logical_ty);
        let sources = function
            .graph
            .nodes
            .iter()
            .filter_map(|(value, definition)| {
                matches!(
                    definition.kind(),
                    ValueKind::FuncParam { parameter } if *parameter == old_parameter
                )
                .then_some(value)
            })
            .collect::<Vec<_>>();
        if sources.is_empty() {
            return Err(format!(
                "function `{}` parameter {old_index} has no value binding",
                function.name
            ));
        }
        for source in &sources {
            super::graph_ops::retype_projection_tree(&mut function.graph, *source, &physical_ty);
        }

        let mut group = Vec::new();
        for (path, leaf) in logical_abi.destination_leaves_with_paths() {
            let name = path.iter().fold(parameter.name().to_owned(), |name, index| {
                format!("{name}_{index}")
            });
            if WynLanguage::is_materialized_aggregate(leaf.ty()) || WynLanguage::is_view(leaf.ty()) {
                let id = function.params.push(FuncParam::place(
                    name,
                    super::types::PlaceType {
                        pointee: super::graph_ops::materialized_array_type(leaf.ty()),
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
        let abi = super::types::by_value_function_result::<WynLanguage>(physical_ty);
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
        for source in sources {
            super::graph_ops::rebind_result_projection_references(&mut function.graph, source, &binding)?;
        }
    }
    super::graph_ops::fold_exposed_projections(&mut function.graph);
    let mut roots = Vec::new();
    for (_, block) in &function.graph.skeleton.blocks {
        for effect in &block.side_effects {
            roots.extend(function.graph.effect_boundary_value_dependencies(effect));
        }
        roots.extend(block.term.referenced_nodes());
    }
    let live = wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(function.graph.nodes[node].children());
        match function.graph.nodes[node].kind() {
            ValueKind::CallResult { call, .. } => out.extend(function.graph.call_value_dependencies(*call)),
            ValueKind::PlaceLength { place } | ValueKind::PlaceView { place } => {
                out.extend(function.graph.place_value_dependencies(*place))
            }
            _ => {}
        }
    });
    let obsolete = function
        .graph
        .nodes
        .iter()
        .filter_map(|(value, definition)| match definition.kind() {
            ValueKind::FuncParam { parameter } if function.params.get(*parameter).is_none() => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    for parameter in obsolete {
        if live.contains(&parameter) {
            return Err(format!(
                "function `{}` retains a live binding for a removed parameter",
                function.name
            ));
        }
        function.graph.remove_func_param(parameter);
    }
    super::graph_ops::synchronize_soac_input_types(&mut function.graph);
    Ok(())
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
        let abi = super::types::by_value_function_result::<WynLanguage>(ty);
        let binding = super::graph_ops::bind_by_value_result(graph, &abi, value);
        flattened.extend(binding.values().into_iter().map(|value| graph.operand_ref(value)));
    }
    Ok(flattened)
}

fn physicalize_function_results(
    function: &mut Func<Physical, WynLanguage, super::ir::PhysicalizingCallableAbi>,
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

fn reconcile_calls(
    graph: &mut EGraph<Physical>,
    boundaries: &LookupMap<
        FunctionId,
        (
            Parameters<BindingRef, Type<TypeName>>,
            FunctionResult<Type<TypeName>>,
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
        FunctionResult<Type<TypeName>>,
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
    let result = result.expect("existing call preserves its logical result bindings");
    let mut return_values = StableMap::new();
    for (route, binding) in boundary.1.destination_leaves().into_iter().zip(result.destination_leaves()) {
        if let (
            Some((_, ResultDestination::ReturnValue(slot))),
            Some((_, ResultDestination::ReturnValue(value))),
        ) = (route.single_destination(), binding.single_destination())
        {
            return_values.insert(*slot, *value);
        }
    }
    graph.rebind_call_boundary(
        site,
        &boundary.0,
        &boundary.1,
        arguments,
        &return_values,
        boundary.2,
    )?;
    Ok(())
}
