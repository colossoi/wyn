//! The only EGIR application path for a finalized fusion plan.
use super::{
    planner::Planned,
    projection, recipe,
    snapshot::{ArrayInput, Catalog, Kind, Operation, ScopeKey},
    FusionError, FusionResult,
};
use crate::ast::TypeName;
use crate::egir::program::{Func, OutputWriter, ProgramIdentities, SemanticProgramData};
use crate::egir::soac::{filter, hist, lambda as lambda_ops, screma};
use crate::egir::types::{
    EGraph, EffectToken, PureOp, ResultBinding, Semantic, SideEffect, SideEffectKind, SkeletonTerminator,
    Soac, SoacEffect, ValueId,
};
use crate::egir::{
    graph_ops,
    ir::{Body, BodySite},
    reify::Segmented,
};
use crate::flow::{BlockId, ControlHeader};
use crate::types::TypeExt;
use crate::{LookupMap, LookupSet};
use polytype::Type;
use smallvec::smallvec;
use wyn_fusion::PortId;

fn required<T>(value: Option<T>, message: &str) -> FusionResult<T> {
    value.ok_or_else(|| FusionError::invalid(message))
}

struct Emitter<'a> {
    program: &'a Segmented,
    catalog: &'a Catalog,
    recipes: &'a recipe::Recipes,
    identities: ProgramIdentities,
    functions: Vec<Func<Semantic>>,
    lambdas: LookupMap<recipe::LambdaId, screma::Lambda>,
    plan: &'a wyn_fusion::Plan<ScopeKey, crate::BindingRef, Operation>,
    projections: projection::HelperCache,
}

impl Emitter<'_> {
    fn canonical(&self, port: PortId) -> PortId {
        self.plan.canonical(port).expect("recipe port belongs to the finalized graph")
    }

    fn value(
        &self,
        graph: &mut EGraph,
        bindings: &LookupMap<PortId, ResultBinding<Type<TypeName>>>,
        port: PortId,
    ) -> FusionResult<ValueId> {
        let port = self.canonical(port);
        if let Some(binding) = bindings.get(&port) {
            return graph_ops::pack_result_values(graph, binding).map_err(FusionError::invalid);
        }
        Ok(required(
            self.catalog.values.get(&port),
            "recipe input has no source or final result",
        )?
        .value)
    }

    fn lambda(
        &mut self,
        lambda: &recipe::Lambda,
        scope: ScopeKey,
        graph: &mut EGraph,
        bindings: &LookupMap<PortId, ResultBinding<Type<TypeName>>>,
    ) -> FusionResult<screma::Lambda> {
        let parameter_types =
            lambda.parameter_types.iter().map(|ty| self.catalog.types[*ty].clone()).collect::<Vec<_>>();
        let Some(body) = &lambda.body else {
            return Ok(screma::Lambda::identity(parameter_types));
        };
        let captures = lambda
            .captures()
            .iter()
            .map(|capture| self.value(graph, bindings, capture.value).map(|value| graph.operand_ref(value)))
            .collect::<FusionResult<Vec<_>>>()?;
        if let Some(source) = lambda.original {
            let super::snapshot::SourceCode::Lambda(original) = &self.catalog.lambdas[source].source else {
                return Err(FusionError::invalid(
                    "source lambda catalog entry is not a function",
                ));
            };
            let mut lambda = original.clone();
            required(lambda.seg_body_mut(), "source lambda lost its region")?.captures = captures;
            return Ok(lambda);
        }
        let recipe::Code::Expression(id) = body.code else {
            unreachable!()
        };
        if let Some(lambda) = self.lambdas.get(&id) {
            return Ok(lambda.clone());
        }
        let mut target = EGraph::new();
        let mut params = lambda_ops::named_parameters(&parameter_types, "value");
        params.append(lambda_ops::named_parameters(
            &lambda
                .captures()
                .iter()
                .map(|capture| self.catalog.types[capture.ty].clone())
                .collect::<Vec<_>>(),
            "capture",
        ));
        let arguments = lambda_ops::function_parameters(&mut target, &params)
            .into_iter()
            .map(|value| value.value())
            .collect::<Vec<_>>();
        let mut block = target.skeleton.entry;
        let results = self.inline(
            lambda,
            &arguments,
            &(0..lambda.result_types.len()).collect::<Vec<_>>(),
            scope,
            &mut target,
            &mut block,
        )?;
        let source = &self.catalog.scopes[&scope];
        let result_types = lambda.result_types.iter().map(|ty| self.catalog.types[*ty].clone()).collect();
        let (lambda, function) = lambda_ops::finish_region_lambda(
            &mut self.identities,
            &source.name,
            &self.recipes.expressions[id].label,
            source.span,
            target,
            block,
            params,
            captures,
            parameter_types,
            result_types,
            results,
            true,
        );
        self.functions.extend(function);
        self.lambdas.insert(id, lambda.clone());
        Ok(lambda)
    }

    fn inline(
        &mut self,
        lambda: &recipe::Lambda,
        arguments: &[Option<ValueId>],
        selected: &[usize],
        scope: ScopeKey,
        graph: &mut EGraph,
        block: &mut BlockId,
    ) -> FusionResult<Vec<ValueId>> {
        let Some(body) = &lambda.body else {
            return selected
                .iter()
                .map(|slot| {
                    required(
                        arguments.get(*slot).copied().flatten(),
                        "identity recipe has an unrouted input",
                    )
                })
                .collect();
        };
        match body.code {
            recipe::Code::Source(id) => {
                let source = &self.catalog.lambdas[id];
                if let super::snapshot::SourceCode::Projection(projection) = &source.source {
                    let location = &self.catalog.scopes[&scope];
                    let mut context = projection::Context {
                        program: self.program,
                        identities: &mut self.identities,
                        scope: &location.name,
                        span: location.span,
                        block: *block,
                        helpers: &mut self.projections,
                    };
                    let (values, helpers) = required(
                        projection.emit(graph, &mut context, arguments),
                        "checked selection could not be emitted",
                    )?;
                    self.functions.extend(helpers);
                    return selected
                        .iter()
                        .map(|slot| required(values.get(*slot).copied(), "projected result is absent"))
                        .collect();
                }
                let super::snapshot::SourceCode::Lambda(original) = &source.source else {
                    unreachable!()
                };
                if let Some(args) = arguments.iter().copied().collect::<Option<Vec<_>>>() {
                    let region =
                        required(original.seg_body(), "source recipe is missing its lambda")?.region;
                    let callee =
                        required(self.program.region(region), "source lambda function is missing")?;
                    let operands = args.iter().map(|value| graph.operand_ref(*value)).collect();
                    let results = lambda_ops::emit_call(graph, *block, original, Some(callee), operands);
                    let values = lambda_ops::result_argument_values(graph, &results);
                    return selected
                        .iter()
                        .map(|slot| required(values.get(*slot).copied(), "source recipe result is absent"))
                        .collect();
                }
                let mut values = Vec::new();
                for slot in selected {
                    let projection = required(
                        source.projections.get(*slot).and_then(Option::as_ref),
                        "source projection was not checked during planning",
                    )?;
                    let location = &self.catalog.scopes[&scope];
                    let mut context = projection::Context {
                        program: self.program,
                        identities: &mut self.identities,
                        scope: &location.name,
                        span: location.span,
                        block: *block,
                        helpers: &mut self.projections,
                    };
                    let (projected, helpers) = required(
                        projection.emit(graph, &mut context, arguments),
                        "checked source projection could not be emitted",
                    )?;
                    values.extend(projected);
                    self.functions.extend(helpers);
                }
                Ok(values)
            }
            recipe::Code::Expression(id) => {
                let expression = self.recipes.expressions[id].clone();
                let selected_calls = expression.selected_calls(selected);
                let mut nodes = LookupMap::new();
                let mut calls = LookupMap::new();
                let guards = selected
                    .iter()
                    .map(|slot| match &expression.nodes[expression.results[*slot]].kind {
                        recipe::NodeKind::Select { predicate, yes, no } => Some((
                            *predicate,
                            *yes,
                            *no,
                            expression.nodes[expression.results[*slot]].ty,
                        )),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();
                if let Some(guards) = guards
                    .filter(|guards| guards.len() > 1 && guards.iter().all(|guard| guard.0 == guards[0].0))
                {
                    let predicate = self.node(
                        &expression,
                        guards[0].0,
                        arguments,
                        scope,
                        graph,
                        block,
                        &mut nodes,
                        &mut calls,
                        &selected_calls,
                    )?;
                    let branches = guards
                        .into_iter()
                        .map(|(_, yes, no, ty)| {
                            let yes = self.node(
                                &expression,
                                yes,
                                arguments,
                                scope,
                                graph,
                                block,
                                &mut nodes,
                                &mut calls,
                                &selected_calls,
                            )?;
                            let no = self.node(
                                &expression,
                                no,
                                arguments,
                                scope,
                                graph,
                                block,
                                &mut nodes,
                                &mut calls,
                                &selected_calls,
                            )?;
                            Ok((self.catalog.types[ty].clone(), yes, no))
                        })
                        .collect::<FusionResult<Vec<_>>>()?;
                    return Ok(select_values(graph, block, predicate, branches));
                }
                selected
                    .iter()
                    .map(|slot| {
                        let node = required(
                            expression.results.get(*slot).copied(),
                            "symbolic projection result is absent",
                        )?;
                        self.node(
                            &expression,
                            node,
                            arguments,
                            scope,
                            graph,
                            block,
                            &mut nodes,
                            &mut calls,
                            &selected_calls,
                        )
                    })
                    .collect()
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn node(
        &mut self,
        expression: &recipe::Expression,
        id: recipe::NodeId,
        arguments: &[Option<ValueId>],
        scope: ScopeKey,
        graph: &mut EGraph,
        block: &mut BlockId,
        nodes: &mut LookupMap<recipe::NodeId, ValueId>,
        calls: &mut LookupMap<recipe::CallId, LookupMap<usize, ValueId>>,
        selected_calls: &LookupMap<recipe::CallId, crate::SortedSet<usize>>,
    ) -> FusionResult<ValueId> {
        if let Some(value) = nodes.get(&id) {
            return Ok(*value);
        }
        let node = &expression.nodes[id];
        let ty = self.catalog.types[node.ty].clone();
        let value = match &node.kind {
            recipe::NodeKind::Parameter(slot) => required(
                arguments.get(*slot).copied().flatten(),
                "symbolic parameter is not routed",
            )?,
            recipe::NodeKind::Integer(value) => integer(graph, *value, ty),
            recipe::NodeKind::Primitive { id, operands, .. } => {
                let values = operands
                    .iter()
                    .map(|node| {
                        self.node(
                            expression,
                            *node,
                            arguments,
                            scope,
                            graph,
                            block,
                            nodes,
                            calls,
                            selected_calls,
                        )
                    })
                    .collect::<FusionResult<Vec<_>>>()?;
                match &self.catalog.primitives[*id] {
                    super::snapshot::Primitive::Constant(value) => graph.intern_constant(value.clone(), ty),
                    super::snapshot::Primitive::Pure(op, span) => {
                        let values = smallvec::SmallVec::from_vec(values);
                        graph
                            .try_algebraic_fold(op, &values, &ty)
                            .unwrap_or_else(|| graph.intern_pure(op.clone(), values, ty, *span))
                    }
                    super::snapshot::Primitive::Union => graph.add_union(values[0], values[1]),
                }
            }
            recipe::NodeKind::Add(left, right) => {
                let left = self.node(
                    expression,
                    *left,
                    arguments,
                    scope,
                    graph,
                    block,
                    nodes,
                    calls,
                    selected_calls,
                )?;
                let right = self.node(
                    expression,
                    *right,
                    arguments,
                    scope,
                    graph,
                    block,
                    nodes,
                    calls,
                    selected_calls,
                )?;
                graph.intern_pure(
                    PureOp::BinOp(crate::op::BinaryOperator::Add),
                    smallvec![left, right],
                    ty,
                    None,
                )
            }
            recipe::NodeKind::CallResult(call_id, slot) => {
                if !calls.contains_key(call_id) {
                    let call = &expression.calls[*call_id];
                    let selected_slots = &selected_calls[call_id];
                    let results = selected_slots.iter().map(|slot| call.results[*slot]).collect::<Vec<_>>();
                    let needed = results
                        .iter()
                        .flat_map(|slot| call.lambda.dependencies[*slot].inputs.iter().copied())
                        .collect::<LookupSet<_>>();
                    let args = call
                        .arguments
                        .iter()
                        .enumerate()
                        .map(|(slot, value)| {
                            if !needed.contains(&slot) {
                                return Ok(None);
                            }
                            value
                                .map(|node| {
                                    self.node(
                                        expression,
                                        node,
                                        arguments,
                                        scope,
                                        graph,
                                        block,
                                        nodes,
                                        calls,
                                        selected_calls,
                                    )
                                })
                                .transpose()
                        })
                        .collect::<FusionResult<Vec<_>>>()?;
                    let values = self.inline(&call.lambda, &args, &results, scope, graph, block)?;
                    calls.insert(*call_id, selected_slots.iter().copied().zip(values).collect());
                }
                required(
                    calls[call_id].get(slot).copied(),
                    "symbolic call result is absent",
                )?
            }
            recipe::NodeKind::Select { predicate, yes, no } => {
                let predicate = self.node(
                    expression,
                    *predicate,
                    arguments,
                    scope,
                    graph,
                    block,
                    nodes,
                    calls,
                    selected_calls,
                )?;
                // Pure map/pre operands are evaluated before guarded selection.
                let yes = self.node(
                    expression,
                    *yes,
                    arguments,
                    scope,
                    graph,
                    block,
                    nodes,
                    calls,
                    selected_calls,
                )?;
                let no = self.node(
                    expression,
                    *no,
                    arguments,
                    scope,
                    graph,
                    block,
                    nodes,
                    calls,
                    selected_calls,
                )?;
                select_values(graph, block, predicate, vec![(ty, yes, no)]).remove(0)
            }
        };
        nodes.insert(id, value);
        Ok(value)
    }

    fn neutral(
        &self,
        neutral: recipe::Neutral,
        graph: &mut EGraph,
        bindings: &LookupMap<PortId, ResultBinding<Type<TypeName>>>,
    ) -> FusionResult<ValueId> {
        match neutral {
            recipe::Neutral::Value(port) => self.value(graph, bindings, port),
            recipe::Neutral::Zero(ty) => Ok(integer(graph, 0, self.catalog.types[ty].clone())),
        }
    }

    fn form(
        &mut self,
        form: &recipe::ScremaForm,
        scope: ScopeKey,
        graph: &mut EGraph,
        bindings: &LookupMap<PortId, ResultBinding<Type<TypeName>>>,
    ) -> FusionResult<screma::ScremaForm> {
        Ok(screma::ScremaForm {
            pre: self.lambda(&form.pre, scope, graph, bindings)?,
            post: self.lambda(&form.post, scope, graph, bindings)?,
            scans: form
                .scans
                .iter()
                .map(|scan| {
                    Ok(screma::Scan {
                        operator: self.lambda(&scan.operator, scope, graph, bindings)?,
                        neutral: scan
                            .neutral
                            .iter()
                            .map(|neutral| self.neutral(*neutral, graph, bindings))
                            .collect::<FusionResult<_>>()?,
                    })
                })
                .collect::<FusionResult<_>>()?,
            reductions: form
                .reductions
                .iter()
                .map(|reduce| {
                    Ok(screma::Reduce {
                        operator: self.lambda(&reduce.operator, scope, graph, bindings)?,
                        neutral: reduce
                            .neutral
                            .iter()
                            .map(|neutral| self.neutral(*neutral, graph, bindings))
                            .collect::<FusionResult<_>>()?,
                        commutative: reduce.commutative,
                    })
                })
                .collect::<FusionResult<_>>()?,
        })
    }

    fn input(
        &self,
        input: &ArrayInput,
        graph: &mut EGraph,
        bindings: &LookupMap<PortId, ResultBinding<Type<TypeName>>>,
    ) -> FusionResult<(ValueId, crate::egir::types::SoacInputType)> {
        let mut value = self.value(graph, bindings, input.node)?;
        let mut ty = self.catalog.input_types[input.ty].clone();
        for slice in &input.slices {
            let start = self.value(graph, bindings, slice.start)?;
            let extent = self.value(graph, bindings, slice.extent)?;
            ty.array = required(
                array_with_outer_size(&ty.array, &self.catalog.types[slice.size]),
                "symbolic slice has an unsupported array type",
            )?;
            let extent_ty = graph.nodes[extent].ty.clone();
            if ty.array.array_variant().is_some_and(crate::types::is_array_variant_view) {
                let length = if slice.length {
                    extent
                } else {
                    graph_ops::intern_binop(
                        graph,
                        crate::op::BinaryOperator::Subtract,
                        extent,
                        start,
                        extent_ty,
                        None,
                    )
                };
                value =
                    graph_ops::intern_inherited_view(graph, value, start, length, ty.array.clone(), None);
            } else {
                let end = if slice.length {
                    graph_ops::intern_binop(
                        graph,
                        crate::op::BinaryOperator::Add,
                        start,
                        extent,
                        extent_ty,
                        None,
                    )
                } else {
                    extent
                };
                value = graph.intern_pure(
                    PureOp::Intrinsic {
                        id: crate::builtins::catalog().known().slice,
                        overload_idx: 0,
                    },
                    smallvec![value, start, end],
                    ty.array.clone(),
                    None,
                );
            }
        }
        Ok((value, ty))
    }

    fn operation(
        &mut self,
        op: &Operation,
        scope: ScopeKey,
        graph: &mut EGraph,
        bindings: &LookupMap<PortId, ResultBinding<Type<TypeName>>>,
        result: ResultBinding<Type<TypeName>>,
        outputs: &[PortId],
    ) -> FusionResult<Vec<SideEffect>> {
        let block = self.catalog.scopes[&scope].block;
        let inputs = op
            .inputs
            .iter()
            .map(|input| self.input(input, graph, bindings))
            .collect::<FusionResult<Vec<_>>>()?;
        let input_types = inputs.iter().map(|(_, ty)| ty.clone()).collect();
        let operands = inputs.iter().map(|(value, _)| graph.operand_ref(*value)).collect();
        if let Kind::Indexed { lambda, demands } = &op.kind {
            let mut emitted = Vec::new();
            let scalar = self.lambda(lambda, scope, graph, bindings)?;
            let callee = scalar
                .seg_body()
                .map(|body| {
                    self.functions
                        .iter()
                        .find(|function| function.region == body.region)
                        .or_else(|| self.program.region(body.region))
                        .cloned()
                })
                .flatten();
            for ((index, slot, path), port) in demands.iter().zip(outputs) {
                let index = self.value(graph, bindings, *index)?;
                let mut arguments = inputs
                    .iter()
                    .map(|(input, ty)| {
                        Some(graph.intern_pure(PureOp::Index, smallvec![*input, index], ty.element(), None))
                    })
                    .collect::<Vec<_>>();
                for capture in lambda.captures() {
                    arguments.push(Some(self.value(graph, bindings, capture.value)?));
                }
                let before = graph.skeleton.blocks[block].side_effects.len();
                let operands =
                    arguments.into_iter().map(|value| graph.operand_ref(value.unwrap())).collect();
                let results = lambda_ops::emit_call(graph, block, &scalar, callee.as_ref(), operands);
                let mut selected = required(results.get(*slot), "indexed recipe result is absent")?.clone();
                for field in path {
                    selected = required(selected.field(*field), "indexed recipe field is absent")?;
                }
                emitted.extend(graph.skeleton.blocks[block].side_effects.drain(before..));
                let old = required(bindings.get(port), "indexed recipe result binding is absent")?;
                graph_ops::rebind_result_value_references(graph, old, &selected)
                    .map_err(FusionError::invalid)?;
            }
            return Ok(emitted);
        }
        let kind = match &op.kind {
            Kind::Screma(form) => {
                let operation = screma::Op {
                    inputs: input_types,
                    form: self.form(form, scope, graph, bindings)?,
                    result_state: op.result_state.clone(),
                    state: screma::SemanticState::Segmented {
                        space: required(op.space.clone(), "Screma recipe has no space")?,
                        output_slots: op.output_slots.clone(),
                        resources: op.resources.clone(),
                    },
                };
                operation.validate().map_err(FusionError::invalid)?;
                Soac::Screma(operation)
            }
            Kind::Filter { map, predicate } => {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(mut original))) =
                    self.catalog.effects[&op.anchor].effect.kind.clone()
                else {
                    return Err(FusionError::invalid("filter recipe lost its source payload"));
                };
                original.body.inputs = input_types;
                original.body.map = self.lambda(map, scope, graph, bindings)?;
                original.body.predicate = self.lambda(predicate, scope, graph, bindings)?;
                original.state.space = required(op.space.clone(), "filter recipe has no space")?;
                original.state.resources = op.resources.clone();
                if let filter::Output::Local { ownership, .. } = &mut original.state.output {
                    if *ownership == crate::types::SoacOwnership::UniqueInput {
                        *ownership = crate::types::SoacOwnership::Fresh;
                    }
                }
                original.body.validate().map_err(FusionError::invalid)?;
                Soac::Filter(original)
            }
            Kind::Hist { bucket } => {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Hist(mut original))) =
                    self.catalog.effects[&op.anchor].effect.kind.clone()
                else {
                    return Err(FusionError::invalid("histogram recipe lost its source payload"));
                };
                original.inputs = input_types;
                original.form.bucket = self.lambda(bucket, scope, graph, bindings)?;
                original.state = hist::SemanticState::Segmented(required(
                    op.space.clone(),
                    "histogram recipe has no space",
                )?);
                Soac::Hist(original)
            }
            Kind::Opaque | Kind::Indexed { .. } => {
                return Err(FusionError::invalid("unsupported composed operation"))
            }
        };
        let mut effect = self.catalog.effects[&op.anchor].effect.clone();
        effect.kind = SideEffectKind::Soac(SoacEffect(
            required(op.semantic_id, "composed operation has no semantic identity")?,
            kind,
        ));
        effect.operands = operands;
        effect.result = Some(result);
        effect.effects = op.effect_tokens;
        Ok(vec![effect])
    }
}

fn select_values(
    graph: &mut EGraph,
    block: &mut BlockId,
    predicate: ValueId,
    branches: Vec<(Type<TypeName>, ValueId, ValueId)>,
) -> Vec<ValueId> {
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let results = branches.iter().map(|(ty, _, _)| graph.add_block_param(merge, ty.clone())).collect();
    graph.skeleton.blocks[*block].term = SkeletonTerminator::CondBranch {
        cond: predicate,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    graph.skeleton.blocks[*block].control_header = Some(ControlHeader::Selection { merge });
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: graph.admit_flow_values(branches.iter().map(|(_, yes, _)| *yes)),
    };
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: graph.admit_flow_values(branches.iter().map(|(_, _, no)| *no)),
    };
    *block = merge;
    results
}

fn integer(graph: &mut EGraph, value: u32, ty: Type<TypeName>) -> ValueId {
    let op = if matches!(ty, Type::Constructed(TypeName::UInt(_), _)) {
        PureOp::Uint(value.to_string())
    } else {
        PureOp::Int(value.to_string())
    };
    graph.intern_pure(op, smallvec![], ty, None)
}

fn array_with_outer_size(array: &Type<TypeName>, size: &Type<TypeName>) -> Option<Type<TypeName>> {
    match array {
        Type::Constructed(TypeName::Array, args) if args.len() >= 4 => {
            let mut args = args.clone();
            args[2] = size.clone();
            Some(Type::Constructed(TypeName::Array, args))
        }
        Type::Constructed(TypeName::Tuple(arity), fields) => Some(Type::Constructed(
            TypeName::Tuple(*arity),
            fields.iter().map(|field| array_with_outer_size(field, size)).collect::<Option<Vec<_>>>()?,
        )),
        _ => None,
    }
}

struct PreparedBody {
    site: BodySite,
    graph: EGraph,
    replacements: Vec<(ValueId, ValueId)>,
    effects: LookupMap<EffectToken, EffectToken>,
    retired_writers: LookupSet<ValueId>,
}

pub(super) fn apply(
    mut program: Segmented,
    planned: Planned,
    catalog: Catalog,
) -> FusionResult<(Segmented, crate::egir::semantic_opt::SemanticOptimizationTrace)> {
    let mut emitter = Emitter {
        program: &program,
        catalog: &catalog,
        recipes: &planned.recipes,
        identities: program.data.identities.clone(),
        functions: vec![],
        lambdas: LookupMap::new(),
        plan: &planned.plan,
        projections: LookupMap::new(),
    };
    let mut seen = LookupSet::new();
    let bodies = planned
        .plan
        .changed_scopes()
        .map(|scope| scope.0)
        .filter(|body| seen.insert(*body))
        .collect::<Vec<_>>();
    let mut prepared = Vec::new();
    for site in bodies {
        let mut graph = required(program.body_graph(site), "planned body is missing")?.clone();
        let body_groups = planned
            .plan
            .scopes()
            .filter(|(scope, _)| scope.0 == site)
            .flat_map(|(scope, groups)| groups.iter().map(move |id| (*id, scope)))
            .collect::<Vec<_>>();
        let mut bindings = LookupMap::new();
        let mut result_bindings = LookupMap::new();
        for (id, _) in &body_groups {
            let group = planned.plan.group(*id)?;
            let op = group.payload();
            if !planned.plan.changed(*id) {
                bindings
                    .extend(group.outputs().iter().map(|port| (*port, catalog.results[port].1.clone())));
                continue;
            }
            let ty = match op.kind {
                Kind::Screma(_) | Kind::Indexed { .. } => Type::Constructed(
                    TypeName::Tuple(op.result_types.len()),
                    op.result_types.iter().map(|ty| catalog.types[*ty].clone()).collect(),
                ),
                _ => required(
                    catalog.effects[&op.anchor].effect.result.as_ref(),
                    "anchored recipe has no result",
                )?
                .ty()
                .clone(),
            };
            let result = graph_ops::alloc_by_value_effect_result(&mut graph, ty);
            let fields = result.top_level_fields();
            if fields.len() != group.outputs().len() {
                return Err(FusionError::invalid(
                    "final recipe result arity differs from graph ports",
                ));
            }
            bindings.extend(group.outputs().iter().copied().zip(fields));
            result_bindings.insert(*id, result);
        }
        let mut sequences = LookupMap::<BlockId, Vec<SideEffect>>::new();
        let mut token_mapping = LookupMap::new();
        let mut retired_writers = LookupSet::new();
        for (id, scope) in &body_groups {
            let block = catalog.scopes[&*scope].block;
            if !planned.plan.changed(*id) {
                sequences.entry(block).or_default().push(catalog.effects[id].effect.clone());
                continue;
            }
            let group = planned.plan.group(*id)?;
            let op = group.payload();
            let effects = emitter.operation(
                op,
                *scope,
                &mut graph,
                &bindings,
                result_bindings[id].clone(),
                group.outputs(),
            )?;
            sequences.entry(block).or_default().extend(effects);
            for source in planned.plan.group(*id)?.members() {
                let source = &catalog.effects[source];
                if let Some((input, output)) = source.effect.effects {
                    let replacement = if matches!(op.kind, Kind::Indexed { .. }) {
                        op.effect_tokens.map(|(input, _)| input).unwrap_or(input)
                    } else {
                        op.effect_tokens.map(|(_, output)| output).unwrap_or(input)
                    };
                    if output != replacement {
                        token_mapping.insert(output, replacement);
                    }
                }
            }
        }
        for (block, sequence) in sequences {
            graph.skeleton.blocks[block].side_effects = sequence;
        }
        for (port, (scope, result)) in &catalog.results {
            if scope.0 == site && !planned.plan.survives(*port)? {
                retired_writers.extend(result.values());
            }
        }
        let mut replacements = Vec::new();
        for (port, target) in planned.plan.replacements() {
            if let Some((_, old)) = catalog.results.get(&port).filter(|(scope, _)| scope.0 == site) {
                replacements.extend(
                    graph_ops::rebind_result_value_references(&mut graph, old, &bindings[&target])
                        .map_err(FusionError::invalid)?,
                );
            }
        }
        for (port, target) in planned.plan.replacements() {
            if catalog.results.contains_key(&port) {
                continue;
            }
            let Some(source) = catalog.values.get(&port).filter(|source| source.body == site) else {
                continue;
            };
            let old = source.value;
            let new = emitter.value(&mut graph, &bindings, target)?;
            let new = graph.canonical_value(new);
            graph.replace_value_references(old, new);
            graph.install_aliases([(old, new)]);
            replacements.push((old, new));
        }
        for (_, block) in &mut graph.skeleton.blocks {
            for effect in &mut block.side_effects {
                if let Some((input, _)) = &mut effect.effects {
                    while let Some(next) = token_mapping.get(input) {
                        *input = *next;
                    }
                }
            }
        }
        graph_ops::fold_exposed_projections(&mut graph);
        graph = EGraph::from_parts(graph.into_parts());
        graph.verify_hash_cons().map_err(FusionError::invalid)?;
        graph.skeleton.verify_branch_arities().map_err(FusionError::invalid)?;
        prepared.push(PreparedBody {
            site,
            graph,
            replacements,
            effects: token_mapping,
            retired_writers,
        });
    }
    let identities = emitter.identities;
    let functions = emitter.functions;
    // All fallible preparation finishes against the unchanged snapshot before
    // any prepared body is installed into the owned program.
    for prepared in prepared {
        program = program.try_rewrite_body(prepared.site, |body| -> FusionResult<_> {
            match body {
                Body::Entry(mut entry) => {
                    entry.graph = prepared.graph;
                    for route in entry.routes_mut() {
                        route.replace_values(&prepared.replacements);
                        route.writers.retain(|writer| !matches!(writer, OutputWriter::Value(value) if prepared.retired_writers.contains(value)));
                        for writer in &mut route.writers {
                            if let OutputWriter::Effect(token) = writer {
                                while let Some(next) = prepared.effects.get(token) { *token = *next; }
                            }
                        }
                    }
                    Ok(Body::Entry(entry))
                }
                Body::Function(mut function) => { function.graph = prepared.graph; Ok(Body::Function(function)) }
                Body::Constant(_) => Err(FusionError::invalid("fusion attempted to replace a constant body")),
            }
        })?;
    }
    Ok((
        program.extend_functions(functions).map_data(|data| SemanticProgramData { identities, ..data }),
        planned.trace,
    ))
}
