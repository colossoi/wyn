//! One read of the source program: payload catalogs and owned planning facts.
use super::{projection, recipe, FusionError, FusionResult};
use crate::ast::{Span, TypeName};
use crate::egir::program::{OutputSlotId, SemanticOpId};
use crate::egir::soac::{hist, screma};
use crate::egir::types::{
    EGraph, PureOp, ResourceAccess, ResultBinding, SegResourceAccess, SegSpace, SideEffect, SideEffectKind,
    Soac, SoacEffect, SoacInputType, ValueId, ValueKind,
};
use crate::egir::{graph_ops, ir::BodySite, reify::Segmented};
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::{BindingRef, LookupMap, StableMap};
use polytype::Type;
use recipe::{Capture, SourceLambdaId, TypeId};
use wyn_base::IdArena;
use wyn_fusion::{GroupId, OrderingReason, PortId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct InputTypeId(u32);
impl From<u32> for InputTypeId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SourceValue {
    pub body: BodySite,
    pub value: ValueId,
}

pub(super) type ScopeKey = (BodySite, BlockId);
type Graph = wyn_fusion::Graph<ScopeKey, BindingRef, Operation>;
type Builder = wyn_fusion::Builder<ScopeKey, BindingRef>;

pub(super) struct Scope {
    pub body: BodySite,
    pub block: BlockId,
    pub name: String,
    pub span: Span,
}

pub(super) struct SourceEffect {
    pub scope: ScopeKey,
    pub effect: SideEffect,
}

pub(super) struct SourceLambda {
    pub source: SourceCode,
    pub projections: Vec<Option<projection::ProjectionRecipe>>,
}

pub(super) enum SourceCode {
    Lambda(screma::Lambda),
    Projection(projection::ProjectionRecipe),
}

#[derive(Clone)]
pub(super) enum Primitive {
    Constant(crate::ssa::types::ConstantValue),
    Pure(PureOp, Option<Span>),
    Union,
}

#[derive(Default)]
pub(super) struct Catalog {
    pub types: IdArena<TypeId, Type<TypeName>>,
    pub primitives: IdArena<recipe::PrimitiveId, Primitive>,
    pub input_types: IdArena<InputTypeId, SoacInputType>,
    pub lambdas: IdArena<SourceLambdaId, SourceLambda>,
    pub scopes: StableMap<ScopeKey, Scope>,
    pub effects: LookupMap<GroupId, SourceEffect>,
    pub results: LookupMap<PortId, (ScopeKey, ResultBinding<Type<TypeName>>)>,
    pub values: LookupMap<PortId, SourceValue>,
}

impl Catalog {
    pub fn ty(&mut self, ty: &Type<TypeName>) -> TypeId {
        if let Some((id, _)) = self.types.iter().find(|(_, existing)| *existing == ty) {
            return *id;
        }
        self.types.alloc(ty.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Slice {
    pub start: PortId,
    pub extent: PortId,
    pub length: bool,
    pub size: TypeId,
}

#[derive(Clone, Debug)]
pub(super) struct ArrayInput {
    pub node: PortId,
    pub ty: InputTypeId,
    pub element_type: TypeId,
    pub slices: Vec<Slice>,
    pub resource: Option<BindingRef>,
}
impl ArrayInput {
    pub fn element(&self) -> TypeId {
        self.element_type
    }
}

pub(super) fn deduplicate_inputs(inputs: Vec<ArrayInput>) -> (Vec<ArrayInput>, Vec<usize>) {
    let mut unique: Vec<ArrayInput> = Vec::new();
    let remap = inputs
        .into_iter()
        .map(|input| {
            if let Some(slot) =
                unique.iter().position(|old| old.node == input.node && old.slices == input.slices)
            {
                slot
            } else {
                let slot = unique.len();
                unique.push(input);
                slot
            }
        })
        .collect();
    (unique, remap)
}

#[derive(Clone, Debug)]
pub(super) enum Use {
    Boundary,
    Slice {
        base: PortId,
        slice: Slice,
    },
    Index {
        base: PortId,
        index: PortId,
    },
    Project {
        base: PortId,
        path: Vec<usize>,
    },
    Length {
        base: PortId,
    },
    Other,
}

#[derive(Clone, Debug)]
pub(super) struct ValueFact {
    pub ty: TypeId,
    pub usage: Use,
}

#[derive(Clone, Debug)]
pub(super) enum Kind {
    Opaque,
    Screma(recipe::ScremaForm),
    Hist {
        bucket: recipe::Lambda,
    },
    Filter {
        map: recipe::Lambda,
        predicate: recipe::Lambda,
    },
    Indexed {
        lambda: recipe::Lambda,
        demands: Vec<(PortId, usize, Vec<usize>)>,
    },
}

#[derive(Clone, Debug)]
pub(super) struct Operation {
    pub kind: Kind,
    pub anchor: GroupId,
    pub semantic_id: Option<SemanticOpId>,
    pub inputs: Vec<ArrayInput>,
    pub outputs: Vec<PortId>,
    pub result_types: Vec<TypeId>,
    pub result_state: Vec<screma::ResultState>,
    pub space: Option<SegSpace>,
    pub output_slots: Vec<OutputSlotId>,
    pub owned_resources: Vec<BindingRef>,
    pub resources: Vec<SegResourceAccess>,
    pub effect_tokens: Option<(crate::egir::types::EffectToken, crate::egir::types::EffectToken)>,
}

pub(super) struct Snapshot {
    pub recipes: recipe::Recipes,
    pub graph: Graph,
    pub values: LookupMap<PortId, ValueFact>,
}

struct Extract<'a> {
    selections: LookupMap<(crate::FunctionId, Vec<usize>), Option<projection::ProjectionRecipe>>,
    recipes: recipe::Recipes,
    program: &'a Segmented,
    builder: Builder,
    catalog: Catalog,
    values: LookupMap<PortId, ValueFact>,
    ports: LookupMap<(ScopeKey, ValueId), PortId>,
    producers: LookupMap<SourceValue, (ScopeKey, PortId)>,
    results: LookupMap<BodySite, Vec<(ScopeKey, PortId, ResultBinding<Type<TypeName>>)>>,
}

impl Snapshot {
    pub fn build(program: &Segmented) -> FusionResult<(Self, Catalog)> {
        let mut extract = Extract {
            selections: LookupMap::new(),
            recipes: recipe::Recipes::default(),
            program,
            builder: Builder::new(),
            catalog: Catalog::default(),
            values: LookupMap::new(),
            ports: LookupMap::new(),
            producers: LookupMap::new(),
            results: LookupMap::new(),
        };
        // Allocate every effect before extracting forward incidences.
        let bodies = program
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, entry)| (BodySite::Entry(index), &entry.graph))
            .chain(
                program
                    .functions
                    .iter()
                    .map(|function| (BodySite::Function(function.region), &function.graph)),
            );
        for (body, graph) in bodies {
            let (span, name) = match body {
                BodySite::Entry(index) => (
                    program.entry_points[index].span,
                    program.entry_points[index].name.clone(),
                ),
                BodySite::Function(id) => {
                    let function = program
                        .region(id)
                        .ok_or_else(|| FusionError::invalid("missing source function"))?;
                    (function.span, function.name.clone())
                }
                BodySite::Constant(_) => unreachable!(),
            };
            for (block, contents) in &graph.skeleton.blocks {
                let scope = (body, block);
                extract.catalog.scopes.insert(
                    scope,
                    Scope {
                        body,
                        block,
                        span,
                        name: name.clone(),
                    },
                );
                for effect in &contents.side_effects {
                    let results = graph
                        .effect_result_binding(effect)
                        .map(|result| result.top_level_fields())
                        .unwrap_or_default();
                    let group = extract.builder.operation(scope, vec![], results.len())?;
                    let ports = extract.builder.outputs(group)?.to_vec();
                    for (result, port) in results.iter().zip(ports) {
                        let ty = extract.catalog.ty(result.ty());
                        extract.values.insert(
                            port,
                            ValueFact {
                                ty,
                                usage: Use::Boundary,
                            },
                        );
                        for (path, leaf) in result.destination_leaves_with_paths() {
                            let crate::egir::ir::ResultDestination::ReturnValue(value) = leaf.destination()
                            else {
                                continue;
                            };
                            let leaf_port = if path.is_empty() {
                                port
                            } else {
                                let leaf_port = extract.builder.value(vec![port])?;
                                let ty = extract.catalog.ty(&graph.nodes[*value].ty);
                                extract.values.insert(
                                    leaf_port,
                                    ValueFact {
                                        ty,
                                        usage: Use::Project {
                                            base: port,
                                            path: path.to_vec(),
                                        },
                                    },
                                );
                                leaf_port
                            };
                            extract
                                .producers
                                .insert(SourceValue { body, value: *value }, (scope, leaf_port));
                            extract.catalog.values.insert(leaf_port, SourceValue { body, value: *value });
                        }
                        extract.results.entry(body).or_default().push((scope, port, result.clone()));
                        extract.catalog.results.insert(port, (scope, result.clone()));
                    }
                    extract.catalog.effects.insert(
                        group,
                        SourceEffect {
                            scope,
                            effect: effect.clone(),
                        },
                    );
                }
            }
        }
        let mut records = extract.catalog.effects.keys().copied().collect::<Vec<_>>();
        records.sort_unstable();
        let mut operations = Vec::new();
        for group in records {
            let source = &extract.catalog.effects[&group];
            let scope = source.scope;
            let body = extract.catalog.scopes[&scope].body;
            let graph =
                program.body_graph(body).ok_or_else(|| FusionError::invalid("missing source body"))?;
            let effect = source.effect.clone();
            let inputs = graph_ops::effect_value_inputs(graph, &effect)
                .into_iter()
                .map(|value| extract.port(scope, graph, value))
                .collect::<FusionResult<Vec<_>>>()?;
            extract.builder.set_inputs(group, inputs)?;
            let mut operation = Operation {
                kind: Kind::Opaque,
                anchor: group,
                semantic_id: None,
                inputs: vec![],
                outputs: extract.builder.outputs(group)?.to_vec(),
                result_types: vec![],
                result_state: vec![],
                space: None,
                output_slots: vec![],
                owned_resources: vec![],
                resources: vec![],
                effect_tokens: effect.effects,
            };
            operation.result_types = operation
                .outputs
                .iter()
                .map(|port| extract.catalog.results[port].1.ty().clone())
                .collect::<Vec<_>>()
                .iter()
                .map(|ty| extract.catalog.ty(ty))
                .collect();
            if let SideEffectKind::Soac(SoacEffect(id, soac)) = &effect.kind {
                operation.semantic_id = Some(*id);
                let input_types = match soac {
                    Soac::Screma(op) => &op.inputs,
                    Soac::Filter(op) => &op.body.inputs,
                    Soac::Hist(op) => &op.inputs,
                };
                for (operand, ty) in effect.operands.iter().zip(input_types) {
                    let Some(value) = operand.value() else {
                        return Err(FusionError::invalid("fusion input is not a value"));
                    };
                    let node = extract.port(scope, graph, value)?;
                    let element_type = extract.catalog.ty(&ty.element());
                    let input_ty = extract.catalog.input_types.alloc(ty.clone());
                    let resource = ty.array.array_buffer().and_then(|buffer| match buffer {
                        Type::Constructed(TypeName::Buffer(binding), _) => Some(*binding),
                        _ => None,
                    });
                    operation.inputs.push(ArrayInput {
                        node,
                        ty: input_ty,
                        element_type,
                        slices: vec![],
                        resource,
                    });
                }
                match soac {
                    Soac::Screma(op) => {
                        if let screma::SemanticState::Segmented {
                            space,
                            output_slots,
                            resources,
                        } = op.semantic_state()
                        {
                            operation.kind = Kind::Screma(extract.form(scope, graph, &op.form)?);
                            operation.space = Some(space.clone());
                            operation.output_slots = output_slots.clone();
                            operation.resources = resources.clone();
                            operation.result_state = op.result_state.clone();
                        }
                    }
                    Soac::Filter(op) => {
                        operation.kind = Kind::Filter {
                            map: extract.lambda(scope, graph, &op.body.map)?,
                            predicate: extract.lambda(scope, graph, &op.body.predicate)?,
                        };
                        operation.space = Some(op.state.space.clone());
                        operation.resources = op.state.resources.clone();
                    }
                    Soac::Hist(op) => {
                        if let hist::SemanticState::Segmented(space) = &op.state {
                            operation.kind = Kind::Hist {
                                bucket: extract.lambda(scope, graph, &op.form.bucket)?,
                            };
                            operation.space = Some(space.clone());
                            operation.resources =
                                crate::egir::semantic_graph::read_resources(graph, &effect);
                            for destination in op
                                .form
                                .operations
                                .iter()
                                .flat_map(|operation| &operation.destinations)
                                .filter_map(|view| {
                                    graph_ops::extract_storage_view_source(graph, view.value())
                                })
                            {
                                if let Some(resource) = operation
                                    .resources
                                    .iter_mut()
                                    .find(|resource| resource.resource == destination)
                                {
                                    resource.access = ResourceAccess::ReadWrite;
                                } else {
                                    operation.resources.push(SegResourceAccess {
                                        resource: destination,
                                        access: ResourceAccess::ReadWrite,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            if let BodySite::Entry(index) = body {
                operation.owned_resources = operation
                    .output_slots
                    .iter()
                    .filter_map(|slot| {
                        program.entry_points[index].outputs.get(slot.0).and_then(|output| output.resource)
                    })
                    .collect();
            }
            for resource in &operation.resources {
                extract.builder.access(
                    group,
                    resource.resource,
                    resource.access != ResourceAccess::Read,
                )?;
            }
            operations.push((group, operation));
        }
        let scopes = extract.catalog.scopes.keys().copied().collect::<Vec<_>>();
        for scope in scopes {
            let source = &extract.catalog.scopes[&scope];
            let body = source.body;
            let block = source.block;
            let graph = program.body_graph(body).unwrap();
            for value in graph.skeleton.blocks[block].term.referenced_nodes() {
                let port = extract.port(scope, graph, value)?;
                extract.builder.observe(port)?;
            }
            if let BodySite::Entry(index) = body {
                for route in
                    program.entry_points[index].routes().filter(|route| route.source.block == block)
                {
                    let port = extract.port(scope, graph, route.source.value)?;
                    extract.builder.observe(port)?;
                }
            }
            // Index/length uses are facts even when nested under other pure
            // expressions. Ports are already restricted to executable roots.
            let scoped = operations
                .iter()
                .map(|(_, op)| op)
                .filter(|op| extract.catalog.effects[&op.anchor].scope == scope)
                .collect::<Vec<_>>();
            for (index, before) in scoped.iter().enumerate() {
                for after in &scoped[index + 1..] {
                    let left = &extract.catalog.effects[&before.anchor].effect;
                    let right = &extract.catalog.effects[&after.anchor].effect;
                    let opaque = |op: &Operation, effect: &SideEffect| {
                        matches!(op.kind, Kind::Opaque)
                            && !matches!(&effect.kind, SideEffectKind::Effect(crate::egir::ir::EffectOp::Call { site }) if matches!(graph.call(*site).effects(), crate::egir::types::CallEffects::Pure))
                    };
                    if opaque(before, left) || opaque(after, right) {
                        extract.builder.order(before.anchor, after.anchor, OrderingReason::Opaque)?;
                    }
                    if matches!((left.effects, right.effects), (Some((_, output)), Some((input, _))) if output == input)
                    {
                        extract.builder.order(before.anchor, after.anchor, OrderingReason::Effect)?;
                    }
                }
            }
        }
        Ok((
            Self {
                recipes: extract.recipes,
                graph: extract.builder.finish(operations)?,
                values: extract.values,
            },
            extract.catalog,
        ))
    }
}

impl Extract<'_> {
    fn port(&mut self, scope: ScopeKey, graph: &EGraph, value: ValueId) -> FusionResult<PortId> {
        if let Some(port) = self.ports.get(&(scope, value)) {
            return Ok(*port);
        }
        let body = self.catalog.scopes[&scope].body;
        let key = SourceValue { body, value };
        let producer = self.producers.get(&key).copied().or_else(|| {
            self.results
                .get(&body)?
                .iter()
                .find(|(_, _, result)| graph.value_has_result_origin(value, result))
                .map(|(scope, port, _)| (*scope, *port))
        });
        if let Some((producer_scope, port)) = producer {
            if producer_scope == scope {
                self.ports.insert((scope, value), port);
                return Ok(port);
            }
            self.builder.observe(port)?;
            let input = self.builder.input();
            self.ports.insert((scope, value), input);
            self.catalog.values.insert(input, key);
            self.values.insert(
                input,
                ValueFact {
                    ty: self.catalog.ty(&graph.nodes[value].ty),
                    usage: Use::Boundary,
                },
            );
            return Ok(input);
        }
        let definition = graph
            .nodes
            .get(value)
            .ok_or_else(|| FusionError::invalid("missing value in fusion snapshot"))?;
        if let Some(alias) = definition.alias {
            let port = self.port(scope, graph, alias)?;
            self.ports.insert((scope, value), port);
            return Ok(port);
        }
        if let Some(field) = graph_ops::projected_tuple_field(graph, value) {
            let port = self.port(scope, graph, field)?;
            self.ports.insert((scope, value), port);
            return Ok(port);
        }
        let ty = self.catalog.ty(&definition.ty);
        let dependencies = match definition.kind() {
            ValueKind::BlockParam { .. } | ValueKind::FuncParam { .. } | ValueKind::Constant(_) => vec![],
            _ => graph.value_dependencies(value),
        };
        let ports = dependencies
            .iter()
            .map(|value| self.port(scope, graph, *value))
            .collect::<FusionResult<Vec<_>>>()?;
        let usage = match definition.kind() {
            ValueKind::Pure {
                op: PureOp::Index, ..
            } if ports.len() == 2 => Use::Index {
                base: ports[0],
                index: ports[1],
            },
            ValueKind::Pure {
                op: PureOp::Intrinsic { id, .. },
                ..
            } if *id == crate::builtins::catalog().known().length && ports.len() == 1 => {
                Use::Length { base: ports[0] }
            }
            ValueKind::Pure { op, .. } if ports.len() == 3 => {
                let size = definition.ty.array_size().map(|size| self.catalog.ty(size));
                match (op, size) {
                    (PureOp::Intrinsic { id, .. }, Some(size))
                        if *id == crate::builtins::catalog().known().slice =>
                    {
                        Use::Slice {
                            base: ports[0],
                            slice: Slice {
                                start: ports[1],
                                extent: ports[2],
                                length: false,
                                size,
                            },
                        }
                    }
                    (PureOp::StorageView(crate::op::PureViewSource::Inherited), Some(size)) => Use::Slice {
                        base: ports[2],
                        slice: Slice {
                            start: ports[0],
                            extent: ports[1],
                            length: true,
                            size,
                        },
                    },
                    _ => Use::Other,
                }
            }
            _ => Use::Other,
        };
        let port = if ports.is_empty() { self.builder.input() } else { self.builder.value(ports)? };
        self.ports.insert((scope, value), port);
        self.catalog.values.insert(port, key);
        self.values.insert(port, ValueFact { ty, usage });
        Ok(port)
    }

    fn selection(
        &mut self,
        lambda: &screma::Lambda,
        results: &[usize],
    ) -> Option<projection::ProjectionRecipe> {
        let key = (lambda.seg_body()?.region, results.to_vec());
        self.selections
            .entry(key)
            .or_insert_with(|| projection::build_projection_recipe(self.program, lambda, results))
            .clone()
    }

    fn lambda(
        &mut self,
        scope: ScopeKey,
        graph: &EGraph,
        lambda: &screma::Lambda,
    ) -> FusionResult<recipe::Lambda> {
        let parameter_types =
            lambda.parameter_types.iter().map(|ty| self.catalog.ty(ty)).collect::<Vec<_>>();
        if lambda.is_identity() {
            return Ok(recipe::Lambda::identity(parameter_types));
        }
        let result_types = lambda.result_types.iter().map(|ty| self.catalog.ty(ty)).collect::<Vec<_>>();
        let captures = lambda
            .captures()
            .iter()
            .map(|capture| {
                let value =
                    capture.value().ok_or_else(|| FusionError::invalid("lambda capture is not a value"))?;
                Ok(Capture {
                    value: self.port(scope, graph, value)?,
                    ty: self.catalog.ty(&graph.nodes[value].ty),
                })
            })
            .collect::<FusionResult<Vec<_>>>()?;
        let projections =
            (0..result_types.len()).map(|slot| self.selection(lambda, &[slot])).collect::<Vec<_>>();
        let dependencies = projections
            .iter()
            .map(|projection| recipe::Dependency {
                inputs: projection
                    .as_ref()
                    .map(|projection| projection.input_dependencies().into_iter().collect())
                    .unwrap_or_else(|| (0..parameter_types.len() + captures.len()).collect()),
                projectable: projection.is_some(),
            })
            .collect();
        let id = self.catalog.lambdas.alloc(SourceLambda {
            source: SourceCode::Lambda(lambda.clone()),
            projections,
        });
        if let Some(projection) = self.selection(lambda, &(0..result_types.len()).collect::<Vec<_>>()) {
            if let Some(mut symbolic) = projection.symbolic(
                self.program,
                &mut self.catalog,
                &mut self.recipes,
                parameter_types.clone(),
                captures.clone(),
            ) {
                symbolic.original = Some(id);
                return Ok(symbolic);
            }
        }
        Ok(recipe::Lambda {
            original: Some(id),
            body: Some(recipe::Body {
                code: recipe::Code::Source(id),
                captures,
            }),
            parameter_types,
            result_types,
            dependencies,
        })
    }

    fn form(
        &mut self,
        scope: ScopeKey,
        graph: &EGraph,
        form: &screma::ScremaForm,
    ) -> FusionResult<recipe::ScremaForm> {
        Ok(recipe::ScremaForm {
            pre: self.lambda(scope, graph, &form.pre)?,
            post: self.lambda(scope, graph, &form.post)?,
            scans: form
                .scans
                .iter()
                .map(|scan| {
                    Ok(recipe::Scan {
                        operator: self.lambda(scope, graph, &scan.operator)?,
                        neutral: scan
                            .neutral
                            .iter()
                            .map(|value| self.port(scope, graph, *value).map(recipe::Neutral::Value))
                            .collect::<FusionResult<_>>()?,
                    })
                })
                .collect::<FusionResult<_>>()?,
            reductions: form
                .reductions
                .iter()
                .map(|reduce| {
                    Ok(recipe::Reduce {
                        operator: self.lambda(scope, graph, &reduce.operator)?,
                        neutral: reduce
                            .neutral
                            .iter()
                            .map(|value| self.port(scope, graph, *value).map(recipe::Neutral::Value))
                            .collect::<FusionResult<_>>()?,
                        commutative: reduce.commutative,
                    })
                })
                .collect::<FusionResult<_>>()?,
        })
    }
}
