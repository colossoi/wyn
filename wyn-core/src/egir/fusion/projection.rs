//! Source projection analysis and emission. Planning retains checked selections only.
use crate::ast::{Span, TypeName};
use crate::egir::graph_projector::{GraphProjector, ValueFlowSelection};
use crate::egir::ir::CallArgument;
use crate::egir::program::{fresh_region_name, Func, ProgramIdentities};
use crate::egir::soac::{lambda as lambda_ops, screma};
use crate::egir::types::{
    CallEffects, EGraph, ParameterId, PureOp, ResultBinding, Semantic, SkeletonTerminator, ValueId,
    ValueKind,
};
use crate::egir::{inlining, reify::Segmented};
use crate::{flow, ssa, FunctionId, LookupMap, LookupSet, StableMap};
use polytype::Type;
use wyn_base::IdArena;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProjectionId(u32);
impl From<u32> for ProjectionId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
pub(super) type HelperCache = LookupMap<(FunctionId, Vec<ValueId>), Func<Semantic>>;

pub(super) struct Context<'a> {
    pub program: &'a Segmented,
    pub identities: &'a mut ProgramIdentities,
    pub scope: &'a str,
    pub span: Span,
    pub block: flow::BlockId,
    pub helpers: &'a mut HelperCache,
}
#[derive(Clone)]
enum ProjectedValue {
    Input(usize),
    Constant {
        value: ssa::types::ConstantValue,
        ty: Type<TypeName>,
    },
    Pure {
        op: PureOp,
        operands: Vec<ProjectedValue>,
        ty: Type<TypeName>,
        span: Option<Span>,
    },
    Union(Box<ProjectedValue>, Box<ProjectedValue>),
    RegionResult {
        projection: ProjectionId,
        result: usize,
    },
}

#[derive(Clone)]
struct RegionProjection {
    region: FunctionId,
    roots: Vec<ValueId>,
    result_types: Vec<Type<TypeName>>,
    selection: ValueFlowSelection,
    arguments: Vec<(ValueId, ProjectedValue)>,
}

#[derive(Clone, Default)]
pub(super) struct ProjectionRecipe {
    values: Vec<ProjectedValue>,
    regions: IdArena<ProjectionId, RegionProjection>,
}

fn function_return_site(
    function: &Func<Semantic>,
) -> Option<(flow::BlockId, ResultBinding<Type<TypeName>>)> {
    let mut returns = function.graph.skeleton.blocks.iter().filter_map(|(block, body)| match &body.term {
        SkeletonTerminator::Return(Some(result)) => Some((block, result.clone())),
        _ => None,
    });
    let result = returns.next()?;
    returns.next().is_none().then_some(result)
}

fn function_result_field(function: &Func<Semantic>, index: usize) -> Option<ValueId> {
    function_return_site(function)?.1.values().get(index).copied()
}

struct ProjectionBuilder<'a> {
    program: &'a Segmented,
    projections: IdArena<ProjectionId, RegionProjection>,
    region_stack: Vec<FunctionId>,
}

pub(super) fn build_projection_recipe(
    program: &Segmented,
    lambda: &screma::Lambda,
    results: &[usize],
) -> Option<ProjectionRecipe> {
    ProjectionBuilder {
        program,
        projections: IdArena::new(),
        region_stack: Vec::new(),
    }
    .lambda(lambda, results)
}

impl ProjectionBuilder<'_> {
    fn lambda(mut self, lambda: &screma::Lambda, results: &[usize]) -> Option<ProjectionRecipe> {
        let body = lambda.seg_body()?;
        let function = self.program.region(body.region)?;
        let (_, result) = function_return_site(function)?;
        let fields = match lambda.result_types.as_slice() {
            [] => vec![],
            [_] => vec![result],
            _ => result.top_level_fields(),
        };
        if fields.len() != lambda.result_types.len() {
            return None;
        }
        let arguments = function
            .params()
            .ids()
            .enumerate()
            .map(|(index, parameter)| (parameter, ProjectedValue::Input(index)))
            .collect::<StableMap<_, _>>();
        let values = results
            .iter()
            .map(|slot| self.binding(function, fields.get(*slot)?, &arguments))
            .collect::<Option<Vec<_>>>()?;
        Some(ProjectionRecipe {
            values,
            regions: self.projections,
        })
    }

    fn binding(
        &mut self,
        function: &Func<Semantic>,
        result: &ResultBinding<Type<TypeName>>,
        arguments: &StableMap<ParameterId, ProjectedValue>,
    ) -> Option<ProjectedValue> {
        if result.is_product() {
            let fields = result
                .top_level_fields()
                .iter()
                .map(|field| self.binding(function, field, arguments))
                .collect::<Option<Vec<_>>>()?;
            Some(ProjectedValue::Pure {
                op: PureOp::Tuple(fields.len()),
                operands: fields,
                ty: result.ty().clone(),
                span: None,
            })
        } else {
            self.function(
                function,
                &[result.single_value()?],
                &[result.ty().clone()],
                arguments,
            )?
            .into_iter()
            .next()
        }
    }

    fn function(
        &mut self,
        function: &Func<Semantic>,
        roots: &[ValueId],
        result_types: &[Type<TypeName>],
        arguments: &StableMap<ParameterId, ProjectedValue>,
    ) -> Option<Vec<ProjectedValue>> {
        if roots.len() != result_types.len()
            || arguments.len() != function.params().len()
            || self.region_stack.contains(&function.region)
        {
            return None;
        }
        self.region_stack.push(function.region);
        let result = (|| {
            if inlining::inlineable_node_count(function).is_some() {
                let mut memo = LookupMap::new();
                return roots
                    .iter()
                    .map(|root| self.value(function, *root, arguments, &mut memo))
                    .collect();
            }

            roots
                .iter()
                .zip(result_types)
                .map(|(root, ty)| {
                    // Keep tuple lanes explicit, selecting each field with the
                    // projector so its control dependencies remain part of the recipe.
                    if let ValueKind::Pure {
                        op: PureOp::Tuple(arity),
                        operands,
                    } = function.graph.nodes[*root].kind()
                    {
                        let fields = operands
                            .iter()
                            .map(|field| {
                                self.region(
                                    function,
                                    &[*field],
                                    &[function.graph.nodes[*field].ty.clone()],
                                    arguments,
                                )?
                                .into_iter()
                                .next()
                            })
                            .collect::<Option<Vec<_>>>()?;
                        Some(ProjectedValue::Pure {
                            op: PureOp::Tuple(*arity),
                            operands: fields,
                            ty: ty.clone(),
                            span: None,
                        })
                    } else {
                        self.region(function, &[*root], &[ty.clone()], arguments)?.into_iter().next()
                    }
                })
                .collect()
        })();
        self.region_stack.pop();
        result
    }

    fn region(
        &mut self,
        function: &Func<Semantic>,
        roots: &[ValueId],
        result_types: &[Type<TypeName>],
        arguments: &StableMap<ParameterId, ProjectedValue>,
    ) -> Option<Vec<ProjectedValue>> {
        let selection = GraphProjector::new(&function.graph).select_value_flow(roots.to_vec()).ok()?;
        let inputs = GraphProjector::new(&function.graph).value_flow_inputs(&selection, roots);
        let mut memo = LookupMap::new();
        let arguments = inputs
            .into_iter()
            .map(|value| Some((value, self.value(function, value, arguments, &mut memo)?)))
            .collect::<Option<Vec<_>>>()?;
        let projection = self.projections.alloc(RegionProjection {
            selection,
            region: function.region,
            roots: roots.to_vec(),
            result_types: result_types.to_vec(),
            arguments,
        });
        Some((0..roots.len()).map(|result| ProjectedValue::RegionResult { projection, result }).collect())
    }

    fn value(
        &mut self,
        function: &Func<Semantic>,
        source: ValueId,
        arguments: &StableMap<ParameterId, ProjectedValue>,
        memo: &mut LookupMap<ValueId, ProjectedValue>,
    ) -> Option<ProjectedValue> {
        if let Some(value) = memo.get(&source) {
            return Some(value.clone());
        }
        let definition = function.graph.nodes.get(source)?;
        if let Some(alias) = definition.alias {
            return self.value(function, alias, arguments, memo);
        }
        let value = match &definition.kind {
            ValueKind::FuncParam { parameter } => arguments.get(parameter)?.clone(),
            ValueKind::Constant(value) => ProjectedValue::Constant {
                value: value.clone(),
                ty: definition.ty.clone(),
            },
            ValueKind::Union { left, right } => ProjectedValue::Union(
                Box::new(self.value(function, *left, arguments, memo)?),
                Box::new(self.value(function, *right, arguments, memo)?),
            ),
            ValueKind::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.len() == 1 => {
                match function.graph.nodes.get(operands[0]).map(|node| &node.kind) {
                    Some(ValueKind::Pure {
                        op: PureOp::Tuple(arity),
                        operands: fields,
                    }) if *arity == fields.len() => {
                        self.value(function, *fields.get(*index as usize)?, arguments, memo)?
                    }
                    _ => {
                        let operand = self.value(function, operands[0], arguments, memo)?;
                        match &operand {
                            ProjectedValue::Pure {
                                op: PureOp::Tuple(arity),
                                operands: fields,
                                ..
                            } if *arity == fields.len() => fields.get(*index as usize)?.clone(),
                            _ => ProjectedValue::Pure {
                                op: PureOp::Project { index: *index },
                                operands: vec![operand],
                                ty: definition.ty.clone(),
                                span: definition.span,
                            },
                        }
                    }
                }
            }
            ValueKind::CallResult { call, .. } => {
                let boundary = function.graph.call(*call);
                if !matches!(boundary.effects(), CallEffects::Pure) {
                    return None;
                }
                let result = boundary.result().values().iter().position(|value| *value == source)?;
                self.call(
                    function,
                    &boundary.callee(),
                    boundary.argument_bindings(),
                    result,
                    arguments,
                    memo,
                )?
            }
            ValueKind::Pure { op, operands } => ProjectedValue::Pure {
                op: op.clone(),
                operands: operands
                    .iter()
                    .map(|operand| self.value(function, *operand, arguments, memo))
                    .collect::<Option<Vec<_>>>()?,
                ty: definition.ty.clone(),
                span: definition.span,
            },
            ValueKind::BlockParam { .. }
            | ValueKind::PlaceLength { .. }
            | ValueKind::PlaceView { .. }
            | ValueKind::SideEffectResult => return None,
        };
        memo.insert(source, value.clone());
        Some(value)
    }

    fn call(
        &mut self,
        caller: &Func<Semantic>,
        callee: &FunctionId,
        call_arguments: &[CallArgument],
        result: usize,
        caller_arguments: &StableMap<ParameterId, ProjectedValue>,
        caller_memo: &mut LookupMap<ValueId, ProjectedValue>,
    ) -> Option<ProjectedValue> {
        let callee = self.program.region(*callee)?;
        if call_arguments.len() != callee.params().len() {
            return None;
        }
        let root = function_result_field(callee, result)?;
        let result_ty = callee.graph.nodes.get(root)?.ty.clone();
        let arguments = callee
            .params()
            .ids()
            .map(|parameter| {
                let argument = call_arguments.iter().find(|argument| argument.parameter() == parameter)?;
                Some((
                    parameter,
                    self.value(caller, argument.value()?, caller_arguments, caller_memo)?,
                ))
            })
            .collect::<Option<StableMap<_, _>>>()?;
        self.function(callee, &[root], &[result_ty], &arguments)?.into_iter().next()
    }
}
impl ProjectionRecipe {
    pub(super) fn symbolic(
        &self,
        program: &Segmented,
        catalog: &mut super::snapshot::Catalog,
        recipes: &mut super::recipe::Recipes,
        parameter_types: Vec<super::recipe::TypeId>,
        captures: Vec<super::recipe::Capture>,
    ) -> Option<super::recipe::Lambda> {
        let mut builder = super::recipe::Builder::new(parameter_types, captures);
        let mut regions = LookupMap::new();
        let results = self
            .values
            .iter()
            .map(|value| self.symbolic_value(value, program, catalog, recipes, &mut builder, &mut regions))
            .collect::<Option<Vec<_>>>()?;
        Some(builder.finish(recipes, results))
    }

    fn symbolic_value(
        &self,
        value: &ProjectedValue,
        program: &Segmented,
        catalog: &mut super::snapshot::Catalog,
        recipes: &super::recipe::Recipes,
        builder: &mut super::recipe::Builder,
        regions: &mut LookupMap<ProjectionId, Vec<super::recipe::NodeId>>,
    ) -> Option<super::recipe::NodeId> {
        use super::{
            recipe::{self, Shape},
            snapshot::{Primitive, SourceCode, SourceLambda},
        };
        match value {
            ProjectedValue::Input(slot) => builder.arguments.get(*slot).copied(),
            ProjectedValue::Constant { value, ty } => {
                let ty = catalog.ty(ty);
                let id = catalog.primitives.alloc(Primitive::Constant(value.clone()));
                builder.primitive(id, Shape::Other, vec![], ty)
            }
            ProjectedValue::Pure {
                op,
                operands,
                ty,
                span,
            } => {
                let operands = operands
                    .iter()
                    .map(|value| self.symbolic_value(value, program, catalog, recipes, builder, regions))
                    .collect::<Option<Vec<_>>>()?;
                let ty = catalog.ty(ty);
                let shape = match op {
                    PureOp::Tuple(_) => Shape::Tuple,
                    PureOp::Project { index } => Shape::Project(*index as usize),
                    _ => Shape::Other,
                };
                let id = catalog.primitives.alloc(Primitive::Pure(op.clone(), *span));
                builder.primitive(id, shape, operands, ty)
            }
            ProjectedValue::Union(left, right) => {
                let left = self.symbolic_value(left, program, catalog, recipes, builder, regions)?;
                let right = self.symbolic_value(right, program, catalog, recipes, builder, regions)?;
                let id = catalog.primitives.alloc(Primitive::Union);
                builder.primitive(id, Shape::Union, vec![left, right], builder.ty(left))
            }
            ProjectedValue::RegionResult { projection, result } => {
                if !regions.contains_key(projection) {
                    let source = &self.regions[*projection];
                    let function = program.region(source.region)?;
                    let args = source
                        .arguments
                        .iter()
                        .map(|(_, value)| {
                            self.symbolic_value(value, program, catalog, recipes, builder, regions)
                        })
                        .collect::<Option<Vec<_>>>()?;
                    let parameter_types = source
                        .arguments
                        .iter()
                        .map(|(value, _)| catalog.ty(&function.graph.nodes[*value].ty))
                        .collect::<Vec<_>>();
                    let result_types =
                        source.result_types.iter().map(|ty| catalog.ty(ty)).collect::<Vec<_>>();
                    let mut normalized = source.clone();
                    normalized.arguments = source
                        .arguments
                        .iter()
                        .enumerate()
                        .map(|(slot, (parameter, _))| (*parameter, ProjectedValue::Input(slot)))
                        .collect();
                    let mut selections = IdArena::new();
                    let id = selections.alloc(normalized);
                    let selected = ProjectionRecipe {
                        values: (0..result_types.len())
                            .map(|result| ProjectedValue::RegionResult {
                                projection: id,
                                result,
                            })
                            .collect(),
                        regions: selections,
                    };
                    let id = catalog.lambdas.alloc(SourceLambda {
                        source: SourceCode::Projection(selected),
                        projections: vec![],
                    });
                    let dependencies = result_types
                        .iter()
                        .map(|_| recipe::Dependency {
                            inputs: (0..parameter_types.len()).collect(),
                            projectable: true,
                        })
                        .collect();
                    let lambda = recipe::Lambda {
                        original: None,
                        body: Some(recipe::Body {
                            code: recipe::Code::Source(id),
                            captures: vec![],
                        }),
                        parameter_types,
                        result_types,
                        dependencies,
                    };
                    let values = builder.invoke(recipes, &lambda, args)?;
                    regions.insert(*projection, values);
                }
                regions.get(projection)?.get(*result).copied()
            }
        }
    }

    pub(super) fn input_dependencies(&self) -> Vec<usize> {
        let mut inputs = crate::SortedSet::new();
        let mut regions = LookupSet::new();
        for value in &self.values {
            self.collect_inputs(value, &mut inputs, &mut regions);
        }
        inputs.into_iter().collect()
    }

    fn collect_inputs(
        &self,
        value: &ProjectedValue,
        inputs: &mut crate::SortedSet<usize>,
        regions: &mut LookupSet<ProjectionId>,
    ) {
        match value {
            ProjectedValue::Input(index) => {
                inputs.insert(*index);
            }
            ProjectedValue::Pure { operands, .. } => {
                for operand in operands {
                    self.collect_inputs(operand, inputs, regions);
                }
            }
            ProjectedValue::Union(left, right) => {
                self.collect_inputs(left, inputs, regions);
                self.collect_inputs(right, inputs, regions);
            }
            ProjectedValue::RegionResult { projection, .. } if regions.insert(*projection) => {
                for (_, argument) in &self.regions[*projection].arguments {
                    self.collect_inputs(argument, inputs, regions);
                }
            }
            ProjectedValue::Constant { .. } | ProjectedValue::RegionResult { .. } => {}
        }
    }
}

struct ProjectionEmitter<'a, 'program> {
    graph: &'a mut EGraph,
    context: &'a mut Context<'program>,
    label: &'a str,
    recipe: &'a ProjectionRecipe,
    inputs: &'a [Option<ValueId>],
    emitted_regions: LookupMap<ProjectionId, Vec<ValueId>>,
    synthesized: Vec<Func<Semantic>>,
}

impl ProjectionEmitter<'_, '_> {
    fn emit(mut self) -> Option<(Vec<ValueId>, Vec<Func<Semantic>>)> {
        let values =
            self.recipe.values.iter().map(|value| self.value(value)).collect::<Option<Vec<_>>>()?;
        Some((values, self.synthesized))
    }

    fn value(&mut self, value: &ProjectedValue) -> Option<ValueId> {
        match value {
            ProjectedValue::Input(index) => self.inputs.get(*index).copied().flatten(),
            ProjectedValue::Constant { value, ty } => {
                Some(self.graph.intern_constant(value.clone(), ty.clone()))
            }
            ProjectedValue::Union(left, right) => {
                let left = self.value(left)?;
                let right = self.value(right)?;
                Some(self.graph.add_union(left, right))
            }
            ProjectedValue::Pure {
                op,
                operands,
                ty,
                span,
            } => {
                let operands =
                    operands.iter().map(|operand| self.value(operand)).collect::<Option<Vec<_>>>()?;
                Some(self.graph.intern_pure(
                    op.clone(),
                    smallvec::SmallVec::from_vec(operands),
                    ty.clone(),
                    *span,
                ))
            }
            ProjectedValue::RegionResult { projection, result } => {
                if !self.emitted_regions.contains_key(projection) {
                    let values = self.region(*projection)?;
                    self.emitted_regions.insert(*projection, values);
                }
                self.emitted_regions.get(projection)?.get(*result).copied()
            }
        }
    }

    fn region(&mut self, index: ProjectionId) -> Option<Vec<ValueId>> {
        let projected = self.recipe.regions.get(index)?.clone();
        let key = (projected.region, projected.roots.clone());
        let call_arguments = projected
            .arguments
            .iter()
            .map(|(_, argument)| self.value(argument).map(|value| self.graph.operand_ref(value)))
            .collect::<Option<Vec<_>>>()?;
        if let Some(helper) = self.context.helpers.get(&key) {
            let (_, result) = self
                .graph
                .emit_call(
                    self.context.block,
                    helper.region,
                    helper.params(),
                    helper.result(),
                    call_arguments,
                    helper.effects(),
                    None,
                    None,
                )
                .ok()?;
            return Some(lambda_ops::result_argument_values(
                self.graph,
                &lambda_ops::logical_result_fields(&result, &projected.result_types),
            ));
        }
        let function = self.context.program.region(projected.region)?.clone();
        let (source_return_block, _) = function_return_site(&function)?;
        let types = projected
            .arguments
            .iter()
            .map(|(value, _)| function.graph.nodes[*value].ty.clone())
            .collect::<Vec<_>>();
        let params = lambda_ops::named_parameters(&types, "selected");
        let inputs =
            projected.arguments.iter().map(|(value, _)| *value).zip(params.ids()).collect::<Vec<_>>();
        let projection =
            GraphProjector::new(&function.graph).emit_value_flow(&projected.selection, &inputs).ok()?;
        let projected_results =
            projected.roots.iter().map(|root| projection.node(*root)).collect::<Option<Vec<_>>>()?;
        let projected_return_block = projection.block(source_return_block)?;

        let name = fresh_region_name(
            self.context.identities,
            &format!("{}_{}", self.context.scope, self.label),
        );
        let region = self.context.identities.alloc_function(name.clone());
        let projected_function = lambda_ops::finish_function(
            projection.graph,
            projected_return_block,
            region,
            name.clone(),
            self.context.span,
            params,
            &projected.result_types,
            &projected_results,
        );
        let (_, result) = self
            .graph
            .emit_call(
                self.context.block,
                region,
                projected_function.params(),
                projected_function.result(),
                call_arguments,
                projected_function.effects(),
                None,
                None,
            )
            .ok()?;
        self.context.helpers.insert(key, projected_function.clone());
        self.synthesized.push(projected_function);
        Some(lambda_ops::result_argument_values(
            self.graph,
            &lambda_ops::logical_result_fields(&result, &projected.result_types),
        ))
    }
}

impl ProjectionRecipe {
    pub(super) fn emit(
        &self,
        graph: &mut EGraph,
        context: &mut Context<'_>,
        arguments: &[Option<ValueId>],
    ) -> Option<(Vec<ValueId>, Vec<Func<Semantic>>)> {
        ProjectionEmitter {
            graph,
            context,
            label: "projection",
            recipe: self,
            inputs: arguments,
            emitted_regions: LookupMap::new(),
            synthesized: Vec::new(),
        }
        .emit()
    }
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
