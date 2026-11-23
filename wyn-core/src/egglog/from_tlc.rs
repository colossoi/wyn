//! Import typed TLC into flat operation, region, and value relationships.
use super::bindings::Bindings;
use super::dataflow::ir::{self, Bounds, Code, Name, OperationId, Parameter, RegionId};
use super::payload::Payload;
use super::{fusion, output_error, OptimizeError};
use crate::ast::Span;
use crate::builtins::{by_id, catalog, BuiltinId, Purity};
use crate::egglog::graph::CollectiveInput;
use crate::egglog::graph::Formula;
use crate::egglog::graph::{ArgumentSite, Graph, ScalarCode, ScalarExpressionId, State};
use crate::egglog::graph::{Dimension, IterationDomain};
use crate::egglog::graph::{OutputBinding, ParallelResults};
use crate::op::BinaryOperator;
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::stage::OwnershipApplied;
use crate::tlc::{
    self, extract_lambda_params_ref, ArrayExpr, DefMeta, LoopKind, SoacBody, SoacOp, TermKind, VarRef,
};
use crate::types::{self, extract_function_signature, strip_existentials, Diet, Type, TypeExt, TypeName};
use crate::{LookupMap, SymbolId};
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, IntoValues, RawValues, Value, Write};
use smallvec::SmallVec;
use wyn_base::{IdSource, Interner};

#[cfg(test)]
#[path = "from_tlc_tests.rs"]
mod tests;

pub(super) type Term = tlc::Term<ExplicitClosurePayload, ExplicitCapturesPayload>;
type Callback = SoacBody<ExplicitClosurePayload, ExplicitCapturesPayload>;
type SourceArray = ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload>;
type Operator = SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>;
mod layout;

pub(super) fn import(source: &OwnershipApplied) -> Result<(Graph, Interner<i64, SymbolId>), OptimizeError> {
    let mut graph = fusion::new_graph()?;
    let mut symbols = Interner::default();
    graph.update(|sink| {
        let mut import = Import {
            sink,
            symbols: &mut symbols,
            source,
            names: IdSource::new(),
            statements: IdSource::new(),
            regions: IdSource::new(),
            domains: IdSource::new(),
            bindings: Bindings::default(),
            span: Span::generated(),
        };
        Ok(import.program())
    })??;
    Ok((graph, symbols))
}
#[derive(Clone)]
struct Leaf {
    name: Value,
    info: Value,
    ty: Type,
}

#[derive(Clone)]
enum Imported<'source> {
    Leaf(Leaf),
    Callable(&'source Term, Vec<Imported<'source>>),
}

#[derive(Default)]
struct Scope {
    statements: Vec<OperationId>,
    parameters: Vec<Parameter>,
    scalar_results: LookupMap<ScalarExpressionId, Leaf>,
}

struct Domain {
    key: Value,
    parameters: Vec<Parameter>,
}

enum Clause {
    Collect(Vec<Name>),
    Compact {
        predicate: Value,
        values: Vec<Name>,
    },
    Contribute(Vec<Name>),
    Fold {
        kind: Value,
        scan: bool,
        neutral: Vec<Name>,
        combine: RegionId,
        values: Vec<Name>,
    },
}

struct AccumulatorInput {
    array: Value,
    policy: Value,
    combine: Option<RegionId>,
}

struct Import<'graph, 'db, 'operations, 'ids, 'source> {
    sink: State<'graph, 'db, 'operations>,
    symbols: &'ids mut Interner<i64, SymbolId>,
    source: &'source OwnershipApplied,
    names: IdSource<i64>,
    statements: IdSource<i64>,
    regions: IdSource<i64>,
    domains: IdSource<i64>,
    bindings: Bindings<SymbolId, Imported<'source>>,
    span: Span,
}

impl<'source> Import<'_, '_, '_, '_, 'source> {
    fn node(&mut self, name: &str, values: impl IntoValues) -> Result<Value, OptimizeError> {
        Ok(self.sink.add(name, values)?)
    }

    fn unit(&mut self, name: &str) -> Result<Value, OptimizeError> {
        self.node(name, RawValues(Vec::new()))
    }

    fn vector(&mut self, values: Vec<Value>) -> Value {
        self.sink.container_to_value(VecContainer {
            data: values,
            do_rebuild: true,
        })
    }

    fn fresh(&mut self, constructor: &str) -> Result<Value, OptimizeError> {
        let id = self.names.next_id();
        self.node(constructor, id)
    }

    fn binding(&self, leaf: &Leaf) -> Name {
        Name::Value(leaf.name)
    }

    fn operation(&mut self, code: Code, inputs: &[Name]) -> Result<OperationId, OptimizeError> {
        let id = self.statements.next_id();
        let operation = ir::operation(&mut self.sink, id, code, inputs)?;
        if !self.span.is_generated() {
            self.sink.set_span(operation.0, self.span)?;
        }
        Ok(operation)
    }

    fn statement(
        &mut self,
        pattern: Vec<Name>,
        expression: OperationId,
        scope: &mut Scope,
    ) -> Result<(), OptimizeError> {
        ir::outputs(&mut self.sink, expression, &pattern)?;
        scope.statements.push(expression);
        Ok(())
    }

    fn function_name(&mut self, symbol: SymbolId) -> Result<Value, OptimizeError> {
        let id = self.symbols.intern(&symbol);
        let Some(name) = self.source.symbols.get(symbol) else {
            return Err(output_error("function has no source name"));
        };
        let linkage = self
            .source
            .defs
            .iter()
            .find(|definition| definition.name == symbol)
            .and_then(
                |definition| match extract_lambda_params_ref(&definition.body).0.kind {
                    TermKind::Extern(ref linkage) => Some(linkage.as_str()),
                    _ => None,
                },
            )
            .unwrap_or(name.as_str());
        self.node("FunctionId", (id, linkage))
    }

    fn resolve(&mut self, symbol: SymbolId) -> Result<Imported<'source>, OptimizeError> {
        if let Some(value) = self.bindings.get(&symbol) {
            return Ok(value.clone());
        }
        let Some(definition) = self.source.defs.iter().find(|definition| definition.name == symbol) else {
            return Err(output_error(format!("unbound source symbol {symbol:?}")));
        };
        Ok(Imported::Callable(&definition.body, Vec::new()))
    }

    fn leaf(value: Imported<'source>) -> Result<Leaf, OptimizeError> {
        let Imported::Leaf(leaf) = value else {
            return Err(output_error("expected one primitive or array value"));
        };
        Ok(leaf)
    }

    fn atom(&mut self, term: &'source Term, scope: &mut Scope) -> Result<Value, OptimizeError> {
        let value = Self::leaf(self.visit(term, scope)?)?;
        Ok(value.name)
    }

    fn primitive_type(&mut self, ty: &Type) -> Result<Value, OptimizeError> {
        let name = match strip_existentials(ty) {
            Type::Constructed(TypeName::Int(8), _) => "I8",
            Type::Constructed(TypeName::Int(16), _) => "I16",
            Type::Constructed(TypeName::Int(32), _) => "I32",
            Type::Constructed(TypeName::Int(64), _) => "I64",
            Type::Constructed(TypeName::UInt(8), _) => "U8",
            Type::Constructed(TypeName::UInt(16), _) => "U16",
            Type::Constructed(TypeName::UInt(32), _) => "U32",
            Type::Constructed(TypeName::UInt(64), _) => "U64",
            Type::Constructed(TypeName::Float(16), _) => "F16",
            Type::Constructed(TypeName::Float(32), _) => "F32",
            Type::Constructed(TypeName::Float(64), _) => "F64",
            Type::Constructed(TypeName::Bool, _) => "Bool",
            Type::Constructed(TypeName::Unit, _) => "UnitType",
            _ => {
                return Err(output_error(format!(
                    "memory scalar type is not primitive: {ty:?}"
                )))
            }
        };
        self.unit(name)
    }

    fn element(&mut self, ty: &Type) -> Result<Value, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = strip_existentials(ty)
        {
            let elements = fields.iter().map(|ty| self.element(ty)).collect::<Result<Vec<_>, _>>()?;
            let elements = self.vector(elements);
            return self.node("ProductElement", elements);
        }
        if ty.is_mat() {
            let (Some(element), Some(rows), Some(columns)) = (ty.elem_type(), ty.mat_rows(), ty.mat_cols())
            else {
                return Err(output_error("matrix dimensions missing"));
            };
            let primitive = self.primitive_type(element)?;
            return self.node("MatrixElement", (primitive, rows as i64, columns as i64));
        }
        if let Some(size) = ty.vec_size() {
            let Some(element) = ty.elem_type() else {
                return Err(output_error("vector element type missing"));
            };
            let primitive = self.primitive_type(element)?;
            self.node("VectorElement", (primitive, size as i64))
        } else {
            let primitive = self.primitive_type(ty)?;
            self.node("ScalarElement", primitive)
        }
    }

    fn resource_type(&mut self, ty: &Type) -> Result<Option<Value>, OptimizeError> {
        Ok(match strip_existentials(ty) {
            Type::Constructed(TypeName::Texture2D, _) => Some(self.unit("TextureType")?),
            Type::Constructed(TypeName::Sampler, _) => Some(self.unit("SamplerType")?),
            Type::Constructed(TypeName::StorageTexture, _) => {
                let Some(binding) = types::storage_image_buffer(ty) else {
                    return Err(output_error("storage image has no descriptor identity"));
                };
                Some(self.node("ImageType", (i64::from(binding.set), i64::from(binding.binding)))?)
            }
            _ => None,
        })
    }

    fn literal(
        &mut self,
        ty: &Type,
        bits: i64,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let code = Code::Constant { ty: ty.clone(), bits };
        let info = self.value_info(ty)?;
        let expression = self.sink.intern_scalar(ScalarCode::Literal(bits), &[], info);
        if let Some(result) = scope.scalar_results.get(&expression) {
            return Ok(Imported::Leaf(result.clone()));
        }
        let operation = self.operation(code, &[])?;
        self.emit_scalar(operation, ty, info, Some(expression), scope)
    }

    fn primitive(
        &mut self,
        payload: Payload,
        atoms: Vec<Value>,
        ty: &Type,
        pure: bool,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let scalar = ty.as_tensor().map_or(ty, |tensor| tensor.elem);
        let pure = pure
            || (matches!(
                payload,
                Payload::Binary(BinaryOperator::Divide | BinaryOperator::Remainder)
            ) && matches!(scalar, Type::Constructed(TypeName::Float(_), _)));
        let repeatable = match payload {
            Payload::Binary(_) => true,
            Payload::Intrinsic { id, overload } => {
                let builtin = by_id(id);
                let Some(overload) = builtin.overloads().get(overload) else {
                    return Err(output_error("invalid builtin overload"));
                };
                builtin.raw.purity == Purity::Pure && overload.lowering.is_reusable()
            }
            _ => pure,
        };
        let code = Code::Scalar(payload);
        let info = self.value_info(ty)?;
        let inputs: Vec<_> = atoms.into_iter().map(Name::Value).collect();
        let shared = if pure && repeatable {
            let values = inputs
                .iter()
                .map(|input| input.fact(&mut self.sink))
                .collect::<Result<SmallVec<[Value; 2]>, _>>()?;
            Some(self.sink.intern_scalar(ScalarCode::Operation(payload), &values, info))
        } else {
            None
        };
        if let Some(result) = shared.and_then(|key| scope.scalar_results.get(&key)) {
            return Ok(Imported::Leaf(result.clone()));
        }
        let operation = self.operation(code, &inputs)?;
        if pure {
            self.sink.add("DiscardableOperation", operation.0)?;
        } else {
            self.sink.add("OrderedOperation", operation.0)?;
        }
        if repeatable {
            self.sink.add("RepeatableOperation", operation.0)?;
        }
        if let Payload::Intrinsic { id, .. } = payload {
            let known = catalog().known();
            if [
                known.thread_id,
                known.thread_id_y,
                known.thread_id_z,
                known.local_id,
                known.workgroup_id,
                known.num_workgroups,
            ]
            .contains(&id)
            {
                self.sink.add("LaneDependentOperation", operation.0)?;
            }
        }
        self.emit_scalar(operation, ty, info, shared, scope)
    }

    fn emit_scalar(
        &mut self,
        operation: OperationId,
        ty: &Type,
        info: Value,
        expression: Option<ScalarExpressionId>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let value = self.typed_leaf(ty, info)?;
        self.statement(vec![Name::Value(value.name)], operation, scope)?;
        if let Some(expression) = expression {
            scope.scalar_results.insert(expression, value.clone());
        }
        Ok(Imported::Leaf(value))
    }

    fn index(&mut self, term: &'source Term, scope: &mut Scope) -> Result<Value, OptimizeError> {
        let leaf = Self::leaf(self.visit(term, scope)?)?;
        self.node("IndexValue", leaf.name)
    }

    fn apply(
        &mut self,
        function: &'source Term,
        arguments: &'source [Term],
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let arguments =
            arguments.iter().map(|arg| self.visit(arg, scope)).collect::<Result<Vec<_>, _>>()?;
        if let TermKind::Var(VarRef::Symbol(symbol)) = function.kind {
            if !self.bindings.contains_key(&symbol) {
                return self.call(symbol, arguments, ty, scope);
            }
        }
        let Imported::Callable(function, captures) = self.visit(function, scope)? else {
            return Err(output_error("application target is not callable"));
        };
        self.apply_value(function, arguments, captures, ty, scope)
    }

    fn apply_value(
        &mut self,
        function: &'source Term,
        mut arguments: Vec<Imported<'source>>,
        captures: Vec<Imported<'source>>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        arguments.extend(captures);
        if matches!(function.kind, TermKind::Lambda(_)) {
            let (body, params) = extract_lambda_params_ref(function);
            if params.len() != arguments.len() {
                return Err(output_error("call argument count differs from parameter count"));
            }
            let checkpoint = self.bindings.checkpoint();
            for ((name, _), argument) in params.iter().zip(arguments) {
                self.bindings.insert(*name, argument);
            }
            let result = self.visit(body, scope);
            self.bindings.restore(checkpoint);
            return result;
        }
        let (payload, pure) = match &function.kind {
            TermKind::BinOp(operator) => (Payload::Binary(operator.op), operator.op.is_speculatable()),
            TermKind::UnOp(operator) => (Payload::Unary(operator.op), true),
            TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                if let Some(value) = self.array_builtin(*id, &arguments, ty, scope)? {
                    return Ok(value);
                }
                let builtin = by_id(*id);
                let Some(overload) = builtin.overloads().get(*overload_idx) else {
                    return Err(output_error("invalid builtin overload"));
                };
                (
                    Payload::Intrinsic {
                        id: *id,
                        overload: *overload_idx,
                    },
                    builtin.raw.purity == Purity::Pure && overload.lowering.is_speculatable(),
                )
            }
            TermKind::Var(VarRef::Symbol(symbol)) => {
                return self.call(*symbol, arguments, ty, scope);
            }
            _ => return Err(output_error("non-callable source expression")),
        };
        let aggregate_update = matches!(&function.kind,
            TermKind::Var(VarRef::Builtin { id, .. }) if *id == catalog().known().record_with);
        let mut atoms = Vec::new();
        for argument in arguments {
            let leaf = Self::leaf(argument)?;
            if leaf.ty.is_array() && !aggregate_update {
                return Err(output_error(
                    "array operation was not normalized to memory statements",
                ));
            }
            atoms.push(leaf.name);
        }
        self.primitive(payload, atoms, ty, pure, scope)
    }

    fn logical_type(&mut self, ty: &Type) -> Result<Value, OptimizeError> {
        let ty = strip_existentials(ty);
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
            let fields = fields.iter().map(|ty| self.logical_type(ty)).collect::<Result<Vec<_>, _>>()?;
            let fields = self.vector(fields);
            return self.node("ProductType", fields);
        }
        if ty.is_array() {
            let Some(element) = ty.elem_type() else {
                return Err(output_error("array element type missing"));
            };
            let element = self.logical_type(element)?;
            let ownership = self.unit(if ty.array_variant().is_some_and(types::is_array_variant_view) {
                "StorageArray"
            } else {
                "ValueArray"
            })?;
            let Some(dimensions) = ty.array_dims() else {
                return Err(output_error("array dimensions missing"));
            };
            let mut extents = Vec::new();
            for dimension in dimensions {
                let extent = match dimension {
                    Type::Constructed(TypeName::Size(size), _) => self.node(
                        "FixedExtent",
                        i64::try_from(*size).map_err(|_| output_error("array extent exceeds i64"))?,
                    )?,
                    _ => self.unit("UnspecifiedExtent")?,
                };
                extents.push(extent);
            }
            let extents = self.vector(extents);
            return self.node("ArrayType", (ownership, extents, element));
        }
        if let Some(resource) = self.resource_type(ty)? {
            return self.node("ResourceValue", resource);
        }
        let element = self.element(ty)?;
        self.node("ElementValue", element)
    }

    fn value_info(&mut self, ty: &Type) -> Result<Value, OptimizeError> {
        let ty = strip_existentials(ty);
        if ty.is_array()
            || matches!(
                strip_existentials(ty),
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
            )
        {
            let ty = self.logical_type(ty)?;
            self.node("LogicalValue", ty)
        } else if let Some(resource) = self.resource_type(ty)? {
            self.node("ResourceInfo", resource)
        } else {
            let element = self.element(ty)?;
            self.node("ScalarInfo", element)
        }
    }

    fn scalar(&mut self, ty: &Type) -> Result<Leaf, OptimizeError> {
        let info = self.value_info(ty)?;
        self.typed_leaf(ty, info)
    }

    fn typed_leaf(&mut self, ty: &Type, info: Value) -> Result<Leaf, OptimizeError> {
        let ty = strip_existentials(ty);
        let name = self.fresh("ValueId")?;
        self.sink.set_value_type(name, info)?;
        Ok(Leaf {
            name,
            info,
            ty: ty.clone(),
        })
    }

    fn return_info(&mut self, ty: &Type) -> Result<Value, OptimizeError> {
        let ty = strip_existentials(ty);
        if ty.is_array()
            || matches!(
                strip_existentials(ty),
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
            )
        {
            let ty = self.logical_type(ty)?;
            self.node("LogicalReturn", ty)
        } else if let Some(resource) = self.resource_type(ty)? {
            self.node("ReturnResource", resource)
        } else {
            let element = self.element(ty)?;
            self.node("ReturnScalar", element)
        }
    }

    fn diet(&mut self, diet: &Diet) -> Result<Value, OptimizeError> {
        if diet.is_consuming_at_root() {
            return self.unit("Consume");
        }
        if let Diet::Aggregate { components, .. } = diet {
            let fields = components.iter().map(|diet| self.diet(diet)).collect::<Result<Vec<_>, _>>()?;
            let fields = self.vector(fields);
            return self.node("ProductDiet", fields);
        }
        self.unit("Observe")
    }

    fn emit(
        &mut self,
        expression: OperationId,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let value = self.scalar(ty)?;
        let binding = self.binding(&value);
        self.statement(vec![binding], expression, scope)?;
        Ok(Imported::Leaf(value))
    }

    fn finish_region(
        &mut self,
        scope: Scope,
        results: &[Name],
        returns: &[Value],
    ) -> Result<RegionId, OptimizeError> {
        let id = self.regions.next_id();
        let region = ir::region(&mut self.sink, id, &scope.statements)?;
        ir::parameters(&mut self.sink, region, &scope.parameters)?;
        ir::yields(&mut self.sink, region, results)?;
        for (index, &info) in returns.iter().enumerate() {
            self.sink.set("RegionReturnType", (region.0, index as i64), info)?;
        }
        Ok(region)
    }

    fn finish(&mut self, scope: Scope, value: &Imported<'source>) -> Result<RegionId, OptimizeError> {
        let value = Self::leaf(value.clone())?;
        let result = self.binding(&value);
        let info = self.return_info(&value.ty)?;
        self.finish_region(scope, &[result], &[info])
    }

    fn program(&mut self) -> Result<(), OptimizeError> {
        if !self.source.defs.iter().any(|definition| matches!(definition.meta, DefMeta::EntryPoint(_))) {
            self.sink.set("RetainAllFunctions", RawValues(Vec::new()), true)?;
        }
        for definition in &self.source.defs {
            let name = self.function_name(definition.name)?;
            self.sink.set("InvocationEntry", name, false)?;
            let (body, source_params) = extract_lambda_params_ref(&definition.body);
            let checkpoint = self.bindings.checkpoint();
            let mut params = Vec::new();
            let (external_params, result_type) = if matches!(&body.kind, TermKind::Extern(_)) {
                extract_function_signature(&body.ty)
            } else {
                (Vec::new(), body.ty.clone())
            };
            for (i, (symbol, ty)) in source_params.iter().enumerate() {
                let value = self.scalar(ty)?;
                let diet = self.diet(&definition.param_diets.get(i).cloned().unwrap_or_default())?;
                params.push(Parameter {
                    name: self.binding(&value),
                    info: value.info,
                    diet,
                });
                self.bindings.insert(*symbol, Imported::Leaf(value));
            }
            for (i, ty) in external_params.iter().enumerate() {
                let value = self.scalar(ty)?;
                let diet = self.diet(&definition.param_diets.get(i).cloned().unwrap_or_default())?;
                params.push(Parameter {
                    name: self.binding(&value),
                    info: value.info,
                    diet,
                });
            }
            let result = self.return_info(&result_type)?;
            self.sink.set("FunctionResultCount", name, 1_i64)?;
            self.sink.set("FunctionResultType", (name, 0_i64), result)?;
            if matches!(&body.kind, TermKind::Extern(_)) {
                let mut parameters = Vec::with_capacity(params.len());
                for parameter in params {
                    let Name::Value(name) = parameter.name else {
                        return Err(output_error("external parameter is an accumulator capability"));
                    };
                    parameters.push(super::graph::ExternalParameter {
                        name,
                        info: parameter.info,
                        diet: parameter.diet,
                    });
                }
                self.sink.set_external(name, parameters);
                let behavior = self.unit("MayTrapCall")?;
                self.sink.set("FunctionBehavior", name, behavior)?;
            } else {
                let mut scope = Scope {
                    parameters: params,
                    ..Scope::default()
                };
                let value = self.visit(body, &mut scope)?;
                let body = self.finish(scope, &value)?;

                self.sink.set_function_root(name, body.0);
            }
            self.bindings.restore(checkpoint);
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                let id = self.symbols.intern(&definition.name);
                self.sink.set_entry(name, id);
                if entry.declaration.entry_kind != crate::interface::EntryKind::Compute {
                    self.sink.set("InvocationEntry", name, true)?;
                }
                let grid = if let Some(grid) = entry.declaration.compute_dispatch {
                    let groups = [grid.x, grid.y, grid.z]
                        .into_iter()
                        .map(|n| self.node("IndexConstant", i64::from(n)))
                        .collect::<Result<Vec<_>, _>>()?;
                    let one = self.node("IndexConstant", 1_i64)?;
                    let width = self.node("IndexConstant", 128_i64)?;
                    let groups = self.vector(groups);
                    let sizes = self.vector(vec![width, one, one]);
                    self.node("Grid3D", (groups, sizes))?
                } else {
                    self.unit("AutomaticGrid")?
                };
                self.sink.set("EntryGrid", name, grid)?;
            }
        }
        Ok(())
    }

    fn call(
        &mut self,
        symbol: SymbolId,
        arguments: Vec<Imported<'source>>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let name = self.function_name(symbol)?;
        let Some(definition) = self.source.defs.iter().find(|definition| definition.name == symbol) else {
            return Err(output_error("call target has no definition"));
        };
        let mut inputs = Vec::new();
        let mut diets = Vec::new();
        for (i, argument) in arguments.into_iter().enumerate() {
            let value = Self::leaf(argument)?;
            inputs.push(self.binding(&value));
            diets.push(self.diet(&definition.param_diets.get(i).cloned().unwrap_or_default())?);
        }
        let operation = self.operation(Code::Call, &inputs)?;
        self.sink.set_call_target(operation.0, name);
        for (index, &diet) in diets.iter().enumerate() {
            self.sink.set("InputDiet", (operation.0, index as i64), diet)?;
        }
        let result = self.return_info(ty)?;
        self.sink.set("CallReturnType", (operation.0, 0_i64), result)?;
        self.emit(operation, ty, scope)
    }

    fn visit(
        &mut self,
        term: &'source Term,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let previous = std::mem::replace(&mut self.span, term.span);
        let result = self.visit_term(term, scope);
        self.span = previous;
        result
    }

    fn visit_term(
        &mut self,
        term: &'source Term,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let mut term = term;
        while let TermKind::Coerce { inner, .. } = &term.kind {
            term = inner;
        }
        match &term.kind {
            TermKind::Var(VarRef::Symbol(symbol)) => self.resolve(*symbol),
            TermKind::Let { name, rhs, body, .. } => {
                let value = self.visit(rhs, scope)?;
                let checkpoint = self.bindings.checkpoint();
                self.bindings.insert(*name, value);
                let result = self.visit(body, scope);
                self.bindings.restore(checkpoint);
                result
            }
            TermKind::Coerce { .. } => unreachable!("coercions were peeled before import"),
            TermKind::Tuple(fields) => {
                let fields =
                    fields.iter().map(|field| self.atom(field, scope)).collect::<Result<Vec<_>, _>>()?;
                let fields = fields.into_iter().map(Name::Value).collect::<Vec<_>>();
                let operation = self.operation(Code::Product, &fields)?;
                self.emit(operation, &term.ty, scope)
            }
            TermKind::TupleProj { tuple, idx } => {
                let value = Self::leaf(self.visit(tuple, scope)?)?;
                let atom = value.name;
                if value.ty.is_vec() {
                    let index =
                        u32::try_from(*idx).map_err(|_| output_error("vector projection is too large"))?;
                    self.primitive(Payload::Project(index), vec![atom], &term.ty, true, scope)
                } else {
                    let operation = self.operation(Code::Project { index: *idx }, &[Name::Value(atom)])?;
                    self.emit(operation, &term.ty, scope)
                }
            }
            TermKind::Lambda(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Var(VarRef::Builtin { .. }) => Ok(Imported::Callable(term, Vec::new())),
            TermKind::Closure(closure) => {
                let Imported::Callable(function, _) = self.resolve(closure.code)? else {
                    return Err(output_error("closure code is not callable"));
                };
                let captures = closure
                    .captures
                    .iter()
                    .map(|capture| self.visit(capture, scope))
                    .collect::<Result<_, _>>()?;
                Ok(Imported::Callable(function, captures))
            }
            TermKind::App { func, args } => self.apply(func, args, &term.ty, scope),
            TermKind::IntLit(text) => {
                let bits = text
                    .parse::<i64>()
                    .or_else(|_| text.parse::<u64>().map(|n| n as i64))
                    .map_err(|_| output_error("invalid integer literal"))?;
                self.literal(&term.ty, bits, scope)
            }
            TermKind::FloatLit(number) => {
                let bits = if matches!(
                    strip_existentials(&term.ty),
                    Type::Constructed(TypeName::Float(64), _)
                ) {
                    f64::from(*number).to_bits() as i64
                } else {
                    i64::from(number.to_bits())
                };
                self.literal(&term.ty, bits, scope)
            }
            TermKind::BoolLit(value) => self.literal(&term.ty, i64::from(*value), scope),
            TermKind::UnitLit => self.literal(&term.ty, 0, scope),
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let condition = self.atom(cond, scope)?;
                let mut left = Scope::default();
                let mut right = Scope::default();
                let yes = self.visit(then_branch, &mut left)?;
                let no = self.visit(else_branch, &mut right)?;
                let yes = self.finish(left, &yes)?;
                let no = self.finish(right, &no)?;
                let expression = self.operation(Code::Branch, &[Name::Value(condition)])?;
                self.sink.set_control(expression.0, 0_i64, yes.0);
                self.sink.set_control(expression.0, 1_i64, no.0);
                self.emit(expression, &term.ty, scope)
            }
            TermKind::ArrayExpr(array) => self.array(array, scope),
            TermKind::Index { array, index } => {
                let array = Self::leaf(self.visit(array, scope)?)?;
                let index = self.atom(index, scope)?;
                if array.ty.is_vec() {
                    let array = array.name;
                    self.primitive(Payload::Index, vec![array, index], &term.ty, true, scope)
                } else {
                    let operation = self.index_operation(array.name, index)?;
                    self.emit(operation, &term.ty, scope)
                }
            }
            TermKind::VecLit(fields) => {
                let atoms =
                    fields.iter().map(|field| self.atom(field, scope)).collect::<Result<Vec<_>, _>>()?;
                self.primitive(Payload::Vector, atoms, &term.ty, true, scope)
            }
            TermKind::Soac(operator) => self.soac(operator, &term.ty, scope),
            TermKind::Loop { .. } => self.loop_expression(term, scope),
            TermKind::Extern(_) => Err(output_error("external declaration in an expression")),
        }
    }

    fn array_names(
        &mut self,
        arrays: &'source [SourceArray],
        scope: &mut Scope,
    ) -> Result<Vec<Name>, OptimizeError> {
        arrays
            .iter()
            .map(|array| {
                let value = Self::leaf(self.array(array, scope)?)?;
                Ok(Name::Value(value.name))
            })
            .collect()
    }

    fn index_operation(&mut self, source: Value, index: Value) -> Result<OperationId, OptimizeError> {
        self.operation(Code::Index, &[Name::Value(source), Name::Value(index)])
    }

    fn row_type(ty: &Type) -> Result<Type, OptimizeError> {
        if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) =
            strip_existentials(ty)
        {
            let fields = fields.iter().map(Self::row_type).collect::<Result<Vec<_>, _>>()?;
            return Ok(Type::Constructed(name.clone(), fields));
        }
        let Type::Constructed(TypeName::Array, mut fields) = strip_existentials(ty).clone() else {
            return Err(output_error(format!("array input type missing: {ty:?}")));
        };
        if ty.array_rank().is_some_and(|rank| rank > 1) {
            fields.remove(2);
            Ok(Type::Constructed(TypeName::Array, fields))
        } else {
            Ok(fields.remove(0))
        }
    }

    fn iteration(&mut self, input: &Leaf, scope: &mut Scope) -> Result<(Domain, Leaf), OptimizeError> {
        let length = self.operation(Code::Length { axis: 0 }, &[Name::Value(input.name)])?;
        let length = Self::leaf(self.emit(length, &types::i32(), scope)?)?;
        let count = self.node("IndexValue", length.name)?;
        let (domain, mut axes) = self.domain(&[count])?;
        let Some(axis) = axes.pop() else {
            return Err(output_error("iteration domain is missing its axis"));
        };
        Ok((domain, axis))
    }

    fn domain(&mut self, extents: &[Value]) -> Result<(Domain, Vec<Leaf>), OptimizeError> {
        let ordinal = self.scalar(&types::i32())?;
        let id = self.domains.next_id();
        let domain = self.node("DomainId", id)?;
        let mut dimensions = Vec::with_capacity(extents.len());
        let mut axes = Vec::with_capacity(extents.len());
        for &extent in extents {
            let index = self.scalar(&types::i32())?;
            dimensions.push(Dimension::new(index.name, extent));
            axes.push(index);
        }
        self.sink.set_domain(
            domain,
            IterationDomain {
                ordinal: Some(ordinal.name),
                dimensions,
            },
        );
        let observe = self.unit("Observe")?;
        let parameters = std::iter::once(&ordinal)
            .chain(&axes)
            .map(|value| Parameter {
                name: Name::Value(value.name),
                info: value.info,
                diet: observe,
            })
            .collect();
        Ok((
            Domain {
                key: domain,
                parameters,
            },
            axes,
        ))
    }

    fn parallel_operation(
        &mut self,
        domain: Domain,
        mut inner: Scope,
        clauses: Vec<Clause>,
    ) -> Result<OperationId, OptimizeError> {
        inner.parameters = domain.parameters;
        let body = self.finish_region(inner, &[], &[])?;
        let operation = self.operation(Code::Parallel, &[])?;
        self.sink.set_operation_domain(operation.0, domain.key);
        self.sink.set_control(operation.0, 0_i64, body.0);
        let execution = self.unit("Unscheduled")?;
        self.sink.set("OperationExecution", operation.0, execution)?;
        let mut results = ParallelResults::default();
        for (index, clause) in clauses.into_iter().enumerate() {
            let index = index as i64;
            let (kind, role, values) = match clause {
                Clause::Collect(values) => (self.unit("Collected")?, self.unit("CollectedOutput")?, values),
                Clause::Compact { predicate, values } => {
                    self.sink.set_collective_inputs(
                        operation.0,
                        index,
                        CollectiveInput::Predicate,
                        vec![predicate],
                    )?;
                    (self.unit("Compacted")?, self.unit("CompactedOutput")?, values)
                }
                Clause::Contribute(values) => {
                    (self.unit("Contributed")?, self.unit("UpdatedOutput")?, values)
                }
                Clause::Fold {
                    kind,
                    scan,
                    neutral,
                    combine,
                    values,
                } => {
                    self.sink.set_reducer(operation.0, index, combine.0);
                    let neutral = neutral
                        .into_iter()
                        .map(|value| value.fact(&mut self.sink))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.sink.set_collective_inputs(
                        operation.0,
                        index,
                        CollectiveInput::Neutral,
                        neutral,
                    )?;
                    (
                        self.node("Folded", kind)?,
                        self.unit(if scan { "PrefixOutput" } else { "TotalOutput" })?,
                        values,
                    )
                }
            };
            results.kinds.push(Some(kind));
            let values = values
                .into_iter()
                .map(|value| value.fact(&mut self.sink))
                .collect::<Result<Vec<_>, _>>()?;
            for field in 0..values.len() {
                results.bindings.push(OutputBinding::new(index, field as i64, role));
            }
            self.sink.set_collective_inputs(operation.0, index, CollectiveInput::Contribution, values)?;
        }
        self.sink.set_parallel_results(operation.0, results);
        Ok(operation)
    }

    fn parallel(
        &mut self,
        domain: Domain,
        inner: Scope,
        clauses: Vec<Clause>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let operation = self.parallel_operation(domain, inner, clauses)?;
        self.emit(operation, ty, scope)
    }

    fn result(&self, value: &Imported<'source>) -> Result<Name, OptimizeError> {
        let value = Self::leaf(value.clone())?;
        Ok(Name::Value(value.name))
    }

    fn array(
        &mut self,
        array: &'source SourceArray,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match array {
            ArrayExpr::Var(VarRef::Symbol(symbol), _) => self.resolve(*symbol),
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => Err(output_error("builtin used as an array")),
            ArrayExpr::Literal(elements) => {
                let elements =
                    elements.iter().map(|term| self.atom(term, scope)).collect::<Result<Vec<_>, _>>()?;
                let inputs = elements.into_iter().map(Name::Value).collect::<Vec<_>>();
                let expression = self.operation(Code::ArrayLiteral, &inputs)?;
                self.emit(expression, &array.array_type(), scope)
            }
            ArrayExpr::Zip(inputs) => {
                let values = inputs
                    .iter()
                    .map(|input| Self::leaf(self.array(input, scope)?))
                    .collect::<Result<Vec<_>, _>>()?;
                let Some(first) = values.first() else {
                    return Err(output_error("zip has no inputs"));
                };
                let (domain, index) = self.iteration(first, scope)?;
                let position = index.name;
                let mut inner = Scope::default();
                let mut fields = Vec::new();
                let mut types = Vec::new();
                for source in values {
                    let row_type = Self::row_type(&source.ty)?;
                    let expression = self.index_operation(source.name, position)?;
                    let value = Self::leaf(self.emit(expression, &row_type, &mut inner)?)?;
                    fields.push(value.name);
                    types.push(row_type);
                }
                let inputs = fields.into_iter().map(Name::Value).collect::<Vec<_>>();
                let expression = self.operation(Code::Product, &inputs)?;
                let value = self.emit(expression, &types::tuple(types), &mut inner)?;
                let result = self.result(&value)?;
                let clause = Clause::Collect(vec![result]);
                self.parallel(domain, inner, vec![clause], &array.array_type(), scope)
            }
            ArrayExpr::Range { start, len, step } => {
                let start_atom = self.atom(start, scope)?;
                let length = self.index(len, scope)?;
                let step = if let Some(step) = step {
                    self.atom(step, scope)?
                } else {
                    let bits = match strip_existentials(&start.ty) {
                        Type::Constructed(TypeName::Float(64), _) => 1.0f64.to_bits() as i64,
                        Type::Constructed(TypeName::Float(_), _) => i64::from(1.0f32.to_bits()),
                        _ => 1,
                    };
                    Self::leaf(self.literal(&start.ty, bits, scope)?)?.name
                };
                let expression =
                    self.operation(Code::Range, &[Name::Value(start_atom), Name::Value(step)])?;
                self.sink.set_formula(expression.0, Formula::Index, length)?;
                self.emit(expression, &array.array_type(), scope)
            }
        }
    }

    fn array_builtin(
        &mut self,
        id: BuiltinId,
        arguments: &[Imported<'source>],
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Option<Imported<'source>>, OptimizeError> {
        let known = catalog().known();
        if id == known.transpose || id == known.flatten {
            let [source] = arguments else {
                return Err(output_error("array layout transform expects one array"));
            };
            return self
                .array_layout(Self::leaf(source.clone())?, ty, id == known.transpose, scope)
                .map(Some);
        }
        if ![
            known.length,
            known.storage_len,
            known.storage_index,
            known.slice,
            known.array_with,
            known.array_with_in_place,
            known.scratch_annotation,
            known.scratch_alloc,
        ]
        .contains(&id)
        {
            return Ok(None);
        }
        let values = arguments.iter().cloned().map(Self::leaf).collect::<Result<Vec<_>, _>>()?;
        let Some(source) = values.first() else {
            return Err(output_error("array operand missing"));
        };
        let expression = if id == known.length || id == known.storage_len {
            self.operation(Code::Length { axis: 0 }, &[Name::Value(source.name)])?
        } else if id == known.scratch_annotation {
            self.operation(Code::Scratch, &[Name::Value(source.name)])?
        } else if id == known.scratch_alloc {
            let count = self.node("IndexValue", source.name)?;
            let shape = self.vector(vec![count]);
            let operation = self.operation(Code::Scratch, &[])?;
            self.sink.set_formula(operation.0, Formula::Extents, shape)?;
            operation
        } else {
            let Some(index) = values.get(1) else {
                return Err(output_error("array index missing"));
            };
            if id == known.storage_index {
                self.index_operation(source.name, index.name)?
            } else {
                let index = self.node("IndexValue", index.name)?;
                let Some(value) = values.get(2) else {
                    return Err(output_error("array final operand missing"));
                };
                if id == known.slice {
                    let end = self.node("IndexValue", value.name)?;
                    let count = self.node("IndexSub", (end, index))?;
                    let one = self.node("IndexConstant", 1_i64)?;
                    let slice = self.node("Slice", (index, count, one))?;
                    let slices = self.vector(vec![slice]);
                    let operation = self.operation(Code::View, &[Name::Value(source.name)])?;
                    self.sink.set_formula(operation.0, Formula::Slices, slices)?;
                    operation
                } else {
                    let destination = if id == known.array_with
                        && !source.ty.array_variant().is_some_and(types::is_array_variant_view)
                    {
                        let copy = self.operation(Code::Copy, &[Name::Value(source.name)])?;
                        Self::leaf(self.emit(copy, &source.ty, scope)?)?.name
                    } else {
                        source.name
                    };
                    let slice = self.node("At", index)?;
                    let slices = self.vector(vec![slice]);
                    let operation = self.operation(
                        Code::Update {
                            bounds: Bounds::Ignore,
                        },
                        &[Name::Value(destination), Name::Value(value.name)],
                    )?;
                    self.sink.set_formula(operation.0, Formula::Slices, slices)?;
                    operation
                }
            }
        };
        self.emit(expression, ty, scope).map(Some)
    }

    fn apply_callback(
        &mut self,
        callback: &'source Callback,
        arguments: Vec<Imported<'source>>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let checkpoint = self.bindings.checkpoint();
        let captures = callback
            .data
            .captures
            .iter()
            .map(|(name, _, term)| self.visit(term, scope).map(|value| (*name, value)))
            .collect::<Result<Vec<_>, _>>()?;
        if callback.lam.params.len() != arguments.len() {
            return Err(output_error("callback parameter count mismatch"));
        }
        for ((name, _), value) in callback.lam.params.iter().zip(&arguments) {
            self.bindings.insert(*name, value.clone());
        }
        for (name, value) in &captures {
            self.bindings.insert(*name, value.clone());
        }
        let body = &callback.lam.body;
        let result = if matches!(body.kind, TermKind::Var(VarRef::Symbol(_))) {
            let Imported::Callable(function, closure) = self.visit(body, scope)? else {
                return Err(output_error("callback envelope does not reference a function"));
            };
            let mut arguments = arguments;
            arguments.extend(captures.into_iter().map(|(_, value)| value));
            self.apply_value(function, arguments, closure, &callback.lam.ret_ty, scope)
        } else {
            self.visit(body, scope)
        };
        self.bindings.restore(checkpoint);
        result
    }

    fn soac(
        &mut self,
        operator: &'source Operator,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match operator {
            SoacOp::Map { lam, inputs } => {
                let inputs = inputs
                    .iter()
                    .map(|input| Self::leaf(self.array(input, scope)?))
                    .collect::<Result<Vec<_>, _>>()?;
                let Some(first) = inputs.first() else {
                    return Err(output_error("map has no inputs"));
                };
                let (domain, index) = self.iteration(first, scope)?;
                let position = index.name;
                let mut inner = Scope::default();
                let mut arguments = Vec::new();
                for input in inputs {
                    let expression = self.index_operation(input.name, position)?;
                    arguments.push(self.emit(expression, &Self::row_type(&input.ty)?, &mut inner)?);
                }
                let value = self.apply_callback(lam, arguments, &mut inner)?;
                let result = self.result(&value)?;
                let clause = Clause::Collect(vec![result]);
                self.parallel(domain, inner, vec![clause], ty, scope)
            }
            SoacOp::Reduce { op, ne, input } | SoacOp::Scan { op, ne, input } => {
                let input = Self::leaf(self.array(input, scope)?)?;
                let neutral = self.atom(ne, scope)?;
                let combine = self.callback(op, scope)?;
                let (domain, index) = self.iteration(&input, scope)?;
                let position = index.name;
                let mut inner = Scope::default();
                let expression = self.index_operation(input.name, position)?;
                let value = self.emit(expression, &Self::row_type(&input.ty)?, &mut inner)?;
                let result = self.result(&value)?;
                let kind = self.unit(if matches!(operator, SoacOp::Scan { .. }) {
                    "InclusiveScan"
                } else {
                    "Total"
                })?;
                let clause = Clause::Fold {
                    kind,
                    scan: matches!(operator, SoacOp::Scan { .. }),
                    neutral: vec![Name::Value(neutral)],
                    combine,
                    values: vec![result],
                };
                self.parallel(domain, inner, vec![clause], ty, scope)
            }
            SoacOp::Filter { pred, input } => {
                let input = Self::leaf(self.array(input, scope)?)?;
                let (domain, index) = self.iteration(&input, scope)?;
                let position = index.name;
                let mut inner = Scope::default();
                let expression = self.index_operation(input.name, position)?;
                let value = self.emit(expression, &Self::row_type(&input.ty)?, &mut inner)?;
                let condition = Self::leaf(self.apply_callback(pred, vec![value.clone()], &mut inner)?)?;
                let condition = condition.name;
                let result = self.result(&value)?;
                let clause = Clause::Compact {
                    predicate: condition,
                    values: vec![result],
                };
                self.parallel(domain, inner, vec![clause], ty, scope)
            }
            SoacOp::Scatter { dest, lam, inputs } => self.scatter(dest, lam, inputs, ty, scope),
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => self.indexed_fold(dest, op, ne, indices, values, ty, scope),
            SoacOp::BucketScatter {
                dest,
                lam,
                inputs,
                input_dimensions,
                domain_rank,
            } => {
                let destination = Self::leaf(self.visit(dest, scope)?)?;
                let arrays = self.array_names(inputs, scope)?;
                let mut inputs = vec![Name::Value(destination.name)];
                inputs.extend(arrays);
                let callback = self.callback(lam, scope)?;
                let coordinates = input_dimensions
                    .iter()
                    .map(|axes| axes.iter().map(|&axis| i64::from(axis)).collect())
                    .collect();
                let expression = self.operation(
                    Code::BucketScatter {
                        coordinates,
                        depth: *domain_rank as usize,
                    },
                    &inputs,
                )?;
                self.sink.set_control(expression.0, 0_i64, callback.0);
                self.emit(expression, ty, scope)
            }
        }
    }

    fn accumulator(
        &mut self,
        destination: &Leaf,
        policy: Value,
        combine: Option<RegionId>,
        inputs: &mut Vec<AccumulatorInput>,
        parameters: &mut Vec<Parameter>,
    ) -> Result<(Value, Value), OptimizeError> {
        let ty = self.logical_type(&destination.ty)?;
        let info = self.node("LogicalAccumulator", ty)?;
        let token = self.fresh("AccumulatorId")?;
        self.sink.set("AccumulatorType", token, info)?;
        let consume = self.unit("Consume")?;
        parameters.push(Parameter {
            name: Name::Accumulator(token),
            info,
            diet: consume,
        });
        inputs.push(AccumulatorInput {
            array: destination.name,
            policy,
            combine,
        });
        Ok((token, info))
    }

    fn contribute(
        &mut self,
        tokens: Vec<(Value, Value)>,
        domain: Domain,
        lane: Scope,
        inputs: Vec<AccumulatorInput>,
        parameters: Vec<Parameter>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let names = tokens.iter().map(|&(name, _)| Name::Accumulator(name)).collect();
        let parallel = self.parallel_operation(domain, lane, vec![Clause::Contribute(names)])?;
        let mut pattern = Vec::new();
        let mut returns = Vec::new();
        for (_, info) in tokens {
            let name = self.fresh("AccumulatorId")?;
            self.sink.set("AccumulatorType", name, info)?;
            pattern.push(Name::Accumulator(name));
            returns.push(self.node("ReturnAccumulator", info)?);
        }
        let mut body = Scope {
            parameters,
            ..Scope::default()
        };
        self.statement(pattern.clone(), parallel, &mut body)?;
        let body = self.finish_region(body, &pattern, &returns)?;
        let operation = self.operation(Code::WithAcc, &[])?;
        self.sink.set_control(operation.0, 0_i64, body.0);
        let policies = inputs.iter().map(|input| Some(input.policy)).collect();
        self.sink.set_accumulator_policies(operation.0, policies);
        for (group, input) in inputs.into_iter().enumerate() {
            let group = group as i64;
            self.sink.set_collective_inputs(
                operation.0,
                group,
                CollectiveInput::Backing,
                vec![input.array],
            )?;
            if let Some(combine) = input.combine {
                self.sink.set_combiner(operation.0, group, combine.0);
            }
        }
        self.emit(operation, ty, scope)
    }

    fn accumulate(
        &mut self,
        token: Value,
        info: Value,
        index: Value,
        value: Value,
        scope: &mut Scope,
    ) -> Result<(Value, Value), OptimizeError> {
        let index = self.node("IndexValue", index)?;
        let indices = self.vector(vec![index]);
        let operation = self.operation(
            Code::Accumulate {
                bounds: Bounds::Ignore,
            },
            &[Name::Accumulator(token), Name::Value(value)],
        )?;
        self.sink.set_formula(operation.0, Formula::Extents, indices)?;
        let next = self.fresh("AccumulatorId")?;
        self.sink.set("AccumulatorType", next, info)?;
        self.statement(vec![Name::Accumulator(next)], operation, scope)?;
        Ok((next, info))
    }

    fn project(
        &mut self,
        value: &Leaf,
        index: usize,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Leaf, OptimizeError> {
        let atom = value.name;
        let expression = self.operation(Code::Project { index }, &[Name::Value(atom)])?;
        Self::leaf(self.emit(expression, ty, scope)?)
    }

    fn scatter_groups(
        &mut self,
        destination: &Leaf,
        contribution: &Leaf,
        policy: Value,
        inputs: &mut Vec<AccumulatorInput>,
        parameters: &mut Vec<Parameter>,
        tokens: &mut Vec<(Value, Value)>,
        scope: &mut Scope,
        lane: &mut Scope,
    ) -> Result<(), OptimizeError> {
        let Type::Constructed(TypeName::Tuple(_), fields) = strip_existentials(&contribution.ty) else {
            return Err(output_error("scatter callback must return index/value pairs"));
        };
        if let [index_ty, value_ty] = fields.as_slice() {
            if matches!(
                strip_existentials(index_ty),
                Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _)
            ) {
                let (token, info) = self.accumulator(destination, policy, None, inputs, parameters)?;
                let index = self.project(contribution, 0, index_ty, lane)?;
                let value = self.project(contribution, 1, value_ty, lane)?;
                let value = value.name;
                tokens.push(self.accumulate(token, info, index.name, value, lane)?);
                return Ok(());
            }
        }
        let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), destinations) =
            strip_existentials(&destination.ty)
        else {
            return Err(output_error(
                "scatter destination does not match callback product",
            ));
        };
        if destinations.len() != fields.len() {
            return Err(output_error("scatter product arity mismatch"));
        }
        for (i, (destination_ty, value_ty)) in destinations.iter().zip(fields).enumerate() {
            let destination = self.project(destination, i, destination_ty, scope)?;
            let value = self.project(contribution, i, value_ty, lane)?;
            self.scatter_groups(
                &destination,
                &value,
                policy,
                inputs,
                parameters,
                tokens,
                scope,
                lane,
            )?;
        }
        Ok(())
    }

    fn scatter(
        &mut self,
        dest: &'source Term,
        callback: &'source Callback,
        sources: &'source [SourceArray],
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let destination = Self::leaf(self.visit(dest, scope)?)?;
        let destination = if destination.ty.array_variant().is_some_and(types::is_array_variant_view) {
            destination
        } else {
            let copy = self.operation(Code::Copy, &[Name::Value(destination.name)])?;
            Self::leaf(self.emit(copy, &destination.ty, scope)?)?
        };
        let sources = sources
            .iter()
            .map(|input| Self::leaf(self.array(input, scope)?))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(first) = sources.first() else {
            return Err(output_error("scatter has no input"));
        };
        let (domain, index) = self.iteration(first, scope)?;
        let position = index.name;
        let mut lane = Scope::default();
        let mut arguments = Vec::new();
        for source in sources {
            let expression = self.index_operation(source.name, position)?;
            arguments.push(self.emit(expression, &Self::row_type(&source.ty)?, &mut lane)?);
        }
        let contribution = Self::leaf(self.apply_callback(callback, arguments, &mut lane)?)?;
        let policy = self.unit("IdenticalWrites")?;
        let mut inputs = Vec::new();
        let mut parameters = Vec::new();
        let mut tokens = Vec::new();
        self.scatter_groups(
            &destination,
            &contribution,
            policy,
            &mut inputs,
            &mut parameters,
            &mut tokens,
            scope,
            &mut lane,
        )?;
        self.contribute(tokens, domain, lane, inputs, parameters, ty, scope)
    }

    fn indexed_fold(
        &mut self,
        dest: &'source Term,
        callback: &'source Callback,
        neutral: &'source Term,
        indices: &'source SourceArray,
        values: &'source SourceArray,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let destination = Self::leaf(self.visit(dest, scope)?)?;
        let destination = if destination.ty.array_variant().is_some_and(types::is_array_variant_view) {
            destination
        } else {
            let copy = self.operation(Code::Copy, &[Name::Value(destination.name)])?;
            Self::leaf(self.emit(copy, &destination.ty, scope)?)?
        };
        self.visit(neutral, scope)?;
        let indices = Self::leaf(self.array(indices, scope)?)?;
        let values = Self::leaf(self.array(values, scope)?)?;
        let combine = self.callback(callback, scope)?;
        let policy = self.unit("CombineWrites")?;
        let mut inputs = Vec::new();
        let mut parameters = Vec::new();
        let (token, info) =
            self.accumulator(&destination, policy, Some(combine), &mut inputs, &mut parameters)?;
        let (domain, index) = self.iteration(&indices, scope)?;
        let position = index.name;
        let mut lane = Scope::default();
        let expression = self.index_operation(indices.name, position)?;
        let index = Self::leaf(self.emit(expression, &Self::row_type(&indices.ty)?, &mut lane)?)?;
        let expression = self.index_operation(values.name, position)?;
        let value = Self::leaf(self.emit(expression, &Self::row_type(&values.ty)?, &mut lane)?)?;
        let value = value.name;
        let token = self.accumulate(token, info, index.name, value, &mut lane)?;
        self.contribute(vec![token], domain, lane, inputs, parameters, ty, scope)
    }

    fn callback(
        &mut self,
        callback: &'source Callback,
        scope: &mut Scope,
    ) -> Result<RegionId, OptimizeError> {
        let checkpoint = self.bindings.checkpoint();
        let mut captures = Vec::new();
        for (name, _, term) in &callback.data.captures {
            let value = self.visit(term, scope)?;
            self.bindings.insert(*name, value.clone());
            captures.push(value);
        }
        let mut parameters = Vec::new();
        let mut arguments = Vec::new();
        for (name, ty) in &callback.lam.params {
            let value = self.scalar(ty)?;
            let binding = self.binding(&value);
            let diet = self.unit("Observe")?;
            parameters.push(Parameter {
                name: binding,
                info: value.info,
                diet,
            });
            self.bindings.insert(*name, Imported::Leaf(value.clone()));
            arguments.push(Imported::Leaf(value));
        }
        let mut inner = Scope {
            parameters,
            ..Scope::default()
        };
        let body = &callback.lam.body;
        let result = if matches!(body.kind, TermKind::Var(VarRef::Symbol(_))) {
            let Imported::Callable(function, closure) = self.visit(body, &mut inner)? else {
                return Err(output_error("callback envelope does not reference a function"));
            };
            arguments.extend(captures);
            self.apply_value(function, arguments, closure, &callback.lam.ret_ty, &mut inner)?
        } else {
            self.visit(body, &mut inner)?
        };
        let body = self.finish(inner, &result)?;
        self.bindings.restore(checkpoint);
        Ok(body)
    }

    fn loop_expression(
        &mut self,
        term: &'source Term,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
            ..
        } = &term.kind
        else {
            return Err(output_error("loop importer received another expression"));
        };
        let initial = Self::leaf(self.visit(init, scope)?)?;
        let carried = self.scalar(loop_var_ty)?;
        let checkpoint = self.bindings.checkpoint();
        let mut inner = Scope::default();
        let mut initial_values = vec![initial.clone()];
        let mut carried_values = vec![carried.clone()];
        let mut induction = None;
        let form = match kind {
            LoopKind::ForRange { var, var_ty, bound } => {
                let bound = self.index(bound, scope)?;
                let index = self.scalar(&types::i32())?;
                let atom = index.name;
                let value = self.primitive(Payload::IndexCast, vec![atom], var_ty, true, &mut inner)?;
                self.bindings.insert(*var, value);
                induction = Some((index, bound));
                Code::CountedLoop
            }
            LoopKind::For { var, var_ty, iter } => {
                let array = Self::leaf(self.visit(iter, scope)?)?;
                let length = self.operation(Code::Length { axis: 0 }, &[Name::Value(array.name)])?;
                let count = Self::leaf(self.emit(length, &types::i32(), scope)?)?;
                let count = self.node("IndexValue", count.name)?;
                let index = self.scalar(&types::i32())?;
                let atom = index.name;
                let item = self.index_operation(array.name, atom)?;
                let value = self.emit(item, var_ty, &mut inner)?;
                self.bindings.insert(*var, value);
                induction = Some((index, count));
                Code::CountedLoop
            }
            LoopKind::While { cond } => {
                self.bindings.insert(*loop_var, Imported::Leaf(initial));
                for (symbol, _, extraction) in init_bindings {
                    let value = self.visit(extraction, scope)?;
                    self.bindings.insert(*symbol, value);
                }
                initial_values.push(Self::leaf(self.visit(cond, scope)?)?);
                let parameter = self.scalar(&types::bool_type())?;
                carried_values.push(parameter.clone());
                Code::WhileLoop
            }
        };
        self.bindings.insert(*loop_var, Imported::Leaf(carried));
        for (symbol, _, extraction) in init_bindings {
            let value = self.visit(extraction, &mut inner)?;
            self.bindings.insert(*symbol, value);
        }
        let next = self.visit(body, &mut inner)?;
        let mut next_values = vec![Self::leaf(next.clone())?];
        if let LoopKind::While { cond } = kind {
            self.bindings.insert(*loop_var, next);
            for (symbol, _, extraction) in init_bindings {
                let value = self.visit(extraction, &mut inner)?;
                self.bindings.insert(*symbol, value);
            }
            next_values.push(Self::leaf(self.visit(cond, &mut inner)?)?);
        }
        self.bindings.restore(checkpoint);
        let observe = self.unit("Observe")?;
        let mut results = Vec::new();
        let mut returns = Vec::new();
        let mut pattern = Vec::new();
        let output = self.scalar(&term.ty)?;
        for (i, (carried, next)) in carried_values.iter().zip(&next_values).enumerate() {
            inner.parameters.push(Parameter {
                name: self.binding(carried),
                info: carried.info,
                diet: observe,
            });
            results.push(self.binding(next));
            returns.push(self.return_info(&next.ty)?);
            let value = if i == 0 { output.clone() } else { self.scalar(&next.ty)? };
            pattern.push(self.binding(&value));
        }
        if let Some((index, _)) = &induction {
            inner.parameters.push(Parameter {
                name: self.binding(index),
                info: index.info,
                diet: observe,
            });
        }
        let body = self.finish_region(inner, &results, &returns)?;
        let operation = self.operation(form, &[])?;
        self.sink.set_control(operation.0, 0_i64, body.0);
        for (index, initial) in initial_values.iter().enumerate() {
            let name = self.binding(initial).fact(&mut self.sink)?;
            self.sink.set_argument(ArgumentSite(operation.0, 0, index), name)?;
        }
        if let Some((_, count)) = induction {
            self.sink.set_formula(operation.0, Formula::LoopCount, count)?;
        }
        self.statement(pattern, operation, scope)?;
        Ok(Imported::Leaf(output))
    }
}
