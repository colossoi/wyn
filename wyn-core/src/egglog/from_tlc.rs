//! Translate typed TLC into logical operations in the common statement IR.
use super::bindings::Bindings;
use super::{fusion, output_error, OptimizeError};
use crate::builtins::{by_id, catalog, BuiltinId, Purity};
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::stage::OwnershipApplied;
use crate::tlc::{
    self, extract_lambda_params_ref, ArrayExpr, DefMeta, LoopKind, SoacBody, SoacOp, TermKind, VarRef,
};
use crate::types::{self, extract_function_signature, strip_existentials, Diet, Type, TypeExt, TypeName};
use crate::SymbolId;
use egglog_engine::sort::{SetContainer, VecContainer};
use egglog_engine::{Core, EGraph, FullState, IntoValues, RawValues, Value, Write};
use wyn_base::{IdSource, Interner};

#[cfg(test)]
#[path = "from_tlc_tests.rs"]
mod tests;

pub(super) type Term = tlc::Term<ExplicitClosurePayload, ExplicitCapturesPayload>;
type Callback = SoacBody<ExplicitClosurePayload, ExplicitCapturesPayload>;
type SourceArray = ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload>;
type Operator = SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>;
pub(super) fn import(
    source: &OwnershipApplied,
) -> Result<(EGraph, Interner<i64, SymbolId>), OptimizeError> {
    let mut graph = fusion::new_graph()?;
    let mut symbols = Interner::default();
    graph.update(|sink| {
        let mut import = Import {
            sink,
            symbols: &mut symbols,
            source,
            names: IdSource::new(),
            statements: IdSource::new(),
            bindings: Bindings::default(),
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
    statements: Vec<Value>,
}

struct Import<'graph, 'db, 'ids, 'source> {
    sink: FullState<'graph, 'db>,
    symbols: &'ids mut Interner<i64, SymbolId>,
    source: &'source OwnershipApplied,
    names: IdSource<i64>,
    statements: IdSource<i64>,
    bindings: Bindings<SymbolId, Imported<'source>>,
}

impl<'source> Import<'_, '_, '_, 'source> {
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

    fn operand(&mut self, leaf: &Leaf) -> Result<Value, OptimizeError> {
        let atom = self.node("Var", leaf.name)?;
        self.node("ValueOperand", atom)
    }

    fn binding(&mut self, leaf: &Leaf) -> Result<Value, OptimizeError> {
        self.node("BindValue", (leaf.name, leaf.info))
    }

    fn statement(
        &mut self,
        pattern: Vec<Value>,
        expression: Value,
        scope: &mut Scope,
    ) -> Result<(), OptimizeError> {
        let number = self.statements.next_id();
        let id = self.node("StatementTag", number)?;
        let pattern = self.vector(pattern);
        let certificates = self.vector(Vec::new());
        scope.statements.push(self.node("Let", (id, pattern, certificates, expression))?);
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
        self.node("Var", value.name)
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
        let primitive = self.primitive_type(ty)?;
        let atom = self.node("LiteralBits", (primitive, bits))?;
        let operand = self.node("ValueOperand", atom)?;
        let expression = self.node("Use", operand)?;
        let leaf = self.scalar(ty)?;
        let binding = self.binding(&leaf)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(Imported::Leaf(leaf))
    }

    fn primitive(
        &mut self,
        opcode: &str,
        atoms: Vec<Value>,
        ty: &Type,
        pure: bool,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let scalar = ty.as_tensor().map_or(ty, |tensor| tensor.elem);
        let pure = pure
            || (matches!(opcode, "/" | "%") && matches!(scalar, Type::Constructed(TypeName::Float(_), _)));
        let data = if pure { Vec::new() } else { vec![self.unit("OrderedEffect")?] };
        let effects = self.sink.container_to_value(SetContainer {
            data: data.into_iter().collect(),
            do_rebuild: true,
        });
        let leaf = self.scalar(ty)?;
        self.sink.set("PrimitiveEffects", (opcode, leaf.info), effects)?;
        if pure {
            self.sink.add("RepeatableOpcode", opcode)?;
        }
        let atoms = self.vector(atoms);
        let expression = self.node("PrimOp", (opcode, atoms))?;
        let binding = self.binding(&leaf)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(Imported::Leaf(leaf))
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
        let (opcode, pure) = match &function.kind {
            TermKind::BinOp(operator) => {
                self.sink.add("RepeatableOpcode", operator.op.symbol())?;
                (operator.op.symbol().to_owned(), operator.op.is_speculatable())
            }
            TermKind::UnOp(operator) => (operator.op.symbol().to_owned(), true),
            TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                if let Some(value) = self.array_builtin(*id, &arguments, ty, scope)? {
                    return Ok(value);
                }
                let builtin = by_id(*id);
                let Some(overload) = builtin.overloads().get(*overload_idx) else {
                    return Err(output_error("invalid builtin overload"));
                };
                let opcode = format!("builtin:{}:{overload_idx}", builtin.dispatch_name());
                if builtin.raw.purity == Purity::Pure && overload.lowering.is_reusable() {
                    self.sink.add("RepeatableOpcode", opcode.as_str())?;
                }
                let known = catalog().known();
                if [
                    known.thread_id,
                    known.thread_id_y,
                    known.thread_id_z,
                    known.local_id,
                    known.workgroup_id,
                    known.num_workgroups,
                ]
                .contains(id)
                {
                    self.sink.add("LaneBuiltin", opcode.as_str())?;
                }
                (
                    opcode,
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
            atoms.push(self.node("Var", leaf.name)?);
        }
        self.primitive(&opcode, atoms, ty, pure, scope)
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

    fn scalar(&mut self, ty: &Type) -> Result<Leaf, OptimizeError> {
        let ty = strip_existentials(ty);
        let info = if ty.is_array()
            || matches!(
                strip_existentials(ty),
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
            ) {
            let ty = self.logical_type(ty)?;
            self.node("LogicalValue", ty)?
        } else if let Some(resource) = self.resource_type(ty)? {
            self.node("ResourceInfo", resource)?
        } else {
            let element = self.element(ty)?;
            self.node("ScalarInfo", element)?
        };
        Ok(Leaf {
            name: self.fresh("ValueId")?,
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
        expression: Value,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let value = self.scalar(ty)?;
        let binding = self.binding(&value)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(Imported::Leaf(value))
    }

    fn finish(&mut self, scope: Scope, value: &Imported<'source>) -> Result<Value, OptimizeError> {
        let value = Self::leaf(value.clone())?;
        let operand = self.operand(&value)?;
        let certificates = self.vector(Vec::new());
        let result = self.node("Yield", (certificates, operand))?;
        let results = self.vector(vec![result]);
        let statements = self.vector(scope.statements);
        self.node("Block", (statements, results))
    }

    fn program(&mut self) -> Result<(), OptimizeError> {
        let mut functions = Vec::new();
        let mut externals = Vec::new();
        let mut entries = Vec::new();
        for definition in &self.source.defs {
            let name = self.function_name(definition.name)?;
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
                let binding = self.binding(&value)?;
                let diet = self.diet(&definition.param_diets.get(i).cloned().unwrap_or_default())?;
                params.push(self.node("Parameter", (binding, diet))?);
                self.bindings.insert(*symbol, Imported::Leaf(value));
            }
            for (i, ty) in external_params.iter().enumerate() {
                let value = self.scalar(ty)?;
                let binding = self.binding(&value)?;
                let diet = self.diet(&definition.param_diets.get(i).cloned().unwrap_or_default())?;
                params.push(self.node("Parameter", (binding, diet))?);
            }
            let params = self.vector(params);
            let result = self.return_info(&result_type)?;
            let returns = self.vector(vec![result]);
            if matches!(&body.kind, TermKind::Extern(_)) {
                let behavior = self.unit("MayTrapCall")?;
                let resources = self.vector(Vec::new());
                externals.push(self.node("DeclareExternal", (name, params, returns, behavior, resources))?);
            } else {
                let mut scope = Scope::default();
                let value = self.visit(body, &mut scope)?;
                let body = self.finish(scope, &value)?;
                functions.push(self.node("DefineFunction", (name, params, returns, body))?);
            }
            self.bindings.restore(checkpoint);
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                let id = self.symbols.intern(&definition.name);
                entries.push(self.node("EntryPoint", (name, id))?);
                if entry.declaration.entry_kind != crate::interface::EntryKind::Compute {
                    self.sink.set("InvocationEntry", name, true)?;
                }
                let grid = if let Some(grid) = entry.declaration.compute_dispatch {
                    let groups = [grid.x, grid.y, grid.z]
                        .into_iter()
                        .map(|n| self.node("IndexConstant", i64::from(n)))
                        .collect::<Result<Vec<_>, _>>()?;
                    let one = self.node("IndexConstant", 1i64)?;
                    let width = self.node("IndexConstant", 128i64)?;
                    let groups = self.vector(groups);
                    let sizes = self.vector(vec![width, one, one]);
                    self.node("Grid3D", (groups, sizes))?
                } else {
                    self.unit("AutomaticGrid")?
                };
                self.sink.set("EntryGrid", name, grid)?;
            }
        }
        let functions = self.vector(functions);
        let externals = self.vector(externals);
        let entries = self.vector(entries);
        let program = self.node("MemoryProgram", (functions, externals, entries))?;
        self.sink.add("CurrentProgram", program)?;
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
        let mut passed = Vec::new();
        for (i, argument) in arguments.into_iter().enumerate() {
            let value = Self::leaf(argument)?;
            let operand = self.operand(&value)?;
            let diet = self.diet(&definition.param_diets.get(i).cloned().unwrap_or_default())?;
            passed.push(self.node("Pass", (operand, diet))?);
        }
        let passed = self.vector(passed);
        let result = self.return_info(ty)?;
        let returns = self.vector(vec![result]);
        let expression = self.node("Call", (name, passed, returns))?;
        self.emit(expression, ty, scope)
    }

    fn visit(
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
                let fields = self.vector(fields);
                let operation = self.node("Product", fields)?;
                self.emit(operation, &term.ty, scope)
            }
            TermKind::TupleProj { tuple, idx } => {
                let value = Self::leaf(self.visit(tuple, scope)?)?;
                let atom = self.node("Var", value.name)?;
                if value.ty.is_vec() {
                    self.primitive(&format!("project:{idx}"), vec![atom], &term.ty, true, scope)
                } else {
                    let operation = self.node("Project", (atom, *idx as i64))?;
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
                let result = self.return_info(&term.ty)?;
                let returns = self.vector(vec![result]);
                let expression = self.node("If", (condition, returns, yes, no))?;
                self.emit(expression, &term.ty, scope)
            }
            TermKind::ArrayExpr(array) => self.array(array, scope),
            TermKind::Index { array, index } => {
                let array = Self::leaf(self.visit(array, scope)?)?;
                let index = self.atom(index, scope)?;
                if array.ty.is_vec() {
                    let array = self.node("Var", array.name)?;
                    self.primitive("vector-index", vec![array, index], &term.ty, true, scope)
                } else {
                    let operation = self.index_operation(array.name, index)?;
                    self.emit(operation, &term.ty, scope)
                }
            }
            TermKind::VecLit(fields) => {
                let atoms =
                    fields.iter().map(|field| self.atom(field, scope)).collect::<Result<Vec<_>, _>>()?;
                self.primitive("vector", atoms, &term.ty, true, scope)
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
    ) -> Result<Value, OptimizeError> {
        let arrays = arrays
            .iter()
            .map(|array| Self::leaf(self.array(array, scope)?).map(|value| value.name))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.vector(arrays))
    }

    fn index_operation(&mut self, source: Value, atom: Value) -> Result<Value, OptimizeError> {
        let indices = self.vector(vec![atom]);
        self.node("Index", (source, indices))
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

    fn iteration(&mut self, input: &Leaf, scope: &mut Scope) -> Result<(Value, Leaf), OptimizeError> {
        let length = self.node("Length", (input.name, 0i64))?;
        let length = Self::leaf(self.emit(length, &types::i32(), scope)?)?;
        let count = self.node("IndexValue", length.name)?;
        let index = self.scalar(&types::i32())?;
        let axis = self.node("DenseAxis", (index.name, count))?;
        let axes = self.vector(vec![axis]);
        let ordinal = self.fresh("ValueId")?;
        Ok((self.node("IterationSpace", (ordinal, axes))?, index))
    }

    fn parallel(
        &mut self,
        domain: Value,
        inner: Scope,
        clauses: Vec<Value>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let statements = self.vector(inner.statements);
        let clauses = self.vector(clauses);
        let body = self.node("ParallelBlock", (statements, clauses))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        self.emit(expression, ty, scope)
    }

    fn result(&mut self, value: &Imported<'source>) -> Result<Value, OptimizeError> {
        let value = Self::leaf(value.clone())?;
        let operand = self.operand(&value)?;
        let certs = self.vector(Vec::new());
        self.node("Yield", (certs, operand))
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
                let elements = self.vector(elements);
                let generator = self.node("ArrayLiteral", elements)?;
                let expression = self.node("GenerateArray", generator)?;
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
                let position = self.node("Var", index.name)?;
                let mut inner = Scope::default();
                let mut fields = Vec::new();
                let mut types = Vec::new();
                for source in values {
                    let row_type = Self::row_type(&source.ty)?;
                    let expression = self.index_operation(source.name, position)?;
                    let value = Self::leaf(self.emit(expression, &row_type, &mut inner)?)?;
                    fields.push(self.node("Var", value.name)?);
                    types.push(row_type);
                }
                let fields = self.vector(fields);
                let expression = self.node("Product", fields)?;
                let value = self.emit(expression, &types::tuple(types), &mut inner)?;
                let result = self.result(&value)?;
                let results = self.vector(vec![result]);
                let clause = self.node("Collect", results)?;
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
                    let primitive = self.primitive_type(&start.ty)?;
                    self.node("LiteralBits", (primitive, bits))?
                };
                let generator = self.node("RangeValues", (start_atom, length, step))?;
                let expression = self.node("GenerateArray", generator)?;
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
            self.node("Length", (source.name, 0i64))?
        } else if id == known.scratch_annotation {
            self.node("Scratch", source.name)?
        } else if id == known.scratch_alloc {
            let count = self.node("IndexValue", source.name)?;
            let shape = self.vector(vec![count]);
            let generator = self.node("UninitializedShape", shape)?;
            self.node("GenerateArray", generator)?
        } else {
            let Some(index) = values.get(1) else {
                return Err(output_error("array index missing"));
            };
            let index_atom = self.node("Var", index.name)?;
            if id == known.storage_index {
                self.index_operation(source.name, index_atom)?
            } else {
                let index = self.node("IndexValue", index.name)?;
                let Some(value) = values.get(2) else {
                    return Err(output_error("array final operand missing"));
                };
                if id == known.slice {
                    let end = self.node("IndexValue", value.name)?;
                    let count = self.node("IndexSub", (end, index))?;
                    let one = self.node("IndexConstant", 1i64)?;
                    let slice = self.node("Slice", (index, count, one))?;
                    let slices = self.vector(vec![slice]);
                    self.node("View", (source.name, slices))?
                } else {
                    let destination = if id == known.array_with
                        && !source.ty.array_variant().is_some_and(types::is_array_variant_view)
                    {
                        let copy = self.node("Copy", source.name)?;
                        Self::leaf(self.emit(copy, &source.ty, scope)?)?.name
                    } else {
                        source.name
                    };
                    let value = self.node("Var", value.name)?;
                    let slice = self.node("At", index)?;
                    let slices = self.vector(vec![slice]);
                    let bounds = self.unit("IgnoreOutOfBounds")?;
                    self.node("Update", (bounds, destination, slices, value))?
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
                let position = self.node("Var", index.name)?;
                let mut inner = Scope::default();
                let mut arguments = Vec::new();
                for input in inputs {
                    let expression = self.index_operation(input.name, position)?;
                    arguments.push(self.emit(expression, &Self::row_type(&input.ty)?, &mut inner)?);
                }
                let value = self.apply_callback(lam, arguments, &mut inner)?;
                let result = self.result(&value)?;
                let results = self.vector(vec![result]);
                let clause = self.node("Collect", results)?;
                self.parallel(domain, inner, vec![clause], ty, scope)
            }
            SoacOp::Reduce { op, ne, input } | SoacOp::Scan { op, ne, input } => {
                let input = Self::leaf(self.array(input, scope)?)?;
                let neutral = self.atom(ne, scope)?;
                let neutral = self.vector(vec![neutral]);
                let combine = self.callback(op, scope)?;
                let order = self.unit("Ordered")?;
                let reducer = self.node("Reduction", (order, neutral, combine))?;
                let (domain, index) = self.iteration(&input, scope)?;
                let position = self.node("Var", index.name)?;
                let mut inner = Scope::default();
                let expression = self.index_operation(input.name, position)?;
                let value = self.emit(expression, &Self::row_type(&input.ty)?, &mut inner)?;
                let result = self.result(&value)?;
                let results = self.vector(vec![result]);
                let kind = self.unit(if matches!(operator, SoacOp::Scan { .. }) {
                    "InclusiveScan"
                } else {
                    "Total"
                })?;
                let clause = self.node("Fold", (kind, reducer, results))?;
                self.parallel(domain, inner, vec![clause], ty, scope)
            }
            SoacOp::Filter { pred, input } => {
                let input = Self::leaf(self.array(input, scope)?)?;
                let (domain, index) = self.iteration(&input, scope)?;
                let position = self.node("Var", index.name)?;
                let mut inner = Scope::default();
                let expression = self.index_operation(input.name, position)?;
                let value = self.emit(expression, &Self::row_type(&input.ty)?, &mut inner)?;
                let condition = Self::leaf(self.apply_callback(pred, vec![value.clone()], &mut inner)?)?;
                let condition = self.node("Var", condition.name)?;
                let result = self.result(&value)?;
                let results = self.vector(vec![result]);
                let clause = self.node("Compact", (condition, results))?;
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
                let inputs = self.array_names(inputs, scope)?;
                let callback = self.callback(lam, scope)?;
                let coordinates = input_dimensions
                    .iter()
                    .map(|axes| {
                        let values =
                            axes.iter().map(|&axis| self.sink.base_to_value(i64::from(axis))).collect();
                        self.vector(values)
                    })
                    .collect();
                let coordinates = self.vector(coordinates);
                let expression = self.node(
                    "BucketScatter",
                    (
                        destination.name,
                        callback,
                        inputs,
                        coordinates,
                        i64::from(*domain_rank),
                    ),
                )?;
                self.emit(expression, ty, scope)
            }
        }
    }

    fn accumulator(
        &mut self,
        destination: &Leaf,
        policy: Value,
        inputs: &mut Vec<Value>,
        parameters: &mut Vec<Value>,
    ) -> Result<(Value, Value), OptimizeError> {
        let ty = self.logical_type(&destination.ty)?;
        let info = self.node("LogicalAccumulator", ty)?;
        let token = self.fresh("AccumulatorId")?;
        let binding = self.node("BindAccumulator", (token, info))?;
        let consume = self.unit("Consume")?;
        parameters.push(self.node("Parameter", (binding, consume))?);
        let arrays = self.vector(vec![destination.name]);
        inputs.push(self.node("AccumulatorInput", (arrays, policy))?);
        Ok((token, info))
    }

    fn contribute(
        &mut self,
        tokens: Vec<(Value, Value)>,
        domain: Value,
        lane: Scope,
        inputs: Vec<Value>,
        parameters: Vec<Value>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let names = self.vector(tokens.iter().map(|(name, _)| *name).collect());
        let clause = self.node("Contribute", names)?;
        let clauses = self.vector(vec![clause]);
        let statements = self.vector(lane.statements);
        let body = self.node("ParallelBlock", (statements, clauses))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        let mut pattern = Vec::new();
        let mut returns = Vec::new();
        let mut results = Vec::new();
        let certs = self.vector(Vec::new());
        for (_, info) in tokens {
            let name = self.fresh("AccumulatorId")?;
            pattern.push(self.node("BindAccumulator", (name, info))?);
            returns.push(self.node("ReturnAccumulator", info)?);
            let operand = self.node("AccOperand", name)?;
            results.push(self.node("Yield", (certs, operand))?);
        }
        let mut body = Scope::default();
        self.statement(pattern, expression, &mut body)?;
        let statements = self.vector(body.statements);
        let results = self.vector(results);
        let body = self.node("Block", (statements, results))?;
        let parameters = self.vector(parameters);
        let returns = self.vector(returns);
        let lambda = self.node("FunctionBody", (parameters, returns, body))?;
        let inputs = self.vector(inputs);
        let expression = self.node("WithAcc", (inputs, lambda))?;
        self.emit(expression, ty, scope)
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
        let values = self.vector(vec![value]);
        let bounds = self.unit("IgnoreOutOfBounds")?;
        let expression = self.node("Accumulate", (bounds, token, indices, values))?;
        let next = self.fresh("AccumulatorId")?;
        let binding = self.node("BindAccumulator", (next, info))?;
        self.statement(vec![binding], expression, scope)?;
        Ok((next, info))
    }

    fn project(
        &mut self,
        value: &Leaf,
        index: usize,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Leaf, OptimizeError> {
        let atom = self.node("Var", value.name)?;
        let expression = self.node("Project", (atom, index as i64))?;
        Self::leaf(self.emit(expression, ty, scope)?)
    }

    fn scatter_groups(
        &mut self,
        destination: &Leaf,
        contribution: &Leaf,
        policy: Value,
        inputs: &mut Vec<Value>,
        parameters: &mut Vec<Value>,
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
                let (token, info) = self.accumulator(destination, policy, inputs, parameters)?;
                let index = self.project(contribution, 0, index_ty, lane)?;
                let value = self.project(contribution, 1, value_ty, lane)?;
                let value = self.node("Var", value.name)?;
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
            let copy = self.node("Copy", destination.name)?;
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
        let position = self.node("Var", index.name)?;
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
            let copy = self.node("Copy", destination.name)?;
            Self::leaf(self.emit(copy, &destination.ty, scope)?)?
        };
        self.visit(neutral, scope)?;
        let indices = Self::leaf(self.array(indices, scope)?)?;
        let values = Self::leaf(self.array(values, scope)?)?;
        let combine = self.callback(callback, scope)?;
        let policy = self.node("CombineWrites", combine)?;
        let mut inputs = Vec::new();
        let mut parameters = Vec::new();
        let (token, info) = self.accumulator(&destination, policy, &mut inputs, &mut parameters)?;
        let (domain, index) = self.iteration(&indices, scope)?;
        let position = self.node("Var", index.name)?;
        let mut lane = Scope::default();
        let expression = self.index_operation(indices.name, position)?;
        let index = Self::leaf(self.emit(expression, &Self::row_type(&indices.ty)?, &mut lane)?)?;
        let expression = self.index_operation(values.name, position)?;
        let value = Self::leaf(self.emit(expression, &Self::row_type(&values.ty)?, &mut lane)?)?;
        let value = self.node("Var", value.name)?;
        let token = self.accumulate(token, info, index.name, value, &mut lane)?;
        self.contribute(vec![token], domain, lane, inputs, parameters, ty, scope)
    }

    fn callback(&mut self, callback: &'source Callback, scope: &mut Scope) -> Result<Value, OptimizeError> {
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
            let binding = self.binding(&value)?;
            let diet = self.unit("Observe")?;
            parameters.push(self.node("Parameter", (binding, diet))?);
            self.bindings.insert(*name, Imported::Leaf(value.clone()));
            arguments.push(Imported::Leaf(value));
        }
        let mut inner = Scope::default();
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
        let result = self.return_info(&callback.lam.ret_ty)?;
        let returns = self.vector(vec![result]);
        let parameters = self.vector(parameters);
        self.node("FunctionBody", (parameters, returns, body))
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
        let zero = self.node("IndexConstant", 0i64)?;
        let one = self.node("IndexConstant", 1i64)?;
        let form = match kind {
            LoopKind::ForRange { var, var_ty, bound } => {
                let bound = self.index(bound, scope)?;
                let index = self.scalar(&types::i32())?;
                let atom = self.node("Var", index.name)?;
                let value = self.primitive("index-cast", vec![atom], var_ty, true, &mut inner)?;
                self.bindings.insert(*var, value);
                self.node("Counted", (index.name, zero, bound, one))?
            }
            LoopKind::For { var, var_ty, iter } => {
                let array = Self::leaf(self.visit(iter, scope)?)?;
                let length = self.node("Length", (array.name, 0i64))?;
                let count = Self::leaf(self.emit(length, &types::i32(), scope)?)?;
                let count = self.node("IndexValue", count.name)?;
                let index = self.scalar(&types::i32())?;
                let atom = self.node("Var", index.name)?;
                let item = self.index_operation(array.name, atom)?;
                let value = self.emit(item, var_ty, &mut inner)?;
                self.bindings.insert(*var, value);
                self.node("Counted", (index.name, zero, count, one))?
            }
            LoopKind::While { cond } => {
                self.bindings.insert(*loop_var, Imported::Leaf(initial));
                for (symbol, _, extraction) in init_bindings {
                    let value = self.visit(extraction, scope)?;
                    self.bindings.insert(*symbol, value);
                }
                initial_values.push(Self::leaf(self.visit(cond, scope)?)?);
                let condition = self.scalar(&types::bool_type())?;
                carried_values.push(condition.clone());
                self.node("While", condition.name)?
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
        let mut merges = Vec::new();
        let mut results = Vec::new();
        let mut returns = Vec::new();
        let mut pattern = Vec::new();
        let output = self.scalar(&term.ty)?;
        for (i, ((initial, carried), next)) in
            initial_values.iter().zip(&carried_values).zip(&next_values).enumerate()
        {
            let binding = self.binding(carried)?;
            let parameter = self.node("Parameter", (binding, observe))?;
            let operand = self.operand(initial)?;
            merges.push(self.node("Carry", (parameter, operand))?);
            let operand = self.operand(next)?;
            let certs = self.vector(Vec::new());
            results.push(self.node("Yield", (certs, operand))?);
            returns.push(self.return_info(&next.ty)?);
            let value = if i == 0 { output.clone() } else { self.scalar(&next.ty)? };
            pattern.push(self.binding(&value)?);
        }
        let statements = self.vector(inner.statements);
        let results = self.vector(results);
        let body = self.node("Block", (statements, results))?;
        let merges = self.vector(merges);
        let returns = self.vector(returns);
        let expression = self.node("Loop", (merges, form, returns, body))?;
        self.statement(pattern, expression, scope)?;
        Ok(Imported::Leaf(output))
    }
}
