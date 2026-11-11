//! Construct the memory IR directly from typed TLC. Import-local values are
//! discarded with the builder; the graph owns every executable statement.
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

// These records hold construction operands, never graph queries or pass results.
#[derive(Clone)]
struct Index {
    expression: Value,
    atom: Value,
}

#[derive(Clone)]
struct Array {
    dimensions: Vec<Index>,
    element: Value,
}

#[derive(Clone)]
struct Leaf {
    name: Value,
    info: Value,
    ty: Type,
    array: Option<Array>,
}

#[derive(Clone)]
enum Imported<'source> {
    Leaf(Leaf),
    Product(Vec<Imported<'source>>),
    Ragged {
        offsets: Leaf,
        data: Box<Imported<'source>>,
    },
    Callable(&'source Term, Vec<Imported<'source>>),
}

// Products distribute through every array, including ABI-backed arrays. Public
// buffer packing is retained by entry metadata, not by the computation types.
fn component_type(ty: &Type) -> Type {
    let ty = strip_existentials(ty);
    match ty {
        Type::Constructed(TypeName::Array, arguments) => {
            let Some(element) = arguments.first() else {
                return ty.clone();
            };
            let element = component_type(element);
            let array_of = |element| {
                let mut arguments = arguments.clone();
                arguments[0] = element;
                Type::Constructed(TypeName::Array, arguments)
            };
            match element {
                Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) => {
                    Type::Constructed(
                        name,
                        fields.into_iter().map(|field| component_type(&array_of(field))).collect(),
                    )
                }
                element => array_of(element),
            }
        }
        Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) => {
            Type::Constructed(name.clone(), fields.iter().map(component_type).collect())
        }
        _ => ty.clone(),
    }
}

// Offset arrays and payloads are ordinary IR values. This decomposition exists
// only while translating source types; no ragged descriptor survives import.
pub(super) fn ragged_parts(ty: &Type) -> Option<(Type, Type)> {
    let element = strip_existentials(ty).elem_type()?;
    if !element.is_array() {
        return None;
    }
    let mut inner = element;
    let mut irregular = false;
    while inner.is_array() {
        for axis in 0..inner.array_rank().unwrap_or(1) {
            irregular |= !matches!(
                inner.array_dim(axis),
                Some(Type::Constructed(TypeName::Size(_), _))
            );
        }
        inner = inner.elem_type()?;
    }
    if !irregular {
        return None;
    }
    let offsets = types::view_array_with_size(
        &types::i32(),
        Type::Constructed(TypeName::SizePlaceholder, vec![]),
        types::no_buffer(),
    );
    let Type::Constructed(TypeName::Array, mut payload) = element.clone() else {
        return None;
    };
    payload[2] = Type::Constructed(TypeName::SizePlaceholder, vec![]);
    Some((offsets, Type::Constructed(TypeName::Array, payload)))
}

fn contains_ragged(ty: &Type) -> bool {
    ragged_parts(ty).is_some()
        || matches!(ty, Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) if fields.iter().any(contains_ragged))
}

// A growable component is ordinary loop state: used rows and a capacity-sized
// array. Ragged values recursively carry the offset and payload components.
fn growing_type(ty: &Type) -> Type {
    if let Some((offsets, data)) = ragged_parts(ty) {
        return types::tuple(vec![growing_type(&offsets), growing_type(&data)]);
    }
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
        return types::tuple(fields.iter().map(growing_type).collect());
    }
    types::tuple(vec![types::i32(), ty.clone()])
}

impl<'source> Imported<'source> {
    fn build(
        ty: &Type,
        operation: &mut impl FnMut(&Type) -> Result<Leaf, OptimizeError>,
    ) -> Result<Self, OptimizeError> {
        if let Some((offsets, data)) = ragged_parts(ty) {
            return Ok(Self::Ragged {
                offsets: operation(&offsets)?,
                data: Box::new(Self::build(&data, operation)?),
            });
        }
        match ty {
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields
                .iter()
                .map(|field| Self::build(field, operation))
                .collect::<Result<_, _>>()
                .map(Self::Product),
            ty => operation(ty).map(Self::Leaf),
        }
    }

    fn map(
        self,
        operation: &mut impl FnMut(Leaf) -> Result<Leaf, OptimizeError>,
    ) -> Result<Self, OptimizeError> {
        match self {
            Self::Leaf(leaf) => operation(leaf).map(Self::Leaf),
            Self::Ragged { offsets, data } => Ok(Self::Ragged {
                offsets: operation(offsets)?,
                data: Box::new(data.map(operation)?),
            }),
            Self::Product(fields) => fields
                .into_iter()
                .map(|field| field.map(operation))
                .collect::<Result<_, _>>()
                .map(Self::Product),
            Self::Callable(_, _) => Err(output_error("callable used as a data component")),
        }
    }

    fn has_segments(&self) -> bool {
        match self {
            Self::Ragged { .. } => true,
            Self::Product(fields) => fields.iter().any(Self::has_segments),
            _ => false,
        }
    }

    fn leaves(&self) -> Result<Vec<&Leaf>, OptimizeError> {
        Ok(self.diet_leaves(&Diet::observing())?.into_iter().map(|(leaf, _)| leaf).collect())
    }

    fn diet_leaves(&self, diet: &Diet) -> Result<Vec<(&Leaf, bool)>, OptimizeError> {
        match self {
            Self::Leaf(leaf) => Ok(vec![(leaf, diet.is_consuming_at_root())]),
            Self::Ragged { offsets, data } => {
                let mut leaves = vec![(offsets, false)];
                leaves.extend(data.diet_leaves(diet)?);
                Ok(leaves)
            }
            Self::Product(fields) => {
                fields.iter().enumerate().try_fold(Vec::new(), |mut leaves, (i, field)| {
                    let component =
                        if diet.is_consuming_at_root() { Diet::Leaf(true) } else { diet.component(i) };
                    leaves.extend(field.diet_leaves(&component)?);
                    Ok(leaves)
                })
            }
            Self::Callable(_, _) => Err(output_error("callable used as a data component")),
        }
    }
}

#[derive(Clone)]
struct Accumulator {
    name: Value,
    info: Value,
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

    fn program(&mut self) -> Result<(), OptimizeError> {
        let mut functions = Vec::new();
        let mut externals = Vec::new();
        let mut entries = Vec::new();
        for definition in &self.source.defs {
            let name = self.function_name(definition.name)?;
            let (body, parameters) = extract_lambda_params_ref(&definition.body);
            let checkpoint = self.bindings.checkpoint();
            let mut params = Vec::new();
            let mut scope = Scope::default();
            let entry = matches!(definition.meta, DefMeta::EntryPoint(_));
            for (i, (symbol, ty)) in parameters.iter().enumerate() {
                let diet = definition.param_diets.get(i).cloned().unwrap_or_default();
                let value = self.parameter_with_diet_mode(ty, &diet, &mut params, !entry)?;
                let value = if entry { self.unpack_boundary(value, ty, &mut scope)? } else { value };
                self.bindings.insert(*symbol, value);
            }
            let result_type = if matches!(&body.kind, TermKind::Extern(_)) {
                let (parameters, result) = extract_function_signature(&body.ty);
                for (i, ty) in parameters.iter().enumerate() {
                    let diet = definition.param_diets.get(i).cloned().unwrap_or_default();
                    self.parameter_with_diet(ty, &diet, &mut params)?;
                }
                result
            } else {
                body.ty.clone()
            };
            let params = self.vector(params);
            let mut pattern = Vec::new();
            let mut returns = Vec::new();
            let output = self.value_pattern(
                if entry { result_type.clone() } else { component_type(&result_type) },
                &mut pattern,
                &mut returns,
            )?;
            let returns = self.vector(returns);
            if matches!(&body.kind, TermKind::Extern(_)) {
                let behavior = self.unit("MayTrapCall")?;
                let resources = self.vector(Vec::new());
                externals.push(self.node("DeclareExternal", (name, params, returns, behavior, resources))?);
            } else {
                let value = self.visit(body, &mut scope)?;
                let value =
                    if entry { self.pack_boundary(value, &result_type, &mut scope)? } else { value };
                let body = self.finish(scope, &value)?;
                functions.push(self.node("DefineFunction", (name, params, returns, body))?);
            }
            // Result patterns describe the caller's binders; only return slots
            // belong to this definition's signature.
            drop(output);
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

    fn resolve(&mut self, symbol: SymbolId) -> Result<Imported<'source>, OptimizeError> {
        if let Some(value) = self.bindings.get(&symbol) {
            return Ok(value.clone());
        }
        let Some(definition) = self.source.defs.iter().find(|definition| definition.name == symbol) else {
            return Err(output_error(format!("unbound source symbol {symbol:?}")));
        };
        Ok(Imported::Callable(&definition.body, Vec::new()))
    }

    fn finish(&mut self, scope: Scope, value: &Imported<'source>) -> Result<Value, OptimizeError> {
        let operands = self.pack(value)?;
        let mut results = Vec::with_capacity(operands.len());
        for operand in operands {
            let certificates = self.vector(Vec::new());
            results.push(self.node("Yield", (certificates, operand))?);
        }
        let statements = self.vector(scope.statements);
        let results = self.vector(results);
        self.node("Block", (statements, results))
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
            TermKind::Tuple(fields) => Ok(Imported::Product(
                fields.iter().map(|field| self.visit(field, scope)).collect::<Result<_, _>>()?,
            )),
            TermKind::TupleProj { tuple, idx } => match self.visit(tuple, scope)? {
                Imported::Product(fields) => {
                    let Some(field) = fields.into_iter().nth(*idx) else {
                        return Err(output_error("product projection is out of bounds"));
                    };
                    Ok(field)
                }
                Imported::Leaf(value) if value.ty.vec_size().is_some_and(|size| *idx < size) => {
                    let atom = self.node("Var", value.name)?;
                    self.primitive(&format!("project:{idx}"), vec![atom], &term.ty, true, scope)
                }
                _ => Err(output_error(
                    "projection has no matching product or vector component",
                )),
            },
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
                let mut pattern = Vec::new();
                let mut returns = Vec::new();
                let result = self.result_pattern(&term.ty, &mut pattern, &mut returns)?;
                let returns = self.vector(returns);
                let expression = self.node("If", (condition, returns, yes, no))?;
                self.statement(pattern, expression, scope)?;
                Ok(result)
            }
            TermKind::ArrayExpr(array) => self.array(array, scope),
            TermKind::Index { array, index } => {
                let array = self.visit(array, scope)?;
                let index = self.index(index, scope)?;
                self.input_at(&array, &index, scope)
            }
            TermKind::VecLit(fields) => {
                let atoms = fields.iter().map(|field| self.atom(field, scope)).collect::<Result<_, _>>()?;
                self.primitive("vector", atoms, &term.ty, true, scope)
            }
            TermKind::Soac(operator) => self.soac(operator, &term.ty, scope),
            TermKind::Loop { .. } => self.loop_expression(term, scope),
            TermKind::Extern(_) => Err(output_error("external declaration in an expression")),
        }
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

    fn scalar(&mut self, ty: &Type) -> Result<Leaf, OptimizeError> {
        let info = if let Some(resource) = self.resource_type(ty)? {
            self.node("ResourceInfo", resource)?
        } else {
            let element = self.element(ty)?;
            self.node("ScalarInfo", element)?
        };
        let name = self.fresh("ValueId")?;
        Ok(Leaf {
            name,
            info,
            ty: ty.clone(),
            array: None,
        })
    }

    fn array_leaf(&mut self, ty: &Type, array: Array) -> Result<Leaf, OptimizeError> {
        let shape = self.vector(array.dimensions.iter().map(|length| length.expression).collect());
        let info = self.node("ArrayValue", (array.element, shape))?;
        let name = self.fresh("ValueId")?;
        Ok(Leaf {
            name,
            info,
            ty: ty.clone(),
            array: Some(array),
        })
    }

    fn index_leaf(&mut self, leaf: &Leaf) -> Result<Index, OptimizeError> {
        Ok(Index {
            expression: self.node("IndexValue", leaf.name)?,
            atom: self.node("Var", leaf.name)?,
        })
    }

    fn constant(&mut self, number: i64) -> Result<Index, OptimizeError> {
        let ty = self.unit("I32")?;
        Ok(Index {
            expression: self.node("IndexConstant", number)?,
            atom: self.node("LiteralBits", (ty, number))?,
        })
    }

    fn index_binding(
        &mut self,
        pattern: &mut Vec<Value>,
        returns: &mut Vec<Value>,
    ) -> Result<Index, OptimizeError> {
        let ty = Type::Constructed(TypeName::Int(32), Vec::new());
        let leaf = Self::leaf(self.result_pattern(&ty, pattern, returns)?)?;
        self.index_leaf(&leaf)
    }

    fn parameter(
        &mut self,
        ty: &Type,
        params: &mut Vec<Value>,
    ) -> Result<Imported<'source>, OptimizeError> {
        self.parameter_with_diet(ty, &Diet::observing(), params)
    }

    fn parameter_with_diet(
        &mut self,
        ty: &Type,
        diet: &Diet,
        params: &mut Vec<Value>,
    ) -> Result<Imported<'source>, OptimizeError> {
        self.parameter_with_diet_mode(ty, diet, params, true)
    }

    fn parameter_with_diet_mode(
        &mut self,
        ty: &Type,
        diet: &Diet,
        params: &mut Vec<Value>,
        normalize: bool,
    ) -> Result<Imported<'source>, OptimizeError> {
        let mut bindings = Vec::new();
        let value = self.value_pattern(
            if normalize { component_type(ty) } else { ty.clone() },
            &mut bindings,
            &mut Vec::new(),
        )?;
        let mut bindings = bindings.into_iter();
        let observe = self.unit("Observe")?;
        for (leaf, consuming) in value.diet_leaves(diet)? {
            for _ in leaf.array.iter().flat_map(|array| &array.dimensions) {
                let Some(binding) = bindings.next() else {
                    return Err(output_error("array shape parameter missing"));
                };
                params.push(self.node("Parameter", (binding, observe))?);
            }
            let Some(binding) = bindings.next() else {
                return Err(output_error("value parameter missing"));
            };
            let diet = if consuming { self.unit("Consume")? } else { observe };
            params.push(self.node("Parameter", (binding, diet))?);
        }
        Ok(value)
    }

    fn result_pattern(
        &mut self,
        ty: &Type,
        pattern: &mut Vec<Value>,
        returns: &mut Vec<Value>,
    ) -> Result<Imported<'source>, OptimizeError> {
        self.value_pattern(component_type(ty), pattern, returns)
    }

    fn value_pattern(
        &mut self,
        ty: Type,
        pattern: &mut Vec<Value>,
        returns: &mut Vec<Value>,
    ) -> Result<Imported<'source>, OptimizeError> {
        Imported::build(strip_existentials(&ty), &mut |ty| {
            let leaf = if ty.is_array() {
                let mut element_ty = ty;
                let mut sizes = Vec::new();
                while element_ty.is_array() {
                    for axis in 0..element_ty.array_rank().unwrap_or(1) {
                        sizes.push(element_ty.array_dim(axis));
                    }
                    let Some(element) = element_ty.elem_type() else {
                        return Err(output_error("array element type missing"));
                    };
                    element_ty = element;
                }
                let element = self.element(element_ty)?;
                let mut dimensions = Vec::with_capacity(sizes.len());
                let mut shape = Vec::with_capacity(sizes.len());
                for size in sizes {
                    let slot = returns.len() as i64;
                    let witness = self.index_binding(pattern, returns)?;
                    if let Some(Type::Constructed(TypeName::Size(size), _)) = size {
                        let size =
                            i64::try_from(*size).map_err(|_| output_error("array extent exceeds i64"))?;
                        let index = self.constant(size)?;
                        shape.push(self.node("FreeIndex", index.expression)?);
                        dimensions.push(index);
                    } else {
                        shape.push(self.node("ResultIndex", slot)?);
                        dimensions.push(witness);
                    }
                }
                let shape = self.vector(shape);
                returns.push(self.node("ReturnArrayValue", (element, shape))?);
                self.array_leaf(ty, Array { dimensions, element })?
            } else {
                let leaf = self.scalar(ty)?;
                if let Some(resource) = self.resource_type(ty)? {
                    returns.push(self.node("ReturnResource", resource)?);
                } else {
                    let element = self.element(ty)?;
                    returns.push(self.node("ReturnScalar", element)?);
                }
                leaf
            };
            pattern.push(self.binding(&leaf)?);
            Ok(leaf)
        })
    }

    fn unpack_boundary(
        &mut self,
        value: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if let Some((_, payload)) = ragged_parts(ty) {
            let Imported::Ragged { offsets, data } = value else {
                return Err(output_error("ragged input components missing"));
            };
            let data = self.unpack_boundary(*data, &payload, scope)?;
            return Ok(Self::distribute_offsets(offsets, data));
        }
        match (value, strip_existentials(ty)) {
            (
                Imported::Product(values),
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types),
            ) => Ok(Imported::Product(
                values
                    .into_iter()
                    .zip(types)
                    .map(|(value, ty)| self.unpack_boundary(value, ty, scope))
                    .collect::<Result<_, _>>()?,
            )),
            (Imported::Leaf(value), _) if ty.is_array() && component_type(ty) != *ty => {
                self.boundary_map(Imported::Leaf(value), ty, true, scope)
            }
            (value, _) => Ok(value),
        }
    }

    fn pack_boundary(
        &mut self,
        value: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if let Some((_, payload)) = ragged_parts(ty) {
            let (offsets, data) = self.gather_offsets(value, scope)?;
            let data = Box::new(self.pack_boundary(data, &payload, scope)?);
            return Ok(Imported::Ragged { offsets, data });
        }
        match strip_existentials(ty) {
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) => {
                let Imported::Product(values) = value else {
                    return Err(output_error("entry product result missing"));
                };
                Ok(Imported::Product(
                    values
                        .into_iter()
                        .zip(types)
                        .map(|(value, ty)| self.pack_boundary(value, ty, scope))
                        .collect::<Result<_, _>>()?,
                ))
            }
            _ if ty.is_array() && component_type(ty) != *ty => self.boundary_map(value, ty, false, scope),
            _ => Ok(value),
        }
    }

    fn distribute_offsets(offsets: Leaf, data: Imported<'source>) -> Imported<'source> {
        match data {
            Imported::Product(fields) => Imported::Product(
                fields.into_iter().map(|field| Self::distribute_offsets(offsets.clone(), field)).collect(),
            ),
            data => Imported::Ragged {
                offsets,
                data: Box::new(data),
            },
        }
    }

    fn gather_offsets(
        &mut self,
        value: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<(Leaf, Imported<'source>), OptimizeError> {
        match value {
            Imported::Ragged { offsets, data } => Ok((offsets, *data)),
            Imported::Product(fields) => {
                let mut fields = fields.into_iter();
                let Some(first) = fields.next() else {
                    return Err(output_error("empty ragged product"));
                };
                let (offsets, first) = self.gather_offsets(first, scope)?;
                let mut data = vec![first];
                for field in fields {
                    let (other_offsets, field) = self.gather_offsets(field, scope)?;
                    self.require_equal_offsets(&offsets, &other_offsets, scope)?;
                    data.push(field);
                }
                Ok((offsets, Imported::Product(data)))
            }
            _ => Err(output_error("ragged output components missing")),
        }
    }

    fn require_equal_offsets(
        &mut self,
        left: &Leaf,
        right: &Leaf,
        scope: &mut Scope,
    ) -> Result<(), OptimizeError> {
        if left.name == right.name {
            return Ok(());
        }
        let a = self.input_length(&Imported::Leaf(left.clone()), scope)?;
        let b = self.input_length(&Imported::Leaf(right.clone()), scope)?;
        let equal = self.predicate("==", a.atom, b.atom, scope)?;
        let assertion = self.node("Assert", (equal, "ragged product lengths differ"))?;
        self.statement(Vec::new(), assertion, scope)?;
        let initial = self.literal(&types::unit(), 0, scope)?;
        self.counted_state(&a, initial, &types::unit(), scope, |this, index, state, inner| {
            let a = Self::leaf(this.index_array(left, index.clone(), &types::i32(), inner)?)?;
            let b = Self::leaf(this.index_array(right, index, &types::i32(), inner)?)?;
            let a = this.node("Var", a.name)?;
            let b = this.node("Var", b.name)?;
            let equal = this.predicate("==", a, b, inner)?;
            let assertion = this.node("Assert", (equal, "ragged product row shapes differ"))?;
            this.statement(Vec::new(), assertion, inner)?;
            Ok(state)
        })?;
        Ok(())
    }

    fn boundary_map(
        &mut self,
        value: Imported<'source>,
        ty: &Type,
        unpack: bool,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let length = self.input_length(&value, scope)?;
        let index = self.scalar(&Type::Constructed(TypeName::Int(32), Vec::new()))?;
        let position = self.index_leaf(&index)?;
        let axis = self.node("DenseAxis", (index.name, length.expression))?;
        let axes = self.vector(vec![axis]);
        let flat = self.fresh("ValueId")?;
        let domain = self.node("IterationSpace", (flat, axes))?;
        let Some(element_ty) = ty.elem_type() else {
            return Err(output_error("boundary map element missing"));
        };
        let mut inner = Scope::default();
        let item = self.input_at(&value, &position, &mut inner)?;
        let item = if unpack {
            let item = self.unpack_element(item, element_ty, &mut inner)?;
            self.unpack_boundary(item, element_ty, &mut inner)?
        } else {
            let item = self.pack_boundary(item, element_ty, &mut inner)?;
            self.pack_element(item, element_ty, &mut inner)?
        };
        let results = self.lane_results(&item)?;
        let results = self.vector(results);
        let collect = self.node("Collect", results)?;
        let clauses = self.vector(vec![collect]);
        let statements = self.vector(inner.statements);
        let body = self.node("ParallelBlock", (statements, clauses))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        let output_ty = if unpack { component_type(ty) } else { ty.clone() };
        let mut pattern = Vec::new();
        let output = Imported::build(&output_ty, &mut |ty| {
            let mut lengths = vec![length.clone()];
            let Some(mut element) = ty.elem_type() else {
                return Err(output_error("boundary array element missing"));
            };
            while element.is_array() {
                let Some(Type::Constructed(TypeName::Size(n), _)) = element.array_size() else {
                    return Err(output_error(
                        "nested boundary array requires its rectangular shape",
                    ));
                };
                lengths.push(self.constant(*n as i64)?);
                let Some(next) = element.elem_type() else {
                    return Err(output_error("boundary array element missing"));
                };
                element = next;
            }
            let array = self.array_value(ty, lengths)?;
            pattern.push(self.binding(&array)?);
            Ok(array)
        })?;
        self.statement(pattern, expression, scope)?;
        Ok(output)
    }

    fn unpack_element(
        &mut self,
        value: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = strip_existentials(ty)
        else {
            return Ok(value);
        };
        let value = Self::leaf(value)?;
        let atom = self.node("Var", value.name)?;
        let mut fields = Vec::new();
        for (i, ty) in types.iter().enumerate() {
            let field = self.primitive(&format!("project:{i}"), vec![atom], ty, true, scope)?;
            fields.push(self.unpack_element(field, ty, scope)?);
        }
        Ok(Imported::Product(fields))
    }

    fn pack_element(
        &mut self,
        value: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = strip_existentials(ty)
        else {
            return Ok(value);
        };
        let Imported::Product(values) = value else {
            return Err(output_error("packed element has no product components"));
        };
        let mut atoms = Vec::new();
        for (value, ty) in values.into_iter().zip(types) {
            let field = Self::leaf(self.pack_element(value, ty, scope)?)?;
            atoms.push(self.node("Var", field.name)?);
        }
        self.primitive("tuple", atoms, ty, true, scope)
    }

    fn pack(&mut self, value: &Imported<'source>) -> Result<Vec<Value>, OptimizeError> {
        let mut operands = Vec::new();
        for leaf in value.leaves()? {
            if let Some(array) = &leaf.array {
                for length in &array.dimensions {
                    operands.push(self.node("ValueOperand", length.atom)?);
                }
            }
            operands.push(self.operand(leaf)?);
        }
        Ok(operands)
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
        let element = self.element(ty)?;
        self.sink.set("PrimitiveEffects", (opcode, element), effects)?;
        if pure {
            self.sink.add("RepeatableOpcode", opcode)?;
        }
        let atoms = self.vector(atoms);
        let expression = self.node("PrimOp", (opcode, atoms))?;
        let leaf = self.scalar(ty)?;
        let binding = self.binding(&leaf)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(Imported::Leaf(leaf))
    }

    fn index(&mut self, term: &'source Term, scope: &mut Scope) -> Result<Index, OptimizeError> {
        let leaf = Self::leaf(self.visit(term, scope)?)?;
        self.index_leaf(&leaf)
    }

    fn index_binary(
        &mut self,
        op: &str,
        a: &Index,
        b: &Index,
        scope: &mut Scope,
    ) -> Result<Index, OptimizeError> {
        let constructor = match op {
            "+" => "IndexAdd",
            "-" => "IndexSub",
            "*" => "IndexMul",
            "/" => "IndexDiv",
            "%" => "IndexMod",
            "min" => "IndexMin",
            "max" => "IndexMax",
            _ => return Err(output_error("invalid index arithmetic operator")),
        };
        let expression = self.node(constructor, (a.expression, b.expression))?;
        let expression = self.node("IndexOp", expression)?;
        let leaf = self.scalar(&Type::Constructed(TypeName::Int(32), Vec::new()))?;
        let binding = self.binding(&leaf)?;
        self.statement(vec![binding], expression, scope)?;
        self.index_leaf(&leaf)
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
        let mut atoms = Vec::new();
        for argument in arguments {
            let leaf = Self::leaf(argument)?;
            if leaf.array.is_some() {
                return Err(output_error(
                    "array operation was not normalized to memory statements",
                ));
            }
            atoms.push(self.node("Var", leaf.name)?);
        }
        self.primitive(&opcode, atoms, ty, pure, scope)
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
        let observe = self.unit("Observe")?;
        for (i, argument) in arguments.iter().enumerate() {
            let diet = definition.param_diets.get(i).cloned().unwrap_or_default();
            for (leaf, consuming) in argument.diet_leaves(&diet)? {
                for dimension in leaf.array.iter().flat_map(|array| &array.dimensions) {
                    let operand = self.node("ValueOperand", dimension.atom)?;
                    passed.push(self.node("Pass", (operand, observe))?);
                }
                let operand = self.operand(leaf)?;
                let diet = if consuming { self.unit("Consume")? } else { observe };
                passed.push(self.node("Pass", (operand, diet))?);
            }
        }
        let arguments = self.vector(passed);
        let mut pattern = Vec::new();
        let mut returns = Vec::new();
        let value = self.result_pattern(ty, &mut pattern, &mut returns)?;
        let returns = self.vector(returns);
        let expression = self.node("Call", (name, arguments, returns))?;
        self.statement(pattern, expression, scope)?;
        Ok(value)
    }

    fn array_builtin(
        &mut self,
        id: BuiltinId,
        arguments: &[Imported<'source>],
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Option<Imported<'source>>, OptimizeError> {
        let known = catalog().known();
        if id == known.scratch_annotation || id == known.scratch_alloc {
            let Some(source) = arguments.first() else {
                return Err(output_error("scratch is missing its shape operand"));
            };
            if id == known.scratch_annotation {
                return self.scratch_like(source.clone(), scope).map(Some);
            }
            let count = self.index_leaf(&Self::leaf(source.clone())?)?;
            let output = self.parallel_output(ty, &count, &mut Vec::new())?;
            return self.initialize_scratch(output, scope).map(Some);
        }
        if id != known.length
            && id != known.storage_len
            && id != known.slice
            && id != known.array_with
            && id != known.array_with_in_place
            && id != known.storage_index
        {
            return Ok(None);
        }
        let Some(source) = arguments.first() else {
            return Err(output_error("array builtin is missing its array operand"));
        };
        if id == known.length || id == known.storage_len {
            let length = self.input_length(source, scope)?;
            return self.primitive("index-cast", vec![length.atom], ty, true, scope).map(Some);
        }
        let Some(Imported::Leaf(index)) = arguments.get(1) else {
            return Err(output_error("array builtin is missing its index"));
        };
        let index = self.index_leaf(index)?;
        if id == known.storage_index {
            return self.input_at(source, &index, scope).map(Some);
        }
        if id == known.slice {
            let Some(end) = arguments.get(2) else {
                return Err(output_error("slice end missing"));
            };
            let end = self.index_leaf(&Self::leaf(end.clone())?)?;
            return self.slice_value(source, &index, &end, scope).map(Some);
        }
        if contains_ragged(&component_type(ty)) {
            let Some(row) = arguments.get(2) else {
                return Err(output_error("array update value missing"));
            };
            return self
                .update_if_valid(source, &index, ty, scope, |this, destination, scope| {
                    this.update_row(destination.clone(), row.clone(), &index, scope)
                })
                .map(Some);
        }
        let Some(last) = arguments.get(2) else {
            return Err(output_error("array builtin is missing its final operand"));
        };
        let source =
            if id == known.array_with { self.copy_arrays(source.clone(), scope)? } else { source.clone() };
        // Read every component before any destination is overwritten.
        let last = last.clone().map(&mut |leaf| self.snapshot(leaf, scope))?;
        let source_leaves = source.leaves()?;
        let mut sources = source_leaves.into_iter();
        let mut values = last.leaves()?.into_iter();
        let result = Imported::build(&component_type(ty), &mut |ty| {
            let Some(source) = sources.next() else {
                return Err(output_error("array result component count mismatch"));
            };
            let Some(array) = &source.array else {
                return Err(output_error("array builtin has a scalar component"));
            };
            let array = array.clone();
            let expression = {
                let Some(value) = values.next() else {
                    return Err(output_error("array update component count mismatch"));
                };
                let atom = self.node("Var", value.name)?;
                let at = self.node("At", index.expression)?;
                let slices = self.vector(vec![at]);
                let bounds = self.unit("IgnoreOutOfBounds")?;
                self.node("Update", (bounds, source.name, slices, atom))?
            };
            let leaf = self.array_leaf(ty, array)?;
            let binding = self.binding(&leaf)?;
            self.statement(vec![binding], expression, scope)?;
            Ok(leaf)
        })?;
        if sources.next().is_some() || values.next().is_some() {
            return Err(output_error("array operation component count mismatch"));
        }
        Ok(Some(result))
    }

    fn array_value(&mut self, ty: &Type, dimensions: Vec<Index>) -> Result<Leaf, OptimizeError> {
        let mut element_ty = strip_existentials(ty);
        while element_ty.is_array() {
            let Some(element) = element_ty.elem_type() else {
                return Err(output_error("array element missing"));
            };
            element_ty = strip_existentials(element);
        }
        let element = self.element(element_ty)?;
        self.array_leaf(ty, Array { dimensions, element })
    }

    fn array(
        &mut self,
        array: &'source ArrayExpr<
            crate::tlc::data::ExplicitClosurePayload,
            crate::tlc::data::ExplicitCapturesPayload,
        >,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let ty = array.array_type();
        match array {
            ArrayExpr::Var(VarRef::Symbol(symbol), _) => return self.resolve(*symbol),
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => {
                return Err(output_error("builtin used as an array"))
            }
            ArrayExpr::Zip(arrays) => {
                return Ok(Imported::Product(
                    arrays.iter().map(|array| self.array(array, scope)).collect::<Result<_, _>>()?,
                ))
            }
            ArrayExpr::Literal(elements) => {
                let values =
                    elements.iter().map(|term| self.visit(term, scope)).collect::<Result<Vec<_>, _>>()?;
                if contains_ragged(&component_type(&ty)) {
                    let mut result = self.empty_array(&component_type(&ty), scope)?;
                    for row in values {
                        result = self.append_row(result, row, scope)?;
                    }
                    return Ok(result);
                }
                let rows = values.iter().map(Imported::leaves).collect::<Result<Vec<_>, _>>()?;
                let length = self.constant(elements.len() as i64)?;
                let mut column = 0;
                return Imported::build(&component_type(&ty), &mut |ty| {
                    let mut atoms = Vec::new();
                    let mut dimensions = vec![length.clone()];
                    for (i, row) in rows.iter().enumerate() {
                        let Some(leaf) = row.get(column) else {
                            return Err(output_error("array literal component count mismatch"));
                        };
                        atoms.push(self.node("Var", leaf.name)?);
                        if i == 0 {
                            if let Some(array) = &leaf.array {
                                dimensions.extend(array.dimensions.clone());
                            }
                        }
                    }
                    column += 1;
                    let leaf = self.array_value(ty, dimensions)?;
                    let atoms = self.vector(atoms);
                    let generator = self.node("ArrayLiteral", atoms)?;
                    let expression = self.node("GenerateArray", generator)?;
                    let binding = self.binding(&leaf)?;
                    self.statement(vec![binding], expression, scope)?;
                    Ok(leaf)
                });
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
                    let value = Self::leaf(self.literal(&start.ty, bits, scope)?)?;
                    self.node("Var", value.name)?
                };
                let output = self.array_value(&ty, vec![length])?;
                self.generate_array(output, scope, |this, indices, inner| {
                    let ordinal = Self::leaf(this.primitive(
                        "index-cast",
                        vec![indices[0].atom],
                        &start.ty,
                        true,
                        inner,
                    )?)?;
                    let ordinal = this.node("Var", ordinal.name)?;
                    let delta =
                        Self::leaf(this.primitive("*", vec![ordinal, step], &start.ty, true, inner)?)?;
                    let delta = this.node("Var", delta.name)?;
                    let value = Self::leaf(this.primitive(
                        "+",
                        vec![start_atom, delta],
                        &start.ty,
                        true,
                        inner,
                    )?)?;
                    this.node("Var", value.name)
                })
            }
        }
    }

    fn generate_array(
        &mut self,
        output: Leaf,
        scope: &mut Scope,
        element: impl FnOnce(&mut Self, &[Index], &mut Scope) -> Result<Value, OptimizeError>,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Some(array) = &output.array else {
            return Err(output_error("parallel generation requires an array result"));
        };
        let mut axes = Vec::new();
        let mut indices = Vec::new();
        for length in &array.dimensions {
            let index = self.scalar(&types::i32())?;
            axes.push(self.node("DenseAxis", (index.name, length.expression))?);
            indices.push(self.index_leaf(&index)?);
        }
        let axes = self.vector(axes);
        let flat = self.fresh("ValueId")?;
        let domain = self.node("IterationSpace", (flat, axes))?;
        let mut inner = Scope::default();
        let atom = element(self, &indices, &mut inner)?;
        let operand = self.node("ValueOperand", atom)?;
        let certificates = self.vector(Vec::new());
        let result = self.node("Yield", (certificates, operand))?;
        let results = self.vector(vec![result]);
        let collect = self.node("Collect", results)?;
        let results = self.vector(vec![collect]);
        let statements = self.vector(inner.statements);
        let body = self.node("ParallelBlock", (statements, results))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        let binding = self.binding(&output)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(Imported::Leaf(output))
    }

    fn index_array(
        &mut self,
        source: &Leaf,
        index: Index,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Some(array) = &source.array else {
            return Err(output_error("indexed value is not an array"));
        };
        let (leaf, expression) = if ty.is_array() {
            let mut result = array.clone();
            result.dimensions.remove(0);
            let at = self.node("At", index.expression)?;
            let slices = self.vector(vec![at]);
            (
                self.array_leaf(ty, result)?,
                self.node("View", (source.name, slices))?,
            )
        } else {
            let indices = self.vector(vec![index.expression]);
            (self.scalar(ty)?, self.node("Load", (source.name, indices))?)
        };
        let binding = self.binding(&leaf)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(Imported::Leaf(leaf))
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
        let initial = self.visit(init, scope)?;
        let mut initial_operands = self.pack(&initial)?;
        let mut carried_pattern = Vec::new();
        let carried = self.result_pattern(loop_var_ty, &mut carried_pattern, &mut Vec::new())?;
        let checkpoint = self.bindings.checkpoint();
        let mut inner = Scope::default();
        let mut condition_carry = None;
        let form = match kind {
            LoopKind::ForRange { var, var_ty, bound } => {
                let bound = self.index(bound, scope)?;
                let index = self.scalar(&Type::Constructed(TypeName::Int(32), Vec::new()))?;
                let atom = self.node("Var", index.name)?;
                let value = self.primitive("index-cast", vec![atom], var_ty, true, &mut inner)?;
                self.bindings.insert(*var, value);
                let zero = self.constant(0)?;
                let one = self.constant(1)?;
                self.node(
                    "Counted",
                    (index.name, zero.expression, bound.expression, one.expression),
                )?
            }
            LoopKind::For { var, iter, .. } => {
                let array = self.visit(iter, scope)?;
                let count = self.input_length(&array, scope)?.expression;
                let index = self.scalar(&Type::Constructed(TypeName::Int(32), Vec::new()))?;
                let position = self.index_leaf(&index)?;
                let value = self.input_at(&array, &position, &mut inner)?;
                self.bindings.insert(*var, value);
                let zero = self.constant(0)?;
                let one = self.constant(1)?;
                self.node("Counted", (index.name, zero.expression, count, one.expression))?
            }
            LoopKind::While { cond } => {
                self.bindings.insert(*loop_var, initial.clone());
                for (symbol, _, extraction) in init_bindings {
                    let value = self.visit(extraction, scope)?;
                    self.bindings.insert(*symbol, value);
                }
                let initial_test = Self::leaf(self.visit(cond, scope)?)?;
                initial_operands.push(self.operand(&initial_test)?);
                let condition = self.scalar(&Type::Constructed(TypeName::Bool, Vec::new()))?;
                carried_pattern.push(self.binding(&condition)?);
                condition_carry = Some(condition.clone());
                self.node("While", condition.name)?
            }
        };
        self.bindings.insert(*loop_var, carried);
        for (symbol, _, extraction) in init_bindings {
            let value = self.visit(extraction, &mut inner)?;
            self.bindings.insert(*symbol, value);
        }
        let next = self.visit(body, &mut inner)?;
        let mut next_operands = self.pack(&next)?;
        if let LoopKind::While { cond } = kind {
            self.bindings.insert(*loop_var, next);
            for (symbol, _, extraction) in init_bindings {
                let value = self.visit(extraction, &mut inner)?;
                self.bindings.insert(*symbol, value);
            }
            let condition = Self::leaf(self.visit(cond, &mut inner)?)?;
            next_operands.push(self.operand(&condition)?);
        }
        self.bindings.restore(checkpoint);
        let mut result_type = component_type(&term.ty);
        if let Some(condition) = condition_carry {
            result_type = types::tuple(vec![result_type, condition.ty]);
        }
        let output = self.carried_loop(
            initial_operands,
            carried_pattern,
            form,
            inner,
            next_operands,
            &result_type,
            scope,
        )?;
        if matches!(kind, LoopKind::While { .. }) {
            let Imported::Product(mut fields) = output else {
                return Err(output_error("while result is not a product"));
            };
            fields.pop();
            let Some(output) = fields.pop() else {
                return Err(output_error("while result is missing"));
            };
            Ok(output)
        } else {
            Ok(output)
        }
    }

    fn carried_loop(
        &mut self,
        initial: Vec<Value>,
        parameters: Vec<Value>,
        form: Value,
        inner: Scope,
        next: Vec<Value>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if initial.len() != parameters.len() || next.len() != parameters.len() {
            return Err(output_error("loop merge signature changed across an iteration"));
        }
        let observe = self.unit("Observe")?;
        let mut merges = Vec::new();
        for (binding, initial) in parameters.into_iter().zip(initial) {
            let parameter = self.node("Parameter", (binding, observe))?;
            merges.push(self.node("Carry", (parameter, initial))?);
        }
        let merges = self.vector(merges);
        let mut results = Vec::new();
        for operand in next {
            let certificates = self.vector(Vec::new());
            results.push(self.node("Yield", (certificates, operand))?);
        }
        let results = self.vector(results);
        let statements = self.vector(inner.statements);
        let body = self.node("Block", (statements, results))?;
        let mut pattern = Vec::new();
        let mut returns = Vec::new();
        let output = self.result_pattern(ty, &mut pattern, &mut returns)?;
        let returns = self.vector(returns);
        let expression = self.node("Loop", (merges, form, returns, body))?;
        self.statement(pattern, expression, scope)?;
        Ok(output)
    }

    fn callback(
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
            return Err(output_error("parallel callback parameter count mismatch"));
        }
        for ((name, _), value) in callback.lam.params.iter().zip(arguments.iter()) {
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

    fn input_length(
        &mut self,
        value: &Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Index, OptimizeError> {
        match value {
            Imported::Ragged { offsets, .. } => {
                let length = self.input_length(&Imported::Leaf(offsets.clone()), scope)?;
                let one = self.constant(1)?;
                self.index_binary("-", &length, &one, scope)
            }
            Imported::Leaf(leaf) => {
                let Some(array) = &leaf.array else {
                    return Err(output_error("parallel input has no array binding"));
                };
                let Some(length) = array.dimensions.first() else {
                    return Err(output_error("parallel input has no iteration dimension"));
                };
                Ok(length.clone())
            }
            Imported::Product(fields) => {
                let Some(field) = fields.first() else {
                    return Err(output_error("parallel input product is empty"));
                };
                self.input_length(field, scope)
            }
            Imported::Callable(_, _) => Err(output_error("parallel input is callable")),
        }
    }

    fn input_at(
        &mut self,
        value: &Imported<'source>,
        index: &Index,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match value {
            Imported::Ragged { offsets, data } => {
                let start = Self::leaf(self.index_array(offsets, index.clone(), &types::i32(), scope)?)?;
                let start = self.index_leaf(&start)?;
                let one = self.constant(1)?;
                let next = self.index_binary("+", index, &one, scope)?;
                let end = Self::leaf(self.index_array(offsets, next, &types::i32(), scope)?)?;
                let end = self.index_leaf(&end)?;
                return self.slice_value(data, &start, &end, scope);
            }
            Imported::Product(fields) => {
                return fields
                    .iter()
                    .map(|field| self.input_at(field, index, scope))
                    .collect::<Result<Vec<_>, _>>()
                    .map(Imported::Product)
            }
            Imported::Leaf(leaf) if leaf.ty.is_vec() => {
                let Some(element) = leaf.ty.elem_type() else {
                    return Err(output_error("vector element missing"));
                };
                let source = self.node("Var", leaf.name)?;
                return self.primitive("vector-index", vec![source, index.atom], element, true, scope);
            }
            _ => {}
        }
        value.clone().map(&mut |leaf| {
            let Type::Constructed(TypeName::Array, mut fields) = leaf.ty.clone() else {
                return Err(output_error("parallel input is not an array"));
            };
            let element = if leaf.ty.array_rank().is_some_and(|rank| rank > 1) {
                fields.remove(2);
                Type::Constructed(TypeName::Array, fields)
            } else {
                fields.remove(0)
            };
            Self::leaf(self.index_array(&leaf, index.clone(), &element, scope)?)
        })
    }

    fn lane_results(&mut self, value: &Imported<'source>) -> Result<Vec<Value>, OptimizeError> {
        value
            .leaves()?
            .into_iter()
            .map(|leaf| {
                let operand = self.operand(leaf)?;
                let certificates = self.vector(Vec::new());
                self.node("Yield", (certificates, operand))
            })
            .collect()
    }

    fn parallel_output(
        &mut self,
        ty: &Type,
        length: &Index,
        pattern: &mut Vec<Value>,
    ) -> Result<Imported<'source>, OptimizeError> {
        Imported::build(&component_type(ty), &mut |ty| {
            let mut lengths = vec![length.clone()];
            let Some(mut element) = ty.elem_type() else {
                return Err(output_error("parallel collect result is not an array"));
            };
            while element.is_array() {
                let Some(Type::Constructed(TypeName::Size(size), _)) = element.array_size() else {
                    return Err(output_error("ragged collect requires offset construction"));
                };
                lengths.push(self.constant(*size as i64)?);
                let Some(next) = element.elem_type() else {
                    return Err(output_error("nested array element missing"));
                };
                element = next;
            }
            let leaf = self.array_value(ty, lengths)?;
            pattern.push(self.binding(&leaf)?);
            Ok(leaf)
        })
    }

    fn soac(
        &mut self,
        operator: &'source Operator,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let (callback, inputs) = match operator {
            SoacOp::Map { lam, inputs } => (
                lam,
                inputs.iter().map(|input| self.array(input, scope)).collect::<Result<Vec<_>, _>>()?,
            ),
            SoacOp::Reduce { op, input, .. } | SoacOp::Scan { op, input, .. } => {
                (op, vec![self.array(input, scope)?])
            }
            SoacOp::Scatter { dest, lam, inputs } => {
                let destination = self.visit(dest, scope)?;
                let destination = self.copy_arrays(destination, scope)?;
                let inputs =
                    inputs.iter().map(|input| self.array(input, scope)).collect::<Result<Vec<_>, _>>()?;
                let policy = self.unit("IdenticalWrites")?;
                if contains_ragged(&component_type(ty)) {
                    return self.scatter_loop(destination, lam, &inputs, ty, scope);
                }
                return self.scatter(destination, policy, lam, inputs, scope);
            }
            SoacOp::Filter { pred, input } => {
                let input = self.array(input, scope)?;
                if input
                    .leaves()?
                    .iter()
                    .any(|leaf| leaf.array.as_ref().is_some_and(|array| array.dimensions.len() > 1))
                    || contains_ragged(&component_type(ty))
                {
                    let length = self.input_length(&input, scope)?;
                    let index = self.scalar(&types::i32())?;
                    let position = self.index_leaf(&index)?;
                    let mut inner = Scope::default();
                    let row = self.input_at(&input, &position, &mut inner)?;
                    let condition = Self::leaf(self.callback(pred, vec![row.clone()], &mut inner)?)?;
                    let condition = self.node("Var", condition.name)?;
                    return self.collect_body(&length, index.name, inner, row, condition, ty, scope);
                }
                return self.filter(pred, input, ty, scope);
            }
            SoacOp::ReduceByIndex { .. } | SoacOp::BucketScatter { .. } => {
                return self.indexed_operator(operator, ty, scope);
            }
        };
        let Some(first) = inputs.first() else {
            return Err(output_error("parallel operator has no input"));
        };
        let length = self.input_length(first, scope)?;
        let index = self.scalar(&Type::Constructed(TypeName::Int(32), Vec::new()))?;
        let position = self.index_leaf(&index)?;
        let flat = self.fresh("ValueId")?;
        let axis = self.node("DenseAxis", (index.name, length.expression))?;
        let axes = self.vector(vec![axis]);
        let domain = self.node("IterationSpace", (flat, axes))?;
        let mut inner = Scope::default();
        let mut pattern = Vec::new();
        let (output, result) = match operator {
            SoacOp::Map { .. } => {
                let arguments = inputs
                    .iter()
                    .map(|input| self.input_at(input, &position, &mut inner))
                    .collect::<Result<Vec<_>, _>>()?;
                let value = self.callback(callback, arguments, &mut inner)?;
                if contains_ragged(&component_type(ty)) {
                    let mut clauses = Vec::new();
                    let output = self.collect_segments(
                        value,
                        &component_type(ty),
                        &length,
                        &mut pattern,
                        &mut clauses,
                        scope,
                    )?;
                    let statements = self.vector(inner.statements);
                    let clauses = self.vector(clauses);
                    let body = self.node("ParallelBlock", (statements, clauses))?;
                    let execution = self.unit("Unscheduled")?;
                    let expression = self.node("Parallel", (execution, domain, body))?;
                    self.statement(pattern, expression, scope)?;
                    return Ok(output);
                }
                let output = self.parallel_output(ty, &length, &mut pattern)?;
                let results = self.lane_results(&value)?;
                let results = self.vector(results);
                let result = self.node("Collect", results)?;
                (output, result)
            }
            SoacOp::Reduce { ne, .. } | SoacOp::Scan { ne, .. } => {
                let neutral = self.visit(ne, scope)?;
                if matches!(operator, SoacOp::Scan { .. }) && contains_ragged(&component_type(ty)) {
                    return self.scan_loop(callback, first, neutral, ty, scope);
                }
                let neutrals = self.packed_atoms(&neutral)?;
                let neutrals = self.vector(neutrals);
                let mut parameters = Vec::new();
                let arguments = callback
                    .lam
                    .params
                    .iter()
                    .map(|(_, ty)| self.parameter(ty, &mut parameters))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut combine = Scope::default();
                let combined = self.callback(callback, arguments, &mut combine)?;
                let body = self.finish(combine, &combined)?;
                let parameters = self.vector(parameters);
                let mut returns = Vec::new();
                self.result_pattern(&callback.lam.ret_ty, &mut Vec::new(), &mut returns)?;
                let returns = self.vector(returns);
                let combine = self.node("FunctionBody", (parameters, returns, body))?;
                let ordered = self.unit("Ordered")?;
                let reducer = self.node("Reduction", (ordered, neutrals, combine))?;
                let input = self.input_at(first, &position, &mut inner)?;
                let values = self.packed_atoms(&input)?;
                let proofs = self.vector(Vec::new());
                let values = values
                    .into_iter()
                    .map(|atom| {
                        let operand = self.node("ValueOperand", atom)?;
                        self.node("Yield", (proofs, operand))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let values = self.vector(values);
                let scan = matches!(operator, SoacOp::Scan { .. });
                let kind = self.unit(if scan { "InclusiveScan" } else { "Total" })?;
                let result = self.node("Fold", (kind, reducer, values))?;
                let output = if scan {
                    let output = self.parallel_output(ty, &length, &mut Vec::new())?;
                    let integer = self.element(&types::i32())?;
                    let shape = self.vector(vec![length.expression]);
                    let witness = self.node("ArrayValue", (integer, shape))?;
                    for leaf in output.leaves()? {
                        for _ in leaf.array.iter().flat_map(|array| array.dimensions.iter().skip(1)) {
                            let name = self.fresh("ValueId")?;
                            pattern.push(self.node("BindValue", (name, witness))?);
                        }
                        pattern.push(self.binding(leaf)?);
                    }
                    output
                } else {
                    self.result_pattern(ty, &mut pattern, &mut Vec::new())?
                };
                (output, result)
            }
            _ => return Err(output_error("invalid parallel normalization state")),
        };
        let statements = self.vector(inner.statements);
        let results = self.vector(vec![result]);
        let body = self.node("ParallelBlock", (statements, results))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        self.statement(pattern, expression, scope)?;
        Ok(output)
    }

    // Array state uses the same existential carries as source loops. Parallel
    // work inside the combiner remains visible to fusion and placement.
    fn empty_array(&mut self, ty: &Type, scope: &mut Scope) -> Result<Imported<'source>, OptimizeError> {
        if let Some((offsets, data)) = ragged_parts(ty) {
            let one = self.constant(1)?;
            let offsets = Self::leaf(self.fill_array(&offsets, vec![one], 0, scope)?)?;
            return Ok(Imported::Ragged {
                offsets,
                data: Box::new(self.empty_array(&data, scope)?),
            });
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
            return fields
                .iter()
                .map(|ty| self.empty_array(ty, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product);
        }
        if !ty.is_array() {
            return Err(output_error("empty array has a scalar type"));
        }
        let mut dimensions = Vec::new();
        let mut element = ty;
        while element.is_array() {
            for axis in 0..element.array_rank().unwrap_or(1) {
                let n = if dimensions.is_empty() {
                    0
                } else {
                    match element.array_dim(axis) {
                        Some(Type::Constructed(TypeName::Size(n), _)) => *n as i64,
                        _ => 0,
                    }
                };
                dimensions.push(self.constant(n)?);
            }
            let Some(next) = element.elem_type() else {
                return Err(output_error("array element missing"));
            };
            element = next;
        }
        let leaf = self.array_value(ty, dimensions)?;
        self.allocate_array(&leaf, scope)?;
        Ok(Imported::Leaf(leaf))
    }

    fn allocate_array(&mut self, leaf: &Leaf, scope: &mut Scope) -> Result<(), OptimizeError> {
        let Some(array) = &leaf.array else {
            return Err(output_error("allocation has a scalar type"));
        };
        let shape = self.vector(array.dimensions.iter().map(|length| length.expression).collect());
        let generator = self.node("Uninitialized", (array.element, shape))?;
        let expression = self.node("GenerateArray", generator)?;
        let binding = self.binding(leaf)?;
        self.statement(vec![binding], expression, scope)
    }

    fn offset_values(
        &mut self,
        source: Leaf,
        base: &Index,
        opcode: &str,
        scope: &mut Scope,
    ) -> Result<Leaf, OptimizeError> {
        let Some(array) = &source.array else {
            return Err(output_error("offsets are not an array"));
        };
        let output = self.array_leaf(&source.ty, array.clone())?;
        let length = self.input_length(&Imported::Leaf(source.clone()), scope)?;
        let name = self.fresh("ValueId")?;
        let index = Index {
            expression: self.node("IndexValue", name)?,
            atom: self.node("Var", name)?,
        };
        let mut inner = Scope::default();
        let value = Self::leaf(self.index_array(&source, index, &types::i32(), &mut inner)?)?;
        let value = self.index_leaf(&value)?;
        let value = self.index_binary(opcode, &value, base, &mut inner)?;
        let operand = self.node("ValueOperand", value.atom)?;
        let certs = self.vector(Vec::new());
        let result = self.node("Yield", (certs, operand))?;
        let results = self.vector(vec![result]);
        let collect = self.node("Collect", results)?;
        let results = self.vector(vec![collect]);
        let statements = self.vector(inner.statements);
        let body = self.node("ParallelBlock", (statements, results))?;
        let axis = self.node("DenseAxis", (name, length.expression))?;
        let axes = self.vector(vec![axis]);
        let flat = self.fresh("ValueId")?;
        let domain = self.node("IterationSpace", (flat, axes))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        let binding = self.binding(&output)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(output)
    }

    fn slice_value(
        &mut self,
        source: &Imported<'source>,
        start: &Index,
        end: &Index,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match source {
            Imported::Product(fields) => fields
                .iter()
                .map(|field| self.slice_value(field, start, end, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product),
            Imported::Ragged { offsets, data } => {
                let first = Self::leaf(self.index_array(offsets, start.clone(), &types::i32(), scope)?)?;
                let first = self.index_leaf(&first)?;
                let last = Self::leaf(self.index_array(offsets, end.clone(), &types::i32(), scope)?)?;
                let last = self.index_leaf(&last)?;
                let one = self.constant(1)?;
                let limit = self.index_binary("+", end, &one, scope)?;
                let offsets = Self::leaf(self.slice_value(
                    &Imported::Leaf(offsets.clone()),
                    start,
                    &limit,
                    scope,
                )?)?;
                let offsets = self.offset_values(offsets, &first, "-", scope)?;
                let data = Box::new(self.slice_value(data, &first, &last, scope)?);
                Ok(Imported::Ragged { offsets, data })
            }
            Imported::Leaf(source) => {
                let Some(array) = &source.array else {
                    return Err(output_error("slice source is not an array"));
                };
                let mut array = array.clone();
                let length = self.index_binary("-", end, start, scope)?;
                array.dimensions[0] = length.clone();
                let one = self.constant(1)?;
                let slice = self.node("Slice", (start.expression, length.expression, one.expression))?;
                let slices = self.vector(vec![slice]);
                let expression = self.node("View", (source.name, slices))?;
                let output = self.array_leaf(&source.ty, array)?;
                let binding = self.binding(&output)?;
                self.statement(vec![binding], expression, scope)?;
                Ok(Imported::Leaf(output))
            }
            Imported::Callable(_, _) => Err(output_error("callable used as an array")),
        }
    }

    fn write_slice(
        &mut self,
        destination: Leaf,
        start: &Index,
        source: &Leaf,
        scope: &mut Scope,
    ) -> Result<Leaf, OptimizeError> {
        let length = self.input_length(&Imported::Leaf(source.clone()), scope)?;
        let one = self.constant(1)?;
        let slice = self.node("Slice", (start.expression, length.expression, one.expression))?;
        let slices = self.vector(vec![slice]);
        let bounds = self.unit("ProvenInBounds")?;
        let atom = self.node("Var", source.name)?;
        let expression = self.node("Update", (bounds, destination.name, slices, atom))?;
        let mut updated = destination;
        updated.name = self.fresh("ValueId")?;
        let binding = self.binding(&updated)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(updated)
    }

    fn concatenate(
        &mut self,
        left: Imported<'source>,
        right: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match (left, right) {
            (Imported::Product(left), Imported::Product(right)) if left.len() == right.len() => left
                .into_iter()
                .zip(right)
                .map(|(a, b)| self.concatenate(a, b, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product),
            (
                Imported::Ragged {
                    offsets: a,
                    data: left,
                },
                Imported::Ragged {
                    offsets: b,
                    data: right,
                },
            ) => {
                let zero = self.constant(0)?;
                let one = self.constant(1)?;
                let length = self.input_length(&Imported::Leaf(a.clone()), scope)?;
                let length = self.index_binary("-", &length, &one, scope)?;
                let a = self.slice_value(&Imported::Leaf(a), &zero, &length, scope)?;
                let base = self.input_length(&left, scope)?;
                let b = self.offset_values(b, &base, "+", scope)?;
                let offsets = Self::leaf(self.concatenate(a, Imported::Leaf(b), scope)?)?;
                let data = Box::new(self.concatenate(*left, *right, scope)?);
                Ok(Imported::Ragged { offsets, data })
            }
            (Imported::Leaf(left), Imported::Leaf(right)) => {
                let a = self.input_length(&Imported::Leaf(left.clone()), scope)?;
                let b = self.input_length(&Imported::Leaf(right.clone()), scope)?;
                let Some(array) = &right.array else {
                    return Err(output_error("concatenation requires arrays"));
                };
                let mut array = array.clone();
                array.dimensions[0] = self.index_binary("+", &a, &b, scope)?;
                let output = self.array_leaf(&left.ty, array)?;
                self.allocate_array(&output, scope)?;
                let zero = self.constant(0)?;
                let output = self.write_slice(output, &zero, &left, scope)?;
                let output = self.write_slice(output, &a, &right, scope)?;
                Ok(Imported::Leaf(output))
            }
            _ => Err(output_error("concatenation component shapes differ")),
        }
    }

    fn branch_value(
        &mut self,
        condition: Value,
        ty: &Type,
        scope: &mut Scope,
        yes: impl FnOnce(&mut Self, &mut Scope) -> Result<Imported<'source>, OptimizeError>,
        no: impl FnOnce(&mut Self, &mut Scope) -> Result<Imported<'source>, OptimizeError>,
    ) -> Result<Imported<'source>, OptimizeError> {
        let mut yes_scope = Scope::default();
        let mut no_scope = Scope::default();
        let yes = yes(self, &mut yes_scope)?;
        let no = no(self, &mut no_scope)?;
        let yes = self.finish(yes_scope, &yes)?;
        let no = self.finish(no_scope, &no)?;
        let mut pattern = Vec::new();
        let mut returns = Vec::new();
        let value = self.result_pattern(ty, &mut pattern, &mut returns)?;
        let returns = self.vector(returns);
        let expression = self.node("If", (condition, returns, yes, no))?;
        self.statement(pattern, expression, scope)?;
        Ok(value)
    }

    fn counted_state(
        &mut self,
        length: &Index,
        initial: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
        step: impl FnOnce(
            &mut Self,
            Index,
            Imported<'source>,
            &mut Scope,
        ) -> Result<Imported<'source>, OptimizeError>,
    ) -> Result<Imported<'source>, OptimizeError> {
        let mut parameters = Vec::new();
        let state = self.result_pattern(ty, &mut parameters, &mut Vec::new())?;
        let name = self.fresh("ValueId")?;
        let index = Index {
            expression: self.node("IndexValue", name)?,
            atom: self.node("Var", name)?,
        };
        let mut inner = Scope::default();
        let next = step(self, index, state, &mut inner)?;
        let zero = self.constant(0)?;
        let one = self.constant(1)?;
        let form = self.node(
            "Counted",
            (name, zero.expression, length.expression, one.expression),
        )?;
        let initial = self.pack(&initial)?;
        let next = self.pack(&next)?;
        self.carried_loop(initial, parameters, form, inner, next, ty, scope)
    }

    fn fold_domain(
        &mut self,
        sources: &[Imported<'source>],
        coordinates: &[Vec<u8>],
        rank: u8,
        positions: &mut Vec<Index>,
        initial: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
        step: &mut impl FnMut(
            &mut Self,
            Vec<Imported<'source>>,
            Imported<'source>,
            &mut Scope,
        ) -> Result<Imported<'source>, OptimizeError>,
    ) -> Result<Imported<'source>, OptimizeError> {
        if sources.len() != coordinates.len() {
            return Err(output_error("domain input mappings differ from sources"));
        }
        if positions.len() == usize::from(rank) {
            let mut arguments = Vec::new();
            for (source, mapping) in sources.iter().zip(coordinates) {
                let mut value = source.clone();
                for &axis in mapping {
                    let Some(index) = positions.get(usize::from(axis)) else {
                        return Err(output_error("input coordinate exceeds its domain"));
                    };
                    value = self.input_at(&value, index, scope)?;
                }
                arguments.push(value);
            }
            return step(self, arguments, initial, scope);
        }
        let mut length = None;
        for (source, mapping) in sources.iter().zip(coordinates) {
            let Some(dimension) = mapping.iter().position(|&axis| usize::from(axis) == positions.len())
            else {
                continue;
            };
            let mut value = source.clone();
            for &axis in &mapping[..dimension] {
                let Some(index) = positions.get(usize::from(axis)) else {
                    return Err(output_error("dependent domain axes are not in lexical order"));
                };
                value = self.input_at(&value, index, scope)?;
            }
            length = Some(self.input_length(&value, scope)?);
            break;
        }
        let Some(length) = length else {
            return Err(output_error("domain axis has no input extent"));
        };
        self.counted_state(&length, initial, ty, scope, |this, index, state, inner| {
            positions.push(index);
            let result = this.fold_domain(sources, coordinates, rank, positions, state, ty, inner, step);
            positions.pop();
            result
        })
    }

    fn update_if_valid(
        &mut self,
        destination: &Imported<'source>,
        index: &Index,
        ty: &Type,
        scope: &mut Scope,
        update: impl FnOnce(
            &mut Self,
            &Imported<'source>,
            &mut Scope,
        ) -> Result<Imported<'source>, OptimizeError>,
    ) -> Result<Imported<'source>, OptimizeError> {
        let length = self.input_length(destination, scope)?;
        let zero = self.constant(0)?;
        let below = self.predicate("<", index.atom, length.atom, scope)?;
        let nonnegative = self.predicate(">=", index.atom, zero.atom, scope)?;
        let valid = self.predicate("&&", below, nonnegative, scope)?;
        self.branch_value(
            valid,
            ty,
            scope,
            |this, scope| update(this, destination, scope),
            |_, _| Ok(destination.clone()),
        )
    }

    fn update_row(
        &mut self,
        destination: Imported<'source>,
        row: Imported<'source>,
        index: &Index,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let row = row.map(&mut |leaf| self.snapshot(leaf, scope))?;
        self.write_row(destination, row, index, scope)
    }

    fn write_row(
        &mut self,
        destination: Imported<'source>,
        row: Imported<'source>,
        index: &Index,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match (destination, row) {
            (Imported::Product(arrays), Imported::Product(rows)) if arrays.len() == rows.len() => arrays
                .into_iter()
                .zip(rows)
                .map(|(array, row)| self.write_row(array, row, index, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product),
            (Imported::Leaf(mut array), Imported::Leaf(row)) => {
                let source = array.name;
                array.name = self.fresh("ValueId")?;
                let binding = self.binding(&array)?;
                let atom = self.node("Var", row.name)?;
                let slice = self.node("At", index.expression)?;
                let slices = self.vector(vec![slice]);
                let bounds = self.unit("ProvenInBounds")?;
                let expression = self.node("Update", (bounds, source, slices, atom))?;
                self.statement(vec![binding], expression, scope)?;
                Ok(Imported::Leaf(array))
            }
            (destination @ Imported::Ragged { .. }, row) => {
                let zero = self.constant(0)?;
                let one = self.constant(1)?;
                let length = self.input_length(&destination, scope)?;
                let next = self.index_binary("+", index, &one, scope)?;
                let prefix = self.slice_value(&destination, &zero, index, scope)?;
                let prefix = self.append_row(prefix, row, scope)?;
                let suffix = self.slice_value(&destination, &next, &length, scope)?;
                self.concatenate(prefix, suffix, scope)
            }
            _ => Err(output_error("array update row components differ")),
        }
    }

    fn append_row(
        &mut self,
        array: Imported<'source>,
        row: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match (array, row) {
            (Imported::Product(arrays), Imported::Product(rows)) if arrays.len() == rows.len() => arrays
                .into_iter()
                .zip(rows)
                .map(|(a, row)| self.append_row(a, row, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product),
            (Imported::Ragged { offsets, data }, row) => {
                let data = self.concatenate(*data, row, scope)?;
                let end = self.input_length(&data, scope)?;
                let end = self.primitive("index-cast", vec![end.atom], &types::i32(), true, scope)?;
                let offsets = Self::leaf(self.append_row(Imported::Leaf(offsets), end, scope)?)?;
                Ok(Imported::Ragged {
                    offsets,
                    data: Box::new(data),
                })
            }
            (Imported::Leaf(array), Imported::Leaf(row)) => {
                let mut dimensions = vec![self.constant(1)?];
                if let Some(array) = &row.array {
                    dimensions.extend(array.dimensions.clone());
                }
                let singleton = self.array_value(&array.ty, dimensions)?;
                let atom = self.node("Var", row.name)?;
                let atoms = self.vector(vec![atom]);
                let generator = self.node("ArrayLiteral", atoms)?;
                let expression = self.node("GenerateArray", generator)?;
                let binding = self.binding(&singleton)?;
                self.statement(vec![binding], expression, scope)?;
                self.concatenate(Imported::Leaf(array), Imported::Leaf(singleton), scope)
            }
            _ => Err(output_error("array row components differ")),
        }
    }

    fn growing_initial(
        &mut self,
        value: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match value {
            Imported::Product(fields) => fields
                .into_iter()
                .map(|field| self.growing_initial(field, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product),
            Imported::Ragged { offsets, data } => Ok(Imported::Product(vec![
                self.growing_initial(Imported::Leaf(offsets), scope)?,
                self.growing_initial(*data, scope)?,
            ])),
            value @ Imported::Leaf(_) => {
                let length = self.input_length(&value, scope)?;
                let used = self.primitive("index-cast", vec![length.atom], &types::i32(), true, scope)?;
                Ok(Imported::Product(vec![used, value]))
            }
            Imported::Callable(_, _) => Err(output_error("growable storage requires an array")),
        }
    }

    fn growing_parts(
        value: Imported<'source>,
    ) -> Result<(Imported<'source>, Imported<'source>), OptimizeError> {
        let Imported::Product(fields) = value else {
            return Err(output_error("growable state is not a product"));
        };
        let Ok([first, second]) = <[_; 2]>::try_from(fields) else {
            return Err(output_error("growable state requires two components"));
        };
        Ok((first, second))
    }

    fn growing_length(
        &mut self,
        ty: &Type,
        state: &Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Index, OptimizeError> {
        let Imported::Product(fields) = state else {
            return Err(output_error("growable state is not a product"));
        };
        let Some(first) = fields.first() else {
            return Err(output_error("growable state is empty"));
        };
        if let Some((offset_ty, _)) = ragged_parts(ty) {
            let length = self.growing_length(&offset_ty, first, scope)?;
            let one = self.constant(1)?;
            return self.index_binary("-", &length, &one, scope);
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = ty {
            let Some(ty) = types.first() else {
                return Err(output_error("growable product type is empty"));
            };
            return self.growing_length(ty, first, scope);
        }
        let Imported::Leaf(used) = first else {
            return Err(output_error("growable length is not scalar"));
        };
        self.index_leaf(used)
    }

    fn growing_finish(
        &mut self,
        ty: &Type,
        state: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if let Some((offset_ty, data_ty)) = ragged_parts(ty) {
            let (offsets, data) = Self::growing_parts(state)?;
            let offsets = Self::leaf(self.growing_finish(&offset_ty, offsets, scope)?)?;
            let data = Box::new(self.growing_finish(&data_ty, data, scope)?);
            return Ok(Imported::Ragged { offsets, data });
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = ty {
            let Imported::Product(fields) = state else {
                return Err(output_error("growable product state missing"));
            };
            if fields.len() != types.len() {
                return Err(output_error("growable product arity mismatch"));
            }
            return types
                .iter()
                .zip(fields)
                .map(|(ty, field)| self.growing_finish(ty, field, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product);
        }
        let (used, buffer) = Self::growing_parts(state)?;
        let used = self.index_leaf(&Self::leaf(used)?)?;
        let zero = self.constant(0)?;
        self.slice_value(&buffer, &zero, &used, scope)
    }

    fn growing_extend(
        &mut self,
        ty: &Type,
        state: Imported<'source>,
        value: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if let Some((offset_ty, data_ty)) = ragged_parts(ty) {
            let (offsets, data) = Self::growing_parts(state)?;
            let Imported::Ragged {
                offsets: appended,
                data: payload,
            } = value
            else {
                return Err(output_error("ragged append requires offsets and payload"));
            };
            let base = self.growing_length(&data_ty, &data, scope)?;
            let one = self.constant(1)?;
            let end = self.input_length(&Imported::Leaf(appended.clone()), scope)?;
            let appended = Self::leaf(self.slice_value(&Imported::Leaf(appended), &one, &end, scope)?)?;
            let appended = self.offset_values(appended, &base, "+", scope)?;
            return Ok(Imported::Product(vec![
                self.growing_extend(&offset_ty, offsets, Imported::Leaf(appended), scope)?,
                self.growing_extend(&data_ty, data, *payload, scope)?,
            ]));
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = ty {
            let (Imported::Product(states), Imported::Product(values)) = (state, value) else {
                return Err(output_error("growable product components missing"));
            };
            if states.len() != types.len() || values.len() != types.len() {
                return Err(output_error("growable product arity mismatch"));
            }
            return types
                .iter()
                .zip(states)
                .zip(values)
                .map(|((ty, state), value)| self.growing_extend(ty, state, value, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product);
        }
        let (used, storage) = Self::growing_parts(state)?;
        let used = self.index_leaf(&Self::leaf(used)?)?;
        let storage = Self::leaf(storage)?;
        let value = Self::leaf(value)?;
        let capacity = self.input_length(&Imported::Leaf(storage.clone()), scope)?;
        let added = self.input_length(&Imported::Leaf(value.clone()), scope)?;
        let required = self.index_binary("+", &used, &added, scope)?;
        let zero = self.constant(0)?;
        let empty = self.predicate("==", used.atom, zero.atom, scope)?;
        let full = self.predicate("<", capacity.atom, required.atom, scope)?;
        let grow = self.predicate("||", empty, full, scope)?;
        let two = self.constant(2)?;
        let doubled = self.index_binary("*", &capacity, &two, scope)?;
        let capacity = self.index_binary("max", &doubled, &required, scope)?;
        let Some(previous) = &storage.array else {
            return Err(output_error("growable storage is not an array"));
        };
        let Some(mut array) = value.array.clone() else {
            return Err(output_error("appended value is not an array"));
        };
        if array.dimensions.len() != previous.dimensions.len() {
            return Err(output_error("append changes array rank"));
        }
        for (old, new) in previous.dimensions.iter().zip(&array.dimensions).skip(1) {
            let equal = self.predicate("==", old.atom, new.atom, scope)?;
            let valid = self.predicate("||", empty, equal, scope)?;
            let assertion = self.node("Assert", (valid, "append changes rectangular row shape"))?;
            self.statement(Vec::new(), assertion, scope)?;
        }
        let Some(length) = array.dimensions.first_mut() else {
            return Err(output_error("growable array has no outer dimension"));
        };
        *length = capacity;
        let fresh = self.array_leaf(&storage.ty, array)?;
        let mut yes = Scope::default();
        self.allocate_array(&fresh, &mut yes)?;
        let zero = self.constant(0)?;
        let live =
            Self::leaf(self.slice_value(&Imported::Leaf(storage.clone()), &zero, &used, &mut yes)?)?;
        let fresh = self.write_slice(fresh, &zero, &live, &mut yes)?;
        let yes = self.finish(yes, &Imported::Leaf(fresh))?;
        let no = self.finish(Scope::default(), &Imported::Leaf(storage))?;
        let mut pattern = Vec::new();
        let mut returns = Vec::new();
        let buffer = Self::leaf(self.result_pattern(ty, &mut pattern, &mut returns)?)?;
        let returns = self.vector(returns);
        let expression = self.node("If", (grow, returns, yes, no))?;
        self.statement(pattern, expression, scope)?;
        let buffer = self.write_slice(buffer, &used, &value, scope)?;
        let used = self.primitive("index-cast", vec![required.atom], &types::i32(), true, scope)?;
        Ok(Imported::Product(vec![used, Imported::Leaf(buffer)]))
    }

    fn growing_append(
        &mut self,
        ty: &Type,
        state: Imported<'source>,
        row: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if let Some((offset_ty, data_ty)) = ragged_parts(ty) {
            let (offsets, data) = Self::growing_parts(state)?;
            let data = self.growing_extend(&data_ty, data, row, scope)?;
            let end = self.growing_length(&data_ty, &data, scope)?;
            let end = self.primitive("index-cast", vec![end.atom], &types::i32(), true, scope)?;
            let offsets = self.growing_append(&offset_ty, offsets, end, scope)?;
            return Ok(Imported::Product(vec![offsets, data]));
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = ty {
            let (Imported::Product(states), Imported::Product(rows)) = (state, row) else {
                return Err(output_error("growable row components missing"));
            };
            if states.len() != types.len() || rows.len() != types.len() {
                return Err(output_error("growable row arity mismatch"));
            }
            return types
                .iter()
                .zip(states)
                .zip(rows)
                .map(|((ty, state), row)| self.growing_append(ty, state, row, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product);
        }
        let row = Self::leaf(row)?;
        let mut dimensions = vec![self.constant(1)?];
        if let Some(array) = &row.array {
            dimensions.extend(array.dimensions.clone());
        }
        let singleton = self.array_value(ty, dimensions)?;
        let atom = self.node("Var", row.name)?;
        let atoms = self.vector(vec![atom]);
        let generator = self.node("ArrayLiteral", atoms)?;
        let expression = self.node("GenerateArray", generator)?;
        let binding = self.binding(&singleton)?;
        self.statement(vec![binding], expression, scope)?;
        self.growing_extend(ty, state, Imported::Leaf(singleton), scope)
    }

    fn concatenated(
        &mut self,
        row: Leaf,
        offsets: bool,
        length: &Index,
        pattern: &mut Vec<Value>,
        clauses: &mut Vec<Value>,
        scope: &mut Scope,
    ) -> Result<(Leaf, Leaf), OptimizeError> {
        let Some(mut array) = row.array.clone() else {
            return Err(output_error("concatenated lane result is not an array"));
        };
        let one = self.constant(1)?;
        let count = self.index_binary("+", length, &one, scope)?;
        let offset_ty = types::view_array_with_size(
            &types::i32(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            types::no_buffer(),
        );
        let segments = self.array_value(&offset_ty, vec![count])?;
        let size = self.scalar(&types::i32())?;
        let Some(first) = array.dimensions.first_mut() else {
            return Err(output_error("concatenated lane result has no leading dimension"));
        };
        *first = self.index_leaf(&size)?;
        let payload = self.array_leaf(&row.ty, array)?;
        pattern.push(self.binding(&segments)?);
        pattern.push(self.binding(&size)?);
        pattern.push(self.binding(&payload)?);
        let mode = self.unit(if offsets { "OffsetSequences" } else { "Elements" })?;
        clauses.push(self.node("Concatenate", (mode, row.name))?);
        Ok((segments, payload))
    }

    fn concatenate_components(
        &mut self,
        row: Imported<'source>,
        length: &Index,
        pattern: &mut Vec<Value>,
        clauses: &mut Vec<Value>,
        scope: &mut Scope,
    ) -> Result<(Leaf, Imported<'source>), OptimizeError> {
        match row {
            Imported::Leaf(row) => {
                let (segments, data) = self.concatenated(row, false, length, pattern, clauses, scope)?;
                Ok((segments, Imported::Leaf(data)))
            }
            Imported::Ragged { offsets, data } => {
                let (segments, offsets) =
                    self.concatenated(offsets, true, length, pattern, clauses, scope)?;
                let (_, data) = self.concatenate_components(*data, length, pattern, clauses, scope)?;
                Ok((
                    segments,
                    Imported::Ragged {
                        offsets,
                        data: Box::new(data),
                    },
                ))
            }
            _ => Err(output_error("concatenation requires normalized array components")),
        }
    }

    fn collect_segments(
        &mut self,
        row: Imported<'source>,
        ty: &Type,
        length: &Index,
        pattern: &mut Vec<Value>,
        clauses: &mut Vec<Value>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if let Imported::Product(fields) = row {
            let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = ty else {
                return Err(output_error("collected product has no component types"));
            };
            if fields.len() != types.len() {
                return Err(output_error("collected product arity differs from its type"));
            }
            return fields
                .into_iter()
                .zip(types)
                .map(|(field, ty)| self.collect_segments(field, ty, length, pattern, clauses, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product);
        }
        if !contains_ragged(ty) {
            let output = self.parallel_output(ty, length, pattern)?;
            let results = self.lane_results(&row)?;
            let results = self.vector(results);
            clauses.push(self.node("Collect", results)?);
            return Ok(output);
        }
        let (offsets, data) = self.concatenate_components(row, length, pattern, clauses, scope)?;
        Ok(Imported::Ragged {
            offsets,
            data: Box::new(data),
        })
    }

    fn collect_body(
        &mut self,
        length: &Index,
        index: Value,
        mut inner: Scope,
        row: Imported<'source>,
        condition: Value,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let ty = component_type(ty);
        let state_ty = growing_type(&ty);
        let empty = self.empty_array(&ty, scope)?;
        let initial = self.growing_initial(empty, scope)?;
        let mut parameters = Vec::new();
        let state = self.result_pattern(&state_ty, &mut parameters, &mut Vec::new())?;
        let next = self.branch_value(
            condition,
            &state_ty,
            &mut inner,
            |this, scope| this.growing_append(&ty, state.clone(), row, scope),
            |_, _| Ok(state.clone()),
        )?;
        let initial = self.pack(&initial)?;
        let next = self.pack(&next)?;
        let zero = self.constant(0)?;
        let one = self.constant(1)?;
        let form = self.node(
            "Counted",
            (index, zero.expression, length.expression, one.expression),
        )?;
        let result = self.carried_loop(initial, parameters, form, inner, next, &state_ty, scope)?;
        self.growing_finish(&ty, result, scope)
    }

    fn scan_loop(
        &mut self,
        callback: &'source Callback,
        input: &Imported<'source>,
        neutral: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let length = self.input_length(input, scope)?;
        let component_ty = component_type(ty);
        let state_ty = types::tuple(vec![callback.lam.ret_ty.clone(), growing_type(&component_ty)]);
        let empty = self.empty_array(&component_ty, scope)?;
        let prefixes = self.growing_initial(empty, scope)?;
        let initial = Imported::Product(vec![neutral, prefixes]);
        let output =
            self.counted_state(&length, initial, &state_ty, scope, |this, index, state, inner| {
                let Imported::Product(fields) = state else {
                    return Err(output_error("scan state is not a product"));
                };
                let element = this.input_at(input, &index, inner)?;
                let combined = this.callback(callback, vec![fields[0].clone(), element], inner)?;
                let prefixes =
                    this.growing_append(&component_ty, fields[1].clone(), combined.clone(), inner)?;
                Ok(Imported::Product(vec![combined, prefixes]))
            })?;
        let Imported::Product(mut fields) = output else {
            return Err(output_error("scan result is not a product"));
        };
        let Some(prefixes) = fields.pop() else {
            return Err(output_error("scan prefixes missing"));
        };
        self.growing_finish(&component_ty, prefixes, scope)
    }

    fn packed_atoms(&mut self, value: &Imported<'source>) -> Result<Vec<Value>, OptimizeError> {
        let mut atoms = Vec::new();
        for leaf in value.leaves()? {
            if let Some(array) = &leaf.array {
                atoms.extend(array.dimensions.iter().map(|length| length.atom));
            }
            atoms.push(self.node("Var", leaf.name)?);
        }
        Ok(atoms)
    }

    fn scalar_atoms(
        &mut self,
        value: &Imported<'source>,
        atoms: &mut Vec<Value>,
    ) -> Result<(), OptimizeError> {
        for leaf in value.leaves()? {
            if leaf.array.is_some() {
                return Err(output_error("expected scalar/vector components"));
            }
            atoms.push(self.node("Var", leaf.name)?);
        }
        Ok(())
    }

    fn snapshot(&mut self, source: Leaf, scope: &mut Scope) -> Result<Leaf, OptimizeError> {
        let Some(array) = &source.array else {
            return Ok(source);
        };
        let result = self.array_value(&source.ty, array.dimensions.clone())?;
        let expression = self.node("Copy", source.name)?;
        let binding = self.binding(&result)?;
        self.statement(vec![binding], expression, scope)?;
        Ok(result)
    }

    fn copy_arrays(
        &mut self,
        value: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        value.map(&mut |source| {
            if source.ty.array_variant().is_some_and(types::is_array_variant_view) {
                Ok(source)
            } else {
                self.snapshot(source, scope)
            }
        })
    }

    fn accumulator_parameters(
        &mut self,
        value: &Imported<'source>,
        policy: Value,
        inputs: &mut Vec<Value>,
        parameters: &mut Vec<Value>,
        tokens: &mut Vec<Accumulator>,
    ) -> Result<(), OptimizeError> {
        match value {
            Imported::Product(fields) => {
                for field in fields {
                    self.accumulator_parameters(field, policy, inputs, parameters, tokens)?;
                }
            }
            Imported::Leaf(leaf) => {
                let Some(array) = &leaf.array else {
                    return Err(output_error("accumulator destination has no memory"));
                };
                let shape = self.vector(array.dimensions.iter().map(|length| length.expression).collect());
                let elements = self.vector(vec![array.element]);
                let info = self.node("AccumulatorInfo", (shape, elements))?;
                let name = self.fresh("AccumulatorId")?;
                let binding = self.node("BindAccumulator", (name, info))?;
                let consume = self.unit("Consume")?;
                parameters.push(self.node("Parameter", (binding, consume))?);
                let arrays = self.vector(vec![leaf.name]);
                inputs.push(self.node("AccumulatorInput", (arrays, policy))?);
                tokens.push(Accumulator { name, info });
            }
            Imported::Ragged { offsets, data } => {
                self.accumulator_parameters(
                    &Imported::Leaf(offsets.clone()),
                    policy,
                    inputs,
                    parameters,
                    tokens,
                )?;
                self.accumulator_parameters(data, policy, inputs, parameters, tokens)?;
            }
            Imported::Callable(_, _) => return Err(output_error("callable accumulator input")),
        }
        Ok(())
    }

    fn updated_bindings(
        &mut self,
        value: Imported<'source>,
        pattern: &mut Vec<Value>,
    ) -> Result<Imported<'source>, OptimizeError> {
        value.map(&mut |mut leaf| {
            leaf.name = self.fresh("ValueId")?;
            pattern.push(self.binding(&leaf)?);
            Ok(leaf)
        })
    }

    fn finish_accumulator(
        &mut self,
        destination: Imported<'source>,
        inputs: Vec<Value>,
        parameters: Vec<Value>,
        tokens: Vec<Accumulator>,
        body: Scope,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let mut returns = Vec::new();
        let mut results = Vec::new();
        for token in tokens {
            returns.push(self.node("ReturnAccumulator", token.info)?);
            let operand = self.node("AccOperand", token.name)?;
            let certs = self.vector(Vec::new());
            results.push(self.node("Yield", (certs, operand))?);
        }
        let results = self.vector(results);
        let statements = self.vector(body.statements);
        let body = self.node("Block", (statements, results))?;
        let parameters = self.vector(parameters);
        let returns = self.vector(returns);
        let callback = self.node("FunctionBody", (parameters, returns, body))?;
        let inputs = self.vector(inputs);
        let expression = self.node("WithAcc", (inputs, callback))?;
        let mut pattern = Vec::new();
        let output = self.updated_bindings(destination, &mut pattern)?;
        self.statement(pattern, expression, scope)?;
        Ok(output)
    }

    fn domain(&mut self, length: &Index) -> Result<(Value, Index), OptimizeError> {
        let name = self.fresh("ValueId")?;
        let ordinal = self.fresh("ValueId")?;
        let axis = self.node("DenseAxis", (name, length.expression))?;
        let axes = self.vector(vec![axis]);
        let domain = self.node("IterationSpace", (ordinal, axes))?;
        Ok((
            domain,
            Index {
                expression: self.node("IndexValue", name)?,
                atom: self.node("Var", name)?,
            },
        ))
    }

    fn contribute(
        &mut self,
        tokens: Vec<Accumulator>,
        domain: Value,
        body: Scope,
        scope: &mut Scope,
    ) -> Result<Vec<Accumulator>, OptimizeError> {
        let values = self.vector(tokens.iter().map(|token| token.name).collect());
        let contribution = self.node("Contribute", values)?;
        let results = self.vector(vec![contribution]);
        let statements = self.vector(body.statements);
        let body = self.node("ParallelBlock", (statements, results))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        let mut pattern = Vec::new();
        let mut output = Vec::new();
        for token in tokens {
            let name = self.fresh("AccumulatorId")?;
            pattern.push(self.node("BindAccumulator", (name, token.info))?);
            output.push(Accumulator {
                name,
                info: token.info,
            });
        }
        self.statement(pattern, expression, scope)?;
        Ok(output)
    }

    fn accumulate(
        &mut self,
        token: &Accumulator,
        indices: Vec<Index>,
        atoms: Vec<Value>,
        scope: &mut Scope,
    ) -> Result<Accumulator, OptimizeError> {
        let bounds = self.unit("IgnoreOutOfBounds")?;
        let indices = self.vector(indices.into_iter().map(|index| index.expression).collect());
        let atoms = self.vector(atoms);
        let expression = self.node("Accumulate", (bounds, token.name, indices, atoms))?;
        let name = self.fresh("AccumulatorId")?;
        let binding = self.node("BindAccumulator", (name, token.info))?;
        self.statement(vec![binding], expression, scope)?;
        Ok(Accumulator {
            name,
            info: token.info,
        })
    }

    fn accumulate_value(
        &mut self,
        token: &Accumulator,
        indices: Vec<Index>,
        value: &Leaf,
        scope: &mut Scope,
    ) -> Result<Accumulator, OptimizeError> {
        if value.array.is_none() {
            let atom = self.node("Var", value.name)?;
            return self.accumulate(token, indices, vec![atom], scope);
        }
        let source = Imported::Leaf(value.clone());
        let length = self.input_length(&source, scope)?;
        let (domain, index) = self.domain(&length)?;
        let mut lane = Scope::default();
        let row = Self::leaf(self.input_at(&source, &index, &mut lane)?)?;
        let mut indices = indices;
        indices.push(index);
        let next = self.accumulate_value(token, indices, &row, &mut lane)?;
        let mut result = self.contribute(vec![next], domain, lane, scope)?;
        let Some(token) = result.pop() else {
            return Err(output_error("array contribution lost its accumulator"));
        };
        Ok(token)
    }

    fn scatter_components(
        index: &Index,
        value: &Imported<'source>,
        writes: &mut Vec<(Index, Imported<'source>)>,
    ) {
        if let Imported::Product(fields) = value {
            for field in fields {
                Self::scatter_components(index, field, writes);
            }
        } else {
            writes.push((index.clone(), value.clone()));
        }
    }

    fn scatter_values(
        &mut self,
        value: Imported<'source>,
        writes: &mut Vec<(Index, Imported<'source>)>,
    ) -> Result<(), OptimizeError> {
        let Imported::Product(fields) = value else {
            return Err(output_error("scatter callback must return index/value pairs"));
        };
        if let [Imported::Leaf(index), values] = fields.as_slice() {
            if index.array.is_none() {
                let index = self.index_leaf(index)?;
                Self::scatter_components(&index, values, writes);
                return Ok(());
            }
        }
        for field in fields {
            self.scatter_values(field, writes)?;
        }
        Ok(())
    }

    fn scatter(
        &mut self,
        destination: Imported<'source>,
        policy: Value,
        callback: &'source Callback,
        sources: Vec<Imported<'source>>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Some(first) = sources.first() else {
            return Err(output_error("scatter has no input"));
        };
        let length = self.input_length(first, scope)?;
        let (domain, index) = self.domain(&length)?;
        let mut inputs = Vec::new();
        let mut parameters = Vec::new();
        let mut tokens = Vec::new();
        self.accumulator_parameters(&destination, policy, &mut inputs, &mut parameters, &mut tokens)?;
        let mut lane = Scope::default();
        let arguments = sources
            .iter()
            .map(|source| self.input_at(source, &index, &mut lane))
            .collect::<Result<Vec<_>, _>>()?;
        let value = self.callback(callback, arguments, &mut lane)?;
        let mut writes = Vec::new();
        self.scatter_values(value, &mut writes)?;
        if writes.len() != tokens.len() {
            return Err(output_error("scatter results differ from destination fields"));
        }
        let tokens = tokens
            .iter()
            .zip(writes)
            .map(|(token, (index, value))| {
                let value = Self::leaf(value)?;
                self.accumulate_value(token, vec![index], &value, &mut lane)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut body = Scope::default();
        let tokens = self.contribute(tokens, domain, lane, &mut body)?;
        self.finish_accumulator(destination, inputs, parameters, tokens, body, scope)
    }

    fn scatter_loop(
        &mut self,
        initial: Imported<'source>,
        callback: &'source Callback,
        sources: &[Imported<'source>],
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let state_ty = component_type(ty);
        let coordinates = sources.iter().map(|_| vec![0]).collect::<Vec<_>>();
        self.fold_domain(
            sources,
            &coordinates,
            1,
            &mut Vec::new(),
            initial,
            &state_ty,
            scope,
            &mut |this, arguments, state, inner| {
                let value = this.callback(callback, arguments, inner)?;
                let mut writes = Vec::new();
                this.scatter_values(value, &mut writes)?;
                let mut writes = writes.into_iter();
                let next = this.scatter_rows(state, &state_ty, &mut writes, inner)?;
                if writes.next().is_some() {
                    return Err(output_error(
                        "scatter has more writes than destination components",
                    ));
                }
                Ok(next)
            },
        )
    }

    fn scatter_rows(
        &mut self,
        destination: Imported<'source>,
        ty: &Type,
        writes: &mut impl Iterator<Item = (Index, Imported<'source>)>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        if let Imported::Product(fields) = destination {
            let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = ty else {
                return Err(output_error("scatter destination product type missing"));
            };
            if fields.len() != types.len() {
                return Err(output_error("scatter destination product arity mismatch"));
            }
            return fields
                .into_iter()
                .zip(types)
                .map(|(field, ty)| self.scatter_rows(field, ty, writes, scope))
                .collect::<Result<Vec<_>, _>>()
                .map(Imported::Product);
        }
        let Some((index, row)) = writes.next() else {
            return Err(output_error("scatter destination has no write"));
        };
        self.update_if_valid(&destination, &index, ty, scope, |this, destination, scope| {
            this.update_row(destination.clone(), row, &index, scope)
        })
    }

    fn sum_reducer(&mut self) -> Result<Value, OptimizeError> {
        let ty = Type::Constructed(TypeName::Int(32), Vec::new());
        let mut parameters = Vec::new();
        let a = Self::leaf(self.parameter(&ty, &mut parameters)?)?;
        let b = Self::leaf(self.parameter(&ty, &mut parameters)?)?;
        let a = self.node("Var", a.name)?;
        let b = self.node("Var", b.name)?;
        let mut body = Scope::default();
        let result = self.primitive("+", vec![a, b], &ty, true, &mut body)?;
        let body = self.finish(body, &result)?;
        let parameters = self.vector(parameters);
        let element = self.element(&ty)?;
        let result = self.node("ReturnScalar", element)?;
        let returns = self.vector(vec![result]);
        let callback = self.node("FunctionBody", (parameters, returns, body))?;
        let zero = self.constant(0)?;
        let neutral = self.vector(vec![zero.atom]);
        let commutative = self.unit("Commutative")?;
        self.node("Reduction", (commutative, neutral, callback))
    }

    fn filter(
        &mut self,
        callback: &'source Callback,
        input: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let length = self.input_length(&input, scope)?;
        let (domain, position) = self.domain(&length)?;
        let index_ty = Type::Constructed(TypeName::Int(32), Vec::new());
        let bool_ty = Type::Constructed(TypeName::Bool, Vec::new());
        let index_array = types::view_array_with_size(
            &index_ty,
            Type::Constructed(TypeName::SizePlaceholder, Vec::new()),
            types::no_buffer(),
        );
        let flag_array = types::view_array_with_size(
            &bool_ty,
            Type::Constructed(TypeName::SizePlaceholder, Vec::new()),
            types::no_buffer(),
        );
        let prefixes = self.array_value(&index_array, vec![length.clone()])?;
        let flags = self.array_value(&flag_array, vec![length.clone()])?;
        let total = self.scalar(&index_ty)?;
        let mut lane = Scope::default();
        let value = self.input_at(&input, &position, &mut lane)?;
        let predicate = Self::leaf(self.callback(callback, vec![value], &mut lane)?)?;
        let flag = self.node("Var", predicate.name)?;
        let zero = self.constant(0)?;
        let one = self.constant(1)?;
        let count = self.primitive(
            "select",
            vec![zero.atom, one.atom, flag],
            &index_ty,
            true,
            &mut lane,
        )?;
        let count = self.lane_results(&count)?;
        let count = self.vector(count);
        let reducer = self.sum_reducer()?;
        let mode = self.unit("InclusiveScanAndTotal")?;
        let fold = self.node("Fold", (mode, reducer, count))?;
        let flag = self.lane_results(&Imported::Leaf(predicate))?;
        let flag = self.vector(flag);
        let collect = self.node("Collect", flag)?;
        let results = self.vector(vec![fold, collect]);
        let statements = self.vector(lane.statements);
        let body = self.node("ParallelBlock", (statements, results))?;
        let execution = self.unit("Unscheduled")?;
        let expression = self.node("Parallel", (execution, domain, body))?;
        let pattern = vec![
            self.binding(&prefixes)?,
            self.binding(&total)?,
            self.binding(&flags)?,
        ];
        self.statement(pattern, expression, scope)?;
        let count = self.index_leaf(&total)?;
        let mut output_pattern = Vec::new();
        let destination = self.parallel_output(ty, &count, &mut output_pattern)?;
        let destination = self.initialize_scratch(destination, scope)?;
        let policy = self.unit("DisjointWrites")?;
        let mut inputs = Vec::new();
        let mut parameters = Vec::new();
        let mut tokens = Vec::new();
        self.accumulator_parameters(&destination, policy, &mut inputs, &mut parameters, &mut tokens)?;
        let (domain, position) = self.domain(&length)?;
        let mut lane = Scope::default();
        let prefix = Self::leaf(self.index_array(&prefixes, position.clone(), &index_ty, &mut lane)?)?;
        let prefix = self.index_leaf(&prefix)?;
        let index = self.index_binary("-", &prefix, &one, &mut lane)?;
        let flag = Self::leaf(self.index_array(&flags, position.clone(), &bool_ty, &mut lane)?)?;
        let flag = self.node("Var", flag.name)?;
        let ignored = self.constant(-1)?;
        let selected = Self::leaf(self.primitive(
            "select",
            vec![ignored.atom, index.atom, flag],
            &index_ty,
            true,
            &mut lane,
        )?)?;
        let index = self.index_leaf(&selected)?;
        let value = self.input_at(&input, &position, &mut lane)?;
        let mut atoms = Vec::new();
        self.scalar_atoms(&value, &mut atoms)?;
        if tokens.len() != atoms.len() {
            return Err(output_error("filter result fields differ from destination"));
        }
        let tokens = tokens
            .iter()
            .zip(atoms)
            .map(|(token, atom)| self.accumulate(token, vec![index.clone()], vec![atom], &mut lane))
            .collect::<Result<Vec<_>, _>>()?;
        let mut body = Scope::default();
        let tokens = self.contribute(tokens, domain, lane, &mut body)?;
        self.finish_accumulator(destination, inputs, parameters, tokens, body, scope)
    }

    fn scratch_like(
        &mut self,
        value: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match value {
            Imported::Leaf(source) => {
                let Some(array) = source.array else {
                    return Err(output_error("scratch requires array components"));
                };
                let output = self.array_leaf(&source.ty, array)?;
                self.initialize_scratch(Imported::Leaf(output), scope)
            }
            Imported::Product(fields) => fields.into_iter()
                .map(|field| self.scratch_like(field, scope))
                .collect::<Result<_, _>>().map(Imported::Product),
            Imported::Ragged { offsets, data } => Ok(Imported::Ragged {
                // Offsets describe shape; only the payload is uninitialized.
                offsets: self.snapshot(offsets, scope)?,
                data: Box::new(self.scratch_like(*data, scope)?),
            }),
            Imported::Callable(_, _) => Err(output_error("scratch requires an array")),
        }
    }

    fn initialize_scratch(
        &mut self,
        value: Imported<'source>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        value.map(&mut |leaf| {
            let Some(array) = &leaf.array else {
                return Err(output_error("scratch has no array binding"));
            };
            let shape = self.vector(array.dimensions.iter().map(|length| length.expression).collect());
            let generator = self.node("Uninitialized", (array.element, shape))?;
            let expression = self.node("GenerateArray", generator)?;
            let binding = self.binding(&leaf)?;
            self.statement(vec![binding], expression, scope)?;
            Ok(leaf)
        })
    }

    fn leaf_arrays<'a>(
        value: &'a Imported<'source>,
        leaves: &mut Vec<&'a Leaf>,
    ) -> Result<(), OptimizeError> {
        for leaf in value.leaves()? {
            if leaf.array.is_none() {
                return Err(output_error("expected array components"));
            }
            leaves.push(leaf);
        }
        Ok(())
    }

    fn combine_lambda(&mut self, callback: &'source Callback) -> Result<Value, OptimizeError> {
        let mut parameters = Vec::new();
        let arguments = callback
            .lam
            .params
            .iter()
            .map(|(_, ty)| self.parameter(ty, &mut parameters))
            .collect::<Result<Vec<_>, _>>()?;
        let mut body = Scope::default();
        let value = self.callback(callback, arguments, &mut body)?;
        let body = self.finish(body, &value)?;
        let parameters = self.vector(parameters);
        let mut returns = Vec::new();
        self.result_pattern(&callback.lam.ret_ty, &mut Vec::new(), &mut returns)?;
        let returns = self.vector(returns);
        self.node("FunctionBody", (parameters, returns, body))
    }

    fn indexed_fold_loop(
        &mut self,
        combine: &'source Callback,
        initial: Imported<'source>,
        indices: Imported<'source>,
        values: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let count = self.input_length(&indices, scope)?;
        self.counted_state(&count, initial, ty, scope, |this, position, state, inner| {
            let index = Self::leaf(this.input_at(&indices, &position, inner)?)?;
            let index = this.index_leaf(&index)?;
            this.update_if_valid(&state, &index, ty, inner, |this, state, scope| {
                let previous = this.input_at(state, &index, scope)?;
                let value = this.input_at(&values, &position, scope)?;
                let combined = this.callback(combine, vec![previous, value], scope)?;
                this.update_row(state.clone(), combined, &index, scope)
            })
        })
    }

    fn indexed_operator(
        &mut self,
        operator: &'source Operator,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        match operator {
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => {
                let destination = self.visit(dest, scope)?;
                let destination = self.copy_arrays(destination, scope)?;
                self.visit(ne, scope)?;
                let indices = self.array(indices, scope)?;
                let values = self.array(values, scope)?;
                if destination
                    .leaves()?
                    .iter()
                    .any(|leaf| leaf.array.as_ref().is_some_and(|array| array.dimensions.len() > 1))
                    || contains_ragged(&component_type(ty))
                {
                    return self.indexed_fold_loop(op, destination, indices, values, ty, scope);
                }
                let length = self.input_length(&indices, scope)?;
                let (domain, position) = self.domain(&length)?;
                let combine = self.combine_lambda(op)?;
                let policy = self.node("CombineWrites", combine)?;
                let mut arrays = Vec::new();
                Self::leaf_arrays(&destination, &mut arrays)?;
                let Some(first) = arrays.first().and_then(|leaf| leaf.array.as_ref()) else {
                    return Err(output_error("indexed reduction destination is empty"));
                };
                let shape = self.vector(first.dimensions.iter().map(|length| length.expression).collect());
                let elements = arrays
                    .iter()
                    .map(|leaf| leaf.array.as_ref().map(|array| array.element))
                    .collect::<Option<Vec<_>>>();
                let Some(elements) = elements else {
                    return Err(output_error("indexed reduction array metadata missing"));
                };
                let elements = self.vector(elements);
                let info = self.node("AccumulatorInfo", (shape, elements))?;
                let name = self.fresh("AccumulatorId")?;
                let token = Accumulator { name, info };
                let binding = self.node("BindAccumulator", (name, info))?;
                let diet = self.unit("Consume")?;
                let parameter = self.node("Parameter", (binding, diet))?;
                let arrays = self.vector(arrays.into_iter().map(|leaf| leaf.name).collect());
                let input = self.node("AccumulatorInput", (arrays, policy))?;
                let mut lane = Scope::default();
                let index = Self::leaf(self.input_at(&indices, &position, &mut lane)?)?;
                let index = self.index_leaf(&index)?;
                let values = self.input_at(&values, &position, &mut lane)?;
                let mut atoms = Vec::new();
                self.scalar_atoms(&values, &mut atoms)?;
                let token = self.accumulate(&token, vec![index], atoms, &mut lane)?;
                let mut body = Scope::default();
                let tokens = self.contribute(vec![token], domain, lane, &mut body)?;
                self.finish_accumulator(destination, vec![input], vec![parameter], tokens, body, scope)
            }
            SoacOp::BucketScatter {
                dest,
                lam,
                inputs,
                input_dimensions,
                domain_rank,
            } => {
                let destination = self.visit(dest, scope)?;
                let destination = self.copy_arrays(destination, scope)?;
                let inputs =
                    inputs.iter().map(|input| self.array(input, scope)).collect::<Result<Vec<_>, _>>()?;
                self.bucket_scatter(
                    destination,
                    lam,
                    inputs,
                    input_dimensions,
                    *domain_rank,
                    ty,
                    scope,
                )
            }
            _ => Err(output_error("indexed importer received another operator")),
        }
    }

    fn fill_array(
        &mut self,
        ty: &Type,
        lengths: Vec<Index>,
        bits: i64,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let leaf = self.array_value(ty, lengths)?;
        let Some(element) = ty.elem_type() else {
            return Err(output_error("fill element missing"));
        };
        let primitive = self.primitive_type(element)?;
        let value = self.node("LiteralBits", (primitive, bits))?;
        self.generate_array(leaf, scope, |_, _, _| Ok(value))
    }

    fn token_body(&mut self, body: Scope, tokens: &[Accumulator]) -> Result<Value, OptimizeError> {
        let mut results = Vec::new();
        for token in tokens {
            let operand = self.node("AccOperand", token.name)?;
            let certs = self.vector(Vec::new());
            results.push(self.node("Yield", (certs, operand))?);
        }
        let statements = self.vector(body.statements);
        let results = self.vector(results);
        self.node("Block", (statements, results))
    }

    fn acc_if(
        &mut self,
        condition: Value,
        before: &[Accumulator],
        yes: impl FnOnce(&mut Self, &mut Scope) -> Result<Vec<Accumulator>, OptimizeError>,
        no: impl FnOnce(&mut Self, &mut Scope) -> Result<Vec<Accumulator>, OptimizeError>,
        scope: &mut Scope,
    ) -> Result<Vec<Accumulator>, OptimizeError> {
        let mut left = Scope::default();
        let mut right = Scope::default();
        let yes = yes(self, &mut left)?;
        let no = no(self, &mut right)?;
        if yes.len() != before.len() || no.len() != before.len() {
            return Err(output_error("branch changed accumulator arity"));
        }
        let yes = self.token_body(left, &yes)?;
        let no = self.token_body(right, &no)?;
        let mut returns = Vec::new();
        let mut pattern = Vec::new();
        let mut output = Vec::new();
        for token in before {
            let name = self.fresh("AccumulatorId")?;
            pattern.push(self.node("BindAccumulator", (name, token.info))?);
            returns.push(self.node("ReturnAccumulator", token.info)?);
            output.push(Accumulator {
                name,
                info: token.info,
            });
        }
        let returns = self.vector(returns);
        let expression = self.node("If", (condition, returns, yes, no))?;
        self.statement(pattern, expression, scope)?;
        Ok(output)
    }

    fn predicate(
        &mut self,
        opcode: &str,
        a: Value,
        b: Value,
        scope: &mut Scope,
    ) -> Result<Value, OptimizeError> {
        let ty = Type::Constructed(TypeName::Bool, Vec::new());
        let value = Self::leaf(self.primitive(opcode, vec![a, b], &ty, true, scope)?)?;
        self.node("Var", value.name)
    }

    fn reserve(
        &mut self,
        token: &Accumulator,
        index: &Index,
        scope: &mut Scope,
    ) -> Result<(Accumulator, Index), OptimizeError> {
        let ty = Type::Constructed(TypeName::UInt(32), Vec::new());
        let previous = self.scalar(&ty)?;
        let primitive = self.primitive_type(&ty)?;
        let one = self.node("LiteralBits", (primitive, 1i64))?;
        let indices = self.vector(vec![index.expression]);
        let expression = self.node("Reserve", (token.name, indices, one))?;
        let name = self.fresh("AccumulatorId")?;
        let pattern = vec![
            self.node("BindAccumulator", (name, token.info))?,
            self.binding(&previous)?,
        ];
        self.statement(pattern, expression, scope)?;
        Ok((
            Accumulator {
                name,
                info: token.info,
            },
            self.index_leaf(&previous)?,
        ))
    }

    fn bucket_loop(
        &mut self,
        destination: Imported<'source>,
        callback: &'source Callback,
        sources: &[Imported<'source>],
        coordinates: &[Vec<u8>],
        rank: u8,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
            return Err(output_error("bucket result is not a tuple"));
        };
        let [_, counts_ty, overflow_ty] = fields.as_slice() else {
            return Err(output_error("bucket result arity"));
        };
        let buckets = self.input_length(&destination, scope)?;
        let counts = self.fill_array(counts_ty, vec![buckets.clone()], 0, scope)?;
        let flag = self.literal(overflow_ty, 0, scope)?;
        let initial = Imported::Product(vec![destination, counts, flag]);
        self.fold_domain(
            sources,
            coordinates,
            rank,
            &mut Vec::new(),
            initial,
            ty,
            scope,
            &mut |this, arguments, state, scope| {
                let contribution = this.callback(callback, arguments, scope)?;
                let Imported::Product(fields) = contribution else {
                    return Err(output_error("bucket callback must return active/key/value"));
                };
                let [Imported::Leaf(active), Imported::Leaf(key), row] = fields.as_slice() else {
                    return Err(output_error("bucket callback must return active/key/value"));
                };
                let active = this.node("Var", active.name)?;
                let key = this.index_leaf(key)?;
                this.branch_value(
                    active,
                    ty,
                    scope,
                    |this, scope| this.bucket_row(state.clone(), &key, row.clone(), ty, scope),
                    |_, _| Ok(state.clone()),
                )
            },
        )
    }

    fn bucket_row(
        &mut self,
        state: Imported<'source>,
        key: &Index,
        row: Imported<'source>,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Imported::Product(fields) = &state else {
            return Err(output_error("bucket state is not a product"));
        };
        let [destination, counts, overflow] = fields.as_slice() else {
            return Err(output_error("bucket state arity"));
        };
        let overflow_ty = Self::leaf(overflow.clone())?.ty;
        let flag = self.literal(&overflow_ty, 1, scope)?;
        let buckets = self.input_length(destination, scope)?;
        let zero = self.constant(0)?;
        let below = self.predicate("<", key.atom, buckets.atom, scope)?;
        let nonnegative = self.predicate(">=", key.atom, zero.atom, scope)?;
        let valid = self.predicate("&&", below, nonnegative, scope)?;
        self.branch_value(
            valid,
            ty,
            scope,
            |this, scope| {
                let bucket = this.input_at(destination, key, scope)?;
                let capacity = this.input_length(&bucket, scope)?;
                let counter = Self::leaf(this.input_at(counts, key, scope)?)?;
                let slot = this.index_leaf(&counter)?;
                let unsigned = this.primitive_type(&counter.ty)?;
                let one = this.node("LiteralBits", (unsigned, 1i64))?;
                let next = this.primitive("+", vec![slot.atom, one], &counter.ty, true, scope)?;
                let counts = this.update_row(counts.clone(), next, key, scope)?;
                let fits = this.predicate("<", slot.atom, capacity.atom, scope)?;
                this.branch_value(
                    fits,
                    ty,
                    scope,
                    |this, scope| {
                        let bucket = this.update_row(bucket, row, &slot, scope)?;
                        let destination = this.update_row(destination.clone(), bucket, key, scope)?;
                        Ok(Imported::Product(vec![
                            destination,
                            counts.clone(),
                            overflow.clone(),
                        ]))
                    },
                    |_, _| {
                        Ok(Imported::Product(vec![
                            destination.clone(),
                            counts.clone(),
                            flag.clone(),
                        ]))
                    },
                )
            },
            |_, _| {
                Ok(Imported::Product(vec![
                    destination.clone(),
                    counts.clone(),
                    flag.clone(),
                ]))
            },
        )
    }

    fn bucket_scatter(
        &mut self,
        destination: Imported<'source>,
        callback: &'source Callback,
        sources: Vec<Imported<'source>>,
        coordinates: &[Vec<u8>],
        rank: u8,
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
            return Err(output_error("bucket results are not a product"));
        };
        let [_, counts_ty, overflow_ty] = fields.as_slice() else {
            return Err(output_error("bucket result needs arrays, counts, overflow"));
        };
        if destination.has_segments() || sources.iter().any(Imported::has_segments) {
            return self.bucket_loop(destination, callback, &sources, coordinates, rank, ty, scope);
        }
        let mut leaves = Vec::new();
        Self::leaf_arrays(&destination, &mut leaves)?;
        let Some(array) = leaves.first().and_then(|leaf| leaf.array.as_ref()) else {
            return Err(output_error("bucket destination is empty"));
        };
        let [buckets, capacity, ..] = array.dimensions.as_slice() else {
            return Err(output_error(
                "bucket destination needs bucket and capacity dimensions",
            ));
        };
        let buckets = buckets.clone();
        let capacity = capacity.clone();
        let counts = self.fill_array(counts_ty, vec![buckets.clone()], 0, scope)?;
        let one = self.constant(1)?;
        let overflow_array_ty = types::view_array_with_size(
            overflow_ty,
            Type::Constructed(TypeName::Size(1), Vec::new()),
            types::no_buffer(),
        );
        let overflow = self.fill_array(&overflow_array_ty, vec![one.clone()], 0, scope)?;
        let mut inputs = Vec::new();
        let mut parameters = Vec::new();
        let mut tokens = Vec::new();
        let writes = self.unit("IdenticalWrites")?;
        self.accumulator_parameters(&destination, writes, &mut inputs, &mut parameters, &mut tokens)?;
        let data_count = tokens.len();
        let adding = self.unit("FetchAddWrites")?;
        self.accumulator_parameters(&counts, adding, &mut inputs, &mut parameters, &mut tokens)?;
        self.accumulator_parameters(&overflow, writes, &mut inputs, &mut parameters, &mut tokens)?;
        let mut positions = Vec::new();
        let mut axes = Vec::new();
        for axis in 0..rank {
            let mut length = None;
            for (source, mapping) in sources.iter().zip(coordinates) {
                if let Some(dimension) = mapping.iter().position(|&value| value == axis) {
                    let mut leaves = Vec::new();
                    Self::leaf_arrays(source, &mut leaves)?;
                    length = leaves
                        .first()
                        .and_then(|leaf| leaf.array.as_ref())
                        .and_then(|array| array.dimensions.get(dimension))
                        .cloned();
                    if length.is_some() {
                        break;
                    }
                }
            }
            let Some(length) = length else {
                return Err(output_error("bucket domain axis has no input extent"));
            };
            let name = self.fresh("ValueId")?;
            positions.push(Index {
                expression: self.node("IndexValue", name)?,
                atom: self.node("Var", name)?,
            });
            axes.push(self.node("DenseAxis", (name, length.expression))?);
        }
        let axes = self.vector(axes);
        let flat = self.fresh("ValueId")?;
        let domain = self.node("IterationSpace", (flat, axes))?;
        let mut lane = Scope::default();
        let mut arguments = Vec::new();
        for (source, mapping) in sources.iter().zip(coordinates) {
            let mut value = source.clone();
            for &axis in mapping {
                let Some(position) = positions.get(usize::from(axis)) else {
                    return Err(output_error("bucket coordinate is outside its domain"));
                };
                value = self.input_at(&value, position, &mut lane)?;
            }
            arguments.push(value);
        }
        let value = self.callback(callback, arguments, &mut lane)?;
        let Imported::Product(fields) = value else {
            return Err(output_error("bucket callback must return active/key/value"));
        };
        let [Imported::Leaf(active), Imported::Leaf(key), values] = fields.as_slice() else {
            return Err(output_error("bucket callback must return active/key/value"));
        };
        let active = self.node("Var", active.name)?;
        let key = self.index_leaf(key)?;
        let contributions = values.leaves()?;
        if contributions.len() != data_count {
            return Err(output_error("bucket callback fields differ from destination"));
        }
        let zero = self.constant(0)?;
        let overflow_primitive = self.primitive_type(overflow_ty)?;
        let flag = self.node("LiteralBits", (overflow_primitive, 1i64))?;
        let updated = self.acc_if(
            active,
            &tokens,
            |this, lane| {
                let below = this.predicate("<", key.atom, buckets.atom, lane)?;
                let nonnegative = this.predicate(">=", key.atom, zero.atom, lane)?;
                let valid = this.predicate("&&", below, nonnegative, lane)?;
                this.acc_if(
                    valid,
                    &tokens,
                    |this, lane| {
                        let (counter, slot) = this.reserve(&tokens[data_count], &key, lane)?;
                        let mut updated = tokens.clone();
                        updated[data_count] = counter;
                        for (i, value) in contributions.iter().enumerate() {
                            updated[i] = this.accumulate_value(
                                &tokens[i],
                                vec![key.clone(), slot.clone()],
                                value,
                                lane,
                            )?;
                        }
                        let full = this.predicate(">=", slot.atom, capacity.atom, lane)?;
                        this.acc_if(
                            full,
                            &updated,
                            |this, lane| {
                                let mut overflowed = updated.clone();
                                overflowed[data_count + 1] = this.accumulate(
                                    &updated[data_count + 1],
                                    vec![zero.clone()],
                                    vec![flag],
                                    lane,
                                )?;
                                Ok(overflowed)
                            },
                            |_, _| Ok(updated.clone()),
                            lane,
                        )
                    },
                    |this, lane| {
                        let mut overflowed = tokens.clone();
                        overflowed[data_count + 1] =
                            this.accumulate(&tokens[data_count + 1], vec![zero.clone()], vec![flag], lane)?;
                        Ok(overflowed)
                    },
                    lane,
                )
            },
            |_, _| Ok(tokens.clone()),
            &mut lane,
        )?;
        let mut body = Scope::default();
        let tokens = self.contribute(updated, domain, lane, &mut body)?;
        let result = self.finish_accumulator(
            Imported::Product(vec![destination, counts, overflow]),
            inputs,
            parameters,
            tokens,
            body,
            scope,
        )?;
        let Imported::Product(mut fields) = result else {
            return Err(output_error("bucket result structure missing"));
        };
        let Some(overflow) = fields.pop() else {
            return Err(output_error("bucket overflow result missing"));
        };
        let overflow = Self::leaf(overflow)?;
        fields.push(self.index_array(&overflow, zero, overflow_ty, scope)?);
        Ok(Imported::Product(fields))
    }
}
