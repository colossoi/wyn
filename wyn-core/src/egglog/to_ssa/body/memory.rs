//! Emit statements in lexical order; descriptor arithmetic comes from bindings.
use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::builtins::catalog;
use crate::egglog::query::Query;
use crate::egglog::to_ssa::{element_type, host, primitive_type, runtime_binding};
use crate::op::{BinaryOperator, OpTag, UnaryOperator};
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::ssa::types::{AtomicOp, InstKind, PlaceId};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::{LookupSet, ResourceAccess};
use egglog_engine::sort::S;

impl Body<'_, '_, '_> {
    pub fn named(&self, name: Value) -> Result<Typed, OptimizeError> {
        let Some(value) = self.values.get(&name) else {
            return Err(error(format!("unbound emitted value {name:?}")));
        };
        Ok(value.clone())
    }

    pub fn atom(&mut self, atom: Value) -> Result<Typed, OptimizeError> {
        let (kind, fields) = self.query().application(atom, &["Var", "LiteralBits"])?;
        if kind == "Var" {
            return self.named(fields[0]);
        }
        let ty = primitive_type(self.query(), fields[0])?;
        let bits = self.compiler.program.graph.value_to_base::<i64>(fields[1]);
        if ty == types::unit() {
            return self.op(OpTag::Unit, vec![], ty);
        }
        self.literal(bits, &ty)
    }

    pub fn index_expression(&mut self, expression: Value) -> Result<Typed, OptimizeError> {
        let (kind, fields) = self.query().application(
            expression,
            &[
                "IndexConstant",
                "IndexValue",
                "IndexAdd",
                "IndexSub",
                "IndexMul",
                "IndexDiv",
                "IndexMod",
                "IndexMin",
                "IndexMax",
            ],
        )?;
        let ty = types::i32();
        if kind == "IndexConstant" {
            let n = self.compiler.program.graph.value_to_base::<i64>(fields[0]);
            i32::try_from(n).map_err(|_| error("memory index exceeds the shader address range"))?;
            return self.literal(n, &ty);
        }
        if kind == "IndexValue" {
            let value = self.named(fields[0])?;
            return self.cast(value, &ty);
        }
        let a = self.index_expression(fields[0])?;
        let b = self.index_expression(fields[1])?;
        if matches!(kind, "IndexMin" | "IndexMax") {
            let compare = if kind == "IndexMin" { BinaryOperator::Less } else { BinaryOperator::Greater };
            let condition = self.binary(compare, a.clone(), b.clone())?;
            return self.select(condition, a, b);
        }
        let operator = match kind {
            "IndexAdd" => BinaryOperator::Add,
            "IndexSub" => BinaryOperator::Subtract,
            "IndexMul" => BinaryOperator::Multiply,
            "IndexDiv" => BinaryOperator::Divide,
            "IndexMod" => BinaryOperator::Remainder,
            _ => unreachable!(),
        };
        self.binary(operator, a, b)
    }

    pub fn operand(&mut self, operand: Value) -> Result<Option<Typed>, OptimizeError> {
        let (kind, fields) = self.query().application(
            operand,
            &["ValueOperand", "IndexOperand", "BlockOperand", "AccOperand"],
        )?;
        match kind {
            "AccOperand" => Ok(None),
            "BlockOperand" => self.named(fields[0]).map(Some),
            "IndexOperand" => self.index_expression(fields[0]).map(Some),
            "ValueOperand" => {
                if let Some(var) = self.query().enode("Var", fields[0])? {
                    let info = self.query().required("BindingInfo", (self.function, var[0]))?;
                    if self.query().enode("ArrayInfo", info)?.is_some() {
                        return Ok(None);
                    }
                }
                self.atom(fields[0]).map(Some)
            }
            _ => unreachable!(),
        }
    }

    pub fn results(&mut self, results: Value) -> Result<Vec<Typed>, OptimizeError> {
        let mut values = Vec::new();
        for result in self.query().vector(results)? {
            let fields = self.query().fields("Yield", result)?;
            if let Some(value) = self.operand(fields[1])? {
                values.push(value);
            }
        }
        Ok(values)
    }

    pub fn block(&mut self, block: Value) -> Result<Vec<Typed>, OptimizeError> {
        let fields = self.query().fields("Block", block)?;
        self.statements(fields[0])?;
        self.results(fields[1])
    }

    pub fn statements(&mut self, statements: Value) -> Result<(), OptimizeError> {
        for statement in self.query().vector(statements)? {
            self.statement(statement)?;
        }
        Ok(())
    }

    pub(in crate::egglog::to_ssa) fn bind(
        &mut self,
        pattern: &[Value],
        values: Vec<Typed>,
    ) -> Result<(), OptimizeError> {
        let mut values = values.into_iter();
        for &binding in pattern {
            if let Some((name, ty)) = runtime_binding(self.query(), binding)? {
                let Some(value) = values.next() else {
                    return Err(error("statement result arity is too small"));
                };
                let value = self.cast(value, &ty)?;
                self.values.insert(name, value);
            }
        }
        if values.next().is_some() {
            return Err(error("statement result arity is too large"));
        }
        Ok(())
    }

    pub fn lambda(&mut self, lambda: Value, arguments: Vec<Typed>) -> Result<Vec<Typed>, OptimizeError> {
        let fields = self.query().fields("FunctionBody", lambda)?;
        let checkpoint = self.values.checkpoint();
        let pattern = self
            .query()
            .vector(fields[0])?
            .into_iter()
            .map(|param| Ok(self.query().fields("Parameter", param)?[0]))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        self.bind(&pattern, arguments)?;
        let result = self.block(fields[2]);
        self.values.restore(checkpoint);
        result
    }

    fn unpack(&mut self, value: Typed) -> Result<Vec<Typed>, OptimizeError> {
        if value.ty == types::unit() {
            return Ok(Vec::new());
        }
        let Type::Constructed(TypeName::Tuple(_), types) = &value.ty else {
            return Ok(vec![value]);
        };
        (0..types.len()).map(|i| self.field(value.clone(), i)).collect()
    }

    pub(in crate::egglog::to_ssa) fn statement(&mut self, statement: Value) -> Result<(), OptimizeError> {
        host::remember(
            Query(&self.compiler.program.graph),
            &mut self.host_values,
            statement,
        )?;
        let fields = self.query().fields("Let", statement)?;
        let pattern = self.query().vector(fields[1])?;
        let expression = fields[3];
        let (kind, arguments) = self.query().application(
            expression,
            &[
                "Use",
                "PrimOp",
                "IndexOp",
                "Assert",
                "Call",
                "If",
                "Loop",
                "Parallel",
                "Launch",
                "WithAcc",
                "Alloc",
                "View",
                "Reshape",
                "Permute",
                "Load",
                "GenerateArray",
                "Copy",
                "Update",
                "Store",
                "Accumulate",
                "Reserve",
                "Barrier",
            ],
        )?;
        if (kind == "Launch"
            || (kind == "Parallel" && self.query().enode("DeviceKernel", arguments[0])?.is_some()))
            && self.active_dispatch != Some(statement)
        {
            return Err(error("device parallel region requires a dispatch entry"));
        }
        let values = match kind {
            "Launch" => self.block(arguments[1])?,
            "Use" => self.operand(arguments[0])?.into_iter().collect(),
            "IndexOp" => vec![self.index_expression(arguments[0])?],
            "PrimOp" => {
                let opcode = self.compiler.program.graph.value_to_base::<S>(arguments[0]).to_string();
                let args = self
                    .query()
                    .vector(arguments[1])?
                    .into_iter()
                    .map(|atom| self.atom(atom))
                    .collect::<Result<Vec<_>, _>>()?;
                let [binding] = pattern.as_slice() else {
                    return Err(error("scalar instruction must bind one result"));
                };
                let Some((_, ty)) = runtime_binding(self.query(), *binding)? else {
                    return Err(error("scalar result is not a value"));
                };
                vec![self.primitive(&opcode, args, ty)?]
            }
            "If" => {
                let condition = self.atom(arguments[0])?;
                let result = self.branch(
                    condition,
                    |body| {
                        let values = body.block(arguments[2])?;
                        body.tuple(values)
                    },
                    |body| {
                        let values = body.block(arguments[3])?;
                        body.tuple(values)
                    },
                )?;
                self.unpack(result)?
            }
            "Loop" => {
                let mut params = Vec::new();
                let mut initial = Vec::new();
                for merge in self.query().vector(arguments[0])? {
                    let fields = self.query().fields("Carry", merge)?;
                    let param = self.query().fields("Parameter", fields[0])?;
                    if let Some((name, _)) = runtime_binding(self.query(), param[0])? {
                        let Some(value) = self.operand(fields[1])? else {
                            return Err(error("loop carry has no runtime value"));
                        };
                        params.push(name);
                        initial.push(value);
                    }
                }
                let install = |body: &mut Self, state: &[Typed]| {
                    for (&name, value) in params.iter().zip(state) {
                        body.values.insert(name, value.clone());
                    }
                };
                if let Some(form) = self.query().enode("Counted", arguments[1])? {
                    let start = self.index_expression(form[1])?;
                    let count = self.index_expression(form[2])?;
                    let step = self.index_expression(form[3])?;
                    self.counted(
                        Self::number(0),
                        count,
                        Self::number(1),
                        initial,
                        |body, ordinal, state| {
                            install(body, &state);
                            let ordinal = body.cast(ordinal, &start.ty)?;
                            let offset = body.binary(BinaryOperator::Multiply, ordinal, step)?;
                            let index = body.binary(BinaryOperator::Add, start, offset)?;
                            body.values.insert(form[0], index);
                            body.block(arguments[3])
                        },
                    )?
                } else {
                    let form = self.query().fields("While", arguments[1])?;
                    self.while_loop(
                        initial,
                        |body, state| {
                            install(body, state);
                            body.named(form[0])
                        },
                        |body, state| {
                            install(body, &state);
                            body.block(arguments[3])
                        },
                    )?
                }
            }
            "Call" => {
                let mut args = Vec::new();
                for arg in self.query().vector(arguments[1])? {
                    let arg = self.query().fields("Pass", arg)?;
                    if let Some(value) = self.operand(arg[0])? {
                        args.push(value);
                    }
                }
                let (id, ty) = self.compiler.function(arguments[0])?;
                let result = self.op(OpTag::Call(id), args, ty)?;
                self.unpack(result)?
            }
            "View" | "Reshape" | "Permute" => Vec::new(),
            "Assert" => {
                let condition = self.atom(arguments[0])?;
                self.require_condition(condition)?;
                Vec::new()
            }
            "Barrier" => {
                self.workgroup_barrier()?;
                Vec::new()
            }
            _ => self.memory_statement(kind, &arguments, &pattern)?,
        };
        self.bind(&pattern, values)
    }

    fn primitive(&mut self, opcode: &str, arguments: Vec<Typed>, ty: Type) -> Result<Typed, OptimizeError> {
        if opcode == "vector-index" {
            let [vector, index]: [Typed; 2] =
                arguments.try_into().map_err(|_| error("vector indexing arity"))?;
            return self.op(OpTag::Index, vec![vector, index], ty);
        }
        if opcode == "index-cast" {
            let [value]: [Typed; 1] = arguments.try_into().map_err(|_| error("index conversion arity"))?;
            return self.cast(value, &ty);
        }
        if opcode == "select" {
            let [no, yes, condition] = arguments.as_slice() else {
                return Err(error("select requires a condition and two values"));
            };
            return self.select(condition.clone(), yes.clone(), no.clone());
        }
        let tag = match opcode {
            "unit" => OpTag::Unit,
            "vector" => OpTag::Vector,
            "tuple" => OpTag::Tuple,
            _ if opcode.starts_with("project:") => OpTag::Project {
                index: opcode[8..].parse().map_err(|_| error("invalid vector projection"))?,
            },
            _ => {
                if let Some((name, overload)) =
                    opcode.strip_prefix("builtin:").and_then(|name| name.rsplit_once(':'))
                {
                    let Some(builtin) = catalog().lookup_by_any_name(name) else {
                        return Err(error(format!("unknown primitive {opcode}")));
                    };
                    OpTag::Intrinsic {
                        id: builtin.id,
                        overload_idx: overload.parse().map_err(|_| error("invalid builtin overload"))?,
                    }
                } else if let Some(builtin) = catalog().lookup_by_any_name(opcode) {
                    OpTag::Intrinsic {
                        id: builtin.id,
                        overload_idx: 0,
                    }
                } else if arguments.len() == 1 {
                    OpTag::UnaryOp(
                        UnaryOperator::try_from(opcode)
                            .map_err(|_| error(format!("unknown unary primitive {opcode}")))?,
                    )
                } else {
                    OpTag::BinOp(
                        BinaryOperator::try_from(opcode)
                            .map_err(|_| error(format!("unknown binary primitive {opcode}")))?,
                    )
                }
            }
        };
        self.op(tag, arguments, ty)
    }
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn storage_roots(
        &self,
        name: Value,
    ) -> Result<Vec<(Value, i32, Typed)>, OptimizeError> {
        let mut roots = Vec::new();
        let mut pending = vec![name];
        let mut visited = LookupSet::default();
        while let Some(name) = pending.pop() {
            if !visited.insert(name) {
                continue;
            }
            if let Some(&tag) = self.compiler.block_ids.get(&name) {
                if let Some(storage) = self.memories.get(&tag) {
                    roots.push((name, tag, storage.clone()));
                    continue;
                }
            }
            self.query().for_each("BlockAlias", |row| {
                if row[0] == self.function && row[1] == name {
                    pending.push(row[2]);
                }
                Ok(())
            })?;
        }
        Ok(roots)
    }

    pub fn register_memory(&mut self, name: Value, mut storage: Typed) -> Result<Typed, OptimizeError> {
        if storage.ty.array_variant().is_some_and(types::is_array_variant_composite)
            && !self.local_arrays.contains_key(&storage.value)
        {
            let array = self.local_array(storage.ty.clone())?;
            let Some(&place) = self.local_arrays.get(&array.value) else {
                return Err(error("local memory place missing"));
            };
            self.builder
                .push_void_inst(InstKind::Store {
                    place,
                    value: storage.value,
                })
                .map_err(builder_error)?;
            storage = array;
        }
        let next =
            i32::try_from(self.compiler.block_ids.len()).map_err(|_| error("too many memory blocks"))?;
        let id = *self.compiler.block_ids.entry(name).or_insert(next);
        self.memories.insert(id, storage);
        self.literal(i64::from(id), &types::i32())
    }

    pub(in crate::egglog::to_ssa) fn array_descriptor(
        &self,
        name: Value,
    ) -> Result<(Type, Value, Vec<Value>), OptimizeError> {
        let info = self.query().required("BindingInfo", (self.function, name))?;
        let info = self.query().fields("ArrayInfo", info)?;
        let storage = self.query().fields("ArrayIn", info[1])?;
        let layout = self.query().fields("Layout", storage[1])?;
        Ok((element_type(self.query(), info[0])?, storage[0], layout))
    }

    pub fn array_shape(&mut self, name: Value) -> Result<Vec<Typed>, OptimizeError> {
        let (_, _, layout) = self.array_descriptor(name)?;
        self.query()
            .vector(layout[1])?
            .into_iter()
            .map(|dim| {
                let fields = self.query().fields("Dim", dim)?;
                self.index_expression(fields[1])
            })
            .collect()
    }

    pub(in crate::egglog::to_ssa) fn array_address(
        &mut self,
        name: Value,
        indices: &[Typed],
    ) -> Result<(Typed, Typed, Type), OptimizeError> {
        let (ty, block, layout) = self.array_descriptor(name)?;
        let handle = self.named(block)?;
        let dimensions = self.query().vector(layout[1])?;
        if indices.len() != dimensions.len() {
            return Err(error("memory access rank mismatch"));
        }
        let mut offset = self.index_expression(layout[0])?;
        for (dimension, index) in dimensions.into_iter().zip(indices) {
            let dim = self.query().fields("Dim", dimension)?;
            let stride = self.index_expression(dim[0])?;
            let index = self.cast(index.clone(), &stride.ty)?;
            let delta = self.binary(BinaryOperator::Multiply, stride, index)?;
            offset = self.binary(BinaryOperator::Add, offset, delta)?;
        }
        Ok((handle, offset, ty))
    }

    // Block-valued joins select among typed storage roots. No pointer phi or
    // default value is introduced by the target's finite block dispatch.
    pub(in crate::egglog::to_ssa) fn access(
        &mut self,
        handle: Typed,
        index: Typed,
        element: &Type,
        mode: ResourceAccess,
        emit: impl Fn(&mut Self, PlaceId, Type, bool) -> Result<Typed, OptimizeError>,
    ) -> Result<Typed, OptimizeError> {
        if let Some(id) = self.integer_literal(&handle) {
            let Some(storage) = i32::try_from(id).ok().and_then(|id| self.memories.get(&id)).cloned()
            else {
                return Err(error("invalid memory block handle"));
            };
            self.storage_access(&storage.ty, mode);
            let private = self.local_arrays.contains_key(&storage.value);
            let (place, ty) = self.memory_place(storage, index, element)?;
            return emit(self, place, ty, private);
        }
        let physical = storage_value_type(element);
        let mut choices = self
            .memories
            .iter()
            .filter(|(_, storage)| {
                memory_element(&storage.ty).is_some_and(|ty| compatible_element(ty, &physical))
            })
            .map(|(&i, storage)| (i, storage.clone()))
            .collect::<Vec<_>>();
        choices.sort_by_key(|(id, _)| *id);
        if choices.is_empty() {
            return Err(error("memory join has no compatible storage"));
        }
        for (_, storage) in &choices {
            self.storage_access(&storage.ty, mode);
        }
        self.access_choice(&choices, handle, index, element, &emit)
    }

    fn access_choice(
        &mut self,
        choices: &[(i32, Typed)],
        handle: Typed,
        index: Typed,
        element: &Type,
        emit: &impl Fn(&mut Self, PlaceId, Type, bool) -> Result<Typed, OptimizeError>,
    ) -> Result<Typed, OptimizeError> {
        let Some(((tag, storage), rest)) = choices.split_first() else {
            return Err(error("empty memory choice"));
        };
        let tag = self.literal(*tag as i64, &handle.ty)?;
        let matches = self.binary(BinaryOperator::Equal, handle.clone(), tag)?;
        if rest.is_empty() {
            self.require_condition(matches)?;
            let private = self.local_arrays.contains_key(&storage.value);
            let (place, ty) = self.memory_place(storage.clone(), index, element)?;
            return emit(self, place, ty, private);
        }
        self.branch(
            matches,
            |body| {
                let private = body.local_arrays.contains_key(&storage.value);
                let (place, ty) = body.memory_place(storage.clone(), index.clone(), element)?;
                emit(body, place, ty, private)
            },
            |body| body.access_choice(rest, handle, index.clone(), element, emit),
        )
    }

    fn memory_place(
        &mut self,
        storage: Typed,
        index: Typed,
        element: &Type,
    ) -> Result<(PlaceId, Type), OptimizeError> {
        let Some(physical) = memory_element(&storage.ty) else {
            return Err(error("memory block has no element type"));
        };
        if !compatible_element(physical, &storage_value_type(element)) {
            return Err(error("memory block element type disagrees with its view"));
        }
        let Some(mut ty) = storage.ty.elem_type().cloned() else {
            return Err(error("memory storage element missing"));
        };
        let mut dimensions = Vec::new();
        while ty.is_array() {
            let Some(Type::Constructed(TypeName::Size(size), _)) = ty.array_size() else {
                return Err(error("nested physical array must have fixed dimensions"));
            };
            dimensions
                .push(u32::try_from(*size).map_err(|_| error("physical array dimension exceeds u32"))?);
            let Some(element) = ty.elem_type() else {
                return Err(error("physical array element missing"));
            };
            ty = element.clone();
        }
        let mut stride = dimensions
            .iter()
            .try_fold(1u32, |stride, &size| stride.checked_mul(size))
            .ok_or_else(|| error("physical array size exceeds u32"))?;
        let outer = if dimensions.is_empty() {
            index.clone()
        } else {
            self.binary(BinaryOperator::Divide, index.clone(), Self::number(stride.max(1)))?
        };
        let (mut place, mut ty) = self.index_place(storage, outer)?;
        for size in dimensions {
            stride /= size.max(1);
            let offset = self.binary(BinaryOperator::Divide, index.clone(), Self::number(stride.max(1)))?;
            let offset = self.binary(BinaryOperator::Remainder, offset, Self::number(size.max(1)))?;
            let Some(element) = ty.elem_type() else {
                return Err(error("physical array projection has no element"));
            };
            let element = element.clone();
            let next = self.builder.new_place(element.clone());
            self.builder
                .push_void_inst(InstKind::PlaceIndex {
                    place,
                    index: offset.value,
                    result: next,
                })
                .map_err(builder_error)?;
            place = next;
            ty = element;
        }
        Ok((place, ty))
    }

    pub fn read_array(&mut self, name: Value, indices: &[Typed]) -> Result<Typed, OptimizeError> {
        let (handle, offset, ty) = self.array_address(name, indices)?;
        self.access(
            handle,
            offset,
            &ty,
            ResourceAccess::Read,
            |body, place, physical, _| {
                let value = body
                    .builder
                    .push_inst(InstKind::Load { place }, physical.clone())
                    .map_err(builder_error)?;
                body.cast(
                    Typed {
                        value: value.into(),
                        ty: physical,
                    },
                    &ty,
                )
            },
        )
    }

    pub fn write_array(
        &mut self,
        name: Value,
        indices: &[Typed],
        value: Typed,
    ) -> Result<(), OptimizeError> {
        let (handle, offset, element) = self.array_address(name, indices)?;
        self.access(
            handle,
            offset,
            &element,
            ResourceAccess::Write,
            |body, place, physical, _| {
                let value = body.cast(value.clone(), &physical)?;
                body.builder
                    .push_void_inst(InstKind::Store {
                        place,
                        value: value.value,
                    })
                    .map_err(builder_error)?;
                body.op(OpTag::Unit, vec![], types::unit())
            },
        )?;
        Ok(())
    }

    pub fn shape_count(&mut self, shape: &[Typed]) -> Result<Typed, OptimizeError> {
        let mut count = Self::number(1);
        for length in shape {
            count = self.binary(BinaryOperator::Multiply, count, length.clone())?;
        }
        Ok(count)
    }

    pub fn coordinates(
        &mut self,
        mut ordinal: Typed,
        shape: &[Typed],
    ) -> Result<Vec<Typed>, OptimizeError> {
        let mut coordinates = Vec::new();
        for length in shape.iter().rev() {
            let positive = self.binary(BinaryOperator::Greater, length.clone(), Self::number(0))?;
            let one = self.literal(1, &length.ty)?;
            let divisor = self.select(positive, length.clone(), one)?;
            coordinates.push(self.binary(BinaryOperator::Remainder, ordinal.clone(), divisor.clone())?);
            ordinal = self.binary(BinaryOperator::Divide, ordinal, divisor)?;
        }
        coordinates.reverse();
        Ok(coordinates)
    }

    pub(in crate::egglog::to_ssa) fn block_element(&self, block: Value) -> Result<Type, OptimizeError> {
        let mut element = None;
        self.query().for_each("ContainsValueInfo", |row| {
            let Some(root) = self.query().constructor("FunctionRoot", (self.function,))? else {
                return Err(error("function index missing"));
            };
            if row[0] == root {
                if let Some(info) = self.query().enode("ArrayInfo", row[1])? {
                    let storage = self.query().fields("ArrayIn", info[1])?;
                    if storage[0] == block {
                        let ty = element_type(self.query(), info[0])?;
                        if element.as_ref().is_some_and(|previous| *previous != ty) {
                            return Err(error("allocation has incompatible typed views"));
                        }
                        element = Some(ty);
                    }
                }
            }
            Ok(())
        })?;
        let Some(element) = element else {
            return Err(error("memory block has no typed use"));
        };
        Ok(element)
    }

    pub(in crate::egglog::to_ssa) fn transfer_block(
        &mut self,
        handle: Typed,
        bytes: Typed,
        element: &Type,
        destination: Typed,
    ) -> Result<(), OptimizeError> {
        let Some(stride) = storage_elem_stride(&storage_value_type(element)) else {
            return Err(error("block element stride missing"));
        };
        let count = self.binary(BinaryOperator::Divide, bytes, Self::number(stride))?;
        self.counted(
            Self::number(0),
            count,
            Self::number(1),
            vec![],
            |body, index, _| {
                let value = body.access(
                    handle,
                    index.clone(),
                    element,
                    ResourceAccess::Read,
                    |body, place, ty, _| {
                        let value = body
                            .builder
                            .push_inst(InstKind::Load { place }, ty.clone())
                            .map_err(builder_error)?;
                        Ok(Typed {
                            value: value.into(),
                            ty,
                        })
                    },
                )?;
                body.store(destination, index, value)?;
                Ok(vec![])
            },
        )?;
        Ok(())
    }

    pub(in crate::egglog::to_ssa) fn write_atom(
        &mut self,
        destination: Value,
        prefix: &[Typed],
        atom: Value,
    ) -> Result<(), OptimizeError> {
        if let Some(var) = self.query().enode("Var", atom)? {
            let info = self.query().required("BindingInfo", (self.function, var[0]))?;
            if self.query().enode("ArrayInfo", info)?.is_some() {
                return self.copy_array(destination, prefix, var[0]);
            }
        }
        let value = self.atom(atom)?;
        self.write_array(destination, prefix, value)
    }

    fn copy_array(
        &mut self,
        destination: Value,
        prefix: &[Typed],
        source: Value,
    ) -> Result<(), OptimizeError> {
        let source_shape = self.array_shape(source)?;
        let destination_shape = self.array_shape(destination)?;
        let Some(row_shape) = destination_shape.get(prefix.len()..) else {
            return Err(error("array transfer prefix exceeds destination rank"));
        };
        let count = self.shape_count(row_shape)?;
        self.counted(
            Self::number(0),
            count,
            Self::number(1),
            vec![],
            |body, ordinal, _| {
                let from = body.coordinates(ordinal.clone(), &source_shape)?;
                let mut to = prefix.to_vec();
                to.extend(body.coordinates(ordinal, row_shape)?);
                let value = body.read_array(source, &from)?;
                body.write_array(destination, &to, value)?;
                Ok(vec![])
            },
        )?;
        Ok(())
    }

    fn memory_statement(
        &mut self,
        kind: &str,
        args: &[Value],
        pattern: &[Value],
    ) -> Result<Vec<Typed>, OptimizeError> {
        let destinations = pattern
            .iter()
            .filter_map(|&binding| match self.query().enode("BindValue", binding) {
                Ok(Some(fields)) => match self.query().enode("ArrayInfo", fields[1]) {
                    Ok(Some(_)) => Some(Ok(fields[0])),
                    Ok(None) => None,
                    Err(error) => Some(Err(error)),
                },
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        match kind {
            "Alloc" => {
                let [binding] = pattern else {
                    return Err(error("allocation binds exactly one block"));
                };
                let block = self.query().fields("BindBlock", *binding)?[0];
                let element = self.block_element(block)?;
                if self.query().enode("DeviceSpace", args[0])?.is_some() {
                    let capacity =
                        host::guarded_index(self.query(), &self.host_values, args[1], self.guard.as_ref())
                            .map_err(|failure| error(format!("allocation {block:?}: {failure}")))?;
                    let bytes = self.index_expression(args[1])?;
                    let Some(stride) = storage_elem_stride(&storage_value_type(&element)) else {
                        return Err(error("device element stride missing"));
                    };
                    let count = self.binary(BinaryOperator::Divide, bytes, Self::number(stride))?;
                    let (binding, storage) = self.device_buffer(&element, capacity, count)?;
                    let Some(generated) = self.generated.as_mut() else {
                        return Err(error("device allocation buffers missing"));
                    };
                    generated.memories.insert(block, (binding, storage_value_type(&element)));
                    return Ok(vec![self.register_memory(block, storage)?]);
                }
                let bytes = self.index_expression(args[1])?;
                let bound = self
                    .query()
                    .lookup("StaticIndex", (self.function, args[1]))?
                    .map(|value| self.query().0.value_to_base::<i64>(value));
                let bytes = self.integer_literal(&bytes).or(bound);
                let Some(bytes) = bytes else {
                    return Err(error("local allocation has no static capacity bound"));
                };
                let Some(stride) = storage_elem_stride(&storage_value_type(&element)) else {
                    return Err(error("allocation element has no storage size"));
                };
                let count = u32::try_from(bytes)
                    .map_err(|_| error("allocation size exceeds u32"))?
                    .div_ceil(stride.max(1));
                let storage = if self.query().enode("WorkgroupSpace", args[0])?.is_some() {
                    self.workgroup_array(count, &element)?
                } else if self.query().enode("PrivateSpace", args[0])?.is_some() {
                    self.local_array(types::sized_array(count.max(1) as usize, element))?
                } else {
                    return Err(error("device allocation has not been assigned its public buffer"));
                };
                Ok(vec![self.register_memory(block, storage)?])
            }
            "Load" => {
                let indices = self
                    .query()
                    .vector(args[1])?
                    .into_iter()
                    .map(|index| self.index_expression(index))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(vec![self.read_array(args[0], &indices)?])
            }
            "Copy" => {
                let [destination] = destinations.as_slice() else {
                    return Err(error("copy must have one array destination"));
                };
                self.copy_array(*destination, &[], args[0])?;
                Ok(Vec::new())
            }
            "GenerateArray" => {
                let [destination] = destinations.as_slice() else {
                    return Err(error("generator must have one array destination"));
                };
                let (generator, fields) =
                    self.query().application(args[0], &["ArrayLiteral", "Uninitialized"])?;
                if generator == "Uninitialized" {
                    return Ok(Vec::new());
                }
                for (i, atom) in self.query().vector(fields[0])?.into_iter().enumerate() {
                    self.write_atom(*destination, &[Self::number(i as u32)], atom)?;
                }
                Ok(Vec::new())
            }
            "Update" | "Store" => {
                let slices = self.query().vector(args[2])?;
                let shape = self.array_shape(args[1])?;
                let source = if let Some(var) = self.query().enode("Var", args[3])? {
                    let info = self.query().required("BindingInfo", (self.function, var[0]))?;
                    self.query().enode("ArrayInfo", info)?.map(|_| var[0])
                } else {
                    None
                };
                let source_shape =
                    if let Some(source) = source { self.array_shape(source)? } else { vec![] };
                let count = self.shape_count(&source_shape)?;
                self.counted(
                    Self::number(0),
                    count,
                    Self::number(1),
                    vec![],
                    |body, ordinal, _| {
                        let coordinates = body.coordinates(ordinal, &source_shape)?;
                        let mut remaining = coordinates.iter();
                        let mut target = Vec::new();
                        let mut valid = body.op(OpTag::Bool(true), vec![], types::bool_type())?;
                        for (axis, length) in shape.iter().enumerate() {
                            let index = if let Some(&slice) = slices.get(axis) {
                                if let Some(at) = body.query().enode("At", slice)? {
                                    body.index_expression(at[0])?
                                } else {
                                    let slice = body.query().fields("Slice", slice)?;
                                    let Some(coordinate) = remaining.next() else {
                                        return Err(error("array update rank mismatch"));
                                    };
                                    let extent = body.index_expression(slice[1])?;
                                    let inside =
                                        body.binary(BinaryOperator::Less, coordinate.clone(), extent)?;
                                    valid = body.binary(BinaryOperator::LogicalAnd, valid, inside)?;
                                    let start = body.index_expression(slice[0])?;
                                    let stride = body.index_expression(slice[2])?;
                                    let delta =
                                        body.binary(BinaryOperator::Multiply, coordinate.clone(), stride)?;
                                    body.binary(BinaryOperator::Add, start, delta)?
                                }
                            } else {
                                let Some(coordinate) = remaining.next() else {
                                    return Err(error("array update rank mismatch"));
                                };
                                coordinate.clone()
                            };
                            let zero = body.literal(0, &index.ty)?;
                            let nonnegative =
                                body.binary(BinaryOperator::GreaterEqual, index.clone(), zero)?;
                            let below = body.binary(BinaryOperator::Less, index.clone(), length.clone())?;
                            let inbounds = body.binary(BinaryOperator::LogicalAnd, nonnegative, below)?;
                            valid = body.binary(BinaryOperator::LogicalAnd, valid, inbounds)?;
                            target.push(index);
                        }
                        if remaining.next().is_some() || slices.len() > shape.len() {
                            return Err(error("array update rank mismatch"));
                        }
                        body.when(valid, |body| {
                            let value = if let Some(source) = source {
                                body.read_array(source, &coordinates)?
                            } else {
                                body.atom(args[3])?
                            };
                            body.write_array(args[1], &target, value)
                        })?;
                        Ok(vec![])
                    },
                )?;
                Ok(Vec::new())
            }
            "WithAcc" => {
                let lambda = self.query().fields("FunctionBody", args[1])?;
                self.block(lambda[2])
            }
            "Parallel" => self.parallel(args, &destinations, pattern),
            "Accumulate" | "Reserve" => self.accumulate(kind, args),
            _ => Err(error(format!("unknown memory instruction {kind}"))),
        }
    }
}

impl Body<'_, '_, '_> {
    fn accumulate(&mut self, kind: &str, arguments: &[Value]) -> Result<Vec<Typed>, OptimizeError> {
        let capability = self.query().required(
            "TokenCapability",
            (self.function, arguments[usize::from(kind != "Reserve")]),
        )?;
        let capability = self.query().fields("AccumulatorInput", capability)?;
        let arrays = self.query().vector(capability[0])?;
        let policy = capability[1];
        let token = usize::from(kind != "Reserve");
        let indices = self
            .query()
            .vector(arguments[token + 1])?
            .into_iter()
            .map(|index| self.index_expression(index))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(&first) = arrays.first() else {
            return Err(error("empty accumulator destination"));
        };
        let (_, block, _) = self.array_descriptor(first)?;
        let mut known = false;
        let mut private = true;
        self.query().for_each("BlockSpace", |row| {
            if row[0] == self.function && row[1] == block {
                known = true;
                private &= self.query().enode("PrivateSpace", row[2])?.is_some();
            }
            Ok(())
        })?;
        let private = known && private;
        let concurrent = if let Some(statement) = self.active_dispatch {
            let dispatch = self.compiler.execution.dispatch(statement)?;
            !private && (dispatch.width.get() > 1 || dispatch.grid != Some((1, 1, 1)))
        } else {
            false
        };
        let shape = self.array_shape(first)?;
        if shape.len() != indices.len() {
            return Err(error("accumulator index rank mismatch"));
        }
        let mut valid = self.op(OpTag::Bool(true), vec![], types::bool_type())?;
        for (index, length) in indices.iter().zip(shape) {
            let zero = self.literal(0, &index.ty)?;
            let nonnegative = self.binary(BinaryOperator::GreaterEqual, index.clone(), zero)?;
            let below = self.binary(BinaryOperator::Less, index.clone(), length)?;
            let inside = self.binary(BinaryOperator::LogicalAnd, nonnegative, below)?;
            valid = self.binary(BinaryOperator::LogicalAnd, valid, inside)?;
        }
        if kind == "Reserve" {
            if self.query().enode("FetchAddWrites", policy)?.is_none() || arrays.len() != 1 {
                return Err(error("reserve requires one fetch-add accumulator"));
            }
            let delta = self.atom(arguments[token + 2])?;
            let ty = delta.ty.clone();
            let value = self.branch(
                valid,
                |body| {
                    if !concurrent {
                        let previous = body.read_array(first, &indices)?;
                        let next = body.binary(BinaryOperator::Add, previous.clone(), delta.clone())?;
                        body.write_array(first, &indices, next)?;
                        return Ok(previous);
                    }
                    let (handle, offset, _) = body.array_address(first, &indices)?;
                    body.access(
                        handle,
                        offset,
                        &ty,
                        ResourceAccess::ReadWrite,
                        |body, place, element, private| {
                            let value = body.cast(delta.clone(), &element)?;
                            if private {
                                let previous = body
                                    .builder
                                    .push_inst(InstKind::Load { place }, element.clone())
                                    .map_err(builder_error)?;
                                let previous = Typed {
                                    value: previous.into(),
                                    ty: element.clone(),
                                };
                                let next = body.binary(BinaryOperator::Add, previous.clone(), value)?;
                                body.builder
                                    .push_void_inst(InstKind::Store {
                                        place,
                                        value: next.value,
                                    })
                                    .map_err(builder_error)?;
                                return body.cast(previous, &ty);
                            }
                            let result = body
                                .builder
                                .push_inst(
                                    InstKind::Atomic {
                                        place,
                                        op: AtomicOp::Add,
                                        values: vec![value.value],
                                    },
                                    element.clone(),
                                )
                                .map_err(builder_error)?;
                            body.cast(
                                Typed {
                                    value: result.into(),
                                    ty: element,
                                },
                                &ty,
                            )
                        },
                    )
                },
                |body| body.literal(0, &ty),
            )?;
            return Ok(vec![value]);
        }
        let values = self
            .query()
            .vector(arguments[3])?
            .into_iter()
            .map(|atom| self.atom(atom))
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != arrays.len() {
            return Err(error("accumulator value arity mismatch"));
        }
        self.when(valid, |body| {
            if let Some(combine) = body.query().enode("CombineWrites", policy)? {
                if concurrent {
                    let [value] = values.as_slice() else {
                        return Err(error("concurrent reducer must have one atomic result"));
                    };
                    let (handle, offset, ty) = body.array_address(first, &indices)?;
                    if !matches!(ty, Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _)) {
                        return Err(error("concurrent reducer needs a 32-bit integer"));
                    }
                    body.access(
                        handle,
                        offset,
                        &ty,
                        ResourceAccess::ReadWrite,
                        |body, place, ty, private| {
                            if private {
                                let old = body
                                    .builder
                                    .push_inst(InstKind::Load { place }, ty.clone())
                                    .map_err(builder_error)?;
                                let old = Typed {
                                    value: old.into(),
                                    ty: ty.clone(),
                                };
                                let mut next = body.lambda(combine[0], vec![old, value.clone()])?;
                                let Some(next) = next.pop() else {
                                    return Err(error("combiner has no result"));
                                };
                                let next = body.cast(next, &ty)?;
                                body.builder
                                    .push_void_inst(InstKind::Store {
                                        place,
                                        value: next.value,
                                    })
                                    .map_err(builder_error)?;
                                return body.op(OpTag::Unit, vec![], types::unit());
                            }
                            let old = body
                                .builder
                                .push_inst(
                                    InstKind::Atomic {
                                        place,
                                        op: AtomicOp::Load,
                                        values: vec![],
                                    },
                                    ty.clone(),
                                )
                                .map_err(builder_error)?;
                            let old = Typed {
                                value: old.into(),
                                ty: ty.clone(),
                            };
                            let done = body.op(OpTag::Bool(false), vec![], types::bool_type())?;
                            let initial = body.tuple(vec![old, done])?;
                            body.retry(initial, |body, state| {
                                let old = body.field(state.clone(), 0)?;
                                let mut next = body.lambda(combine[0], vec![old.clone(), value.clone()])?;
                                if next.len() != 1 {
                                    return Err(error("atomic combiner changes arity"));
                                }
                                let Some(next) = next.pop() else {
                                    return Err(error("atomic combiner has no result"));
                                };
                                let next = body.cast(next, &ty)?;
                                let result = body
                                    .builder
                                    .push_inst(
                                        InstKind::Atomic {
                                            place,
                                            op: AtomicOp::CompareExchange,
                                            values: vec![old.value, next.value],
                                        },
                                        state.ty.clone(),
                                    )
                                    .map_err(builder_error)?;
                                Ok(Typed {
                                    value: result.into(),
                                    ty: state.ty,
                                })
                            })?;
                            body.op(OpTag::Unit, vec![], types::unit())
                        },
                    )?;
                    return Ok(());
                }
                let mut previous = arrays
                    .iter()
                    .map(|&array| body.read_array(array, &indices))
                    .collect::<Result<Vec<_>, _>>()?;
                previous.extend(values);
                let next = body.lambda(combine[0], previous)?;
                for (&array, value) in arrays.iter().zip(next) {
                    body.write_array(array, &indices, value)?;
                }
            } else {
                for (&array, value) in arrays.iter().zip(values) {
                    body.write_array(array, &indices, value)?;
                }
            }
            Ok(())
        })?;
        Ok(Vec::new())
    }
}

fn memory_element(mut ty: &Type) -> Option<&Type> {
    if !ty.is_array() {
        return None;
    }
    while ty.is_array() {
        ty = ty.elem_type()?;
    }
    Some(ty)
}

fn compatible_element(left: &Type, right: &Type) -> bool {
    match (left, right) {
        (
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), left),
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), right),
        ) => left.len() == right.len() && left.iter().zip(right).all(|(a, b)| compatible_element(a, b)),
        _ => left == right,
    }
}
