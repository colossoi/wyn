use super::{
    builder_error, concrete, error, u32_type, uint, Body, InstKind, OpTag, OptimizeError, ParameterId,
    PureViewSource, Storage, Type, TypeExt, TypeName, Typed, Value,
};
use crate::ast::Span;
use crate::builtins::{catalog, select};
use crate::egglog::abi::storage_type;
use crate::egglog::data::is_slice;
use crate::egglog::{Array, ExprId, ExprKind, PlacementSite};
use crate::flow::{BlockId, ControlHeader};
use crate::op::{BinaryOperator, UnaryOperator};
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{ConstantValue, Function, PlaceId, Terminator, ValueRef};
use crate::types::{
    array_variant_bounded, bool_type, buffer_tag, extract_function_signature, i32, is_array_variant_view,
    is_array_variant_virtual, make_array1, no_buffer, sized_array, strip_existentials, view_array_of,
    view_array_with_size,
};
use crate::{BindingRef, FunctionId};
use std::collections::HashMap;

// One evaluation of an expression or argument list. Block keys prevent a value
// emitted in one conditional arm from escaping into its sibling or the join.
// The cache is dropped before subsequent bindings or memory effects.
type ExpressionCache = HashMap<(Option<BlockId>, ExprId), Typed>;

fn static_array_length(ty: &Type) -> Option<usize> {
    let ty = strip_existentials(ty);
    if let Type::Constructed(TypeName::Tuple(_), fields) = ty {
        return static_array_length(fields.first()?);
    }
    // Bounded arrays encode capacity here. An abstract representation may
    // specialize to bounded, so neither variant proves an exact live length.
    if !matches!(
        ty.array_variant(),
        Some(Type::Constructed(
            TypeName::ArrayVariantComposite | TypeName::ArrayVariantView | TypeName::ArrayVariantVirtual,
            _
        ))
    ) {
        return None;
    }
    ty.as_tensor()?.concrete_dim(0)
}

impl Body<'_, '_> {
    /// Preserve the physical interface without emitting runtime setup.
    pub(super) fn declare_input(&mut self, id: ParameterId) -> Result<(), OptimizeError> {
        if self.declared_inputs.contains_key(&id) {
            return Ok(());
        }
        if self.entry.is_none() {
            return Err(error(format!("unbound helper parameter {id:?}")));
        }
        let Some(inputs) = self.compiler.data.state.abi.inputs.get(&id).cloned() else {
            return Err(error(format!("no source ABI for parameter {id:?}")));
        };
        let mut declared = vec![];
        for input in inputs {
            let ty = if let Some(binding) = input.storage_binding() {
                let Some(element) = input.ty.elem_type() else {
                    return Err(error("storage input element"));
                };
                view_array_of(&storage_type(element)?, buffer_tag(binding))
            } else {
                concrete(&input.ty)?
            };
            let value = self.builder.func_mut().add_function_param(ty.clone(), input.name.clone()).into();
            self.inputs.push(input.clone());
            declared.push((input, Typed { value, ty }));
        }
        self.declared_inputs.insert(id, declared);
        Ok(())
    }

    pub(super) fn input(&mut self, id: ParameterId) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.environment.parameters.get(&id) {
            return Ok(value.clone());
        }
        self.declare_input(id)?;
        let mut values = vec![];
        for (input, parameter) in self.declared_inputs[&id].clone() {
            let value = if let Some(binding) = input.storage_binding() {
                let length = if let Some(Type::Constructed(TypeName::Size(n), _)) = input.ty.array_size() {
                    Self::number(u32::try_from(*n).map_err(|_| error("array size exceeds u32"))?)
                } else {
                    self.op(
                        OpTag::Intrinsic {
                            id: catalog().known().storage_len,
                            overload_idx: 0,
                        },
                        vec![Self::number(binding.set), Self::number(binding.binding)],
                        u32_type(),
                    )?
                };
                let Some(element) = input.ty.elem_type() else {
                    return Err(error("storage input element"));
                };
                self.view(binding, storage_type(element)?, length)?
            } else {
                parameter
            };
            values.push(value);
        }
        let value = if values.len() == 1 { values.remove(0) } else { self.tuple(values)? };
        let value = self.cast(
            value,
            &self.compiler.data.types[self.compiler.data.parameters[id].ty].ty,
        )?;
        self.environment.parameters.insert(id, value.clone());
        Ok(value)
    }
    fn view(&mut self, binding: BindingRef, element: Type, len: Typed) -> Result<Typed, OptimizeError> {
        let ty = view_array_with_size(
            &element,
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            buffer_tag(binding),
        );
        self.op(
            OpTag::StorageView(PureViewSource::Storage(binding)),
            vec![Self::number(0), len],
            ty,
        )
    }
    fn number(n: u32) -> Typed {
        Typed {
            value: uint(n),
            ty: u32_type(),
        }
    }
    pub(super) fn value(&mut self, value: &Value) -> Result<Typed, OptimizeError> {
        self.value_cached(value, &mut ExpressionCache::new())
    }
    pub(super) fn values(&mut self, values: &[Value]) -> Result<Vec<Typed>, OptimizeError> {
        self.values_cached(values, &mut ExpressionCache::new())
    }
    fn values_cached(
        &mut self,
        values: &[Value],
        cache: &mut ExpressionCache,
    ) -> Result<Vec<Typed>, OptimizeError> {
        values.iter().map(|v| self.value_cached(v, cache)).collect()
    }
    fn value_cached(&mut self, value: &Value, cache: &mut ExpressionCache) -> Result<Typed, OptimizeError> {
        match value {
            Value::Discarded => Err(error("unused result has no value")),
            Value::Int(n) => Ok(Self::number(*n)),
            Value::Local(name) => {
                if !self.environment.locals.contains_key(name) {
                    if let Some(&parameter) = self.input_locals.get(name) {
                        return self.input(parameter);
                    }
                }
                let Some(value) = self.environment.locals.get(name).cloned() else {
                    return Err(error(format!("unbound scaffold local {name}")));
                };
                Ok(value)
            }
            Value::Source(id) => self.expression_cached(*id, cache),
            Value::Tuple(values) => {
                let values = self.values_cached(values, cache)?;
                self.tuple(values)
            }
            Value::Field(value, index) => {
                let value = self.value_cached(value, cache)?;
                self.field(value, *index)
            }
            Value::Array(array) => self.array_cached(array, cache),
            Value::Workgroup { id, count, element } => {
                let ty = view_array_of(&concrete(&self.compiler.data.types[*element].ty)?, no_buffer());
                self.op(
                    OpTag::StorageView(PureViewSource::Workgroup {
                        id: *id,
                        count: *count,
                    }),
                    vec![Self::number(0), Self::number(*count)],
                    ty,
                )
            }
            Value::Buffer(id) => {
                if let Some((place, ty)) = self.environment.buffers.get(id).cloned() {
                    let value = self
                        .builder
                        .push_inst(InstKind::Load { place }, ty.clone())
                        .map_err(builder_error)?;
                    return Ok(Typed {
                        value: value.into(),
                        ty,
                    });
                }
                let buffer = &self.compiler.data.state.buffers[*id];
                if let Storage::View(expr) = buffer.storage {
                    return self.expression_cached(expr, cache);
                }
                if buffer.storage == Storage::Function {
                    return Err(error("local buffer used before allocation"));
                }
                let Some(declaration) = self.compiler.data.state.abi.bindings.get(id).cloned() else {
                    return Err(error(format!("buffer {id:?} has no Allocation fact")));
                };
                self.compiler.used.insert(*id);
                let len = self.value_cached(&buffer.length, cache)?;
                let len = self.cast(len, &u32_type())?;
                self.view(declaration.binding, declaration.elem_ty, len)
            }
            Value::Primitive(name, args) => {
                if let ("length", [array]) = (*name, args.as_slice()) {
                    return self.length_cached(array, cache);
                }
                let values = self.values_cached(args, cache)?;
                self.primitive(name, values)
            }
        }
    }

    // Generated shape queries need only the array's extent. Source operations
    // still execute in their scheduled instructions; reading an extent must not
    // reload their captured results just to assemble the array's elements.
    fn length_cached(
        &mut self,
        array: &Value,
        cache: &mut ExpressionCache,
    ) -> Result<Typed, OptimizeError> {
        if let Some(length) = self.length_type(array, cache).and_then(static_array_length) {
            return Ok(Self::number(
                u32::try_from(length).map_err(|_| error("array size exceeds u32"))?,
            ));
        }
        match array {
            Value::Primitive("slice", args) if args.len() == 2 => {
                return self.value_cached(&args[1], cache);
            }
            Value::Array(array) => return self.array_length_cached(array, cache),
            Value::Tuple(fields) => {
                let Some(first) = fields.first() else {
                    return Err(error("empty logical array"));
                };
                return self.length_cached(first, cache);
            }
            Value::Source(id)
                if !self.environment.expressions.contains_key(id)
                    && !cache.contains_key(&(self.builder.current_block(), *id)) =>
            {
                if let Some(length) = self.compiler.data.state.execution.view_lengths.get(id) {
                    return self.value_cached(length, cache);
                }
                match &self.compiler.data.expressions[*id].kind {
                    ExprKind::OperationResult(op) if !self.environment.operations.contains_key(op) => {
                        if let Some(value) = self.compiler.data.state.materialized.get(op) {
                            return self.length_cached(value, cache);
                        }
                    }
                    ExprKind::Array(array) => return self.array_length_cached(array, cache),
                    ExprKind::Coerce(inner) => return self.length_cached(&Value::Source(*inner), cache),
                    ExprKind::PureApp { function, args }
                        if is_slice(&self.compiler.data.ir, *function) && args.len() == 3 =>
                    {
                        let start = self.expression_cached(args[1], cache)?;
                        let end = self.expression_cached(args[2], cache)?;
                        let start = self.cast(start, &u32_type())?;
                        let end = self.cast(end, &u32_type())?;
                        return self.primitive("sub", vec![end, start]);
                    }
                    ExprKind::Tuple(fields) => {
                        let Some(first) = fields.first() else {
                            return Err(error("empty logical array"));
                        };
                        return self.length_cached(&Value::Source(*first), cache);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        let array = self.value_cached(array, cache)?;
        self.primitive("length", vec![array])
    }

    // Consult the current binding before source metadata: a scheduled value can
    // have a different array representation (and length semantics) after rebinding.
    fn length_type<'c>(&'c self, array: &Value, cache: &'c ExpressionCache) -> Option<&'c Type> {
        match array {
            Value::Source(id) => {
                let expression = &self.compiler.data.expressions[*id];
                let bound = self
                    .environment
                    .expressions
                    .get(id)
                    .or_else(|| cache.get(&(self.builder.current_block(), *id)))
                    .or_else(|| match expression.kind {
                        ExprKind::Parameter(id) => self.environment.parameters.get(&id),
                        ExprKind::OperationResult(id) => self.environment.operations.get(&id),
                        _ => None,
                    });
                Some(bound.map_or(&self.compiler.data.types[expression.ty].ty, |value| &value.ty))
            }
            Value::Local(name) => self.environment.locals.get(name).map(|value| &value.ty),
            Value::Field(value, index) => match strip_existentials(self.length_type(value, cache)?) {
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.get(*index),
                _ => None,
            },
            _ => None,
        }
    }

    fn array_length_cached(
        &mut self,
        array: &Array,
        cache: &mut ExpressionCache,
    ) -> Result<Typed, OptimizeError> {
        match array {
            Array::Value(id) => self.length_cached(&Value::Source(*id), cache),
            Array::Zip(arrays) => {
                let Some(first) = arrays.first() else {
                    return Err(error("empty logical array"));
                };
                self.array_length_cached(first, cache)
            }
            Array::Literal(items) => Ok(Self::number(
                u32::try_from(items.len()).map_err(|_| error("array size exceeds u32"))?,
            )),
            Array::Range { len, .. } => {
                let length = self.expression_cached(*len, cache)?;
                self.cast(length, &u32_type())
            }
        }
    }

    pub(super) fn expression(&mut self, id: ExprId) -> Result<Typed, OptimizeError> {
        self.expression_cached(id, &mut ExpressionCache::new())
    }
    pub(super) fn expressions(&mut self, ids: &[ExprId]) -> Result<Vec<Typed>, OptimizeError> {
        let mut cache = ExpressionCache::new();
        ids.iter().map(|&id| self.expression_cached(id, &mut cache)).collect()
    }
    fn expression_cached(
        &mut self,
        id: ExprId,
        cache: &mut ExpressionCache,
    ) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.environment.expressions.get(&id) {
            return Ok(value.clone());
        }
        if let Some(value) = cache.get(&(self.builder.current_block(), id)) {
            return Ok(value.clone());
        }
        if let Some(binding) = self.compiler.host.captures.get(&(self.root, id)).copied() {
            let ty = self.compiler.data.types[self.compiler.data.expressions[id].ty].ty.clone();
            let element = storage_type(&ty)?;
            let view = self.view(binding, element, Self::number(1))?;
            let value = self.index(view, Self::number(0))?;
            let value = self.cast(value, &ty)?;
            self.environment.expressions.insert(id, value.clone());
            return Ok(value);
        }
        let data = self.compiler.data;
        let record = &data.expressions[id];
        let ty = data.types[record.ty].ty.clone();
        let result = match &record.kind {
            ExprKind::Parameter(p) => self.input(*p),
            ExprKind::OperationResult(op) => {
                if let Some(v) = self.environment.operations.get(op) {
                    Ok(v.clone())
                } else if let Some((function, captures)) = data.state.execution.expansions.get(op) {
                    let args = captures
                        .iter()
                        .map(|&e| self.expression_cached(e, cache))
                        .collect::<Result<Vec<_>, _>>()?;
                    let id =
                        self.compiler.function(*function, args.iter().map(|v| v.ty.clone()).collect())?;
                    self.op(OpTag::Call(id), args, ty)
                } else {
                    let Some(value) = data.state.materialized.get(op) else {
                        return Err(error(format!("unmaterialized capture {op:?}")));
                    };
                    let value = self.value_cached(value, cache)?;
                    self.cast(value, &ty)
                }
            }
            ExprKind::Int(s) => self.op(
                if matches!(ty, Type::Constructed(TypeName::UInt(_), _)) {
                    OpTag::Uint(s.clone())
                } else {
                    OpTag::Int(s.clone())
                },
                vec![],
                ty,
            ),
            ExprKind::FloatBits(bits) if ty == Type::Constructed(TypeName::Float(32), vec![]) => {
                Ok(Typed {
                    value: ValueRef::Const(ConstantValue::F32(*bits)),
                    ty,
                })
            }
            ExprKind::FloatBits(bits) => {
                self.op(OpTag::Float(f32::from_bits(*bits).to_string()), vec![], ty)
            }
            ExprKind::Bool(b) => Ok(Typed {
                value: ValueRef::Const(ConstantValue::Bool(*b)),
                ty,
            }),
            ExprKind::Unit => self.op(OpTag::Unit, vec![], ty),
            ExprKind::Tuple(items) => {
                let values = items
                    .iter()
                    .map(|&id| self.expression_cached(id, cache))
                    .collect::<Result<Vec<_>, _>>()?;
                self.tuple(values)
            }
            ExprKind::Vector(items) => {
                let values = items
                    .iter()
                    .map(|&id| self.expression_cached(id, cache))
                    .collect::<Result<Vec<_>, _>>()?;
                self.op(OpTag::Vector(items.len()), values, ty)
            }
            ExprKind::Project { tuple, index } => self.projection(*tuple, *index, cache),
            ExprKind::Coerce(inner) => {
                let value = self.expression_cached(*inner, cache)?;
                self.cast(value, &ty)
            }
            ExprKind::Array(array) => self.array_cached(array, cache),
            ExprKind::PureApp { function, args } => {
                let args = args
                    .iter()
                    .map(|&id| self.expression_cached(id, cache))
                    .collect::<Result<Vec<_>, _>>()?;
                self.apply(*function, args, ty)
            }
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => {
                for value in self
                    .compiler
                    .placements
                    .get(&PlacementSite::Expression(id))
                    .cloned()
                    .unwrap_or_default()
                {
                    let computed = self.expression(value)?;
                    self.environment.expressions.insert(value, computed);
                }
                let c = self.expression_cached(*condition, cache)?;
                let saved = self.environment.clone();
                let Some(start) = self.builder.current_block() else {
                    return Err(error("no current block"));
                };
                let yes = self.builder.create_block();
                let no = self.builder.create_block();
                let end = self.builder.create_block();
                self.builder.set_control_header(start, ControlHeader::Selection { merge: end });
                self.builder
                    .terminate(Terminator::CondBranch {
                        cond: c.value,
                        then_target: yes,
                        then_args: vec![],
                        else_target: no,
                        else_args: vec![],
                    })
                    .map_err(builder_error)?;
                self.builder.switch_to_block_unchecked(yes);
                let a = self.expression_cached(*then_value, cache)?;
                let p = self.builder.add_block_param(end, a.ty.clone());
                self.builder
                    .terminate(Terminator::Branch {
                        target: end,
                        args: vec![a.value],
                    })
                    .map_err(builder_error)?;
                self.builder.switch_to_block_unchecked(no);
                self.environment = saved.clone();
                let b = self.expression_cached(*else_value, cache)?;
                let b = self.cast(b, &a.ty)?;
                self.builder
                    .terminate(Terminator::Branch {
                        target: end,
                        args: vec![b.value],
                    })
                    .map_err(builder_error)?;
                self.builder.switch_to_block_unchecked(end);
                self.environment = saved;
                Ok(Typed {
                    value: p.into(),
                    ty: a.ty,
                })
            }
            other => Err(error(format!("TODO: first-class value {other:?}"))),
        }?;
        // Interned values can have several source uses. Retain a concrete
        // origin without replacing a more specific child expression's span.
        if let (Some(&span), ValueRef::Ssa(value)) = (self.compiler.origins.get(&id), &result.value) {
            let body = self.builder.func_mut();
            if let Some(inst) = body.inst_of_value(*value) {
                body.insts[inst].span.get_or_insert(span);
            }
        }
        cache.insert((self.builder.current_block(), id), result.clone());
        Ok(result)
    }
    fn projection(
        &mut self,
        tuple: ExprId,
        index: usize,
        cache: &mut ExpressionCache,
    ) -> Result<Typed, OptimizeError> {
        let data = self.compiler.data;
        if !self.environment.expressions.contains_key(&tuple)
            && !cache.contains_key(&(self.builder.current_block(), tuple))
        {
            if let ExprKind::OperationResult(op) = data.expressions[tuple].kind {
                if !self.environment.operations.contains_key(&op) {
                    // Split results may contain discarded slots. Resolve only
                    // the requested slot instead of loading the whole tuple.
                    if let Some(Value::Tuple(fields)) = data.state.materialized.get(&op) {
                        let Some(value) = fields.get(index) else {
                            return Err(error("result slot"));
                        };
                        return self.value_cached(value, cache);
                    }
                }
            }
        }
        // An aggregate stored in one buffer has one SSA identity. All of its
        // projections use the same cached OperationResult within this emission.
        let value = self.expression_cached(tuple, cache)?;
        self.field(value, index)
    }
    pub(super) fn apply(
        &mut self,
        function: ExprId,
        args: Vec<Typed>,
        ty: Type,
    ) -> Result<Typed, OptimizeError> {
        let tag = match &self.compiler.data.expressions[function].kind {
            ExprKind::BinOp(op) => OpTag::BinOp(
                BinaryOperator::try_from(op.as_str()).map_err(|_| error("unknown binary operator"))?,
            ),
            ExprKind::UnOp(op) => OpTag::UnaryOp(
                UnaryOperator::try_from(op.as_str()).map_err(|_| error("unknown unary operator"))?,
            ),
            ExprKind::Builtin(id) => {
                let b = &self.compiler.data.builtins[*id];
                OpTag::Intrinsic {
                    id: b.builtin,
                    overload_idx: b.overload_idx,
                }
            }
            ExprKind::Extern(external) => {
                let id = if let Some(&id) = self.compiler.externs.get(external) {
                    id
                } else {
                    let index = u32::try_from(self.compiler.functions.len())
                        .map_err(|_| error("too many helpers"))?;
                    let id = FunctionId::from(index);
                    let name = &self.compiler.data.externs[*external].linkage_name;
                    let signature =
                        &self.compiler.data.types[self.compiler.data.expressions[function].ty].ty;
                    let (params, result) = extract_function_signature(signature);
                    self.compiler.functions.push(Function {
                        id,
                        name: name.clone(),
                        body: FuncBuilder::new(
                            params.into_iter().enumerate().map(|(i, t)| (t, format!("arg{i}"))).collect(),
                            result,
                        )
                        .finish_unchecked(),
                        span: Span::generated(),
                        linkage_name: Some(name.clone()),
                    });
                    self.compiler.externs.insert(*external, id);
                    id
                };
                OpTag::Call(id)
            }
            other => return Err(error(format!("TODO: unresolved call target {other:?}"))),
        };
        self.op(tag, args, ty)
    }
    fn array_cached(&mut self, array: &Array, cache: &mut ExpressionCache) -> Result<Typed, OptimizeError> {
        match array {
            Array::Value(id) => self.expression_cached(*id, cache),
            Array::Zip(arrays) => {
                let values =
                    arrays.iter().map(|a| self.array_cached(a, cache)).collect::<Result<Vec<_>, _>>()?;
                self.tuple(values)
            }
            Array::Literal(items) => {
                let values = items
                    .iter()
                    .map(|&id| self.expression_cached(id, cache))
                    .collect::<Result<Vec<_>, _>>()?;
                let Some(first) = values.first() else {
                    return Err(error("TODO: element type of empty array operand"));
                };
                let ty = sized_array(values.len(), first.ty.clone());
                self.op(OpTag::ArrayLit(values.len()), values, ty)
            }
            Array::Range { start, len, step } => {
                let a = self.expression_cached(*start, cache)?;
                let n = self.expression_cached(*len, cache)?;
                let n = self.cast(n, &a.ty)?;
                let ty = make_array1(
                    a.ty.clone(),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    no_buffer(),
                );
                let mut args = vec![a, n];
                if let Some(step) = step {
                    args.push(self.expression_cached(*step, cache)?);
                }
                self.op(
                    OpTag::ArrayRange {
                        has_step: step.is_some(),
                    },
                    args,
                    ty,
                )
            }
        }
    }
    pub(super) fn field(&mut self, value: Typed, index: usize) -> Result<Typed, OptimizeError> {
        let Some(ty) = (match &value.ty {
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
                fields.get(index).cloned()
            }
            Type::Constructed(TypeName::Vec, fields) => fields.first().cloned(),
            _ => None,
        }) else {
            return Err(error(format!("invalid projection {index} of {:?}", value.ty)));
        };
        self.op(OpTag::Project { index: index as u32 }, vec![value], ty)
    }
    pub(super) fn indexed_destination(
        &mut self,
        buffer: &Value,
        index: Typed,
    ) -> Result<(PlaceId, Type), OptimizeError> {
        let base = match buffer {
            Value::Primitive("index", args) => {
                let [base, outer] = args.as_slice() else {
                    return Err(error("destination index requires an array and index"));
                };
                let outer = self.value(outer)?;
                Some(self.indexed_destination(base, outer)?)
            }
            Value::Buffer(id) => self.environment.buffers.get(id).cloned(),
            _ => None,
        };
        let Some((place, ty)) = base else {
            let array = self.value(buffer)?;
            return self.index_place(array, index);
        };
        let Some(ty) = ty.elem_type().cloned() else {
            return Err(error("indexed destination has no element type"));
        };
        let result = self.builder.new_place(ty.clone());
        self.builder
            .push_void_inst(InstKind::PlaceIndex {
                place,
                index: index.value,
                result,
            })
            .map_err(builder_error)?;
        Ok((result, ty))
    }
    fn index_place(&mut self, array: Typed, index: Typed) -> Result<(PlaceId, Type), OptimizeError> {
        if !array.ty.array_variant().is_some_and(is_array_variant_view) {
            return Err(error("writable array requires a storage view or local place"));
        }
        let Some(ty) = array.ty.elem_type().cloned() else {
            return Err(error("indexed value has no element type"));
        };
        let place = self.builder.new_place(ty.clone());
        self.builder
            .push_void_inst(InstKind::ViewIndex {
                view: array.value,
                index: index.value,
                result: place,
            })
            .map_err(builder_error)?;
        Ok((place, ty))
    }
    pub(super) fn index(&mut self, array: Typed, index: Typed) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_), fields) = &array.ty {
            let mut values = vec![];
            for i in 0..fields.len() {
                let part = self.field(array.clone(), i)?;
                values.push(self.index(part, index.clone())?);
            }
            return self.tuple(values);
        }
        if array.ty.array_variant().is_some_and(is_array_variant_view) {
            let (place, ty) = self.index_place(array, index)?;
            let value =
                self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
            return Ok(Typed {
                value: value.into(),
                ty,
            });
        }
        let Some(ty) = array.ty.elem_type().cloned() else {
            return Err(error("index of non-array"));
        };
        let index = if array.ty.array_variant().is_some_and(is_array_variant_virtual) {
            self.cast(index, &ty)?
        } else {
            index
        };
        self.op(OpTag::Index, vec![array, index], ty)
    }
    pub(super) fn cast(&mut self, value: Typed, ty: &Type) -> Result<Typed, OptimizeError> {
        let ty = strip_existentials(ty);
        if value.ty == *ty || (value.ty.is_array() && ty.is_array()) {
            return Ok(value);
        }
        if let (
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), a),
            Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), b),
        ) = (&value.ty, ty)
        {
            if a.len() == b.len() {
                let mut fields = vec![];
                for (i, target) in b.iter().enumerate() {
                    let field = self.field(value.clone(), i)?;
                    fields.push(self.cast(field, target)?);
                }
                let ty = Type::Constructed(name.clone(), fields.iter().map(|f| f.ty.clone()).collect());
                return self.op(OpTag::Tuple(fields.len()), fields, ty);
            }
        }
        if *ty == bool_type() {
            return self.binary(BinaryOperator::NotEqual, value, Self::number(0));
        }
        if value.ty == bool_type() {
            let one = self.cast(Self::number(1), ty)?;
            let zero = self.cast(Self::number(0), ty)?;
            return self.select(value, one, zero);
        }
        let (Type::Constructed(target, _), Type::Constructed(source, _)) = (ty, &value.ty) else {
            return Err(error("unsupported conversion"));
        };
        let Some(id) = catalog().conversion(target, source) else {
            return Err(error(format!("TODO: convert {:?} to {ty:?}", value.ty)));
        };
        self.op(OpTag::Intrinsic { id, overload_idx: 0 }, vec![value], ty.clone())
    }
    fn binary(&mut self, op: BinaryOperator, a: Typed, b: Typed) -> Result<Typed, OptimizeError> {
        let b = self.cast(b, &a.ty)?;
        let ty = if matches!(
            op,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
                | BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr
        ) {
            bool_type()
        } else {
            a.ty.clone()
        };
        self.op(OpTag::BinOp(op), vec![a, b], ty)
    }
    fn select(&mut self, c: Typed, a: Typed, b: Typed) -> Result<Typed, OptimizeError> {
        if a.ty == b.ty && select::supported_type(&a.ty) {
            let ty = a.ty.clone();
            return self.op(
                OpTag::Intrinsic {
                    id: catalog().known().select,
                    overload_idx: 0,
                },
                vec![b, a, c],
                ty,
            );
        }
        let Some(start) = self.builder.current_block() else {
            return Err(error("no selection block"));
        };
        let yes = self.builder.create_block();
        let no = self.builder.create_block();
        let (end, p) = self.builder.create_block_with_params(vec![a.ty.clone()]);
        self.builder.set_control_header(start, ControlHeader::Selection { merge: end });
        self.builder
            .terminate(Terminator::CondBranch {
                cond: c.value,
                then_target: yes,
                then_args: vec![],
                else_target: no,
                else_args: vec![],
            })
            .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(yes);
        self.builder
            .terminate(Terminator::Branch {
                target: end,
                args: vec![a.value],
            })
            .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(no);
        let b = self.cast(b, &a.ty)?;
        self.builder
            .terminate(Terminator::Branch {
                target: end,
                args: vec![b.value],
            })
            .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(end);
        Ok(Typed {
            value: p[0].into(),
            ty: a.ty,
        })
    }
    fn primitive(&mut self, name: &str, args: Vec<Typed>) -> Result<Typed, OptimizeError> {
        let known = catalog().known();
        match (name, args.as_slice()) {
            ("local_id", []) => self.op(
                OpTag::Intrinsic {
                    id: known.local_id,
                    overload_idx: 0,
                },
                vec![],
                u32_type(),
            ),
            ("global_id", [_]) => {
                let mut index = self.op(
                    OpTag::Intrinsic {
                        id: known.thread_id,
                        overload_idx: 0,
                    },
                    vec![],
                    u32_type(),
                )?;
                if self.grid_yz != [1, 1] {
                    let y = self.op(
                        OpTag::Intrinsic {
                            id: known.thread_id_y,
                            overload_idx: 0,
                        },
                        vec![],
                        u32_type(),
                    )?;
                    let z = self.op(
                        OpTag::Intrinsic {
                            id: known.thread_id_z,
                            overload_idx: 0,
                        },
                        vec![],
                        u32_type(),
                    )?;
                    let z = self.binary(BinaryOperator::Multiply, z, Self::number(self.grid_yz[0]))?;
                    let yz = self.binary(BinaryOperator::Add, y, z)?;
                    let nx = self.op(
                        OpTag::Intrinsic {
                            id: known.num_workgroups,
                            overload_idx: 0,
                        },
                        vec![],
                        u32_type(),
                    )?;
                    let nx = self.binary(BinaryOperator::Multiply, nx, Self::number(self.width))?;
                    let offset = self.binary(BinaryOperator::Multiply, yz, nx)?;
                    index = self.binary(BinaryOperator::Add, index, offset)?;
                }
                Ok(index)
            }
            ("global_size", [_]) => {
                let n = self.op(
                    OpTag::Intrinsic {
                        id: known.num_workgroups,
                        overload_idx: 0,
                    },
                    vec![],
                    u32_type(),
                )?;
                self.binary(
                    BinaryOperator::Multiply,
                    n,
                    Self::number(self.width * self.grid_yz[0] * self.grid_yz[1]),
                )
            }
            ("length", [a]) => {
                let a = if matches!(a.ty, Type::Constructed(TypeName::Tuple(_), _)) {
                    self.field(a.clone(), 0)?
                } else {
                    a.clone()
                };
                let length = self.op(
                    OpTag::Intrinsic {
                        id: known.length,
                        overload_idx: 0,
                    },
                    vec![a],
                    i32(),
                )?;
                self.cast(length, &u32_type())
            }
            ("dimension", [a, axis]) => {
                let ValueRef::Const(ConstantValue::U32(axis)) = axis.value else {
                    return Err(error("generated dimension requires a constant axis"));
                };
                if axis == 0 {
                    return self.primitive("length", vec![a.clone()]);
                }
                let mut ty = &a.ty;
                let mut axis = axis as usize;
                loop {
                    if let Type::Constructed(TypeName::Tuple(_), fields) = ty {
                        let Some(first) = fields.first() else {
                            return Err(error("empty ranked input"));
                        };
                        ty = first;
                        continue;
                    }
                    let Some(dims) = ty.array_dims() else {
                        return Err(error("array dimension out of rank"));
                    };
                    if let Some(Type::Constructed(TypeName::Size(n), _)) = dims.get(axis) {
                        return Ok(Self::number(
                            u32::try_from(*n).map_err(|_| error("array dimension exceeds u32"))?,
                        ));
                    }
                    if axis < dims.len() {
                        return Err(error("inner array dimension must be statically sized"));
                    }
                    axis -= dims.len();
                    let Some(element) = ty.elem_type() else {
                        return Err(error("array dimension out of rank"));
                    };
                    ty = element;
                }
            }
            ("index", [a, i]) => self.index(a.clone(), i.clone()),
            ("bool_to_u32", [a]) => self.cast(a.clone(), &u32_type()),
            ("slice", [a, n]) => {
                if !a.ty.array_variant().is_some_and(is_array_variant_view) {
                    let (Some(element), Some(size)) = (a.ty.elem_type(), a.ty.array_size()) else {
                        return Err(error("local slice requires an array with fixed capacity"));
                    };
                    let ty = make_array1(
                        element.clone(),
                        array_variant_bounded(),
                        size.clone(),
                        no_buffer(),
                    );
                    let length = self.cast(n.clone(), &i32())?;
                    return self.op(OpTag::Tuple(2), vec![a.clone(), length], ty);
                }
                self.op(
                    OpTag::StorageView(PureViewSource::Inherited),
                    vec![Self::number(0), n.clone(), a.clone()],
                    a.ty.clone(),
                )
            }
            ("max" | "min", [a, b]) => {
                let b = self.cast(b.clone(), &a.ty)?;
                let Type::Constructed(scalar, _) = &a.ty else {
                    return Err(error("min/max requires a numeric scalar"));
                };
                let generic = if name == "min" { known.min } else { known.max };
                let Some(id) = catalog().specialize_numeric(generic, scalar) else {
                    return Err(error(format!("unsupported {name} type {:?}", a.ty)));
                };
                self.op(
                    OpTag::Intrinsic { id, overload_idx: 0 },
                    vec![a.clone(), b],
                    a.ty.clone(),
                )
            }
            ("ceil_div", [a, b]) => {
                let sum = self.binary(BinaryOperator::Add, a.clone(), b.clone())?;
                let sub = self.binary(BinaryOperator::Subtract, sum, Self::number(1))?;
                self.binary(BinaryOperator::Divide, sub, b.clone())
            }
            (_, [a, b]) => {
                let op = match name {
                    "add" => BinaryOperator::Add,
                    "sub" => BinaryOperator::Subtract,
                    "mul" => BinaryOperator::Multiply,
                    "div" => BinaryOperator::Divide,
                    "rem" => BinaryOperator::Remainder,
                    "lt" => BinaryOperator::Less,
                    "ge" => BinaryOperator::GreaterEqual,
                    "ne" => BinaryOperator::NotEqual,
                    "eq" => BinaryOperator::Equal,
                    "and" => BinaryOperator::LogicalAnd,
                    _ => return Err(error(format!("TODO: generated primitive {name}"))),
                };
                self.binary(op, a.clone(), b.clone())
            }
            _ => Err(error(format!("TODO: generated primitive {name}"))),
        }
    }
}
