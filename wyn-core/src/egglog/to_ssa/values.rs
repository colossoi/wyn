use super::{
    builder_error, concrete, error, u32_type, uint, Body, InstKind, OpTag, OptimizeError, PureViewSource,
    Storage, Type, TypeExt, TypeName, Typed, Value,
};
use crate::builtins;
use crate::egglog::{Array, ExprId, ExprKind};
use crate::flow::ControlHeader;
use crate::op::{BinaryOperator, UnaryOperator};
use crate::ssa::types::{ConstantValue, Terminator, ValueRef};
use crate::types;

impl Body<'_, '_> {
    pub(super) fn placeholder(&mut self, key: &str, ty: &Type) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_), fields) = ty {
            let values = fields
                .iter()
                .enumerate()
                .map(|(i, t)| self.placeholder(&format!("{key}_{i}"), t))
                .collect::<Result<Vec<_>, _>>()?;
            return self.tuple(values);
        }
        let (element, len, array) = if let Some(element) = ty.elem_type().filter(|_| ty.is_array()) {
            let len = match ty.array_size() {
                Some(Type::Constructed(TypeName::Size(n), _)) => *n as u32,
                _ => 64,
            };
            (concrete(element)?, len, true)
        } else {
            (
                if *ty == types::bool_type() { u32_type() } else { concrete(ty)? },
                1,
                false,
            )
        };
        let binding = self.compiler.binding(key.into(), element.clone())?;
        let view = self.view(binding, element, len)?;
        if array {
            Ok(view)
        } else {
            let value = self.index(view, Self::number(0))?;
            self.cast(value, ty)
        }
    }
    pub(super) fn seed(&mut self, id: ExprId) -> Result<Typed, OptimizeError> {
        let data = self.compiler.data;
        match &data.expressions[id].kind {
            ExprKind::Tuple(items) => {
                let values = items.iter().map(|&id| self.seed(id)).collect::<Result<Vec<_>, _>>()?;
                self.tuple(values)
            }
            ExprKind::Vector(items) => {
                let values = items.iter().map(|&id| self.seed(id)).collect::<Result<Vec<_>, _>>()?;
                self.op(
                    OpTag::Vector(items.len()),
                    values,
                    data.types[data.expressions[id].ty].ty.clone(),
                )
            }
            ExprKind::Int(_) | ExprKind::FloatBits(_) | ExprKind::Bool(_) | ExprKind::Unit => {
                self.expression(id)
            }
            // Preserve exact identities of materialized producer buffers where
            // scheduling already knows the alias. Other captures are ABI inputs.
            ExprKind::Project { tuple, index } => {
                if let ExprKind::OperationResult(op) = data.expressions[*tuple].kind {
                    if let Some(value) =
                        data.bodies.values().flat_map(|b| &b.instructions).find_map(|i| match i {
                            super::Instruction::BindResult(candidate, value) if *candidate == op => {
                                Some(value)
                            }
                            _ => None,
                        })
                    {
                        if let Value::Tuple(fields) = value {
                            if let Some(Value::Buffer(buffer)) = fields.get(*index) {
                                return self.value(&Value::Buffer(*buffer));
                            }
                        }
                    }
                }
                let value = self.seed(*tuple)?;
                self.field(value, *index)
            }
            ExprKind::Coerce(value) => {
                let value = self.seed(*value)?;
                self.cast(value, &data.types[data.expressions[id].ty].ty)
            }
            ExprKind::PureApp { function, args } => {
                let args = args.iter().map(|&id| self.seed(id)).collect::<Result<Vec<_>, _>>()?;
                self.apply(*function, args, data.types[data.expressions[id].ty].ty.clone())
            }
            ExprKind::Array(array) => self.seed_array(array),
            _ => self.placeholder(
                &format!("capture{}", id.as_u32()),
                &data.types[data.expressions[id].ty].ty,
            ),
        }
    }
    fn seed_array(&mut self, array: &Array) -> Result<Typed, OptimizeError> {
        match array {
            Array::Value(id) => self.seed(*id),
            Array::Zip(arrays) => {
                let values = arrays.iter().map(|a| self.seed_array(a)).collect::<Result<Vec<_>, _>>()?;
                self.tuple(values)
            }
            Array::Literal(items) => {
                for &id in items {
                    let v = self.seed(id)?;
                    self.environment.expressions.insert(id, v);
                }
                self.array(array)
            }
            Array::Range { start, len, step } => {
                for id in [Some(*start), Some(*len), *step].into_iter().flatten() {
                    let value = self.seed(id)?;
                    self.environment.expressions.insert(id, value);
                }
                self.array(array)
            }
        }
    }
    fn view(
        &mut self,
        binding: crate::BindingRef,
        element: Type,
        len: u32,
    ) -> Result<Typed, OptimizeError> {
        let ty = types::view_array_with_size(
            &element,
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            types::buffer_tag(binding),
        );
        self.op(
            OpTag::StorageView(PureViewSource::Storage(binding)),
            vec![Self::number(0), Self::number(len)],
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
        match value {
            Value::Int(n) => Ok(Self::number(*n)),
            Value::Local(name) => self
                .environment
                .locals
                .get(name)
                .cloned()
                .ok_or_else(|| error(format!("unbound scaffold local {name}"))),
            Value::Source(id) => self.expression(*id),
            Value::Tuple(values) => {
                let values = self.values(values)?;
                self.tuple(values)
            }
            Value::Field(value, index) => {
                let value = self.value(value)?;
                self.field(value, *index)
            }
            Value::Array(array) => self.array(array),
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
                let buffer = &self.compiler.data.buffers[*id];
                if let Storage::External(expr) = buffer.storage {
                    return self.expression(expr);
                }
                if buffer.storage == Storage::Function {
                    return Err(error("local buffer used before allocation"));
                }
                let element = concrete(&buffer.element)?;
                let binding = self.compiler.binding(format!("buffer{}", id.as_u32()), element.clone())?;
                let len = if let Value::Int(n) = buffer.length { n } else { 64 };
                self.view(binding, element, len)
            }
            Value::Primitive(name, args) => {
                let values = self.values(args)?;
                self.primitive(name, values)
            }
        }
    }
    pub(super) fn expression(&mut self, id: ExprId) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.environment.expressions.get(&id) {
            return Ok(value.clone());
        }
        let data = self.compiler.data;
        let record = &data.expressions[id];
        let ty = data.types[record.ty].ty.clone();
        match &record.kind {
            ExprKind::Parameter(p) => self
                .environment
                .parameters
                .get(p)
                .cloned()
                .ok_or_else(|| error(format!("unbound parameter {p:?}"))),
            ExprKind::OperationResult(op) => self
                .environment
                .operations
                .get(op)
                .cloned()
                .ok_or_else(|| error(format!("unbound operation result {op:?}"))),
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
                let values = items.iter().map(|&id| self.expression(id)).collect::<Result<Vec<_>, _>>()?;
                self.tuple(values)
            }
            ExprKind::Vector(items) => {
                let values = items.iter().map(|&id| self.expression(id)).collect::<Result<Vec<_>, _>>()?;
                self.op(OpTag::Vector(items.len()), values, ty)
            }
            ExprKind::Project { tuple, index } => {
                let value = self.expression(*tuple)?;
                self.field(value, *index)
            }
            ExprKind::Coerce(inner) => {
                let value = self.expression(*inner)?;
                self.cast(value, &ty)
            }
            ExprKind::Array(array) => self.array(array),
            ExprKind::PureApp { function, args } => {
                let args = args.iter().map(|&id| self.expression(id)).collect::<Result<Vec<_>, _>>()?;
                self.apply(*function, args, ty)
            }
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => {
                let saved = self.environment.expressions.clone();
                for value in self
                    .compiler
                    .placements
                    .get(&crate::egglog::PlacementSite::Expression(id))
                    .cloned()
                    .unwrap_or_default()
                {
                    let computed = self.expression(value)?;
                    self.environment.expressions.insert(value, computed);
                }
                let c = self.expression(*condition)?;
                let start = self.builder.current_block().ok_or_else(|| error("no current block"))?;
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
                let a = self.expression(*then_value)?;
                let p = self.builder.add_block_param(end, a.ty.clone());
                self.builder
                    .terminate(Terminator::Branch {
                        target: end,
                        args: vec![a.value],
                    })
                    .map_err(builder_error)?;
                self.builder.switch_to_block_unchecked(no);
                let b = self.expression(*else_value)?;
                let b = self.cast(b, &a.ty)?;
                self.builder
                    .terminate(Terminator::Branch {
                        target: end,
                        args: vec![b.value],
                    })
                    .map_err(builder_error)?;
                self.builder.switch_to_block_unchecked(end);
                self.environment.expressions = saved;
                Ok(Typed {
                    value: p.into(),
                    ty: a.ty,
                })
            }
            other => Err(error(format!("TODO: first-class value {other:?}"))),
        }
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
            other => return Err(error(format!("TODO: unresolved call target {other:?}"))),
        };
        self.op(tag, args, ty)
    }
    fn array(&mut self, array: &Array) -> Result<Typed, OptimizeError> {
        match array {
            Array::Value(id) => self.expression(*id),
            Array::Zip(arrays) => {
                let values = arrays.iter().map(|a| self.array(a)).collect::<Result<Vec<_>, _>>()?;
                self.tuple(values)
            }
            Array::Literal(items) => {
                let values = items.iter().map(|&id| self.expression(id)).collect::<Result<Vec<_>, _>>()?;
                let Some(first) = values.first() else {
                    return Err(error("TODO: element type of empty array operand"));
                };
                let ty = types::sized_array(values.len(), first.ty.clone());
                self.op(OpTag::ArrayLit(values.len()), values, ty)
            }
            Array::Range { start, len, step } => {
                let a = self.expression(*start)?;
                let n = self.expression(*len)?;
                let n = self.cast(n, &types::i32())?;
                let ty = types::make_array1(
                    a.ty.clone(),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    types::no_buffer(),
                );
                let mut args = vec![a, n];
                if let Some(step) = step {
                    args.push(self.expression(*step)?);
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
        let ty = match &value.ty {
            Type::Constructed(TypeName::Tuple(_), fields) => fields.get(index).cloned(),
            Type::Constructed(TypeName::Vec, fields) => fields.first().cloned(),
            _ => None,
        }
        .ok_or_else(|| error(format!("invalid projection {index} of {:?}", value.ty)))?;
        self.op(OpTag::Project { index: index as u32 }, vec![value], ty)
    }
    pub(super) fn index_place(
        &mut self,
        array: Typed,
        index: Typed,
    ) -> Result<(crate::ssa::types::PlaceId, Type), OptimizeError> {
        if !array.ty.array_variant().is_some_and(types::is_array_variant_view) {
            return Err(error("TODO: writable nested/composite array view"));
        }
        let ty = array.ty.elem_type().cloned().ok_or_else(|| error("indexed value has no element type"))?;
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
        if array.ty.array_variant().is_some_and(types::is_array_variant_view) {
            let (place, ty) = self.index_place(array, index)?;
            let value =
                self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
            return Ok(Typed {
                value: value.into(),
                ty,
            });
        }
        let ty = array.ty.elem_type().cloned().ok_or_else(|| error("index of non-array"))?;
        let index = if array.ty.array_variant().is_some_and(types::is_array_variant_virtual) {
            self.cast(index, &ty)?
        } else {
            index
        };
        self.op(OpTag::Index, vec![array, index], ty)
    }
    pub(super) fn cast(&mut self, value: Typed, ty: &Type) -> Result<Typed, OptimizeError> {
        if value.ty == *ty || (value.ty.is_array() && ty.is_array()) {
            return Ok(value);
        }
        if *ty == types::bool_type() {
            return self.binary(BinaryOperator::NotEqual, value, Self::number(0));
        }
        if value.ty == types::bool_type() {
            let one = self.cast(Self::number(1), ty)?;
            let zero = self.cast(Self::number(0), ty)?;
            return self.select(value, one, zero);
        }
        let (Type::Constructed(target, _), Type::Constructed(source, _)) = (ty, &value.ty) else {
            return Err(error("unsupported conversion"));
        };
        let Some(id) = builtins::catalog().conversion(target, source) else {
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
            types::bool_type()
        } else {
            a.ty.clone()
        };
        self.op(OpTag::BinOp(op), vec![a, b], ty)
    }
    fn select(&mut self, c: Typed, a: Typed, b: Typed) -> Result<Typed, OptimizeError> {
        let start = self.builder.current_block().ok_or_else(|| error("no selection block"))?;
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
        let known = builtins::catalog().known();
        match (name, args.as_slice()) {
            ("global_id", [_]) => self.op(
                OpTag::Intrinsic {
                    id: known.thread_id,
                    overload_idx: 0,
                },
                vec![],
                u32_type(),
            ),
            ("global_size", [_]) => {
                let n = self.op(
                    OpTag::Intrinsic {
                        id: known.num_workgroups,
                        overload_idx: 0,
                    },
                    vec![],
                    u32_type(),
                )?;
                self.binary(BinaryOperator::Multiply, n, Self::number(self.width))
            }
            ("length", [a]) => {
                let a = if matches!(a.ty, Type::Constructed(TypeName::Tuple(_), _)) {
                    self.field(a.clone(), 0)?
                } else {
                    a.clone()
                };
                self.op(
                    OpTag::Intrinsic {
                        id: known.length,
                        overload_idx: 0,
                    },
                    vec![a],
                    u32_type(),
                )
            }
            ("index", [a, i]) => self.index(a.clone(), i.clone()),
            ("bool_to_u32", [a]) => self.cast(a.clone(), &u32_type()),
            ("slice", [a, n]) => {
                if !a.ty.array_variant().is_some_and(types::is_array_variant_view) {
                    return Err(error("TODO: slice local composite array"));
                }
                self.op(
                    OpTag::StorageView(PureViewSource::Inherited),
                    vec![Self::number(0), n.clone(), a.clone()],
                    a.ty.clone(),
                )
            }
            ("max" | "min", [a, b]) => {
                let b = self.cast(b.clone(), &a.ty)?;
                self.op(
                    OpTag::Intrinsic {
                        id: if name == "min" { known.min } else { known.max },
                        overload_idx: 0,
                    },
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
                    "mul" => BinaryOperator::Multiply,
                    "div" => BinaryOperator::Divide,
                    "rem" => BinaryOperator::Remainder,
                    "lt" => BinaryOperator::Less,
                    "ge" => BinaryOperator::GreaterEqual,
                    "ne" => BinaryOperator::NotEqual,
                    "and" => BinaryOperator::LogicalAnd,
                    _ => return Err(error(format!("TODO: generated primitive {name}"))),
                };
                self.binary(op, a.clone(), b.clone())
            }
            _ => Err(error(format!("TODO: generated primitive {name}"))),
        }
    }
}
