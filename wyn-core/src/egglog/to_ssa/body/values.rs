use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::builtins::{catalog, select};
use crate::host::{Expr, ScalarExpr, ScalarSource};
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::ir::ValueDef;
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::ssa::types::ConstantValue;
use crate::ssa::types::ValueRef;
use crate::ssa::types::{InstKind, PlaceId};
use crate::types::{self, bool_type, strip_existentials, Type, TypeExt, TypeName};
use egglog_engine::{RawValues, Value};

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn fixed_loop_state(
        &mut self,
        value: Typed,
        initial: Option<Value>,
    ) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = &value.ty {
            let name = name.clone();
            let mut values = Vec::new();
            for i in 0..fields.len() {
                let field = self.field(value.clone(), i)?;
                let root = self
                    .query()
                    .constructor("RootPath", RawValues(vec![]))?
                    .ok_or_else(|| error("logical root path missing"))?;
                let path = self.query().constructor("FieldPath", (root, i as i64))?;
                let component = if let (Some(initial), Some(path)) = (initial, path) {
                    self.query()
                        .row("ProductComponent", |row| {
                            row[0] == self.function && row[1] == initial && row[2] == path
                        })?
                        .map(|row| row[3])
                } else {
                    None
                };
                values.push(self.fixed_loop_state(field, component)?);
            }
            let ty = Type::Constructed(name, values.iter().map(|value| value.ty.clone()).collect());
            return self.op(OpTag::Tuple, values, ty);
        }
        if value.ty.array_variant().is_some_and(types::is_array_variant_view) {
            let exact = if let Some(initial) = initial {
                self.query()
                    .lookup("LogicalOuterCount", (self.function, initial))?
                    .map(|count| self.query().lookup("StaticIndex", (self.function, count)))
                    .transpose()?
                    .flatten()
                    .and_then(|size| usize::try_from(self.query().0.value_to_base::<i64>(size)).ok())
            } else {
                None
            };
            let size = exact.or_else(|| match value.ty.array_size() {
                Some(Type::Constructed(TypeName::Size(size), _)) => Some(*size),
                _ => match self.host_array_length(&value) {
                    Ok(Expr::Integer(size)) => usize::try_from(size).ok(),
                    _ => None,
                },
            });
            if let Some(size) = size {
                let element =
                    value.ty.elem_type().cloned().ok_or_else(|| error("loop array element missing"))?;
                let ty = types::sized_array(size, element);
                return self.cast(value, &ty);
            }
        }
        Ok(value)
    }

    pub(in crate::egglog::to_ssa) fn copy_elements(
        &mut self,
        source: Typed,
        destination: Typed,
        count: Typed,
    ) -> Result<Typed, OptimizeError> {
        self.counted(
            Self::number(0),
            count,
            Self::number(1),
            vec![],
            |body, index, _| {
                let value = body.index(source, index.clone())?;
                body.store(destination, index, value)?;
                Ok(vec![])
            },
        )?;
        self.op(OpTag::Unit, vec![], types::unit())
    }
    pub(in crate::egglog::to_ssa) fn coalesce_arrays(
        &mut self,
        arrays: Vec<Typed>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let Some(element) = arrays.first().and_then(|array| array.ty.elem_type()).cloned() else {
            return Err(error("array collection has no element type"));
        };
        if arrays.iter().any(|array| array.ty.elem_type() != Some(&element)) {
            return Err(error(
                "nested collection requires a common payload representation",
            ));
        }
        let Some(stride) = storage_elem_stride(&storage_value_type(&element)) else {
            return Err(error("array collection element has no storage stride"));
        };
        let mut total = self.literal(0, &types::i32())?;
        let mut host_total = Expr::Integer(0);
        let mut lengths = Vec::new();
        for array in &arrays {
            let length = self.array_length(array.clone())?;
            total = self.binary(BinaryOperator::Add, total, length.clone())?;
            host_total = Expr::Add(Box::new(host_total), Box::new(self.host_array_length(array)?));
            lengths.push(length);
        }
        let bytes = Expr::Multiply(
            Box::new(host_total.clone()),
            Box::new(Expr::Integer(stride.into())),
        );
        let (_, payload) = self.device_buffer(&element, bytes, total)?;
        self.host_lengths.insert(payload.value, host_total);
        let mut offset = self.literal(0, &types::i32())?;
        let mut views = Vec::new();
        for (array, length) in arrays.into_iter().zip(lengths) {
            self.counted(
                Self::number(0),
                length.clone(),
                Self::number(1),
                vec![],
                |body, index, _| {
                    let value = body.index(array, index.clone())?;
                    let destination = body.binary(BinaryOperator::Add, offset.clone(), index)?;
                    body.store(payload.clone(), destination, value)?;
                    Ok(vec![])
                },
            )?;
            let view = self.op(
                OpTag::StorageView(PureViewSource::Inherited),
                vec![offset.clone(), length.clone(), payload.clone()],
                payload.ty.clone(),
            )?;
            views.push(view);
            offset = self.binary(BinaryOperator::Add, offset, length)?;
        }
        Ok(views)
    }

    pub(in crate::egglog::to_ssa) fn array_totals(
        &mut self,
        array: Typed,
        depth: usize,
        mut totals: Vec<Typed>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let length = self.array_length(array.clone())?;
        totals[depth] = self.binary(BinaryOperator::Add, totals[depth].clone(), length.clone())?;
        if depth + 1 == totals.len() {
            return Ok(totals);
        }
        self.counted(
            Self::number(0),
            length,
            Self::number(1),
            totals,
            |body, index, totals| {
                let row = body.index(array, index)?;
                body.array_totals(row, depth + 1, totals)
            },
        )
    }

    pub(in crate::egglog::to_ssa) fn flatten_array(
        &mut self,
        array: Typed,
        payload: &Typed,
        offsets: &[Typed],
        depth: usize,
        positions: Vec<Typed>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let length = self.array_length(array.clone())?;
        self.counted(
            Self::number(0),
            length,
            Self::number(1),
            positions,
            |body, index, mut positions| {
                let element = body.index(array, index)?;
                if depth == offsets.len() {
                    body.store(payload.clone(), positions[depth].clone(), element)?;
                } else {
                    positions = body.flatten_array(element, payload, offsets, depth + 1, positions)?;
                    let next =
                        body.binary(BinaryOperator::Add, positions[depth].clone(), Self::number(1))?;
                    body.store(offsets[depth].clone(), next, positions[depth + 1].clone())?;
                }
                positions[depth] =
                    body.binary(BinaryOperator::Add, positions[depth].clone(), Self::number(1))?;
                Ok(positions)
            },
        )
    }

    pub(in crate::egglog::to_ssa) fn snapshot(&mut self, array: Typed) -> Result<Typed, OptimizeError> {
        if array.ty.array_variant().is_some_and(types::is_array_variant_bounded) {
            let storage = self.field(array.clone(), 0)?;
            let storage = self.snapshot(storage)?;
            let count = self.field(array.clone(), 1)?;
            return self.op(OpTag::Tuple, vec![storage, count], array.ty);
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = &array.ty {
            let mut values = Vec::new();
            for i in 0..fields.len() {
                let field = self.field(array.clone(), i)?;
                values.push(self.snapshot(field)?);
            }
            return self.tuple(values);
        }
        let Some(&place) = self.local_arrays.get(&array.value) else {
            return Ok(array);
        };
        let value =
            self.builder.push_inst(InstKind::Load { place }, array.ty.clone()).map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty: array.ty,
        })
    }
    pub(in crate::egglog::to_ssa) fn array_length(&mut self, array: Typed) -> Result<Typed, OptimizeError> {
        if array.ty.array_variant().is_some_and(types::is_array_variant_bounded) {
            return self.field(array, 1);
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = &array.ty {
            if fields.is_empty() {
                return Err(error("empty product has no array length"));
            }
            let first = self.field(array, 0)?;
            return self.array_length(first);
        }
        if array.ty.array_variant().is_some_and(types::is_array_variant_composite) {
            if let Some(Type::Constructed(TypeName::Size(size), _)) = array.ty.array_size() {
                return self.literal(
                    i64::try_from(*size).map_err(|_| error("array length exceeds i64"))?,
                    &types::i32(),
                );
            }
        }
        if array.ty.array_variant().is_some_and(types::is_array_variant_virtual) {
            if let ValueRef::Ssa(id) = array.value {
                let function = self.builder.func();
                if let ValueDef::Inst { inst } = function.values[id].def {
                    if let InstKind::Op {
                        tag: OpTag::Tuple,
                        operands,
                    } = &function.insts[inst].data
                    {
                        if let (Some(value), Some(ty)) = (operands.get(2), array.ty.elem_type()) {
                            return self.cast(
                                Typed {
                                    value: *value,
                                    ty: ty.clone(),
                                },
                                &types::i32(),
                            );
                        }
                    }
                }
            }
        }
        if array.ty.array_variant().is_some_and(types::is_array_variant_view) {
            if let ValueRef::Ssa(id) = array.value {
                let function = self.builder.func();
                if let ValueDef::Inst { inst } = function.values[id].def {
                    if let InstKind::Op {
                        tag: OpTag::StorageView(_),
                        operands,
                    } = &function.insts[inst].data
                    {
                        let value = operands[1];
                        let ty = match value {
                            ValueRef::Ssa(id) => function.values[id].ty.clone(),
                            ValueRef::Const(ConstantValue::I32(_)) => types::i32(),
                            ValueRef::Const(ConstantValue::U32(_)) => Self::number(0).ty,
                            _ => return Err(error("view length must be an integer")),
                        };
                        return self.cast(Typed { value, ty }, &types::i32());
                    }
                }
            }
        }
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().length,
                overload_idx: 0,
            },
            vec![array],
            types::i32(),
        )
    }

    pub(in crate::egglog::to_ssa) fn host_array_length(
        &self,
        array: &Typed,
    ) -> Result<Expr, OptimizeError> {
        if let Some(length) = self.host_lengths.get(&array.value) {
            return Ok(length.clone());
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = &array.ty {
            if let (Some(ty), ValueRef::Ssa(id)) = (fields.first(), array.value) {
                let function = self.builder.func();
                if let ValueDef::Inst { inst } = function.values[id].def {
                    if let InstKind::Op {
                        tag: OpTag::Tuple,
                        operands,
                    } = &function.insts[inst].data
                    {
                        if let Some(value) = operands.first() {
                            return self.host_array_length(&Typed {
                                value: *value,
                                ty: ty.clone(),
                            });
                        }
                    }
                }
            }
        }
        if array.ty.array_variant().is_some_and(types::is_array_variant_bounded) {
            return Err(error("bounded array length has no host witness"));
        }
        if let Some(Type::Constructed(TypeName::Size(size), _)) = array.ty.array_size() {
            return Ok(Expr::Integer(
                i64::try_from(*size).map_err(|_| error("array length exceeds i64"))?,
            ));
        }
        if let ValueRef::Ssa(id) = array.value {
            let function = self.builder.func();
            if let ValueDef::Inst { inst } = function.values[id].def {
                if matches!(
                    &function.insts[inst].data,
                    InstKind::Op {
                        tag: OpTag::StorageView(PureViewSource::Inherited),
                        ..
                    }
                ) {
                    return Err(error("slice length has no host witness"));
                }
            }
        }
        if let Some(binding) = types::array_view_buffer(&array.ty) {
            let Some(element) = array.ty.elem_type() else {
                return Err(error("array element missing"));
            };
            let Some(stride) = crate::ssa::layout::storage_elem_stride(element) else {
                return Err(error("array element has no stride"));
            };
            return Ok(Expr::Scalar(ScalarExpr::BufferLength {
                source: ScalarSource::Binding {
                    set: binding.set,
                    binding: binding.binding,
                },
                stride,
            }));
        }
        Err(error("array length has no host witness"))
    }

    pub(in crate::egglog::to_ssa) fn field(
        &mut self,
        value: Typed,
        index: usize,
    ) -> Result<Typed, OptimizeError> {
        let Some(ty) = (match &value.ty {
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
                fields.get(index).cloned()
            }
            Type::Constructed(TypeName::Vec, fields) => fields.first().cloned(),
            _ if value.ty.array_variant().is_some_and(types::is_array_variant_view)
                && value.ty.array_rank().is_some_and(|rank| index <= rank) =>
            {
                Some(Self::number(0).ty)
            }
            _ if value.ty.array_variant().is_some_and(types::is_array_variant_bounded) => match index {
                0 => Some(types::make_array1(
                    value.ty.elem_type().cloned().ok_or_else(|| error("bounded array element missing"))?,
                    types::array_variant_composite(),
                    value
                        .ty
                        .array_size()
                        .cloned()
                        .ok_or_else(|| error("bounded array capacity missing"))?,
                    types::no_buffer(),
                )),
                1 => Some(types::i32()),
                _ => None,
            },
            _ => None,
        }) else {
            return Err(error(format!("invalid projection {index} of {:?}", value.ty)));
        };
        // Keep physical resource identity when projecting an emitted aggregate.
        // A shared descriptor's storage is not encoded in its source-level type.
        if let ValueRef::Ssa(id) = value.value {
            let function = self.builder.func();
            if let ValueDef::Inst { inst } = function.values[id].def {
                if let InstKind::Op {
                    tag: OpTag::Tuple,
                    operands,
                } = &function.insts[inst].data
                {
                    if let Some(value) = operands.get(index) {
                        return Ok(Typed {
                            value: value.clone(),
                            ty,
                        });
                    }
                }
            }
        }
        self.op(OpTag::Project { index: index as u32 }, vec![value], ty)
    }
    pub(in crate::egglog::to_ssa) fn index_place(
        &mut self,
        array: Typed,
        index: Typed,
    ) -> Result<(PlaceId, Type), OptimizeError> {
        let Some(ty) = array.ty.elem_type().cloned() else {
            return Err(error("indexed value has no element type"));
        };
        let place = self.builder.new_place(ty.clone());
        if let Some(&array) = self.local_arrays.get(&array.value) {
            self.builder
                .push_void_inst(InstKind::PlaceIndex {
                    place: array,
                    index: index.value,
                    result: place,
                })
                .map_err(builder_error)?;
            return Ok((place, ty));
        }
        self.builder
            .push_void_inst(InstKind::ViewIndex {
                view: array.value,
                index: index.value,
                result: place,
            })
            .map_err(builder_error)?;
        Ok((place, ty))
    }
    pub(in crate::egglog::to_ssa) fn canonical_value(
        &mut self,
        value: Typed,
        logical: &Type,
    ) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = logical {
            let mut values = Vec::new();
            for (i, field) in fields.iter().enumerate() {
                let value = self.field(value.clone(), i)?;
                values.push(self.canonical_value(value, field)?);
            }
            let ty = Type::Constructed(
                name.clone(),
                values.iter().map(|value| value.ty.clone()).collect(),
            );
            return self.op(OpTag::Tuple, values, ty);
        }
        if logical.is_array()
            && !(logical.array_variant().is_some_and(types::is_array_variant_composite)
                && matches!(
                    logical.array_size(),
                    Some(Type::Constructed(TypeName::Size(_), _))
                ))
        {
            return Ok(value);
        }
        self.cast(value, logical)
    }

    pub(in crate::egglog::to_ssa) fn cast(
        &mut self,
        value: Typed,
        ty: &Type,
    ) -> Result<Typed, OptimizeError> {
        let ty = strip_existentials(ty);
        if value.ty == *ty {
            return Ok(value);
        }
        if value.ty.array_variant().is_some_and(types::is_array_variant_composite)
            && ty.array_variant().is_some_and(types::is_array_variant_composite)
            && value.ty.elem_type() == ty.elem_type()
            && value.ty.array_dims() == ty.array_dims()
        {
            return Ok(value);
        }
        if value.ty.is_array() && ty.array_variant().is_some_and(types::is_array_variant_composite) {
            if let Some(Type::Constructed(TypeName::Size(count), _)) = ty.array_size() {
                let count = u32::try_from(*count).map_err(|_| error("array capacity exceeds u32"))?;
                let destination = self.local_array(ty.clone())?;
                self.counted(
                    Self::number(0),
                    Self::number(count),
                    Self::number(1),
                    vec![],
                    |body, index, _| {
                        let element = body.index(value.clone(), index.clone())?;
                        body.store(destination.clone(), index, element)?;
                        Ok(vec![])
                    },
                )?;
                return self.snapshot(destination);
            }
        }
        if matches!(ty, Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _)) {
            if let Some(number) = self.integer_literal(&value) {
                return self.literal(number, ty);
            }
        }
        if let (
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), targets),
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), sources),
        ) = (ty, &value.ty)
        {
            if targets.len() != sources.len() {
                return Err(error("physical product conversion arity mismatch"));
            }
            let fields = targets
                .iter()
                .enumerate()
                .map(|(i, ty)| {
                    let field = self.field(value.clone(), i)?;
                    self.cast(field, ty)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return self.op(OpTag::Tuple, fields, ty.clone());
        }
        if *ty == bool_type() {
            let zero = self.literal(0, &value.ty)?;
            return self.binary(BinaryOperator::NotEqual, value, zero);
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
            return Err(error(format!(
                "unsupported physical conversion from {:?} to {ty:?}",
                value.ty
            )));
        };
        self.op(OpTag::Intrinsic { id, overload_idx: 0 }, vec![value], ty.clone())
    }
    pub(in crate::egglog::to_ssa) fn binary(
        &mut self,
        op: BinaryOperator,
        a: Typed,
        b: Typed,
    ) -> Result<Typed, OptimizeError> {
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
    pub(in crate::egglog::to_ssa) fn tuple(&mut self, values: Vec<Typed>) -> Result<Typed, OptimizeError> {
        let ty = types::tuple(values.iter().map(|value| value.ty.clone()).collect());
        self.op(OpTag::Tuple, values, ty)
    }
    pub fn number(n: u32) -> Typed {
        Typed {
            value: ValueRef::Const(ConstantValue::U32(n)),
            ty: Type::Constructed(TypeName::UInt(32), vec![]),
        }
    }
    pub fn local_id(&mut self) -> Result<Typed, OptimizeError> {
        if let Some(subgroup) = &self.subgroup {
            return Ok(subgroup.lane.clone());
        }
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().local_id,
                overload_idx: 0,
            },
            vec![],
            Type::Constructed(TypeName::UInt(32), vec![]),
        )
    }
    pub(in crate::egglog::to_ssa) fn select(
        &mut self,
        c: Typed,
        a: Typed,
        b: Typed,
    ) -> Result<Typed, OptimizeError> {
        if a.ty != b.ty || !select::supported_type(&a.ty) {
            return Err(error("select requires matching scalar or vector types"));
        }
        let ty = a.ty.clone();
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().select,
                overload_idx: 0,
            },
            vec![b, a, c],
            ty,
        )
    }
    pub(in crate::egglog::to_ssa) fn literal(
        &mut self,
        bits: i64,
        ty: &Type,
    ) -> Result<Typed, OptimizeError> {
        let tag = match strip_existentials(ty) {
            Type::Constructed(TypeName::UInt(_), _) => OpTag::Uint((bits as u64).to_string()),
            Type::Constructed(TypeName::Int(_), _) => OpTag::Int(bits.to_string()),
            Type::Constructed(TypeName::Bool, _) => OpTag::Bool(bits != 0),
            Type::Constructed(TypeName::Float(64), _) => {
                OpTag::Float(f64::from_bits(bits as u64).to_string())
            }
            Type::Constructed(TypeName::Float(_), _) => {
                let bits = u32::try_from(bits).map_err(|_| error("invalid folded float bits"))?;
                if matches!(ty, Type::Constructed(TypeName::Float(32), _)) {
                    return Ok(Typed {
                        value: ValueRef::Const(ConstantValue::F32(bits)),
                        ty: ty.clone(),
                    });
                }
                OpTag::Float(f32::from_bits(bits).to_string())
            }
            _ => return Err(error("unsupported scalar literal type")),
        };
        self.op(tag, vec![], ty.clone())
    }
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn local_place(&mut self, ty: Type) -> Result<PlaceId, OptimizeError> {
        let place = self.builder.new_place(ty);
        self.builder.push_void_inst(InstKind::Alloca { result: place }).map_err(builder_error)?;
        Ok(place)
    }

    pub(in crate::egglog::to_ssa) fn load_place(&mut self, place: PlaceId) -> Result<Typed, OptimizeError> {
        let ty = self.builder.place_elem_ty(place).clone();
        let value = self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }

    pub(in crate::egglog::to_ssa) fn store_place(
        &mut self,
        place: PlaceId,
        value: Typed,
    ) -> Result<(), OptimizeError> {
        let ty = self.builder.place_elem_ty(place).clone();
        let value = self.cast(value, &ty)?;
        self.builder
            .push_void_inst(InstKind::Store {
                place,
                value: value.value,
            })
            .map(|_| ())
            .map_err(builder_error)
    }

    /// Allocate mutable local storage; reads used as values are explicit snapshots.
    pub(in crate::egglog::to_ssa) fn local_array(&mut self, ty: Type) -> Result<Typed, OptimizeError> {
        // Preserve the allocation's logical lifetime for reuse across loop iterations.
        let place = self.local_place(ty)?;
        let array = self.load_place(place)?;
        self.local_arrays.insert(array.value, place);
        Ok(array)
    }

    pub(in crate::egglog::to_ssa) fn local_array_with_length(
        &mut self,
        ty: Type,
        count: Typed,
    ) -> Result<Typed, OptimizeError> {
        let array = self.local_array(ty)?;
        let Some(Type::Constructed(TypeName::Size(capacity), _)) = array.ty.array_size() else {
            return Err(error("local array capacity missing"));
        };
        if self.integer_literal(&count).and_then(|value| usize::try_from(value).ok()) == Some(*capacity) {
            return Ok(array);
        }
        let ty = types::make_array1(
            array.ty.elem_type().cloned().ok_or_else(|| error("local array element missing"))?,
            types::array_variant_bounded(),
            Type::Constructed(TypeName::Size(*capacity), vec![]),
            types::no_buffer(),
        );
        let place = self
            .local_arrays
            .get(&array.value)
            .copied()
            .ok_or_else(|| error("local array place missing"))?;
        let count = self.cast(count, &types::i32())?;
        let value = self.op(OpTag::Tuple, vec![array, count], ty)?;
        self.local_arrays.insert(value.value, place);
        Ok(value)
    }
}

impl Body<'_, '_, '_> {
    pub fn index(&mut self, array: Typed, index: Typed) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = &array.ty {
            let mut elements = Vec::new();
            for field in 0..fields.len() {
                let component = self.field(array.clone(), field)?;
                elements.push(self.index(component, index.clone())?);
            }
            return self.tuple(elements);
        }
        if array.ty.array_rank().is_some_and(|rank| rank > 1)
            && array.ty.array_variant().is_some_and(types::is_array_variant_view)
        {
            let shape = self.view_dimensions(array.clone())?;
            let mut offset = self.cast(index, &types::i32())?;
            for dimension in &shape[1..] {
                offset = self.binary(BinaryOperator::Multiply, offset, dimension.clone())?;
            }
            let Type::Constructed(TypeName::Array, mut fields) = array.ty.clone() else {
                return Err(error("ranked view type missing"));
            };
            fields.remove(2);
            let ty = Type::Constructed(TypeName::Array, fields);
            let host_shape = self.host_shapes.get(&array.value).map(|shape| shape[1..].to_vec());
            let mut operands = vec![offset];
            operands.extend_from_slice(&shape[1..]);
            operands.push(array);
            let row = self.op(
                OpTag::StorageView(crate::op::PureViewSource::Inherited),
                operands,
                ty,
            )?;
            if let Some(shape) = host_shape {
                self.host_lengths.insert(row.value, shape[0].clone());
                self.host_shapes.insert(row.value, shape);
            }
            return Ok(row);
        }
        if array.ty.array_variant().is_some_and(|variant| {
            types::is_array_variant_composite(variant)
                || types::is_array_variant_virtual(variant)
                || types::is_array_variant_bounded(variant)
        }) && !self.local_arrays.contains_key(&array.value)
        {
            let Some(element) = array.ty.elem_type().cloned() else {
                return Err(error("indexed array has no element type"));
            };
            let index = if array.ty.array_variant().is_some_and(types::is_array_variant_virtual) {
                self.cast(index, &element)?
            } else {
                index
            };
            return self.op(OpTag::Index, vec![array, index], element);
        }
        self.storage_access(&array.ty, crate::ResourceAccess::Read);
        let (place, ty) = self.index_place(array, index)?;
        let value = self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }

    pub fn store(&mut self, array: Typed, index: Typed, value: Typed) -> Result<(), OptimizeError> {
        if array.ty.array_rank().is_some_and(|rank| rank > 1) {
            let row = self.index(array, index)?;
            let count = self.array_length(value.clone())?;
            self.copy_elements(value, row, count)?;
            return Ok(());
        }
        self.storage_access(&array.ty, crate::ResourceAccess::Write);
        let (place, _) = self.index_place(array, index)?;
        self.store_place(place, value)
    }

    pub(in crate::egglog::to_ssa) fn view_dimensions(
        &mut self,
        array: Typed,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let rank = array.ty.array_rank().ok_or_else(|| error("view rank missing"))?;
        if let ValueRef::Ssa(value) = array.value {
            let function = self.builder.func();
            if let Some(inst) = function.inst_of_value(value) {
                if let InstKind::Op {
                    tag: OpTag::StorageView(_),
                    operands,
                } = &function.insts[inst].data
                {
                    let dimensions = operands
                        .get(1..=rank)
                        .ok_or_else(|| error("view descriptor dimensions missing"))?
                        .iter()
                        .map(|value| {
                            let ty = match value {
                                ValueRef::Ssa(value) => function.value_type(*value).clone(),
                                ValueRef::Const(ConstantValue::I32(_)) => types::i32(),
                                ValueRef::Const(ConstantValue::U32(_)) => Self::number(0).ty,
                                _ => return Err(error("view dimension is not an integer")),
                            };
                            Ok(Typed { value: *value, ty })
                        })
                        .collect::<Result<Vec<_>, OptimizeError>>()?;
                    return dimensions
                        .into_iter()
                        .map(|dimension| self.cast(dimension, &types::i32()))
                        .collect();
                }
            }
        }
        (0..rank)
            .map(|axis| {
                let value = self.field(array.clone(), axis + 1)?;
                self.cast(value, &types::i32())
            })
            .collect()
    }

    pub(super) fn valid_indices(
        &mut self,
        array: Typed,
        indices: &[Typed],
    ) -> Result<Typed, OptimizeError> {
        let Some((index, rest)) = indices.split_first() else {
            return self.op(OpTag::Bool(true), vec![], types::bool_type());
        };
        let length = self.array_length(array.clone())?;
        let length = self.cast(length, &index.ty)?;
        let zero = self.literal(0, &index.ty)?;
        let above = self.binary(BinaryOperator::GreaterEqual, index.clone(), zero)?;
        let below = self.binary(BinaryOperator::Less, index.clone(), length)?;
        let valid = self.binary(BinaryOperator::LogicalAnd, above, below)?;
        if rest.is_empty() {
            return Ok(valid);
        }
        self.branch(
            valid,
            |body| {
                let row = body.index(array, index.clone())?;
                body.valid_indices(row, rest)
            },
            |body| body.op(OpTag::Bool(false), vec![], types::bool_type()),
        )
    }

    pub fn workgroup_array(&mut self, count: u32, element: &Type) -> Result<Typed, OptimizeError> {
        let id = self.next_workgroup_array;
        self.next_workgroup_array += 1;
        self.op(
            OpTag::StorageView(PureViewSource::Workgroup {
                id,
                count: count.max(1),
            }),
            vec![Self::number(0), Self::number(count)],
            types::view_array_with_size(
                &crate::ssa::layout::storage_value_type(element),
                Type::Constructed(TypeName::Size(count as usize), vec![]),
                types::no_buffer(),
            ),
        )
    }

    pub fn workgroup_barrier(&mut self) -> Result<(), OptimizeError> {
        self.builder.push_void_inst(InstKind::ControlBarrier).map(|_| ()).map_err(builder_error)
    }
}
