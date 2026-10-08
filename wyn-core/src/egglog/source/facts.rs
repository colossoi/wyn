//! Operation-local facts emitted during the authoritative TLC walk.
use super::{Import, Summary, Term};
use crate::builtins::lowering::PrimOp;
use crate::builtins::{by_id, catalog, BuiltinLowering, Purity};
use crate::egglog::OptimizeError;
use crate::ssa::layout::{std430_type_layout, storage_elem_stride, storage_value_type, type_byte_size};
use crate::tlc::{SoacOp, TermKind, VarRef};
use crate::types::{
    array_view_buffer, as_soa_tuple, is_array_variant_composite, is_array_variant_view, is_copy,
    strip_existentials, Type, TypeExt, TypeName,
};
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, RawValues, Value, Write};

impl<'source> Import<'_, '_, '_, 'source> {
    pub(super) fn register_source(&mut self, value: Value) -> Result<(), OptimizeError> {
        if !self.source_ordinals.contains_key(&value) {
            let ordinal = self.source_ordinals.len() as i64;
            self.sink.set("SourceOrdinal", value, ordinal)?;
            self.source_ordinals.insert(value, ordinal);
        }
        Ok(())
    }

    pub(super) fn ty(&mut self, ty: &Type) -> Result<Value, OptimizeError> {
        let token = self.identities.types.intern(ty);
        if let Some(&key) = self.imported_types.get(&token) {
            return Ok(key);
        }
        let key = self.sink.add("TypeId", token)?;
        let semantic = strip_existentials(ty);
        if semantic.is_array() {
            self.sink.add("SourceArrayType", key)?;
            if semantic
                .array_variant()
                .is_some_and(|variant| !crate::types::is_array_variant_virtual(variant))
            {
                self.sink.add("SourceBufferedArrayType", key)?;
            }
            if !semantic.array_variant().is_some_and(is_array_variant_view) {
                self.sink.add("SourceOwnedArrayType", key)?;
            }
        }
        let mut array = semantic;
        while let Some(fields) = as_soa_tuple(array) {
            let Some(first) = fields.first() else {
                break;
            };
            array = first;
        }
        if array.is_array() {
            if let Some(dimension) = array.array_size() {
                let dimension = self.ty(dimension)?;
                self.sink.set("SourceArrayDimension", key, dimension)?;
            }
            let size = if let Some(Type::Constructed(TypeName::Size(n), _)) = array.array_size() {
                self.sink.add("FixedSize", *n as i64)?
            } else {
                self.sink.add("DynamicSize", RawValues(vec![]))?
            };
            self.sink.set("SourceArraySize", key, size)?;
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = semantic {
            let Type::Constructed(name, _) = semantic else {
                unreachable!()
            };
            let shape = Type::Constructed(name.clone(), vec![crate::types::unit(); fields.len()]);
            let shape = self.identities.types.intern(&shape);
            let shape = self.sink.add("TypeId", shape)?;
            self.sink.set("SourceAggregateShape", key, shape)?;
            let fields = fields.iter().map(|field| self.ty(field)).collect::<Result<Vec<_>, _>>()?;
            let fields = self.sink.container_to_value(VecContainer {
                data: fields,
                do_rebuild: true,
            });
            self.sink.set("SourceTypeFields", key, fields)?;
        } else if !semantic.is_array() && matches!(semantic, Type::Constructed(_, _)) {
            self.sink.add("SourceAtomicType", key)?;
        }
        let storage = storage_value_type(ty);
        if let Some((size, align)) = std430_type_layout(&storage) {
            self.sink.add("SourceBlockLayout", (key, i64::from(size), i64::from(align)))?;
        }
        let byte_size = type_byte_size(&storage);
        if let Some(size) = byte_size {
            self.sink.set("SourceByteSize", key, i64::from(size))?;
        }
        if let Some(stride) = storage_elem_stride(&storage) {
            self.sink.set("SourceStorageStride", key, i64::from(stride))?;
        }
        if semantic.array_variant().is_some_and(is_array_variant_view) {
            self.sink.add("SourceViewType", key)?;
        }
        // Booleans are local scalar values even though their storage form is u32.
        // Classify aggregates using that form so boolean loop state remains a
        // device-local control boundary rather than exposing its inner SOACs.
        if byte_size.is_some_and(|size| size > 0) {
            self.sink.add("SourceScalarLayout", key)?;
        }
        fn element(ty: &Type) -> Option<Type> {
            if let Some(fields) = crate::types::as_soa_tuple(ty) {
                return Some(crate::types::tuple(
                    fields.iter().map(element).collect::<Option<Vec<_>>>()?,
                ));
            }
            ty.elem_type().cloned()
        }
        if let Some(element) = element(semantic) {
            let element = self.ty(&element)?;
            self.sink.add("SourceArrayElement", (key, element))?;
        }
        // Aggregate shape tokens can be interned before their facts are emitted.
        // Cache only completed imports, independently of the type interner.
        self.imported_types.insert(token, key);
        Ok(key)
    }

    pub(super) fn value_type(&mut self, value: Value, ty: &Type) -> Result<(), OptimizeError> {
        self.register_source(value)?;
        if let Some(binding) = array_view_buffer(ty) {
            let Some(element) = ty.elem_type() else {
                return Err(OptimizeError::Output("bound array has no element".into()));
            };
            let Some(stride) = crate::ssa::layout::storage_elem_stride(&storage_value_type(element)) else {
                return Err(OptimizeError::Output("bound array has no storage stride".into()));
            };
            self.storage_binding(value, binding.set, binding.binding, stride)?;
        }
        let ty = self.ty(ty)?;
        self.sink.set("SourceType", value, ty)?;
        Ok(())
    }

    pub(super) fn properties(&mut self, term: &Term, value: Value) -> Result<(), OptimizeError> {
        let (device, pure, readonly, duplicate, work) = match &term.kind {
            TermKind::App { func, args } => match &func.kind {
                TermKind::BinOp(operator) => {
                    let speculative = operator.op.is_speculatable();
                    (false, speculative, true, true, 1)
                }
                TermKind::UnOp(_) => (false, true, true, true, 1),
                TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                    let builtin = by_id(*id);
                    let Some(overload) = builtin.overloads().get(*overload_idx) else {
                        return Err(OptimizeError::Output("invalid TLC builtin overload".into()));
                    };
                    // An authored update of a storage view already writes
                    // shared storage in both backends. Preserve its effect
                    // ordering even while the TLC builtin remains functional.
                    let storage_update = *id == catalog().known().array_with
                        && args
                            .first()
                            .is_some_and(|arg| arg.ty.array_variant().is_some_and(is_array_variant_view));
                    let pure = builtin.raw.purity == Purity::Pure && !storage_update;
                    let reusable = pure && overload.lowering.is_reusable();
                    let structural = *id == catalog().known().length || *id == catalog().known().slice;
                    (
                        !reusable && !structural,
                        (pure && overload.lowering.is_speculatable()) || structural,
                        pure,
                        reusable || structural,
                        1,
                    )
                }
                TermKind::Var(VarRef::Symbol(symbol)) if self.globals.contains(symbol) => {
                    (false, true, true, true, 1)
                }
                TermKind::Lambda(_) | TermKind::Closure(_) => (false, true, true, true, 1),
                _ => (true, false, false, false, 65),
            },
            TermKind::Index { array, .. } => (
                true,
                false,
                true,
                array_view_buffer(&array.ty).is_some()
                    || array.ty.array_variant().is_some_and(is_array_variant_composite),
                1,
            ),
            TermKind::Soac(soac) => {
                let readonly = matches!(
                    soac,
                    SoacOp::Map { .. }
                        | SoacOp::Reduce { .. }
                        | SoacOp::Scan { .. }
                        | SoacOp::Filter { .. }
                );
                (true, readonly, readonly, false, 65)
            }
            TermKind::Loop { .. } => (false, true, true, false, 65),
            TermKind::Extern(_) => (true, false, false, false, 65),
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Lambda(_)
            | TermKind::Closure(_)
            | TermKind::Let { .. }
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Coerce { .. }
            | TermKind::If { .. }
            | TermKind::ArrayExpr(_)
            | TermKind::Tuple(_)
            | TermKind::TupleProj { .. }
            | TermKind::VecLit(_) => (false, true, true, true, 0),
        };
        self.flags(value, device, pure, readonly, duplicate, work)?;
        if let TermKind::Index { array, .. } = &term.kind {
            if let Some(binding) = array_view_buffer(&array.ty) {
                self.summaries.values.entry(value).or_default().reads.insert(binding);
            }
        }
        if matches!(term.kind, TermKind::Extern(_)) {
            self.sink.add("SourceUnknownWrite", RawValues(vec![]))?;
        }
        Ok(())
    }

    pub(super) fn flags(
        &mut self,
        value: Value,
        device: bool,
        pure: bool,
        readonly: bool,
        duplicate: bool,
        work: i64,
    ) -> Result<(), OptimizeError> {
        self.summaries.values.insert(
            value,
            Summary {
                device,
                pure,
                readonly,
                duplicate,
                work,
                ..Summary::default()
            },
        );
        Ok(())
    }

    pub(super) fn application(
        &mut self,
        value: Value,
        func: &Term,
        args: &[Value],
    ) -> Result<(), OptimizeError> {
        match (&func.kind, args) {
            (TermKind::Var(VarRef::Builtin { id, .. }), &[array, _, _])
                if *id == catalog().known().array_with || *id == catalog().known().array_with_in_place =>
            {
                self.sink.add("SourceSameShape", (value, array))?;
                self.sink.add("SourceResultOperand", (value, 0i64))?;
            }
            (TermKind::Var(VarRef::Builtin { id, .. }), &[array]) if *id == catalog().known().length => {
                self.sink.set("SourceLength", value, array)?;
                self.summaries.lengths.insert(value, array);
            }
            (TermKind::Var(VarRef::Builtin { id, .. }), &[array, start, end])
                if *id == catalog().known().slice =>
            {
                self.sink.add("SourceSlice", (value, array, start, end))?;
            }
            (TermKind::Var(VarRef::Builtin { id, .. }), &[length])
                if *id == catalog().known().scratch_alloc =>
            {
                let extent = self.sink.add("Scalar", length)?;
                self.sink.add("SourceScratch", (value, extent))?;
            }
            (TermKind::Var(VarRef::Builtin { id, .. }), &[array])
                if *id == catalog().known().scratch_annotation =>
            {
                self.summaries.lengths.insert(value, array);
                let extent = self.sink.add("Length", array)?;
                self.sink.add("SourceScratch", (value, extent))?;
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn output(
        &mut self,
        entry: i64,
        value: Value,
        ty: &Type,
        outputs: &[crate::interface::EntryOutputDecl<crate::interface::ResolvedAttribute>],
    ) -> Result<(), OptimizeError> {
        // A single declared aggregate output is one storage value. In
        // particular, indirect commands cannot be split into field buffers.
        if outputs.len() == 1
            && outputs[0].ty == *ty
            && matches!(
                outputs[0].attribute,
                Some(crate::interface::Attribute::Storage { .. })
            )
            && matches!(ty, Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _))
        {
            return self.output_leaf(entry, 0, value, ty, outputs.first());
        }
        match strip_existentials(ty) {
            Type::Constructed(TypeName::Unit | TypeName::SideEffect | TypeName::StorageTexture, _) => {}
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
                for (index, field) in fields.iter().enumerate() {
                    let component = self.sink.add("SourceProjected", (value, index as i64))?;
                    self.value_type(component, field)?;
                    self.use_summary(component, value, false);
                    if !self.projection_summary(component, value, index as i64)? {
                        self.sink.add("SourceProjection", (component, value, index as i64))?;
                    }
                    self.output_leaf(entry, index, component, field, outputs.get(index))?;
                }
            }
            ty => self.output_leaf(entry, 0, value, ty, outputs.first())?,
        }
        Ok(())
    }

    pub(super) fn output_leaf(
        &mut self,
        entry: i64,
        slot: usize,
        value: Value,
        ty: &Type,
        output: Option<&crate::interface::EntryOutputDecl<crate::interface::ResolvedAttribute>>,
    ) -> Result<(), OptimizeError> {
        let id = self.outputs.next_id();
        let binding = output.and_then(|o| o.attribute.as_ref());
        self.sink.set(
            "OutputPinned",
            id,
            matches!(binding, Some(crate::interface::Attribute::Storage { .. })),
        )?;
        if let Some(crate::interface::Attribute::Storage { set, binding, .. }) = binding {
            self.sink.add("AbiOutputBinding", (id, i64::from(*set), i64::from(*binding)))?;
        }
        let ty_key = self.ty(ty)?;
        let array = ty.is_array() || crate::types::as_soa_tuple(ty).is_some();
        self.sink.add("SourceOutput", (id, entry, value, ty_key, array))?;
        self.sink.add("SourceResultSlot", (entry, slot as i64, id))?;
        Ok(())
    }
}

/// Scalar applications remain opaque source expressions for structural passes.
/// Context-dependent queries, memory operations and non-copy values retain sites.
pub(super) fn scalar_application(func: &Term, args: &[Term], result: &Type) -> bool {
    match &func.kind {
        TermKind::BinOp(_) | TermKind::UnOp(_) => true,
        TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
            if *id == catalog().known().slice {
                return true;
            }
            let builtin = by_id(*id);
            let Some(overload) = builtin.overloads().get(*overload_idx) else {
                return false;
            };
            let movable = match &overload.lowering {
                BuiltinLowering::PrimOp(PrimOp::DPdx | PrimOp::DPdy | PrimOp::Fwidth) => false,
                BuiltinLowering::PrimOp(_) | BuiltinLowering::ExtInstSplat { .. } => !args.is_empty(),
                _ => false,
            };
            *id != catalog().known().storage_index
                && movable
                && builtin.raw.purity == Purity::Pure
                && is_copy(result)
                && args.iter().all(|arg| is_copy(&arg.ty))
        }
        _ => false,
    }
}
