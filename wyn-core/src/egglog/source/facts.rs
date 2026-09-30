//! Operation-local facts emitted during the authoritative TLC walk.
use super::{Import, Summary, Term};
use crate::builtins::lowering::PrimOp;
use crate::builtins::{by_id, catalog, BuiltinLowering, Purity};
use crate::egglog::OptimizeError;
use crate::ssa::layout::type_byte_size;
use crate::tlc::{SoacOp, TermKind, VarRef};
use crate::types::{
    array_view_buffer, is_array_variant_composite, is_array_variant_view, is_copy, strip_existentials,
    Type, TypeExt, TypeName,
};
use egglog_engine::{RawValues, Value, Write};

impl<'source> Import<'_, '_, '_, 'source> {
    pub(super) fn expression_key(&mut self, value: Value) -> Result<Value, OptimizeError> {
        let token = self.identities.values.intern(&value);
        let key = self.sink.add("ExprId", token)?;
        self.sink.set("SourceExprKey", value, key)?;
        Ok(key)
    }

    pub(super) fn ty(&mut self, ty: &Type) -> Result<Value, OptimizeError> {
        let token = self.identities.types.intern(ty);
        let key = self.sink.add("TypeId", token)?;
        let semantic = strip_existentials(ty);
        if semantic.array_variant().is_some_and(is_array_variant_view) {
            self.sink.add("SourceViewType", key)?;
        }
        // Booleans are local scalar values even though their storage form is u32.
        // Classify aggregates using that form so boolean loop state remains a
        // device-local control boundary rather than exposing its inner SOACs.
        if type_byte_size(&crate::ssa::layout::storage_value_type(ty)).is_some_and(|size| size > 0) {
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
        Ok(key)
    }

    pub(super) fn value_type(&mut self, value: Value, ty: &Type) -> Result<(), OptimizeError> {
        self.expression_key(value)?;
        if let Some(binding) = array_view_buffer(ty) {
            self.sink.add(
                "SourceBinding",
                (value, i64::from(binding.set), i64::from(binding.binding)),
            )?;
        }
        let ty = self.ty(ty)?;
        self.sink.set("SourceType", value, ty)?;
        Ok(())
    }

    pub(super) fn properties(&mut self, term: &Term, value: Value) -> Result<(), OptimizeError> {
        let (device, pure, readonly, duplicate, work) = match &term.kind {
            TermKind::App { func, .. } => match &func.kind {
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
                    let pure = builtin.raw.purity == Purity::Pure;
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
            (TermKind::Var(VarRef::Builtin { id, .. }), &[array]) if *id == catalog().known().length => {
                self.sink.add("SourceLength", (value, array))?;
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
                self.sink.add("SourceScratch", (value, length))?;
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
                    self.output_leaf(entry, component, field, outputs.get(index))?;
                }
            }
            ty => self.output_leaf(entry, value, ty, outputs.first())?,
        }
        Ok(())
    }

    fn output_leaf(
        &mut self,
        entry: i64,
        value: Value,
        ty: &Type,
        output: Option<&crate::interface::EntryOutputDecl<crate::interface::ResolvedAttribute>>,
    ) -> Result<(), OptimizeError> {
        if let Some(summary) = self.summaries.values.get(&value) {
            for &operation in &summary.dependencies {
                self.sink.add("SourceObserved", operation)?;
            }
        }
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
