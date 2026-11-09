//! Structural TLC metadata consumed by the retained placement rules.
use super::Import;
use crate::egglog::OptimizeError;
use crate::interface::{EntryParamBinding, EntryParamBindingKind};
use egglog_engine::{Value, Write};

impl Import<'_, '_, '_, '_> {
    pub(super) fn entry_binding(&mut self, binding: &EntryParamBinding) -> Result<(), OptimizeError> {
        let value = self.resolve(binding.param_sym)?;
        self.sink.set("SourceStorage", value, true)?;
        self.sink.set("SourceRequiresDevice", value, true)?;
        match &binding.kind {
            EntryParamBindingKind::Single {
                binding, elem_bytes, ..
            } => {
                self.storage_binding(value, binding.set, binding.binding, *elem_bytes)?;
            }
            EntryParamBindingKind::TupleOfViews(fields) => {
                for (index, field) in fields.iter().enumerate() {
                    let component = self.sink.add("SourceProjected", (value, index as i64))?;
                    self.register_source(component)?;
                    let info = self.sink.add("ProjectionInfo", (value, index as i64))?;
                    self.sink.set("SourceProjection", component, info)?;
                    self.sink.add("SourceField", (value, index as i64, component))?;
                    self.storage_binding(
                        component,
                        field.binding.set,
                        field.binding.binding,
                        field.elem_bytes,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn storage_binding(
        &mut self,
        value: Value,
        set: u32,
        binding: u32,
        stride: u32,
    ) -> Result<(), OptimizeError> {
        self.sink.add("SourceBinding", (value, i64::from(set), i64::from(binding)))?;
        self.register_source(value)?;
        let binding = self.sink.add("InputBinding", (i64::from(set), i64::from(binding)))?;
        let storage = self.sink.add("StorageInput", (binding, i64::from(stride)))?;
        self.sink.set("AbiStorage", value, storage)?;
        Ok(())
    }
}
