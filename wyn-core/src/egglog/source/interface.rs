//! Import declared cross-entry resource identities without inspecting a plan.
use super::Import;
use crate::binding_layout::{
    extract_sampler_binding, extract_storage_access, extract_storage_binding,
    extract_storage_image_binding, extract_storage_image_resource, extract_texture_backing,
    extract_texture_binding, extract_texture_resource, extract_uniform_binding,
};
use crate::egglog::OptimizeError;
use crate::interface::{Attribute, DrawBufferOperand, EntryParamBindingKind, StorageAccess};
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::tlc::data::EntryInputBounds;
use crate::tlc::EntryPoint;
use crate::types::{Type, TypeExt};
use crate::{BindingRef, SymbolId};
use egglog_engine::{Value, Write};

impl Import<'_, '_, '_, '_> {
    fn interface_named(&mut self, kind: &str, name: &str) -> Result<Value, OptimizeError> {
        Ok(self.sink.add("NamedResource", (kind, name))?)
    }
    fn interface_binding(
        &mut self,
        owner: i64,
        slot: BindingRef,
        resource: Value,
    ) -> Result<(), OptimizeError> {
        let binding = self.sink.add("InputBinding", (i64::from(slot.set), i64::from(slot.binding)))?;
        self.sink.add("SourceInterfaceBinding", (owner, binding, resource))?;
        Ok(())
    }
    pub(super) fn interface(
        &mut self,
        entry: &EntryPoint<EntryInputBounds>,
        owner: i64,
        parameters: &[(SymbolId, Type)],
    ) -> Result<(), OptimizeError> {
        for (index, param) in entry.declaration.params.iter().enumerate() {
            let Some((symbol, ty)) = parameters.get(index) else {
                return Err(OptimizeError::Output("interface parameter missing".into()));
            };
            let value = self.resolve(*symbol)?;
            let bound = entry.data.param_bindings.get(index).and_then(Option::as_ref);
            if let Some(EntryParamBindingKind::TupleOfViews(fields)) = bound.map(|b| &b.kind) {
                for (index, field) in fields.iter().enumerate() {
                    let component = self.sink.add("SourceProjected", (value, index as i64))?;
                    let resource = self.interface_named("buffer", &format!("{}_{index}", param.name))?;
                    self.sink.add("SourceInterfaceParameter", (component, resource, 1i64))?;
                    self.interface_binding(owner, field.binding, resource)?;
                }
                continue;
            }
            let buffer =
                bound.map(|binding| binding.first_buffer().0).or_else(|| extract_storage_binding(param));
            if let Some(binding) = buffer {
                let element = ty.elem_type().unwrap_or(ty);
                let Some(stride) = storage_elem_stride(&storage_value_type(element)) else {
                    return Err(OptimizeError::Output(
                        "storage parameter has no element stride".into(),
                    ));
                };
                self.storage_binding(value, binding.set, binding.binding, stride)?;
            }
            let (resource, slot, access) = if let Some(binding) = buffer {
                (
                    self.interface_named("buffer", &param.name)?,
                    Some(binding),
                    extract_storage_access(param).unwrap_or(StorageAccess::ReadOnly),
                )
            } else if let Some(binding) = extract_uniform_binding(param) {
                (
                    self.sink.add(
                        "DescriptorResource",
                        ("uniform", i64::from(binding.set), i64::from(binding.binding)),
                    )?,
                    Some(binding),
                    StorageAccess::ReadOnly,
                )
            } else if let Some(binding) = extract_sampler_binding(param) {
                (
                    self.sink.add(
                        "DescriptorResource",
                        ("sampler", i64::from(binding.set), i64::from(binding.binding)),
                    )?,
                    Some(binding),
                    StorageAccess::ReadOnly,
                )
            } else if let Some((binding, _, access, _)) = extract_storage_image_binding(param) {
                let resource = if let Some(name) = extract_storage_image_resource(param) {
                    self.interface_named("texture", &name)?
                } else {
                    self.sink.add(
                        "DescriptorResource",
                        (
                            "storage-texture",
                            i64::from(binding.set),
                            i64::from(binding.binding),
                        ),
                    )?
                };
                (resource, Some(binding), access)
            } else if let Some(binding) = extract_texture_binding(param) {
                let resource = if let Some(name) = extract_texture_resource(param) {
                    self.interface_named("texture", &name)?
                } else if let Some(backing) = extract_texture_backing(param) {
                    self.sink.add(
                        "DescriptorResource",
                        (
                            "storage-texture",
                            i64::from(backing.set),
                            i64::from(backing.binding),
                        ),
                    )?
                } else {
                    self.interface_named("texture", &param.name)?
                };
                (resource, Some(binding), StorageAccess::ReadOnly)
            } else {
                if param.attributes.iter().any(|a| matches!(a, Attribute::VertexSlot(_))) {
                    let resource = self.interface_named("buffer", &param.name)?;
                    self.sink.add("SourceDrawRead", (owner, resource))?;
                }
                continue;
            };
            let flags: i64 = match access {
                StorageAccess::ReadOnly => 1,
                StorageAccess::WriteOnly => 2,
                StorageAccess::ReadWrite => 3,
            };
            self.sink.add("SourceInterfaceParameter", (value, resource, flags))?;
            if let Some(slot) = slot {
                self.interface_binding(owner, slot, resource)?;
            }
        }
        for (index, output) in entry.declaration.outputs.iter().enumerate() {
            match &output.attribute {
                Some(Attribute::Target(name)) => {
                    let resource = self.interface_named("texture", name)?;
                    self.sink.add("SourceInterfaceOutput", (owner, resource, false))?;
                }
                Some(Attribute::Storage { set, binding, .. }) => {
                    let resource = self
                        .interface_named("buffer", &format!("{}_output_{index}", entry.declaration.name))?;
                    self.interface_binding(owner, BindingRef::new(*set, *binding), resource)?;
                    self.sink.add("SourceInterfaceOutput", (owner, resource, true))?;
                }
                _ => {}
            }
        }
        if let Some(group) = &entry.declaration.graphics_group {
            for buffer in [
                group.invocation.draw.indirect_commands(),
                group.invocation.draw.indices(),
            ]
            .into_iter()
            .flatten()
            {
                match buffer {
                    DrawBufferOperand::Input(buffer) => {
                        let resource = self.interface_named("buffer", buffer.frame_name())?;
                        self.sink.add("SourceDrawRead", (owner, resource))?;
                    }
                    DrawBufferOperand::Result { entry, slot } => {
                        let producer = self.identities.symbols.intern(entry);
                        self.sink.add("SourceDrawValue", (owner, producer, *slot as i64))?;
                    }
                }
            }
        }
        Ok(())
    }
}
