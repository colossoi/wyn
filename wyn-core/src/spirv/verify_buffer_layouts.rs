//! Backend-boundary invariant: every storage-binding-typed field in
//! the finalized SSA `Program` must carry an element type that the
//! SPIR-V layout machinery can measure.
//!
//! SPIR-V's `create_storage_buffer` (`spirv/mod.rs`) ends in a
//! `type_byte_size(elem).expect("…")` call that crashes the compiler
//! whenever upstream hands it a non-concrete element shape (view
//! wrappers, abstract array variants, composites with a runtime size
//! at the leaf). Construction sites that produce these typed fields
//! — there are roughly eight, scattered across the EGIR `from_tlc`,
//! `realize_outputs`, `parallelize`, and `soac_expand` passes — each
//! does its own ad-hoc normalization. This pass is the single
//! chokepoint they must implicitly satisfy: it walks every
//! `EntryInput.ty` / `EntryOutput.ty` / `StorageBindingDecl.elem_ty`
//! that carries a `storage_binding`, derives the element type the
//! backend will see, and rejects anything `type_byte_size` can't
//! lay out.
//!
//! Failures surface as a structured `SpirvError` naming the entry,
//! the binding's (set, binding), and the offending type — not a
//! `panic!` from inside the layout helper a few hundred frames
//! deep. That diagnostic also points at the construction site to
//! fix, rather than burying the problem in the backend.
//!
//! Plumbed in `lib.rs` between SSA construction and SPIR-V emission
//! (`SsaConverted::lower`), right after `verify_no_abstract::run`.

use crate::ast::TypeName;
use crate::error::CompilerError;
use crate::interface::StorageBindingDecl;
use crate::ssa::layout::type_byte_size;
use crate::ssa::types::{EntryInput, EntryPoint, Program};
use crate::BindingRef;
use polytype::Type;

pub type Result<T> = std::result::Result<T, CompilerError>;

pub fn run(program: &Program) -> Result<()> {
    for entry in &program.entry_points {
        check_entry(entry)?;
    }
    Ok(())
}

fn check_entry(entry: &EntryPoint) -> Result<()> {
    for input in &entry.inputs {
        if let Some(br) = input.storage_binding {
            check_buffer_elem(&entry.name, br, &input.ty, &input_label(input))?;
        }
    }
    for (i, output) in entry.outputs.iter().enumerate() {
        if let Some(br) = output.storage_binding {
            check_buffer_elem(&entry.name, br, &output.ty, &format!("output #{}", i))?;
        }
    }
    for sb in &entry.storage_bindings {
        check_storage_binding_decl(&entry.name, sb)?;
    }
    Ok(())
}

fn check_storage_binding_decl(entry_name: &str, sb: &StorageBindingDecl) -> Result<()> {
    // `StorageBindingDecl.elem_ty` is conventionally already the array
    // *element* type, not the array — `create_storage_buffer` doesn't
    // strip an array wrapper off it. So measure it directly.
    if type_byte_size(&sb.elem_ty).is_some() {
        return Ok(());
    }
    Err(CompilerError::SpirvError(
        format!(
            "internal: entry `{}` storage binding (set={}, binding={}) elem type {:?} \
             has no static size; SPIR-V's `create_storage_buffer` cannot lay it out. \
             The construction site (likely a EGIR pass producing this `StorageBindingDecl`) \
             needs to reduce the element shape to a concrete fixed-size type before the \
             SSA boundary. See `spirv/verify_buffer_layouts.rs` for context.",
            entry_name, sb.binding.set, sb.binding.binding, sb.elem_ty
        ),
        None,
    ))
}

fn check_buffer_elem(
    entry_name: &str,
    br: BindingRef,
    array_ty: &Type<TypeName>,
    field_label: &str,
) -> Result<()> {
    // Mirror `create_storage_buffer`'s own derivation: try
    // `array_elem`, fall back to `array_ty` itself (the scalar/vec
    // outputs that get packed into a single-element runtime array).
    let elem_ty = crate::types::array_elem(array_ty).unwrap_or(array_ty);
    if type_byte_size(elem_ty).is_some() {
        return Ok(());
    }
    Err(CompilerError::SpirvError(
        format!(
            "internal: entry `{}` {} (set={}, binding={}) lowered to a storage buffer whose \
             element type {:?} has no static size — derived from declared field type {:?}. \
             SPIR-V's `create_storage_buffer` requires a fixed-size element layout. \
             A construction site upstream (likely in `egir::from_tlc`, \
             `egir::realize_outputs`, `egir::parallelize`, or `egir::soac_expand`) is \
             handing through a view-wrapped or composite-element type that didn't get \
             reduced to its concrete leaf. See `spirv/verify_buffer_layouts.rs` for \
             context.",
            entry_name, field_label, br.set, br.binding, elem_ty, array_ty
        ),
        None,
    ))
}

fn input_label(input: &EntryInput) -> String {
    if input.name.is_empty() {
        "input".to_string()
    } else {
        format!("input `{}`", input.name)
    }
}

#[cfg(test)]
#[path = "verify_buffer_layouts_tests.rs"]
mod tests;
