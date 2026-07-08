use crate::ast::TypeName;
use crate::error::CompilerError;
use crate::interface::{StorageBindingDecl, StorageRole};
use crate::spirv::verify_buffer_layouts;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{EntryInput, EntryOutput, EntryPoint, ExecutionModel, Program};
use crate::types;
use crate::BindingRef;
use polytype::Type;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

/// Composite runtime-sized `[]u32` — concrete element, runtime length;
/// the canonical legal shape for a storage buffer.
fn runtime_u32_array() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            u32_ty(),
            types::array_variant_composite(),
            Type::Variable(101),
            types::no_region(),
        ],
    )
}

/// `Composite[Variable, NoRegion]` ARRAY whose element is itself a
/// runtime-sized array — the leaf has no fixed size. The shape that
/// reproduces the `*[]u32 with [0] = 42` panic in
/// `spirv/mod.rs::create_storage_buffer`.
fn runtime_array_of_runtime_array() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            runtime_u32_array(),
            types::array_variant_composite(),
            Type::Variable(102),
            types::no_region(),
        ],
    )
}

fn empty_program() -> Program {
    Program {
        functions: Vec::new(),
        entry_points: Vec::new(),
        constants: Vec::new(),
    }
}

fn entry(name: &str) -> EntryPoint {
    let builder = FuncBuilder::new(vec![], Type::Constructed(TypeName::Unit, vec![]));
    EntryPoint {
        name: name.into(),
        body: builder.finish_unchecked(),
        execution_model: ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        storage_bindings: Vec::new(),
        span: crate::ast::Span::dummy(),
    }
}

fn storage_input(name: &str, set: u32, binding: u32, ty: Type<TypeName>) -> EntryInput {
    EntryInput {
        name: name.into(),
        ty,
        decoration: None,
        size_hint: None,
        storage_binding: Some(BindingRef::new(set, binding)),
        storage_access: Some(crate::interface::StorageAccess::ReadWrite),
        uniform_binding: None,
        push_constant: None,
        texture_binding: None,
        texture_backing: None,
        sampler_binding: None,
        storage_image_binding: None,
        length: None,
    }
}

fn storage_output(set: u32, binding: u32, ty: Type<TypeName>) -> EntryOutput {
    EntryOutput {
        ty,
        decoration: None,
        target: None,
        storage_binding: Some(BindingRef::new(set, binding)),
        length: None,
    }
}

#[test]
fn accepts_runtime_array_of_concrete_scalar() {
    let mut e = entry("ok");
    e.inputs.push(storage_input("buf", 2, 0, runtime_u32_array()));
    let mut program = empty_program();
    program.entry_points.push(e);

    verify_buffer_layouts::run(&program).expect("runtime []u32 is the canonical legal storage shape");
}

#[test]
fn rejects_input_whose_elem_is_a_runtime_array() {
    // This is the shape that crashes `create_storage_buffer` —
    // an array whose element type lacks a static byte size.
    let mut e = entry("bad_input");
    e.inputs.push(storage_input("buf", 2, 0, runtime_array_of_runtime_array()));
    let mut program = empty_program();
    program.entry_points.push(e);

    let err = verify_buffer_layouts::run(&program).expect_err("nested-runtime elem must be rejected");
    match err {
        CompilerError::SpirvError(msg, _) => {
            assert!(msg.contains("bad_input"), "names the entry: {msg}");
            assert!(
                msg.contains("set=2") && msg.contains("binding=0"),
                "names the binding: {msg}"
            );
            assert!(msg.contains("has no static size"), "explains the failure: {msg}");
        }
        other => panic!("expected SpirvError, got {other:?}"),
    }
}

#[test]
fn rejects_output_whose_elem_is_a_runtime_array() {
    // Same shape on an output — the path that reproduces the original
    // `buf with [0] = 42u32` panic (returned `*[]u32` becomes an
    // EntryOutput with a non-measurable elem).
    let mut e = entry("bad_output");
    e.outputs.push(storage_output(2, 0, runtime_array_of_runtime_array()));
    let mut program = empty_program();
    program.entry_points.push(e);

    let err = verify_buffer_layouts::run(&program).expect_err("nested-runtime output must be rejected");
    assert!(matches!(err, CompilerError::SpirvError(_, _)));
}

#[test]
fn rejects_compiler_introduced_binding_with_runtime_elem() {
    // `StorageBindingDecl.elem_ty` is the leaf — measured directly,
    // not via `array_elem`. A runtime-array elem at this slot is
    // unmeasurable.
    let mut e = entry("bad_intermediate");
    e.storage_bindings.push(StorageBindingDecl {
        binding: BindingRef::new(3, 4),
        role: StorageRole::Output,
        elem_ty: runtime_u32_array(), // runtime array as the leaf elem
        length: None,
    });
    let mut program = empty_program();
    program.entry_points.push(e);

    let err = verify_buffer_layouts::run(&program).expect_err("runtime-array elem must be rejected");
    match err {
        CompilerError::SpirvError(msg, _) => {
            assert!(msg.contains("bad_intermediate") && msg.contains("set=3") && msg.contains("binding=4"));
        }
        other => panic!("expected SpirvError, got {other:?}"),
    }
}

#[test]
fn accepts_empty_program() {
    verify_buffer_layouts::run(&empty_program()).expect("empty program passes");
}
