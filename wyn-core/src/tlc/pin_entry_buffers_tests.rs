use super::*;
use crate::ast::TypeName;
use crate::tlc::{self, DefMeta};
use crate::types::TypeExt;
use crate::Compiler;
use polytype::Type;

/// Compile `src` through type-check → TLC → region-pinning and return the
/// pinned program.
fn pin(src: &str) -> Program<tlc::stage::BuffersPinned> {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let type_checked = Compiler::parse(src, &mut node_counter)
        .expect("parse")
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    let program = type_checked.to_tlc(&module_manager, false);
    tlc::pin_entry_buffers(program).expect("pin_entry_buffers")
}

/// The buffer slot of the sole entry's `param_index`-th flattened param.
fn entry_param_buffer(program: &Program<tlc::stage::BuffersPinned>, param_index: usize) -> Type<TypeName> {
    let def = program
        .defs
        .iter()
        .find(|d| matches!(&d.meta, DefMeta::EntryPoint(_)))
        .expect("program has an entry point");
    let (_, params) = extract_lambda_params_ref(&def.body);
    let ty = &params[param_index].1;
    ty.array_buffer().expect("param is an array").clone()
}

#[test]
fn single_view_param_pins_to_binding_zero() {
    // The lone `[]f32` storage param auto-allocates `(set 0, binding 0)`,
    // so after pinning its buffer slot is `Region(0, 0)` — concrete, not a
    // variable.
    let program = pin("#[compute]\n\
         entry sum_array(data: []f32) f32 =\n\
         \x20   reduce(|a: f32, b: f32| a + b, 0.0, data)\n");
    let region = entry_param_buffer(&program, 0);
    assert_eq!(
        region,
        Type::Constructed(TypeName::Buffer(crate::BindingRef::new(0, 0)), vec![]),
        "view param region should be pinned to its auto-allocated binding"
    );
}

#[test]
fn two_view_params_pin_to_distinct_bindings() {
    // Two storage params auto-allocate `(0,0)` and `(0,1)` in declaration
    // order; pinning records each region distinctly.
    let program = pin("#[compute]\n\
         entry add(xs: []f32, ys: []f32) f32 =\n\
         \x20   reduce(|a: f32, b: f32| a + b, 0.0, xs) + reduce(|a: f32, b: f32| a + b, 0.0, ys)\n");
    assert_eq!(
        entry_param_buffer(&program, 0),
        Type::Constructed(TypeName::Buffer(crate::BindingRef::new(0, 0)), vec![]),
    );
    assert_eq!(
        entry_param_buffer(&program, 1),
        Type::Constructed(TypeName::Buffer(crate::BindingRef::new(0, 1)), vec![]),
    );
}

#[test]
fn explicit_storage_attribute_pins_to_its_binding() {
    // An explicit `#[storage(set, binding)]` wins over auto-allocation: the
    // region is pinned to the attribute's `(2, 3)`.
    let program = pin("#[compute]\n\
         entry consume(#[storage(set=2, binding=3, access=read)] data: []f32) f32 =\n\
         \x20   reduce(|a: f32, b: f32| a + b, 0.0, data)\n");
    assert_eq!(
        entry_param_buffer(&program, 0),
        Type::Constructed(TypeName::Buffer(crate::BindingRef::new(2, 3)), vec![]),
    );
}
