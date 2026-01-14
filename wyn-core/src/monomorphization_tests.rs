use crate::error::CompilerError;

/// Helper to run full pipeline through monomorphization AND lowering
fn compile_through_lowering(input: &str) -> Result<(), CompilerError> {
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(input, &mut node_counter)?;
    let alias_checked = parsed
        .desugar(&mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&module_manager)?
        .alias_check()?;
    let (flattened, mut backend) = alias_checked
        .lower_to_sir(node_counter)?
        .skip_fusion().skip_parallelize()
        .flatten()?;
    flattened
        .hoist_materializations()
        .normalize()
        .monomorphize()?
        .filter_reachable()
        .fold_constants()?
        .lift_bindings()?
        .lower()?;
    Ok(())
}

/// Helper to run full pipeline through monomorphization only
fn compile_through_monomorphization(input: &str) -> Result<(), CompilerError> {
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(input, &mut node_counter)?;
    let alias_checked = parsed
        .desugar(&mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&module_manager)?
        .alias_check()?;
    let (flattened, _backend) = alias_checked
        .lower_to_sir(node_counter)?
        .skip_fusion().skip_parallelize()
        .flatten()?;
    flattened.hoist_materializations().normalize().monomorphize()?;
    Ok(())
}

#[test]
fn test_monomorphization_asserts_on_unresolved_size_params() {
    // This test verifies that monomorphization properly asserts when
    // there are unresolved type variables in the MIR from type checking.
    //
    // The bug is that function bodies with size parameters contain
    // unresolved type variables (like Variable(46)) that should have been
    // resolved during type checking but weren't.

    let source = r#"
def sum<[n]>(arr:[n]f32) f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

#[vertex]
entry vertex_main(vertex_id: i32) #[builtin(position)] vec4f32 =
  let result = sum([1.0f32, 1.0f32, 1.0f32]) in
  @[result, result, 0.0f32, 1.0f32]
"#;

    // The assertion in monomorphization should catch unresolved variables
    // and panic with a clear message about which function/expression has the issue
    compile_through_monomorphization(source).expect("Should compile after bug is fixed");

    // Also test that lowering works
    compile_through_lowering(source).expect("Should compile through lowering after bug is fixed");
}

#[test]
fn test_monomorphization_with_map_and_size_params() {
    // This is the original pattern from de_rasterizer.wyn line 158:
    //   f32.sum (map (\e -> ...) edges)
    // where sum has a size parameter [n]

    let source = r#"
def sum<[n]>(arr:[n]f32) f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

#[vertex]
entry vertex_main(vertex_id: i32) #[builtin(position)] vec4f32 =
  let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
  let result = sum(map((|e:[2]i32| 1.0f32), edges)) in
  @[result, result, 0.0f32, 1.0f32]
"#;
    compile_through_monomorphization(source).expect("Should compile after bug is fixed");
}
