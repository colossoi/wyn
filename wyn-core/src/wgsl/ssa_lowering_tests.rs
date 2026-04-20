//! WGSL backend unit tests.
//!
//! Populated in later commits once the lowering has real functionality.

#[test]
fn scaffold_placeholder() {
    // Sanity: the module compiles and the `lower` entry point returns
    // an error for an empty program (until the real lowering lands).
    let program = crate::ssa::types::Program {
        functions: Vec::new(),
        entry_points: Vec::new(),
        constants: Vec::new(),
        uniforms: Vec::new(),
        storage: Vec::new(),
    };
    let result = super::lower(&program);
    assert!(
        result.is_err(),
        "scaffold should error until lowering is implemented"
    );
}
