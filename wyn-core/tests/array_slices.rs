#[path = "support/pipeline.rs"]
mod pipeline;

use wyn_core::{host, lower_ssa_to_wgsl_with_program, CodegenTarget};

#[test]
fn returned_slice_does_not_publish_its_unsliced_backing_buffer() {
    let source = pipeline::source("entry main(xs:[]i32) []i32 = xs[1..length(xs)]");
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap();
    let interface = &output.program.interface;
    let host::Pipeline::Compute(plan) = &interface.pipelines[0] else {
        panic!("compute pipeline");
    };
    let source = plan
        .bindings
        .iter()
        .find_map(|binding| match binding {
            host::Binding::StorageBuffer {
                name, set, binding, ..
            } if name == "xs" => Some((*set, *binding)),
            _ => None,
        })
        .expect("source buffer");
    let result = &interface.source_results[0];
    assert_ne!(
        (result.set, result.binding),
        source,
        "public results have no field for a slice's base offset"
    );
    let host::ResultShape::Rectangular(lengths) = &result.shape else {
        panic!("rectangular slice");
    };
    assert_eq!(lengths.len(), 1);
    assert!(
        !matches!(
            &lengths[0],
            host::Expr::Scalar(host::ScalarExpr::BufferLength { .. })
        ),
        "slice length must not be inferred from backing-buffer capacity"
    );
}
