use crate::egglog::{convert_program, insert_expressions, optimize, schedule, to_ssa};
use crate::{compile_thru_tlc, lower_ssa_to_wgsl, tlc};

fn compile(source: &str) -> naga::Module {
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program =
        schedule(insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap()).unwrap();
    let ssa = to_ssa(&program.data, crate::CodegenTarget::Wgsl).unwrap();
    let source = lower_ssa_to_wgsl(ssa).unwrap();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{source}"));
    module
}

#[test]
fn fused_map_reduce_reaches_the_existing_wgsl_backend() {
    compile("entry main(xs: []f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, map(|x: f32| x * 2.0, xs))");
}

#[test]
fn scalar_loop_reaches_the_existing_wgsl_backend() {
    compile("entry main(n: i32) i32 = loop acc = 0 for i < n do acc + i");
}

#[test]
fn scan_and_filter_emit_all_scheduled_kernels() {
    let scan = compile("entry main(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)");
    let filter = compile("entry main(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 3 == 1, xs)");
    assert_eq!(scan.entry_points.len(), 3);
    assert_eq!(filter.entry_points.len(), 4);
}

#[test]
fn tuple_collectives_and_runtime_captures_reach_wgsl() {
    compile("entry main(xs: []i32, bias: i32) (i32, i32) = reduce(|a: (i32, i32), b: (i32, i32)| (a.0 + b.0, a.1 + b.1), (0, 0), map(|x: i32| (x + bias, 1), xs))");
}

#[test]
fn nested_device_collectives_reach_wgsl() {
    compile("entry main(xs: []i32) []i32 = map(|x: i32| reduce(|a: i32, b: i32| a + b, 0, map(|y: i32| y + x, iota(5))), xs)");
}

#[test]
fn array_and_while_loops_reach_wgsl() {
    compile(
        "entry main(xs: [4]i32, n: i32) (i32, i32) =
        let a = loop acc = 0 for x in xs do acc + x in
        let (b, _) = loop (acc, i) = (0, 0) while i < n do (acc + i, i + 1) in (a, b)",
    );
}

#[test]
fn conditional_expressions_inside_device_loops_reach_wgsl() {
    compile("entry main(xs: []i32) []i32 = map(|x: i32| loop acc = 0 for i < x do if i % 2 == 0 then acc + i else acc, xs)");
}

#[test]
fn kernels_inside_host_control_are_emitted_without_a_host_descriptor() {
    let source =
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)";
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program =
        schedule(insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap()).unwrap();
    let ssa = to_ssa(&program.data, crate::CodegenTarget::Wgsl).unwrap();
    let lowered = crate::lower_ssa_to_wgsl_with_pipeline(ssa).unwrap();
    assert!(lowered.pipeline.pipelines.is_empty());
    assert!(lowered.pipeline.source_results.is_empty());
    assert!(lowered.pipeline.frame_graph.is_empty());
    assert_eq!(compile(source).entry_points.len(), 1);
}

#[test]
fn existing_spirv_backend_also_accepts_the_handoff() {
    let tlc = tlc::infer_input_slice_bounds(
        compile_thru_tlc("entry main(xs: []i32) []i32 = map(|x: i32| x * 2, xs)").unwrap(),
    );
    let program =
        schedule(insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap()).unwrap();
    let ssa = to_ssa(&program.data, crate::CodegenTarget::Spirv).unwrap();
    let output = crate::lower_ssa_to_spirv(ssa).unwrap();
    let module = wspirv::dr::load_words(output.spirv).unwrap();
    assert_eq!(module.entry_points.len(), 1);
}

#[test]
fn vector_input_and_runtime_gather_after_scan_reach_wgsl() {
    compile(include_str!("../../../testfiles/gather_scan_chain.wyn"));
}
