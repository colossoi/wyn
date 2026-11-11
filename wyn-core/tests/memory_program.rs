//! End-to-end contracts for the authoritative memory program.
#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::{egglog, CodegenTarget, PipelineTopologyPolicy};

#[test]
fn scalar_entry_survives_direct_memory_emission() {
    let source = pipeline::source("entry scalar(x:i32) i32=x+1");
    let module = pipeline::lower(&source, CodegenTarget::Portable);
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].inputs[0].name, "x");
    assert_eq!(module.entry_points[0].outputs.len(), 1);
    let spirv = wyn_core::lower_ssa_to_spirv(module.clone()).unwrap();
    assert!(!spirv.spirv.is_empty());
    let wgsl = wyn_core::lower_ssa_to_wgsl_with_program(module).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap())
    .unwrap();
}

#[test]
fn packed_product_arrays_pass_through_common_allocation() {
    for source in [
        "entry packed(xs:[](i32,f32)) [](i32,f32)=map(|(i,f)|(i+1,f+2.0),xs)",
        "entry packed(xs:[]((i32,f32),i32)) []((i32,f32),i32)=xs",
    ] {
        let source = pipeline::source(source);
        let program = egglog::from_tlc(&source).unwrap();
        let program = egglog::fuse(program).unwrap();
        let program = egglog::place(program, PipelineTopologyPolicy::AllowGenerated).unwrap();
        let program = egglog::schedule(program).unwrap();
        let program = egglog::prepare_interfaces(program).unwrap();
        let program = egglog::optimize(program).unwrap();
        egglog::prepare_host(program).unwrap();
    }
}

#[test]
fn invocation_local_arrays_use_their_explicit_memory_layouts() {
    let source = pipeline::source("entry mapped(xs:[4]i32) [4]i32=map(|x|x+1,xs)");
    let module = pipeline::lower_with_options(
        &source,
        CodegenTarget::Portable,
        egglog::ScalarOptimization::Basic,
        PipelineTopologyPolicy::AuthoredOnly,
    );
    let spirv = wyn_core::lower_ssa_to_spirv(module.clone()).unwrap();
    assert!(!spirv.spirv.is_empty());
    let wgsl = wyn_core::lower_ssa_to_wgsl_with_program(module).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap())
    .unwrap();
}

#[test]
fn local_control_and_folds_preserve_memory_results() {
    for source in [
        "entry choose(b:bool) i32 = let xs=if b then [1,2] else [3,4] in xs[1]",
        "entry count(n:i32) i32=loop acc=0 for i<n do acc+i",
        "entry total(xs:[4]i32) i32=reduce((+),0,xs)",
        "entry prefix(xs:[4]i32) [4]i32=scan((+),0,xs)",
    ] {
        eprintln!("checking {source}");
        let owned = pipeline::source(source);
        let module = pipeline::lower_with_options(
            &owned,
            CodegenTarget::Portable,
            egglog::ScalarOptimization::Basic,
            PipelineTopologyPolicy::AuthoredOnly,
        );
        wyn_core::lower_ssa_to_spirv(module.clone())
            .unwrap_or_else(|failure| panic!("{source}: {failure}"));
        let wgsl = wyn_core::lower_ssa_to_wgsl_with_program(module)
            .unwrap_or_else(|failure| panic!("{source}: {failure}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap())
        .unwrap();
    }
}

#[test]
fn generated_dispatches_use_common_memory_bindings() {
    for source in [
        "entry mapped(xs:[]i32) []i32=map(|x|x+1,xs)",
        "entry total(xs:[]i32) i32=reduce((+),0,xs)",
        "entry prefix(xs:[]i32) []i32=scan((+),0,xs)",
        "entry packed(xs:[](i32,f32)) [](i32,f32)=map(|(i,f)|(i+1,f+2.0),xs)",
        "entry twice(xs:[]i32) []i32=let n=reduce((+),0,xs) in map(|x|x+n,xs)",
    ] {
        eprintln!("checking {source}");
        let owned = pipeline::source(source);
        let module = pipeline::lower(&owned, CodegenTarget::Portable);
        wyn_core::lower_ssa_to_spirv(module.clone())
            .unwrap_or_else(|failure| panic!("{source}: {failure}"));
        let wgsl = wyn_core::lower_ssa_to_wgsl_with_program(module)
            .unwrap_or_else(|failure| panic!("{source}: {failure}"));
        let parsed = naga::front::wgsl::parse_str(&wgsl.wgsl)
            .unwrap_or_else(|failure| panic!("{source}: {}", failure.emit_to_string(&wgsl.wgsl)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .unwrap_or_else(|failure| panic!("{source}: {failure:?}"));
    }
}

#[test]
fn conditional_dispatches_join_memory_witnesses() {
    for source in [
        "entry choose(b:bool,xs:[]i32) []i32=if b then map(|x|x+1,xs) else map(|x|x*2,xs)",
        "entry choose(b:bool,xs:[]i32) i32=if b then reduce((+),0,xs) else reduce((+),10,xs)",
    ] {
        eprintln!("checking {source}");
        let owned = pipeline::source(source);
        let module = pipeline::lower(&owned, CodegenTarget::Portable);
        wyn_core::lower_ssa_to_spirv(module.clone())
            .unwrap_or_else(|failure| panic!("{source}: {failure}"));
        let wgsl = wyn_core::lower_ssa_to_wgsl_with_program(module)
            .unwrap_or_else(|failure| panic!("{source}: {failure}"));
        let parsed = naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .unwrap();
        let host = wgsl.program.to_whl("choose", wyn_core::host::ShaderFormat::Wgsl).unwrap();
        wyn_host_interp::Program::parse(&host).unwrap();
    }
}

#[test]
fn dispatch_loops_carry_memory_and_scalar_witnesses() {
    check_dispatch_loop("entry repeated(xs:[]i32,n:i32) []i32=loop ys=xs for i<n do map(|x|x+i,ys)");
}

#[test]
fn growing_dispatch_loops_carry_resized_storage() {
    check_dispatch_loop("entry repeated(xs:[]i32,n:i32) []i32=loop ys=map(|x|x,xs) for i<n do map(|j|if j<length(ys) then ys[j]+1 else i,iota(length(ys)+1))");
}

#[test]
fn while_dispatch_loops_carry_memory_and_predicates() {
    check_dispatch_loop("entry repeated(xs:[]i32,n:i32) ([]i32,i32)=loop (ys,i)=(map(|x|x,xs),0) while i<n do (map(|x|x+1,ys),i+1)");
}

fn check_dispatch_loop(source: &str) {
    eprintln!("checking {source}");
    let owned = pipeline::source(source);
    let module = pipeline::lower(&owned, CodegenTarget::Portable);
    wyn_core::lower_ssa_to_spirv(module.clone()).unwrap_or_else(|failure| panic!("{source}: {failure}"));
    let wgsl = wyn_core::lower_ssa_to_wgsl_with_program(module)
        .unwrap_or_else(|failure| panic!("{source}: {failure}"));
    let parsed = naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .unwrap();
    assert_eq!(wgsl.program.interface.dispatch_loops.len(), 1);
    let host = wgsl.program.to_whl("repeated", wyn_core::host::ShaderFormat::Wgsl).unwrap();
    wyn_host_interp::Program::parse(&host).unwrap();
    wgsl.program.to_rust_wgpu("repeated", wyn_core::host::ShaderFormat::Wgsl).unwrap();
}

#[test]
fn typed_views_preserve_rectangular_and_product_storage() {
    for source in [
        "entry rows(xs:[][3]i32) []i32=map(|row|reduce((+),0,row),xs)",
        "entry rows(xs:[][3]i32) [][3]i32=xs",
        "entry nested(xs:[]((i32,f32),i32)) []((i32,f32),i32)=xs",
    ] {
        eprintln!("checking {source}");
        let owned = pipeline::source(source);
        let module = pipeline::lower(&owned, CodegenTarget::Portable);
        wyn_core::lower_ssa_to_spirv(module.clone()).unwrap();
        let wgsl = wyn_core::lower_ssa_to_wgsl_with_program(module).unwrap();
        let parsed = naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .unwrap();
        assert_eq!(wgsl.program.interface.source_results.len(), 1);
    }
}
