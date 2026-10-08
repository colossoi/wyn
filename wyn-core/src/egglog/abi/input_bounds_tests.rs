//! Check inferred bounds at the interface that consumes them.

use crate::host::BufferLen;
use crate::interface::EntryInputKind;

fn input_length(source: &str, name: &str) -> Option<BufferLen> {
    let program = crate::compile_thru_ssa(source).expect("compile entry interface");
    let mut lengths =
        program.entry_points.iter().flat_map(|entry| &entry.inputs).filter_map(|input| match &input.kind {
            EntryInputKind::Storage { length, .. } if input.name == name => Some(length.clone()),
            _ => None,
        });
    let first = lengths.next().expect("storage input must be published");
    for length in lengths {
        assert_eq!(length, first, "generated stages must agree on the input bound");
    }
    first
}

#[test]
fn slice_only_param_gets_bound() {
    let source = "def N:i32 = 8
        entry e(xs: []vec4f32) vec4f32 =
          let xs = xs[0..N] in
          reduce(|a,b| a+b, @[0.0,0.0,0.0,0.0], xs)";
    assert_eq!(
        input_length(source, "xs"),
        Some(BufferLen::Fixed { bytes: 8 * 16 })
    );
}

#[test]
fn length_call_disqualifies() {
    assert_eq!(
        input_length("entry e(xs: []vec4f32) i32 = length(xs)", "xs"),
        None
    );
}

#[test]
fn let_shadowing_does_not_disqualify_outer() {
    let source = "def N:i32 = 4
        entry e(xs: []vec4f32) vec4f32 =
          let xs = xs[0..N] in
          reduce(|a,b| a+b, @[0.0,0.0,0.0,0.0], xs)";
    assert_eq!(
        input_length(source, "xs"),
        Some(BufferLen::Fixed { bytes: 4 * 16 })
    );
}

#[test]
fn multiple_slices_take_max() {
    let source = "def SMALL:i32 = 4
        def BIG:i32 = 16
        entry e(xs: []vec4f32) vec4f32 =
          let a = xs[0..SMALL] in
          let b = xs[0..BIG] in
          reduce(|x,y| x+y, @[0.0,0.0,0.0,0.0], a) +
            reduce(|x,y| x+y, @[0.0,0.0,0.0,0.0], b)";
    assert_eq!(
        input_length(source, "xs"),
        Some(BufferLen::Fixed { bytes: 16 * 16 })
    );
}

#[test]
fn other_uses_disqualify_a_prefix_bound() {
    for expression in [
        "reduce(|a,b|a+b,0,xs[0..4])+length(xs)",
        "reduce(|a,b|a+b,0,xs[0..4])+xs[7]",
        "reduce(|a,b|a+b,0,xs[1..4])",
        "reduce(|a,b|a+b,0,xs[0..n])",
        "reduce(|a,b|a+b,0,xs[0..4])+reduce(|a,b|a+b,0,xs)",
        "reduce(|a,b|a+b,0,xs[0..4])+reduce(|a,b|a+b,0,map(|i|xs[i],iota(n)))",
    ] {
        let source = format!("entry e(xs:[]i32,n:i32) i32={expression}");
        assert_eq!(input_length(&source, "xs"), None, "{expression}");
    }
}

#[test]
fn input_bounds_are_independent() {
    let source = "entry e(xs:[]i32,ys:[]i32) i32 =
        reduce(|a,b|a+b,0,xs[0..4])+length(ys)";
    assert_eq!(input_length(source, "xs"), Some(BufferLen::Fixed { bytes: 16 }));
    assert_eq!(input_length(source, "ys"), None);
}

#[test]
fn declared_size_remains_the_fallback() {
    // Exceed the push-constant budget so the array uses a storage descriptor.
    assert_eq!(
        input_length("entry e(xs:[64]i32) [64]i32=map(|x|x+1,xs)", "xs"),
        Some(BufferLen::Fixed { bytes: 256 })
    );
}

#[test]
fn prefix_uses_in_branches_take_the_maximum() {
    let source = "entry e(xs:[]i32,choose:bool) i32 =
        if choose then reduce(|a,b|a+b,0,xs[0..4])
        else reduce(|a,b|a+b,0,xs[0..8])";
    assert_eq!(input_length(source, "xs"), Some(BufferLen::Fixed { bytes: 32 }));
}

#[test]
fn lexical_aliases_share_the_complete_use_set() {
    let source = "entry e(xs:[]i32) i32 =
        let ys=xs in reduce(|a,b|a+b,0,ys[0..4])";
    assert_eq!(input_length(source, "xs"), Some(BufferLen::Fixed { bytes: 16 }));
    assert_eq!(input_length(&format!("{source}+length(ys)"), "xs"), None);
}

#[test]
fn a_return_without_an_operand_use_blocks_the_graph_proof() {
    use crate::egglog::{from_tlc, fuse, place, query::Query};
    use egglog_engine::Write;

    let tlc = crate::compile_thru_tlc("entry e(xs:[]i32) i32=reduce(|a,b|a+b,0,xs[0..4])").unwrap();
    let mut program = place(
        fuse(from_tlc(&tlc).unwrap()).unwrap(),
        crate::PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let scope = Query(&program.graph).row("SourceEntryPoint", |_| true).unwrap().unwrap()[1];
    let input = Query(&program.graph).required("SourceParameter", (scope, 0i64)).unwrap();
    assert!(Query(&program.graph).lookup("MaxInputPrefix", (input,)).unwrap().is_some());

    // Add a region returning the input directly. No SourceDirectUse edge can
    // describe this escape; it must still defeat the prefix-only proof.
    program
        .graph
        .update(|mut sink| {
            let returning = sink.add("RegionId", i64::MAX)?;
            sink.set("SourceResult", returning, input)
        })
        .unwrap();
    program
        .graph
        .parse_and_run_program(
            None,
            "(run-schedule (saturate planning-import) physical-bindings)",
        )
        .unwrap();
    assert!(Query(&program.graph).lookup("ParameterPrefixElements", (scope, 0i64)).unwrap().is_none());
}
