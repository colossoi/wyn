//! Public-boundary checks for planning without early memory representation.
#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::{egglog, PipelineTopologyPolicy};

fn plan(source: &str) {
    let source = pipeline::source(source);
    let imported = egglog::from_tlc(&source).expect("logical import");
    let fused = egglog::fuse(imported).expect("logical fusion");
    let optimized = egglog::optimize(fused).expect("scalar optimization");
    let placed =
        egglog::place(optimized, PipelineTopologyPolicy::AllowGenerated).expect("logical placement");
    // schedule validates that no physical array bindings or loads/stores were
    // introduced, and every surviving parallel body has an execution strategy.
    egglog::schedule(placed).expect("complete logical strategy selection");
}

#[test]
fn nested_maps_keep_logical_values_through_planning() {
    plan("entry main(xs:[]i32,n:i32) [][]i32=map(|x|map(|y|x+y,iota(n)),xs)");
}

#[test]
fn compaction_accepts_whole_rows_without_offsets() {
    plan("entry main(xs:[][4]i32) [][4]i32=filter(|row|row[0]>0,xs)");
}

#[test]
fn variable_row_collection_is_rejected_before_placement() {
    let source = pipeline::source("entry main(xs:[]i32) [][]i32=map(|x|iota(x),xs)");
    let imported = egglog::from_tlc(&source).expect("logical import");
    let Err(egglog::OptimizeError::Output(message)) = egglog::fuse(imported) else {
        panic!("logical fusion must reject variable rows");
    };
    assert!(message.contains("map callback results must have the same array shape"));
}

#[test]
fn nested_collections_preserve_product_elements() {
    plan("entry main(xs:[]i32,n:i32) [][](i32,bool)=map(|x|map(|i|(x+i,i%2==0),iota(n)),xs)");
}

#[test]
fn scalar_reduction_keeps_its_callback_until_emission() {
    plan("entry main(xs:[]i32) i32=reduce((+),0i32,xs)");
}

#[test]
fn nested_bucket_scatter_keeps_its_callback_and_coordinate_maps() {
    plan("entry main(n:i32) ([][][]i32,[]u32,u32)=bucket_scatter_1d([[iota(0),iota(1)],[iota(2),iota(0)]],[(0,iota(n)),(1,iota(n+1))])");
}

#[test]
fn array_scan_preserves_reducer_scope() {
    plan("entry main(xs:[][2]i32) [][2]i32=scan(|a,b|map2((+),a,b),[0,0],xs)");
}
