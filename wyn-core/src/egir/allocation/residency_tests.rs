use super::*;
use crate::egir;
use crate::egir::program::GeneratedStageKind;
use crate::tlc;

fn allocate_before_residency(source: &str) -> ResidencyDraft {
    let program = crate::compile_thru_tlc(source).expect("compile residency fixture through TLC");
    let program = tlc::infer_input_slice_bounds(program);
    let program = crate::to_egraph(program).expect("convert residency fixture to EGIR");
    let program = egir::reify_soacs(program);
    let program = egir::optimize_semantic_operations(program).expect("semantic EGIR optimization failed");
    let program = egir::lift_stage_uniform_values(program);
    super::super::allocate_semantic_resources(program).expect("allocate host resources before residency")
}

fn materialization_signature(program: &ResidencyDraft) -> (Vec<GeneratedStageKind>, usize) {
    (
        program.data.stages.stages().filter_map(|(_, origin, _)| origin.generated_kind()).collect(),
        program.data.core.resources.len(),
    )
}

#[test]
fn operation_materializations_restart_to_a_stable_fixpoint() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let program = allocate_before_residency(
                r#"
def N: i32 = 8
entry e() [8]i32 =
    let ys = map(|i: i32| i + 1, 0i32 ..< N) in
    let s = reduce(|a: i32, b: i32| a + b, 0, ys) in
    map(|y: i32| y + s, ys)
"#,
            );

            let resolved = resolve_residency(program).expect("resolve all residency candidates");
            let first = materialization_signature(&resolved);
            assert_eq!(
                first.0,
                vec![GeneratedStageKind::SharedArray, GeneratedStageKind::Scalar],
                "the shared producer is materialized before its scalar consumer"
            );

            let resolved_again = resolve_residency(resolved).expect("re-run residency at its fixpoint");
            assert_eq!(
                materialization_signature(&resolved_again),
                first,
                "a completed residency fixpoint must not add materializations or resources"
            );
        })
        .expect("spawn residency test thread")
        .join()
        .expect("residency test thread panicked");
}
