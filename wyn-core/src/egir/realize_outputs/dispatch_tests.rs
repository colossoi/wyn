//! Unit tests for the dispatch helpers.

use super::*;
use crate::egir::types::EGraph;

/// A runtime-sized compute output that no retargetable Map/Scan produced
/// must surface a clean `Unsupported` error.
#[test]
fn compute_slot_source_rejects_unsized_array_without_soac() {
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    // Array args = [elem, variant, size]. Use a View variant + a free
    // type variable for the size to model "unsized runtime array."
    let unsized_arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            f32_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(99),
            crate::types::no_region(),
        ],
    );

    let mut graph = EGraph::new();
    let source = graph.alloc_side_effect_result(unsized_arr_ty.clone());
    let block = graph.skeleton.entry;
    let mut next_effect = 1u32;
    let mut aliases = std::collections::HashMap::new();

    let err = compute_slot_source(
        &mut graph,
        &mut aliases,
        &mut next_effect,
        block,
        source,
        0,
        &unsized_arr_ty,
        crate::BindingRef::new(0, 1),
        false,
    )
    .expect_err("runtime-sized array without a producing SOAC must be rejected");
    match err {
        ConvertError::Unsupported(msg) => {
            assert!(msg.contains("runtime-sized array"), "unexpected message: {msg}")
        }
        other => panic!("expected ConvertError::Unsupported, got {other:?}"),
    }
}
