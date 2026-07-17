use super::*;
use crate::ast::TypeName;
use crate::egir::program::SemanticRegion;
use crate::egir::types::{EGraph, PureOp, SkeletonTerminator};
use polytype::Type;
use smallvec::smallvec;

/// Affine composition `(a,b) o (c,d) = (a*c, a*d+b)` is associative
/// but noncommutative, making operand-order bugs visible.
fn compose(left: &(i64, i64), right: &(i64, i64)) -> (i64, i64) {
    (
        left.0.wrapping_mul(right.0),
        left.0.wrapping_mul(right.1).wrapping_add(left.1),
    )
}

fn affine_region() -> (RegionId, LookupMap<RegionId, SemanticRegion>) {
    let int = Type::Constructed(TypeName::Int(64), vec![]);
    let pair = Type::Constructed(TypeName::Tuple(2), vec![int.clone(), int.clone()]);
    let mut graph = EGraph::new();
    let left = graph.add_func_param(0, pair.clone());
    let right = graph.add_func_param(1, pair.clone());
    let la = graph.intern_pure(PureOp::Project { index: 0 }, smallvec![left], int.clone(), None);
    let lb = graph.intern_pure(PureOp::Project { index: 1 }, smallvec![left], int.clone(), None);
    let ra = graph.intern_pure(PureOp::Project { index: 0 }, smallvec![right], int.clone(), None);
    let rb = graph.intern_pure(PureOp::Project { index: 1 }, smallvec![right], int.clone(), None);
    let out_a = graph.intern_pure(PureOp::BinOp("*".into()), smallvec![la, ra], int.clone(), None);
    let scaled = graph.intern_pure(PureOp::BinOp("*".into()), smallvec![la, rb], int.clone(), None);
    let out_b = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![scaled, lb],
        int.clone(),
        None,
    );
    let result = graph.intern_pure(PureOp::Tuple(2), smallvec![out_a, out_b], pair.clone(), None);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    let id = RegionId::from_index(0);
    let mut regions = LookupMap::new();
    regions.insert(
        id,
        SemanticRegion {
            name: "affine_compose".to_string(),
            params: vec![(pair.clone(), "left".into()), (pair.clone(), "right".into())],
            return_ty: pair,
            graph,
            control_headers: LookupMap::new(),
        },
    );
    (id, regions)
}

#[test]
fn region_executor_runs_noncommutative_reduce_and_scan_values() {
    let (region, regions) = affine_region();
    let executor = RegionExecutor::new(&regions);
    for len in [0usize, 1, 63, 64, 65, 513] {
        let inputs: Vec<_> =
            (0..len).map(|index| Value::Tuple(vec![Value::Int(2), Value::Int(index as i64 + 1)])).collect();
        let neutral = Value::Tuple(vec![Value::Int(1), Value::Int(0)]);
        let mut reduction = neutral.clone();
        let mut scan = Vec::new();
        for value in &inputs {
            reduction = executor.call(&region, &[reduction, value.clone()]).unwrap();
            scan.push(reduction.clone());
        }
        assert_eq!(scan.last().cloned().unwrap_or(neutral), reduction, "len={len}");
    }
}

#[test]
fn semantic_reduce_and_scan_cover_dispatch_boundaries() {
    for len in [0usize, 1, 63, 64, 65, 513] {
        let values: Vec<_> = (0..len).map(|i| (2, i as i64 + 1)).collect();
        let reduced = reduce(&values, (1, 0), compose);
        let scanned = inclusive_scan(&values, (1, 0), compose);
        assert_eq!(scanned.last().copied().unwrap_or((1, 0)), reduced, "len={len}");

        let reversed = reduce(&values.iter().cloned().rev().collect::<Vec<_>>(), (1, 0), compose);
        if len > 1 {
            assert_ne!(reduced, reversed, "oracle must detect noncommutative reordering");
        }
    }
}

#[test]
fn semantic_scanomap_routes_both_outputs() {
    let (mapped, scanned) = scanomap(&[1, 2, 3, 4], |x| x * 3, 0, |a, b| a + b);
    assert_eq!(mapped, [3, 6, 9, 12]);
    assert_eq!(scanned, [3, 9, 18, 30]);
}

#[test]
fn semantic_filter_and_ordered_scatter_match_source_contract() {
    assert_eq!(filter(&[1, 2, 3, 4, 5], |x| x % 2 == 1), [1, 3, 5]);
    assert_eq!(scatter(&[0, 0, 0], &[(1, 4), (1, 9), (8, 7)]), [0, 9, 0]);
}

#[test]
fn stepped_range_semantics_are_value_level() {
    let range: Vec<i32> = (0..6).map(|i| 3 + i * 4).collect();
    assert_eq!(map(&range, |x| x * 2), [6, 14, 22, 30, 38, 46]);
}
