// End-to-end local-storage coverage lives in tests/compiler/cases/local_storage*.wyn.

use super::*;
use crate::op::BinaryOperator;
use crate::ssa::builder::FuncBuilder;
use crate::types::{i32, sized_array};

#[test]
fn scalar_padding_is_contracted_but_later_alias_access_still_blocks_reuse() {
    for observe_alias in [false, true] {
        let mut body = FuncBuilder::new(vec![], i32()).finish_unchecked();
        let entry = body.inner.entry;
        let middle = body.inner.create_block();
        let exit = body.inner.create_block();
        let array = sized_array(4, i32());
        let base = body.places.insert(PlaceInfo {
            elem_ty: array.clone(),
            origin: PlaceOrigin::Instruction,
        });
        let element = body.places.insert(PlaceInfo {
            elem_ty: i32(),
            origin: PlaceOrigin::Instruction,
        });
        let zero = ValueRef::Const(ConstantValue::I32(0));
        body.inner.append_void_inst(entry, InstKind::Alloca { result: base });
        body.inner.append_void_inst(
            entry,
            InstKind::PlaceIndex {
                place: base,
                index: zero,
                result: element,
            },
        );
        for _ in 0..64 {
            body.inner.append_void_inst(
                entry,
                InstKind::Store {
                    place: element,
                    value: zero,
                },
            );
        }
        let initial = body.inner.append_inst(
            entry,
            InstKind::Op {
                tag: OpTag::ArrayLit,
                operands: vec![zero; 4],
            },
            array.clone(),
        );
        body.inner.append_void_inst(
            entry,
            InstKind::Store {
                place: base,
                value: initial.into(),
            },
        );
        let snapshot = body.inner.append_inst(entry, InstKind::Load { place: base }, array);
        body.inner.blocks[entry].term = Terminator::Branch {
            target: middle,
            args: vec![],
        };
        for _ in 0..100 {
            body.inner.append_inst(
                middle,
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Add),
                    operands: vec![zero, zero],
                },
                i32(),
            );
        }
        let unrelated = body.places.insert(PlaceInfo {
            elem_ty: i32(),
            origin: PlaceOrigin::Instruction,
        });
        body.inner.append_void_inst(middle, InstKind::Alloca { result: unrelated });
        body.inner.append_void_inst(
            middle,
            InstKind::Store {
                place: unrelated,
                value: zero,
            },
        );
        body.inner.append_inst(middle, InstKind::Load { place: unrelated }, i32());
        body.inner.blocks[middle].term = Terminator::Branch {
            target: exit,
            args: vec![],
        };
        let result = body.inner.append_inst(
            exit,
            InstKind::Op {
                tag: OpTag::Index,
                operands: vec![snapshot.into(), zero],
            },
            i32(),
        );
        if observe_alias {
            body.inner.append_inst(exit, InstKind::Load { place: element }, i32());
        }
        body.inner.blocks[exit].term = Terminator::Return(Some(result.into()));
        let arrays = [(initial, 0), (snapshot, 1)].into_iter().collect();
        let places = storage_places(&body, &arrays);
        assert_eq!(places, [base, element].into_iter().collect());
        let events = storage_events(&body, &arrays, &places);
        assert!(events[&middle].is_empty());
        assert_eq!(events[&entry].len(), 5);
        for block in [entry, exit] {
            assert_eq!(events[&block].concat(), body.inner.blocks[block].insts);
        }
        let analysis = analyze(&body).unwrap();
        assert!(analysis.candidates.contains(&snapshot));
        assert_eq!(
            analysis.existing.get(&snapshot).copied(),
            (!observe_alias).then_some(base)
        );
    }
}

#[test]
fn allocation_reachability_is_limited_to_eligible_webs_and_respects_backedges() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, include_str!("local_storage.egg")).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (Array 1) (Origin 1) (Updated 1) (OriginLoad 1 10 100) (Allocation 10)
        (Flow 100 101)
        (Array 2) (Origin 2) (OriginLoad 2 20 200) (Allocation 20)
        (Flow 200 201)
        (Array 3) (Origin 3) (Updated 3) (OriginLoad 3 30 300) (Allocation 30)
        (Array 4) (Origin 4) (Link 3 4) (Flow 300 301)
        (Array 5) (Origin 5) (Updated 5) (OriginLoad 5 50 500) (Allocation 50)
        (Flow 500 501) (AllocationStep 50 501 502) (Use 501 5)
        (Array 6) (Origin 6) (Updated 6) (OriginLoad 6 60 600) (Allocation 60)
        (Escape 6) (Flow 600 601)
    "#,
        )
        .unwrap();
    graph.parse_and_run_program(None, ANALYSIS_SCHEDULE).unwrap();
    let existing = graph.read(|r| r.lookup("Existing", (1i64,))).unwrap().unwrap();
    assert_eq!(graph.value_to_base::<i64>(existing), 10);
    assert!(graph.read(|r| r.contains("AfterOrigin", (1i64, 101i64))).unwrap());
    for (value, point) in [(2i64, 200i64), (3, 300), (6, 600)] {
        assert!(!graph.read(|r| r.contains("AfterOrigin", (value, point))).unwrap());
        assert!(graph.read(|r| r.lookup("Existing", (value,))).unwrap().is_none());
    }
    assert!(graph.read(|r| r.lookup("Existing", (5i64,))).unwrap().is_none());
    let accessed = graph.read(|r| r.lookup("AccessedLater", (5i64,))).unwrap().unwrap();
    assert!(graph.value_to_base::<bool>(accessed));
}
