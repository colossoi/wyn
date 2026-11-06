// End-to-end local-storage coverage lives in tests/compiler/cases/local_storage*.wyn.

use super::*;
use crate::op::BinaryOperator;
use crate::ssa::builder::FuncBuilder;
use crate::types::{i32, sized_array};
use egglog_engine::EGraph;

#[test]
fn storage_validation_requires_initialization_on_every_incoming_path() {
    for omit_initializer in [false, true] {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(
                None,
                concat!(
                    include_str!("../egglog/flow.egg"),
                    "\n",
                    include_str!("local_storage.egg")
                ),
            )
            .unwrap();
        graph
            .parse_and_run_program(
                None,
                r#"
            (Array 1) (Array 2) (Link 1 2) (Origin 1) (Origin 2) (Updated 1)
            (StorageEntry (FlowNode 0) 3)
            (StorageDefinition (FlowNode 1) 1)
            (FlowNext (FlowNode 0) (FlowNode 1)) (FlowNext (FlowNode 0) (FlowNode 2))
            (FlowNext (FlowNode 1) (FlowNode 3)) (FlowNext (FlowNode 2) (FlowNode 3))
            (FlowUse (FlowNode 3) 2)
            ; Later initialized iterations must not hide an uninitialized first trip.
            (FlowNext (FlowNode 3) (FlowNode 1)) (FlowNext (FlowNode 3) (FlowNode 2))
        "#,
            )
            .unwrap();
        if !omit_initializer {
            graph.parse_and_run_program(None, "(StorageDefinition (FlowNode 2) 2)").unwrap();
        }
        graph.parse_and_run_program(None, ANALYSIS_SCHEDULE).unwrap();
        assert!(graph.read(|r| r.contains("Reuse", 1i64)).unwrap());
        assert_eq!(
            graph.read(|r| r.contains("StorageUninitialized", 1i64)).unwrap(),
            omit_initializer
        );
    }
}

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
        let flow = Dataflow::memory(&body).unwrap();
        let places = body
            .places
            .keys()
            .filter(|&p| flow.graph.read(|r| r.contains("MemoryStoragePlace", memory_id(p))).unwrap())
            .collect();
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
    graph
        .parse_and_run_program(
            None,
            concat!(
                include_str!("../egglog/flow.egg"),
                "\n",
                include_str!("local_storage.egg")
            ),
        )
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (Array 1) (Origin 1) (Updated 1) (OriginLoad 1 10 (FlowNode 100)) (Allocation 10)
        (FlowNext (FlowNode 100) (FlowNode 101))
        (Array 2) (Origin 2) (OriginLoad 2 20 (FlowNode 200)) (Allocation 20)
        (FlowNext (FlowNode 200) (FlowNode 201))
        (Array 3) (Origin 3) (Updated 3) (OriginLoad 3 30 (FlowNode 300)) (Allocation 30)
        (Array 4) (Origin 4) (Link 3 4) (FlowNext (FlowNode 300) (FlowNode 301))
        (Array 5) (Origin 5) (Updated 5) (OriginLoad 5 50 (FlowNode 500)) (Allocation 50)
        (FlowNext (FlowNode 500) (FlowNode 501)) (set (AllocationBoundary (FlowNode 501) 50) true) (FlowUse (FlowNode 501) 5)
        (Array 6) (Origin 6) (Updated 6) (OriginLoad 6 60 (FlowNode 600)) (Allocation 60)
        (Escape 6) (FlowNext (FlowNode 600) (FlowNode 601))
    "#,
        )
        .unwrap();
    graph.parse_and_run_program(None, ANALYSIS_SCHEDULE).unwrap();
    let existing = graph.read(|r| r.lookup("Existing", (1i64,))).unwrap().unwrap();
    assert_eq!(graph.value_to_base::<i64>(existing), 10);
    let point = graph.read(|r| r.eclass_of("FlowNode", 101i64)).unwrap().unwrap();
    assert!(graph.read(|r| r.contains("AfterOrigin", (1i64, point))).unwrap());
    for (value, point) in [(2i64, 200i64), (3, 300), (6, 600)] {
        let point = graph.read(|r| r.eclass_of("FlowNode", point)).unwrap().unwrap();
        assert!(!graph.read(|r| r.contains("AfterOrigin", (value, point))).unwrap());
        assert!(graph.read(|r| r.lookup("Existing", (value,))).unwrap().is_none());
    }
    assert!(graph.read(|r| r.lookup("Existing", (5i64,))).unwrap().is_none());
    let accessed = graph.read(|r| r.lookup("AccessedLater", (5i64,))).unwrap().unwrap();
    assert!(graph.value_to_base::<bool>(accessed));
}
