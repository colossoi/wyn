#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::projection::side_effect_output_slots;
use super::*;
use crate::ast::Span;
use crate::egir;
use crate::egir::allocation::ResourcesAllocated;
use crate::egir::ir::RealizedOutputRoute;
use crate::egir::program::SlotSource;
use crate::egir::soac::screma;
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, EffectOp, EffectToken, OperandRef,
    Parameters, PlaceAccess, PlaceRegion, PlaceType, WynLanguage,
};
use crate::flow::ExecutionModel;
use crate::interface;
use crate::op;
use crate::types;
use crate::FunctionId;
use wyn_base::IdSource;

pub(crate) const FILTER_SCAN_GROUPS: u32 = model::FILTER_SCAN_GROUPS;
pub(crate) const REDUCE_PHASE1_WIDTH: u32 = model::REDUCE_PHASE1_WIDTH;

pub(crate) fn planned_callable_names(
    program: ResourcesAllocated,
) -> std::result::Result<Vec<String>, String> {
    let existing = program.functions.len();
    let (program, _) = build_parallel_plan(program).map_err(|error| error.to_string())?;
    let names = program.functions[existing..].iter().map(|function| function.name.clone()).collect();
    Ok(names)
}

/// Opaque region used by canonical operator-lambda fixtures.
const OPERATOR_REGION: FunctionId = FunctionId::from_index(0);

fn reduce_operator(neutral: ValueId, captures: Vec<ValueId>) -> screma::Reduce {
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    screma::Reduce {
        operator: screma::Lambda::region(
            SegBody {
                region: OPERATOR_REGION,
                captures: captures.into_iter().map(OperandRef::Value).collect(),
            },
            vec![unit.clone(), unit.clone()],
            vec![unit],
        ),
        neutral: vec![neutral],
        commutative: false,
    }
}

fn scan_operator(neutral: ValueId, captures: Vec<ValueId>) -> screma::Scan {
    let reduction = reduce_operator(neutral, captures);
    screma::Scan {
        operator: reduction.operator,
        neutral: reduction.neutral,
    }
}

fn neutral(graph: &mut EGraph, _index: usize) -> ValueId {
    graph.add_block_param(graph.skeleton.entry, Type::Constructed(TypeName::Unit, vec![]))
}

#[test]
fn output_ownership_comes_from_explicit_route_writer() {
    let mut graph = EGraph::new();
    let block = graph.skeleton.entry;
    let source = neutral(&mut graph, 0);
    let place = graph.add_alloca_place(
        PlaceType {
            pointee: Type::Constructed(TypeName::Unit, vec![]),
            region: PlaceRegion::Function,
            access: PlaceAccess::ReadWrite,
        },
        None,
    );
    let writer = EffectToken::from(9);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Store { place }),
        operands: smallvec![OperandRef::Value(source)],
        result: None,
        effects: Some((EffectToken::from(8), writer)),
        span: None,
    });
    let mut identities = egir::program::ProgramIdentities::default();
    let mut entry = egir::program::AllocatedEntry::new_with_resources(
        "route_test".into(),
        identities.alloc_entry("route_test".into()),
        Span::dummy(),
        ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        (0..4)
            .map(|_| interface::EntryOutput {
                ty: Type::Constructed(TypeName::Unit, vec![]),
                kind: interface::EntryOutputKind::Value {
                    destination: interface::EntryOutputDestination::Plain,
                },
            })
            .collect(),
        vec![],
        Parameters::new(),
        by_value_function_result::<WynLanguage>(Type::Constructed(TypeName::Unit, vec![])),
        graph,
    );
    entry.outputs[3].routes.push(RealizedOutputRoute {
        source: SlotSource { block, value: source },
        writers: vec![OutputWriter::Effect(writer)],
    });
    let resource = SemanticResourceRef(ResourceId::for_test(0));
    entry.outputs[3].resource = Some(resource);

    let entry = egir::program::PlannedEntry::project(&entry).expect("project route fixture");
    assert_eq!(entry.outputs[3].resource, Some(resource));
    let effect = entry
        .graph
        .skeleton
        .blocks
        .values()
        .flat_map(|block| &block.side_effects)
        .next()
        .expect("projected store effect");
    assert_eq!(side_effect_output_slots(&entry, effect), vec![3]);
}

#[test]
fn disjoint_sets_merge_transitive_components() {
    let mut sets = DisjointSets::new(5);
    sets.merge(0, 1);
    sets.merge(1, 3);
    sets.merge(2, 4);

    assert_eq!(sets.representative(0), sets.representative(3));
    assert_eq!(sets.representative(2), sets.representative(4));
    assert_ne!(sets.representative(0), sets.representative(2));
}

#[test]
fn reduction_keeps_canonical_operator_lambda_together() {
    let mut graph = EGraph::new();
    let neutral = neutral(&mut graph, 0);
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    let op = screma::Op::<Semantic> {
        inputs: vec![],
        form: screma::ScremaForm {
            pre: screma::Lambda::identity(vec![unit]),
            scans: vec![],
            reductions: vec![reduce_operator(neutral, vec![neutral])],
            post: screma::Lambda::identity(vec![]),
        },
        result_state: vec![screma::ResultState {
            ownership: types::SoacOwnership::Fresh,
        }],
        state: screma::SemanticState::Serial,
    };
    assert!(op.is_reduce());
    let reduction = &op.form.reductions[0];
    let body = reduction.operator.seg_body().unwrap();
    assert_eq!(body.region, OPERATOR_REGION);
    assert_eq!(body.captures, vec![OperandRef::Value(neutral)]);
    assert_eq!(reduction.neutral, vec![neutral]);
    assert!(!reduction.commutative, "Wyn does not yet declare commutativity");
}

#[test]
fn scan_form_carries_operator_and_neutral() {
    let mut graph = EGraph::new();
    let neutral = neutral(&mut graph, 0);
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    let op = screma::Op::<Semantic> {
        inputs: vec![],
        form: screma::ScremaForm {
            pre: screma::Lambda::identity(vec![unit.clone()]),
            scans: vec![scan_operator(neutral, vec![])],
            reductions: vec![],
            post: screma::Lambda::identity(vec![unit]),
        },
        result_state: vec![screma::ResultState {
            ownership: types::SoacOwnership::Fresh,
        }],
        state: screma::SemanticState::Serial,
    };
    assert!(!op.form.scans.is_empty() && op.form.reductions.is_empty());
    assert_eq!(op.form.scans.len(), 1);
}

#[test]
fn screma_form_carries_scan_and_reduction_operators() {
    let mut graph = EGraph::new();
    let reduce_neutral = neutral(&mut graph, 0);
    let scan_neutral = neutral(&mut graph, 1);
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    let op = screma::Op::<Semantic> {
        inputs: vec![],
        form: screma::ScremaForm {
            pre: screma::Lambda::identity(vec![unit.clone(), unit.clone()]),
            scans: vec![scan_operator(scan_neutral, vec![])],
            reductions: vec![reduce_operator(reduce_neutral, vec![])],
            post: screma::Lambda::identity(vec![unit]),
        },
        result_state: vec![
            screma::ResultState {
                ownership: types::SoacOwnership::Fresh,
            },
            screma::ResultState {
                ownership: types::SoacOwnership::Fresh,
            },
        ],
        state: screma::SemanticState::Serial,
    };
    assert!(!op.form.scans.is_empty() && !op.form.reductions.is_empty());
    assert_eq!(op.form.reductions.len(), 1);
    assert_eq!(op.form.scans.len(), 1);
}

#[test]
fn idle_chunk_start_is_clamped_before_remaining_subtraction() {
    let mut graph = EGraph::new();
    let len = graph.add_block_param(
        graph.skeleton.entry,
        Type::Constructed(TypeName::UInt(32), vec![]),
    );
    let (_, start, _) =
        emit_chunk_arithmetic(&mut graph, REDUCE_PHASE1_WIDTH, len).expect("u32 chunk arithmetic");
    assert!(matches!(
        &graph.nodes[start].kind,
        super::super::types::ValueKind::Pure {
            op: PureOp::Intrinsic { .. },
            operands,
        } if operands.as_slice().last() == Some(&len)
    ));
}

#[test]
fn scan_phase2_writes_exclusive_prefix_before_combining_current_block() {
    let elem_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let mut phase1 = EGraph::new();
    let neutral = phase1.intern_pure(PureOp::Int("0".into()), smallvec![], elem_ty.clone(), None);
    let sums = ResourceId::for_test(40);
    let offsets = ResourceId::for_test(41);
    let mut semantic_ids = egir::program::SemanticOpIdSource::default();
    let mut effect_ids = IdSource::new();
    let mut identities = egir::program::ProgramIdentities::default();
    let operator_id = identities.alloc_function("combine".into());
    let operator_params = [
        callable_parameter::<SemanticResourceRef, WynLanguage>("left".into(), elem_ty.clone()),
        callable_parameter::<SemanticResourceRef, WynLanguage>("right".into(), elem_ty.clone()),
    ]
    .into_iter()
    .collect::<Parameters<_, _>>();
    let parameter_ids = operator_params.ids().collect::<Vec<_>>();
    let mut operator_graph = EGraph::new();
    let left = operator_graph.add_test_value_parameter(parameter_ids[0], elem_ty.clone());
    let right = operator_graph.add_test_value_parameter(parameter_ids[1], elem_ty.clone());
    let combined = operator_graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![left, right],
        elem_ty.clone(),
        None,
    );
    operator_graph.skeleton.blocks[operator_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(operator_graph.value_result(combined)));
    let operator = Func::<Semantic>::new(
        operator_id,
        "combine".into(),
        Span::dummy(),
        None,
        operator_params,
        by_value_function_result::<WynLanguage>(elem_ty.clone()),
        CallEffects::Pure,
        operator_graph,
    );
    let phase2 = ScanPhase2Spec {
        entry_name: "prefix".into(),
        operator: &operator,
        elem_ty,
        source_graph: &phase1,
        operator_captures: &[],
        capture_inputs: &[],
        neutral,
        scratch: ScanScratch {
            block_sums: sums,
            block_offsets: offsets,
        },
        total_out: None,
        reduction_output: None,
    }
    .build(&mut identities, &mut semantic_ids, &mut effect_ids)
    .expect("phase2 synthesis");

    let stored_value = phase2
        .body
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .find_map(|effect| {
            let SideEffectKind::Effect(EffectOp::Store { place }) = &effect.kind else {
                return None;
            };
            let value = effect.operands.first()?.value()?;
            let resource = match &phase2.body.graph.place(*place).ty().region {
                egir::types::PlaceRegion::Resource(resource) => Some(*resource),
                _ => None,
            };
            (resource == Some(SemanticResourceRef(offsets))).then_some(value)
        })
        .expect("block-offset store");
    assert!(matches!(
        phase2.body.graph.nodes[stored_value].kind,
        super::super::types::ValueKind::BlockParam { .. }
    ));
}
