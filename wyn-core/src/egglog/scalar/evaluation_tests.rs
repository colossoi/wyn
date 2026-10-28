use super::*;
use crate::egglog::{from_tlc, fuse, optimize_for_test, place, schedule};
use crate::PipelineTopologyPolicy;
use egglog_engine::{Core, TermDag};

#[test]
fn shared_ranks_preserve_request_ordering_and_choice_boundaries() {
    let mut dag = TermDag::default();
    let leaf = dag.app("ExprLeaf".into(), vec![]);
    let a = dag.app("ExprChoice".into(), vec![leaf; 6]);
    let b = dag.app("ExprChoice".into(), vec![leaf, leaf, leaf, a, leaf, leaf]);
    let nested = dag.app("ExprChoice".into(), vec![leaf, leaf, a, leaf, b, leaf]);
    let nil = dag.app("ExprNil".into(), vec![leaf]);
    let tail = dag.app("ExprCons".into(), vec![leaf, nested, nil]);
    let middle = dag.app("ExprCons".into(), vec![leaf, b, tail]);
    let args = dag.app("ExprCons".into(), vec![leaf, a, middle]);
    let pure = dag.app("ExprTuple".into(), vec![leaf, leaf, args]);
    let effectful = dag.app("ExprVector".into(), vec![leaf, leaf, args]);
    let n = dag.size();
    let mut plan = Evaluation {
        dependencies: vec![Vec::new(); n],
        readonly: vec![true; n],
        total: vec![false; n],
        targets: LookupMap::default(),
        enclosing: vec![Vec::new(); n],
        owners: vec![None; n],
        ordinals: vec![None; n],
        before: LookupMap::default(),
        shared: LookupMap::default(),
        schedules: LookupMap::default(),
        common: LookupMap::default(),
    };
    plan.readonly[effectful] = false;
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, include_str!("evaluation.egg")).unwrap();
    let keys = graph
        .update(|mut sink| {
            for term in 0..n {
                let Term::App(name, fields) = dag.get(term) else {
                    unreachable!()
                };
                let deps = if name == "ExprChoice" { vec![fields[2]] } else { operands(name, fields) };
                for &child in &deps {
                    sink.add("ExprScheduleOperand", (term as i64, child as i64))?;
                }
                plan.dependencies[term] = deps;
                sink.set("ExprScheduleWeight", term as i64, i64::from(name == "ExprChoice"))?;
                if name == "ExprChoice" {
                    sink.set("ExprScheduleCondition", term as i64, fields[2] as i64)?;
                }
            }
            Ok::<_, egglog_engine::Error>([sink.base_to_value(0i64), sink.base_to_value(1i64)])
        })
        .unwrap();
    // Producers behind leaves have separate anchors, even when their ranks are shared.
    plan.dependencies[leaf] = vec![nested];
    let selected = Selected {
        dag,
        values: Vec::new(),
        roots: LookupMap::default(),
        extents: [(keys[0], pure), (keys[1], effectful)].into_iter().collect(),
    };
    plan.schedule_anchors(&selected, &mut graph).unwrap();
    let pure_order = plan.schedule(&[pure]).unwrap();
    assert_eq!(pure_order, &[vec![leaf], vec![a, b], vec![nested], vec![pure]]);
    assert_eq!(
        plan.schedule(&[effectful]).unwrap(),
        &[vec![leaf], vec![a], vec![b], vec![nested], vec![effectful]],
    );
    assert_eq!(plan.schedule(&[leaf]).unwrap(), &[vec![leaf]]);
    assert!(
        plan.schedule(&[leaf, a]).is_ok(),
        "grouped arms receive their own schedule"
    );
}

#[test]
fn shared_placement_queries_match_individual_legal_site_selection() {
    let tlc = crate::compile_thru_tlc(
        "def walk(x:f32,n:i32) f32 =
           loop sum=0.0f32 for k<n do sum+x*x+f32(k)
         entry first(xs:[]f32,n:i32) []f32 = map(|x|walk(x,n)+x*2.0f32,xs)
         entry second(xs:[]f32,n:i32) []f32 = map(|x|walk(x+1.0f32,n)+x*3.0f32,xs)",
    )
    .unwrap();
    let program = optimize_for_test(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let placement = Placement::new(&program).unwrap();
    let plan = Evaluation::new(&program, &placement).unwrap();
    let mut classes = LookupSet::default();
    let mut contexts = LookupSet::default();
    let mut targets = LookupSet::default();
    let mut repeated = 0;
    for (&(scope, term), &actual) in &plan.targets {
        let (_, fields) = program.stage.selected.app(term).unwrap();
        let context = program.stage.selected.values[fields[0]];
        let expected = placement
            .sites(context, scope, term)
            .into_iter()
            .min_by_key(|&(_, loops, depth)| (loops, std::cmp::Reverse(depth)))
            .map(|(site, _, _)| site);
        assert_eq!(Some(actual), expected);
        assert!(
            plan.total[term],
            "non-total expressions cannot share a hoisting decision"
        );
        contexts.insert(context);
        targets.insert(actual);
        repeated += usize::from(!classes.insert((context, scope, placement.required_scopes(term))));
    }
    assert!(repeated > 0, "exercise expressions sharing a placement class");
    assert!(contexts.len() > 1, "exercise separate entry contexts");
    assert!(targets.len() > 1, "exercise separate lexical placement targets");
}
