use super::super::extract::operands;
use super::*;
use crate::egglog::{from_tlc, fuse, optimize_for_test, place, schedule};
use crate::PipelineTopologyPolicy;
use egglog_engine::{Core, Read, TermDag, Write};

#[test]
fn shared_mandatory_operands_match_reference_closure_and_preserve_readonly_guards() {
    let dependencies = [
        vec![],
        vec![0],
        vec![0],
        vec![1, 2],
        vec![5],
        vec![4, 6],
        vec![],
        vec![3, 4],
        vec![7],
        vec![3, 6],
        vec![],
        vec![10],
    ];
    let pairs: LookupSet<_> = [(1, 2), (3, 7), (4, 5), (8, 9), (9, 3), (10, 11)].into_iter().collect();
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            concat!(
                include_str!("../flow.egg"),
                "\n(sort ExprContext)\n",
                include_str!("evaluation.egg")
            ),
        )
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
            (relation ReferenceOperand (i64 i64))
            (relation ReferenceDemand (i64 i64))
            (relation ReferenceShared (i64 i64 i64))
            (ruleset reference-reachability)
            (rule ((EvaluationPair yes no))
              ((ReferenceDemand yes yes) (ReferenceDemand no no)) :ruleset reference-reachability)
            (rule ((ReferenceDemand root parent) (ReferenceOperand parent child))
              ((ReferenceDemand root child)) :ruleset reference-reachability)
            (rule ((EvaluationPair yes no) (EvaluationReadOnly yes) (EvaluationReadOnly no)
                   (ReferenceDemand yes term) (ReferenceDemand no term))
              ((ReferenceShared yes no term)) :ruleset reference-reachability)
            "#,
        )
        .unwrap();
    graph
        .update(|mut sink| {
            for (parent, children) in dependencies.iter().enumerate() {
                for &child in children {
                    sink.add("ReferenceOperand", (parent as i64, child as i64))?;
                    let from = sink.add("FlowNode", parent as i64)?;
                    let to = sink.add("FlowNode", child as i64)?;
                    sink.add("FlowEvaluation", (from, to))?;
                }
                if parent != 9 {
                    sink.add("EvaluationReadOnly", parent as i64)?;
                }
            }
            for &(yes, no) in &pairs {
                sink.add("EvaluationPair", (yes as i64, no as i64))?;
            }
            Ok(())
        })
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            "(run-schedule (saturate reference-reachability) (saturate flow-evaluation) (saturate evaluation-placement))",
        )
        .unwrap();
    let rows = |table| {
        let mut rows = LookupSet::default();
        graph
            .constructor_enodes(table, |row| {
                rows.insert((
                    graph.value_to_base::<i64>(row.children[0]),
                    graph.value_to_base::<i64>(row.children[1]),
                    graph.value_to_base::<i64>(row.children[2]),
                ));
            })
            .unwrap();
        rows
    };
    let actual = rows("EvaluationShared");
    assert_eq!(actual, rows("ReferenceShared"));
    assert!(actual.contains(&(1, 2, 0)));
    assert!(
        actual.contains(&(4, 5, 6)),
        "cyclic dependencies retain their reachable operand"
    );
    assert!(actual.iter().all(|&(yes, no, _)| yes != 9 && no != 9));
}

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
        available: LookupSet::default(),
        readonly: vec![true; n],
        targets: LookupMap::default(),
        ordinals: vec![None; n],
        before: LookupMap::default(),
        shared: LookupMap::default(),
        schedules: LookupMap::default(),
        common: LookupMap::default(),
    };
    plan.readonly[effectful] = false;
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            concat!(
                include_str!("../flow.egg"),
                "\n(sort ExprContext)\n",
                include_str!("evaluation.egg")
            ),
        )
        .unwrap();
    let keys = graph
        .update(|mut sink| {
            for term in 0..n {
                let Term::App(name, fields) = dag.get(term) else {
                    unreachable!()
                };
                let deps = if name == "ExprChoice" { vec![fields[2]] } else { operands(name, fields) };
                let parent = sink.add("FlowNode", term as i64)?;
                for (index, &child) in deps.iter().enumerate() {
                    let node = sink.add("FlowNode", child as i64)?;
                    sink.set("FlowEvaluationOperand", (parent, index as i64), node)?;
                    sink.add("ExprScheduleOperand", (term as i64, child as i64))?;
                }
                sink.set("ExprScheduleWeight", term as i64, i64::from(name == "ExprChoice"))?;
                if name == "ExprChoice" {
                    sink.set("ExprScheduleCondition", term as i64, fields[2] as i64)?;
                }
            }
            Ok::<_, egglog_engine::Error>([sink.base_to_value(0i64), sink.base_to_value(1i64)])
        })
        .unwrap();
    // Producers behind leaves have separate anchors, even when their ranks are shared.
    graph
        .update(|mut sink| {
            let parent = sink.add("FlowNode", leaf as i64)?;
            let child = sink.add("FlowNode", nested as i64)?;
            sink.set("FlowEvaluationOperand", (parent, 0i64), child)?;
            Ok(())
        })
        .unwrap();
    let selected = Selected {
        dag,
        values: Vec::new(),
        roots: LookupMap::default(),
        extents: [(keys[0], pure), (keys[1], effectful)].into_iter().collect(),
    };
    graph.parse_and_run_program(None, "(run-schedule (saturate schedule-rank))").unwrap();
    plan.schedule_anchors(&selected, &graph).unwrap();
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
    let plan = Evaluation::new(&program).unwrap();
    let facts = Query(&program.graph);
    let selected = &program.stage.selected;
    let parent = |scope| {
        (!facts.flag("ExprContextBoundary", scope).unwrap())
            .then(|| program.identities.scopes.get(&scope).and_then(|s| s.0))
            .flatten()
    };
    let ancestors = |mut scope| {
        let mut result = vec![scope];
        while let Some(p) = parent(scope) {
            result.push(p);
            scope = p;
        }
        result
    };
    let mut members = LookupMap::<_, LookupSet<_>>::default();
    program
        .stage
        .scalars
        .constructor_enodes("ExprContextMember", |row| {
            members.entry(row.children[0]).or_default().insert(row.children[1]);
        })
        .unwrap();
    let mut loops = LookupSet::default();
    facts
        .for_function("SourceLoop", |_, info| {
            loops.insert(facts.enode("LoopInfo", info)?.unwrap()[0]);
            Ok(())
        })
        .unwrap();
    let mut contexts = LookupSet::default();
    let mut targets = LookupSet::default();
    for (&(scope, term), &actual) in &plan.targets {
        let (_, fields) = program.stage.selected.app(term).unwrap();
        let context = program.stage.selected.values[fields[0]];
        // Independently walk the expression operands and source scope chain.
        let mut pending = vec![term];
        let mut required = LookupSet::default();
        while let Some(term) = pending.pop() {
            let Term::App(name, fields) = selected.dag.get(term) else {
                continue;
            };
            let owner = match name.as_str() {
                "ExprLeaf" | "ExprExecute" | "ExprInstruction" | "ExprCall" => {
                    facts.lookup("ExprContextOwner", selected.values[fields[2]]).unwrap()
                }
                "ExprParameter" => Some(selected.values[fields[2]]),
                _ => None,
            };
            required.extend(owner);
            pending.extend(operands(name, fields));
        }
        let candidates: Vec<_> =
            ancestors(scope).into_iter().take_while(|s| members[&context].contains(s)).collect();
        let expected = candidates
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, site)| required.iter().all(|r| ancestors(*site).contains(r)))
            .min_by_key(|(distance, site)| {
                (
                    ancestors(*site).iter().filter(|s| loops.contains(s)).count(),
                    *distance,
                )
            })
            .map(|(_, site)| site);
        assert_eq!(Some(actual), expected);
        assert!(
            Query(&program.stage.scalars).contains("ExprTotal", selected.values[term]).unwrap(),
            "non-total expressions cannot share a hoisting decision"
        );
        contexts.insert(context);
        targets.insert(actual);
    }
    assert!(contexts.len() > 1, "exercise separate entry contexts");
    assert!(targets.len() > 1, "exercise separate lexical placement targets");
}

#[test]
fn unselected_expression_alternatives_do_not_add_evaluation_dependencies() {
    let tlc = crate::compile_thru_tlc("entry main(x:i32) i32=x").unwrap();
    let mut program = optimize_for_test(
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
    let selected = &program.stage.selected;
    let term = (0..selected.dag.size()).find(|&term| {
        matches!(selected.dag.get(term), Term::App(name, _) if matches!(name.as_str(), "ExprParameter" | "ExprLeaf"))
    }).unwrap();
    let (_, fields) = selected.app(term).unwrap();
    let expression = selected.dag.to_string(term);
    let context = selected.dag.to_string(fields[0]);
    let ty = selected.dag.to_string(fields[1]);
    let graph = &mut program.stage.scalars;
    // Introduce a valid alternative after selection. Its self-reference would
    // create a dependency cycle if projection followed all equivalent enodes.
    graph
        .parse_and_run_program(
            None,
            &format!(
        "(union {expression} (ExprBinary {context} {ty} \"+\" {expression} (ExprLiteral {context} {ty} 0)))
         (run-schedule (saturate selected-shape) (saturate selected-address) (saturate selected-flow))"
    ),
        )
        .unwrap();
    let query = Query(graph);
    let value = query.required("SelectedExpr", term as i64).unwrap();
    let mut has_binary = false;
    graph.read(|r| r.enodes_for_eclass("ExprBinary", value, |_| has_binary = true)).unwrap();
    assert!(has_binary, "the alternative must remain present in the e-class");
    let node = query.constructor("FlowNode", term as i64).unwrap().unwrap();
    assert!(evaluation_operands(graph, term).unwrap().is_empty());
    query
        .for_each("FlowOperand", |row| {
            assert_ne!(row[0], node, "an unselected alternative leaked an operand");
            Ok(())
        })
        .unwrap();
}
