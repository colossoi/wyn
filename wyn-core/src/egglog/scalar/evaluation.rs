//! Immutable evaluation facts and control-boundary schedules. Built once before
//! emitting any function; emission never queries effects or walks hidden source
//! regions to discover an earlier producer.
use super::extract::operands;
use super::placement::Placement;
use crate::egglog::facts::Facts;
use crate::egglog::query::Query;
use crate::egglog::{OptimizeError, Optimized, Program};
use crate::{LookupMap, LookupSet};
use egglog_engine::{EGraph, Read, Term, TermId, Value, Write};
use wyn_graph::dag_postorder;

pub(in crate::egglog) struct Evaluation {
    dependencies: Vec<Vec<TermId>>,
    readonly: Vec<bool>,
    total: Vec<bool>,
    targets: LookupMap<(Value, TermId), Value>,
    enclosing: Vec<Vec<TermId>>,
    owners: Vec<Option<Value>>,
    ordinals: Vec<Option<i64>>,
    before: LookupMap<(Value, Value), Vec<TermId>>,
    shared: LookupMap<(TermId, TermId), Vec<(TermId, bool)>>,
    schedules: LookupMap<Vec<TermId>, Vec<Vec<TermId>>>,
    common: LookupMap<(Vec<TermId>, Vec<TermId>), Vec<(TermId, bool)>>,
}
impl Evaluation {
    pub fn new(program: &Program<'_, Optimized>, placement: &Placement) -> Result<Self, OptimizeError> {
        let selected = &program.stage.selected;
        let facts = Facts { program };
        let scalars = &program.stage.scalars;
        let n = selected.dag.size();
        let mut plan = Self {
            dependencies: vec![Vec::new(); n],
            readonly: vec![false; n],
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
        let mut control_edges = vec![Vec::new(); n];
        let mut leaves = LookupSet::default();
        let mut boundaries = Vec::new();
        let mut pairs = LookupSet::default();
        let mut choices = LookupMap::<TermId, (Vec<TermId>, Vec<TermId>)>::default();
        for term in 0..n {
            let Term::App(name, fields) = selected.dag.get(term) else {
                continue;
            };
            if !name.starts_with("Scalar") {
                continue;
            }
            let value = selected.values[term];
            let table = if matches!(name.as_str(), "ScalarCons" | "ScalarNil") {
                "ScalarArgsReadOnly"
            } else {
                "ScalarReadOnly"
            };
            plan.readonly[term] =
                scalars.read(|r| r.lookup(table, value))?.is_some_and(|v| scalars.value_to_base::<bool>(v));
            plan.total[term] = scalars.read(|r| r.contains("ScalarTotal", value))?;
            let mut deps = operands(name, fields);
            if name == "ScalarLeaf" {
                let source = selected.values[fields[2]];
                leaves.insert(term);
                plan.owners[term] = facts.lookup("ScalarOwner", (source,));
                plan.ordinals[term] =
                    facts.lookup("SourceEvaluationPosition", (source,)).map(|v| facts.integer(v));
                if let Some(&root) = selected.roots.get(&(selected.values[fields[0]], source)) {
                    if root != term {
                        deps = vec![root];
                    }
                }
            }
            if name == "ScalarChoice" {
                deps = vec![fields[2]];
                let arms = choices.entry(fields[2]).or_default();
                arms.0.push(fields[3]);
                arms.1.push(fields[4]);
            }
            if name == "ScalarInstruction" && matches!(selected.text(fields[3]), Ok("index")) {
                if let Ok(args) = selected.arguments(fields[4]) {
                    if let [array, index] = args.as_slice() {
                        if let Ok((base, _)) = selected.projected_array(*array) {
                            deps = vec![base, *index];
                        }
                    }
                }
            }
            plan.dependencies[term] = deps.clone();
            if name == "ScalarChoice" {
                deps.extend_from_slice(&fields[3..5]);
            }
            if name == "ScalarExecute" {
                let context = selected.values[fields[0]];
                let source = selected.values[fields[2]];
                let mut sources = Vec::new();
                if let Some((yes, no)) = facts.branches(source) {
                    sources.extend(facts.lookup("SsaBranchCondition", (source,)));
                    sources.extend(facts.result(yes));
                    sources.extend(facts.result(no));
                    if let (Some(a), Some(b)) = (facts.result(yes), facts.result(no)) {
                        if let (Some(&a), Some(&b)) = (
                            selected.roots.get(&(context, a)),
                            selected.roots.get(&(context, b)),
                        ) {
                            pairs.insert((a, b));
                        }
                    }
                }
                if let Some((header, iteration)) = facts.loops(source) {
                    sources.extend(facts.result(iteration));
                    sources.extend(facts.lookup("SsaLoopInitial", (header,)));
                    if let Some(form) = facts.lookup("SourceLoopForm", (source,)) {
                        for name in ["WhileCondition", "ForCount", "ForEach"] {
                            if let Some(fields) = facts.enode(name, form) {
                                sources.push(fields[0]);
                            }
                        }
                    }
                    if let Some(owner) = facts.lookup("ScalarOwner", (source,)) {
                        boundaries.push((context, source, owner, term));
                    }
                }
                deps.extend(sources.into_iter().filter_map(|s| selected.roots.get(&(context, s)).copied()));
            }
            control_edges[term] = deps;
        }
        for (_, (yes, no)) in choices {
            for &a in &yes {
                for &b in &no {
                    pairs.insert((a, b));
                }
            }
        }
        let mut graph = EGraph::default();
        graph.parse_and_run_program(None, include_str!("evaluation.egg"))?;
        graph.update(|mut sink| {
            for (term, children) in control_edges.iter().enumerate() {
                for &child in children {
                    sink.add("EvaluationEdge", (term as i64, child as i64))?;
                }
                if let Term::App(name, fields) = selected.dag.get(term) {
                    sink.add("ScheduleNode", (term as i64, i64::from(name == "ScalarChoice")))?;
                    if name != "ScalarLeaf" {
                        for &child in &plan.dependencies[term] {
                            sink.add("ScheduleOperand", (term as i64, child as i64))?;
                        }
                    }
                    if name == "ScalarChoice" {
                        sink.add("ScheduleChoice", (term as i64, fields[2] as i64))?;
                    }
                }
                if plan.readonly[term] {
                    sink.add("EvaluationReadOnly", (term as i64,))?;
                }
                if plan.total[term] {
                    sink.add("EvaluationTotal", (term as i64,))?;
                }
            }
            for (term, children) in plan.dependencies.iter().enumerate() {
                for &child in children {
                    sink.add("EvaluationOperand", (term as i64, child as i64))?;
                }
            }
            for &(a, b) in &pairs {
                sink.add("EvaluationPair", (a as i64, b as i64))?;
            }
            for &leaf in &leaves {
                sink.add("EvaluationLeaf", (leaf as i64,))?;
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate evaluation-reachability))")?;
        Query(&graph).for_each("EvaluationContains", |row| {
            let root = graph.value_to_base::<i64>(row[0]) as usize;
            let leaf = graph.value_to_base::<i64>(row[1]) as usize;
            plan.enclosing[root].push(leaf);
            Ok(())
        })?;
        graph.update(|mut sink| {
            for &(_, source, owner, term) in &boundaries {
                for &leaf in &plan.enclosing[term] {
                    if plan.owners[leaf].is_some_and(|o| placement.available(leaf, o, owner))
                        && selected.app(leaf).is_ok_and(|(_, f)| selected.values[f[2]] != source)
                    {
                        sink.add("EvaluationAvailable", (term as i64, leaf as i64))?;
                    }
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate evaluation-placement))")?;
        let mut schedules = LookupMap::<usize, Vec<usize>>::default();
        Query(&graph).for_each("EvaluationBefore", |row| {
            let root = graph.value_to_base::<i64>(row[0]) as usize;
            let leaf = graph.value_to_base::<i64>(row[1]) as usize;
            schedules.entry(root).or_default().push(leaf);
            Ok(())
        })?;
        for (context, source, _, term) in boundaries {
            let mut schedule = schedules.remove(&term).unwrap_or_default();
            schedule.sort_by_key(|t| (plan.ordinals[*t], *t));
            plan.before.insert((context, source), schedule);
        }
        for (table, binding) in [("EvaluationShared", false), ("EvaluationSharedBinding", true)] {
            Query(&graph).for_each(table, |row| {
                let a = graph.value_to_base::<i64>(row[0]) as usize;
                let b = graph.value_to_base::<i64>(row[1]) as usize;
                let term = graph.value_to_base::<i64>(row[2]) as usize;
                plan.shared.entry((a, b)).or_default().push((term, binding));
                Ok(())
            })?;
        }
        plan.total.fill(false);
        Query(&graph).for_each("EvaluationHoistable", |row| {
            plan.total[graph.value_to_base::<i64>(row[0]) as usize] = true;
            Ok(())
        })?;
        let mut queries = LookupSet::default();
        let mut pending = Vec::new();
        scalars.constructor_enodes("ScalarRoot", |row| {
            if let Some(&term) = selected.roots.get(&(row.children[0], row.children[2])) {
                pending.push((row.children[1], term));
            }
        })?;
        while let Some((scope, term)) = pending.pop() {
            if !queries.insert((scope, term)) {
                continue;
            }
            pending.extend(plan.dependencies[term].iter().map(|t| (scope, *t)));
            if let Term::App(name, fields) = selected.dag.get(term) {
                if name == "ScalarChoice" {
                    pending.extend(fields[3..5].iter().map(|t| (scope, *t)));
                }
            }
        }
        let mut queries: Vec<_> = queries.into_iter().collect();
        queries.sort();
        let mut sites = Vec::new();
        graph.update(|mut sink| {
            for (query, &(scope, term)) in queries.iter().enumerate() {
                if !plan.total[term] {
                    continue;
                }
                let Term::App(_, fields) = selected.dag.get(term) else {
                    continue;
                };
                let context = selected.values[fields[0]];
                for (site, loops, depth) in placement.sites(context, scope, term) {
                    let index = sites.len();
                    sites.push(site);
                    sink.add(
                        "EvaluationSite",
                        (
                            query as i64,
                            term as i64,
                            index as i64,
                            loops as i64,
                            depth as i64,
                        ),
                    )?;
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate evaluation-frequency) (saturate evaluation-lifetime) (saturate evaluation-target))")?;
        Query(&graph).for_function("EvaluationTarget", |row, target| {
            let query = graph.value_to_base::<i64>(row[0]) as usize;
            let site = graph.value_to_base::<i64>(target) as usize;
            plan.targets.insert(queries[query], sites[site]);
            Ok(())
        })?;
        for &(a, b) in &pairs {
            plan.plan_common(vec![a], vec![b]);
        }
        plan.schedule_anchors(program, &mut graph)?;
        Ok(plan)
    }
    pub fn schedule(&self, roots: &[TermId]) -> Result<&[Vec<TermId>], OptimizeError> {
        let Some(schedule) = self.schedules.get(roots) else {
            return Err(OptimizeError::Output(format!(
                "missing evaluation anchor for {roots:?}"
            )));
        };
        Ok(schedule)
    }
    fn schedule_anchors(
        &mut self,
        program: &Program<'_, Optimized>,
        graph: &mut EGraph,
    ) -> Result<(), OptimizeError> {
        let selected = &program.stage.selected;
        graph.parse_and_run_program(None, "(run-schedule (saturate schedule-rank))")?;
        let mut pending: Vec<_> = selected.roots.values().map(|t| vec![*t]).collect();
        pending.extend((0..selected.dag.size()).filter_map(|term| {
            matches!(selected.dag.get(term), Term::App(name, _) if name == "ScalarChoice")
                .then_some(vec![term])
        }));
        pending.extend(self.shared.values().flatten().map(|(t, _)| vec![*t]));
        pending.extend(self.before.values().flatten().map(|t| vec![*t]));
        let mut next = 0i64;
        while !pending.is_empty() {
            let mut requests = Vec::new();
            for roots in std::mem::take(&mut pending) {
                if self.schedules.contains_key(&roots) {
                    continue;
                }
                self.schedules.insert(roots.clone(), Vec::new());
                let order = dag_postorder(
                    roots.iter().copied(),
                    |_| false,
                    |term, out| {
                        if !matches!(selected.dag.get(term), Term::App(name, _) if name == "ScalarLeaf") {
                            out.extend(self.dependencies[term].iter().copied());
                        }
                    },
                );
                requests.push((next, roots, order));
                next += 1;
            }
            graph.update(|mut sink| {
                for (id, roots, order) in &requests {
                    sink.add("ScheduleRequest", (*id, roots.iter().all(|t| self.readonly[*t])))?;
                    for &term in order {
                        sink.add("ScheduleMember", (*id, term as i64))?;
                    }
                }
                Ok::<_, egglog_engine::Error>(())
            })?;
            graph.parse_and_run_program(None, "(run-schedule (saturate schedule-anchors))")?;
            for (id, roots, mut order) in requests {
                let mut ranks = LookupMap::default();
                let mut groups = LookupMap::<_, Vec<_>>::default();
                for &term in &order {
                    let rank = Query(graph).required("ScheduleOrder", (id, term as i64))?;
                    ranks.insert(term, graph.value_to_base::<i64>(rank));
                    if let Some(group) = Query(graph).lookup("ScheduleGroup", (id, term as i64))? {
                        groups.entry(group).or_default().push(term);
                    }
                }
                order.sort_by_key(|t| ranks[t]);
                let mut schedule = Vec::new();
                for term in order {
                    let (name, _) = selected.app(term)?;
                    if matches!(name, "ScalarCons" | "ScalarNil") {
                        continue;
                    }
                    let terms =
                        if let Some(group) = Query(graph).lookup("ScheduleGroup", (id, term as i64))? {
                            let Some(terms) = groups.remove(&group) else {
                                continue;
                            };
                            let arms = [3, 4].map(|arm| {
                                terms
                                    .iter()
                                    .map(|t| selected.app(*t).map(|(_, f)| f[arm]))
                                    .collect::<Result<Vec<_>, _>>()
                            });
                            let [yes, no] = arms;
                            let (yes, no) = (yes?, no?);
                            self.plan_common(yes.clone(), no.clone());
                            pending.extend([yes, no]);
                            terms
                        } else {
                            vec![term]
                        };
                    schedule.push(terms);
                }
                self.schedules.insert(roots, schedule);
            }
        }
        Ok(())
    }
    pub fn target(&self, scope: Value, term: TermId) -> Option<Value> {
        self.targets.get(&(scope, term)).copied()
    }
    pub fn before(&self, context: Value, source: Value) -> &[TermId] {
        self.before.get(&(context, source)).map_or(&[], Vec::as_slice)
    }
    fn plan_common(&mut self, yes: Vec<TermId>, no: Vec<TermId>) {
        let mut terms = LookupMap::<TermId, bool>::default();
        for &a in &yes {
            for &b in &no {
                if let Some(shared) = self.shared.get(&(a, b)) {
                    for &(term, binding) in shared {
                        terms.entry(term).and_modify(|b| *b &= binding).or_insert(binding);
                    }
                }
            }
        }
        let mut terms: Vec<_> = terms.into_iter().collect();
        terms.sort_by_key(|(t, _)| (self.ordinals[*t], *t));
        self.common.insert((yes, no), terms);
    }
    pub fn common(&self, placement: &Placement, scope: Value, arms: [&[TermId]; 2]) -> Vec<TermId> {
        self.common
            .get(&(arms[0].to_vec(), arms[1].to_vec()))
            .into_iter()
            .flatten()
            .filter(|(term, binding)| {
                !binding || self.owners[*term].is_some_and(|owner| placement.available(*term, owner, scope))
            })
            .map(|(term, _)| *term)
            .collect()
    }
}
