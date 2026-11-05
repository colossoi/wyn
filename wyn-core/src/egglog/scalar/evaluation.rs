//! Immutable evaluation facts and control-boundary schedules. Built once before
//! emitting any function; emission never queries effects or walks hidden source
//! regions to discover an earlier producer.
use super::extract::operands;
use super::placement::Placement;
use super::Selected;
use crate::egglog::facts::Facts;
use crate::egglog::query::Query;
use crate::egglog::{timing, OptimizeError, Optimized, Program};
use crate::{LookupMap, LookupSet};
use egglog_engine::{EGraph, Read, Term, TermId, Value, Write};
use wyn_graph::{dag_postorder, reachable_targets};

pub(in crate::egglog) struct Evaluation {
    dependencies: Vec<Vec<TermId>>,
    readonly: Vec<bool>,
    total: Vec<bool>,
    targets: LookupMap<(Value, TermId), Value>,
    owners: Vec<Option<Value>>,
    ordinals: Vec<Option<i64>>,
    before: LookupMap<(Value, Value), Vec<TermId>>,
    shared: LookupMap<(TermId, TermId), Vec<(TermId, bool)>>,
    schedules: LookupMap<Vec<TermId>, Vec<Vec<TermId>>>,
    common: LookupMap<(Vec<TermId>, Vec<TermId>), Vec<(TermId, bool)>>,
}
impl Evaluation {
    pub fn new(program: &Program<'_, Optimized>, placement: &Placement) -> Result<Self, OptimizeError> {
        let inputs = timing::span("egglog evaluation / collect facts");
        let selected = &program.stage.selected;
        let facts = Facts { program };
        let scalars = &program.stage.scalars;
        let n = selected.dag.size();
        let mut plan = Self {
            dependencies: vec![Vec::new(); n],
            readonly: vec![false; n],
            total: vec![false; n],
            targets: LookupMap::default(),
            owners: vec![None; n],
            ordinals: vec![None; n],
            before: LookupMap::default(),
            shared: LookupMap::default(),
            schedules: LookupMap::default(),
            common: LookupMap::default(),
        };
        let mut control_edges = vec![Vec::new(); n];
        let mut expressions = Vec::new();
        let mut leaves = LookupSet::default();
        let mut boundaries = Vec::new();
        let mut pairs = LookupSet::default();
        let mut choices = LookupMap::<TermId, (Vec<TermId>, Vec<TermId>)>::default();
        for term in 0..n {
            let Term::App(name, fields) = selected.dag.get(term) else {
                continue;
            };
            if !matches!(
                name.as_str(),
                "ExprLeaf"
                    | "ExprExecute"
                    | "ExprInstruction"
                    | "ExprCall"
                    | "ExprParameter"
                    | "ExprLength"
                    | "ExprLiteral"
                    | "ExprUnary"
                    | "ExprBinary"
                    | "ExprOp"
                    | "ExprInvoke"
                    | "ExprChoice"
                    | "ExprTuple"
                    | "ExprVector"
                    | "ExprProject"
                    | "ExprCoerce"
                    | "ExprNil"
                    | "ExprCons"
            ) {
                continue;
            }
            expressions.push(term);
            let value = selected.values[term];
            let table = if matches!(name.as_str(), "ExprCons" | "ExprNil") {
                "ExprArgsReadOnly"
            } else {
                "ExprReadOnly"
            };
            plan.readonly[term] =
                scalars.read(|r| r.lookup(table, value))?.is_some_and(|v| scalars.value_to_base::<bool>(v));
            plan.total[term] = scalars.read(|r| r.contains("ExprTotal", value))?;
            let mut deps = operands(name, fields);
            if name == "ExprLeaf" {
                let source = selected.values[fields[2]];
                leaves.insert(term);
                plan.owners[term] = facts.lookup("ExprContextOwner", (source,));
                plan.ordinals[term] =
                    facts.lookup("SourceEvaluationPosition", (source,)).map(|v| facts.integer(v));
                if let Some(&root) = selected.roots.get(&(selected.values[fields[0]], source)) {
                    if root != term {
                        deps = vec![root];
                    }
                }
            }
            if name == "ExprChoice" {
                deps = vec![fields[2]];
                let arms = choices.entry(fields[2]).or_default();
                arms.0.push(fields[3]);
                arms.1.push(fields[4]);
            }
            if name == "ExprInstruction" && matches!(selected.text(fields[3]), Ok("index")) {
                if let Ok(args) = selected.arguments(fields[4]) {
                    if let [array, index] = args.as_slice() {
                        if let Ok((mut base, _)) = selected.projected_array(*array) {
                            let mut indices = vec![*index];
                            while let Some((parent, index)) = selected.index_operands(base)? {
                                base = selected.projected_array(parent)?.0;
                                indices.push(index);
                            }
                            // The emitter follows this address path directly;
                            // eagerly scheduling intermediate reads would copy
                            // whole rows before it has a chance to do so.
                            deps = vec![base];
                            deps.extend(indices.into_iter().rev());
                        }
                    }
                }
            }
            plan.dependencies[term] = deps.clone();
            if name == "ExprChoice" {
                deps.extend_from_slice(&fields[3..5]);
            }
            if name == "ExprExecute" {
                let context = selected.values[fields[0]];
                let source = selected.values[fields[2]];
                let mut sources = Vec::new();
                if let Some((yes, no)) = facts.branches(source) {
                    sources.extend(facts.branch_condition(source));
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
                    sources.extend(
                        facts.record("SourceLoopInitial", "LoopInitialInfo", (header,)).map(|info| info[1]),
                    );
                    if let Some(form) = facts.lookup("SourceLoopForm", (source,)) {
                        for name in ["WhileCondition", "ForCount", "ForEach"] {
                            if let Some(fields) = facts.enode(name, form) {
                                sources.push(fields[0]);
                            }
                        }
                    }
                    if let Some(owner) = facts.lookup("ExprContextOwner", (source,)) {
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
        if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
            eprintln!(
                "egglog evaluation facts: {n} DAG nodes, {} roots, {} leaves, {} boundaries, {} branch pairs",
                selected.roots.len(), leaves.len(), boundaries.len(), pairs.len()
            );
        }
        drop(inputs);
        let containment = timing::span("egglog evaluation / binding reachability");
        let mut contained = reachable_targets(
            control_edges
                .iter()
                .enumerate()
                .flat_map(|(parent, children)| children.iter().map(move |&child| (parent, child))),
            leaves.iter().copied(),
        );
        let containment_roots: LookupSet<_> = boundaries
            .iter()
            .map(|&(_, _, _, term)| term)
            .chain(pairs.iter().flat_map(|&(a, b)| [a, b]))
            .collect();
        let enclosing: LookupMap<_, _> = containment_roots
            .iter()
            .map(|&root| {
                let mut leaves: Vec<_> = contained.remove(&root).unwrap_or_default().into_iter().collect();
                leaves.sort_unstable();
                (root, leaves)
            })
            .collect();
        drop(contained);
        drop(containment);
        let common_operands = timing::time("egglog evaluation / branch reachability", || {
            shared_mandatory_operands(&plan.dependencies, &pairs)
        });
        if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
            eprintln!(
                "egglog evaluation reachability: {} contained bindings, {} common operands",
                enclosing.values().map(Vec::len).sum::<usize>(),
                common_operands.len()
            );
        }
        let schema = timing::span("egglog evaluation / schema");
        let mut graph = EGraph::default();
        graph.parse_and_run_program(None, include_str!("evaluation.egg"))?;
        drop(schema);
        let import = timing::span("egglog evaluation / import facts");
        graph.update(|mut sink| {
            for &term in &expressions {
                if let Term::App(name, fields) = selected.dag.get(term) {
                    sink.set("ExprScheduleWeight", term as i64, i64::from(name == "ExprChoice"))?;
                    if name != "ExprLeaf" {
                        for &child in &plan.dependencies[term] {
                            sink.add("ExprScheduleOperand", (term as i64, child as i64))?;
                        }
                    }
                    if name == "ExprChoice" {
                        sink.set("ExprScheduleCondition", term as i64, fields[2] as i64)?;
                    }
                }
                if plan.readonly[term] {
                    sink.add("EvaluationReadOnly", (term as i64,))?;
                }
                if plan.total[term] {
                    sink.add("EvaluationTotal", (term as i64,))?;
                }
            }
            for &(yes, no, term) in &common_operands {
                sink.add("EvaluationCommonOperand", (yes as i64, no as i64, term as i64))?;
            }
            for &(a, b) in &pairs {
                sink.add("EvaluationPair", (a as i64, b as i64))?;
            }
            for (&root, leaves) in &enclosing {
                for &leaf in leaves {
                    sink.add("EvaluationContains", (root as i64, leaf as i64))?;
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        drop(import);
        let available = timing::span("egglog evaluation / available facts");
        graph.update(|mut sink| {
            for &(_, source, owner, term) in &boundaries {
                for &leaf in &enclosing[&term] {
                    if plan.owners[leaf].is_some_and(|o| placement.available(leaf, o, owner))
                        && selected.app(leaf).is_ok_and(|(_, f)| selected.values[f[2]] != source)
                    {
                        sink.add("EvaluationAvailable", (term as i64, leaf as i64))?;
                    }
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        drop(available);
        timing::time("egglog evaluation / placement rules", || {
            graph.parse_and_run_program(None, "(run-schedule (saturate evaluation-placement))")
        })?;
        let sites_time = timing::span("egglog evaluation / site facts");
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
        scalars.constructor_enodes("ExprRoot", |row| {
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
                if name == "ExprChoice" {
                    pending.extend(fields[3..5].iter().map(|t| (scope, *t)));
                }
            }
        }
        let mut queries: Vec<_> = queries.into_iter().collect();
        queries.sort();
        let query_count = queries.len();
        let mut classes = LookupMap::default();
        let mut groups = Vec::<Vec<(Value, TermId)>>::new();
        for (scope, term) in queries {
            if !plan.total[term] {
                continue;
            }
            let Term::App(_, fields) = selected.dag.get(term) else {
                continue;
            };
            // Equal context, scope, and requirements have the same legal placement sites.
            let key = (selected.values[fields[0]], scope, placement.required_scopes(term));
            let group = *classes.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[group].push((scope, term));
        }
        let mut sites = Vec::new();
        graph.update(|mut sink| {
            for (query, members) in groups.iter().enumerate() {
                let (scope, term) = members[0];
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
        drop(sites_time);
        if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
            eprintln!(
                "egglog evaluation sites: {query_count} queries, {} classes, {} candidates",
                groups.len(),
                sites.len()
            );
        }
        timing::time("egglog evaluation / target rules", || {
            graph.parse_and_run_program(None, "(run-schedule (saturate evaluation-frequency) (saturate evaluation-lifetime) (saturate evaluation-target))")
        })?;
        let readout = timing::span("egglog evaluation / read targets");
        Query(&graph).for_function("EvaluationTarget", |row, target| {
            let query = graph.value_to_base::<i64>(row[0]) as usize;
            let site = graph.value_to_base::<i64>(target) as usize;
            for &member in &groups[query] {
                plan.targets.insert(member, sites[site]);
            }
            Ok(())
        })?;
        for &(a, b) in &pairs {
            plan.plan_common(vec![a], vec![b]);
        }
        drop(readout);
        timing::time("egglog evaluation / schedule anchors", || {
            plan.schedule_anchors(selected, &mut graph)
        })?;
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
    fn schedule_anchors(&mut self, selected: &Selected, graph: &mut EGraph) -> Result<(), OptimizeError> {
        timing::time("egglog anchors / rank rules", || {
            graph.parse_and_run_program(None, "(run-schedule (saturate schedule-rank))")
        })?;
        // Ranks and choice conditions are expression properties, shared by every request.
        let mut ordering = vec![(0, None); selected.dag.size()];
        Query(graph).for_function("ExprScheduleChoiceDepth", |row, rank| {
            ordering[graph.value_to_base::<i64>(row[0]) as usize].0 = graph.value_to_base::<i64>(rank);
            Ok(())
        })?;
        Query(graph).for_function("ExprScheduleCondition", |row, condition| {
            ordering[graph.value_to_base::<i64>(row[0]) as usize].1 =
                Some(graph.value_to_base::<i64>(condition) as usize);
            Ok(())
        })?;
        let mut pending: Vec<_> = selected.roots.values().map(|t| vec![*t]).collect();
        pending.extend(selected.extents.values().map(|t| vec![*t]));
        pending.extend((0..selected.dag.size()).filter_map(|term| {
            matches!(selected.dag.get(term), Term::App(name, _) if name == "ExprChoice")
                .then_some(vec![term])
        }));
        pending.extend(self.shared.values().flatten().map(|(t, _)| vec![*t]));
        pending.extend(self.before.values().flatten().map(|t| vec![*t]));
        while !pending.is_empty() {
            let requests_time = timing::span("egglog anchors / import requests");
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
                        if !matches!(selected.dag.get(term), Term::App(name, _) if name == "ExprLeaf") {
                            out.extend(self.dependencies[term].iter().copied());
                        }
                    },
                );
                requests.push((roots, order));
            }
            if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
                let memberships: usize = requests.iter().map(|(_, order)| order.len()).sum();
                let terms: LookupSet<_> = requests.iter().flat_map(|(_, order)| order.iter()).collect();
                eprintln!(
                    "egglog anchor facts: {} requests, {memberships} memberships, {} distinct terms",
                    requests.len(),
                    terms.len()
                );
            }
            drop(requests_time);
            let _readout = timing::span("egglog anchors / read schedules");
            for (roots, mut order) in requests {
                let reorderable = roots.iter().all(|t| self.readonly[*t]);
                let group = |term: TermId| {
                    ordering[term].1.map(|condition| if reorderable { condition } else { term })
                };
                let mut groups = LookupMap::<_, Vec<_>>::default();
                for &term in &order {
                    if let Some(group) = group(term) {
                        groups.entry(group).or_default().push(term);
                    }
                }
                if reorderable {
                    order.sort_by_key(|t| ordering[*t].0);
                }
                let mut schedule = Vec::new();
                for term in order {
                    let (name, _) = selected.app(term)?;
                    if matches!(name, "ExprCons" | "ExprNil") {
                        continue;
                    }
                    let terms = if let Some(group) = group(term) {
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

fn shared_mandatory_operands(
    dependencies: &[Vec<TermId>],
    pairs: &LookupSet<(TermId, TermId)>,
) -> Vec<(TermId, TermId, TermId)> {
    let roots: LookupSet<_> = pairs.iter().flat_map(|&(yes, no)| [yes, no]).collect();
    let reached_by = reachable_targets(
        dependencies
            .iter()
            .enumerate()
            .flat_map(|(parent, children)| children.iter().map(move |&child| (child, parent))),
        roots,
    );
    let mut demands = LookupMap::<_, LookupSet<_>>::default();
    for (term, roots) in reached_by {
        for root in roots {
            demands.entry(root).or_default().insert(term);
        }
    }
    let mut common = Vec::new();
    for &(yes, no) in pairs {
        if let (Some(yes_terms), Some(no_terms)) = (demands.get(&yes), demands.get(&no)) {
            common.extend(yes_terms.intersection(no_terms).map(|&term| (yes, no, term)));
        }
    }
    common.sort_unstable();
    common
}

#[cfg(test)]
#[path = "evaluation_tests.rs"]
mod tests;
