//! Immutable evaluation facts and control-boundary schedules. Built once before
//! emitting any function; emission never queries effects or walks hidden source
//! regions to discover an earlier producer.
use super::Selected;
use crate::egglog::query::Query;
use crate::egglog::{timing, OptimizeError, Optimized, Program};
use crate::{LookupMap, LookupSet};
use egglog_engine::{EGraph, RawValues, Term, TermId, Value};
use wyn_graph::dag_postorder;

pub(super) fn prepare(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog evaluation / native projection");
    graph.parse_and_run_program(None, include_str!("evaluation.egg"))?;
    graph.parse_and_run_program(None, "(run-schedule (saturate selected-shape) (saturate selected-address) (saturate selected-flow) (saturate flow-scopes))")?;
    if Query(graph).contains("EvaluationScopeCycle", RawValues(Vec::new()))? {
        return Err(OptimizeError::Output("cyclic lexical scope forest".into()));
    }
    graph.parse_and_run_program(None, "(run-schedule (saturate source-summary) (saturate flow-evaluation) (saturate evaluation-placement) (saturate evaluation-candidates) (saturate evaluation-legality) (saturate evaluation-sites) (saturate evaluation-frequency) (saturate evaluation-lifetime) (saturate evaluation-target) (saturate schedule-rank))")?;
    Ok(())
}

pub(in crate::egglog) struct Evaluation {
    available: LookupSet<(TermId, Value)>,
    readonly: Vec<bool>,
    targets: LookupMap<(Value, TermId), Value>,
    ordinals: Vec<Option<i64>>,
    before: LookupMap<(Value, Value), Vec<TermId>>,
    shared: LookupMap<(TermId, TermId), Vec<(TermId, bool)>>,
    schedules: LookupMap<Vec<TermId>, Vec<Vec<TermId>>>,
    common: LookupMap<(Vec<TermId>, Vec<TermId>), Vec<(TermId, bool)>>,
}
impl Evaluation {
    pub fn new(program: &Program<'_, Optimized>) -> Result<Self, OptimizeError> {
        let _timing = timing::span("egglog evaluation / final readout");
        let selected = &program.stage.selected;
        let graph = &program.graph;
        let query = Query(graph);
        let term = |value| graph.value_to_base::<i64>(value) as usize;
        let mut plan = Self {
            available: LookupSet::default(),
            readonly: vec![false; selected.dag.size()],
            targets: LookupMap::default(),
            ordinals: vec![None; selected.dag.size()],
            before: LookupMap::default(),
            shared: LookupMap::default(),
            schedules: LookupMap::default(),
            common: LookupMap::default(),
        };
        query.for_each("EvaluationReadOnly", |row| {
            plan.readonly[term(row[0])] = true;
            Ok(())
        })?;
        query.for_function("EvaluationOrdinal", |row, ordinal| {
            plan.ordinals[term(row[0])] = Some(graph.value_to_base::<i64>(ordinal));
            Ok(())
        })?;
        query.for_each("EvaluationCommonAvailable", |row| {
            plan.available.insert((term(row[0]), row[1]));
            Ok(())
        })?;
        let mut schedules = LookupMap::<usize, Vec<usize>>::default();
        query.for_each("EvaluationBefore", |row| {
            schedules.entry(term(row[0])).or_default().push(term(row[1]));
            Ok(())
        })?;
        query.for_each("EvaluationBoundary", |row| {
            let mut schedule = schedules.remove(&term(row[3])).unwrap_or_default();
            schedule.sort_by_key(|t| (plan.ordinals[*t], *t));
            plan.before.insert((row[0], row[1]), schedule);
            Ok(())
        })?;
        for (table, binding) in [("EvaluationShared", false), ("EvaluationSharedBinding", true)] {
            query.for_each(table, |row| {
                plan.shared.entry((term(row[0]), term(row[1]))).or_default().push((term(row[2]), binding));
                Ok(())
            })?;
        }
        query.for_each("EvaluationPlacement", |row| {
            plan.targets.insert((row[0], term(row[1])), row[2]);
            Ok(())
        })?;
        plan.schedule_anchors(selected, graph)?;
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
    fn schedule_anchors(&mut self, selected: &Selected, graph: &EGraph) -> Result<(), OptimizeError> {
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
        let _timing = timing::span("egglog anchors / schedule requests");
        while let Some(roots) = pending.pop() {
            if self.schedules.contains_key(&roots) {
                continue;
            }
            let mut result = Ok(());
            let mut order = dag_postorder(
                roots.iter().copied(),
                |_| false,
                |term, out| {
                    if !matches!(selected.dag.get(term), Term::App(name, _) if name == "ExprLeaf") {
                        if result.is_ok() {
                            result = evaluation_operands(graph, term).map(|children| out.extend(children));
                        }
                    }
                },
            );
            result?;
            let reorderable = roots.iter().all(|t| self.readonly[*t]);
            let group =
                |term: TermId| ordering[term].1.map(|condition| if reorderable { condition } else { term });
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
    pub fn common(&self, scope: Value, arms: [&[TermId]; 2]) -> Vec<TermId> {
        self.common
            .get(&(arms[0].to_vec(), arms[1].to_vec()))
            .into_iter()
            .flatten()
            .filter(|(term, binding)| !binding || self.available.contains(&(*term, scope)))
            .map(|(term, _)| *term)
            .collect()
    }
}

fn evaluation_operands(graph: &EGraph, term: TermId) -> Result<Vec<TermId>, OptimizeError> {
    let query = Query(graph);
    let Some(parent) = query.constructor("FlowNode", term as i64)? else {
        return Err(OptimizeError::Output(format!("missing evaluation node {term}")));
    };
    let mut operands = Vec::new();
    while let Some(child) = query.record(
        "FlowEvaluationOperand",
        "FlowNode",
        (parent, operands.len() as i64),
    )? {
        operands.push(graph.value_to_base::<i64>(child[0]) as usize);
    }
    Ok(operands)
}

#[cfg(test)]
#[path = "evaluation_tests.rs"]
mod tests;
