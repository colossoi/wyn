//! Compare finite egglog extractions by the work in their shared DAGs.
use crate::egglog::OptimizeError;
use crate::LookupSet;
use egglog_engine::extract::{Cost, CostModel, Extractor};
use egglog_engine::{ArcSort, EGraph, Enode, Function, Term, TermDag, TermId, Value};

// Inlining trades one call for this much scalar work. Representation nodes,
// already computed inputs, and constants require no additional instructions.
const CALL_WORK: u64 = 64;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Estimate {
    calls: u64,
    work: u64,
}

impl Cost for Estimate {
    fn identity() -> Self {
        Self::default()
    }
    fn unit() -> Self {
        Self {
            work: 1,
            ..Self::default()
        }
    }
    fn combine(self, other: &Self) -> Self {
        Self {
            calls: self.calls.saturating_add(other.calls),
            work: self.work.saturating_add(other.work),
        }
    }
}

struct Model {
    prefer_inlining: bool,
}

impl CostModel<Estimate> for Model {
    fn fold(&self, _head: &str, children: &[Estimate], head: Estimate) -> Estimate {
        children.iter().fold(head, Cost::combine)
    }
    fn enode_cost(&self, _graph: &EGraph, function: &Function, _enode: &Enode<'_>) -> Estimate {
        Estimate {
            calls: u64::from(self.prefer_inlining && function.name() == "ScalarInvoke"),
            work: work(function.name()),
        }
    }
    fn base_value_cost(&self, _graph: &EGraph, _sort: &ArcSort, _value: Value) -> Estimate {
        Estimate::default()
    }
}

fn work(name: &str) -> u64 {
    match name {
        "ScalarInvoke" => CALL_WORK,
        "ScalarOp" | "ScalarChoice" | "ScalarTuple" | "ScalarVector" | "ScalarProject" | "ScalarCoerce" => {
            1
        }
        _ => 0,
    }
}

pub(super) struct Candidates {
    compact: Extractor<Estimate>,
    inlined: Extractor<Estimate>,
}

impl Candidates {
    pub fn new(graph: &EGraph) -> Result<Self, OptimizeError> {
        let Some(sort) = graph.get_sort_by_name("ScalarExpr") else {
            return Err(OptimizeError::Output("missing scalar expression sort".into()));
        };
        let extract = |prefer_inlining| {
            Extractor::compute_costs_from_rootsorts(
                Some(vec![sort.clone()]),
                graph,
                Model { prefer_inlining },
            )
        };
        Ok(Self {
            compact: extract(false),
            inlined: extract(true),
        })
    }

    pub fn select(&self, graph: &EGraph, dag: &mut TermDag, value: Value) -> Result<TermId, OptimizeError> {
        // Both seed models are additive, as egglog's extractor requires. The
        // second exposes sharing hidden behind calls in the compact candidate.
        // Reranking these finite DAGs is a heuristic, not globally optimal DAG
        // extraction, and does not introduce expressions or cross contexts.
        let mut best = None;
        for extractor in [&self.compact, &self.inlined] {
            let mut candidate = TermDag::default();
            let Some((_, term)) = extractor.extract_best(graph, &mut candidate, value) else {
                continue;
            };
            let cost = dag_work(&candidate, term);
            if best.as_ref().is_none_or(|(old, _)| cost < *old) {
                best = Some((cost, extractor));
            }
        }
        let Some((_, extractor)) = best else {
            return Err(OptimizeError::Extraction(
                "scalar root has no resolved finite extraction".into(),
            ));
        };
        let Some((_, term)) = extractor.extract_best(graph, dag, value) else {
            return Err(OptimizeError::Extraction(
                "selected scalar extraction disappeared".into(),
            ));
        };
        Ok(term)
    }
}

fn dag_work(dag: &TermDag, root: TermId) -> u64 {
    let mut seen = LookupSet::default();
    let mut pending = vec![root];
    let mut cost = 0u64;
    while let Some(term) = pending.pop() {
        if !seen.insert(term) {
            continue;
        }
        if let Term::App(name, fields) = dag.get(term) {
            cost = cost.saturating_add(work(name));
            pending.extend(operands(name, fields));
        }
    }
    cost
}

pub(super) fn operands(name: &str, fields: &[TermId]) -> Vec<TermId> {
    match name {
        "ScalarOp" | "ScalarInvoke" => vec![fields[3]],
        "ScalarTuple" | "ScalarVector" | "ScalarProject" | "ScalarCoerce" => vec![fields[2]],
        "ScalarChoice" => fields[2..5].to_vec(),
        "ScalarCons" => fields[1..3].to_vec(),
        _ => Vec::new(),
    }
}
