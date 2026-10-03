//! Compare finite egglog extractions by the work in their shared DAGs.
use crate::egglog::{timing, OptimizeError, ScalarOptimization};
use crate::{LookupMap, LookupSet};
use egglog_engine::extract::{Cost, CostModel, Extractor};
use egglog_engine::sort::VecContainer;
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
            calls: u64::from(
                self.prefer_inlining && matches!(function.name(), "ScalarInvoke" | "ScalarCall"),
            ),
            work: work(function.name()),
        }
    }
    fn base_value_cost(&self, _graph: &EGraph, _sort: &ArcSort, _value: Value) -> Estimate {
        Estimate::default()
    }
}

fn work(name: &str) -> u64 {
    match name {
        "ScalarInvoke" | "ScalarCall" => CALL_WORK,
        "ScalarUnary" | "ScalarBinary" | "ScalarOp" | "ScalarChoice" | "ScalarTuple" | "ScalarVector"
        | "ScalarProject" | "ScalarCoerce" | "ScalarInstruction" => 1,
        _ => 0,
    }
}

/// Extract all demanded roots together so egglog reconstructs shared terms once.
/// Candidate DAGs are temporary; only selected reachable terms enter the output.
pub(super) fn select(
    graph: &mut EGraph,
    roots: &[Value],
    policy: ScalarOptimization,
) -> Result<(TermDag, Vec<TermId>), OptimizeError> {
    let Some(sort) = graph.get_sort_by_name("ScalarOutputs").cloned() else {
        return Err(OptimizeError::Output("missing scalar outputs sort".into()));
    };
    let value = graph.container_to_value(VecContainer {
        do_rebuild: true,
        data: roots.to_vec(),
    });
    let extract = |prefer_inlining| {
        let costs = timing::span(if prefer_inlining {
            "egglog scalar / inlined costs"
        } else {
            "egglog scalar / compact costs"
        });
        let extractor = Extractor::compute_costs_from_rootsorts(
            Some(vec![sort.clone()]),
            graph,
            Model { prefer_inlining },
        );
        drop(costs);
        let mut dag = TermDag::default();
        let Some((_, root)) = extractor.extract_best(graph, &mut dag, value) else {
            return Err(OptimizeError::Extraction(
                "scalar roots have no resolved finite extraction".into(),
            ));
        };
        let Term::App(_, children) = dag.get(root) else {
            return Err(OptimizeError::Extraction(
                "expected extracted output vector".into(),
            ));
        };
        let children = children.clone();
        Ok((dag, children))
    };
    // TODO: Move the compact-versus-inlined optimization policy into Egglog;
    // final extraction should consume that choice instead of reranking in Rust.
    // Both seed models are additive, as egglog requires. Reranking their finite
    // DAGs exposes sharing hidden behind calls without expanding alternatives.
    let (compact, compact_roots) = extract(false)?;
    if policy == ScalarOptimization::Basic {
        // Keep only reachable expression terms, excluding the synthetic output
        // vector. Placement resolves every retained term as an engine enode.
        let mut dag = TermDag::default();
        let mut copied = LookupMap::default();
        let roots = compact_roots
            .into_iter()
            .map(|root| copy_term(&compact, root, &mut dag, &mut copied))
            .collect();
        return Ok((dag, roots));
    }
    let (inlined, inlined_roots) = extract(true)?;
    let mut dag = TermDag::default();
    let mut compact_copies = LookupMap::default();
    let mut inlined_copies = LookupMap::default();
    let roots = compact_roots
        .into_iter()
        .zip(inlined_roots)
        .map(|(a, b)| {
            if dag_work(&inlined, b) < dag_work(&compact, a) {
                copy_term(&inlined, b, &mut dag, &mut inlined_copies)
            } else {
                copy_term(&compact, a, &mut dag, &mut compact_copies)
            }
        })
        .collect();
    Ok((dag, roots))
}

fn copy_term(
    source: &TermDag,
    term: TermId,
    target: &mut TermDag,
    copied: &mut LookupMap<TermId, TermId>,
) -> TermId {
    if let Some(&id) = copied.get(&term) {
        return id;
    }
    let id = match source.get(term) {
        Term::Lit(literal) => target.lit(literal.clone()),
        Term::Var(name) => target.var(name.clone()),
        Term::App(name, children) => {
            let children = children.iter().map(|&child| copy_term(source, child, target, copied)).collect();
            target.app(name.clone(), children)
        }
    };
    copied.insert(term, id);
    id
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

pub(in crate::egglog) fn operands(name: &str, fields: &[TermId]) -> Vec<TermId> {
    match name {
        "ScalarUnary" => vec![fields[3]],
        "ScalarBinary" => fields[3..5].to_vec(),
        "ScalarOp" | "ScalarInvoke" => vec![fields[3]],
        "ScalarInstruction" | "ScalarCall" => vec![fields[4]],
        "ScalarTuple" | "ScalarVector" | "ScalarProject" | "ScalarCoerce" => vec![fields[2]],
        "ScalarChoice" => fields[2..5].to_vec(),
        "ScalarCons" => fields[1..3].to_vec(),
        _ => Vec::new(),
    }
}
