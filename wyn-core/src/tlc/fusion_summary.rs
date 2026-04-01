//! Function summaries for interprocedural SOAC fusion.
//!
//! For each def, compute a summary describing what the function does in terms
//! of SOAC operations. Summaries enable interprocedural fusion: instead of
//! inlining function bodies, we reason about their summaries.

use crate::SymbolId;
use std::collections::{HashMap, HashSet};

use super::{
    ArrayExpr, Def, Lambda, Program, ReduceProps, SoacOp, Term, TermKind, collect_var_refs,
    extract_lambda_params,
};

/// Summary of what a def computes.
#[derive(Debug, Clone)]
pub enum DefSummary {
    /// Body is essentially `map(lam, param[param_idx])`
    Map {
        lam: Lambda,
        param_idx: usize,
    },
    /// Body is essentially `reduce(op, ne, param[param_idx])`
    Reduce {
        op: Lambda,
        ne: Term,
        param_idx: usize,
        props: ReduceProps,
    },
    /// Body produces an array via map, but the input is not a parameter
    /// (e.g., comes from globals). The callee's params are stored so the
    /// caller can substitute them with call arguments when inlining.
    ProducesMap {
        lam: Lambda,
        inputs: Vec<super::ArrayExpr>,
        callee_params: Vec<(crate::SymbolId, polytype::Type<crate::ast::TypeName>)>,
    },
    /// Body is not a recognizable SOAC pattern
    Unknown,
}

/// Compute summaries for all defs in a program.
pub fn summarize_program(program: &Program) -> HashMap<SymbolId, DefSummary> {
    let mut summaries = HashMap::new();
    for def in &program.defs {
        summaries.insert(def.name, summarize_def(def));
    }
    summaries
}

/// Compute a summary for a single def.
pub fn summarize_def(def: &Def) -> DefSummary {
    // Extract function parameters by unwrapping nested lambdas
    let (params, inner_body) = extract_lambda_params(&def.body);

    if params.is_empty() {
        return DefSummary::Unknown; // constant, not a function
    }

    let param_syms: Vec<SymbolId> = params.iter().map(|(sym, _)| *sym).collect();

    extract_soac_summary(&inner_body, &param_syms, &params).unwrap_or(DefSummary::Unknown)
}

/// Check if a def body is essentially a single SOAC applied to one of its parameters.
/// Strips trivial let bindings to see through administrative noise.
fn extract_soac_summary(
    body: &Term,
    params: &[SymbolId],
    full_params: &[(SymbolId, polytype::Type<crate::ast::TypeName>)],
) -> Option<DefSummary> {
    match &body.kind {
        TermKind::Soac(soac) => extract_from_soac(soac, params, full_params),

        TermKind::Let { rhs, body, .. } => {
            if matches!(rhs.kind, TermKind::Soac(_)) {
                if matches!(body.kind, TermKind::Var(_)) {
                    if let TermKind::Soac(soac) = &rhs.kind {
                        return extract_from_soac(soac, params, full_params);
                    }
                }
            }
            extract_soac_summary(body, params, full_params)
        }

        // Reject anything with control flow
        TermKind::If { .. } | TermKind::Loop { .. } => None,

        _ => None,
    }
}

/// Extract a summary from a concrete SOAC operation.
fn extract_from_soac(
    soac: &SoacOp,
    params: &[SymbolId],
    full_params: &[(SymbolId, polytype::Type<crate::ast::TypeName>)],
) -> Option<DefSummary> {
    match soac {
        SoacOp::Map { lam, inputs } => {
            // Best case: single input is a parameter
            if inputs.len() == 1 {
                if let Some(param_idx) = input_param_index(&inputs[0], params) {
                    return Some(DefSummary::Map {
                        lam: lam.clone(),
                        param_idx,
                    });
                }
            }
            // Fallback: function produces a map result, but input isn't a param.
            // Still useful — caller can inline to expose the map.
            Some(DefSummary::ProducesMap {
                lam: lam.clone(),
                inputs: inputs.clone(),
                callee_params: full_params.to_vec(),
            })
        }
        SoacOp::Reduce { op, ne, input, props } => {
            let param_idx = input_param_index(input, params)?;
            Some(DefSummary::Reduce {
                op: op.clone(),
                ne: (**ne).clone(),
                param_idx,
                props: props.clone(),
            })
        }
        _ => None,
    }
}

/// If an ArrayExpr is a Ref to one of the function parameters, return its index.
fn input_param_index(input: &ArrayExpr, params: &[SymbolId]) -> Option<usize> {
    match input {
        ArrayExpr::Ref(t) => {
            if let TermKind::Var(sym) = &t.kind {
                params.iter().position(|p| p == sym)
            } else {
                None
            }
        }
        _ => None,
    }
}

// =============================================================================
// Call graph
// =============================================================================

/// Build a call graph: for each def, collect which other defs it references.
pub fn build_call_graph(program: &Program) -> HashMap<SymbolId, Vec<SymbolId>> {
    let def_names: HashSet<SymbolId> = program.defs.iter().map(|d| d.name).collect();

    let mut graph = HashMap::new();
    for def in &program.defs {
        let refs: Vec<SymbolId> = collect_var_refs(&def.body)
            .into_iter()
            .filter(|sym| def_names.contains(sym) && *sym != def.name)
            .collect();
        graph.insert(def.name, refs);
    }
    graph
}

/// Compute reverse topological order of the call graph.
/// Returns SCCs — each SCC is a Vec<SymbolId>. Multi-node SCCs are recursive.
/// Order: leaves (callees) first.
pub fn topo_order(graph: &HashMap<SymbolId, Vec<SymbolId>>) -> Vec<Vec<SymbolId>> {
    // Tarjan's SCC algorithm
    let mut state = TarjanState {
        index_counter: 0,
        stack: Vec::new(),
        on_stack: HashSet::new(),
        indices: HashMap::new(),
        lowlinks: HashMap::new(),
        result: Vec::new(),
    };

    for &node in graph.keys() {
        if !state.indices.contains_key(&node) {
            tarjan_strongconnect(node, graph, &mut state);
        }
    }

    state.result
}

struct TarjanState {
    index_counter: usize,
    stack: Vec<SymbolId>,
    on_stack: HashSet<SymbolId>,
    indices: HashMap<SymbolId, usize>,
    lowlinks: HashMap<SymbolId, usize>,
    result: Vec<Vec<SymbolId>>,
}

fn tarjan_strongconnect(v: SymbolId, graph: &HashMap<SymbolId, Vec<SymbolId>>, state: &mut TarjanState) {
    state.indices.insert(v, state.index_counter);
    state.lowlinks.insert(v, state.index_counter);
    state.index_counter += 1;
    state.stack.push(v);
    state.on_stack.insert(v);

    if let Some(neighbors) = graph.get(&v) {
        for &w in neighbors {
            if !state.indices.contains_key(&w) {
                tarjan_strongconnect(w, graph, state);
                let w_low = state.lowlinks[&w];
                let v_low = state.lowlinks[&v];
                state.lowlinks.insert(v, v_low.min(w_low));
            } else if state.on_stack.contains(&w) {
                let w_idx = state.indices[&w];
                let v_low = state.lowlinks[&v];
                state.lowlinks.insert(v, v_low.min(w_idx));
            }
        }
    }

    // If v is a root node, pop the SCC
    if state.lowlinks[&v] == state.indices[&v] {
        let mut scc = Vec::new();
        loop {
            let w = state.stack.pop().unwrap();
            state.on_stack.remove(&w);
            scc.push(w);
            if w == v {
                break;
            }
        }
        state.result.push(scc);
    }
}

// =============================================================================
// Summary propagation
// =============================================================================

/// Propagate summaries bottom-up through the call graph.
///
/// Processing in topological order (callees before callers):
/// - If a def calls a callee with a Map summary, and the call result is
///   consumed by a Map or Reduce, the caller's summary is refined.
/// - Recursive SCCs get Unknown.
pub fn propagate_summaries(program: &Program, summaries: &mut HashMap<SymbolId, DefSummary>) {
    let graph = build_call_graph(program);
    let order = topo_order(&graph);

    for scc in &order {
        // Recursive SCCs → all Unknown
        if scc.len() > 1 {
            for &sym in scc {
                summaries.insert(sym, DefSummary::Unknown);
            }
            continue;
        }

        let sym = scc[0];

        // Check for self-recursion
        if let Some(callees) = graph.get(&sym) {
            if callees.contains(&sym) {
                summaries.insert(sym, DefSummary::Unknown);
                continue;
            }
        }

        // If already summarized as Map or Reduce, keep it
        if !matches!(summaries.get(&sym), Some(DefSummary::Unknown) | None) {
            continue;
        }

        // Try to refine Unknown summaries by looking at callee summaries.
        // For now, this is a placeholder — full propagation requires
        // recognizing patterns like `map(g, foo(x))` where foo is Map(f).
        // That's Phase 5 territory (call-site rewriting).
    }
}
