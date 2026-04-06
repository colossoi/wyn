//! Producer graph: the fusion substrate.
//!
//! Builds a directed graph of array producers and consumers within a function
//! body. Nodes are array-producing operations (SOACs, literals, ranges, etc.),
//! edges are def-use links (producer's output feeds consumer's input).
//!
//! The graph is built by walking the Let chain in a function body. Each
//! `let name = SOAC(...) in body` creates a node, and references to `name`
//! in downstream SOAC inputs create edges.

use std::collections::HashMap;

use super::array_semantics::{ArraySemantics, FunctionSummary, ResultSemantics, classify_term};
use super::fusion::substitute_sym;
use super::{Lambda, Term, TermIdSource, TermKind};
use crate::SymbolId;
use crate::ast::TypeName;
use polytype::Type;

// =============================================================================
// Graph types
// =============================================================================

/// Unique identifier for a node in the producer graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProducerId(pub u32);

/// A node in the producer graph — one array-producing operation.
#[derive(Debug, Clone)]
pub struct ProducerNode {
    /// What this operation does.
    pub semantics: ArraySemantics,
    /// The let-bound name for this node's output (if any).
    pub binding: Option<SymbolId>,
    /// Result type.
    pub ty: Type<TypeName>,
    /// Number of uses of this node's output in the function body.
    pub use_count: usize,
}

/// An edge in the producer graph: producer output feeds consumer input.
#[derive(Debug, Clone)]
pub struct ProducerEdge {
    /// The producer node.
    pub producer: ProducerId,
    /// The consumer node.
    pub consumer: ProducerId,
    /// Which input of the consumer this edge feeds (index into the consumer's inputs).
    pub input_index: usize,
}

/// The producer graph for a single function body.
#[derive(Debug)]
pub struct ProducerGraph {
    nodes: Vec<ProducerNode>,
    edges: Vec<ProducerEdge>,
    /// Map from let-bound SymbolId to the ProducerId that produces it.
    binding_map: HashMap<SymbolId, ProducerId>,
}

impl ProducerGraph {
    pub fn nodes(&self) -> &[ProducerNode] {
        &self.nodes
    }

    pub fn edges(&self) -> &[ProducerEdge] {
        &self.edges
    }

    pub fn node(&self, id: ProducerId) -> &ProducerNode {
        &self.nodes[id.0 as usize]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the ProducerId for a let-bound symbol, if it's a producer in this graph.
    pub fn producer_for(&self, sym: SymbolId) -> Option<ProducerId> {
        self.binding_map.get(&sym).copied()
    }

    /// Get all consumers of a given producer.
    pub fn consumers_of(&self, producer: ProducerId) -> Vec<&ProducerEdge> {
        self.edges.iter().filter(|e| e.producer == producer).collect()
    }

    /// Get all producers feeding into a given consumer.
    pub fn producers_of(&self, consumer: ProducerId) -> Vec<&ProducerEdge> {
        self.edges.iter().filter(|e| e.consumer == consumer).collect()
    }
}

// =============================================================================
// Graph construction
// =============================================================================

/// Build a ProducerGraph from a function body.
///
/// Walks the Let chain, creating nodes for SOAC/array-producing operations
/// and edges for def-use relationships between them.
pub fn build_producer_graph(
    body: &Term,
    params: &[SymbolId],
    summaries: &HashMap<SymbolId, FunctionSummary>,
) -> ProducerGraph {
    let mut builder = GraphBuilder {
        nodes: Vec::new(),
        edges: Vec::new(),
        binding_map: HashMap::new(),
        param_set: params.iter().copied().collect(),
        summaries,
        term_ids: TermIdSource::new(),
    };

    builder.walk_term(body);
    builder.compute_use_counts();

    ProducerGraph {
        nodes: builder.nodes,
        edges: builder.edges,
        binding_map: builder.binding_map,
    }
}

struct GraphBuilder<'a> {
    nodes: Vec<ProducerNode>,
    edges: Vec<ProducerEdge>,
    binding_map: HashMap<SymbolId, ProducerId>,
    param_set: std::collections::HashSet<SymbolId>,
    summaries: &'a HashMap<SymbolId, FunctionSummary>,
    term_ids: TermIdSource,
}

impl<'a> GraphBuilder<'a> {
    fn add_node(&mut self, node: ProducerNode) -> ProducerId {
        let id = ProducerId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Classify a term as array semantics, using function summaries for App nodes.
    fn classify(&mut self, term: &Term) -> ArraySemantics {
        // First try direct classification (SOACs, array exprs)
        let direct = classify_term(term);
        if !matches!(direct, ArraySemantics::Opaque) {
            return direct;
        }

        // For App nodes, check if the callee has a known summary
        if let TermKind::App { func, args } = &term.kind {
            if let TermKind::Var(callee_sym) = &func.kind {
                if let Some(summary) = self.summaries.get(callee_sym) {
                    if let ResultSemantics::Produces(ref semantics) = summary.result {
                        // Substitute the summary's parameter references with
                        // the actual call arguments
                        return self.substitute_summary_args(semantics, &summary.params, args);
                    }
                }
            }
        }

        ArraySemantics::Opaque
    }

    /// Substitute parameter references in a summary's ArraySemantics with
    /// the actual call arguments.
    fn substitute_summary_args(
        &mut self,
        semantics: &ArraySemantics,
        summary_params: &[(SymbolId, Type<TypeName>)],
        call_args: &[Term],
    ) -> ArraySemantics {
        // Build a param_sym → arg_sym mapping
        let param_to_arg: HashMap<SymbolId, SymbolId> = summary_params
            .iter()
            .zip(call_args)
            .filter_map(|((param_sym, _), arg)| {
                if let TermKind::Var(arg_sym) = &arg.kind { Some((*param_sym, *arg_sym)) } else { None }
            })
            .collect();

        // Apply all param→arg substitutions to every ArrayExpr and Lambda in the semantics
        let mut result = semantics.clone();
        for (&param_sym, &arg_sym) in &param_to_arg {
            result = subst_in_semantics(result, param_sym, arg_sym, &mut self.term_ids);
        }
        result
    }

    /// Walk a term, extracting producer nodes from Let bindings.
    fn walk_term(&mut self, term: &Term) {
        match &term.kind {
            TermKind::Let { name, rhs, body, .. } => {
                let semantics = self.classify(rhs);

                if !matches!(semantics, ArraySemantics::Opaque) {
                    let id = self.add_node(ProducerNode {
                        semantics,
                        binding: Some(*name),
                        ty: rhs.ty.clone(),
                        use_count: 0,
                    });
                    self.wire_edges(id);
                    self.binding_map.insert(*name, id);
                }

                self.walk_term(body);
            }

            // Tail expression — might be a SOAC or interprocedural call
            TermKind::Soac(_) | TermKind::ArrayExpr(_) | TermKind::App { .. } => {
                let semantics = self.classify(term);
                if !matches!(semantics, ArraySemantics::Opaque) {
                    let id = self.add_node(ProducerNode {
                        semantics,
                        binding: None,
                        ty: term.ty.clone(),
                        use_count: 0,
                    });
                    self.wire_edges(id);
                }
            }

            _ => {}
        }
    }

    /// Wire edges from a consumer node's inputs to their producer nodes.
    fn wire_edges(&mut self, consumer_id: ProducerId) {
        // Extract SymbolIds from ArrayExpr inputs
        let input_syms: Vec<(usize, SymbolId)> = self.nodes[consumer_id.0 as usize]
            .semantics
            .input_exprs()
            .iter()
            .enumerate()
            .filter_map(|(i, ae)| {
                if let super::ArrayExpr::Ref(t) = ae {
                    if let TermKind::Var(sym) = &t.kind {
                        return Some((i, *sym));
                    }
                }
                None
            })
            .collect();

        for (input_index, sym) in input_syms {
            if let Some(&producer_id) = self.binding_map.get(&sym) {
                self.edges.push(ProducerEdge {
                    producer: producer_id,
                    consumer: consumer_id,
                    input_index,
                });
            }
        }
    }

    /// After all nodes and edges are built, count how many times each
    /// producer's output is used (both as SOAC inputs and in non-SOAC contexts).
    fn compute_use_counts(&mut self) {
        // Count edges per producer
        let mut counts: HashMap<ProducerId, usize> = HashMap::new();
        for edge in &self.edges {
            *counts.entry(edge.producer).or_insert(0) += 1;
        }
        for (id, count) in counts {
            self.nodes[id.0 as usize].use_count = count;
        }
        // TODO: also count non-SOAC uses by walking the full term body
        // For now, edge count is a lower bound.
    }
}

/// Substitute one param symbol → arg symbol throughout an ArraySemantics
/// (in both ArrayExpr inputs and Lambda bodies).
fn subst_in_semantics(
    sem: ArraySemantics,
    old: SymbolId,
    new: SymbolId,
    term_ids: &mut TermIdSource,
) -> ArraySemantics {
    use super::ArrayExpr;

    fn sub_ae(ae: ArrayExpr, old: SymbolId, new: SymbolId, ids: &mut TermIdSource) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(substitute_sym(*t, old, new, ids))),
            other => other,
        }
    }

    fn sub_lam(lam: Lambda, old: SymbolId, new: SymbolId, ids: &mut TermIdSource) -> Lambda {
        Lambda {
            body: Box::new(substitute_sym(*lam.body, old, new, ids)),
            ..lam
        }
    }

    match sem {
        ArraySemantics::Elementwise { inputs, body } => ArraySemantics::Elementwise {
            inputs: inputs.into_iter().map(|ae| sub_ae(ae, old, new, term_ids)).collect(),
            body: sub_lam(body, old, new, term_ids),
        },
        ArraySemantics::Reduction {
            input,
            op,
            init,
            props,
        } => ArraySemantics::Reduction {
            input: sub_ae(input, old, new, term_ids),
            op: sub_lam(op, old, new, term_ids),
            init,
            props,
        },
        ArraySemantics::PrefixScan { input, op, init } => ArraySemantics::PrefixScan {
            input: sub_ae(input, old, new, term_ids),
            op: sub_lam(op, old, new, term_ids),
            init,
        },
        ArraySemantics::Filter { input, pred } => ArraySemantics::Filter {
            input: sub_ae(input, old, new, term_ids),
            pred: sub_lam(pred, old, new, term_ids),
        },
        other => other,
    }
}

// =============================================================================
// Debug / display
// =============================================================================

impl ProducerGraph {
    /// Pretty-print the graph for debugging.
    pub fn debug_print(&self, symbols: &crate::SymbolTable) {
        eprintln!(
            "ProducerGraph ({} nodes, {} edges):",
            self.nodes.len(),
            self.edges.len()
        );
        for (i, node) in self.nodes.iter().enumerate() {
            let name = node
                .binding
                .and_then(|s| symbols.get(s).map(|n| n.clone()))
                .unwrap_or_else(|| format!("<tail>"));
            eprintln!(
                "  [{}] {} : uses={}, semantics={:?}",
                i,
                name,
                node.use_count,
                std::mem::discriminant(&node.semantics)
            );
        }
        for edge in &self.edges {
            eprintln!(
                "  {} → {} (input {})",
                edge.producer.0, edge.consumer.0, edge.input_index
            );
        }
    }
}
