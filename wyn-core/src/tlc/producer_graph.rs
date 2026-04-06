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

use super::array_semantics::{ArraySemantics, ArraySource, classify_term};
use super::{Term, TermKind};
use crate::ast::TypeName;
use crate::SymbolId;
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
) -> ProducerGraph {
    let mut builder = GraphBuilder {
        nodes: Vec::new(),
        edges: Vec::new(),
        binding_map: HashMap::new(),
        param_set: params.iter().copied().collect(),
    };

    builder.walk_term(body);
    builder.compute_use_counts();

    ProducerGraph {
        nodes: builder.nodes,
        edges: builder.edges,
        binding_map: builder.binding_map,
    }
}

struct GraphBuilder {
    nodes: Vec<ProducerNode>,
    edges: Vec<ProducerEdge>,
    binding_map: HashMap<SymbolId, ProducerId>,
    param_set: std::collections::HashSet<SymbolId>,
}

impl GraphBuilder {
    fn add_node(&mut self, node: ProducerNode) -> ProducerId {
        let id = ProducerId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Walk a term, extracting producer nodes from Let bindings.
    fn walk_term(&mut self, term: &Term) {
        match &term.kind {
            TermKind::Let {
                name, rhs, body, ..
            } => {
                // Classify the RHS
                let semantics = classify_term(rhs);

                // Only create a node if this is actually an array producer
                if !matches!(semantics, ArraySemantics::Opaque) {
                    let id = self.add_node(ProducerNode {
                        semantics,
                        binding: Some(*name),
                        ty: rhs.ty.clone(),
                        use_count: 0, // computed later
                    });
                    self.wire_edges(id);
                    self.binding_map.insert(*name, id);
                }

                // Continue walking the body
                self.walk_term(body);
            }

            // The tail expression might also be a SOAC (the final result)
            TermKind::Soac(_) | TermKind::ArrayExpr(_) => {
                let semantics = classify_term(term);
                if !matches!(semantics, ArraySemantics::Opaque) {
                    let id = self.add_node(ProducerNode {
                        semantics,
                        binding: None, // tail expression, no binding
                        ty: term.ty.clone(),
                        use_count: 0,
                    });
                    // Wire edges from this node's inputs to existing producers
                    self.wire_edges(id);
                }
            }

            _ => {}
        }
    }

    /// Wire edges from a consumer node's inputs to their producer nodes.
    fn wire_edges(&mut self, consumer_id: ProducerId) {
        let sources: Vec<(usize, ArraySource)> = self.nodes[consumer_id.0 as usize]
            .semantics
            .input_sources()
            .iter()
            .enumerate()
            .map(|(i, s)| (i, (*s).clone()))
            .collect();

        for (input_index, source) in sources {
            if let ArraySource::Var(sym) = source {
                if let Some(&producer_id) = self.binding_map.get(&sym) {
                    self.edges.push(ProducerEdge {
                        producer: producer_id,
                        consumer: consumer_id,
                        input_index,
                    });
                }
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

// =============================================================================
// Debug / display
// =============================================================================

impl ProducerGraph {
    /// Pretty-print the graph for debugging.
    pub fn debug_print(&self, symbols: &crate::SymbolTable) {
        eprintln!("ProducerGraph ({} nodes, {} edges):", self.nodes.len(), self.edges.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let name = node
                .binding
                .and_then(|s| symbols.get(s).map(|n| n.clone()))
                .unwrap_or_else(|| format!("<tail>"));
            eprintln!(
                "  [{}] {} : uses={}, semantics={:?}",
                i, name, node.use_count,
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
