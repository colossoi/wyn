//! Semantic classification of array-producing operations.
//!
//! Every array-producing expression (SOAC, literal, range, etc.) is classified
//! by what it *does*, not what syntax produced it. This enables fusion rules
//! based on semantic compatibility rather than syntactic pattern matching.

use super::{ArrayExpr, Lambda, Place, ReduceProps, Shape, SoacOp, Term, TermKind};
use crate::SymbolId;

// =============================================================================
// Core types
// =============================================================================

/// What an array-producing operation does, abstractly.
#[derive(Debug, Clone)]
pub enum ArraySemantics {
    /// Elementwise: output[i] = f(inputs[i]...) — shape-preserving, parallelizable.
    /// Covers Map and multi-input Map (zip-map).
    Elementwise {
        inputs: Vec<ArraySource>,
        body: Lambda,
    },

    /// Reduction: scalar = fold(op, init, input).
    Reduction {
        input: ArraySource,
        op: Lambda,
        init: Box<Term>,
        props: ReduceProps,
    },

    /// Prefix scan: output[i] = fold(op, init, input[0..=i]).
    /// Shape-preserving but with sequential dependency.
    PrefixScan {
        input: ArraySource,
        op: Lambda,
        init: Box<Term>,
    },

    /// Filter: output = input where pred(elem) — shape-changing.
    Filter {
        input: ArraySource,
        pred: Lambda,
    },

    /// Scatter: dest[indices[i]] = values[i] — indexed writes.
    ScatterOp {
        dest: PlaceSource,
        indices: ArraySource,
        values: ArraySource,
    },

    /// Histogram-style: dest[indices[i]] = op(dest[indices[i]], values[i]).
    IndexedReduction {
        dest: PlaceSource,
        indices: ArraySource,
        values: ArraySource,
        op: Lambda,
        init: Box<Term>,
        props: ReduceProps,
    },

    /// Materialized constant array.
    Literal(Vec<Term>),

    /// Generated sequence: output[i] = index_fn(i).
    Generate {
        shape: Shape,
        index_fn: Lambda,
    },

    /// Range/iota: output[i] = start + i (with implicit step of 1).
    Range {
        start: Box<Term>,
        len: Box<Term>,
    },

    /// Storage buffer reference (external memory).
    StorageBuffer {
        set: u32,
        binding: u32,
    },

    /// Opaque — can't classify this operation.
    Opaque,
}

/// Where an array value comes from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArraySource {
    /// Produced by a let-bound name in the current function body.
    Local(SymbolId),
    /// A function parameter.
    Param(SymbolId),
    /// A global definition.
    Global(SymbolId),
    /// A variable whose provenance we haven't resolved yet.
    Var(SymbolId),
}

/// Where a scatter/reduce_by_index destination comes from.
#[derive(Debug, Clone)]
pub enum PlaceSource {
    BufferSlice {
        set: u32,
        binding: u32,
    },
    LocalArray(SymbolId),
    /// Can't classify.
    Opaque,
}

// =============================================================================
// Semantic properties
// =============================================================================

impl ArraySemantics {
    /// Is this operation elementwise (output[i] depends only on input[i])?
    pub fn is_elementwise(&self) -> bool {
        matches!(self, ArraySemantics::Elementwise { .. })
    }

    /// Does this operation preserve the shape of its input?
    pub fn preserves_shape(&self) -> bool {
        matches!(
            self,
            ArraySemantics::Elementwise { .. }
                | ArraySemantics::PrefixScan { .. }
        )
    }

    /// Does this operation produce an array (vs a scalar)?
    pub fn produces_array(&self) -> bool {
        !matches!(self, ArraySemantics::Reduction { .. })
    }

    /// Is this operation pure (no side effects)?
    pub fn is_pure(&self) -> bool {
        !matches!(
            self,
            ArraySemantics::ScatterOp { .. } | ArraySemantics::IndexedReduction { .. }
        )
    }

    /// Get all input sources this operation reads from.
    pub fn input_sources(&self) -> Vec<&ArraySource> {
        match self {
            ArraySemantics::Elementwise { inputs, .. } => inputs.iter().collect(),
            ArraySemantics::Reduction { input, .. } => vec![input],
            ArraySemantics::PrefixScan { input, .. } => vec![input],
            ArraySemantics::Filter { input, .. } => vec![input],
            ArraySemantics::ScatterOp {
                indices, values, ..
            } => vec![indices, values],
            ArraySemantics::IndexedReduction {
                indices, values, ..
            } => vec![indices, values],
            ArraySemantics::Literal(_)
            | ArraySemantics::Generate { .. }
            | ArraySemantics::Range { .. }
            | ArraySemantics::StorageBuffer { .. }
            | ArraySemantics::Opaque => vec![],
        }
    }
}

// =============================================================================
// Extraction from TLC terms
// =============================================================================

/// Extract ArraySemantics from a SoacOp.
pub fn classify_soac(soac: &SoacOp) -> ArraySemantics {
    match soac {
        SoacOp::Map { lam, inputs } => ArraySemantics::Elementwise {
            inputs: inputs.iter().map(classify_array_expr_source).collect(),
            body: lam.clone(),
        },
        SoacOp::Reduce {
            op,
            ne,
            input,
            props,
        } => ArraySemantics::Reduction {
            input: classify_array_expr_source(input),
            op: op.clone(),
            init: ne.clone(),
            props: props.clone(),
        },
        SoacOp::Scan { op, ne, input } => ArraySemantics::PrefixScan {
            input: classify_array_expr_source(input),
            op: op.clone(),
            init: ne.clone(),
        },
        SoacOp::Filter { pred, input } => ArraySemantics::Filter {
            input: classify_array_expr_source(input),
            pred: pred.clone(),
        },
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => ArraySemantics::ScatterOp {
            dest: classify_place(dest),
            indices: classify_array_expr_source(indices),
            values: classify_array_expr_source(values),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            props,
        } => ArraySemantics::IndexedReduction {
            dest: classify_place(dest),
            indices: classify_array_expr_source(indices),
            values: classify_array_expr_source(values),
            op: op.clone(),
            init: ne.clone(),
            props: props.clone(),
        },
    }
}

/// Extract ArraySemantics from an ArrayExpr.
pub fn classify_array_expr(ae: &ArrayExpr) -> ArraySemantics {
    match ae {
        ArrayExpr::Ref(_) => ArraySemantics::Opaque, // just a reference, not a producer
        ArrayExpr::Zip(_) => ArraySemantics::Opaque,  // zip is consumed by enclosing Map
        ArrayExpr::Soac(op) => classify_soac(op),
        ArrayExpr::Generate {
            shape, index_fn, ..
        } => ArraySemantics::Generate {
            shape: shape.clone(),
            index_fn: index_fn.clone(),
        },
        ArrayExpr::Literal(terms) => ArraySemantics::Literal(terms.clone()),
        ArrayExpr::Range { start, len } => ArraySemantics::Range {
            start: start.clone(),
            len: len.clone(),
        },
        ArrayExpr::StorageBuffer { set, binding, .. } => ArraySemantics::StorageBuffer {
            set: *set,
            binding: *binding,
        },
    }
}

/// Classify a term's RHS as array semantics (if it's a SOAC or array expression).
pub fn classify_term(term: &Term) -> ArraySemantics {
    match &term.kind {
        TermKind::Soac(soac) => classify_soac(soac),
        TermKind::ArrayExpr(ae) => classify_array_expr(ae),
        _ => ArraySemantics::Opaque,
    }
}

/// Extract an ArraySource from an ArrayExpr (what array does this refer to?).
fn classify_array_expr_source(ae: &ArrayExpr) -> ArraySource {
    match ae {
        ArrayExpr::Ref(term) => match &term.kind {
            TermKind::Var(sym) => ArraySource::Var(*sym),
            _ => ArraySource::Var(SymbolId(u32::MAX)), // non-var ref, treat as opaque
        },
        // Nested SOAC/Generate/etc. in an input position — these are inline producers,
        // not references. The graph builder handles these separately.
        _ => ArraySource::Var(SymbolId(u32::MAX)),
    }
}

/// Classify a Place destination.
fn classify_place(place: &Place) -> PlaceSource {
    match place {
        Place::BufferSlice { .. } => {
            // TODO: extract set/binding from the base term
            PlaceSource::Opaque
        }
        Place::LocalArray { id, .. } => PlaceSource::LocalArray(*id),
    }
}
