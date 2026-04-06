//! Semantic classification of array-producing operations.
//!
//! Every array-producing expression (SOAC, literal, range, etc.) is classified
//! by what it *does*, not what syntax produced it. This enables fusion rules
//! based on semantic compatibility rather than syntactic pattern matching.

use std::collections::HashMap;

use super::{ArrayExpr, Def, Lambda, Place, Program, ReduceProps, Shape, SoacOp, Term, TermKind,
            extract_lambda_params};
use crate::SymbolId;
use crate::ast::TypeName;
use polytype::Type;

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
// Fusion rules — semantic compatibility
// =============================================================================

/// What kind of fusion is possible between a producer and consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionKind {
    /// Compose elementwise bodies: map(g, map(f, a)) → map(g∘f, a)
    ComposeElementwise,
    /// Compose map into reduce: reduce(op, ne, map(f, a)) → reduce(op∘f, ne, a)
    MapIntoReduce,
    /// Compose map into scan: scan(op, ne, map(f, a)) → scan(op∘f, ne, a)
    MapIntoScan,
    /// Compose map into filter predicate: filter(p, map(f, a)) → filter(p∘f, a) then map(f, ...)
    /// (only when pred doesn't use map's output value — deferred for now)
    // MapIntoFilter,
    /// Inline a generator into an elementwise consumer
    GenerateIntoMap,
    /// Inline a range into an elementwise consumer
    RangeIntoMap,
    /// Not fusible
    NotFusible,
}

/// Determine if a producer can be fused into a consumer.
pub fn can_fuse(producer: &ArraySemantics, consumer: &ArraySemantics) -> FusionKind {
    match (producer, consumer) {
        // Elementwise → Elementwise: compose bodies
        (ArraySemantics::Elementwise { .. }, ArraySemantics::Elementwise { .. }) => {
            FusionKind::ComposeElementwise
        }

        // Elementwise → Reduction: compose map into reduce body
        (ArraySemantics::Elementwise { .. }, ArraySemantics::Reduction { .. }) => {
            FusionKind::MapIntoReduce
        }

        // Elementwise → PrefixScan: compose map into scan body
        (ArraySemantics::Elementwise { .. }, ArraySemantics::PrefixScan { .. }) => {
            FusionKind::MapIntoScan
        }

        // Generate → Elementwise: inline generator into map body
        (ArraySemantics::Generate { .. }, ArraySemantics::Elementwise { .. }) => {
            FusionKind::GenerateIntoMap
        }

        // Range → Elementwise: inline range into map body
        (ArraySemantics::Range { .. }, ArraySemantics::Elementwise { .. }) => {
            FusionKind::RangeIntoMap
        }

        // Everything else: not fusible
        _ => FusionKind::NotFusible,
    }
}

// =============================================================================
// Semantic composition — building fused operations
// =============================================================================

/// Compose two elementwise operations: map(g, map(f, a)) → map(g∘f, a).
///
/// Takes the producer's inputs and body, and the consumer's body.
/// Returns a new Elementwise with the producer's inputs and a composed body.
pub fn compose_elementwise(
    producer: &ArraySemantics,
    consumer: &ArraySemantics,
    symbols: &mut crate::SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> Option<ArraySemantics> {
    let (prod_inputs, prod_body) = match producer {
        ArraySemantics::Elementwise { inputs, body } => (inputs, body),
        _ => return None,
    };
    let cons_body = match consumer {
        ArraySemantics::Elementwise { body, .. } => body,
        _ => return None,
    };

    let composed = compose_lambda_bodies(prod_body, cons_body, symbols, term_ids);
    Some(ArraySemantics::Elementwise {
        inputs: prod_inputs.clone(),
        body: composed,
    })
}

/// Compose a map into a reduce: reduce(op, ne, map(f, a)) → reduce(op∘f, ne, a).
pub fn compose_map_into_reduce(
    producer: &ArraySemantics,
    consumer: &ArraySemantics,
    symbols: &mut crate::SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> Option<ArraySemantics> {
    let (prod_inputs, prod_body) = match producer {
        ArraySemantics::Elementwise { inputs, body } => (inputs, body),
        _ => return None,
    };
    let (cons_op, cons_init, cons_props) = match consumer {
        ArraySemantics::Reduction { op, init, props, .. } => (op, init, props),
        _ => return None,
    };

    // Compose: map body feeds into reduce op's element parameter
    let composed_op = compose_map_into_op(prod_body, cons_op, symbols, term_ids);

    Some(ArraySemantics::Reduction {
        input: prod_inputs.first().cloned().unwrap_or(ArraySource::Var(SymbolId(u32::MAX))),
        op: composed_op,
        init: cons_init.clone(),
        props: cons_props.clone(),
    })
}

/// Compose a map into a scan: scan(op, ne, map(f, a)) → scan(op∘f, ne, a).
pub fn compose_map_into_scan(
    producer: &ArraySemantics,
    consumer: &ArraySemantics,
    symbols: &mut crate::SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> Option<ArraySemantics> {
    let (prod_inputs, prod_body) = match producer {
        ArraySemantics::Elementwise { inputs, body } => (inputs, body),
        _ => return None,
    };
    let (cons_op, cons_init) = match consumer {
        ArraySemantics::PrefixScan { op, init, .. } => (op, init),
        _ => return None,
    };

    let composed_op = compose_map_into_op(prod_body, cons_op, symbols, term_ids);

    Some(ArraySemantics::PrefixScan {
        input: prod_inputs.first().cloned().unwrap_or(ArraySource::Var(SymbolId(u32::MAX))),
        op: composed_op,
        init: cons_init.clone(),
    })
}

/// Compose two lambda bodies: g∘f where f maps input→intermediate, g maps intermediate→output.
/// Result lambda has f's params and g's return type.
fn compose_lambda_bodies(
    f: &Lambda,
    g: &Lambda,
    symbols: &mut crate::SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> Lambda {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let intermediate_ty = f.ret_ty.clone();

    // Substitute g's first parameter with the fresh symbol in g's body
    let g_param = g.params[0].0;
    let g_body_substituted = super::fusion::substitute_sym(*g.body.clone(), g_param, fresh_sym, term_ids);

    let composed_body = Term {
        id: term_ids.next_id(),
        ty: g.ret_ty.clone(),
        span: f.body.span,
        kind: TermKind::Let {
            name: fresh_sym,
            name_ty: intermediate_ty,
            rhs: f.body.clone(),
            body: Box::new(g_body_substituted),
        },
    };

    Lambda {
        params: f.params.clone(),
        body: Box::new(composed_body),
        ret_ty: g.ret_ty.clone(),
        captures: vec![],
    }
}

/// Compose a map body into a reduce/scan operator.
/// map_body: A → B, op: (Acc, B) → Acc → composed: (Acc, A) → Acc
fn compose_map_into_op(
    map_body: &Lambda,
    op: &Lambda,
    symbols: &mut crate::SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> Lambda {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let intermediate_ty = map_body.ret_ty.clone();

    // The op has params [acc, elem]. Substitute elem with fresh.
    let elem_param = op.params[1].0;
    let op_body_substituted = super::fusion::substitute_sym(*op.body.clone(), elem_param, fresh_sym, term_ids);

    let composed_body = Term {
        id: term_ids.next_id(),
        ty: op.ret_ty.clone(),
        span: map_body.body.span,
        kind: TermKind::Let {
            name: fresh_sym,
            name_ty: intermediate_ty,
            rhs: map_body.body.clone(),
            body: Box::new(op_body_substituted),
        },
    };

    Lambda {
        params: vec![op.params[0].clone(), map_body.params[0].clone()],
        body: Box::new(composed_body),
        ret_ty: op.ret_ty.clone(),
        captures: vec![],
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

// =============================================================================
// Function summaries — interprocedural analysis
// =============================================================================

/// Summary of what a function does, seen from outside.
///
/// Computed by analyzing the function body and expressed in terms of
/// ArraySemantics. Enables interprocedural fusion: callers can fuse
/// through function calls without inlining.
#[derive(Debug, Clone)]
pub struct FunctionSummary {
    /// What the function returns (in terms of its parameters).
    pub result: ResultSemantics,
    /// The function's parameter symbols and types.
    pub params: Vec<(SymbolId, Type<TypeName>)>,
}

/// What a function's return value is, semantically.
#[derive(Debug, Clone)]
pub enum ResultSemantics {
    /// The result is a SOAC applied to inputs that are (possibly) parameters.
    /// The ArraySemantics' ArraySources use ArraySource::Param for parameter
    /// references, so the caller can substitute its own arguments.
    Produces(ArraySemantics),

    /// The result is one of the parameters passed through unchanged.
    PassesThrough(usize),

    /// The result is a scalar or non-array value.
    Scalar,

    /// Can't determine what this function does.
    Unknown,
}

/// Compute function summaries for all defs in a program.
pub fn summarize_program(program: &Program) -> HashMap<SymbolId, FunctionSummary> {
    let mut summaries = HashMap::new();
    for def in &program.defs {
        summaries.insert(def.name, summarize_def(def));
    }
    summaries
}

/// Compute a summary for a single def.
pub fn summarize_def(def: &Def) -> FunctionSummary {
    let (params, inner_body) = extract_lambda_params(&def.body);

    if params.is_empty() {
        return FunctionSummary {
            result: ResultSemantics::Unknown,
            params: vec![],
        };
    }

    let param_syms: Vec<SymbolId> = params.iter().map(|(s, _)| *s).collect();
    let result = analyze_body(&inner_body, &param_syms);

    FunctionSummary {
        result,
        params,
    }
}

/// Analyze a function body to determine what it returns.
///
/// Walks through Let chains to find the tail expression, classifying it
/// as a SOAC, parameter passthrough, or unknown.
fn analyze_body(body: &Term, params: &[SymbolId]) -> ResultSemantics {
    match &body.kind {
        // Direct SOAC — classify it, resolving inputs against params
        TermKind::Soac(soac) => {
            let semantics = classify_soac_with_params(soac, params);
            ResultSemantics::Produces(semantics)
        }

        // Let chain — skip through to the inner body
        TermKind::Let { body, .. } => analyze_body(body, params),

        // Variable — might be a parameter passthrough
        TermKind::Var(sym) => {
            if let Some(idx) = params.iter().position(|p| p == sym) {
                ResultSemantics::PassesThrough(idx)
            } else {
                ResultSemantics::Unknown
            }
        }

        // Function call — could be interprocedural (handled by fixpoint later)
        TermKind::App { .. } => ResultSemantics::Unknown,

        // Anything with control flow — bail for now
        _ => ResultSemantics::Unknown,
    }
}

/// Classify a SoacOp, resolving ArrayExpr inputs to parameter references
/// where possible. This is the key difference from `classify_soac` — here
/// we know which SymbolIds are parameters and tag them as `ArraySource::Param`.
fn classify_soac_with_params(soac: &SoacOp, params: &[SymbolId]) -> ArraySemantics {
    let resolve_source = |ae: &ArrayExpr| -> ArraySource {
        match ae {
            ArrayExpr::Ref(term) => match &term.kind {
                TermKind::Var(sym) => {
                    if let Some(_idx) = params.iter().position(|p| p == sym) {
                        ArraySource::Param(*sym)
                    } else {
                        ArraySource::Local(*sym)
                    }
                }
                _ => ArraySource::Var(SymbolId(u32::MAX)),
            },
            _ => ArraySource::Var(SymbolId(u32::MAX)),
        }
    };

    match soac {
        SoacOp::Map { lam, inputs } => ArraySemantics::Elementwise {
            inputs: inputs.iter().map(resolve_source).collect(),
            body: lam.clone(),
        },
        SoacOp::Reduce { op, ne, input, props } => ArraySemantics::Reduction {
            input: resolve_source(input),
            op: op.clone(),
            init: ne.clone(),
            props: props.clone(),
        },
        SoacOp::Scan { op, ne, input } => ArraySemantics::PrefixScan {
            input: resolve_source(input),
            op: op.clone(),
            init: ne.clone(),
        },
        SoacOp::Filter { pred, input } => ArraySemantics::Filter {
            input: resolve_source(input),
            pred: pred.clone(),
        },
        SoacOp::Scatter { dest, indices, values } => ArraySemantics::ScatterOp {
            dest: classify_place(dest),
            indices: resolve_source(indices),
            values: resolve_source(values),
        },
        SoacOp::ReduceByIndex { dest, op, ne, indices, values, props } => {
            ArraySemantics::IndexedReduction {
                dest: classify_place(dest),
                indices: resolve_source(indices),
                values: resolve_source(values),
                op: op.clone(),
                init: ne.clone(),
                props: props.clone(),
            }
        }
    }
}
