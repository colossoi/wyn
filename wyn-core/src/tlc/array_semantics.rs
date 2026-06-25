//! Semantic classification of array-producing operations.
//!
//! Every array-producing expression (SOAC, literal, range, etc.) is classified
//! by what it *does*, not what syntax produced it. This enables fusion rules
//! based on semantic compatibility rather than syntactic pattern matching.

use super::VarRef;
use crate::LookupMap;
use crate::SymbolTable;

use super::{
    extract_lambda_params, ArrayExpr, Def, Lambda, Place, Program, SoacBody, SoacOp, Term, TermKind,
};
use crate::ast::TypeName;
use crate::SymbolId;
use polytype::Type;

// =============================================================================
// Core types
// =============================================================================

/// What an array-producing operation does, abstractly.
///
/// Inputs are stored as `ArrayExpr` directly to preserve
/// type information needed for code generation after fusion.
#[derive(Debug, Clone)]
pub enum ArraySemantics {
    /// Elementwise: output[i] = f(inputs[i]...) — shape-preserving, parallelizable.
    Elementwise {
        inputs: Vec<ArrayExpr>,
        body: SoacBody,
    },

    /// Reduction: scalar = fold(op, init, input).
    Reduction {
        input: ArrayExpr,
        op: SoacBody,
        init: Box<Term>,
    },

    /// Prefix scan: output[i] = fold(op, init, input[0..=i]).
    PrefixScan {
        input: ArrayExpr,
        op: SoacBody,
        init: Box<Term>,
    },

    /// Filter: output = input where pred(elem) — shape-changing.
    Filter {
        input: ArrayExpr,
        pred: SoacBody,
    },

    /// Scatter: over `inputs`, `lam` yields `(index, value)` per element,
    /// written as `dest[index] = value`. Carries the full `dest`/`lam` so a
    /// `MapIntoScatter` fusion can rebuild the `Scatter` SOAC from semantics.
    ScatterOp {
        dest: Place,
        lam: SoacBody,
        inputs: Vec<ArrayExpr>,
    },

    /// Histogram-style: dest[indices[i]] = op(dest[indices[i]], values[i]).
    IndexedReduction {
        dest: PlaceSource,
        indices: ArrayExpr,
        values: ArrayExpr,
        op: SoacBody,
        init: Box<Term>,
    },

    /// Materialized constant array.
    Literal(Vec<Term>),

    /// Range/iota: output[i] = start + i (with implicit step of 1).
    Range {
        start: Box<Term>,
        len: Box<Term>,
    },

    /// Storage buffer reference (external memory).
    StorageBuffer {
        binding: crate::BindingRef,
    },

    /// Opaque — can't classify this operation.
    Opaque,
}

/// Where a scatter/reduce_by_index destination comes from.
#[derive(Debug, Clone)]
pub enum PlaceSource {
    BufferSlice {
        binding: crate::BindingRef,
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
            ArraySemantics::Elementwise { .. } | ArraySemantics::PrefixScan { .. }
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

    /// Get all array inputs this operation reads from.
    pub fn input_exprs(&self) -> Vec<&ArrayExpr> {
        match self {
            ArraySemantics::Elementwise { inputs, .. } => inputs.iter().collect(),
            ArraySemantics::Reduction { input, .. } => vec![input],
            ArraySemantics::PrefixScan { input, .. } => vec![input],
            ArraySemantics::Filter { input, .. } => vec![input],
            ArraySemantics::ScatterOp { inputs, .. } => inputs.iter().collect(),
            ArraySemantics::IndexedReduction { indices, values, .. } => vec![indices, values],
            ArraySemantics::Literal(_)
            | ArraySemantics::Range { .. }
            | ArraySemantics::StorageBuffer { .. }
            | ArraySemantics::Opaque => vec![],
        }
    }
}

// =============================================================================
// Fusion rules — semantic compatibility
// =============================================================================

/// A fusion that is both *eligible* and *buildable*. `can_fuse` returns one only
/// when [`crate::tlc::fusion`]'s builder can construct it, so there is never a
/// verdict the driver has to silently drop — eligibility is coupled to
/// construction. (The old `RangeIntoMap` verdict had no builder and always
/// no-op'd; it is gone until a Range builder exists — see plan item 4.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionRecipe {
    /// Compose elementwise bodies: map(g, map(f, a)) → map(g∘f, a)
    ComposeElementwise,
    /// Compose map into reduce: reduce(op, ne, map(f, a)) → reduce(op∘f, ne, a)
    MapIntoReduce,
    /// Compose map into scan: scan(op, ne, map(f, a)) → scan(op∘f, ne, a)
    MapIntoScan,
    /// Compose a map producer into a scatter's envelope at the fused input slot:
    /// `scatter(g, dest, map(f, a), vs)` → `scatter(g∘f at slot, dest, a, vs)`.
    /// The producer's outputs must be fully consumed (scatter has no
    /// pass-through results), which the single-use edge filter guarantees.
    MapIntoScatter,
    /// Fuse a filter into a reduce: `reduce(op, ne, filter(p, a))` → a
    /// single-`Reduce`-accumulator `Screma` over `a` whose step is `op∘mask`,
    /// where `mask = λx. if p(x) then x else ne`. Valid because `ne` is `op`'s
    /// neutral element (reduce's contract), so the masked-out elements fold in
    /// as no-ops. Avoids materializing the filtered array — the result
    /// parallelizes as an ordinary fused map→reduce.
    FilterIntoReduce,
}

/// Determine if a producer can be fused into a consumer. Returns `None` when no
/// fusion is possible *or* when no builder exists for the pair, so every
/// `Some(recipe)` is guaranteed buildable.
pub fn can_fuse(producer: &ArraySemantics, consumer: &ArraySemantics) -> Option<FusionRecipe> {
    match (producer, consumer) {
        // Elementwise → Elementwise: compose bodies
        (ArraySemantics::Elementwise { .. }, ArraySemantics::Elementwise { .. }) => {
            Some(FusionRecipe::ComposeElementwise)
        }

        // Elementwise → Reduction: compose map into reduce body
        (ArraySemantics::Elementwise { .. }, ArraySemantics::Reduction { .. }) => {
            Some(FusionRecipe::MapIntoReduce)
        }

        // Elementwise → PrefixScan: compose map into scan body
        (ArraySemantics::Elementwise { .. }, ArraySemantics::PrefixScan { .. }) => {
            Some(FusionRecipe::MapIntoScan)
        }

        // Elementwise → ScatterOp: compose map into the scatter envelope at the
        // fused input slot (Futhark thesis §7.3.1 map-scatter rule).
        (ArraySemantics::Elementwise { .. }, ArraySemantics::ScatterOp { .. }) => {
            Some(FusionRecipe::MapIntoScatter)
        }

        // Filter → Reduction: fold the filter into a masked single-accumulator
        // Screma, avoiding the compacted intermediate array entirely.
        (ArraySemantics::Filter { .. }, ArraySemantics::Reduction { .. }) => {
            Some(FusionRecipe::FilterIntoReduce)
        }

        // Everything else: not fusible (or no builder yet).
        _ => None,
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
    symbols: &mut SymbolTable,
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
    symbols: &mut SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> Option<ArraySemantics> {
    let (prod_inputs, prod_body) = match producer {
        ArraySemantics::Elementwise { inputs, body } => (inputs, body),
        _ => return None,
    };
    let (cons_op, cons_init) = match consumer {
        ArraySemantics::Reduction { op, init, .. } => (op, init),
        _ => return None,
    };

    // Compose: map body feeds into reduce op's element parameter
    let composed_op = compose_map_into_op(prod_body, cons_op, symbols, term_ids);

    Some(ArraySemantics::Reduction {
        input: prod_inputs.first().cloned().unwrap_or(ArrayExpr::Literal(vec![])),
        op: composed_op,
        init: cons_init.clone(),
    })
}

/// Compose a map into a scan: scan(op, ne, map(f, a)) → scan(op∘f, ne, a).
pub fn compose_map_into_scan(
    producer: &ArraySemantics,
    consumer: &ArraySemantics,
    symbols: &mut SymbolTable,
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
        input: prod_inputs.first().cloned().unwrap_or(ArrayExpr::Literal(vec![])),
        op: composed_op,
        init: cons_init.clone(),
    })
}

/// Compose two lambda bodies: g∘f where f maps input→intermediate, g maps intermediate→output.
/// Result lambda has f's params and g's return type.
fn compose_lambda_bodies(
    f: &SoacBody,
    g: &SoacBody,
    symbols: &mut SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> SoacBody {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let intermediate_ty = f.lam.ret_ty.clone();

    // Substitute g's first parameter with the fresh symbol in g's body
    let g_param = g.lam.params[0].0;
    let g_body_substituted =
        super::fusion::substitute_sym(*g.lam.body.clone(), g_param, fresh_sym, term_ids);

    let composed_body = Term {
        id: term_ids.next_id(),
        ty: g.lam.ret_ty.clone(),
        span: f.lam.body.span,
        kind: TermKind::Let {
            name: fresh_sym,
            name_ty: intermediate_ty,
            rhs: f.lam.body.clone(),
            body: Box::new(g_body_substituted),
        },
    };

    SoacBody {
        lam: Lambda {
            params: f.lam.params.clone(),
            body: Box::new(composed_body),
            ret_ty: g.lam.ret_ty.clone(),
        },
        captures: vec![],
    }
}

/// Compose a map body into a reduce/scan operator.
/// map_body: A → B, op: (Acc, B) → Acc → composed: (Acc, A) → Acc
fn compose_map_into_op(
    map_body: &SoacBody,
    op: &SoacBody,
    symbols: &mut SymbolTable,
    term_ids: &mut super::TermIdSource,
) -> SoacBody {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let intermediate_ty = map_body.lam.ret_ty.clone();

    // The op has params [acc, elem]. Substitute elem with fresh.
    let elem_param = op.lam.params[1].0;
    let op_body_substituted =
        super::fusion::substitute_sym(*op.lam.body.clone(), elem_param, fresh_sym, term_ids);

    let composed_body = Term {
        id: term_ids.next_id(),
        ty: op.lam.ret_ty.clone(),
        span: map_body.lam.body.span,
        kind: TermKind::Let {
            name: fresh_sym,
            name_ty: intermediate_ty,
            rhs: map_body.lam.body.clone(),
            body: Box::new(op_body_substituted),
        },
    };

    SoacBody {
        lam: Lambda {
            params: vec![op.lam.params[0].clone(), map_body.lam.params[0].clone()],
            body: Box::new(composed_body),
            ret_ty: op.lam.ret_ty.clone(),
        },
        captures: vec![],
    }
}

// =============================================================================
// Extraction from TLC terms
// =============================================================================

/// Extract ArraySemantics from a SoacOp, preserving ArrayExpr inputs with types.
pub fn classify_soac(soac: &SoacOp) -> ArraySemantics {
    match soac {
        SoacOp::Map { lam, inputs, .. } => ArraySemantics::Elementwise {
            inputs: inputs.clone(),
            body: lam.clone(),
        },
        SoacOp::Reduce { op, ne, input } => ArraySemantics::Reduction {
            input: input.clone(),
            op: op.clone(),
            init: ne.clone(),
        },
        SoacOp::Scan { op, ne, input, .. } => ArraySemantics::PrefixScan {
            input: input.clone(),
            op: op.clone(),
            init: ne.clone(),
        },
        SoacOp::Filter { pred, input, .. } => ArraySemantics::Filter {
            input: input.clone(),
            pred: pred.clone(),
        },
        SoacOp::Scatter { dest, lam, inputs } => ArraySemantics::ScatterOp {
            dest: dest.clone(),
            lam: lam.clone(),
            inputs: inputs.clone(),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
        } => ArraySemantics::IndexedReduction {
            dest: classify_place(dest),
            indices: indices.clone(),
            values: values.clone(),
            op: op.clone(),
            init: ne.clone(),
        },
        // A multi-result map+accumulator SOAC (the fused form, including a
        // map-into-reduce); classified as opaque because it is not analyzed
        // further by the semantic framework.
        SoacOp::Screma { .. } => ArraySemantics::Opaque,
    }
}

/// Extract ArraySemantics from an ArrayExpr.
pub fn classify_array_expr(ae: &ArrayExpr) -> ArraySemantics {
    match ae {
        ArrayExpr::Ref(_) => ArraySemantics::Opaque, // just a reference, not a producer
        ArrayExpr::Zip(_) => ArraySemantics::Opaque, // zip is consumed by enclosing Map
        ArrayExpr::Soac(op) => classify_soac(op),
        ArrayExpr::Literal(terms) => ArraySemantics::Literal(terms.clone()),
        ArrayExpr::Range { start, len, .. } => ArraySemantics::Range {
            start: start.clone(),
            len: len.clone(),
        },
        ArrayExpr::StorageView(crate::tlc::StorageView { binding: br, .. }) => {
            ArraySemantics::StorageBuffer { binding: *br }
        }
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
    /// The ArrayExpr inputs may reference the callee's parameter symbols,
    /// which the ProducerGraph substitutes with call arguments.
    Produces(ArraySemantics),

    /// The result is one of the parameters passed through unchanged.
    PassesThrough(usize),

    /// The result is a scalar or non-array value.
    Scalar,

    /// Can't determine what this function does.
    Unknown,
}

/// Compute function summaries for all defs in a program, with fixpoint
/// propagation for interprocedural analysis.
pub fn summarize_program(program: &Program) -> LookupMap<SymbolId, FunctionSummary> {
    // Initial pass: summarize each def without interprocedural info
    let mut summaries: LookupMap<SymbolId, FunctionSummary> = LookupMap::new();
    for def in &program.defs {
        let summary = summarize_def(def);
        summaries.insert(def.name, summary);
    }

    // Fixpoint: re-analyze defs that returned Unknown, using existing summaries
    // to see through function calls. Repeat until stable.
    let mut changed = true;
    while changed {
        changed = false;
        for def in &program.defs {
            let current = summaries.get(&def.name).unwrap();
            if !matches!(current.result, ResultSemantics::Unknown) {
                continue; // already resolved
            }

            let (params, inner_body) = extract_lambda_params(&def.body);
            if params.is_empty() {
                continue;
            }

            let param_syms: Vec<SymbolId> = params.iter().map(|(s, _)| *s).collect();
            let new_result = analyze_body_with_summaries(&inner_body, &param_syms, &summaries);

            if !matches!(new_result, ResultSemantics::Unknown) {
                summaries.insert(
                    def.name,
                    FunctionSummary {
                        result: new_result,
                        params,
                    },
                );
                changed = true;
            }
        }
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

    FunctionSummary { result, params }
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
        TermKind::Var(VarRef::Symbol(sym)) => {
            if let Some(idx) = params.iter().position(|p| p == sym) {
                ResultSemantics::PassesThrough(idx)
            } else {
                ResultSemantics::Unknown
            }
        }

        // Function call — unknown without summary context
        TermKind::App { .. } => ResultSemantics::Unknown,

        _ => ResultSemantics::Unknown,
    }
}

/// Analyze a body with access to existing function summaries.
/// This extends `analyze_body` to see through function calls.
fn analyze_body_with_summaries(
    body: &Term,
    params: &[SymbolId],
    summaries: &LookupMap<SymbolId, FunctionSummary>,
) -> ResultSemantics {
    match &body.kind {
        TermKind::Soac(soac) => {
            let semantics = classify_soac_with_params(soac, params);
            ResultSemantics::Produces(semantics)
        }

        TermKind::Let { body, .. } => analyze_body_with_summaries(body, params, summaries),

        TermKind::Var(VarRef::Symbol(sym)) => {
            if let Some(idx) = params.iter().position(|p| p == sym) {
                ResultSemantics::PassesThrough(idx)
            } else {
                ResultSemantics::Unknown
            }
        }

        // Function call: look up callee summary
        TermKind::App { func, args } => {
            if let TermKind::Var(VarRef::Symbol(callee_sym)) = &func.kind {
                if let Some(callee_summary) = summaries.get(callee_sym) {
                    match &callee_summary.result {
                        ResultSemantics::Produces(semantics) => {
                            // The callee produces an array — this call does too.
                            // The semantics reference the callee's params;
                            // the graph builder will substitute them with call args later.
                            ResultSemantics::Produces(semantics.clone())
                        }
                        ResultSemantics::PassesThrough(idx) => {
                            // Callee passes through one of its params — check if the
                            // corresponding call arg is one of OUR params.
                            if *idx < args.len() {
                                if let TermKind::Var(VarRef::Symbol(arg_sym)) = &args[*idx].kind {
                                    if let Some(our_idx) = params.iter().position(|p| p == arg_sym) {
                                        return ResultSemantics::PassesThrough(our_idx);
                                    }
                                }
                            }
                            ResultSemantics::Unknown
                        }
                        _ => ResultSemantics::Unknown,
                    }
                } else {
                    ResultSemantics::Unknown
                }
            } else {
                ResultSemantics::Unknown
            }
        }

        _ => ResultSemantics::Unknown,
    }
}

/// Classify a SoacOp for summary extraction. Same as `classify_soac` —
/// we preserve the ArrayExpr inputs directly (they carry type info).
fn classify_soac_with_params(soac: &SoacOp, _params: &[SymbolId]) -> ArraySemantics {
    classify_soac(soac)
}
