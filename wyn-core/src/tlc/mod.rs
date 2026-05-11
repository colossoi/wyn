//! Typed Lambda Calculus (TLC) representation.
//!
//! A minimal typed lambda calculus IR for SOAC fusion analysis.
//! Lambdas remain as values (not yet defunctionalized).

pub mod array_semantics;
pub mod buffer_specialize;
pub mod closure_calls_lower;
pub mod closure_convert;
pub mod defaults;
pub mod fusion;
pub mod hof_specialize;
pub mod inline;
pub mod monomorphize;
pub mod normalize;
pub mod ownership;
pub mod parallelize;
pub mod partial_eval;
pub mod producer_graph;
pub mod run;
pub mod soa;
pub mod specialize;

use crate::ast::{self, NodeId, Span, TypeName};
use crate::interface;
use crate::types::TypeExt;
use crate::{SymbolId, SymbolTable, TypeTable};
use polytype::Type;
use std::collections::HashMap;

/// SOAC names that are intercepted in transform_application and turned into
/// first-class SOAC nodes rather than intrinsic calls.
const SOAC_NAMES: &[&str] = &[
    "map",
    "reduce",
    "scan",
    "filter",
    "zip",
    "zip2",
    "zip3",
    "zip4",
    "zip5",
    "reduce_by_index",
];

// =============================================================================
// Helper functions
// =============================================================================

/// Count the arity of a function type by counting the number of arrow constructors.
/// For `A -> B -> C`, returns 2.
/// For non-function types, returns 0.
fn count_function_arity(ty: &Type<TypeName>) -> usize {
    match ty {
        Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => 1 + count_function_arity(&args[1]),
        _ => 0,
    }
}

/// Extract lambda parameters and body from a term.
/// If the term is a Lambda, returns its params and body.
/// If not, returns empty params and the term itself.
pub fn extract_lambda_params(term: &Term) -> (Vec<(SymbolId, Type<TypeName>)>, Term) {
    match &term.kind {
        TermKind::Lambda(Lambda { params, body, .. }) => (params.clone(), (**body).clone()),
        _ => (vec![], term.clone()),
    }
}

/// Count the number of nodes in a term tree.
/// Used as a size heuristic for inlining decisions.
pub fn term_size(term: &Term) -> usize {
    let mut count = 1; // count this node
    term.for_each_child(&mut |child| count += term_size(child));
    count
}

/// Collect all `TermKind::Var(crate::tlc::VarRef::Symbol(sym))` SymbolIds referenced anywhere in a term tree.
/// This is a raw collection with no scope tracking — used for DCE reachability.
pub fn collect_var_refs(term: &Term) -> Vec<SymbolId> {
    let mut refs = Vec::new();
    collect_var_refs_inner(term, &mut refs);
    refs
}

fn collect_var_refs_inner(term: &Term, refs: &mut Vec<SymbolId>) {
    // Var leaf: the only TermKind that directly contributes a ref.
    if let TermKind::Var(crate::tlc::VarRef::Symbol(sym)) = &term.kind {
        refs.push(*sym);
    }

    // Place::LocalArray also contributes a non-Term SymbolId ref.
    // for_each_child doesn't expose Place internals, so handle here.
    collect_place_ids_in_soacs(term, refs);

    // Recurse into all Term children.
    term.for_each_child(&mut |child| collect_var_refs_inner(child, refs));
}

/// Collect SymbolIds from Place::LocalArray inside Scatter/ReduceByIndex SOACs.
/// These are non-Term refs that for_each_child can't reach.
fn collect_place_ids_in_soacs(term: &Term, refs: &mut Vec<SymbolId>) {
    if let TermKind::Soac(soac) = &term.kind {
        let place = match soac {
            SoacOp::Scatter { dest, .. } | SoacOp::ReduceByIndex { dest, .. } => Some(dest),
            _ => None,
        };
        if let Some(Place::LocalArray { id, .. }) = place {
            refs.push(*id);
        }
    }
}

// =============================================================================
// TLC Terms
// =============================================================================

/// A unique identifier for TLC terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TermId(pub u32);

/// Counter for generating unique TermIds.
#[derive(Debug, Clone, Default)]
pub struct TermIdSource {
    next: u32,
}

impl TermIdSource {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    pub fn next_id(&mut self) -> TermId {
        let id = TermId(self.next);
        self.next += 1;
        id
    }
}

/// A typed term in the lambda calculus.
#[derive(Debug, Clone)]
pub struct Term {
    pub id: TermId,
    pub ty: Type<TypeName>,
    pub span: Span,
    pub kind: TermKind,
}

/// Reference target for a `Var` term.
///
/// Catalog builtins are identified by `BuiltinId` from type-check
/// onward; everything else (locals, user-defined functions, prelude
/// functions, top-level constants) is a `Symbol(SymbolId)`. The
/// distinction lets TLC → EGIR dispatch structurally without
/// re-deriving builtin identity by string-matching `_w_intrinsic_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarRef {
    /// Catalog builtin — identified by `BuiltinId` plus the overload
    /// index resolved by the type checker. Backends dispatch using
    /// `def.overloads()[overload_idx].lowering`.
    Builtin {
        id: crate::builtins::BuiltinId,
        overload_idx: usize,
    },
    /// Symbol-table reference — locals, user-defined functions,
    /// prelude functions, top-level constants.
    Symbol(SymbolId),
}

/// Resolve a `Var`-position term to its `BuiltinId`, if any. Handles
/// `VarRef::Builtin { id, .. }` directly and looks up `VarRef::Symbol`
/// by name in the catalog. Returns `None` for non-`Var` terms and for
/// symbol references that don't name a catalog entry.
pub fn var_term_builtin_id(term: &Term, symbols: &SymbolTable) -> Option<crate::builtins::BuiltinId> {
    match &term.kind {
        TermKind::Var(VarRef::Builtin { id, .. }) => Some(*id),
        TermKind::Var(VarRef::Symbol(sym)) => {
            let name = symbols.get(*sym)?;
            crate::builtins::catalog().lookup_by_any_name(name).map(|def| def.id)
        }
        _ => None,
    }
}

/// Like `var_term_matches_name`, but returns the matching name (if
/// any). Useful when the caller needs to dispatch on one of several
/// possible names — e.g. `intrinsic_aliasing_arg` in
/// `tlc/ownership.rs`.
pub fn var_term_canonical_name<'a>(term: &'a Term, symbols: &'a SymbolTable) -> Option<&'a str> {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => symbols.get(*sym).map(|s| s.as_str()),
        TermKind::Var(VarRef::Builtin { id, .. }) => {
            // Prefer impl_source_names[0] — that's the form historically
            // used by name-keyed dispatch (`_w_intrinsic_*`). Fall back
            // to surface_name when impl_source_names is empty (per-type
            // ops, compiler-internal entries).
            let def = crate::builtins::by_id(*id);
            def.impl_source_names().first().copied().or(Some(def.raw.surface_name))
        }
        _ => None,
    }
}

/// The kind of term.
#[derive(Debug, Clone)]
pub enum TermKind {
    /// Variable reference.
    Var(VarRef),

    /// Binary operator as a value: +, -, *, /, ==, etc.
    BinOp(ast::BinaryOp),

    /// Unary operator as a value: -, !
    UnOp(ast::UnaryOp),

    /// Lambda abstraction (structured).
    Lambda(Lambda),

    /// Application: f(a, b, c) — always fully applied.
    App {
        func: Box<Term>,
        args: Vec<Term>,
    },

    /// Let binding: let x:T = rhs in body
    Let {
        name: SymbolId,
        name_ty: Type<TypeName>,
        rhs: Box<Term>,
        body: Box<Term>,
    },

    /// Integer literal.
    IntLit(String),

    /// Float literal.
    FloatLit(f32),

    /// Boolean literal.
    BoolLit(bool),

    /// External function reference (linked SPIR-V).
    /// The string is the linkage name for spirv-link.
    /// The Wyn-visible name comes from the parent Def.
    Extern(String),

    /// Conditional: if cond then t else e
    If {
        cond: Box<Term>,
        then_branch: Box<Term>,
        else_branch: Box<Term>,
    },

    /// Loop construct (mirrors MIR::Loop).
    Loop {
        /// The loop accumulator variable name.
        loop_var: SymbolId,
        /// Type of the loop variable.
        loop_var_ty: Type<TypeName>,
        /// Initial value for the accumulator.
        init: Box<Term>,
        /// Bindings that extract from loop_var (e.g., for tuple destructuring).
        /// Each is (name, type, extraction_expr).
        init_bindings: Vec<(SymbolId, Type<TypeName>, Term)>,
        /// The kind of loop.
        kind: LoopKind,
        /// Loop body expression.
        body: Box<Term>,
    },

    /// First-class SOAC (second-order array combinator).
    Soac(SoacOp),

    /// Array producer expression.
    ArrayExpr(ArrayExpr),

    /// Materialization barrier — forces an array expression to be computed.
    Force(Box<Term>),

    /// Tuple constructor: `(a, b, c)` or record literal in tuple form.
    Tuple(Vec<Term>),

    /// Tuple projection: `t.idx`. `idx` is the structural field index.
    TupleProj {
        tuple: Box<Term>,
        idx: usize,
    },

    /// Array indexing: `arr[idx]`. The result aliases `array` for
    /// ownership analysis.
    Index {
        array: Box<Term>,
        index: Box<Term>,
    },

    /// Vector literal: `@[x, y, z, w]`.
    VecLit(Vec<Term>),
}

/// The kind of loop (mirrors MIR::LoopKind).
#[derive(Debug, Clone)]
pub enum LoopKind {
    /// For loop over an array: `for x in arr`.
    For {
        var: SymbolId,
        var_ty: Type<TypeName>,
        iter: Box<Term>,
    },
    /// For loop with range bound: `for i < n`.
    ForRange {
        var: SymbolId,
        var_ty: Type<TypeName>,
        bound: Box<Term>,
    },
    /// While loop: `while cond`.
    While {
        cond: Box<Term>,
    },
}

// =============================================================================
// SOAC Types
// =============================================================================

/// A structured lambda with explicit parameters and return type.
#[derive(Debug, Clone)]
pub struct Lambda {
    pub params: Vec<(SymbolId, Type<TypeName>)>,
    pub body: Box<Term>,
    pub ret_ty: Type<TypeName>,
}

/// Lambda + captures bundle used as a SOAC envelope or `Generate` body.
/// The captures are the closure environment threaded into the body at
/// SOAC lowering time. Empty before defunctionalization; filled after.
#[derive(Debug, Clone)]
pub struct SoacBody {
    pub lam: Lambda,
    pub captures: Vec<(SymbolId, Type<TypeName>, Term)>,
}

/// A symbolic dimension expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    Const(i64),
    Var(DimVarId),
    Add(Box<Dim>, Box<Dim>),
    Sub(Box<Dim>, Box<Dim>),
}

/// A dimension variable identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DimVarId(pub u32);

/// A shape is a list of dimensions.
#[derive(Clone, Debug)]
pub struct Shape(pub Vec<Dim>);

/// An array-producing expression.
//
// Two variants are phase-scoped — they're only valid in part of the
// pipeline, with the boundaries enforced at runtime via `unreachable!`
// in passes that can't see them:
//
// * `Zip` is constructed in `transform_soac_zip`, absorbed at
//   `transform_soac_map`, and anything escaping as a standalone term is
//   rewritten to `_w_tuple(...)` by `tlc::soa`. Post-SoA passes don't
//   meaningfully encounter it.
//
// * `StorageBuffer` is introduced by `tlc::buffer_specialize`.
//   Pre-buffer_specialize passes don't see it.
//
// A future refactor could narrow these via per-phase enum splits
// (`PreSoaArrayExpr` / `PostSoaArrayExpr`, etc.) so the invariants are
// type-enforced rather than runtime-checked. The cost is non-trivial —
// `Term`/`TermKind`/`SoacOp`/`Lambda` would need a phase parameter, so
// it cascades across the whole TLC pipeline. The current runtime checks
// are fine until that pays for itself.
#[derive(Debug, Clone)]
pub enum ArrayExpr {
    /// A TLC term producing an array value.
    Ref(Box<Term>),
    /// Logical zip — not materialized, consumed by enclosing Map.
    Zip(Vec<ArrayExpr>),
    /// SOAC producing an array.
    Soac(Box<SoacOp>),
    /// Generator: `elem = index_fn(i)` for `i` in `0..shape`.
    Generate {
        shape: Shape,
        index_fn: SoacBody,
        elem_ty: Type<TypeName>,
    },
    /// Literal small array.
    Literal(Vec<Term>),
    /// Range / iota.
    Range {
        start: Box<Term>,
        len: Box<Term>,
    },
    /// Storage buffer reference (introduced by buffer_specialize).
    /// Represents elements from a storage buffer at (set, binding),
    /// starting at `offset` for `len` elements.
    StorageBuffer {
        set: u32,
        binding: u32,
        offset: Box<Term>,
        len: Box<Term>,
        elem_ty: Type<TypeName>,
    },
}

/// A second-order array combinator (SOAC) operation.
#[derive(Debug, Clone)]
pub enum SoacOp {
    Map {
        lam: SoacBody,
        /// Parallel inputs. `inputs.len() == lam.lam.params.len()`.
        inputs: Vec<ArrayExpr>,
        /// Set by the ownership pass when the map's primary input
        /// is mutable, dead-after, pointwise, and not bound to a
        /// compute-shader output. Read by `egir::from_tlc` to lower
        /// the map to an in-place loop instead of allocating a
        /// fresh output buffer.
        consumes_input: bool,
    },
    Reduce {
        op: SoacBody,
        ne: Box<Term>,
        input: ArrayExpr,
        props: ReduceProps,
    },
    /// Fused map+reduce: `op(acc, x1, ..., xn) -> acc'` over parallel inputs.
    /// Produced by fusion when a Map feeds directly into a Reduce.
    /// Lowered as a single loop without materializing the intermediate array.
    Redomap {
        /// Combined operator: `(acc, x1, ..., xn) -> acc'`
        /// First param is the accumulator, rest are elements from each input.
        op: SoacBody,
        /// Pure reduce combiner: `(acc, acc) -> acc` for parallel phase 2.
        reduce_op: SoacBody,
        /// Initial accumulator value.
        ne: Box<Term>,
        /// Parallel input arrays (one per element param in op).
        inputs: Vec<ArrayExpr>,
        props: ReduceProps,
    },
    Scan {
        op: SoacBody,
        ne: Box<Term>,
        input: ArrayExpr,
    },
    Filter {
        pred: SoacBody,
        input: ArrayExpr,
    },
    Scatter {
        dest: Place,
        indices: ArrayExpr,
        values: ArrayExpr,
    },
    ReduceByIndex {
        dest: Place,
        op: SoacBody,
        ne: Box<Term>,
        indices: ArrayExpr,
        values: ArrayExpr,
        props: ReduceProps,
    },
}

/// Destination-passing for scatter / reduce_by_index.
#[derive(Debug, Clone)]
pub enum Place {
    BufferSlice {
        base: Box<Term>,
        offset: Box<Term>,
        shape: Shape,
        elem_ty: Type<TypeName>,
    },
    LocalArray {
        id: SymbolId,
        shape: Shape,
        elem_ty: Type<TypeName>,
    },
}

/// Properties of a reduction operator.
#[derive(Debug, Clone)]
pub struct ReduceProps {
    pub commutative: bool,
    pub associative: bool,
    pub requires_atomic: bool,
}

impl Default for ReduceProps {
    fn default() -> Self {
        Self {
            commutative: false,
            associative: false,
            requires_atomic: false,
        }
    }
}

// =============================================================================
// TLC Program
// =============================================================================

/// Metadata about how a definition should be lowered to MIR.
#[derive(Debug, Clone)]
pub enum DefMeta {
    /// A regular function or constant.
    Function,
    /// A shader entry point - stores the original AST entry for metadata.
    EntryPoint(Box<interface::EntryDecl>),
}

/// A top-level definition in TLC.
#[derive(Debug, Clone)]
pub struct Def {
    pub name: SymbolId,
    pub ty: Type<TypeName>,
    pub body: Term,
    pub meta: DefMeta,
    /// Number of arguments this function expects.
    pub arity: usize,
}

/// A TLC program (collection of definitions).
#[derive(Debug, Clone)]
pub struct Program {
    pub defs: Vec<Def>,
    /// Uniform declarations (no bodies, just metadata).
    pub uniforms: Vec<interface::UniformDecl>,
    /// Storage buffer declarations (no bodies, just metadata).
    pub storage: Vec<interface::StorageDecl>,
    /// Symbol table: maps SymbolId to original name (for errors/debugging).
    pub symbols: SymbolTable,
    /// Canonical function name → def SymbolId mapping.
    /// Used by fusion to resolve call-site SymbolIds to def SymbolIds.
    pub def_syms: HashMap<String, SymbolId>,
}

impl Program {
    /// Assert no def body contains nested Apps.
    pub fn assert_flat_apps(&self) {
        for def in &self.defs {
            let name = self.symbols.get(def.name).cloned().unwrap_or_else(|| format!("{:?}", def.name));
            def.body.assert_flat_apps_in(&name);
        }
    }
}

/// Parts of a TLC program, without the symbol table.
/// Returned by `Transformer::transform_program` when the caller owns the symbol table.
#[derive(Debug, Clone)]
pub struct ProgramParts {
    pub defs: Vec<Def>,
    pub uniforms: Vec<interface::UniformDecl>,
    pub storage: Vec<interface::StorageDecl>,
}

impl ProgramParts {
    /// Combine with a symbol table to create a complete Program.
    pub fn with_symbols(self, symbols: SymbolTable, def_syms: HashMap<String, SymbolId>) -> Program {
        Program {
            defs: self.defs,
            uniforms: self.uniforms,
            storage: self.storage,
            symbols,
            def_syms,
        }
    }
}

impl Program {
    /// Rebuild a Program, carrying def_syms through.
    pub fn rebuild(self, defs: Vec<Def>, symbols: SymbolTable) -> Program {
        Program {
            defs,
            uniforms: self.uniforms,
            storage: self.storage,
            symbols,
            def_syms: self.def_syms,
        }
    }
}

// =============================================================================
// Generic child traversal
// =============================================================================

impl Term {
    /// If this term is `App { func: Var(sym), args }` — a direct named
    /// call — return `Some((sym, args))`. Returns `None` for operator
    /// dispatch (`App { BinOp/UnOp/Extern, .. }`), partial applications,
    /// non-call terms, etc.
    ///
    /// Use this in post-defunctionalize passes and backends to one-step
    /// destructure named calls instead of nesting two `match`es.
    pub fn as_direct_call(&self) -> Option<(SymbolId, &[Term])> {
        match &self.kind {
            TermKind::App { func, args } => match &func.kind {
                TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => Some((*sym, args.as_slice())),
                _ => None,
            },
            _ => None,
        }
    }

    /// Assert that no App node in this tree has a func that is itself an App.
    pub fn assert_flat_apps(&self) {
        self.assert_flat_apps_in("<unknown>");
    }

    fn assert_flat_apps_in(&self, def_name: &str) {
        if let TermKind::App { func, args } = &self.kind {
            if let TermKind::App { args: inner_args, .. } = &func.kind {
                panic!(
                    "Nested App detected in def '{}': outer has {} args, inner func has {} args. \
                     Inner func kind: {:?}",
                    def_name,
                    args.len(),
                    inner_args.len(),
                    if let TermKind::App { func: f, .. } = &func.kind {
                        format!("App(func={:?})", std::mem::discriminant(&f.kind))
                    } else {
                        format!("{:?}", std::mem::discriminant(&func.kind))
                    }
                );
            }
        }
        self.for_each_child(&mut |child| child.assert_flat_apps_in(def_name));
    }

    /// Apply `f` to every immediate `Term` child, returning a new `Term` with
    /// the same metadata but transformed children. Recurses into Lambda,
    /// SoacOp, ArrayExpr, LoopKind, and Place sub-structures.
    ///
    /// This is the single place that knows the shape of TermKind — passes
    /// that need a uniform bottom-up or top-down walk can use this instead
    /// of hand-rolling a match over every variant.
    pub fn map_children<F>(self, f: &mut F) -> Self
    where
        F: FnMut(Term) -> Term,
    {
        let kind = match self.kind {
            // Leaves — no Term children
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::Extern(_) => self.kind,

            TermKind::App { func, args } => TermKind::App {
                func: Box::new(f(*func)),
                args: args.into_iter().map(&mut *f).collect(),
            },

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => TermKind::Let {
                name,
                name_ty,
                rhs: Box::new(f(*rhs)),
                body: Box::new(f(*body)),
            },

            TermKind::Lambda(lam) => TermKind::Lambda(map_lambda_children(lam, f)),

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TermKind::If {
                cond: Box::new(f(*cond)),
                then_branch: Box::new(f(*then_branch)),
                else_branch: Box::new(f(*else_branch)),
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => TermKind::Loop {
                loop_var,
                loop_var_ty,
                init: Box::new(f(*init)),
                init_bindings: init_bindings.into_iter().map(|(s, t, e)| (s, t, f(e))).collect(),
                kind: map_loop_kind_children(kind, f),
                body: Box::new(f(*body)),
            },

            TermKind::Soac(soac) => TermKind::Soac(map_soac_children(soac, f)),

            TermKind::ArrayExpr(ae) => TermKind::ArrayExpr(map_array_expr_children(ae, f)),

            TermKind::Force(inner) => TermKind::Force(Box::new(f(*inner))),

            TermKind::Tuple(parts) => TermKind::Tuple(parts.into_iter().map(&mut *f).collect()),

            TermKind::TupleProj { tuple, idx } => TermKind::TupleProj {
                tuple: Box::new(f(*tuple)),
                idx,
            },

            TermKind::Index { array, index } => TermKind::Index {
                array: Box::new(f(*array)),
                index: Box::new(f(*index)),
            },

            TermKind::VecLit(parts) => TermKind::VecLit(parts.into_iter().map(&mut *f).collect()),
        };

        Term { kind, ..self }
    }

    /// Visit every immediate `Term` child by reference. This is the by-ref
    /// counterpart to `map_children` — use it for analysis passes that
    /// inspect without transforming.
    pub fn for_each_child<F>(&self, f: &mut F)
    where
        F: FnMut(&Term),
    {
        match &self.kind {
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::Extern(_) => {}

            TermKind::App { func, args } => {
                f(func);
                for a in args {
                    f(a);
                }
            }

            TermKind::Let { rhs, body, .. } => {
                f(rhs);
                f(body);
            }

            TermKind::Lambda(lam) => visit_lambda_children(lam, f),

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                f(cond);
                f(then_branch);
                f(else_branch);
            }

            TermKind::Loop {
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                f(init);
                for (_, _, e) in init_bindings {
                    f(e);
                }
                visit_loop_kind_children(kind, f);
                f(body);
            }

            TermKind::Soac(soac) => visit_soac_children(soac, f),
            TermKind::ArrayExpr(ae) => visit_array_expr_children(ae, f),
            TermKind::Force(inner) => f(inner),

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                for p in parts {
                    f(p);
                }
            }
            TermKind::TupleProj { tuple, .. } => f(tuple),
            TermKind::Index { array, index } => {
                f(array);
                f(index);
            }
        }
    }
}

fn visit_lambda_children<F>(lam: &Lambda, f: &mut F)
where
    F: FnMut(&Term),
{
    f(&lam.body);
}

fn visit_soac_body_children<F>(sb: &SoacBody, f: &mut F)
where
    F: FnMut(&Term),
{
    visit_lambda_children(&sb.lam, f);
    for (_, _, e) in &sb.captures {
        f(e);
    }
}

fn visit_soac_children<F>(soac: &SoacOp, f: &mut F)
where
    F: FnMut(&Term),
{
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            visit_soac_body_children(lam, f);
            for ae in inputs {
                visit_array_expr_children(ae, f);
            }
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(input, f);
        }
        SoacOp::Scan { op, ne, input } => {
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(input, f);
        }
        SoacOp::Filter { pred, input } => {
            visit_soac_body_children(pred, f);
            visit_array_expr_children(input, f);
        }
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => {
            visit_place_children(dest, f);
            visit_array_expr_children(indices, f);
            visit_array_expr_children(values, f);
        }
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            ..
        } => {
            visit_place_children(dest, f);
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(indices, f);
            visit_array_expr_children(values, f);
        }
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
            ..
        } => {
            visit_soac_body_children(op, f);
            visit_soac_body_children(reduce_op, f);
            f(ne);
            for ae in inputs {
                visit_array_expr_children(ae, f);
            }
        }
    }
}

fn visit_array_expr_children<F>(ae: &ArrayExpr, f: &mut F)
where
    F: FnMut(&Term),
{
    match ae {
        ArrayExpr::Ref(t) => f(t),
        ArrayExpr::Zip(aes) => {
            for ae in aes {
                visit_array_expr_children(ae, f);
            }
        }
        ArrayExpr::Soac(op) => visit_soac_children(op, f),
        ArrayExpr::Generate { index_fn, .. } => visit_soac_body_children(index_fn, f),
        ArrayExpr::Literal(terms) => {
            for t in terms {
                f(t);
            }
        }
        ArrayExpr::Range { start, len } => {
            f(start);
            f(len);
        }
        ArrayExpr::StorageBuffer { offset, len, .. } => {
            f(offset);
            f(len);
        }
    }
}

fn visit_loop_kind_children<F>(kind: &LoopKind, f: &mut F)
where
    F: FnMut(&Term),
{
    match kind {
        LoopKind::For { iter, .. } => f(iter),
        LoopKind::ForRange { bound, .. } => f(bound),
        LoopKind::While { cond } => f(cond),
    }
}

fn visit_place_children<F>(place: &Place, f: &mut F)
where
    F: FnMut(&Term),
{
    match place {
        Place::BufferSlice { base, offset, .. } => {
            f(base);
            f(offset);
        }
        Place::LocalArray { .. } => {}
    }
}

fn map_lambda_children<F>(lam: Lambda, f: &mut F) -> Lambda
where
    F: FnMut(Term) -> Term,
{
    Lambda {
        body: Box::new(f(*lam.body)),
        ..lam
    }
}

fn map_soac_body_children<F>(sb: SoacBody, f: &mut F) -> SoacBody
where
    F: FnMut(Term) -> Term,
{
    SoacBody {
        lam: map_lambda_children(sb.lam, f),
        captures: sb.captures.into_iter().map(|(s, t, e)| (s, t, f(e))).collect(),
    }
}

fn map_soac_children<F>(soac: SoacOp, f: &mut F) -> SoacOp
where
    F: FnMut(Term) -> Term,
{
    match soac {
        SoacOp::Map {
            lam,
            inputs,
            consumes_input,
        } => SoacOp::Map {
            lam: map_soac_body_children(lam, f),
            inputs: inputs.into_iter().map(|ae| map_array_expr_children(ae, f)).collect(),
            consumes_input,
        },
        SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            input: map_array_expr_children(input, f),
            props,
        },
        SoacOp::Scan { op, ne, input } => SoacOp::Scan {
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            input: map_array_expr_children(input, f),
        },
        SoacOp::Filter { pred, input } => SoacOp::Filter {
            pred: map_soac_body_children(pred, f),
            input: map_array_expr_children(input, f),
        },
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => SoacOp::Scatter {
            dest: map_place_children(dest, f),
            indices: map_array_expr_children(indices, f),
            values: map_array_expr_children(values, f),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            props,
        } => SoacOp::ReduceByIndex {
            dest: map_place_children(dest, f),
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            indices: map_array_expr_children(indices, f),
            values: map_array_expr_children(values, f),
            props,
        },
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
            props,
        } => SoacOp::Redomap {
            op: map_soac_body_children(op, f),
            reduce_op: map_soac_body_children(reduce_op, f),
            ne: Box::new(f(*ne)),
            inputs: inputs.into_iter().map(|ae| map_array_expr_children(ae, f)).collect(),
            props,
        },
    }
}

fn map_array_expr_children<F>(ae: ArrayExpr, f: &mut F) -> ArrayExpr
where
    F: FnMut(Term) -> Term,
{
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(f(*t))),
        ArrayExpr::Zip(aes) => {
            ArrayExpr::Zip(aes.into_iter().map(|ae| map_array_expr_children(ae, f)).collect())
        }
        ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(map_soac_children(*op, f))),
        ArrayExpr::Generate {
            shape,
            index_fn,
            elem_ty,
        } => ArrayExpr::Generate {
            shape,
            index_fn: map_soac_body_children(index_fn, f),
            elem_ty,
        },
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(terms.into_iter().map(f).collect()),
        ArrayExpr::Range { start, len } => ArrayExpr::Range {
            start: Box::new(f(*start)),
            len: Box::new(f(*len)),
        },
        ArrayExpr::StorageBuffer {
            set,
            binding,
            offset,
            len,
            elem_ty,
        } => ArrayExpr::StorageBuffer {
            set,
            binding,
            offset: Box::new(f(*offset)),
            len: Box::new(f(*len)),
            elem_ty,
        },
    }
}

fn map_loop_kind_children<F>(kind: LoopKind, f: &mut F) -> LoopKind
where
    F: FnMut(Term) -> Term,
{
    match kind {
        LoopKind::For { var, var_ty, iter } => LoopKind::For {
            var,
            var_ty,
            iter: Box::new(f(*iter)),
        },
        LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
            var,
            var_ty,
            bound: Box::new(f(*bound)),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: Box::new(f(*cond)),
        },
    }
}

fn map_place_children<F>(place: Place, f: &mut F) -> Place
where
    F: FnMut(Term) -> Term,
{
    match place {
        Place::BufferSlice {
            base,
            offset,
            shape,
            elem_ty,
        } => Place::BufferSlice {
            base: Box::new(f(*base)),
            offset: Box::new(f(*offset)),
            shape,
            elem_ty,
        },
        Place::LocalArray { .. } => place,
    }
}

// =============================================================================
// AST to TLC Transformation
// =============================================================================

/// A pending let-binding to be applied after all lambdas are created.
#[derive(Debug, Clone)]
struct PendingBinding {
    name: SymbolId,
    ty: Type<TypeName>,
    expr: Term,
}

/// Flattened-no-sharing layout for a structural sum type. Computed
/// once per sum and then consulted by the Constructor and Match
/// transforms for tag values and per-payload slot offsets.
struct SumLayout {
    /// All slot types of the lowered tuple, lowered. Index 0 is the
    /// u32 tag; indices 1.. are the variant payloads concatenated
    /// in source order.
    slot_types: Vec<Type<TypeName>>,
    /// For each constructor name: its tag value (source-order index)
    /// and the starting slot index of its payload in `slot_types`.
    constructor_info: std::collections::HashMap<String, (u32, usize)>,
}

/// Context for transforming AST to TLC.
pub struct Transformer<'a> {
    type_table: &'a TypeTable,
    term_ids: TermIdSource,
    /// Shared symbol table: maps SymbolId to original name (for errors/debugging).
    symbols: &'a mut SymbolTable,
    /// Current scope for name resolution: maps string name to SymbolId.
    scope: HashMap<String, SymbolId>,
    /// Top-level symbols that persist across function transformations.
    /// This ensures function references use the same SymbolId as the Def.
    /// Shared across all transformers via mutable reference.
    top_level_symbols: &'a mut HashMap<String, SymbolId>,
    /// Side table from name resolution: AST NodeId → BuiltinId for
    /// catalog-resolved identifiers. Lets `Var`-position idents be
    /// classified as `VarRef::Builtin(id)` directly without round-
    /// tripping through name strings.
    name_resolution: &'a crate::name_resolution::NameResolution,
    /// Optional namespace prefix for definition names (e.g., "f32" -> "f32.pi")
    namespace: Option<String>,
    /// Shared placeholder symbol for pattern matching scrutinees.
    /// Allocated once and reused to avoid polluting the symbol table.
    placeholder_sym: SymbolId,
    /// When true, `ExprKind::TypeHole` nodes are replaced with a
    /// default-valued term of the hole's inferred type. When false,
    /// TypeHole nodes are invariant-rejected upstream at the
    /// type-check boundary (`TypeChecked::reject_type_holes`), so
    /// reaching one here is a bug — the arm panics.
    fill_holes: bool,
    /// Errors surfaced while defaulting `???` type holes. Populated
    /// only when `fill_holes` is true and a hole's inferred type
    /// can't be default-filled (e.g. a function type or an
    /// unresolved type variable). Owned by the caller so errors
    /// from every Transformer that runs during `to_tlc` accumulate
    /// into one list.
    fill_hole_errors: &'a mut Vec<crate::error::CompilerError>,
}

impl<'a> Transformer<'a> {
    pub fn new(
        type_table: &'a TypeTable,
        symbols: &'a mut SymbolTable,
        top_level_symbols: &'a mut HashMap<String, SymbolId>,
        name_resolution: &'a crate::name_resolution::NameResolution,
        fill_holes: bool,
        fill_hole_errors: &'a mut Vec<crate::error::CompilerError>,
    ) -> Self {
        let placeholder_sym = symbols.alloc("_w_placeholder".to_string());
        Self {
            type_table,
            term_ids: TermIdSource::new(),
            symbols,
            scope: HashMap::new(),
            top_level_symbols,
            name_resolution,
            namespace: None,
            placeholder_sym,
            fill_holes,
            fill_hole_errors,
        }
    }

    /// Create a transformer with a namespace prefix for definition names.
    pub fn with_namespace(
        type_table: &'a TypeTable,
        symbols: &'a mut SymbolTable,
        top_level_symbols: &'a mut HashMap<String, SymbolId>,
        name_resolution: &'a crate::name_resolution::NameResolution,
        namespace: &str,
        fill_holes: bool,
        fill_hole_errors: &'a mut Vec<crate::error::CompilerError>,
    ) -> Self {
        let placeholder_sym = symbols.alloc("_w_placeholder".to_string());
        Self {
            type_table,
            term_ids: TermIdSource::new(),
            symbols,
            scope: HashMap::new(),
            top_level_symbols,
            name_resolution,
            namespace: Some(namespace.to_string()),
            placeholder_sym,
            fill_holes,
            fill_hole_errors,
        }
    }

    /// Define a new symbol, returning its unique ID.
    /// This adds the name to the symbol table and current scope.
    fn define(&mut self, name: &str) -> SymbolId {
        let id = self.symbols.alloc(name.to_string());
        self.scope.insert(name.to_string(), id);
        id
    }

    /// Resolve a name to its SymbolId in current scope.
    /// Returns None for unbound names (e.g., intrinsics, top-level functions).
    fn resolve(&self, name: &str) -> Option<SymbolId> {
        self.scope.get(name).copied()
    }

    /// Resolve a name to its SymbolId.
    /// Checks local scope, then top-level symbols. Compiler-internal names (`_w_*`)
    /// are allocated on demand; all other names must be pre-registered via NameRegistry.
    fn resolve_or_define(&mut self, name: &str) -> SymbolId {
        if let Some(id) = self.resolve(name) {
            id
        } else if let Some(&id) = self.top_level_symbols.get(name) {
            id
        } else if name.starts_with("_w_") {
            let id = self.symbols.alloc(name.to_string());
            self.top_level_symbols.insert(name.to_string(), id);
            id
        } else {
            panic!("ICE: unresolved name '{}' in TLC transform", name);
        }
    }

    /// Check if a name is locally bound (for SOAC renaming check).
    fn is_locally_bound(&self, name: &str) -> bool {
        self.scope.contains_key(name)
    }

    /// Transform an AST program to TLC.
    /// Returns program parts without the symbol table - caller must combine with
    /// their owned symbol table using `ProgramParts::with_symbols`.
    pub fn transform_program(&mut self, program: &ast::Program) -> ProgramParts {
        // First pass: register all top-level function names so that references
        // within function bodies use the same SymbolId as the Def.
        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    let name_str = match &self.namespace {
                        Some(ns) => format!("{}.{}", ns, d.name),
                        None => d.name.clone(),
                    };
                    let sym = self.symbols.alloc(name_str.clone());
                    self.top_level_symbols.insert(name_str, sym);
                }
                ast::Declaration::Entry(e) => {
                    let sym = self.symbols.alloc(e.name.clone());
                    self.top_level_symbols.insert(e.name.clone(), sym);
                }
                ast::Declaration::Extern(e) => {
                    let sym = self.symbols.alloc(e.name.clone());
                    self.top_level_symbols.insert(e.name.clone(), sym);
                }
                _ => {}
            }
        }

        // Second pass: transform function bodies (now resolve_or_define can find top-level symbols)
        let mut defs = Vec::new();
        let mut uniforms = Vec::new();
        let mut storage = Vec::new();

        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    if let Some(def) = self.transform_decl(d) {
                        defs.push(def);
                    }
                }
                ast::Declaration::Entry(e) => {
                    if let Some(def) = self.transform_entry(e) {
                        defs.push(def);
                    }
                }
                ast::Declaration::Uniform(u) => {
                    uniforms.push(u.clone());
                }
                ast::Declaration::Storage(s) => {
                    storage.push(s.clone());
                }
                ast::Declaration::Extern(e) => {
                    // Use the pre-registered symbol
                    let name_sym =
                        *self.top_level_symbols.get(&e.name).expect("BUG: extern not pre-registered");
                    let body = self.mk_term(e.ty.clone(), e.span, TermKind::Extern(e.linkage_name.clone()));
                    let arity = count_function_arity(&e.ty);
                    defs.push(Def {
                        name: name_sym,
                        ty: e.ty.clone(),
                        body,
                        meta: DefMeta::Function,
                        arity,
                    });
                }
                ast::Declaration::Sig(_)
                | ast::Declaration::TypeBind(_)
                | ast::Declaration::Module(_)
                | ast::Declaration::ModuleTypeBind(_)
                | ast::Declaration::Open(_)
                | ast::Declaration::Import(_) => {}
            }
        }

        ProgramParts {
            defs,
            uniforms,
            storage,
        }
    }

    pub fn transform_decl(&mut self, decl: &ast::Decl) -> Option<Def> {
        // Clear scope for each definition to ensure fresh scope
        self.scope.clear();
        let body_ty = self.lookup_type(decl.body.h.id)?;
        let full_ty = self.build_function_type(&decl.params, &body_ty);
        let body = self.transform_with_params(&decl.params, &decl.body, full_ty.clone());

        // Apply namespace prefix if set (e.g., "f32" + "pi" -> "f32.pi")
        let name_str = match &self.namespace {
            Some(ns) => format!("{}.{}", ns, decl.name),
            None => decl.name.clone(),
        };

        // Use pre-registered symbol if available, otherwise allocate and register a new one.
        // Always ensure the symbol is in top_level_symbols for later transformers.
        let name_sym = if let Some(&sym) = self.top_level_symbols.get(&name_str) {
            sym
        } else {
            let sym = self.symbols.alloc(name_str.clone());
            self.top_level_symbols.insert(name_str, sym);
            sym
        };

        Some(Def {
            name: name_sym,
            ty: full_ty,
            body,
            meta: DefMeta::Function,
            arity: decl.params.len(),
        })
    }

    fn transform_entry(&mut self, entry: &interface::EntryDecl) -> Option<Def> {
        // Clear scope for each entry to ensure fresh scope
        self.scope.clear();
        let body_ty = self.lookup_type(entry.body.h.id)?;
        let full_ty = self.build_function_type(&entry.params, &body_ty);
        let body = self.transform_with_params(&entry.params, &entry.body, full_ty.clone());

        // Use pre-registered symbol if available, otherwise allocate and register a new one.
        let name_sym = if let Some(&sym) = self.top_level_symbols.get(&entry.name) {
            sym
        } else {
            let sym = self.symbols.alloc(entry.name.clone());
            self.top_level_symbols.insert(entry.name.clone(), sym);
            sym
        };

        Some(Def {
            name: name_sym,
            ty: full_ty,
            body,
            meta: DefMeta::EntryPoint(Box::new(entry.clone())),
            arity: entry.params.len(),
        })
    }

    fn build_function_type(&self, params: &[ast::Pattern], ret_ty: &Type<TypeName>) -> Type<TypeName> {
        let mut ty = ret_ty.clone();

        for param in params.iter().rev() {
            let param_ty = self.pattern_type(param);
            ty = Type::Constructed(TypeName::Arrow, vec![param_ty, ty]);
        }

        ty
    }

    fn pattern_type(&self, pattern: &ast::Pattern) -> Type<TypeName> {
        match &pattern.kind {
            // For attributed patterns, recurse into the inner pattern
            ast::PatternKind::Attributed(_, inner) => self.pattern_type(inner),
            // Always look up from type_table - the type checker has substituted UserVars
            // with Type::Variables. Using the AST type directly would retain UserVars.
            _ => self.lookup_type(pattern.h.id).expect("Pattern must have type in type table"),
        }
    }

    fn transform_with_params(
        &mut self,
        params: &[ast::Pattern],
        body: &ast::Expression,
        full_ty: Type<TypeName>,
    ) -> Term {
        let span = params.first().map(|p| p.h.span).unwrap_or(body.h.span);
        self.build_lambda_chain(params, body, full_ty, span)
    }

    /// Build a chain of nested lambdas from patterns, deferring all let-bindings
    /// until after all lambdas are created. This ensures no let-bindings appear
    /// between nested lambdas, which is important for consistent capture analysis.
    fn build_lambda_chain(
        &mut self,
        params: &[ast::Pattern],
        body: &ast::Expression,
        full_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if params.is_empty() {
            return self.transform_expr(body);
        }

        // Collect all lambda parameters and their pending bindings
        // compute_pattern_bindings already creates SymbolIds via define()
        let mut lambda_info: Vec<(SymbolId, Type<TypeName>, Vec<PendingBinding>)> = Vec::new();
        let mut current_ty = full_ty;

        // Use the shared placeholder symbol for the scrutinee in compute_pattern_bindings
        let placeholder_sym = self.placeholder_sym;

        for param in params {
            let param_ty = self.get_param_type(&current_ty);

            // Use a placeholder scrutinee - we need to call compute_pattern_bindings to get
            // the param name and projection bindings, but the actual lambda param value
            // won't exist until runtime
            let placeholder = self.mk_term(
                param_ty.clone(),
                span,
                TermKind::Var(crate::tlc::VarRef::Symbol(placeholder_sym)),
            );
            let (param_sym, mut bindings) = self.compute_pattern_bindings(param, placeholder, span);

            // For complex patterns (Tuple/Record), compute_pattern_bindings returns bindings that
            // include the top-level binding (fresh = scrutinee). For lambdas, we don't want this
            // since the lambda param IS the fresh name. Skip the first binding if it matches.
            if !bindings.is_empty() && bindings[0].name == param_sym {
                bindings.remove(0);
            }

            lambda_info.push((param_sym, param_ty.clone(), bindings));
            current_ty = self.get_body_type(&current_ty);
        }

        // Transform the body expression
        let mut result = self.transform_expr(body);

        // Apply all bindings in reverse order (innermost first, so outermost ends up innermost)
        for (_, _, bindings) in lambda_info.iter().rev() {
            for binding in bindings.iter().rev() {
                result = self.mk_term(
                    result.ty.clone(),
                    span,
                    TermKind::Let {
                        name: binding.name,
                        name_ty: binding.ty.clone(),
                        rhs: Box::new(binding.expr.clone()),
                        body: Box::new(result),
                    },
                );
            }
        }

        // Build a single flat lambda with all params
        let all_params: Vec<(SymbolId, Type<TypeName>)> =
            lambda_info.into_iter().map(|(sym, ty, _)| (sym, ty)).collect();
        let ret_ty = result.ty.clone();
        let lam_ty = {
            let mut ty = ret_ty.clone();
            for (_, param_ty) in all_params.iter().rev() {
                ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), ty]);
            }
            ty
        };
        result = self.mk_term(
            lam_ty,
            span,
            TermKind::Lambda(Lambda {
                params: all_params,
                ret_ty,
                body: Box::new(result),
            }),
        );

        result
    }

    /// Compute bindings for a pattern matched against a scrutinee variable.
    /// Returns (bound_symbol, list_of_pending_bindings).
    ///
    /// The bound_symbol is either:
    /// - A symbol for the pattern's name (for simple Name patterns)
    /// - A fresh symbol (for complex patterns like Tuple/Record)
    ///
    /// For Name/Wildcard patterns, no bindings are returned - the caller is responsible
    /// for creating the top-level binding if needed (e.g., for let-in).
    ///
    /// For Tuple/Record patterns, bindings include:
    /// - The top-level binding (fresh_name = scrutinee)
    /// - All projection bindings
    ///
    /// This is the single source of truth for pattern → binding plan transformation.
    fn compute_pattern_bindings(
        &mut self,
        pattern: &ast::Pattern,
        scrutinee: Term,
        span: Span,
    ) -> (SymbolId, Vec<PendingBinding>) {
        self.compute_pattern_bindings_inner(pattern, scrutinee, span, true)
    }

    /// Inner implementation that tracks whether we're at the top level.
    /// At top level, Name/Wildcard don't create bindings (caller handles).
    /// Nested Name/Wildcard DO create bindings (needed for tuple component extraction).
    fn compute_pattern_bindings_inner(
        &mut self,
        pattern: &ast::Pattern,
        scrutinee: Term,
        span: Span,
        is_top_level: bool,
    ) -> (SymbolId, Vec<PendingBinding>) {
        match &pattern.kind {
            ast::PatternKind::Name(name) => {
                let sym = self.define(name);
                if is_top_level {
                    // Top-level Name: no binding needed, caller will use scrutinee directly
                    // or wrap with Let as appropriate
                    (sym, vec![])
                } else {
                    // Nested Name (e.g., inside tuple): need binding for projection result
                    let binding = PendingBinding {
                        name: sym,
                        ty: scrutinee.ty.clone(),
                        expr: scrutinee,
                    };
                    (sym, vec![binding])
                }
            }

            ast::PatternKind::Wildcard => {
                let fresh_name = format!("_w_wild_{}", self.term_ids.next_id().0);
                let sym = self.define(&fresh_name);
                if is_top_level {
                    // Top-level Wildcard: no binding needed
                    (sym, vec![])
                } else {
                    // Nested Wildcard: need binding to evaluate projection
                    let binding = PendingBinding {
                        name: sym,
                        ty: scrutinee.ty.clone(),
                        expr: scrutinee,
                    };
                    (sym, vec![binding])
                }
            }

            ast::PatternKind::Typed(inner, _) | ast::PatternKind::Attributed(_, inner) => {
                self.compute_pattern_bindings_inner(inner, scrutinee, span, is_top_level)
            }

            ast::PatternKind::Tuple(patterns) => {
                let fresh_name = format!("_w_tup_{}", self.term_ids.next_id().0);
                let fresh_sym = self.define(&fresh_name);
                let tuple_ty = scrutinee.ty.clone();
                let component_types = self.extract_tuple_types(&tuple_ty, patterns.len());

                // First bind the scrutinee to the fresh name
                let mut bindings = vec![PendingBinding {
                    name: fresh_sym,
                    ty: tuple_ty.clone(),
                    expr: scrutinee,
                }];

                // Then recursively compute bindings for each component (NOT top-level)
                for (i, sub_pattern) in patterns.iter().enumerate() {
                    let comp_ty = component_types
                        .get(i)
                        .cloned()
                        .expect("BUG: Tuple pattern has more elements than tuple type");

                    let proj = self.build_tuple_projection(fresh_sym, &tuple_ty, i, comp_ty, span);
                    let (_, sub_bindings) =
                        self.compute_pattern_bindings_inner(sub_pattern, proj, span, false);
                    bindings.extend(sub_bindings);
                }

                (fresh_sym, bindings)
            }

            ast::PatternKind::Record(fields) => {
                let fresh_name = format!("_w_rec_{}", self.term_ids.next_id().0);
                let fresh_sym = self.define(&fresh_name);
                let record_ty = scrutinee.ty.clone();
                let field_types = self.extract_record_types(&record_ty);

                // First bind the scrutinee to the fresh name
                let mut bindings = vec![PendingBinding {
                    name: fresh_sym,
                    ty: record_ty.clone(),
                    expr: scrutinee,
                }];

                // Then recursively compute bindings for each field (NOT top-level)
                for field in fields {
                    let field_ty = field_types
                        .get(&field.field)
                        .cloned()
                        .unwrap_or_else(|| panic!("BUG: Record field '{}' not found in type", field.field));

                    let field_idx = self
                        .resolve_field_index(&record_ty, &field.field)
                        .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field.field));

                    let field_access = self.build_tuple_projection(
                        fresh_sym,
                        &record_ty,
                        field_idx,
                        field_ty.clone(),
                        span,
                    );

                    if let Some(pat) = &field.pattern {
                        let (_, sub_bindings) =
                            self.compute_pattern_bindings_inner(pat, field_access, span, false);
                        bindings.extend(sub_bindings);
                    } else {
                        let field_sym = self.define(&field.field);
                        bindings.push(PendingBinding {
                            name: field_sym,
                            ty: field_ty,
                            expr: field_access,
                        });
                    }
                }

                (fresh_sym, bindings)
            }

            ast::PatternKind::Unit => {
                todo!("Unit patterns")
            }

            ast::PatternKind::Literal(_) => {
                todo!("Literal patterns in lambdas")
            }

            ast::PatternKind::Constructor(_, _) => {
                todo!("Constructor patterns in lambdas")
            }
        }
    }

    /// Build a tuple projection: _w_tuple_proj(var, index)
    fn build_tuple_projection(
        &mut self,
        var_sym: SymbolId,
        var_ty: &Type<TypeName>,
        index: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let var_term = self.mk_term(
            var_ty.clone(),
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(var_sym)),
        );
        self.mk_tuple_proj(var_term, index, result_ty, span)
    }

    /// Apply a list of bindings around a body term, creating nested let expressions.
    /// Bindings are applied in reverse order so the first binding is outermost.
    fn apply_bindings_around(&mut self, bindings: Vec<PendingBinding>, body: Term, span: Span) -> Term {
        bindings.into_iter().rev().fold(body, |acc, b| {
            self.mk_term(
                acc.ty.clone(),
                span,
                TermKind::Let {
                    name: b.name,
                    name_ty: b.ty,
                    rhs: Box::new(b.expr),
                    body: Box::new(acc),
                },
            )
        })
    }

    /// Returns Some(name) for simple patterns (Name, Wildcard, or wrapped versions),
    /// None for complex patterns that need destructuring.
    fn simple_pattern_name(&mut self, pattern: &ast::Pattern) -> Option<String> {
        match &pattern.kind {
            ast::PatternKind::Name(name) => Some(name.clone()),
            ast::PatternKind::Wildcard => Some(format!("_w_wild_{}", self.term_ids.next_id().0)),
            ast::PatternKind::Typed(inner, _) | ast::PatternKind::Attributed(_, inner) => {
                self.simple_pattern_name(inner)
            }
            ast::PatternKind::Tuple(_)
            | ast::PatternKind::Record(_)
            | ast::PatternKind::Unit
            | ast::PatternKind::Literal(_)
            | ast::PatternKind::Constructor(_, _) => None,
        }
    }

    fn extract_tuple_types(&self, ty: &Type<TypeName>, _expected_len: usize) -> Vec<Type<TypeName>> {
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
            _ => panic!("BUG: Expected tuple type, got {:?}", ty),
        }
    }

    /// Resolve a field name to its index in a record type
    fn resolve_field_index(&self, ty: &Type<TypeName>, field: &str) -> Option<usize> {
        match ty {
            Type::Constructed(TypeName::Record(fields), _) => fields.iter().position(|f| f == field),
            // Vec swizzle: x=0, y=1, z=2, w=3
            Type::Constructed(TypeName::Vec, _) => match field {
                "x" => Some(0),
                "y" => Some(1),
                "z" => Some(2),
                "w" => Some(3),
                _ => None,
            },
            // Tuple positional access: .0, .1, etc.
            Type::Constructed(TypeName::Tuple(_), _) => field.parse::<usize>().ok(),
            _ => None,
        }
    }

    fn extract_record_types(&self, ty: &Type<TypeName>) -> HashMap<String, Type<TypeName>> {
        match ty {
            Type::Constructed(TypeName::Record(fields), args) => {
                fields.iter().cloned().zip(args.iter().cloned()).collect()
            }
            _ => HashMap::new(),
        }
    }

    fn transform_expr(&mut self, expr: &ast::Expression) -> Term {
        let ty = self.lookup_type(expr.h.id).unwrap_or_else(|| {
            panic!(
                "BUG: Expression must have type in type table. NodeId={:?}, kind={:?}, span={:?}",
                expr.h.id, expr.kind, expr.h.span
            )
        });
        let span = expr.h.span;

        match &expr.kind {
            ast::ExprKind::IntLiteral(s) => self.mk_term(ty, span, TermKind::IntLit(s.0.clone())),

            ast::ExprKind::FloatLiteral(f) => self.mk_term(ty, span, TermKind::FloatLit(*f)),

            ast::ExprKind::BoolLiteral(b) => self.mk_term(ty, span, TermKind::BoolLit(*b)),

            ast::ExprKind::Unit => {
                // Unit value represented as _w_unit intrinsic call
                self.build_intrinsic_call("_w_unit", &[], ty, span)
            }

            ast::ExprKind::Identifier(qualifiers, name) => {
                // First consult the NameResolution side table built at
                // type-check time. If this NodeId was classified as a
                // catalog builtin, emit `Var(Builtin(id))` directly —
                // no SymbolId allocation, no surface→internal name
                // rename, no later string-matching at the EGIR boundary.
                if let Some(crate::name_resolution::ResolvedValueRef::Builtin { id, overload_idx }) =
                    self.name_resolution.get(expr.h.id)
                {
                    let overload_idx = overload_idx.unwrap_or_else(|| {
                        let def = crate::builtins::by_id(*id);
                        panic!(
                            "BUG: builtin '{}' (id={:?}) reached TLC with unresolved overload — \
                             type checker must call NameResolution::set_overload_idx after \
                             overload resolution",
                            def.raw.surface_name, id
                        )
                    });
                    return self.mk_term(
                        ty,
                        span,
                        TermKind::Var(crate::tlc::VarRef::Builtin {
                            id: *id,
                            overload_idx,
                        }),
                    );
                }
                let resolved_name = if qualifiers.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", qualifiers.join("."), name)
                };
                let sym = self.resolve_or_define(&resolved_name);
                self.mk_term(ty, span, TermKind::Var(crate::tlc::VarRef::Symbol(sym)))
            }

            ast::ExprKind::ArrayLiteral(elements) => {
                log::debug!("ArrayLiteral with {} elements", elements.len());
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_array_lit(terms, ty, span)
            }

            ast::ExprKind::VecMatLiteral(elements) => {
                // For matrices, columns are vectors not arrays
                // Check if result type is Mat and transform columns accordingly
                if ty.is_mat() {
                    // Mat[elem, cols, rows] - column type is Vec[elem, rows]
                    if let (Some(elem), Some(rows_ty)) = (ty.elem_type(), ty.mat_rows_type()) {
                        let col_ty = Type::Constructed(TypeName::Vec, vec![elem.clone(), rows_ty.clone()]);
                        // Transform elements, treating ArrayLiterals as vectors
                        let col_terms: Vec<Term> =
                            elements.iter().map(|e| self.transform_as_vector(e, col_ty.clone())).collect();
                        return self.build_vec_lit_from_terms(&col_terms, ty, span);
                    }
                }
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_vec_lit(terms, ty, span)
            }

            ast::ExprKind::ArrayIndex(array, index) => {
                let arr = self.transform_expr(array);
                let idx = self.transform_expr(index);
                self.mk_index(arr, idx, ty, span)
            }

            ast::ExprKind::ArrayWith {
                array, index, value, ..
            } => {
                let arr = self.transform_expr(array);
                let idx = self.transform_expr(index);
                let val = self.transform_expr(value);
                let aw_id = crate::builtins::catalog().known().array_with;
                self.build_call_by_id(aw_id, &[arr, idx, val], ty, span)
            }

            ast::ExprKind::VecWith {
                target,
                components,
                op,
                value,
            } => self.transform_vec_with(target, components, op.as_deref(), value, ty, span),

            ast::ExprKind::RecordWith { record, path, value } => {
                self.transform_record_with(record, path, value, ty, span)
            }

            ast::ExprKind::BinaryOp(op, lhs, rhs) => {
                let l = self.transform_expr(lhs);
                let r = self.transform_expr(rhs);
                self.build_binop(op.clone(), l, r, ty, span)
            }

            ast::ExprKind::UnaryOp(op, operand) => {
                let arg = self.transform_expr(operand);
                self.build_unop(op.clone(), arg, ty, span)
            }

            ast::ExprKind::Tuple(elements) => {
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_tuple(terms, ty, span)
            }

            ast::ExprKind::RecordLiteral(fields) => {
                // Records are tuples - reorder fields to match type's field order
                let field_map: HashMap<&str, &ast::Expression> =
                    fields.iter().map(|(name, expr)| (name.as_str(), expr)).collect();

                let ordered_exprs: Vec<ast::Expression> = match &ty {
                    Type::Constructed(TypeName::Record(type_fields), _) => type_fields
                        .iter()
                        .filter_map(|f| field_map.get(f.as_str()).map(|e| (*e).clone()))
                        .collect(),
                    _ => fields.iter().map(|(_, e)| e.clone()).collect(),
                };

                let terms: Vec<Term> = ordered_exprs.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_tuple(terms, ty, span)
            }

            ast::ExprKind::Lambda(lam) => self.transform_lambda(&lam.params, &lam.body, ty, span),

            ast::ExprKind::Application(func, args) => self.transform_application(func, args, ty, span),

            ast::ExprKind::LetIn(let_in) => {
                // Snapshot scope so a nested let's bindings don't leak past
                // the body. Without this, `let x = ... in let x = ... in ...; ... x ...`
                // resolves the trailing `x` to the inner SymbolId because
                // the inner `define` overwrote `scope["x"]`.
                let saved_scope = self.scope.clone();
                // Check pattern kind to avoid redundant transforms for simple patterns
                let simple_name = self.simple_pattern_name(&let_in.pattern);

                let result = if let Some(name_str) = simple_name {
                    // Simple Name/Wildcard pattern - single Let binding
                    let rhs = self.transform_expr(&let_in.value);
                    // Define the name (adds to scope) before transforming body
                    let name_sym = self.define(&name_str);
                    let body = self.transform_expr(&let_in.body);
                    self.mk_term(
                        body.ty.clone(),
                        span,
                        TermKind::Let {
                            name: name_sym,
                            name_ty: rhs.ty.clone(),
                            rhs: Box::new(rhs),
                            body: Box::new(body),
                        },
                    )
                } else {
                    // Complex pattern - use compute_pattern_bindings
                    let rhs = self.transform_expr(&let_in.value);
                    let (_, bindings) = self.compute_pattern_bindings(&let_in.pattern, rhs, span);
                    // Note: bindings already added names to scope via define() in compute_pattern_bindings
                    let body = self.transform_expr(&let_in.body);
                    self.apply_bindings_around(bindings, body, span)
                };
                self.scope = saved_scope;
                result
            }

            ast::ExprKind::FieldAccess(record, field) => {
                let rec = self.transform_expr(record);
                // Vec swizzle (1–4 letters from a single swizzle set —
                // `xyzw` or `rgba`): build per-letter projections;
                // single letter → scalar, multi → _w_vec_lit.
                if rec.ty.is_vec() && crate::types::is_swizzle_field(field) {
                    let elem_ty = rec.ty.elem_type().cloned().unwrap_or(ty.clone());
                    let components: Vec<Term> = field
                        .chars()
                        .map(|c| {
                            let idx = crate::types::swizzle_component_index(c)
                                .expect("is_swizzle_field already accepted this letter");
                            self.mk_tuple_proj(rec.clone(), idx as usize, elem_ty.clone(), span)
                        })
                        .collect();
                    if components.len() == 1 {
                        return components.into_iter().next().unwrap();
                    }
                    return self.build_vec_lit_from_terms(&components, ty, span);
                }
                // Resolve field name to index, treat record as tuple
                let field_idx = self
                    .resolve_field_index(&rec.ty, field)
                    .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field));
                self.mk_tuple_proj(rec, field_idx, ty, span)
            }

            ast::ExprKind::If(if_expr) => {
                let cond = self.transform_expr(&if_expr.condition);
                let then_branch = self.transform_expr(&if_expr.then_branch);
                let else_branch = self.transform_expr(&if_expr.else_branch);
                self.mk_term(
                    ty,
                    span,
                    TermKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                )
            }

            ast::ExprKind::Loop(loop_expr) => self.transform_loop(loop_expr, ty, span),

            ast::ExprKind::Match(match_expr) => self.transform_match(match_expr, ty, span),

            ast::ExprKind::Constructor(name, args) => {
                // Lower `#ck(a1..am)` to a flat tuple
                // `(tag=k, slot_1, ..., slot_total-1)` where the active
                // constructor's payload occupies slots [offset_k, offset_k+m)
                // and dead slots get zero-filled.
                let raw_sum_ty = self
                    .lookup_type_raw(expr.h.id)
                    .expect("BUG: Constructor expression must have type in type table");
                let variants = match &raw_sum_ty {
                    Type::Constructed(TypeName::Sum(v), _) => v.clone(),
                    _ => panic!("BUG: Constructor `#{}` has non-sum type {:?}", name, raw_sum_ty),
                };
                let layout = Self::sum_layout(&variants);
                let &(tag_value, payload_offset) = layout
                    .constructor_info
                    .get(name)
                    .expect("BUG: Phase B should have validated constructor name");

                let arg_terms: Vec<Term> = args.iter().map(|a| self.transform_expr(a)).collect();

                let tag_term = self.mk_term(
                    Type::Constructed(TypeName::UInt(32), vec![]),
                    span,
                    TermKind::IntLit(tag_value.to_string()),
                );
                let mut slot_terms: Vec<Term> = Vec::with_capacity(layout.slot_types.len());
                slot_terms.push(tag_term);
                for slot_idx in 1..layout.slot_types.len() {
                    let slot_ty = &layout.slot_types[slot_idx];
                    if slot_idx >= payload_offset && slot_idx < payload_offset + arg_terms.len() {
                        slot_terms.push(arg_terms[slot_idx - payload_offset].clone());
                    } else {
                        slot_terms.push(self.build_zero(slot_ty, span));
                    }
                }
                self.mk_tuple(slot_terms, ty, span)
            }

            ast::ExprKind::Range(range) => {
                // Transform range to _w_range intrinsic
                let start = self.transform_expr(&range.start);
                let end = self.transform_expr(&range.end);
                let kind_val = match range.kind {
                    ast::RangeKind::Inclusive => 0,
                    ast::RangeKind::Exclusive => 1,
                    ast::RangeKind::ExclusiveLt => 2,
                    ast::RangeKind::ExclusiveGt => 3,
                };
                let kind_lit = self.mk_term(
                    Type::Constructed(TypeName::Int(32), vec![]),
                    span,
                    TermKind::IntLit(kind_val.to_string()),
                );

                match &range.step {
                    Some(step_expr) => {
                        let step = self.transform_expr(step_expr);
                        // _w_range_step start step end kind
                        self.build_call("_w_range_step", &[start, step, end, kind_lit], ty, span)
                    }
                    None => {
                        // _w_range start end kind
                        self.build_call("_w_range", &[start, end, kind_lit], ty, span)
                    }
                }
            }

            ast::ExprKind::Slice(slice) => {
                // Transform slice to _w_intrinsic_slice(arr, start, end).
                // The slice aliases the source — it's a view, not a copy.
                let arr = self.transform_expr(&slice.array);

                // Omitted start defaults to 0.
                let start = slice
                    .start
                    .as_ref()
                    .map(|e| self.transform_expr(e))
                    .unwrap_or_else(|| self.mk_i32(0, span));

                // Omitted end defaults to `length(arr)`. `_w_intrinsic_length`
                // is registered as returning i32 and works on every array
                // flavor (composite / view / virtual) — the subsequent
                // `buffer_specialize` / SPIR-V passes rewrite it into the
                // right per-flavor lowering.
                let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let known = crate::builtins::catalog().known();
                let end =
                    slice.end.as_ref().map(|e| self.transform_expr(e)).unwrap_or_else(|| {
                        self.build_call_by_id(known.length, &[arr.clone()], i32_ty, span)
                    });

                self.build_call_by_id(known.slice, &[arr, start, end], ty, span)
            }

            ast::ExprKind::TypeAscription(inner, _) => self.transform_expr(inner),

            ast::ExprKind::TypeCoercion(inner, _) => {
                let term = self.transform_expr(inner);
                self.build_call("_w_coerce", &[term], ty, span)
            }

            ast::ExprKind::TypeHole => {
                if !self.fill_holes {
                    unreachable!(
                        "TypeHole should be rejected at type-check when \
                         --fill-holes is not set; see \
                         TypeChecked::reject_type_holes"
                    );
                }
                let hole_ty = self.lookup_type(expr.h.id).unwrap_or(ty.clone());
                defaults::default_term_for_type(self, &hole_ty, span)
            }
        }
    }

    fn transform_lambda(
        &mut self,
        params: &[ast::Pattern],
        body: &ast::Expression,
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        self.build_lambda_chain(params, body, ty, span)
    }

    fn get_param_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => args[0].clone(),
            _ => panic!("BUG: Expected arrow type for function param, got {:?}", ty),
        }
    }

    fn get_body_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => args[1].clone(),
            _ => ty.clone(),
        }
    }

    fn transform_application(
        &mut self,
        func: &ast::Expression,
        args: &[ast::Expression],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Check if func is a bare SOAC name (not locally bound)
        if let Some(soac_name) = self.resolve_soac_name(func) {
            return self.transform_soac_call(&soac_name, args, ty, span);
        }

        let func_term = self.transform_expr(func);

        if args.is_empty() {
            return func_term;
        }

        let arg_terms: Vec<Term> = args.iter().map(|a| self.transform_expr(a)).collect();

        // If func_term is already an App, flatten by merging args.
        // The AST represents chained calls as nested Application nodes.
        if let TermKind::App { .. } = &func_term.kind {
            let TermKind::App {
                func: inner_func,
                args: inner_args,
            } = func_term.kind
            else {
                unreachable!()
            };
            let mut all_args = inner_args;
            all_args.extend(arg_terms);
            return self.mk_term(
                ty,
                span,
                TermKind::App {
                    func: inner_func,
                    args: all_args,
                },
            );
        }

        self.mk_term(
            ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                args: arg_terms,
            },
        )
    }

    /// Check if an expression is a bare SOAC name (not locally bound).
    fn resolve_soac_name(&self, func: &ast::Expression) -> Option<String> {
        if let ast::ExprKind::Identifier(qualifiers, name) = &func.kind {
            if qualifiers.is_empty() && !self.is_locally_bound(name) && SOAC_NAMES.contains(&name.as_str())
            {
                return Some(name.clone());
            }
        }
        None
    }

    /// Dispatch SOAC call by name.
    fn transform_soac_call(
        &mut self,
        name: &str,
        args: &[ast::Expression],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        match name {
            "map" => self.transform_soac_map(args, ty, span),
            "reduce" => self.transform_soac_reduce(args, ty, span),
            "scan" => self.transform_soac_scan(args, ty, span),
            "filter" => self.transform_soac_filter(args, ty, span),
            "zip" | "zip2" | "zip3" | "zip4" | "zip5" => self.transform_soac_zip(args, ty, span),
            "reduce_by_index" => self.transform_soac_reduce_by_index(args, ty, span),
            _ => unreachable!("Unknown SOAC: {}", name),
        }
    }

    /// Transform `map(f, arr)` → `Soac(Map { lam, inputs })`.
    fn transform_soac_map(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 2, "map requires at least 2 arguments");
        let func_term = self.transform_expr(&args[0]);
        let arr_term = self.transform_expr(&args[1]);

        let lam = self.term_to_lambda(func_term);

        // Absorb zip: if arr_term is ArrayExpr(Zip(...)), flatten into inputs.
        // The lambda still takes a single tuple param — the soa::normalize pass
        // will rewrite it to take separate params.
        let inputs = match arr_term.kind {
            TermKind::ArrayExpr(ArrayExpr::Zip(exprs)) => exprs,
            _ => vec![ArrayExpr::Ref(Box::new(arr_term))],
        };

        self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Map {
                lam,
                inputs,
                consumes_input: false,
            }),
        )
    }

    /// Transform `reduce(op, ne, arr)` → `Soac(Reduce { op, ne, input, props })`.
    fn transform_soac_reduce(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 3, "reduce requires 3 arguments");
        let op_term = self.transform_expr(&args[0]);
        let ne_term = self.transform_expr(&args[1]);
        let arr_term = self.transform_expr(&args[2]);

        let op = self.term_to_lambda(op_term);

        self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Reduce {
                op,
                ne: Box::new(ne_term),
                input: ArrayExpr::Ref(Box::new(arr_term)),
                props: ReduceProps::default(),
            }),
        )
    }

    /// Transform `scan(op, ne, arr)` → `Soac(Scan { op, ne, input })`.
    fn transform_soac_scan(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 3, "scan requires 3 arguments");
        let op_term = self.transform_expr(&args[0]);
        let ne_term = self.transform_expr(&args[1]);
        let arr_term = self.transform_expr(&args[2]);

        let op = self.term_to_lambda(op_term);

        self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Scan {
                op,
                ne: Box::new(ne_term),
                input: ArrayExpr::Ref(Box::new(arr_term)),
            }),
        )
    }

    /// Transform `filter(pred, arr)` → `Soac(Filter { pred, input })`.
    fn transform_soac_filter(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 2, "filter requires 2 arguments");
        let pred_term = self.transform_expr(&args[0]);
        let arr_term = self.transform_expr(&args[1]);

        let pred = self.term_to_lambda(pred_term);

        self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Filter {
                pred,
                input: ArrayExpr::Ref(Box::new(arr_term)),
            }),
        )
    }

    /// Transform `zip(a, b, ...)` → `ArrayExpr(Zip(...))`.
    fn transform_soac_zip(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        let exprs: Vec<ArrayExpr> =
            args.iter().map(|a| ArrayExpr::Ref(Box::new(self.transform_expr(a)))).collect();
        self.mk_term(ty, span, TermKind::ArrayExpr(ArrayExpr::Zip(exprs)))
    }

    /// Transform `reduce_by_index(dest, op, ne, indices, values)`.
    fn transform_soac_reduce_by_index(
        &mut self,
        args: &[ast::Expression],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert!(args.len() >= 5, "reduce_by_index requires 5 arguments");
        let dest_term = self.transform_expr(&args[0]);
        let op_term = self.transform_expr(&args[1]);
        let ne_term = self.transform_expr(&args[2]);
        let indices_term = self.transform_expr(&args[3]);
        let values_term = self.transform_expr(&args[4]);

        let op = self.term_to_lambda(op_term);

        // Build a Place from dest_term
        let dest_elem_ty = self.get_array_element_type(&dest_term.ty);
        let dest = Place::LocalArray {
            id: match &dest_term.kind {
                TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => *sym,
                _ => {
                    // Bind dest to a fresh name
                    let fresh = self.define("_w_rbi_dest");
                    fresh
                }
            },
            shape: Shape(vec![]),
            elem_ty: dest_elem_ty,
        };

        self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::ReduceByIndex {
                dest,
                op,
                ne: Box::new(ne_term),
                indices: ArrayExpr::Ref(Box::new(indices_term)),
                values: ArrayExpr::Ref(Box::new(values_term)),
                props: ReduceProps::default(),
            }),
        )
    }

    /// Convert a term to a SoacBody. If it's already a Lambda, wrap it.
    /// Otherwise, eta-expand all parameters: `f : A -> B -> C` → `|a, b| f(a)(b)`.
    /// Captures are always empty here — this runs pre-defunctionalization.
    fn term_to_lambda(&mut self, term: Term) -> SoacBody {
        match term.kind {
            TermKind::Lambda(lam) => SoacBody {
                lam,
                captures: vec![],
            },
            _ => {
                // Decompose the full arrow chain: A -> B -> C gives ([A, B], C)
                let mut param_tys = Vec::new();
                let mut current = term.ty.clone();
                while let Type::Constructed(TypeName::Arrow, ref args) = current {
                    if args.len() == 2 {
                        param_tys.push(args[0].clone());
                        current = args[1].clone();
                    } else {
                        break;
                    }
                }
                assert!(
                    !param_tys.is_empty(),
                    "BUG: Expected arrow type for SOAC function arg, got {:?}",
                    term.ty
                );
                let ret_ty = current;

                // Create parameter symbols. Display names must be distinct
                // per-parameter — SPIR-V keys off numeric parameter ids so
                // it survives duplicate names, but text-emitting backends
                // (GLSL, WGSL) inherit the display name verbatim and reject
                // a function whose parameter list repeats a name.
                let params: Vec<(SymbolId, Type<TypeName>)> = param_tys
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| (self.define(&format!("_soac_arg_{}", i)), ty.clone()))
                    .collect();

                // Build flat App(f, [a, b, ...])
                let span = term.span;
                let arg_terms: Vec<Term> = params
                    .iter()
                    .map(|(sym, ty)| {
                        self.mk_term(ty.clone(), span, TermKind::Var(crate::tlc::VarRef::Symbol(*sym)))
                    })
                    .collect();
                let body = self.mk_term(
                    ret_ty.clone(),
                    span,
                    TermKind::App {
                        func: Box::new(term),
                        args: arg_terms,
                    },
                );

                SoacBody {
                    lam: Lambda {
                        params,
                        body: Box::new(body),
                        ret_ty,
                    },
                    captures: vec![],
                }
            }
        }
    }

    fn transform_loop(&mut self, loop_expr: &ast::LoopExpr, ty: Type<TypeName>, span: Span) -> Term {
        // Get the init expression and accumulator type
        let init_term = loop_expr.init.as_ref().map(|e| self.transform_expr(e)).unwrap_or_else(|| {
            // No accumulator - use unit
            self.build_intrinsic_call("_w_unit", &[], Type::Constructed(TypeName::Unit, vec![]), span)
        });
        let acc_ty = init_term.ty.clone();

        // Build loop_var and init_bindings from the pattern
        let (loop_var, loop_var_ty, init_bindings) =
            self.build_loop_var_and_bindings(&loop_expr.pattern, &acc_ty, span);

        match &loop_expr.form {
            ast::LoopForm::For(idx_var, bound) => {
                let bound_term = self.transform_expr(bound);
                let index_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let idx_var_sym = self.define(idx_var);

                // Transform body after defining the index variable
                let body = self.transform_expr(&loop_expr.body);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::ForRange {
                            var: idx_var_sym,
                            var_ty: index_ty,
                            bound: Box::new(bound_term),
                        },
                        body: Box::new(body),
                    },
                )
            }

            ast::LoopForm::ForIn(elem_pattern, iter) => {
                let iter_term = self.transform_expr(iter);
                let elem_ty = self.get_array_element_type(&iter_term.ty);
                let elem_var_name = elem_pattern.simple_name().unwrap_or("_w_elem").to_string();
                let elem_var_sym = self.define(&elem_var_name);

                // Transform body after defining the element variable
                let body = self.transform_expr(&loop_expr.body);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::For {
                            var: elem_var_sym,
                            var_ty: elem_ty,
                            iter: Box::new(iter_term),
                        },
                        body: Box::new(body),
                    },
                )
            }

            ast::LoopForm::While(cond) => {
                let body = self.transform_expr(&loop_expr.body);
                let cond_term = self.transform_expr(cond);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::While {
                            cond: Box::new(cond_term),
                        },
                        body: Box::new(body),
                    },
                )
            }
        }
    }

    /// Build loop variable name and init_bindings from a pattern.
    fn build_loop_var_and_bindings(
        &mut self,
        pattern: &ast::Pattern,
        acc_ty: &Type<TypeName>,
        span: Span,
    ) -> (SymbolId, Type<TypeName>, Vec<(SymbolId, Type<TypeName>, Term)>) {
        use crate::pattern::binding_paths;

        // For a simple name pattern, use it directly
        if let ast::PatternKind::Name(name) = &pattern.kind {
            let name_sym = self.define(name);
            return (name_sym, acc_ty.clone(), vec![]);
        }

        // For complex patterns, create a fresh loop_var and build projections
        let loop_var_name = format!("_w_loop_{}", self.term_ids.next_id().0);
        let loop_var_sym = self.define(&loop_var_name);
        let paths = binding_paths(pattern);

        let init_bindings = paths
            .into_iter()
            .filter_map(|bp| {
                if bp.path.is_empty() {
                    // This is the root binding - shouldn't happen for complex patterns
                    None
                } else {
                    let binding_ty = self.type_at_path(acc_ty, &bp.path);
                    let proj_term = self.build_projection_chain(loop_var_sym, acc_ty, &bp.path, span);
                    let binding_sym = self.define(&bp.name);
                    Some((binding_sym, binding_ty, proj_term))
                }
            })
            .collect();

        (loop_var_sym, acc_ty.clone(), init_bindings)
    }

    /// Get the type at a given projection path within a tuple/record type.
    fn type_at_path(&self, ty: &Type<TypeName>, path: &[usize]) -> Type<TypeName> {
        let mut current = ty.clone();
        for &idx in path {
            current = match &current {
                Type::Constructed(TypeName::Tuple(_), args) => {
                    args.get(idx).cloned().unwrap_or_else(|| {
                        panic!(
                            "BUG: tuple projection index {} out of bounds for {:?}",
                            idx, current
                        )
                    })
                }
                Type::Constructed(TypeName::Record(fields), args) => {
                    args.get(idx).cloned().unwrap_or_else(|| {
                        panic!(
                            "BUG: record projection index {} out of bounds for {:?} (fields: {:?})",
                            idx, current, fields
                        )
                    })
                }
                _ => panic!("BUG: projection on non-tuple/record type: {:?}", current),
            };
        }
        current
    }

    /// Build a chain of tuple projections: proj[path[n-1]](...proj[path[0]](var))
    fn build_projection_chain(
        &mut self,
        var_sym: SymbolId,
        var_ty: &Type<TypeName>,
        path: &[usize],
        span: Span,
    ) -> Term {
        let mut current_ty = var_ty.clone();
        let mut current = self.mk_term(
            current_ty.clone(),
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(var_sym)),
        );

        for &idx in path {
            let elem_ty = self.type_at_path(&current_ty, &[idx]);
            current = self.mk_tuple_proj(current, idx, elem_ty.clone(), span);
            current_ty = elem_ty;
        }

        current
    }

    fn get_array_element_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        ty.elem_type()
            .filter(|_| ty.is_array())
            .cloned()
            .unwrap_or_else(|| panic!("BUG: Expected array type, got {:?}", ty))
    }

    /// Lower `target with .swizzle [op]= value` into a let-bound
    /// vec-build:
    ///
    /// ```text
    ///   let _t = target in
    ///   let _r = value (or _t.swizzle <op> value, for compound) in
    ///   _w_vec_lit(_t.0, ..., _r.0 or _t.i, ..., _t.{N-1})
    /// ```
    ///
    /// `_t` and `_r` are bound to fresh symbols so the inputs evaluate
    /// once even when they're arbitrary expressions.
    fn transform_vec_with(
        &mut self,
        target: &ast::Expression,
        components: &[u8],
        op: Option<&str>,
        value: &ast::Expression,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let t_term = self.transform_expr(target);
        let target_ty = t_term.ty.clone();
        let elem_ty = target_ty.elem_type().cloned().expect("VecWith target must be a vec by type-check");
        let vec_size = target_ty.vec_size().expect("VecWith target must have known size");

        // Bind `_t = target` so each per-slot projection reads the
        // same evaluated value.
        let t_id = self.term_ids.next_id().0;
        let t_sym = self.define(&format!("_w_vw_t_{}", t_id));
        let t_var = self.mk_term(
            target_ty.clone(),
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(t_sym)),
        );

        // Compute the RHS term. For plain `=`, that's just `value`.
        // For compound `op=`, build `_t.swizzle <op> value` so the
        // existing binary-op machinery handles vec-vec / vec-mat /
        // vec-scalar dispatch identically to a hand-written
        // `t.swizzle op rhs`.
        let v_term_raw = self.transform_expr(value);
        let rhs_term = match op {
            None => v_term_raw,
            Some(binop_str) => {
                let swizzle_read = self.build_swizzle_read(&t_var, components, &elem_ty, span);
                let result_slot_ty = swizzle_read.ty.clone();
                self.build_binop(
                    ast::BinaryOp {
                        op: binop_str.to_string(),
                    },
                    swizzle_read,
                    v_term_raw,
                    result_slot_ty,
                    span,
                )
            }
        };

        // Bind `_r = <rhs>` so per-slot reads share one evaluation.
        let r_id = self.term_ids.next_id().0;
        let r_sym = self.define(&format!("_w_vw_r_{}", r_id));
        let r_var = self.mk_term(
            rhs_term.ty.clone(),
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(r_sym)),
        );

        // Locate each component's position in `components` so we know
        // which RHS slot supplies each target slot.
        let component_pos: Vec<Option<usize>> =
            (0..vec_size).map(|slot| components.iter().position(|&c| c as usize == slot)).collect();

        // Build per-slot terms: RHS slot for swizzle positions,
        // original target slot otherwise.
        let single_component = components.len() == 1;
        let slot_terms: Vec<Term> = component_pos
            .iter()
            .enumerate()
            .map(|(slot, found)| match found {
                Some(rhs_pos) => {
                    if single_component {
                        // RHS is the elem type itself, not a vec.
                        r_var.clone()
                    } else {
                        self.build_proj(&r_var, *rhs_pos, &elem_ty, span)
                    }
                }
                None => self.build_proj(&t_var, slot, &elem_ty, span),
            })
            .collect();

        let body = self.build_vec_lit_from_terms(&slot_terms, result_ty.clone(), span);

        // Wrap: let _t = target in let _r = rhs in body.
        let inner = self.mk_term(
            result_ty.clone(),
            span,
            TermKind::Let {
                name: r_sym,
                name_ty: rhs_term.ty.clone(),
                rhs: Box::new(rhs_term),
                body: Box::new(body),
            },
        );
        self.mk_term(
            result_ty,
            span,
            TermKind::Let {
                name: t_sym,
                name_ty: target_ty,
                rhs: Box::new(t_term),
                body: Box::new(inner),
            },
        )
    }

    /// Lower `r with field = e` (single-level) and `r with a.x = e`
    /// (nested) by binding the record to a fresh symbol and rebuilding
    /// it via `_w_tuple` with the path target replaced. Each level of
    /// the path produces its own bind-and-rebuild; nested paths chain
    /// inside the outer rebuild's replacement slot.
    fn transform_record_with(
        &mut self,
        record: &ast::Expression,
        path: &[String],
        value: &ast::Expression,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let r_term = self.transform_expr(record);
        let record_ty = r_term.ty.clone();
        let new_value = self.transform_expr(value);

        let r_id = self.term_ids.next_id().0;
        let r_sym = self.define(&format!("_w_rw_r_{}", r_id));
        let r_var = self.mk_term(
            record_ty.clone(),
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(r_sym)),
        );

        let body = self.build_record_with_body(&r_var, &record_ty, path, new_value, span);

        self.mk_term(
            result_ty,
            span,
            TermKind::Let {
                name: r_sym,
                name_ty: record_ty,
                rhs: Box::new(r_term),
                body: Box::new(body),
            },
        )
    }

    /// Recursive builder for `transform_record_with`. `target` is a
    /// Var term referring to the record at this level of the path.
    fn build_record_with_body(
        &mut self,
        target: &Term,
        record_ty: &Type<TypeName>,
        path: &[String],
        new_value: Term,
        span: Span,
    ) -> Term {
        let (fields, field_types) = match record_ty {
            Type::Constructed(TypeName::Record(fs), tys) => (fs, tys),
            _ => panic!("BUG: record-with target must be a record type at lowering"),
        };
        let head = &path[0];
        let idx = fields.get_index(head).expect("BUG: typeck verified record field exists");

        let replacement = if path.len() == 1 {
            new_value
        } else {
            let inner_ty = field_types[idx].clone();
            let inner_proj = self.build_proj(target, idx, &inner_ty, span);
            let inner_id = self.term_ids.next_id().0;
            let inner_sym = self.define(&format!("_w_rw_inner_{}", inner_id));
            let inner_var = self.mk_term(
                inner_ty.clone(),
                span,
                TermKind::Var(crate::tlc::VarRef::Symbol(inner_sym)),
            );
            let inner_body =
                self.build_record_with_body(&inner_var, &inner_ty, &path[1..], new_value, span);
            self.mk_term(
                inner_ty.clone(),
                span,
                TermKind::Let {
                    name: inner_sym,
                    name_ty: inner_ty,
                    rhs: Box::new(inner_proj),
                    body: Box::new(inner_body),
                },
            )
        };

        let field_terms: Vec<Term> = (0..fields.len())
            .map(|i| {
                if i == idx {
                    replacement.clone()
                } else {
                    self.build_proj(target, i, &field_types[i], span)
                }
            })
            .collect();

        self.mk_tuple(field_terms, record_ty.clone(), span)
    }

    /// Build a swizzle read on a Var term: emits per-letter
    /// `_w_tuple_proj` calls and assembles them with
    /// `_w_vec_lit_from_terms` (or returns the single term when
    /// `components.len() == 1`).
    fn build_swizzle_read(
        &mut self,
        target_var: &Term,
        components: &[u8],
        elem_ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        let projs: Vec<Term> =
            components.iter().map(|&c| self.build_proj(target_var, c as usize, elem_ty, span)).collect();
        if projs.len() == 1 {
            projs.into_iter().next().unwrap()
        } else {
            let result_ty = Type::Constructed(
                TypeName::Vec,
                vec![
                    elem_ty.clone(),
                    Type::Constructed(TypeName::Size(components.len()), vec![]),
                ],
            );
            self.build_vec_lit_from_terms(&projs, result_ty, span)
        }
    }

    /// Build `TermKind::TupleProj` returning `result_ty`.
    fn build_proj(&mut self, target: &Term, idx: usize, result_ty: &Type<TypeName>, span: Span) -> Term {
        self.mk_tuple_proj(target.clone(), idx, result_ty.clone(), span)
    }

    fn transform_match(&mut self, match_expr: &ast::MatchExpr, ty: Type<TypeName>, span: Span) -> Term {
        let raw_scrutinee_ty = self
            .lookup_type_raw(match_expr.scrutinee.h.id)
            .expect("BUG: match scrutinee must have type in type table");
        let scrutinee = self.transform_expr(&match_expr.scrutinee);

        if match_expr.cases.is_empty() {
            todo!("Empty match")
        }

        // Phase B requires a sum-typed scrutinee with all-Constructor
        // arms. Route every match through the sum-aware path; the
        // legacy `compile_match_cases` is no longer reached.
        let variants = match &raw_scrutinee_ty {
            Type::Constructed(TypeName::Sum(v), _) => v.clone(),
            _ => panic!(
                "BUG: match scrutinee must be a sum type after Phase B, got {:?}",
                raw_scrutinee_ty
            ),
        };
        self.compile_sum_match(scrutinee, &variants, &match_expr.cases, ty, span)
    }

    fn compile_sum_match(
        &mut self,
        scrutinee: Term,
        variants: &[(String, Vec<Type<TypeName>>)],
        cases: &[ast::MatchCase],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let layout = Self::sum_layout(variants);
        let scrutinee_ty = scrutinee.ty.clone();

        // Bind the scrutinee to a fresh symbol so each arm can read
        // it without re-evaluating the input expression.
        let scrut_name = format!("_w_match_scrut_{}", self.term_ids.next_id().0);
        let scrut_sym = self.define(&scrut_name);
        let scrut_var = self.mk_term(
            scrutinee_ty.clone(),
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(scrut_sym)),
        );

        let body = self.build_sum_match_chain(&scrut_var, &layout, variants, cases, ty.clone(), span);

        self.mk_term(
            ty,
            span,
            TermKind::Let {
                name: scrut_sym,
                name_ty: scrutinee_ty,
                rhs: Box::new(scrutinee),
                body: Box::new(body),
            },
        )
    }

    fn build_sum_match_chain(
        &mut self,
        scrut_var: &Term,
        layout: &SumLayout,
        variants: &[(String, Vec<Type<TypeName>>)],
        cases: &[ast::MatchCase],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        debug_assert!(!cases.is_empty(), "Phase B exhaustiveness guarantees ≥1 arm");
        let case = &cases[0];
        let rest = &cases[1..];

        let (ctor_name, sub_patterns) = match &case.pattern.kind {
            ast::PatternKind::Constructor(n, ps) => (n, ps),
            other => panic!(
                "BUG: Phase B should have rejected non-Constructor arm: {:?}",
                other
            ),
        };
        let &(tag_value, payload_offset) = layout
            .constructor_info
            .get(ctor_name)
            .expect("BUG: Phase B should have validated constructor name");
        let payload_types = &variants
            .iter()
            .find(|(n, _)| n == ctor_name)
            .expect("BUG: constructor must exist in variants")
            .1;

        // Project each payload slot and bind it via the regular
        // pattern-binding machinery. Sub-patterns are nested
        // (is_top_level=false) so simple `Name` sub-patterns also
        // produce real bindings.
        let mut bindings = Vec::new();
        for (i, sub_pat) in sub_patterns.iter().enumerate() {
            let lowered_payload_ty = Self::lower_type(payload_types[i].clone());
            let proj = self.mk_tuple_proj(scrut_var.clone(), payload_offset + i, lowered_payload_ty, span);
            let (_, sub_bindings) = self.compute_pattern_bindings_inner(sub_pat, proj, span, false);
            bindings.extend(sub_bindings);
        }

        let body_term = self.transform_expr(&case.body);
        let arm_body = self.apply_bindings_around(bindings, body_term, span);

        // Last arm: Phase B exhaustiveness guarantees we reach it,
        // so emit the body directly with no tag check. This avoids
        // needing an `_w_match_fail` fallthrough on the bottom arm.
        if rest.is_empty() {
            return arm_body;
        }

        // Otherwise: if scrut.tag == tag_value then arm_body else <rest>
        let tag_proj = self.mk_tuple_proj(
            scrut_var.clone(),
            0,
            Type::Constructed(TypeName::UInt(32), vec![]),
            span,
        );
        let tag_lit = self.mk_term(
            Type::Constructed(TypeName::UInt(32), vec![]),
            span,
            TermKind::IntLit(tag_value.to_string()),
        );
        let cond = self.build_binop(
            ast::BinaryOp { op: "==".to_string() },
            tag_proj,
            tag_lit,
            Type::Constructed(TypeName::Bool, vec![]),
            span,
        );

        let else_branch = self.build_sum_match_chain(scrut_var, layout, variants, rest, ty.clone(), span);

        self.mk_term(
            ty,
            span,
            TermKind::If {
                cond: Box::new(cond),
                then_branch: Box::new(arm_body),
                else_branch: Box::new(else_branch),
            },
        )
    }

    /// Produce a typed-zero Term for `ty`. Used to fill dead
    /// constructor-payload slots in a flattened sum-type tuple.
    fn build_zero(&mut self, ty: &Type<TypeName>, span: Span) -> Term {
        match ty {
            Type::Constructed(TypeName::Int(_), _) | Type::Constructed(TypeName::UInt(_), _) => {
                self.mk_term(ty.clone(), span, TermKind::IntLit("0".to_string()))
            }
            Type::Constructed(TypeName::Float(_), _) => {
                self.mk_term(ty.clone(), span, TermKind::FloatLit(0.0))
            }
            Type::Constructed(TypeName::Bool, _) => {
                self.mk_term(ty.clone(), span, TermKind::BoolLit(false))
            }
            Type::Constructed(TypeName::Unit, _) => self.build_call("_w_unit", &[], ty.clone(), span),
            Type::Constructed(TypeName::Tuple(_), elems)
            | Type::Constructed(TypeName::Record(_), elems) => {
                let zero_terms: Vec<Term> = elems.iter().map(|t| self.build_zero(t, span)).collect();
                self.mk_tuple(zero_terms, ty.clone(), span)
            }
            _ => todo!("zero-fill for sum-payload type {:?}", ty),
        }
    }

    // Helper: build binary op application
    fn build_binop(
        &mut self,
        op: ast::BinaryOp,
        lhs: Term,
        rhs: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Build the binop type: lhs.ty -> rhs.ty -> result_ty
        let binop_ty = Type::Constructed(
            TypeName::Arrow,
            vec![
                lhs.ty.clone(),
                Type::Constructed(TypeName::Arrow, vec![rhs.ty.clone(), result_ty.clone()]),
            ],
        );
        let binop_term = self.mk_term(binop_ty, span, TermKind::BinOp(op));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(binop_term),
                args: vec![lhs, rhs],
            },
        )
    }

    // Helper: build unary op application
    fn build_unop(&mut self, op: ast::UnaryOp, arg: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        let unop_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), result_ty.clone()]);
        let unop_term = self.mk_term(unop_ty, span, TermKind::UnOp(op));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(unop_term),
                args: vec![arg],
            },
        )
    }

    // Helper: build flat application for variable number of args
    /// Build a flat call from a function name and already-transformed argument terms.
    /// For f(a, b, c) with result R: builds App { func: Var(f), args: [a, b, c] }
    fn build_call(
        &mut self,
        func_name: &str,
        args: &[Term],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Catalog-resolved builtins emit `Var(Builtin(id))` directly so
        // TLC→EGIR dispatches structurally (no later string match against
        // `_w_intrinsic_*`). Names that aren't in the catalog (e.g.
        // `_w_tuple`, `_w_vec_lit`, `_w_index` — TLC-level pseudo-ops
        // handled by dedicated `convert_named_app` arms) keep the
        // SymbolId path.
        let func_var = if let Some(def) = crate::builtins::catalog().lookup_by_any_name(func_name) {
            // Compiler-emitted catalog calls go through `build_call`
            // with the internal `_w_intrinsic_*` name; these are
            // pre-resolved (the IR site picked the exact builtin), and
            // they target single-overload entries today, so
            // `overload_idx = 0` is correct. If a future caller targets
            // a multi-overload entry from a synthesised TLC site, this
            // assertion will fire and the caller must specify the
            // overload explicitly.
            assert_eq!(
                def.overloads().len(),
                1,
                "build_call({:?}) targets a multi-overload catalog entry; \
                 callers must specify overload_idx explicitly",
                func_name
            );
            crate::tlc::VarRef::Builtin {
                id: def.id,
                overload_idx: 0,
            }
        } else {
            crate::tlc::VarRef::Symbol(self.resolve_or_define(func_name))
        };
        if args.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(func_var));
        }

        // Build the function type: arg1_ty -> arg2_ty -> ... -> result_ty (right-associative)
        let mut func_ty = result_ty.clone();
        for arg in args.iter().rev() {
            func_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), func_ty]);
        }
        let func_term = self.mk_term(func_ty, span, TermKind::Var(func_var));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                args: args.to_vec(),
            },
        )
    }

    /// Build a flat call against a catalog `BuiltinId`.
    fn build_call_by_id(
        &mut self,
        id: crate::builtins::BuiltinId,
        args: &[Term],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_var = crate::tlc::VarRef::Builtin { id, overload_idx: 0 };
        if args.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(func_var));
        }
        let mut func_ty = result_ty.clone();
        for arg in args.iter().rev() {
            func_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), func_ty]);
        }
        let func_term = self.mk_term(func_ty, span, TermKind::Var(func_var));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                args: args.to_vec(),
            },
        )
    }

    fn build_intrinsic_call(
        &mut self,
        name: &str,
        args: &[ast::Expression],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let arg_terms: Vec<Term> = args.iter().map(|a| self.transform_expr(a)).collect();
        self.build_call(name, &arg_terms, result_ty, span)
    }

    /// Construct a `TermKind::Tuple` directly.
    fn mk_tuple(&mut self, parts: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::Tuple(parts))
    }

    /// Construct a `TermKind::TupleProj` directly.
    fn mk_tuple_proj(&mut self, tuple: Term, idx: usize, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(
            result_ty,
            span,
            TermKind::TupleProj {
                tuple: Box::new(tuple),
                idx,
            },
        )
    }

    /// Construct a `TermKind::Index` directly.
    fn mk_index(&mut self, array: Term, index: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(
            result_ty,
            span,
            TermKind::Index {
                array: Box::new(array),
                index: Box::new(index),
            },
        )
    }

    /// Construct a `TermKind::VecLit` directly.
    fn mk_vec_lit(&mut self, parts: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::VecLit(parts))
    }

    /// Construct an array literal `[a, b, c]` as
    /// `TermKind::ArrayExpr(ArrayExpr::Literal(parts))`.
    fn mk_array_lit(&mut self, parts: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::ArrayExpr(ArrayExpr::Literal(parts)))
    }

    fn lookup_type(&self, node_id: NodeId) -> Option<Type<TypeName>> {
        self.lookup_type_raw(node_id).map(Self::lower_type)
    }

    /// Like `lookup_type`, but returns the type *before* sum-type
    /// lowering — used by Constructor and Match transforms that need
    /// to inspect the original `Sum` variants for layout computation.
    fn lookup_type_raw(&self, node_id: NodeId) -> Option<Type<TypeName>> {
        self.type_table.get(&node_id).map(|scheme| self.extract_monotype(scheme))
    }

    /// Recursively rewrite `Sum(variants)` types into a flattened tuple
    /// `(tag: u32, ...all_variant_payload_slots)`. Sum types do not
    /// survive past AST→TLC; downstream passes only see tuples.
    fn lower_type(ty: Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Sum(variants), _) => {
                let layout = Self::sum_layout(&variants);
                Type::Constructed(TypeName::Tuple(layout.slot_types.len()), layout.slot_types)
            }
            Type::Constructed(name, args) => {
                let lowered_args: Vec<_> = args.into_iter().map(Self::lower_type).collect();
                Type::Constructed(name, lowered_args)
            }
            Type::Variable(_) => ty,
        }
    }

    /// Compute the flattened-no-sharing layout for a sum type.
    /// Slot 0 is always the u32 tag; slots 1..end are each
    /// constructor's payloads laid out in source order with no
    /// sharing between variants. The dead slots for an inactive
    /// variant are zero-filled at construction.
    fn sum_layout(variants: &[(String, Vec<Type<TypeName>>)]) -> SumLayout {
        let tag_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut slot_types = vec![tag_ty];
        let mut constructor_info = std::collections::HashMap::new();
        for (i, (name, payload)) in variants.iter().enumerate() {
            constructor_info.insert(name.clone(), (i as u32, slot_types.len()));
            for p in payload {
                slot_types.push(Self::lower_type(p.clone()));
            }
        }
        SumLayout {
            slot_types,
            constructor_info,
        }
    }

    fn extract_monotype(&self, scheme: &polytype::TypeScheme<TypeName>) -> Type<TypeName> {
        match scheme {
            polytype::TypeScheme::Monotype(ty) => ty.clone(),
            polytype::TypeScheme::Polytype { body, .. } => self.extract_monotype(body),
        }
    }

    fn mk_term(&mut self, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind,
        }
    }

    fn mk_i32(&mut self, value: i32, span: Span) -> Term {
        self.mk_term(
            Type::Constructed(TypeName::Int(32), vec![]),
            span,
            TermKind::IntLit(value.to_string()),
        )
    }

    /// Transform an expression as a vector, converting ArrayLiteral to a VecLit term.
    fn transform_as_vector(&mut self, expr: &ast::Expression, vec_ty: Type<TypeName>) -> Term {
        let span = expr.h.span;
        match &expr.kind {
            ast::ExprKind::ArrayLiteral(elements) => {
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_vec_lit(terms, vec_ty, span)
            }
            _ => self.transform_expr(expr),
        }
    }

    /// Build a `TermKind::VecLit` from already-transformed terms.
    fn build_vec_lit_from_terms(&mut self, terms: &[Term], result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_vec_lit(terms.to_vec(), result_ty, span)
    }
}
