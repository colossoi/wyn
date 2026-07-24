//! Typed Lambda Calculus (TLC) representation.
//!
//! A minimal typed lambda calculus IR for source-level specialization.
//! Lambdas remain as values (not yet defunctionalized).

pub mod anf;
pub mod data;
mod dce;
pub mod defaults;
pub mod defunctionalize;
pub mod if_over_producer;
pub mod inline;
pub mod input_slice_bounds;
pub mod monomorphize;
pub mod ownership;
pub mod partial_eval;
pub mod patterns;
pub mod pin_entry_buffers;
pub mod reachability;
pub mod rep_specialize;
#[cfg(test)]
#[path = "rep_specialize_tests.rs"]
mod rep_specialize_tests;
pub mod run;
pub mod runtime_index_producers;
pub mod soa;
pub mod soac_anf;
mod specialize;
pub mod subst;

use crate::ast::{self, NodeId, Span, TypeName};
use crate::builtins::{by_id, catalog, BuiltinId};
use crate::error::CompilerError;
use crate::name_resolution::{NameResolution, ResolvedValueRef, SoacKind};
use crate::types::{SoacOwnership, TypeExt};
use crate::{interface, LookupMap, LookupSet, SymbolId, SymbolTable, TypeTable};
use polytype::Type;
use std::num::NonZeroU32;

// =============================================================================
// TLC phase families and checkpoints
// =============================================================================

/// A phase-selected payload container whose elements may contain tree nodes.
///
/// Rust cannot pass a generic associated type such as `F::ClosureData<T>`
/// directly as another type's generic argument. `Payload` is the small
/// type-constructor adapter that lets `Term<C, S>` receive the closure and
/// SOAC-body dimensions independently after they are peeled off a [`Family`].
pub trait Payload: Clone + std::fmt::Debug {
    type With<T: Clone + std::fmt::Debug>: Clone + std::fmt::Debug;

    fn map<T, U, M>(data: Self::With<T>, map: &mut M) -> Self::With<U>
    where
        T: Clone + std::fmt::Debug,
        U: Clone + std::fmt::Debug,
        M: FnMut(T) -> U;

    fn for_each<T, V>(data: &Self::With<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T);

    fn for_each_rev<T, V>(data: &Self::With<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T);

    fn for_each_mut<T, V>(data: &mut Self::With<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&mut T);
}

/// Whether the generic term walker should visit a term's children.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkDecision {
    /// Visit every immediate child in structural order.
    Recurse,
    /// Do not perform the walker's normal child traversal.
    ///
    /// A visitor may walk selected children itself before returning this,
    /// which supports edge-specific scope and atomic subtree recognition.
    Prune,
}

/// Preorder, read-only traversal over a TLC term tree.
///
/// Implement [`TermVisitor::visit`] for node-specific behavior. Most nodes
/// return [`WalkDecision::Recurse`]. A visitor that needs special treatment
/// for one edge can call [`Term::walk`] on the selected children itself and
/// then return [`WalkDecision::Prune`].
pub trait TermVisitor<C: Payload, S: Payload> {
    fn visit(&mut self, term: &Term<C, S>) -> WalkDecision;

    fn walk(&mut self, term: &Term<C, S>)
    where
        Self: Sized,
    {
        if self.visit(term) == WalkDecision::Recurse {
            term.for_each_child(&mut |child| self.walk(child));
        }
    }
}

impl<C, S, F> TermVisitor<C, S> for F
where
    C: Payload,
    S: Payload,
    F: FnMut(&Term<C, S>) -> WalkDecision,
{
    fn visit(&mut self, term: &Term<C, S>) -> WalkDecision {
        self(term)
    }
}

/// Whether a consuming term rewriter changed the node it was handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteDecision {
    Unchanged,
    Changed,
}

/// Consuming reconstruction of a TLC term tree.
///
/// Node-local rewrites normally run bottom-up through
/// [`TermRewriter::rewrite_node`]. Edge-sensitive passes may also use
/// [`TermRewriter::rewrite_node_before_children`]. Unchanged nodes retain their
/// IDs. When either hook or a child changes the current node, the walker
/// assigns exactly one fresh ID to it. Passes therefore express only their
/// local rewrite and cannot accidentally preserve an ID for a changed semantic
/// node.
pub trait TermRewriter<C: Payload, S: Payload> {
    /// Allocate from the program that owns the tree being rebuilt.
    fn next_term_id(&mut self) -> TermId;

    /// Apply a node-local rewrite before its children are processed.
    ///
    /// Most rewrites should use [`TermRewriter::rewrite_node`]. This hook is
    /// for transformations whose treatment of an edge determines how that
    /// child must subsequently be visited, such as resolving a direct callee
    /// before the callee term is rewritten independently. A hook that directly
    /// changes a child must also refresh that child's ID; the returned decision
    /// refreshes the current node.
    fn rewrite_node_before_children(&mut self, _term: &mut Term<C, S>) -> RewriteDecision {
        RewriteDecision::Unchanged
    }

    /// Apply a node-local rewrite after all children have been processed.
    fn rewrite_node(&mut self, _term: &mut Term<C, S>) -> RewriteDecision {
        RewriteDecision::Unchanged
    }

    /// Rewrite an exclusively owned term and report whether its subtree
    /// changed. Existing boxes and vectors are retained.
    fn rewrite_tracked(&mut self, term: &mut Term<C, S>) -> bool
    where
        Self: Sized,
    {
        let node_changed_before = self.rewrite_node_before_children(term) == RewriteDecision::Changed;
        let mut descendant_changed = false;
        term.for_each_child_mut(&mut |child| {
            descendant_changed |= self.rewrite_tracked(child);
        });
        let node_changed_after = self.rewrite_node(term) == RewriteDecision::Changed;
        let changed = node_changed_before || descendant_changed || node_changed_after;
        if changed {
            term.id = self.next_term_id();
        }
        changed
    }

    /// Consume and rebuild a term.
    fn rewrite(&mut self, mut term: Term<C, S>) -> Term<C, S>
    where
        Self: Sized,
    {
        self.rewrite_tracked(&mut term);
        term
    }
}

/// The phase-varying data structures embedded in the TLC tree.
///
/// Every associated type corresponds to an actual field stored on a tree node.
/// Structural invariants and proof-only state belong to [`Stage`] or to the
/// concrete node shape, not to empty per-node marker payloads.
pub trait Family {
    type DefinitionData: Clone + std::fmt::Debug;
    type EntryData: Clone + std::fmt::Debug;
    type ClosureData: Payload;
    type SoacBodyData: Payload;
}

/// One externally visible TLC pipeline checkpoint.
///
/// `Program<S>` is the compiler state. The stage selects both the recursive
/// tree family and the program-wide context available at that checkpoint.
pub trait Stage {
    type Family: Family;
    type GlobalContext: std::fmt::Debug;
}

/// Program-wide context selected by TLC pipeline checkpoints.
pub mod context {
    use super::*;

    /// Global state immediately after AST-to-TLC conversion.
    ///
    /// Diagnostics and the AST type table intentionally exist only at this
    /// first checkpoint. Function schemes live on definitions instead.
    #[derive(Debug)]
    pub struct TransformedGlobal {
        pub type_table: TypeTable,
        pub known_defs: LookupSet<String>,
        pub fill_hole_errors: Vec<CompilerError>,
        pub auto_storage_binding_ids: crate::IdSource<u32>,
    }

    /// Global state shared by tree-rewriting TLC checkpoints.
    #[derive(Debug, Clone)]
    pub struct RewriteGlobal {
        pub known_defs: LookupSet<String>,
        pub auto_storage_binding_ids: crate::IdSource<u32>,
    }

    /// Global state after closure conversion has consumed `known_defs`.
    #[derive(Debug, Clone)]
    pub struct PostClosureGlobal {
        pub auto_storage_binding_ids: crate::IdSource<u32>,
    }

    /// Global state retained at the TLC-to-EGIR boundary.
    #[derive(Debug, Clone)]
    pub struct BackendGlobal {
        pub auto_storage_binding_ids: crate::IdSource<u32>,
    }
}

/// Stable names for the recursive families defined by their producing passes.
pub mod family {
    pub use super::defunctionalize::ClosureConverted;
    pub use super::input_slice_bounds::InputBounded;
    pub use super::monomorphize::Monomorphic;
    pub use super::run::Polymorphic;
}

/// Stable names for checkpoints defined by their producing passes.
pub mod stage {
    pub use super::defunctionalize::Defunctionalized;
    pub use super::if_over_producer::ConditionalProducersCanonicalized;
    pub use super::inline::{GeneratedLambdasFolded, SmallInlined, SoacHelpersInlined};
    pub use super::input_slice_bounds::InputSliceBoundsInferred;
    pub use super::monomorphize::Monomorphized;
    pub use super::ownership::{OwnershipApplied, OwnershipValidated};
    pub use super::partial_eval::PartialEvaled;
    pub use super::pin_entry_buffers::BuffersPinned;
    pub use super::reachability::Reachable;
    pub use super::rep_specialize::RepSpecialized;
    pub use super::run::Transformed;
    pub use super::runtime_index_producers::RuntimeIndexProducersFloated;
    pub use super::soa::{InlinedSoaNormalized, SoaNormalized};
    pub use super::soac_anf::SoacsAnfNormalized;
}

// Named consuming TLC transitions. The stage types remain visible through
// `tlc::stage`, while callers compose the pipeline as ordinary functions.
pub use defunctionalize::run as defunctionalize;
pub use if_over_producer::run as canonicalize_conditional_producers;
pub use inline::{
    fold_generated_lambdas, run_force_soac_helpers as force_inline_soac_helpers, run_small as inline_small,
};
pub use input_slice_bounds::run as infer_input_slice_bounds;
pub use monomorphize::run as monomorphize;
pub use ownership::{apply_ownership, validate as validate_ownership};
pub use partial_eval::run as partial_eval;
pub use pin_entry_buffers::run as pin_entry_buffers;
pub use reachability::run as filter_reachable;
pub use rep_specialize::run as rep_specialize;
pub use runtime_index_producers::run as float_runtime_index_nested_producers;
pub use soa::{rerun as renormalize_inlined_soa, run as normalize_soacs};
pub use soac_anf::run as normalize_soacs_to_anf;

// =============================================================================
// Helper functions
// =============================================================================

/// A substitution from type variables to concrete TLC types.
pub(crate) type TypeSubstitution = LookupMap<polytype::Variable, Type<TypeName>>;

/// Apply a type-variable substitution everywhere a TLC type can contain a
/// nested type.
///
/// In addition to ordinary `Type::Constructed` arguments, sum variants store
/// their payload types inside `TypeName` itself. Keeping this operation here
/// prevents individual passes from silently disagreeing about that case.
pub(crate) fn apply_type_substitution(ty: &Type<TypeName>, subst: &TypeSubstitution) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Type::Constructed(name, args) => {
            let name = match name {
                TypeName::Sum(variants) => TypeName::Sum(
                    variants
                        .iter()
                        .map(|(name, fields)| {
                            (
                                name.clone(),
                                fields.iter().map(|field| apply_type_substitution(field, subst)).collect(),
                            )
                        })
                        .collect(),
                ),
                _ => name.clone(),
            };
            Type::Constructed(
                name,
                args.iter().map(|arg| apply_type_substitution(arg, subst)).collect(),
            )
        }
    }
}

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
pub fn extract_lambda_params<C: Payload, S: Payload>(
    term: &Term<C, S>,
) -> (Vec<(SymbolId, Type<TypeName>)>, Term<C, S>) {
    match &term.kind {
        TermKind::Lambda(Lambda { params, body, .. }) => (params.clone(), (**body).clone()),
        _ => (vec![], term.clone()),
    }
}

/// Build a curried function type from its parameter and return types.
pub(crate) fn curried_function_type<'a>(
    param_types: impl DoubleEndedIterator<Item = &'a Type<TypeName>>,
    return_ty: &Type<TypeName>,
) -> Type<TypeName> {
    param_types.rev().fold(return_ty.clone(), |result, param_ty| {
        Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), result])
    })
}

/// Build a lambda term from a flattened parameter list and body.
///
/// This is shared by passes that change a callable's parameter ABI. The
/// helper is generic over the two recursive payload dimensions rather than
/// depending on whichever pass first happened to need it.
pub fn rebuild_nested_lam<C: Payload, S: Payload>(
    params: &[(SymbolId, Type<TypeName>)],
    body: Term<C, S>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    let ret_ty = body.ty.clone();
    let lam_ty = curried_function_type(params.iter().map(|(_, ty)| ty), &ret_ty);
    Term {
        id: term_ids.next_id(),
        ty: lam_ty,
        span,
        kind: TermKind::Lambda(Lambda {
            params: params.to_vec(),
            body: Box::new(body),
            ret_ty,
        }),
    }
}

/// Build `App(Var(func_sym), args)` with the function-position type implied
/// by the argument and result types.
pub fn build_app_call<C: Payload, S: Payload>(
    func_sym: SymbolId,
    args: Vec<Term<C, S>>,
    result_ty: Type<TypeName>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    let fn_ty = curried_function_type(args.iter().map(|arg| &arg.ty), &result_ty);
    let func_term = Term {
        id: term_ids.next_id(),
        ty: fn_ty,
        span,
        kind: TermKind::Var(VarRef::Symbol(func_sym)),
    };

    if args.is_empty() {
        return Term {
            ty: result_ty,
            ..func_term
        };
    }

    Term {
        id: term_ids.next_id(),
        ty: result_ty,
        span,
        kind: TermKind::App {
            func: Box::new(func_term),
            args,
        },
    }
}

/// One pending `let` wrapper around a rebuilt term.
#[derive(Debug, Clone)]
pub(crate) struct LetBinding<C: Payload, S: Payload> {
    pub name: SymbolId,
    pub name_ty: Type<TypeName>,
    pub rhs: Term<C, S>,
    pub span: Span,
}

/// Wrap a term in pending lets, preserving binding order.
///
/// This is shared by normalizations that hoist producers or other expressions
/// and then materialize their plans as tree nodes.
pub(crate) fn wrap_let_bindings<C: Payload, S: Payload>(
    bindings: Vec<LetBinding<C, S>>,
    mut body: Term<C, S>,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    for binding in bindings.into_iter().rev() {
        let body_ty = body.ty.clone();
        body = Term {
            id: term_ids.next_id(),
            ty: body_ty,
            span: binding.span,
            kind: TermKind::Let {
                name: binding.name,
                name_ty: binding.name_ty,
                rhs: Box::new(binding.rhs),
                body: Box::new(body),
            },
        };
    }
    body
}

/// Clone a term subtree into the same program with fresh IDs throughout.
///
/// Ordinary `Clone` is appropriate for analysis snapshots. Any clone that is
/// inserted back into a program must use this helper so globally unique term
/// IDs remain unique.
pub fn clone_term_with_fresh_ids<C: Payload, S: Payload>(
    term: &Term<C, S>,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    fn refresh<C: Payload, S: Payload>(term: Term<C, S>, term_ids: &mut TermIdSource) -> Term<C, S> {
        let fresh_id = term_ids.next_id();
        term.map_children(fresh_id, &mut |child| refresh(child, term_ids))
    }

    refresh(term.clone(), term_ids)
}

/// Borrowed, all-levels counterpart to [`extract_lambda_params`]: walk through
/// nested `Lambda`s and return the inner non-lambda body by reference plus the
/// accumulated params. (`extract_lambda_params` clones and peels a single level;
/// this avoids the clone and handles curried nesting.)
pub(crate) fn extract_lambda_params_ref<C: Payload, S: Payload>(
    term: &Term<C, S>,
) -> (&Term<C, S>, Vec<(SymbolId, Type<TypeName>)>) {
    let mut params = Vec::new();
    let mut current = term;
    while let TermKind::Lambda(lam) = &current.kind {
        params.extend(lam.params.iter().cloned());
        current = &lam.body;
    }
    (current, params)
}

/// Count the number of nodes in a term tree.
/// Used as a size heuristic for inlining decisions.
pub fn term_size<C: Payload, S: Payload>(term: &Term<C, S>) -> usize {
    let mut count = 1; // count this node
    term.for_each_child(&mut |child| count += term_size(child));
    count
}

// =============================================================================
// TLC Terms
// =============================================================================

/// A unique identifier for TLC terms.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TermId(Option<NonZeroU32>);

impl TermId {
    /// Ephemeral adapter term that is never stored in a TLC program.
    pub const SYNTHETIC: Self = Self(None);

    /// The allocated nonzero numeric ID, or `None` for a synthetic adapter
    /// term.
    pub fn as_u32(self) -> Option<u32> {
        self.0.map(NonZeroU32::get)
    }
}

impl std::fmt::Display for TermId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_u32() {
            Some(id) => id.fmt(f),
            None => f.write_str("synthetic"),
        }
    }
}

impl std::fmt::Debug for TermId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_u32() {
            Some(id) => write!(f, "TermId({id})"),
            None => f.write_str("TermId::SYNTHETIC"),
        }
    }
}

/// The compiler-wide allocator for TLC term identifiers.
///
/// The cursor uses the same niche representation as `TermId`: nonzero values
/// are available IDs and `None` means that the finite ID space is exhausted.
#[derive(Debug, Clone)]
pub struct TermIdSource {
    next: Option<NonZeroU32>,
}

impl TermIdSource {
    pub fn new() -> Self {
        Self {
            next: Some(NonZeroU32::MIN),
        }
    }

    pub fn next_id(&mut self) -> TermId {
        let next = self.next.expect("TLC TermId space exhausted");
        self.next = next.checked_add(1);
        TermId(Some(next))
    }
}

impl Default for TermIdSource {
    fn default() -> Self {
        Self::new()
    }
}

/// A typed term in the lambda calculus.
#[derive(Debug, Clone)]
pub struct Term<C: Payload = data::Empty, S: Payload = data::Empty> {
    pub id: TermId,
    pub ty: Type<TypeName>,
    pub span: Span,
    pub kind: TermKind<C, S>,
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
        id: BuiltinId,
        overload_idx: usize,
    },
    /// Symbol-table reference — locals, user-defined functions,
    /// prelude functions, top-level constants.
    Symbol(SymbolId),
}

/// Resolve a `Var`-position term to its `BuiltinId`, if any.
///
/// Only `VarRef::Builtin { id, .. }` returns `Some(id)`. `VarRef::Symbol`
/// always returns `None` — it's a user-defined binding (local, top-level,
/// or prelude function) that may legitimately *shadow* a catalog name
/// (e.g. a user `let outer = …` reused as a SOAC-position closure
/// shadows the catalog `outer` outer-product builtin). The previous
/// catalog-by-name lookup in this branch silently mis-routed shadowing
/// user values through builtin lowering.
///
/// Catalog calls reach TLC as `VarRef::Builtin` end-to-end:
/// `NameResolution` classifies AST identifiers, synthesizers
/// (`buffer_specialize::make_app`, `parallelize::intrinsic_term`,
/// etc.) emit `Var(Builtin)` directly, and `transform_expr`'s
/// Identifier path consults the `NameResolution` side table.
pub fn var_term_builtin_id<C: Payload, S: Payload>(
    term: &Term<C, S>,
    _symbols: &SymbolTable,
) -> Option<BuiltinId> {
    match &term.kind {
        TermKind::Var(VarRef::Builtin { id, .. }) => Some(*id),
        TermKind::Var(VarRef::Symbol(_)) => None,
        _ => None,
    }
}

/// The kind of term.
#[derive(Debug, Clone)]
pub enum TermKind<C: Payload = data::Empty, S: Payload = data::Empty> {
    /// Variable reference.
    Var(VarRef),

    /// Binary operator as a value: +, -, *, /, ==, etc.
    BinOp(ast::BinaryOp),

    /// Unary operator as a value: -, !
    UnOp(ast::UnaryOp),

    /// Lambda abstraction (structured).
    Lambda(Lambda<C, S>),

    /// A closure-converted callable value with its lexical environment stored
    /// at the value's tree position rather than in a symbol-keyed side table.
    Closure(C::With<Term<C, S>>),

    /// Application: f(a, b, c) — always fully applied.
    App {
        func: Box<Term<C, S>>,
        args: Vec<Term<C, S>>,
    },

    /// Let binding: let x:T = rhs in body
    Let {
        name: SymbolId,
        name_ty: Type<TypeName>,
        rhs: Box<Term<C, S>>,
        body: Box<Term<C, S>>,
    },

    /// Integer literal.
    IntLit(String),

    /// Float literal.
    FloatLit(f32),

    /// Boolean literal.
    BoolLit(bool),

    /// Unit literal: `()`.
    UnitLit,

    /// Numeric type coercion: `expr :> target_ty`.
    Coerce {
        inner: Box<Term<C, S>>,
        target_ty: Type<TypeName>,
    },

    /// External function reference (linked SPIR-V).
    /// The string is the linkage name for spirv-link.
    /// The Wyn-visible name comes from the parent Def.
    Extern(String),

    /// Conditional: if cond then t else e
    If {
        cond: Box<Term<C, S>>,
        then_branch: Box<Term<C, S>>,
        else_branch: Box<Term<C, S>>,
    },

    /// Loop construct (mirrors MIR::Loop).
    Loop {
        /// The loop accumulator variable name.
        loop_var: SymbolId,
        /// Type of the loop variable.
        loop_var_ty: Type<TypeName>,
        /// Initial value for the accumulator.
        init: Box<Term<C, S>>,
        /// Bindings that extract from loop_var (e.g., for tuple destructuring).
        /// Each is (name, type, extraction_expr).
        init_bindings: Vec<(SymbolId, Type<TypeName>, Term<C, S>)>,
        /// The iteration form.
        kind: LoopKind<C, S>,
        /// Loop body expression.
        body: Box<Term<C, S>>,
    },

    /// First-class SOAC (second-order array combinator).
    Soac(SoacOp<C, S>),

    /// Array producer expression.
    ArrayExpr(ArrayExpr<C, S>),

    /// Tuple constructor: `(a, b, c)` or record literal in tuple form.
    Tuple(Vec<Term<C, S>>),

    /// Tuple projection: `t.idx`. `idx` is the structural field index.
    TupleProj {
        tuple: Box<Term<C, S>>,
        idx: usize,
    },

    /// Array indexing: `arr[idx]`. The result aliases `array` for
    /// ownership analysis.
    Index {
        array: Box<Term<C, S>>,
        index: Box<Term<C, S>>,
    },

    /// Vector literal: `@[x, y, z, w]`.
    VecLit(Vec<Term<C, S>>),
}

/// The kind of loop (mirrors MIR::LoopKind).
#[derive(Debug, Clone)]
pub enum LoopKind<C: Payload = data::Empty, S: Payload = data::Empty> {
    /// For loop over an array: `for x in arr`.
    For {
        var: SymbolId,
        var_ty: Type<TypeName>,
        iter: Box<Term<C, S>>,
    },
    /// For loop with range bound: `for i < n`.
    ForRange {
        var: SymbolId,
        var_ty: Type<TypeName>,
        bound: Box<Term<C, S>>,
    },
    /// While loop: `while cond`.
    While {
        cond: Box<Term<C, S>>,
    },
}

// =============================================================================
// SOAC Types
// =============================================================================

/// A structured lambda with explicit parameters and return type.
#[derive(Debug, Clone)]
pub struct Lambda<C: Payload = data::Empty, S: Payload = data::Empty> {
    pub params: Vec<(SymbolId, Type<TypeName>)>,
    pub body: Box<Term<C, S>>,
    pub ret_ty: Type<TypeName>,
}

/// Lambda plus phase-varying SOAC-body data.
///
/// Before defunctionalization `data` is `()`. Closure-converted families store
/// an `ExplicitCaptures` value containing the environment threaded into the
/// body at SOAC lowering time.
#[derive(Debug, Clone)]
pub struct SoacBody<C: Payload = data::Empty, S: Payload = data::Empty> {
    pub lam: Lambda<C, S>,
    pub data: S::With<(SymbolId, Type<TypeName>, Term<C, S>)>,
}

/// An array-producing expression.
//
// One variant is phase-scoped — it is only valid in part of the
// pipeline, with the boundaries enforced at runtime via `unreachable!`
// in passes that can't see them:
//
// * `Zip` is constructed in `transform_soac_zip`, absorbed at
//   `transform_soac_map`, and anything escaping as a standalone term is
//   rewritten to `_w_tuple(...)` by `tlc::soa`. Post-SoA passes don't
//   meaningfully encounter it.
//
// A future refactor could narrow these via per-phase enum splits
// (`PreSoaArrayExpr` / `PostSoaArrayExpr`, etc.) so the invariants are
// type-enforced rather than runtime-checked. The cost is non-trivial —
// `Term`/`TermKind`/`SoacOp`/`Lambda` would need a phase parameter, so
// it cascades across the whole TLC pipeline. The current runtime checks
// are fine until that pays for itself.
#[derive(Debug, Clone)]
pub enum ArrayExpr<C: Payload = data::Empty, S: Payload = data::Empty> {
    /// A named array value — the canonical atom shape for a SOAC input,
    /// carrying the variable reference and the array's type. A SOAC consumes a
    /// producer only by name: the producer is let-bound and referenced here,
    /// so a producer cannot sit directly in an input position.
    Var(VarRef, Type<TypeName>),
    /// Logical zip — not materialized, consumed by enclosing Map.
    Zip(Vec<ArrayExpr<C, S>>),
    /// Literal small array.
    Literal(Vec<Term<C, S>>),
    /// Range / iota. `step` defaults to 1 when `None`.
    Range {
        start: Box<Term<C, S>>,
        len: Box<Term<C, S>>,
        step: Option<Box<Term<C, S>>>,
    },
}

impl<C: Payload, S: Payload> ArrayExpr<C, S> {
    /// `Var(Symbol(sym), _)` → `Some(sym)`. The canonical "named SOAC input"
    /// shape used by ownership and TLC→EGIR conversion to recognize an entry
    /// parameter, intermediate, or let-bound result of a prior SOAC.
    pub fn as_named_ref(&self) -> Option<SymbolId> {
        if let ArrayExpr::Var(VarRef::Symbol(sym), _) = self {
            return Some(*sym);
        }
        None
    }

    /// The array type of this input atom. `Var` carries its type verbatim
    /// (consumers that need the bare type strip uniqueness themselves);
    /// `Literal` is a composite array sized to its element count; `Range` a
    /// virtual array; `Zip` a virtual array of the tuple of its children's element
    /// types. Inverse of the per-variant `array_expr_type` recomputation that
    /// EGIR lowering and the representation passes each used to carry.
    pub fn array_type(&self) -> Type<TypeName> {
        use crate::types::{make_array1, no_buffer};
        let virtual_array = |elem: Type<TypeName>| {
            make_array1(
                elem,
                Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                no_buffer(),
            )
        };
        match self {
            ArrayExpr::Var(_, ty) => ty.clone(),
            ArrayExpr::Literal(terms) => make_array1(
                terms
                    .first()
                    .map(|t| t.ty.clone())
                    .unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![])),
                Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
                Type::Constructed(TypeName::Size(terms.len()), vec![]),
                no_buffer(),
            ),
            ArrayExpr::Range { start, .. } => virtual_array(start.ty.clone()),
            ArrayExpr::Zip(children) => {
                let elems: Vec<Type<TypeName>> = children
                    .iter()
                    .map(|c| {
                        crate::types::array_elem(&c.array_type())
                            .cloned()
                            .unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]))
                    })
                    .collect();
                virtual_array(Type::Constructed(TypeName::Tuple(elems.len()), elems))
            }
        }
    }
}

/// A second-order array combinator (SOAC) operation.
///
/// `Reduce`, `Scan`, and `ReduceByIndex` parallelize freely on
/// the assumption that their reducer is associative (Futhark convention:
/// the caller asserts associativity by using the SOAC). The compiler
/// never verifies this — float reductions are reordered just like int
/// reductions, so the user accepts non-determinism on `+/f32` etc.
#[derive(Debug, Clone)]
pub enum SoacOp<C: Payload = data::Empty, S: Payload = data::Empty> {
    Map {
        lam: SoacBody<C, S>,
        /// Parallel inputs. `inputs.len() == lam.lam.params.len()`.
        inputs: Vec<ArrayExpr<C, S>>,
        /// Logical uniqueness fact from TLC ownership; EGIR decides whether
        /// the candidate becomes an in-place write.
        destination: SoacOwnership,
    },
    Reduce {
        op: SoacBody<C, S>,
        ne: Box<Term<C, S>>,
        input: ArrayExpr<C, S>,
    },
    Scan {
        /// Pure associative combiner `(acc, x) -> acc'`.
        op: SoacBody<C, S>,
        ne: Box<Term<C, S>>,
        input: ArrayExpr<C, S>,
        /// TLC may mark a pointwise-safe uniquely owned input as
        /// `UniqueInput`; EGIR resolves post-fusion liveness and routing into
        /// the physical destination.
        destination: SoacOwnership,
    },
    Filter {
        pred: SoacBody<C, S>,
        input: ArrayExpr<C, S>,
        /// TLC may mark a pointwise-safe uniquely owned input as
        /// `UniqueInput`; EGIR resolves post-fusion liveness and routing into
        /// the physical destination.
        destination: SoacOwnership,
    },
    /// Indexed writes into `dest`: over the parallel `inputs`, `lam` yields an
    /// `(index, value)` pair per element, written as `dest[index] = value`.
    /// Plain `scatter(dest, is, vs)` carries the identity envelope
    /// `lam = λ(i, v) → (i, v)` with `inputs = [is, vs]`. EGIR receives that
    /// callable ABI and owns any later producer composition.
    Scatter {
        dest: Place,
        lam: SoacBody<C, S>,
        inputs: Vec<ArrayExpr<C, S>>,
    },
    // TODO(reduce_by_index): produced by to_tlc but EGIR rejects
    // (`egir::from_tlc::convert_soac`). Sequential lowering would be a
    // straightforward read-combine-write loop; the parallel path needs
    // atomic-op emission in the backends, which doesn't exist yet.
    ReduceByIndex {
        dest: Place,
        op: SoacBody<C, S>,
        ne: Box<Term<C, S>>,
        indices: ArrayExpr<C, S>,
        values: ArrayExpr<C, S>,
    },
}

/// Destination-passing for scatter / reduce_by_index.
#[derive(Debug, Clone)]
pub struct Place {
    pub id: SymbolId,
    pub elem_ty: Type<TypeName>,
}

// =============================================================================
// TLC Program
// =============================================================================

/// Metadata about how a definition should be lowered to MIR.
#[derive(Debug, Clone)]
pub enum DefMeta<E: Clone + std::fmt::Debug> {
    /// A regular function or constant.
    Function,
    /// A shader entry point with phase-varying entry-local information.
    EntryPoint(EntryPoint<E>),
    /// A lifted lambda produced during defunctionalization. Marks the def so
    /// later passes (inlining, etc.) can recognise it structurally
    /// instead of sniffing the symbol name.
    LiftedLambda,
}

/// Source declaration and compiler information for one shader entry point.
#[derive(Debug, Clone)]
pub struct EntryPoint<E: Clone + std::fmt::Debug> {
    pub declaration: Box<interface::EntryDecl>,
    pub data: E,
}

/// A top-level definition in TLC.
#[derive(Debug, Clone)]
pub struct Def<F: Family> {
    pub data: F::DefinitionData,
    pub name: SymbolId,
    pub ty: Type<TypeName>,
    pub body: Term<F::ClosureData, F::SoacBodyData>,
    pub meta: DefMeta<F::EntryData>,
    /// Number of arguments this function expects.
    pub arity: usize,
    pub param_diets: Vec<crate::types::Diet>,
    pub return_diet: crate::types::Diet,
}

/// A TLC program (collection of definitions).
#[derive(Debug, Clone)]
pub struct Program<S: Stage> {
    pub defs: Vec<Def<S::Family>>,
    /// Symbol table: maps SymbolId to original name (for errors/debugging).
    pub symbols: SymbolTable,
    /// Canonical function name → def SymbolId mapping.
    /// Used by fusion to resolve call-site SymbolIds to def SymbolIds.
    pub def_syms: LookupMap<String, SymbolId>,
    /// The sole allocator for terms added to this program.
    term_ids: TermIdSource,
    /// Program-wide state available at this exact pipeline checkpoint.
    pub global_context: S::GlobalContext,
}

impl<S: Stage> Program<S> {
    /// Assemble a TLC program from nodes and the allocator that created them.
    pub(crate) fn from_parts(
        defs: Vec<Def<S::Family>>,
        symbols: SymbolTable,
        def_syms: LookupMap<String, SymbolId>,
        term_ids: TermIdSource,
        global_context: S::GlobalContext,
    ) -> Self {
        Self {
            defs,
            symbols,
            def_syms,
            term_ids,
            global_context,
        }
    }

    /// Allocate a term ID from this program's unique ID domain.
    pub fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    /// Change only the program's typestate when both stages select exactly the
    /// same stored tree family and global context.
    ///
    /// Read-only validation passes use this after proving their invariant. No
    /// tree nodes, vectors, symbol tables, or allocator state are rebuilt.
    pub fn into_stage<T>(self) -> Program<T>
    where
        T: Stage<Family = S::Family, GlobalContext = S::GlobalContext>,
    {
        self.map_global_context(std::convert::identity)
    }

    /// Change the program-wide context without rebuilding the selected tree
    /// family.
    ///
    /// This is the consuming transition for passes that update or narrow only
    /// global state. The definition vector, definitions, and term trees all
    /// move directly into the destination program.
    pub fn map_global_context<T>(
        self,
        map_global: impl FnOnce(S::GlobalContext) -> T::GlobalContext,
    ) -> Program<T>
    where
        T: Stage<Family = S::Family>,
    {
        let Program {
            defs,
            symbols,
            def_syms,
            term_ids,
            global_context,
        } = self;
        Program {
            defs,
            symbols,
            def_syms,
            term_ids,
            global_context: map_global(global_context),
        }
    }

    /// Assert no def body contains nested Apps.
    pub fn assert_flat_apps(&self) {
        for def in &self.defs {
            let name = self.symbols.get(def.name).cloned().unwrap_or_else(|| format!("{:?}", def.name));
            def.body.assert_flat_apps_in(&name);
        }
    }

    /// Index of arity-0 `Function`-meta defs by their source name. These are
    /// the top-level value bindings (`def foo = expr`) — candidates for being
    /// hoisted to program scope as pure constants, and the resolution target
    /// when a downstream `Var(sym)` reference has lost its symbol context and
    /// needs to look the def up by name.
    pub fn value_defs_by_name(&self) -> LookupMap<String, SymbolId> {
        self.defs
            .iter()
            .filter(|d| d.arity == 0 && matches!(&d.meta, DefMeta::Function))
            .filter_map(|d| self.symbols.get(d.name).map(|n| (n.clone(), d.name)))
            .collect()
    }
}

/// Parts of a TLC program, without the symbol table.
/// Returned by `Transformer::transform_program` when the caller owns the symbol table.
#[derive(Debug, Clone)]
pub struct ProgramParts<F: Family> {
    pub defs: Vec<Def<F>>,
}

impl<F: Family> ProgramParts<F> {
    /// Combine with a symbol table to create a complete Program.
    pub fn with_symbols<S>(
        self,
        symbols: SymbolTable,
        def_syms: LookupMap<String, SymbolId>,
        term_ids: TermIdSource,
        global_context: S::GlobalContext,
    ) -> Program<S>
    where
        S: Stage<Family = F>,
    {
        Program::from_parts(self.defs, symbols, def_syms, term_ids, global_context)
    }
}

impl<S: Stage> Program<S> {
    /// Infallible form of [`Program::try_rebuild`].
    pub fn rebuild<T>(
        self,
        map_global: impl FnOnce(S::GlobalContext) -> T::GlobalContext,
        mut map_def: impl FnMut(Def<S::Family>, &mut TermIdSource) -> Def<T::Family>,
    ) -> Program<T>
    where
        T: Stage,
    {
        let Program {
            defs,
            symbols,
            def_syms,
            mut term_ids,
            global_context,
        } = self;
        Program {
            defs: defs.into_iter().map(|def| map_def(def, &mut term_ids)).collect(),
            symbols,
            def_syms,
            term_ids,
            global_context: map_global(global_context),
        }
    }

    /// Consume this program into another phase, preserving the symbol tables
    /// and the single program-owned term-ID allocator.
    ///
    /// The caller supplies only the pieces that can vary across a phase
    /// boundary: global data and definition reconstruction. A definition
    /// mapper can move unchanged subtrees directly whenever the source and
    /// destination field types agree.
    pub fn try_rebuild<T, E>(
        self,
        map_global: impl FnOnce(S::GlobalContext) -> Result<T::GlobalContext, E>,
        mut map_def: impl FnMut(Def<S::Family>, &mut TermIdSource) -> Result<Def<T::Family>, E>,
    ) -> Result<Program<T>, E>
    where
        T: Stage,
    {
        let Program {
            defs,
            symbols,
            def_syms,
            mut term_ids,
            global_context,
        } = self;
        let global_context = map_global(global_context)?;
        let defs =
            defs.into_iter().map(|def| map_def(def, &mut term_ids)).collect::<Result<Vec<_>, _>>()?;
        Ok(Program {
            defs,
            symbols,
            def_syms,
            term_ids,
            global_context,
        })
    }
}

// =============================================================================
// Generic child traversal
// =============================================================================

impl<C: Payload, S: Payload> Term<C, S> {
    /// Walk this term and its descendants in preorder.
    pub fn walk<V>(&self, visitor: &mut V)
    where
        V: TermVisitor<C, S>,
    {
        visitor.walk(self);
    }

    /// Consume and rebuild this term with a bottom-up rewriter.
    pub fn rewrite<R>(self, rewriter: &mut R) -> Self
    where
        R: TermRewriter<C, S>,
    {
        rewriter.rewrite(self)
    }

    /// Rewrite every type stored in this term tree.
    ///
    /// This includes the term result types and the types stored in binders,
    /// coercions, loop metadata, SOAC capture ABIs, destinations, and named
    /// array-input atoms. Passes provide only the type-local operation.
    ///
    /// Every visited term receives a fresh ID because its stored type may
    /// change. This deliberately errs on the side of invalidating IDs rather
    /// than asking each caller to determine which type rewrites were no-ops.
    pub fn rewrite_types<M>(&mut self, term_ids: &mut TermIdSource, map: &mut M)
    where
        M: FnMut(&Type<TypeName>) -> Type<TypeName>,
    {
        self.ty = map(&self.ty);
        match &mut self.kind {
            TermKind::Lambda(lambda) => rewrite_lambda_types(lambda, map),
            TermKind::Let { name_ty, .. } => *name_ty = map(name_ty),
            TermKind::Coerce { target_ty, .. } => *target_ty = map(target_ty),
            TermKind::Loop {
                loop_var_ty,
                init_bindings,
                kind,
                ..
            } => {
                *loop_var_ty = map(loop_var_ty);
                for (_, ty, _) in init_bindings {
                    *ty = map(ty);
                }
                match kind {
                    LoopKind::For { var_ty, .. } | LoopKind::ForRange { var_ty, .. } => {
                        *var_ty = map(var_ty);
                    }
                    LoopKind::While { .. } => {}
                }
            }
            TermKind::Soac(soac) => rewrite_soac_types(soac, map),
            TermKind::ArrayExpr(array) => rewrite_array_expr_types(array, map),
            _ => {}
        }
        self.for_each_child_mut(&mut |child| child.rewrite_types(term_ids, map));
        self.id = term_ids.next_id();
    }

    /// If this term is `App { func: Var(sym), args }` — a direct named
    /// call — return `Some((sym, args))`. Returns `None` for operator
    /// dispatch (`App { BinOp/UnOp/Extern, .. }`), partial applications,
    /// non-call terms, etc.
    ///
    /// Use this in post-defunctionalize passes and backends to one-step
    /// destructure named calls instead of nesting two `match`es.
    pub fn as_direct_call(&self) -> Option<(SymbolId, &[Term<C, S>])> {
        match &self.kind {
            TermKind::App { func, args } => match &func.kind {
                TermKind::Var(VarRef::Symbol(sym)) => Some((*sym, args.as_slice())),
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

    /// Apply `f` to every immediate `Term` child, returning a rebuilt `Term`
    /// with the caller-provided fresh ID. Recurses into Lambda, SoacOp,
    /// ArrayExpr, LoopKind, and Place sub-structures.
    ///
    /// This is the single place that knows the shape of TermKind — passes
    /// that need a uniform bottom-up or top-down walk can use this instead
    /// of hand-rolling a match over every variant.
    pub fn map_children<F>(self, fresh_id: TermId, f: &mut F) -> Self
    where
        F: FnMut(Term<C, S>) -> Term<C, S>,
    {
        let kind = match self.kind {
            // Leaves — no Term children
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => self.kind,

            TermKind::Closure(data) => TermKind::Closure(C::map(data, f)),

            TermKind::Coerce { inner, target_ty } => TermKind::Coerce {
                inner: Box::new(f(*inner)),
                target_ty,
            },

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

        Term {
            id: fresh_id,
            kind,
            ..self
        }
    }

    /// Visit every immediate `Term` child by reference. This is the by-ref
    /// counterpart to `map_children` — use it for analysis passes that
    /// inspect without transforming.
    pub fn for_each_child<F>(&self, f: &mut F)
    where
        F: FnMut(&Term<C, S>),
    {
        match &self.kind {
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {}

            TermKind::Closure(data) => C::for_each(data, f),

            TermKind::Coerce { inner, .. } => f(inner),

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

    /// Visit every immediate `Term` child by mutable reference — the in-place
    /// counterpart to `map_children`. The method itself writes nothing; it
    /// hands each child out as `&mut Term` so the callback can rewrite (or
    /// wholesale replace) children without rebuilding the tree.
    pub fn for_each_child_mut<F>(&mut self, f: &mut F)
    where
        F: FnMut(&mut Term<C, S>),
    {
        match &mut self.kind {
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {}

            TermKind::Closure(data) => C::for_each_mut(data, f),

            TermKind::Coerce { inner, .. } => f(inner),

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

            TermKind::Lambda(lam) => visit_lambda_children_mut(lam, f),

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
                visit_loop_kind_children_mut(kind, f);
                f(body);
            }

            TermKind::Soac(soac) => visit_soac_children_mut(soac, f),
            TermKind::ArrayExpr(ae) => visit_array_expr_children_mut(ae, f),

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

fn rewrite_lambda_types<C, S, M>(lambda: &mut Lambda<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    for (_, ty) in &mut lambda.params {
        *ty = map(ty);
    }
    lambda.ret_ty = map(&lambda.ret_ty);
}

fn rewrite_soac_body_types<C, S, M>(body: &mut SoacBody<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    rewrite_lambda_types(&mut body.lam, map);
    S::for_each_mut(&mut body.data, &mut |(_, ty, _)| *ty = map(ty));
}

fn rewrite_soac_types<C, S, M>(soac: &mut SoacOp<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            rewrite_soac_body_types(lam, map);
            for input in inputs {
                rewrite_array_expr_types(input, map);
            }
        }
        SoacOp::Reduce { op, input, .. } | SoacOp::Scan { op, input, .. } => {
            rewrite_soac_body_types(op, map);
            rewrite_array_expr_types(input, map);
        }
        SoacOp::Filter { pred, input, .. } => {
            rewrite_soac_body_types(pred, map);
            rewrite_array_expr_types(input, map);
        }
        SoacOp::Scatter {
            dest, lam, inputs, ..
        } => {
            dest.elem_ty = map(&dest.elem_ty);
            rewrite_soac_body_types(lam, map);
            for input in inputs {
                rewrite_array_expr_types(input, map);
            }
        }
        SoacOp::ReduceByIndex {
            dest,
            op,
            indices,
            values,
            ..
        } => {
            dest.elem_ty = map(&dest.elem_ty);
            rewrite_soac_body_types(op, map);
            rewrite_array_expr_types(indices, map);
            rewrite_array_expr_types(values, map);
        }
    }
}

fn rewrite_array_expr_types<C, S, M>(array: &mut ArrayExpr<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    match array {
        ArrayExpr::Var(_, ty) => *ty = map(ty),
        ArrayExpr::Zip(parts) => {
            for part in parts {
                rewrite_array_expr_types(part, map);
            }
        }
        ArrayExpr::Literal(_) | ArrayExpr::Range { .. } => {}
    }
}

fn visit_lambda_children<C, S, V>(lam: &Lambda<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    f(&lam.body);
}

fn visit_soac_body_children<C, S, V>(sb: &SoacBody<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    visit_lambda_children(&sb.lam, f);
    S::for_each(&sb.data, &mut |capture| f(&capture.2));
}

fn visit_soac_children<C, S, V>(soac: &SoacOp<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
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
        SoacOp::Scan { op, ne, input, .. } => {
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(input, f);
        }
        SoacOp::Filter { pred, input, .. } => {
            visit_soac_body_children(pred, f);
            visit_array_expr_children(input, f);
        }
        SoacOp::Scatter { lam, inputs, .. } => {
            visit_soac_body_children(lam, f);
            for input in inputs {
                visit_array_expr_children(input, f);
            }
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(indices, f);
            visit_array_expr_children(values, f);
        }
    }
}

fn visit_array_expr_children<C, S, V>(ae: &ArrayExpr<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    match ae {
        // Visit the named input as a var term, so analyses (free-var / capture
        // collection, etc.) see the reference.
        ArrayExpr::Var(vr, ty) => f(&synthetic_atom_var_term(*vr, ty.clone())),
        ArrayExpr::Zip(aes) => {
            for ae in aes {
                visit_array_expr_children(ae, f);
            }
        }
        ArrayExpr::Literal(terms) => {
            for t in terms {
                f(t);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            f(start);
            f(len);
            if let Some(s) = step {
                f(s);
            }
        }
    }
}

fn visit_loop_kind_children<C, S, V>(kind: &LoopKind<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    match kind {
        LoopKind::For { iter, .. } => f(iter),
        LoopKind::ForRange { bound, .. } => f(bound),
        LoopKind::While { cond } => f(cond),
    }
}

fn visit_lambda_children_mut<C, S, V>(lam: &mut Lambda<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    f(&mut lam.body);
}

fn visit_soac_body_children_mut<C, S, V>(sb: &mut SoacBody<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    visit_lambda_children_mut(&mut sb.lam, f);
    S::for_each_mut(&mut sb.data, &mut |capture| f(&mut capture.2));
}

fn visit_soac_children_mut<C, S, V>(soac: &mut SoacOp<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            visit_soac_body_children_mut(lam, f);
            for ae in inputs {
                visit_array_expr_children_mut(ae, f);
            }
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            visit_soac_body_children_mut(op, f);
            f(ne);
            visit_array_expr_children_mut(input, f);
        }
        SoacOp::Scan { op, ne, input, .. } => {
            visit_soac_body_children_mut(op, f);
            f(ne);
            visit_array_expr_children_mut(input, f);
        }
        SoacOp::Filter { pred, input, .. } => {
            visit_soac_body_children_mut(pred, f);
            visit_array_expr_children_mut(input, f);
        }
        SoacOp::Scatter { lam, inputs, .. } => {
            visit_soac_body_children_mut(lam, f);
            for input in inputs {
                visit_array_expr_children_mut(input, f);
            }
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            visit_soac_body_children_mut(op, f);
            f(ne);
            visit_array_expr_children_mut(indices, f);
            visit_array_expr_children_mut(values, f);
        }
    }
}

fn visit_array_expr_children_mut<C, S, V>(ae: &mut ArrayExpr<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    match ae {
        // Feed the named input through a reconstructed var term (as
        // `map_array_expr_children` does), so rewrites that rename or replace
        // a variable reach SOAC inputs, then re-atomize the result.
        ArrayExpr::Var(vr, ty) => {
            let mut tmp = synthetic_atom_var_term(*vr, ty.clone());
            f(&mut tmp);
            *ae = term_as_input_atom(tmp);
        }
        ArrayExpr::Zip(aes) => {
            for ae in aes {
                visit_array_expr_children_mut(ae, f);
            }
        }
        ArrayExpr::Literal(terms) => {
            for t in terms {
                f(t);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            f(start);
            f(len);
            if let Some(s) = step {
                f(s);
            }
        }
    }
}

fn visit_loop_kind_children_mut<C, S, V>(kind: &mut LoopKind<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    match kind {
        LoopKind::For { iter, .. } => f(iter),
        LoopKind::ForRange { bound, .. } => f(bound),
        LoopKind::While { cond } => f(cond),
    }
}

fn map_lambda_children<C, S, M>(lam: Lambda<C, S>, f: &mut M) -> Lambda<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    Lambda {
        body: Box::new(f(*lam.body)),
        ..lam
    }
}

fn map_soac_body_children<C, S, M>(sb: SoacBody<C, S>, f: &mut M) -> SoacBody<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    SoacBody {
        lam: map_lambda_children(sb.lam, f),
        data: S::map(sb.data, &mut |(symbol, ty, term)| (symbol, ty, f(term))),
    }
}

fn map_soac_children<C, S, M>(soac: SoacOp<C, S>, f: &mut M) -> SoacOp<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    match soac {
        SoacOp::Map {
            lam,
            inputs,
            destination,
        } => SoacOp::Map {
            lam: map_soac_body_children(lam, f),
            inputs: inputs.into_iter().map(|ae| map_array_expr_children(ae, f)).collect(),
            destination,
        },
        SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            input: map_array_expr_children(input, f),
        },
        SoacOp::Scan {
            op,
            ne,
            input,
            destination,
        } => SoacOp::Scan {
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            input: map_array_expr_children(input, f),
            destination,
        },
        SoacOp::Filter {
            pred,
            input,
            destination,
        } => SoacOp::Filter {
            pred: map_soac_body_children(pred, f),
            input: map_array_expr_children(input, f),
            destination,
        },
        SoacOp::Scatter { dest, lam, inputs } => SoacOp::Scatter {
            dest,
            lam: map_soac_body_children(lam, f),
            inputs: inputs.into_iter().map(|ae| map_array_expr_children(ae, f)).collect(),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
        } => SoacOp::ReduceByIndex {
            dest,
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            indices: map_array_expr_children(indices, f),
            values: map_array_expr_children(values, f),
        },
    }
}

/// Build a var-reference `Term` for a SOAC-input atom (`Var(vr, ty)`), for
/// passes that walk a SOAC input as a `Term` (ownership analysis, substitution,
/// EGIR conversion). A SOAC-input atom has no span of its own, but it still
/// gets a real pass-local `TermId` so synthetic terms never alias on a sentinel
/// placeholder.
pub fn atom_var_term<C: Payload, S: Payload>(
    vr: VarRef,
    ty: Type<TypeName>,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    Term {
        id: term_ids.next_id(),
        ty,
        span: Span::new(0, 0, 0, 0),
        kind: TermKind::Var(vr),
    }
}

pub(crate) fn synthetic_atom_var_term<C: Payload, S: Payload>(
    vr: VarRef,
    ty: Type<TypeName>,
) -> Term<C, S> {
    Term {
        id: TermId::SYNTHETIC,
        ty,
        span: Span::new(0, 0, 0, 0),
        kind: TermKind::Var(vr),
    }
}

/// Convert a `Term` standing in a SOAC-input position into an input atom: a bare
/// name stays a name; an array expression is itself an atom; a tuple-of-arrays
/// (the SoA form of a `zip`) becomes a `Zip` of atoms. Used where a `Term`-level
/// rewrite (substitution, inlining) lands on a SOAC input and the result must be
/// re-atomized. A producer term has no atomic form for an input position.
pub fn term_as_input_atom<C: Payload, S: Payload>(t: Term<C, S>) -> ArrayExpr<C, S> {
    match t.kind {
        TermKind::Var(vr) => ArrayExpr::Var(vr, t.ty),
        TermKind::ArrayExpr(ae) => ae,
        TermKind::Tuple(elems) => ArrayExpr::Zip(elems.into_iter().map(term_as_input_atom).collect()),
        other => panic!("ANF: cannot use a non-atom term as a SOAC input: {other:?}"),
    }
}

/// Peel leading `let` bindings off `term`, returning them (outermost first)
/// plus the inner non-`let` term. Lets a SOAC transform lift binding lets above
/// the SOAC so the SOAC input stays a bare zip / atom (ANF).
fn peel_lets<C: Payload, S: Payload>(
    mut term: Term<C, S>,
) -> (Vec<(SymbolId, Type<TypeName>, Term<C, S>)>, Term<C, S>) {
    let mut binds = Vec::new();
    while let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = term.kind
    {
        binds.push((name, name_ty, *rhs));
        term = *body;
    }
    (binds, term)
}

fn map_array_expr_children<C, S, M>(ae: ArrayExpr<C, S>, f: &mut M) -> ArrayExpr<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    match ae {
        // Apply `f` to the named input through a reconstructed var term, so
        // substitutions that rename (or inline) a variable reach SOAC inputs,
        // then re-atomize the result.
        ArrayExpr::Var(vr, ty) => term_as_input_atom(f(synthetic_atom_var_term(vr, ty))),
        ArrayExpr::Zip(aes) => {
            ArrayExpr::Zip(aes.into_iter().map(|ae| map_array_expr_children(ae, f)).collect())
        }
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(terms.into_iter().map(f).collect()),
        ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
            start: Box::new(f(*start)),
            len: Box::new(f(*len)),
            step: step.map(|s| Box::new(f(*s))),
        },
    }
}

fn map_loop_kind_children<C, S, M>(kind: LoopKind<C, S>, f: &mut M) -> LoopKind<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
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

// =============================================================================
// AST to TLC Transformation
// =============================================================================

/// A pending let-binding to be applied after all lambdas are created.
#[derive(Debug, Clone)]
pub(super) struct PendingBinding {
    pub(super) name: SymbolId,
    pub(super) ty: Type<TypeName>,
    pub(super) expr: Term,
}

/// Flattened-no-sharing layout for a structural sum type. Computed
/// once per sum and then consulted by the Constructor and Match
/// transforms for tag values and per-payload slot offsets.
pub(super) struct SumLayout {
    /// All slot types of the lowered tuple, lowered. Index 0 is the
    /// u32 tag; indices 1.. are the variant payloads concatenated
    /// in source order.
    pub(super) slot_types: Vec<Type<TypeName>>,
    /// For each constructor name: its tag value (source-order index)
    /// and the starting slot index of its payload in `slot_types`.
    pub(super) constructor_info: LookupMap<String, (u32, usize)>,
}

/// Context for transforming AST to TLC.
pub struct Transformer<'a> {
    type_table: &'a TypeTable,
    pub(super) term_ids: &'a mut TermIdSource,
    /// Shared symbol table: maps SymbolId to original name (for errors/debugging).
    symbols: &'a mut SymbolTable,
    /// Current scope for name resolution: maps string name to SymbolId.
    scope: LookupMap<String, SymbolId>,
    /// Top-level symbols that persist across function transformations.
    /// This ensures function references use the same SymbolId as the Def.
    /// Shared across all transformers via mutable reference.
    top_level_symbols: &'a mut LookupMap<String, SymbolId>,
    /// Side table from name resolution: AST NodeId → BuiltinId for
    /// catalog-resolved identifiers. Lets `Var`-position idents be
    /// classified as `VarRef::Builtin(id)` directly without round-
    /// tripping through name strings.
    name_resolution: &'a NameResolution,
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
    fill_hole_errors: &'a mut Vec<CompilerError>,
}

impl<'a> Transformer<'a> {
    pub fn new(
        type_table: &'a TypeTable,
        symbols: &'a mut SymbolTable,
        top_level_symbols: &'a mut LookupMap<String, SymbolId>,
        name_resolution: &'a NameResolution,
        fill_holes: bool,
        fill_hole_errors: &'a mut Vec<CompilerError>,
        term_ids: &'a mut TermIdSource,
    ) -> Self {
        let placeholder_sym = symbols.alloc("_w_placeholder".to_string());
        Self {
            type_table,
            term_ids,
            symbols,
            scope: LookupMap::new(),
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
        top_level_symbols: &'a mut LookupMap<String, SymbolId>,
        name_resolution: &'a NameResolution,
        namespace: &str,
        fill_holes: bool,
        fill_hole_errors: &'a mut Vec<CompilerError>,
        term_ids: &'a mut TermIdSource,
    ) -> Self {
        let placeholder_sym = symbols.alloc("_w_placeholder".to_string());
        Self {
            type_table,
            term_ids,
            symbols,
            scope: LookupMap::new(),
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
    pub(super) fn define(&mut self, name: &str) -> SymbolId {
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

    /// Transform an AST program to TLC.
    /// Returns program parts without the symbol table - caller must combine with
    /// their owned symbol table using `ProgramParts::with_symbols`.
    pub fn transform_program(&mut self, program: &ast::Program) -> ProgramParts<run::Polymorphic> {
        // First pass: register all top-level function names so that
        // references within function bodies use the same SymbolId as
        // the Def. Only allocate when the name isn't already pre-
        // registered by the `NameRegistry` walk in `tlc::run` — that
        // walk seeds `top_level_symbols` from the catalog / prelude
        // / elaborated-module items, and a user `def map` that
        // shadows the SOAC `map` must REUSE the existing symbol so
        // module-body references (transformed earlier with the
        // shared `top_level_symbols` map) point at the same Def.
        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    let name_str = match &self.namespace {
                        Some(ns) => format!("{}.{}", ns, d.name),
                        None => d.name.clone(),
                    };
                    if !self.top_level_symbols.contains_key(&name_str) {
                        let sym = self.symbols.alloc(name_str.clone());
                        self.top_level_symbols.insert(name_str, sym);
                    }
                }
                ast::Declaration::Entry(e) => {
                    if !self.top_level_symbols.contains_key(&e.name) {
                        let sym = self.symbols.alloc(e.name.clone());
                        self.top_level_symbols.insert(e.name.clone(), sym);
                    }
                }
                ast::Declaration::Extern(e) => {
                    if !self.top_level_symbols.contains_key(&e.name) {
                        let sym = self.symbols.alloc(e.name.clone());
                        self.top_level_symbols.insert(e.name.clone(), sym);
                    }
                }
                _ => {}
            }
        }

        // Second pass: transform function bodies (now resolve_or_define can find top-level symbols)
        let mut defs = Vec::new();

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
                ast::Declaration::Extern(e) => {
                    // Use the pre-registered symbol
                    let name_sym =
                        *self.top_level_symbols.get(&e.name).expect("BUG: extern not pre-registered");
                    let body = self.mk_term(e.ty.clone(), e.span, TermKind::Extern(e.linkage_name.clone()));
                    let arity = count_function_arity(&e.ty);
                    defs.push(Def {
                        data: data::PolymorphicDefinition { scheme: None },
                        name: name_sym,
                        ty: e.ty.clone(),
                        body,
                        meta: DefMeta::Function,
                        arity,
                        param_diets: e.param_diets.clone(),
                        return_diet: e.return_diet.clone(),
                    });
                }
                ast::Declaration::Sig(_)
                | ast::Declaration::TypeBind(_)
                | ast::Declaration::Module(_)
                | ast::Declaration::ModuleTypeBind(_)
                | ast::Declaration::Open(_)
                | ast::Declaration::Import(_)
                | ast::Declaration::Resource(_) => {}
            }
        }

        ProgramParts { defs }
    }

    pub fn transform_decl(&mut self, decl: &ast::Decl) -> Option<Def<run::Polymorphic>> {
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
            data: data::PolymorphicDefinition { scheme: None },
            name: name_sym,
            ty: full_ty,
            body,
            meta: DefMeta::Function,
            arity: decl.params.len(),
            param_diets: decl.param_diets.clone(),
            return_diet: decl.return_diet.clone(),
        })
    }

    fn transform_entry(&mut self, entry: &interface::EntryDecl) -> Option<Def<run::Polymorphic>> {
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
            data: data::PolymorphicDefinition { scheme: None },
            name: name_sym,
            ty: full_ty,
            body,
            meta: DefMeta::EntryPoint(EntryPoint {
                declaration: Box::new(entry.clone()),
                data: (),
            }),
            arity: entry.params.len(),
            param_diets: entry.param_diets.clone(),
            return_diet: entry.return_diet.clone(),
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
                TermKind::Var(VarRef::Symbol(placeholder_sym)),
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
    // compute_pattern_bindings + compute_pattern_bindings_inner +
    // build_tuple_projection live in tlc/patterns/bindings.rs (extends
    // this type via an `impl` block there).

    /// Apply a list of bindings around a body term, creating nested let expressions.
    /// Bindings are applied in reverse order so the first binding is outermost.
    pub(super) fn apply_bindings_around(
        &mut self,
        bindings: Vec<PendingBinding>,
        body: Term,
        span: Span,
    ) -> Term {
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

    // simple_pattern_name, extract_tuple_types, resolve_field_index,
    // and extract_record_types live in tlc/patterns/bindings.rs.

    pub(super) fn transform_expr(&mut self, expr: &ast::Expression) -> Term {
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

            ast::ExprKind::Unit => self.mk_term(ty, span, TermKind::UnitLit),

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
                        let def = by_id(*id);
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
                        TermKind::Var(VarRef::Builtin {
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
                self.mk_term(ty, span, TermKind::Var(VarRef::Symbol(sym)))
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
                let aw_id = if matches!(ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    catalog().known().image_with
                } else {
                    catalog().known().array_with
                };
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
                let field_map: LookupMap<&str, &ast::Expression> =
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
                    let elem_ty = rec
                        .ty
                        .elem_type()
                        .cloned()
                        .expect("rec.ty.is_vec() above guarantees a vec elem type");
                    let n_components = field.chars().count();

                    // Single-letter swizzle is one projection — no
                    // duplication concern; project the rec term directly.
                    if n_components == 1 {
                        let idx = crate::types::swizzle_component_index(field.chars().next().unwrap())
                            .expect("is_swizzle_field already accepted this letter");
                        return self.mk_tuple_proj(rec, idx as usize, elem_ty, span);
                    }

                    // Multi-letter swizzle desugars to one
                    // `mk_tuple_proj` per component. If `rec` is a
                    // non-trivial producer (App, Soac, If, Loop, …),
                    // cloning it once per component leaves downstream
                    // passes with several independent copies of the
                    // same producer — egregious when the producer is a
                    // `reduce(...)`: the SoA / CSE layers don't share
                    // them, and the compiled output runs the reduce
                    // once per swizzle slot. Let-bind first so each
                    // projection reads the same evaluated value;
                    // mirrors what `transform_vec_with` does for `with`
                    // updates (`_w_vw_t_…`).
                    let needs_share = !matches!(
                        &rec.kind,
                        TermKind::Var(_)
                            | TermKind::IntLit(_)
                            | TermKind::FloatLit(_)
                            | TermKind::BoolLit(_)
                            | TermKind::UnitLit
                    );

                    let (base, wrap_let): (Term, Option<(SymbolId, Type<TypeName>, Term)>) = if needs_share
                    {
                        let t_id = self.term_ids.next_id();
                        let t_sym = self.define(&format!("_w_swz_t_{}", t_id));
                        let t_ty = rec.ty.clone();
                        let var = self.mk_term(t_ty.clone(), span, TermKind::Var(VarRef::Symbol(t_sym)));
                        (var, Some((t_sym, t_ty, rec)))
                    } else {
                        (rec, None)
                    };

                    let components: Vec<Term> = field
                        .chars()
                        .map(|c| {
                            let idx = crate::types::swizzle_component_index(c)
                                .expect("is_swizzle_field already accepted this letter");
                            self.mk_tuple_proj(base.clone(), idx as usize, elem_ty.clone(), span)
                        })
                        .collect();

                    let body = self.build_vec_lit_from_terms(&components, ty.clone(), span);
                    return match wrap_let {
                        Some((name, name_ty, rhs)) => self.mk_term(
                            ty,
                            span,
                            TermKind::Let {
                                name,
                                name_ty,
                                rhs: Box::new(rhs),
                                body: Box::new(body),
                            },
                        ),
                        None => body,
                    };
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
                // and dead slots get blank-filled.
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
                        slot_terms.push(self.build_blank(slot_ty, span));
                    }
                }
                self.mk_tuple(slot_terms, ty, span)
            }

            ast::ExprKind::Range(range) => {
                let start = self.transform_expr(&range.start);
                let end = self.transform_expr(&range.end);
                let step = range.step.as_ref().map(|s| self.transform_expr(s));
                let elem_ty = end.ty.clone();
                let minus = ast::BinaryOp { op: "-".to_string() };
                let plus = ast::BinaryOp { op: "+".to_string() };
                let div = ast::BinaryOp { op: "/".to_string() };

                // Element count per range kind:
                //   `a..b`   (Exclusive)   → b - a
                //   `a..<b`  (ExclusiveLt) → b - a
                //   `a..>b`  (ExclusiveGt) → a - b   (descending half-open)
                //   `a...b`  (Inclusive)   → b - a + 1
                let mut len = match range.kind {
                    ast::RangeKind::Exclusive | ast::RangeKind::ExclusiveLt => {
                        self.build_binop(minus.clone(), end.clone(), start.clone(), elem_ty.clone(), span)
                    }
                    ast::RangeKind::ExclusiveGt => {
                        self.build_binop(minus.clone(), start.clone(), end.clone(), elem_ty.clone(), span)
                    }
                    ast::RangeKind::Inclusive => {
                        let one = self.mk_term(elem_ty.clone(), span, TermKind::IntLit("1".to_string()));
                        let diff =
                            self.build_binop(minus, end.clone(), start.clone(), elem_ty.clone(), span);
                        self.build_binop(plus, diff, one, elem_ty.clone(), span)
                    }
                };
                if let Some(ref step_term) = step {
                    len = self.build_binop(div, len, step_term.clone(), elem_ty.clone(), span);
                }

                let range_ae = ArrayExpr::Range {
                    start: Box::new(start),
                    len: Box::new(len),
                    step: step.map(Box::new),
                };
                self.mk_term(ty, span, TermKind::ArrayExpr(range_ae))
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
                let known = catalog().known();
                let end =
                    slice.end.as_ref().map(|e| self.transform_expr(e)).unwrap_or_else(|| {
                        self.build_call_by_id(known.length, &[arr.clone()], i32_ty, span)
                    });

                self.build_call_by_id(known.slice, &[arr, start, end], ty, span)
            }

            ast::ExprKind::TypeAscription(inner, _) => self.transform_expr(inner),

            ast::ExprKind::TypeCoercion(inner, _) => {
                let term = self.transform_expr(inner);
                let target_ty = ty.clone();
                self.mk_term(
                    ty,
                    span,
                    TermKind::Coerce {
                        inner: Box::new(term),
                        target_ty,
                    },
                )
            }

            ast::ExprKind::TypeHole => {
                if !self.fill_holes {
                    unreachable!(
                        "TypeHole should be rejected at type-check when \
                         --fill-holes is not set; see \
                         TypeChecked::reject_type_holes"
                    );
                }
                // `ty` is the table lookup at the top of this function (which
                // already panics if missing) — no need to look it up again.
                defaults::default_term_for_type(self, &ty, span)
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
        // Lower as a SOAC iff the resolver tagged the callee as one
        // (so a user `def map` shadowing the builtin is a normal call).
        if let Some(kind) = self.resolve_soac(func) {
            return self.transform_soac_call(kind, args, ty, span);
        }

        // Constructor-style vec conversion (`vec2i32(v)`, …).
        // The type checker recorded a `VecConstructor` ResolvedValueRef
        // for the callee. Desugar to a `VecLit` of componentwise scalar
        // conversion calls — `vec2i32(v)` ⟶ `@[i32(v.x), i32(v.y)]`
        // with each `i32(…)` resolved to its concrete per-type catalog
        // entry by the source-component type.
        if let ast::ExprKind::Identifier(_, _) = &func.kind {
            if let Some(crate::name_resolution::ResolvedValueRef::VecConstructor {
                arity,
                target_elem,
                ..
            }) = self.name_resolution.get(func.h.id).cloned()
            {
                debug_assert_eq!(
                    args.len(),
                    1,
                    "BUG: vec constructor expected 1 arg, got {}",
                    args.len()
                );
                return self.transform_vec_constructor(&args[0], &target_elem, arity, ty, span);
            }
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

    /// Synthesise `vec<N><target_elem>(v)` as a `VecLit` of N
    /// componentwise scalar conversion calls. Each component is
    /// `<target_elem>.<source_elem>(v.<i>)`, where `<source_elem>` is
    /// read from `v`'s converted Term type.
    fn transform_vec_constructor(
        &mut self,
        arg: &ast::Expression,
        target_elem: &str,
        arity: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let arg_term = self.transform_expr(arg);
        // The arg is a vec whose elem type tells us the source-elem
        // module name. `i32`-to-`f32` keys map to the catalog entry
        // `f32.i32` (target.source); we form that surface name and
        // dispatch via the builtin catalog.
        let source_elem_ty = arg_term
            .ty
            .elem_type()
            .expect("vec constructor arg must be a vec — type checker enforces this")
            .clone();
        let source_elem_name =
            crate::types::checker::type_name_to_module(&source_elem_ty).unwrap_or_else(|| {
                panic!(
                    "BUG: vec constructor arg's component type {:?} has no surface module name",
                    source_elem_ty
                )
            });
        let conv_surface = format!("{}.{}", target_elem, source_elem_name);
        let conv_builtin =
            crate::builtins::catalog().lookup_by_surface_name(&conv_surface).unwrap_or_else(|| {
                panic!(
                    "BUG: vec constructor desugar can't find catalog entry `{}` — \
                     target_elem `{}` source_elem `{}` arity {}",
                    conv_surface, target_elem, source_elem_name, arity
                )
            });
        let conv_id = conv_builtin.id;
        let target_elem_ty =
            result_ty.elem_type().expect("vec constructor result type is always a vec").clone();

        // Bind the arg once to a synthetic let so each component
        // projection reuses the same evaluation. `SymbolId(0)` is the
        // shared "sequence" sentinel — fine for unit-typed bindings
        // but here we want the value preserved, so allocate a fresh
        // name.
        let arg_sym = self.symbols.alloc("_w_vec_conv_arg".to_string());
        let arg_ref = self.mk_term(arg_term.ty.clone(), span, TermKind::Var(VarRef::Symbol(arg_sym)));

        // Build N per-component conversion calls.
        let mut components: Vec<Term> = Vec::with_capacity(arity);
        for i in 0..arity {
            let proj = self.mk_tuple_proj(arg_ref.clone(), i, source_elem_ty.clone(), span);
            let conv_func = self.mk_term(
                Type::Constructed(
                    TypeName::Arrow,
                    vec![source_elem_ty.clone(), target_elem_ty.clone()],
                ),
                span,
                TermKind::Var(VarRef::Builtin {
                    id: conv_id,
                    overload_idx: 0,
                }),
            );
            let conv_call = self.mk_term(
                target_elem_ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(conv_func),
                    args: vec![proj],
                },
            );
            components.push(conv_call);
        }

        let vec_lit = self.build_vec_lit_from_terms(&components, result_ty.clone(), span);

        // Wrap in `let _w_vec_conv_arg = <arg_term> in <vec_lit>`.
        self.mk_term(
            result_ty.clone(),
            span,
            TermKind::Let {
                name: arg_sym,
                name_ty: arg_term.ty.clone(),
                rhs: Box::new(arg_term),
                body: Box::new(vec_lit),
            },
        )
    }

    /// The SOAC this call's callee denotes, per the frontend resolver —
    /// `None` for everything else, including a user `def` (top-level or
    /// local) that shadows a SOAC name. Structural: no surface-name match
    /// or scope re-derivation here; the resolver already decided.
    fn resolve_soac(&self, func: &ast::Expression) -> Option<SoacKind> {
        match self.name_resolution.get(func.h.id) {
            Some(ResolvedValueRef::Soac(kind)) => Some(*kind),
            _ => None,
        }
    }

    /// Dispatch SOAC call by structural kind.
    fn transform_soac_call(
        &mut self,
        kind: SoacKind,
        args: &[ast::Expression],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        match kind {
            SoacKind::Map => self.transform_soac_map(args, ty, span),
            SoacKind::Reduce => self.transform_soac_reduce(args, ty, span),
            SoacKind::Scan => self.transform_soac_scan(args, ty, span),
            SoacKind::Filter => self.transform_soac_filter(args, ty, span),
            SoacKind::Zip => self.transform_soac_zip(args, ty, span),
            SoacKind::ReduceByIndex => self.transform_soac_reduce_by_index(args, ty, span),
            SoacKind::Scatter => self.transform_soac_scatter(args, ty, span),
        }
    }

    /// Convert a transformed array-argument term into an ANF SOAC input. A bare
    /// variable passes through as `Var`; any other term (a producer SOAC, a
    /// call, …) is let-bound to a fresh `_anf` name, with the binding pushed to
    /// `binds` for the caller to wrap around the SOAC via [`Self::wrap_binds`].
    fn soac_input(
        &mut self,
        arr_term: Term,
        binds: &mut Vec<(SymbolId, Type<TypeName>, Term)>,
    ) -> ArrayExpr {
        // Lift any binding lets above the SOAC (e.g. `iota(N)` desugars to
        // `let arg = N in Range{…}`), keeping the input itself atomic.
        let (mut peeled, core) = peel_lets(arr_term);
        binds.append(&mut peeled);
        match core.kind {
            TermKind::Var(vr) => ArrayExpr::Var(vr, core.ty),
            // An array expression (Range / Literal / Zip) is
            // itself an atomic SOAC input; consume it directly rather than
            // let-binding a name to it.
            TermKind::ArrayExpr(ae) => ae,
            _ => {
                let ty = core.ty.clone();
                let sym = self.symbols.alloc("_anf".to_string());
                binds.push((sym, ty.clone(), core));
                ArrayExpr::Var(VarRef::Symbol(sym), ty)
            }
        }
    }

    /// Wrap `binds` as nested `let`s (outermost first) around `body`.
    fn wrap_binds(&mut self, binds: Vec<(SymbolId, Type<TypeName>, Term)>, body: Term, span: Span) -> Term {
        let mut result = body;
        for (name, name_ty, rhs) in binds.into_iter().rev() {
            let body_ty = result.ty.clone();
            result = self.mk_term(
                body_ty,
                span,
                TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(rhs),
                    body: Box::new(result),
                },
            );
        }
        result
    }

    /// Transform `map(f, arr)` → `Soac(Map { lam, inputs })`.
    fn transform_soac_map(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 2, "map requires at least 2 arguments");
        let func_term = self.transform_expr(&args[0]);
        let arr_term = self.transform_expr(&args[1]);

        let lam = self.term_to_lambda(func_term);

        // Absorb zip: if arr_term is ArrayExpr(Zip(...)), flatten into inputs.
        // The lambda still takes a single tuple param — the soa::normalize pass
        // will rewrite it to take separate params. A zip whose children needed
        // let-binding arrives wrapped in those lets (from `transform_soac_zip`),
        // so peel them off and re-wrap around the whole map.
        let (mut binds, core) = peel_lets(arr_term);
        let inputs = match core.kind {
            TermKind::ArrayExpr(ArrayExpr::Zip(exprs)) => exprs,
            _ => vec![self.soac_input(core, &mut binds)],
        };

        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Map {
                lam,
                inputs,
                destination: SoacOwnership::Fresh,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `reduce(op, ne, arr)` → `Soac(Reduce { op, ne, input })`.
    fn transform_soac_reduce(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 3, "reduce requires 3 arguments");
        let op_term = self.transform_expr(&args[0]);
        let ne_term = self.transform_expr(&args[1]);
        let arr_term = self.transform_expr(&args[2]);

        let op = self.term_to_lambda(op_term);

        let mut binds = Vec::new();
        let input = self.soac_input(arr_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Reduce {
                op,
                ne: Box::new(ne_term),
                input,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `scan(op, ne, arr)` → `Soac(Scan { op, ne, input })`.
    fn transform_soac_scan(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 3, "scan requires 3 arguments");
        let op_term = self.transform_expr(&args[0]);
        let ne_term = self.transform_expr(&args[1]);
        let arr_term = self.transform_expr(&args[2]);

        let op = self.term_to_lambda(op_term);

        let mut binds = Vec::new();
        let input = self.soac_input(arr_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Scan {
                op,
                ne: Box::new(ne_term),
                input,
                // Initial construction; apply_ownership may flip later.
                destination: SoacOwnership::Fresh,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `filter(pred, arr)` → `Soac(Filter { pred, input })`.
    fn transform_soac_filter(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 2, "filter requires 2 arguments");
        let pred_term = self.transform_expr(&args[0]);
        let arr_term = self.transform_expr(&args[1]);

        let pred = self.term_to_lambda(pred_term);

        let mut binds = Vec::new();
        let input = self.soac_input(arr_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Filter {
                pred,
                input,
                // Initial construction; apply_ownership may flip later.
                destination: SoacOwnership::Fresh,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `zip(a, b, ...)` → `ArrayExpr(Zip(...))`. Each child becomes an
    /// ANF atom; any producer child is let-bound, the bindings wrapping the zip
    /// term (a consuming `map` peels them back off — see `transform_soac_map`).
    fn transform_soac_zip(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        let mut binds = Vec::new();
        let mut exprs = Vec::with_capacity(args.len());
        for a in args {
            let t = self.transform_expr(a);
            exprs.push(self.soac_input(t, &mut binds));
        }
        let zip = self.mk_term(ty, span, TermKind::ArrayExpr(ArrayExpr::Zip(exprs)));
        self.wrap_binds(binds, zip, span)
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
        let dest = Place {
            id: match &dest_term.kind {
                TermKind::Var(VarRef::Symbol(sym)) => sym.clone(),
                _ => {
                    // Bind dest to a fresh name
                    let fresh = self.define("_w_rbi_dest");
                    fresh
                }
            },
            elem_ty: dest_elem_ty,
        };

        let mut binds = Vec::new();
        let indices = self.soac_input(indices_term, &mut binds);
        let values = self.soac_input(values_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::ReduceByIndex {
                dest,
                op,
                ne: Box::new(ne_term),
                indices,
                values,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// `scatter(dest, indices, values)` → `SoacOp::Scatter`. Writes
    /// `values[i]` into `dest[indices[i]]` for each `i`; out-of-bounds indices
    /// are ignored (Futhark semantics). The `dest` must be a Var (a `#[storage]`
    /// buffer param in the rasterizer use case) — its `Place`
    /// carries the symbol the EGIR conversion resolves to the dest's view.
    fn transform_soac_scatter(&mut self, args: &[ast::Expression], ty: Type<TypeName>, span: Span) -> Term {
        assert!(args.len() >= 3, "scatter requires 3 arguments");
        let dest_term = self.transform_expr(&args[0]);
        let indices_term = self.transform_expr(&args[1]);
        let values_term = self.transform_expr(&args[2]);

        let dest_elem_ty = self.get_array_element_type(&dest_term.ty);
        let idx_elem_ty = self.get_array_element_type(&indices_term.ty);
        let val_elem_ty = self.get_array_element_type(&values_term.ty);
        let dest = Place {
            id: match &dest_term.kind {
                TermKind::Var(VarRef::Symbol(sym)) => sym.clone(),
                _ => self.define("_w_scatter_dest"),
            },
            elem_ty: dest_elem_ty,
        };

        // Identity envelope `λ(i, v) → (i, v)`. Fusion composes producer
        // lambdas into this and splices their inputs in place of `is`/`vs`.
        let i_sym = self.define("_w_scatter_i");
        let v_sym = self.define("_w_scatter_v");
        let i_var = self.mk_term(idx_elem_ty.clone(), span, TermKind::Var(VarRef::Symbol(i_sym)));
        let v_var = self.mk_term(val_elem_ty.clone(), span, TermKind::Var(VarRef::Symbol(v_sym)));
        let tuple_ty =
            Type::Constructed(TypeName::Tuple(2), vec![idx_elem_ty.clone(), val_elem_ty.clone()]);
        let body = self.mk_tuple(vec![i_var, v_var], tuple_ty.clone(), span);
        let lam = SoacBody {
            lam: Lambda {
                params: vec![(i_sym, idx_elem_ty), (v_sym, val_elem_ty)],
                body: Box::new(body),
                ret_ty: tuple_ty,
            },
            data: (),
        };

        let mut binds = Vec::new();
        let indices = self.soac_input(indices_term, &mut binds);
        let values = self.soac_input(values_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Scatter {
                dest,
                lam,
                inputs: vec![indices, values],
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Convert a term to a SoacBody. If it's already a Lambda, wrap it.
    /// Otherwise, eta-expand all parameters: `f : A -> B -> C` → `|a, b| f(a)(b)`.
    /// Captures are always empty here — this runs pre-defunctionalization.
    fn term_to_lambda(&mut self, term: Term) -> SoacBody {
        match term.kind {
            TermKind::Lambda(lam) => SoacBody { lam, data: () },
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
                // it survives duplicate names, but WGSL inherits the display
                // name verbatim and rejects a function whose parameter list
                // repeats a name.
                let params: Vec<(SymbolId, Type<TypeName>)> = param_tys
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| (self.define(&format!("_soac_arg_{}", i)), ty.clone()))
                    .collect();

                // Build flat App(f, [a, b, ...])
                let span = term.span;
                let arg_terms: Vec<Term> = params
                    .iter()
                    .map(|(sym, ty)| self.mk_term(ty.clone(), span, TermKind::Var(VarRef::Symbol(*sym))))
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
                    data: (),
                }
            }
        }
    }

    fn transform_loop(&mut self, loop_expr: &ast::LoopExpr, ty: Type<TypeName>, span: Span) -> Term {
        // Get the init expression and accumulator type
        let init_term = loop_expr.init.as_ref().map(|e| self.transform_expr(e)).unwrap_or_else(|| {
            // No accumulator - use unit
            self.mk_term(Type::Constructed(TypeName::Unit, vec![]), span, TermKind::UnitLit)
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
        let loop_var_name = format!("_w_loop_{}", self.term_ids.next_id());
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
        let mut current = self.mk_term(current_ty.clone(), span, TermKind::Var(VarRef::Symbol(var_sym)));

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
        let t_id = self.term_ids.next_id();
        let t_sym = self.define(&format!("_w_vw_t_{}", t_id));
        let t_var = self.mk_term(target_ty.clone(), span, TermKind::Var(VarRef::Symbol(t_sym)));

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
        let r_id = self.term_ids.next_id();
        let r_sym = self.define(&format!("_w_vw_r_{}", r_id));
        let r_var = self.mk_term(rhs_term.ty.clone(), span, TermKind::Var(VarRef::Symbol(r_sym)));

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

        let r_id = self.term_ids.next_id();
        let r_sym = self.define(&format!("_w_rw_r_{}", r_id));
        let r_var = self.mk_term(record_ty.clone(), span, TermKind::Var(VarRef::Symbol(r_sym)));

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
            let inner_id = self.term_ids.next_id();
            let inner_sym = self.define(&format!("_w_rw_inner_{}", inner_id));
            let inner_var = self.mk_term(inner_ty.clone(), span, TermKind::Var(VarRef::Symbol(inner_sym)));
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
        debug_assert!(
            !match_expr.cases.is_empty(),
            "checker rejects empty match upstream"
        );
        self.compile_match(match_expr, ty, span)
    }

    /// Produce a typed-blank Term for `ty`. Fills dead
    /// constructor-payload slots in a flattened sum-type tuple.
    fn build_blank(&mut self, ty: &Type<TypeName>, span: Span) -> Term {
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
            Type::Constructed(TypeName::Unit, _) => self.mk_term(ty.clone(), span, TermKind::UnitLit),
            Type::Constructed(TypeName::Tuple(_), elems)
            | Type::Constructed(TypeName::Record(_), elems) => {
                let blank_terms: Vec<Term> = elems.iter().map(|t| self.build_blank(t, span)).collect();
                self.mk_tuple(blank_terms, ty.clone(), span)
            }
            Type::Constructed(TypeName::Array, args) => {
                // Rank-1 invariant: sum payloads can only hold rank-1
                // arrays with a constant size.
                debug_assert_eq!(
                    args.len(),
                    4,
                    "Array sum payload must have [elem, variant, size, region] args"
                );
                let elem_ty = &args[0];
                let n = match &args[2] {
                    Type::Constructed(TypeName::Size(n), _) => *n,
                    other => panic!(
                        "BUG: array-typed sum payload must have constant size (got {:?}); \
                         the type checker should reject symbolic-size sum payloads upstream",
                        other
                    ),
                };
                let elem_blank = self.build_blank(elem_ty, span);
                let elems: Vec<Term> = std::iter::repeat(elem_blank).take(n).collect();
                self.mk_term(ty.clone(), span, TermKind::ArrayExpr(ArrayExpr::Literal(elems)))
            }
            Type::Constructed(TypeName::Vec, args) => {
                debug_assert_eq!(args.len(), 2, "Vec type must have [elem, size] args");
                let elem_ty = &args[0];
                let n = match &args[1] {
                    Type::Constructed(TypeName::Size(n), _) => *n,
                    other => panic!("BUG: Vec sum payload must have constant size (got {:?})", other),
                };
                let elem_blank = self.build_blank(elem_ty, span);
                let elems: Vec<Term> = std::iter::repeat(elem_blank).take(n).collect();
                self.mk_term(ty.clone(), span, TermKind::VecLit(elems))
            }
            Type::Constructed(TypeName::Arrow, _) => {
                panic!(
                    "BUG: function-typed sum payloads are not supported, but reached \
                     build_blank. The type checker should reject this at the Constructor \
                     or Match site."
                );
            }
            Type::Variable(_)
            | Type::Constructed(TypeName::Size(_), _)
            | Type::Constructed(TypeName::SizeVar(_), _)
            | Type::Constructed(TypeName::SizePlaceholder, _)
            | Type::Constructed(TypeName::AddressPlaceholder, _)
            | Type::Constructed(TypeName::ArrayVariantView, _)
            | Type::Constructed(TypeName::ArrayVariantComposite, _)
            | Type::Constructed(TypeName::ArrayVariantVirtual, _)
            | Type::Constructed(TypeName::Skolem(_), _) => {
                panic!(
                    "BUG: build_blank reached a non-value-level type {:?}; \
                     these shouldn't appear in sum payload slot positions",
                    ty
                );
            }
            _ => panic!("blank for sum-payload type {:?} is not yet implemented", ty),
        }
    }

    // Helper: build binary op application
    pub(super) fn build_binop(
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

    /// Build a flat call against a catalog `BuiltinId`.
    fn build_call_by_id(
        &mut self,
        id: BuiltinId,
        args: &[Term],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_var = VarRef::Builtin { id, overload_idx: 0 };
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

    /// Construct a `TermKind::Tuple` directly.
    fn mk_tuple(&mut self, parts: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::Tuple(parts))
    }

    /// Construct a `TermKind::TupleProj` directly.
    pub(super) fn mk_tuple_proj(
        &mut self,
        tuple: Term,
        idx: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
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
    pub(super) fn lower_type(ty: Type<TypeName>) -> Type<TypeName> {
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
    /// variant are blank-filled at construction.
    pub(super) fn sum_layout(variants: &[(String, Vec<Type<TypeName>>)]) -> SumLayout {
        let tag_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut slot_types = vec![tag_ty];
        let mut constructor_info = LookupMap::new();
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

    pub(super) fn mk_term(&mut self, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
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

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;

#[cfg(test)]
#[path = "term_rewriter_tests.rs"]
mod term_rewriter_tests;
