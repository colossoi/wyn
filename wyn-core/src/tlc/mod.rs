//! Typed Lambda Calculus (TLC) representation.
//!
//! A minimal typed lambda calculus IR for source-level specialization.
//! Lambdas remain as values (not yet defunctionalized).

pub mod anf;
pub mod data;
mod dce;
pub mod defaults;
pub mod defunctionalize;
mod from_ast;
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

use crate::ast::{self, Span, TypeName};
use crate::builtins::BuiltinId;
use crate::error::CompilerError;
use crate::types::SoacOwnership;
use crate::{interface, LookupMap, LookupSet, SymbolId, SymbolTable, TypeTable};
use polytype::Type;
use std::num::NonZeroU32;

pub(crate) use from_ast::{PendingBinding, Transformer};

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

    /// Apply the pre-children hook while owning the node.
    ///
    /// Structural rewrites that need to move fields out of a `TermKind` can
    /// override this method. The default delegates to the borrowed hook.
    fn rewrite_owned_node_before_children(
        &mut self,
        mut term: Term<C, S>,
    ) -> (Term<C, S>, RewriteDecision) {
        let decision = self.rewrite_node_before_children(&mut term);
        (term, decision)
    }

    /// Apply the post-children hook while owning the node.
    ///
    /// Structural rewrites that need to move fields out of a `TermKind` can
    /// override this method. The default delegates to the borrowed hook.
    fn rewrite_owned_node(&mut self, mut term: Term<C, S>) -> (Term<C, S>, RewriteDecision) {
        let decision = self.rewrite_node(&mut term);
        (term, decision)
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

    /// Rewrite an owned term in place so existing child allocations survive.
    fn rewrite(&mut self, mut term: Term<C, S>) -> Term<C, S>
    where
        Self: Sized,
    {
        self.rewrite_tracked(&mut term);
        term
    }

    /// Consume and rebuild a term using the owned-node hooks.
    fn rewrite_owned(&mut self, term: Term<C, S>) -> Term<C, S>
    where
        Self: Sized,
    {
        self.rewrite_owned_tracked(term).0
    }

    /// Consume and rebuild a term while reporting whether its subtree changed.
    fn rewrite_owned_tracked(&mut self, term: Term<C, S>) -> (Term<C, S>, bool)
    where
        Self: Sized,
    {
        let (term, node_changed_before) = self.rewrite_owned_node_before_children(term);
        let old_id = term.id;
        let mut descendant_changed = false;
        let term = term.map_children(old_id, &mut |child| {
            let (child, changed) = self.rewrite_owned_tracked(child);
            descendant_changed |= changed;
            child
        });
        let (mut term, node_changed_after) = self.rewrite_owned_node(term);
        let changed = node_changed_before == RewriteDecision::Changed
            || descendant_changed
            || node_changed_after == RewriteDecision::Changed;
        if changed {
            term.id = self.next_term_id();
        }
        (term, changed)
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
    /// types. This is the canonical array-type calculation shared by TLC
    /// representation passes and EGIR lowering.
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

    /// Rewrite this owned term while retaining its existing child allocations.
    pub fn rewrite<R>(self, rewriter: &mut R) -> Self
    where
        R: TermRewriter<C, S>,
    {
        rewriter.rewrite(self)
    }

    /// Consume and rebuild this term with a rewriter's owned-node hooks.
    pub fn rewrite_owned<R>(self, rewriter: &mut R) -> Self
    where
        R: TermRewriter<C, S>,
    {
        rewriter.rewrite_owned(self)
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

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;

#[cfg(test)]
#[path = "term_rewriter_tests.rs"]
mod term_rewriter_tests;
