pub mod ast;
pub mod ast_renumber;
pub mod builtins;
pub mod diags;
pub mod error;
pub mod interface;
pub mod lexer;
pub mod module_manager;
pub mod name_resolution;
pub mod parser;
pub mod pattern;
pub mod scope;
pub mod ssa;
pub mod types;
pub mod visitor;

// Re-export type_checker from its new location for backwards compatibility
pub use types::checker as type_checker;

pub mod ast_const_fold;
pub mod lowering_common;
pub mod name_registry;
pub mod tlc;

pub mod egir;
pub mod glsl;
/// Re-export of the pipeline descriptor format. Lives in its own
/// crate so host runtimes (e.g. `extra/viz`) can deserialize the
/// JSON without pulling in the whole compiler.
pub use wyn_pipeline_descriptor as pipeline_descriptor;
pub mod resolve_imports;
pub mod resolve_opens;
pub mod resolve_placeholders;
pub mod spirv;
pub mod structured;
pub mod wgsl;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod desugar_tests;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use egir::from_tlc::ConvertError;
use egir::program::EgirInner;

use indexmap::IndexMap;

use ast::{NodeCounter, NodeId};
use error::Result;
use polytype::TypeScheme;

// =============================================================================
// Generic ID allocation
// =============================================================================

/// Generic counter for generating unique IDs.
///
/// The ID type must implement `From<u32>` to convert the raw counter value.
#[derive(Debug, Clone)]
pub struct IdSource<Id> {
    next_id: u32,
    _phantom: PhantomData<Id>,
}

impl<Id: From<u32>> IdSource<Id> {
    pub fn new() -> Self {
        IdSource {
            next_id: 0,
            _phantom: PhantomData,
        }
    }

    pub fn next_id(&mut self) -> Id {
        let id = Id::from(self.next_id);
        self.next_id += 1;
        id
    }
}

impl<Id: From<u32>> Default for IdSource<Id> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Symbol Table for TLC
// =============================================================================

/// Unique identifier for a symbol (variable, function, parameter).
/// After AST → TLC conversion, all variable references use SymbolIds
/// instead of strings, eliminating name resolution from later passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

impl From<u32> for SymbolId {
    fn from(v: u32) -> Self {
        SymbolId(v)
    }
}

impl std::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sym{}", self.0)
    }
}

/// Symbol table: maps SymbolId to original name (for errors/debugging).
pub type SymbolTable = IdArena<SymbolId, String>;

/// Arena that allocates IDs and stores associated items.
///
/// Combines ID generation with storage, ensuring each item gets a unique ID.
/// Uses IndexMap for deterministic iteration order (insertion order).
#[derive(Debug, Clone)]
pub struct IdArena<Id, T> {
    source: IdSource<Id>,
    items: IndexMap<Id, T>,
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IdArena<Id, T> {
    pub fn new() -> Self {
        IdArena {
            source: IdSource::new(),
            items: IndexMap::new(),
        }
    }

    /// Allocate a new ID and store the item.
    pub fn alloc(&mut self, item: T) -> Id {
        let id = self.source.next_id();
        self.items.insert(id, item);
        id
    }

    /// Allocate a new ID without storing anything yet.
    /// Use `insert` later to store the item.
    pub fn alloc_id(&mut self) -> Id {
        self.source.next_id()
    }

    /// Insert an item with a pre-allocated ID.
    /// Panics if the ID is already in use.
    pub fn insert(&mut self, id: Id, item: T) {
        let old = self.items.insert(id, item);
        assert!(old.is_none(), "IdArena::insert called with duplicate ID");
    }

    /// Get an item by ID.
    pub fn get(&self, id: Id) -> Option<&T> {
        self.items.get(&id)
    }

    /// Get a mutable reference to an item by ID.
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.items.get_mut(&id)
    }

    /// Iterate over all (id, item) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Id, &T)> {
        self.items.iter()
    }

    /// Iterate over all items (without IDs).
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.items.values()
    }

    /// Number of items in the arena.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> Default for IdArena<Id, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for IdArena<Id, T> {
    type Item = (Id, T);
    type IntoIter = indexmap::map::IntoIter<Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a IdArena<Id, T> {
    type Item = (&'a Id, &'a T);
    type IntoIter = indexmap::map::Iter<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a mut IdArena<Id, T> {
    type Item = (&'a Id, &'a mut T);
    type IntoIter = indexmap::map::IterMut<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

// Re-export key types for the public API
pub use ast::TypeName;
pub use polytype::Context as PolytypeContext;
pub type TypeTable = HashMap<NodeId, TypeScheme<TypeName>>;

// =============================================================================
// Typestate Compiler Pipeline
// =============================================================================
//
// The compiler uses a typestate pattern where each struct represents a stage.
// Methods consume `self` and return the next stage, enforcing valid ordering.
//
//   let (mut node_counter, mut module_manager) = init_compiler();
//   let parsed = Compiler::parse(source, &mut node_counter)?;
//
// FrontEnd (AST) stages:
//     parsed.resolve(&module_manager)              -> Resolved
//       .fold_ast_constants()                      -> AstConstFoldedEarly
//       .type_check(&mut module_manager)           -> TypeChecked
//
// TLC stages (AST → SSA):
//       .to_tlc(&module_manager, fill_holes)       -> TlcTransformed
//       .partial_eval()                            -> TlcPartialEvaled
//       .normalize_soacs()                         -> TlcSoaNormalized
//       .fuse_maps()                               -> TlcFused
//       .apply_ownership()                         -> TlcOwnershipApplied
//       .defunctionalize()                         -> TlcDefunctionalized
//       .monomorphize()                            -> TlcMonomorphized
//       .buffer_specialize()                       -> TlcBufferSpecialized
//       .fold_generated_lambdas()                  -> TlcGeneratedLambdasFolded
//       .inline_small()                            -> TlcSmallInlined
//       .parallelize_soacs(disable)                -> TlcParallelized
//       .filter_reachable()                        -> TlcReachable
//       .to_egraph()                               -> EgirRaw
//
// EGIR stages:
//       .expand_soacs(unroll)                      -> EgirSoacExpanded
//       [.materialize()]                           -> EgirMaterialized
//       .optimize_skeleton()                       -> EgirSkelOptimized
//       .elaborate()                               -> SsaConverted
//
// Backend:
//       .lower() | .lower_glsl() | .lower_wgsl()
//
// Tests should prefer the `compile_thru_*` helpers below, which subsume
// the chain up to a milestone and centralize updates as new passes land.

/// Build a fresh `(NodeCounter, ModuleManager)` pair. The node counter is
/// shared between user code parsing and prelude loading so all NodeIds
/// stay unique. The module manager comes pre-loaded with the parsed
/// prelude.
pub fn init_compiler() -> (NodeCounter, module_manager::ModuleManager) {
    let mut node_counter = NodeCounter::new();
    let module_manager = match module_manager::ModuleManager::create_prelude(&mut node_counter) {
        Ok(prelude) => module_manager::ModuleManager::from_prelude(prelude),
        Err(e) => {
            eprintln!("ERROR creating prelude: {:?}", e);
            module_manager::ModuleManager::new_empty()
        }
    };
    (node_counter, module_manager)
}

/// Build a `(NodeCounter, ModuleManager)` pair from an already-elaborated
/// prelude. Faster than `init_compiler()` when callers can amortize the
/// prelude across multiple compiles.
pub fn init_compiler_from_prelude(
    prelude: module_manager::PreElaboratedPrelude,
    node_counter: NodeCounter,
) -> (NodeCounter, module_manager::ModuleManager) {
    let module_manager = module_manager::ModuleManager::from_prelude(prelude);
    (node_counter, module_manager)
}

// =============================================================================
// Compiler entry point
// =============================================================================

/// Entry point for the compiler. Use `Compiler::parse()` to start the pipeline.
pub struct Compiler;

impl Compiler {
    /// Parse source code into an AST using the provided node counter.
    /// This ensures NodeIds don't collide with prelude modules.
    pub fn parse(source: &str, node_counter: &mut NodeCounter) -> Result<Parsed> {
        let tokens = lexer::tokenize(source).map_err(|e| err_parse!("{}", e))?;
        let mut parser = parser::Parser::new(tokens, node_counter);
        let ast = parser.parse()?;
        Ok(Parsed(FrontInner { ast }))
    }
}

// =============================================================================
// FrontEnd stages (AST-based)
// =============================================================================

/// Shared payload for the front-end AST states (`Parsed`, `Desugared`,
/// `Resolved`, `AstConstFoldedEarly`). Each state is a newtype wrapping this
/// inner; transitions within the group just unwrap / re-wrap.
struct FrontInner {
    ast: ast::Program,
}

/// Source has been parsed into an AST
pub struct Parsed(FrontInner);

impl Parsed {
    /// Resolve `import "path"` declarations against the filesystem,
    /// loading each referenced file relative to `base_dir`, parsing
    /// it, recursively resolving its own imports, and prepending its
    /// declarations into this program. The `Import` nodes themselves
    /// are removed.
    ///
    /// Cycle / re-import safety: a file is loaded at most once per
    /// compilation (keyed by canonical path). Diamond imports work
    /// fine; cycles are silently broken (only the first encounter
    /// loads decls).
    ///
    /// Path resolution: `import "foo"` looks for `<base_dir>/foo.wyn`.
    /// Imports inside `foo.wyn` resolve relative to `foo.wyn`'s
    /// directory.
    pub fn resolve_imports(
        mut self,
        base_dir: &std::path::Path,
        node_counter: &mut NodeCounter,
    ) -> Result<Self> {
        self.0.ast.declarations = resolve_imports::run(self.0.ast.declarations, base_dir, node_counter)?;
        Ok(self)
    }

    /// Elaborate inline module declarations from the parsed program.
    /// This registers modules with the module_manager so they're available during resolution,
    /// then removes the Module declarations from the AST (they've been copied to module_manager).
    /// Should be called before desugar() if the program contains module definitions.
    pub fn elaborate_modules(
        mut self,
        module_manager: &mut module_manager::ModuleManager,
        node_counter: &mut ast::NodeCounter,
    ) -> Result<Self> {
        module_manager.elaborate_modules(&self.0.ast, node_counter)?;
        // Remove Module and ModuleTypeBind declarations - they've been elaborated
        self.0.ast.declarations.retain(|decl| {
            !matches!(
                decl,
                ast::Declaration::Module(_) | ast::Declaration::ModuleTypeBind(_)
            )
        });
        Ok(self)
    }

    /// Resolve names: rewrite FieldAccess -> QualifiedName and load modules
    pub fn resolve(mut self, module_manager: &module_manager::ModuleManager) -> Result<Resolved> {
        name_resolution::run(&mut self.0.ast, module_manager)?;
        Ok(Resolved(self.0))
    }
}

/// Names have been resolved
pub struct Resolved(FrontInner);

impl Resolved {
    /// Fold AST-level integer constants (required before type checking)
    pub fn fold_ast_constants(mut self) -> AstConstFoldedEarly {
        ast_const_fold::run(&mut self.0.ast);
        AstConstFoldedEarly(self.0)
    }
}

/// AST integer constants have been folded (before type checking)
pub struct AstConstFoldedEarly(FrontInner);

impl AstConstFoldedEarly {
    /// Type check the program
    pub fn type_check(mut self, module_manager: &mut module_manager::ModuleManager) -> Result<TypeChecked> {
        let out = types::run::run(&mut self.0.ast, module_manager)?;
        Ok(TypeChecked {
            ast: self.0.ast,
            type_table: out.type_table,
            warnings: out.warnings,
            checker_builtins: out.builtin_names,
            schemes: out.schemes,
            name_resolution: out.name_resolution,
        })
    }
}

/// Program has been type checked
pub struct TypeChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub checker_builtins: Vec<String>,
    pub schemes: HashMap<String, TypeScheme<TypeName>>,
    pub name_resolution: name_resolution::NameResolution,
}

impl TypeChecked {
    /// Print warnings to stderr (convenience method)
    pub fn print_warnings(&self) {
        for warning in &self.warnings {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| types::format_type(t)),
                warning.span()
            );
        }
    }

    /// Reject programs that still contain `???` type holes. Each hole
    /// is listed with its inferred type and source location; the
    /// caller (the driver) maps `CompilerError::TypeHole` to a
    /// distinct exit code so tooling can tell hole-errors apart from
    /// generic compilation failures. Holes are a development aid —
    /// they're not expected to reach the backend, so this is the
    /// right place to bail.
    pub fn reject_type_holes(self) -> Result<Self> {
        use std::fmt::Write;
        let holes: Vec<_> = self
            .warnings
            .iter()
            .filter_map(|w| match w {
                type_checker::TypeWarning::TypeHoleFilled { inferred_type, span } => {
                    Some((inferred_type, span))
                }
            })
            .collect();
        if holes.is_empty() {
            return Ok(self);
        }
        let mut msg = String::from("type hole(s) in program:\n");
        for (ty, span) in &holes {
            let _ = writeln!(
                &mut msg,
                "  at {}:{} — inferred `{}`",
                span.start_line,
                span.start_col,
                types::format_type(ty),
            );
        }
        Err(err_type_hole!("{}", msg.trim_end()))
    }

    /// Transform AST to TLC. `module_manager` provides access to prelude
    /// declarations. `fill_holes` makes `???` type-hole expressions lower
    /// to a default value of the inferred type rather than panicking. The
    /// driver should call `reject_type_holes` first when
    /// `fill_holes = false`; unfillable hole types under
    /// `fill_holes = true` land in the returned program's
    /// `fill_hole_errors`.
    pub fn to_tlc(
        self,
        module_manager: &module_manager::ModuleManager,
        fill_holes: bool,
    ) -> TlcTransformed {
        let out = tlc::run::run(
            &self.ast,
            self.type_table,
            &self.schemes,
            &self.checker_builtins,
            &self.name_resolution,
            module_manager,
            fill_holes,
        );
        TlcTransformed(TlcEarlyInner {
            tlc: out.program,
            type_table: out.type_table,
            known_defs: out.known_defs,
            schemes: out.schemes,
            fill_hole_errors: out.fill_hole_errors,
        })
    }
}

// =============================================================================
// TLC-based pipeline stages
// =============================================================================

/// Shared payload for the TLC early states (`TlcTransformed`,
/// `TlcPartialEvaled`, `TlcSoaNormalized`, `TlcFused`). Wrapped by each
/// newtype; inner fields are crate-pub so the transition impls can move
/// them across group boundaries by value.
pub struct TlcEarlyInner {
    pub tlc: tlc::Program,
    pub type_table: TypeTable,
    /// Built-in names that should not be captured as free variables
    pub(crate) known_defs: std::collections::HashSet<String>,
    /// Type schemes for functions (for monomorphization)
    pub(crate) schemes: HashMap<SymbolId, types::TypeScheme>,
    /// Errors surfaced while default-filling `???` type holes with
    /// `--fill-holes`. Empty unless `to_tlc` was called with
    /// `fill_holes = true` and some hole had a type that couldn't
    /// be defaulted. The driver checks this before proceeding to
    /// later TLC passes and turns a non-empty list into a
    /// `CompilerError::TypeHole`.
    pub fill_hole_errors: Vec<error::CompilerError>,
}

/// AST has been transformed to TLC
pub struct TlcTransformed(pub TlcEarlyInner);

impl std::ops::Deref for TlcTransformed {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcTransformed {
    /// Constant folding and algebraic simplifications.
    pub fn partial_eval(self) -> TlcPartialEvaled {
        let mut inner = self.0;
        inner.tlc = tlc::partial_eval::PartialEvaluator::partial_eval(inner.tlc);
        TlcPartialEvaled(inner)
    }
}

#[cfg(test)]
impl TlcTransformed {
    /// Test-only: run the canonical TLC optimization pipeline to
    /// `TlcReachable` without the per-stage `time(...)` wrappers the
    /// driver uses. `parallelize_compute = false` matches the
    /// `--single-stage` driver mode.
    pub fn optimize_for_test(self, parallelize_compute: bool) -> TlcReachable {
        self.partial_eval()
            .normalize_soacs()
            .fuse_maps()
            .apply_ownership()
            .expect("apply_ownership")
            .defunctionalize()
            .monomorphize()
            .buffer_specialize()
            .fold_generated_lambdas()
            .inline_small()
            .parallelize_soacs(parallelize_compute)
            .expect("parallelize_soacs")
            .filter_reachable()
    }
}

/// TLC after partial evaluation
pub struct TlcPartialEvaled(pub TlcEarlyInner);

impl std::ops::Deref for TlcPartialEvaled {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcPartialEvaled {
    /// SoA transform + SOAC normalization: rewrite array-of-tuple types,
    /// flatten Map+Zip into multi-input Map, and convert standalone Zip to tuple.
    pub fn normalize_soacs(self) -> TlcSoaNormalized {
        let mut inner = self.0;
        inner.tlc = tlc::soa::run(inner.tlc);
        TlcSoaNormalized(inner)
    }
}

/// TLC after SoA normalization (arrays never contain tuples, zips eliminated)
pub struct TlcSoaNormalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcSoaNormalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcSoaNormalized {
    /// Fuse consecutive SOAC operations to eliminate intermediate arrays.
    pub fn fuse_maps(self) -> TlcFused {
        let mut inner = self.0;
        inner.tlc = tlc::fusion::run(inner.tlc);
        TlcFused(inner)
    }
}

/// TLC after map fusion
pub struct TlcFused(pub TlcEarlyInner);

impl std::ops::Deref for TlcFused {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcFused {
    /// Run the TLC ownership/liveness analysis on the post-fusion IR.
    /// Reports use-after-move errors and applies ownership-driven
    /// rewrites: `_w_intrinsic_array_with` → `_w_intrinsic_array_with_inplace`
    /// where the source is mutable and dead-after, and (in subsequent
    /// phases) consuming-input marking on eligible Map SOACs.
    pub fn apply_ownership(self) -> Result<TlcOwnershipApplied> {
        let mut inner = self.0;
        inner.tlc = tlc::ownership::apply_ownership(inner.tlc)?;
        Ok(TlcOwnershipApplied(inner))
    }
}

/// TLC after ownership-driven rewrites. All `with` calls have
/// settled on either the functional or in-place intrinsic; eligible
/// SOACs (Phase C+) carry the consuming-input flag.
pub struct TlcOwnershipApplied(pub TlcEarlyInner);

impl std::ops::Deref for TlcOwnershipApplied {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcOwnershipApplied {
    /// Run the three-phase closure pipeline:
    /// 1. `closure_convert::run` lifts every lambda to a top-level def
    ///    and produces a `ClosureInfo` side-table describing captures
    ///    per callable symbol.
    /// 2. `hof_specialize::run` clones each HOF for the concrete
    ///    callable that flows in, eliminating function-typed params.
    /// 3. `closure_calls_lower::run` threads captures into call sites
    ///    and verifies every call resolves to a direct, fully-applied
    ///    `Var`-position callee.
    pub fn defunctionalize(self) -> TlcDefunctionalized {
        let TlcEarlyInner {
            tlc,
            type_table,
            known_defs,
            schemes,
            fill_hole_errors: _,
        } = self.0;
        let (cc, closure_info) = tlc::closure_convert::run(tlc, &known_defs);
        let hof_free = tlc::hof_specialize::run(cc, &closure_info);
        let lowered = tlc::closure_calls_lower::run(hof_free, &closure_info);
        TlcDefunctionalized {
            tlc: lowered,
            type_table,
            schemes,
        }
    }
}

/// TLC with all lambdas defunctionalized (lifted + SOAC captures flattened)
pub struct TlcDefunctionalized {
    pub tlc: tlc::Program,
    pub type_table: TypeTable,
    /// Type schemes for functions (for monomorphization)
    schemes: HashMap<SymbolId, types::TypeScheme>,
}

impl TlcDefunctionalized {
    /// Specialize polymorphic intrinsics and monomorphize user functions.
    pub fn monomorphize(self) -> TlcMonomorphized {
        let specialized = tlc::specialize::run(self.tlc);
        let monomorphized = tlc::monomorphize::run(specialized, &self.schemes);
        TlcMonomorphized(TlcLateInner {
            tlc: monomorphized,
            type_table: self.type_table,
        })
    }
}

/// Shared payload for TLC states that carry just `{tlc, type_table}`
/// (`TlcMonomorphized`, `TlcBufferSpecialized`, `TlcGeneratedLambdasFolded`,
/// `TlcSmallInlined`).
pub struct TlcLateInner {
    pub tlc: tlc::Program,
    pub type_table: TypeTable,
}

/// TLC with all functions monomorphized (no type variables remain)
pub struct TlcMonomorphized(pub TlcLateInner);

impl std::ops::Deref for TlcMonomorphized {
    type Target = TlcLateInner;
    fn deref(&self) -> &TlcLateInner {
        &self.0
    }
}

impl TlcMonomorphized {
    /// Specialize functions that take view-array parameters per-buffer.
    /// After this pass, no `DefMeta::Function` has view-array parameters.
    pub fn buffer_specialize(self) -> TlcBufferSpecialized {
        let mut inner = self.0;
        inner.tlc = tlc::buffer_specialize::run(inner.tlc);
        TlcBufferSpecialized(inner)
    }
}

/// TLC after buffer specialization (no functions have view-array params)
pub struct TlcBufferSpecialized(pub TlcLateInner);

impl std::ops::Deref for TlcBufferSpecialized {
    type Target = TlcLateInner;
    fn deref(&self) -> &TlcLateInner {
        &self.0
    }
}

impl TlcBufferSpecialized {
    /// Inline compiler-generated `_w_lambda_*` defs back at their call sites,
    /// then remove unreferenced defs (DCE).
    pub fn fold_generated_lambdas(self) -> TlcGeneratedLambdasFolded {
        let mut inner = self.0;
        inner.tlc = tlc::inline::run_large(inner.tlc);
        TlcGeneratedLambdasFolded(inner)
    }
}

/// TLC after inlining compiler-generated lambda defs and DCE
pub struct TlcGeneratedLambdasFolded(pub TlcLateInner);

impl std::ops::Deref for TlcGeneratedLambdasFolded {
    type Target = TlcLateInner;
    fn deref(&self) -> &TlcLateInner {
        &self.0
    }
}

impl TlcGeneratedLambdasFolded {
    /// Inline small user functions and constants at their call/reference sites.
    pub fn inline_small(self) -> TlcSmallInlined {
        let mut inner = self.0;
        inner.tlc = tlc::inline::run_small(inner.tlc);
        TlcSmallInlined(inner)
    }

    /// Eliminate unreachable defs (dead code elimination at TLC level).
    pub fn filter_reachable(self) -> TlcReachable {
        let TlcLateInner { tlc, type_table } = self.0;
        let tlc = tlc::inline::run_reachable(tlc);
        TlcReachable(TlcPipelineInner {
            tlc,
            pipeline: pipeline_descriptor::PipelineDescriptor::default(),
            type_table,
        })
    }

    /// Build the raw EGIR program. Callers chain the pipeline
    /// (`expand_soacs → [materialize →] optimize_skeleton → elaborate`)
    /// explicitly — materialize is the only optional pass and is required for
    /// SPIR-V but not GLSL.
    pub fn to_egraph(self) -> std::result::Result<EgirRaw, ConvertError> {
        egir::from_tlc::run(&self.0.tlc, pipeline_descriptor::PipelineDescriptor::default()).map(EgirRaw)
    }
}

/// TLC after small function and constant inlining
pub struct TlcSmallInlined(pub TlcLateInner);

impl std::ops::Deref for TlcSmallInlined {
    type Target = TlcLateInner;
    fn deref(&self) -> &TlcLateInner {
        &self.0
    }
}

impl TlcSmallInlined {
    /// Parallelize SOACs in compute entry points at the TLC level.
    /// `disable` turns the pass into an effective no-op — compute SOACs
    /// remain as single-threaded sequential loops in their original
    /// entries, graphical entries get no restructuring, and the pipeline
    /// descriptor is built as if every entry runs in one stage.
    pub fn parallelize_soacs(self, disable: bool) -> Result<TlcParallelized> {
        let TlcLateInner { tlc, type_table } = self.0;
        let result = tlc::parallelize::run(tlc, disable)?;
        Ok(TlcParallelized(TlcPipelineInner {
            tlc: result.program,
            pipeline: result.pipeline,
            type_table,
        }))
    }

    /// Eliminate unreachable defs (dead code elimination at TLC level).
    pub fn filter_reachable(self) -> TlcReachable {
        let TlcLateInner { tlc, type_table } = self.0;
        let tlc = tlc::inline::run_reachable(tlc);
        TlcReachable(TlcPipelineInner {
            tlc,
            pipeline: pipeline_descriptor::PipelineDescriptor::default(),
            type_table,
        })
    }

    /// Build the raw EGIR program. Callers chain the pipeline
    /// (`expand_soacs → [materialize →] optimize_skeleton → elaborate`)
    /// explicitly — materialize is the only optional pass and is required for
    /// SPIR-V but not GLSL.
    pub fn to_egraph(self) -> std::result::Result<EgirRaw, ConvertError> {
        egir::from_tlc::run(&self.0.tlc, pipeline_descriptor::PipelineDescriptor::default()).map(EgirRaw)
    }
}

/// Shared payload for TLC states carrying `{tlc, pipeline, type_table}`
/// (`TlcParallelized`, `TlcReachable`).
pub struct TlcPipelineInner {
    pub tlc: tlc::Program,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
    pub type_table: TypeTable,
}

/// TLC after SOAC parallelization
pub struct TlcParallelized(pub TlcPipelineInner);

impl std::ops::Deref for TlcParallelized {
    type Target = TlcPipelineInner;
    fn deref(&self) -> &TlcPipelineInner {
        &self.0
    }
}

impl TlcParallelized {
    /// Eliminate unreachable defs (dead code elimination at TLC level).
    pub fn filter_reachable(self) -> TlcReachable {
        let mut inner = self.0;
        inner.tlc = tlc::inline::run_reachable(inner.tlc);
        TlcReachable(inner)
    }

    pub fn to_egraph(self) -> std::result::Result<EgirRaw, ConvertError> {
        let TlcPipelineInner { tlc, pipeline, .. } = self.0;
        egir::from_tlc::run(&tlc, pipeline).map(EgirRaw)
    }
}

/// TLC after dead code elimination. Always carries a `PipelineDescriptor`
/// (empty for non-parallelized paths).
pub struct TlcReachable(pub TlcPipelineInner);

impl std::ops::Deref for TlcReachable {
    type Target = TlcPipelineInner;
    fn deref(&self) -> &TlcPipelineInner {
        &self.0
    }
}

impl TlcReachable {
    pub fn to_egraph(self) -> std::result::Result<EgirRaw, ConvertError> {
        let TlcPipelineInner { tlc, pipeline, .. } = self.0;
        egir::from_tlc::run(&tlc, pipeline).map(EgirRaw)
    }
}

// =============================================================================
// EGIR typestate chain
//
// Four newtypes over a shared `EgirInner` (defined in `egir::program`).
// Transitions consume `self` and re-wrap the inner into the next newtype.
// Pass modules in `egir::*` are called per-body from inside the transitions
// and are unaware of the newtype wrapping.
// =============================================================================

/// Raw EGIR program, directly produced by TLC → EGIR conversion.
pub struct EgirRaw(EgirInner);

/// EGIR after SOAC lowering: every `PendingSoac::{Map, Scan, Reduce, …}` in
/// the skeleton has been expanded to explicit loops / unrolled code.
pub struct EgirSoacExpanded(EgirInner);

/// EGIR after materialization: dynamic `Index` into non-materialized composite
/// values has been rewritten to `Materialize` + `DynamicExtract`. SPIR-V only.
pub struct EgirMaterialized(EgirInner);

/// EGIR after skeleton-CFG optimizations (LICM, dead-block elim, etc).
pub struct EgirSkelOptimized(EgirInner);

impl EgirRaw {
    /// `unroll_maps`: whether to unroll small-constant-length Maps into
    /// straight-line code. Typically `true` for SPIR-V and `false` for GLSL
    /// (where drivers unroll themselves and the structurizer prefers loops).
    pub fn expand_soacs(self, unroll_maps: bool) -> EgirSoacExpanded {
        let EgirRaw(mut inner) = self;
        egir::soac_expand::run(&mut inner, unroll_maps);
        EgirSoacExpanded(inner)
    }
}

impl EgirSoacExpanded {
    pub fn materialize(self) -> EgirMaterialized {
        let EgirSoacExpanded(mut inner) = self;
        egir::materialize::run(&mut inner);
        EgirMaterialized(inner)
    }

    pub fn optimize_skeleton(self) -> EgirSkelOptimized {
        let EgirSoacExpanded(mut inner) = self;
        egir::skel_opt::run(&mut inner);
        EgirSkelOptimized(inner)
    }
}

impl EgirMaterialized {
    pub fn optimize_skeleton(self) -> EgirSkelOptimized {
        let EgirMaterialized(mut inner) = self;
        egir::skel_opt::run(&mut inner);
        EgirSkelOptimized(inner)
    }
}

impl EgirSkelOptimized {
    /// Terminal step: lower each per-body e-graph to SSA and assemble the
    /// final `SsaConverted`.
    pub fn elaborate(self) -> SsaConverted {
        let EgirSkelOptimized(inner) = self;
        let (ssa, pipeline) = egir::elaborate::run_program(inner);
        SsaConverted { ssa, pipeline }
    }
}

/// TLC has been converted directly to SSA (via EGIR).
pub struct SsaConverted {
    pub ssa: ssa::types::Program,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
}

impl SsaConverted {
    /// Lower SSA to SPIR-V. Materialization (`Index` → `Materialize` +
    /// `DynamicExtract`) was already done by the EGIR pipeline if `to_egir`
    /// was called with `EgirOpts::for_spirv()`.
    pub fn lower(self) -> error::Result<Lowered> {
        let spirv = spirv::lower_ssa_program(&self.ssa)?;
        Ok(Lowered {
            spirv,
            pipeline: self.pipeline,
        })
    }

    pub fn lower_glsl(self) -> error::Result<glsl::GlslOutput> {
        glsl::lower(&self.ssa)
    }

    pub fn lower_shadertoy(self) -> error::Result<String> {
        glsl::lower_shadertoy(&self.ssa)
    }

    pub fn lower_wgsl(self) -> error::Result<String> {
        wgsl::lower(&self.ssa)
    }
}

/// Final SPIR-V output
pub struct Lowered {
    pub spirv: Vec<u32>,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
}

// =============================================================================
// Test utilities - cached prelude for faster test execution
// =============================================================================

#[cfg(test)]
use std::sync::OnceLock;

/// Cached prelude data AND the node counter state after parsing it
#[cfg(test)]
static PRELUDE_CACHE: OnceLock<(module_manager::PreElaboratedPrelude, NodeCounter)> = OnceLock::new();

/// Get the cached prelude and a cloned node counter (test-only)
/// This avoids re-parsing prelude files for each test, providing ~10x speedup
#[cfg(test)]
fn get_prelude_cache() -> (&'static module_manager::PreElaboratedPrelude, NodeCounter) {
    let (prelude, counter) = PRELUDE_CACHE.get_or_init(|| {
        let mut nc = NodeCounter::new();
        let prelude =
            module_manager::ModuleManager::create_prelude(&mut nc).expect("Failed to create prelude cache");
        (prelude, nc)
    });
    (prelude, counter.clone())
}

/// Create a ModuleManager and NodeCounter using the cached prelude (test-only)
#[cfg(test)]
pub fn cached_module_manager() -> (module_manager::ModuleManager, NodeCounter) {
    let (prelude, node_counter) = get_prelude_cache();
    (
        module_manager::ModuleManager::from_prelude(prelude.clone()),
        node_counter,
    )
}

/// Build a `(NodeCounter, ModuleManager)` pair using the cached prelude (test-only).
#[cfg(test)]
pub fn cached_compiler_init() -> (NodeCounter, module_manager::ModuleManager) {
    let (prelude, node_counter) = get_prelude_cache();
    init_compiler_from_prelude(prelude.clone(), node_counter)
}

// =============================================================================
// Test-only milestone helpers
// =============================================================================
//
// `compile_thru_*` helpers run the pipeline up to a milestone and return
// just the milestone value. Each subsumes the previous one:
//
//   compile_thru_frontend  →  TypeChecked          (AST passes done)
//   compile_thru_tlc       →  TlcReachable         (TLC pipeline done)
//   compile_thru_ssa       →  SsaConverted         (EGIR + elaborate done)
//   compile_thru_spirv     →  Lowered              (final SPIR-V binary)
//
// These exist so test files don't have to enumerate every pass — when a
// new pass lands, only the helper that owns its milestone needs updating.
// Tests that need an off-milestone stop call the typestate methods directly.

/// Run AST passes through type checking. Uses the cached prelude.
#[cfg(test)]
pub fn compile_thru_frontend(source: &str) -> error::Result<TypeChecked> {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    Compiler::parse(source, &mut node_counter)?
        .elaborate_modules(&mut module_manager, &mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&mut module_manager)
}

/// Run the canonical TLC optimization pipeline (no compute parallelization,
/// no hole-filling) through `filter_reachable`.
#[cfg(test)]
pub fn compile_thru_tlc(source: &str) -> error::Result<TlcReachable> {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    let type_checked = Compiler::parse(source, &mut node_counter)?
        .elaborate_modules(&mut module_manager, &mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&mut module_manager)?;
    Ok(type_checked.to_tlc(&module_manager, false).optimize_for_test(false))
}

/// Run all the way through EGIR + elaborate to SSA. Materialize is enabled
/// (matches the SPIR-V backend's requirements). Returns the boxed
/// `Result<_, dyn Error>` so callers see both compiler errors and EGIR
/// conversion errors uniformly.
#[cfg(test)]
pub fn compile_thru_ssa(source: &str) -> std::result::Result<SsaConverted, Box<dyn std::error::Error>> {
    let tlc = compile_thru_tlc(source)?;
    let raw = tlc.to_egraph()?;
    Ok(raw.expand_soacs(true).materialize().optimize_skeleton().elaborate())
}

/// Run the full pipeline to a final SPIR-V binary.
#[cfg(test)]
pub fn compile_thru_spirv(source: &str) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(compile_thru_ssa(source)?.lower()?)
}
