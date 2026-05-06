pub mod ast;
pub mod builtins;
pub mod diags;
pub mod error;
pub mod impl_source;
pub mod interface;
pub mod intrinsics;
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
pub mod desugar;
pub mod lowering_common;
pub mod name_registry;
pub mod tlc;

pub mod egir;
pub mod glsl;
/// Re-export of the pipeline descriptor format. Lives in its own
/// crate so host runtimes (e.g. `extra/viz`) can deserialize the
/// JSON without pulling in the whole compiler.
pub use wyn_pipeline_descriptor as pipeline_descriptor;
pub mod resolve_opens;
pub mod resolve_placeholders;
pub mod spirv;
pub mod structured;
pub mod wgsl;

#[cfg(test)]
mod integration_tests;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use egir::from_tlc::ConvertError;
use egir::program::EgirInner;

use indexmap::IndexMap;

use ast::{NodeCounter, NodeId};
use error::Result;
use polytype::{Context, TypeScheme};

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
pub type SpanTable = HashMap<NodeId, ast::Span>;

/// Build a SpanTable from an AST by collecting all NodeId -> Span mappings
pub fn build_span_table(program: &ast::Program) -> SpanTable {
    use std::ops::ControlFlow;
    use visitor::Visitor;

    struct SpanCollector {
        spans: SpanTable,
    }

    impl Visitor for SpanCollector {
        type Break = ();

        fn visit_expression(&mut self, e: &ast::Expression) -> ControlFlow<Self::Break> {
            self.spans.insert(e.h.id, e.h.span);
            visitor::walk_expression(self, e)
        }
    }

    let mut collector = SpanCollector {
        spans: HashMap::new(),
    };

    for decl in &program.declarations {
        let _ = visitor::walk_declaration(&mut collector, decl);
    }

    collector.spans
}

// =============================================================================
// Two-Level Typestate Compiler Pipeline
// =============================================================================
//
// The compiler uses a typestate pattern where each struct represents a stage.
// Methods consume `self` and return the next stage, enforcing valid ordering.
//
// Shared state is held at two top levels and passed as &mut to passes that need it:
//   - FrontEnd: holds module_manager (used by resolve, type_check, flatten)
//   - BackEnd: holds node_counter (used by normalize)
//
// FrontEnd Pipeline (AST):
//   let parsed = Compiler::parse(source)?;
//   let mut frontend = FrontEnd::new(module_manager);
//     -> parsed.desugar(&mut node_counter)              -> Desugared
//       -> .resolve(&frontend.module_manager)           -> Resolved
//       -> .fold_ast_constants()                        -> AstConstFoldedEarly
//       -> .type_check(&frontend.module_manager)        -> TypeChecked
//
// TLC Pipeline (AST -> SSA):
//       -> .to_tlc()                                    -> TlcTransformed
//       -> .partial_eval()                              -> TlcPartialEvaled
//       -> .normalize_soacs()                             -> TlcSoaNormalized
//       -> .fuse_maps()                                 -> TlcFused
//       -> .apply_ownership()                           -> TlcOwnershipApplied
//       -> .defunctionalize()                           -> TlcDefunctionalized
//       -> .monomorphize()                              -> TlcMonomorphized
//       -> .buffer_specialize()                         -> TlcBufferSpecialized
//       -> .fold_generated_lambdas()                    -> TlcGeneratedLambdasFolded
//       -> .inline_small()                              -> TlcSmallInlined
//       -> .parallelize_soacs(false)                         -> TlcParallelized
//       -> .filter_reachable()                          -> TlcReachable
//       -> .to_egir()                                   -> SsaConverted
//
// BackEnd Pipeline (SSA -> output):
//       -> .lower()                                     -> Lowered

// =============================================================================
// Top-level state containers
// =============================================================================

/// Shared state for FrontEnd (AST) passes.
/// Holds the node counter and module manager used by parse, resolve, type_check, and flatten.
pub struct FrontEnd {
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
    /// Type variable allocator for polymorphic types
    pub context: Context<TypeName>,
    /// Maps AST nodes to their inferred type schemes
    pub type_table: TypeTable,
    /// Top-level function type schemes (includes prelude and user-defined functions)
    pub schemes: HashMap<String, TypeScheme<TypeName>>,
    /// Per-module function type schemes cache (populated on first use)
    pub module_schemes: HashMap<String, HashMap<String, TypeScheme<TypeName>>>,
}

impl Default for FrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontEnd {
    /// Create a new FrontEnd with fresh state.
    /// The node counter is shared between user code parsing and prelude loading
    /// to ensure unique NodeIds across all AST nodes.
    pub fn new() -> Self {
        let mut node_counter = NodeCounter::new();

        // Create prelude (parsed/elaborated ASTs only - type-checking happens later)
        let module_manager = match module_manager::ModuleManager::create_prelude(&mut node_counter) {
            Ok(prelude) => module_manager::ModuleManager::from_prelude(prelude),
            Err(e) => {
                eprintln!("ERROR creating prelude: {:?}", e);
                module_manager::ModuleManager::new_empty()
            }
        };

        // Type-related state is populated during type_check()
        let context = Context::default();

        FrontEnd {
            node_counter,
            module_manager,
            context,
            type_table: HashMap::new(),
            schemes: HashMap::new(),
            module_schemes: HashMap::new(),
        }
    }

    /// Create a FrontEnd from a pre-elaborated prelude.
    /// This is faster than `new()` as it reuses an already-parsed prelude.
    pub fn new_from_prelude(
        prelude: module_manager::PreElaboratedPrelude,
        node_counter: NodeCounter,
    ) -> Self {
        let module_manager = module_manager::ModuleManager::from_prelude(prelude);
        let context = Context::default();

        FrontEnd {
            node_counter,
            module_manager,
            context,
            type_table: HashMap::new(),
            schemes: HashMap::new(),
            module_schemes: HashMap::new(),
        }
    }
}

/// Shared state for BackEnd (MIR) passes.
/// Holds the node counter which is used by normalize.
pub struct BackEnd {
    pub node_counter: NodeCounter,
}

impl BackEnd {
    /// Create a new BackEnd with the given node counter.
    pub fn new(node_counter: NodeCounter) -> Self {
        BackEnd { node_counter }
    }
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
        let mut visited: std::collections::HashSet<std::path::PathBuf> = std::collections::HashSet::new();
        self.0.ast.declarations =
            resolve_imports_inner(self.0.ast.declarations, base_dir, node_counter, &mut visited)?;
        Ok(self)
    }

    /// Elaborate inline module declarations from the parsed program.
    /// This registers modules with the module_manager so they're available during resolution,
    /// then removes the Module declarations from the AST (they've been copied to module_manager).
    /// Should be called before desugar() if the program contains module definitions.
    pub fn elaborate_modules(mut self, module_manager: &mut module_manager::ModuleManager) -> Result<Self> {
        module_manager.elaborate_modules(&self.0.ast)?;
        // Remove Module and ModuleTypeBind declarations - they've been elaborated
        self.0.ast.declarations.retain(|decl| {
            !matches!(
                decl,
                ast::Declaration::Module(_) | ast::Declaration::ModuleTypeBind(_)
            )
        });
        Ok(self)
    }

    /// Desugar range and slice expressions into map/iota constructs.
    /// Should be called before resolve() to ensure the generated identifiers
    /// (iota, map) get properly resolved.
    pub fn desugar(mut self, nc: &mut ast::NodeCounter) -> Result<Desugared> {
        desugar::run(&mut self.0.ast, nc).map(|()| Desugared(self.0))
    }
}

/// Recursively expand `Declaration::Import` nodes against the
/// filesystem. Each imported file's declarations replace the import
/// node in-place; transitive imports inside the loaded file are
/// resolved relative to that file's directory. A canonical-path
/// dedup set prevents infinite loops on cyclic imports and dedupes
/// diamond imports.
fn resolve_imports_inner(
    decls: Vec<ast::Declaration>,
    base_dir: &std::path::Path,
    node_counter: &mut NodeCounter,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
) -> Result<Vec<ast::Declaration>> {
    let mut out: Vec<ast::Declaration> = Vec::with_capacity(decls.len());
    for decl in decls {
        let ast::Declaration::Import(rel_path) = decl else {
            out.push(decl);
            continue;
        };

        // Resolve `import "foo"` to `<base_dir>/foo.wyn`. If the user
        // already wrote a `.wyn` extension, don't double-add it.
        let mut joined = base_dir.join(&rel_path);
        if joined.extension().is_none() {
            joined.set_extension("wyn");
        }
        let canonical = joined.canonicalize().map_err(|e| {
            err_module!(
                "import: cannot resolve `{}` (looked for `{}`): {}",
                rel_path,
                joined.display(),
                e
            )
        })?;
        if !visited.insert(canonical.clone()) {
            // Already loaded — diamond import; silently dedupe.
            continue;
        }

        let source = std::fs::read_to_string(&canonical)
            .map_err(|e| err_module!("import: failed to read `{}`: {}", canonical.display(), e))?;
        let imported = Compiler::parse(&source, node_counter)?;
        let imported_dir = canonical.parent().unwrap_or(base_dir);
        let resolved =
            resolve_imports_inner(imported.0.ast.declarations, imported_dir, node_counter, visited)?;
        out.extend(resolved);
    }
    Ok(out)
}

/// Range and slice expressions have been desugared to map/iota
pub struct Desugared(FrontInner);

impl Desugared {
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
    pub fn type_check(
        mut self,
        module_manager: &mut module_manager::ModuleManager,
        schemes: &mut HashMap<String, TypeScheme<TypeName>>,
    ) -> Result<TypeChecked> {
        // Resolve type placeholders to type variables before type checking
        let mut resolver = resolve_placeholders::PlaceholderResolver::new();
        resolver.resolve(module_manager, &mut self.0.ast);
        let (context, spec_schemes) = resolver.into_parts();

        // `open M` name resolution. Builds the open index from the
        // union of (a) module-spec schemes (keyed `M.name`) and (b)
        // catalog per-type and per-intrinsic names (`f32.cos`,
        // `_w_intrinsic_*`, …), then rewrites bare names that uniquely
        // match one open.
        let scheme_keys = spec_schemes.keys().cloned();
        let catalog_keys: Vec<String> = crate::builtins::catalog()
            .defs()
            .iter()
            .flat_map(|d| {
                d.impl_source_names()
                    .iter()
                    .copied()
                    .chain(d.intrinsic_source_names().iter().copied())
                    .map(|s| s.to_string())
            })
            .collect();
        resolve_opens::run(&mut self.0.ast, scheme_keys.chain(catalog_keys))?;

        let mut checker =
            type_checker::TypeChecker::with_context_and_schemes(module_manager, context, spec_schemes);
        checker.set_name_resolution(crate::name_resolution::build_name_resolution(
            &self.0.ast,
            crate::builtins::catalog(),
        ));
        checker.load_builtins()?;
        let type_table = checker.check_program(&self.0.ast)?;
        // Populate schemes with function type schemes from type checking
        *schemes = checker.get_function_schemes();
        let checker_builtins = checker.builtin_names();
        let warnings: Vec<_> = checker.warnings().to_vec();
        let span_table = build_span_table(&self.0.ast);

        Ok(TypeChecked {
            ast: self.0.ast,
            type_table,
            span_table,
            warnings,
            checker_builtins,
        })
    }
}

/// Program has been type checked
pub struct TypeChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub span_table: SpanTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub checker_builtins: Vec<String>,
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

    /// Transform AST to TLC. `schemes` contains type schemes for all
    /// functions (populated during type_check). `module_manager`
    /// provides access to prelude declarations. `fill_holes` makes
    /// `???` type-hole expressions lower to a default value of the
    /// inferred type rather than panicking. The driver should call
    /// `reject_type_holes` first when `fill_holes = false`;
    /// unfillable hole types under `fill_holes = true` land in the
    /// returned program's `fill_hole_errors`.
    pub fn to_tlc(
        mut self,
        schemes: &HashMap<String, types::TypeScheme>,
        module_manager: &module_manager::ModuleManager,
        fill_holes: bool,
    ) -> TlcTransformed {
        // Under `--fill-holes`, rewrite any free type variable in the
        // node-level type table to `i32` so holes whose type stayed
        // unconstrained (no call-site or annotation pinned them) get a
        // ground type instead of surfacing as a fill-hole error.
        if fill_holes {
            tlc::defaults::default_free_vars_in_table(self.type_table.values_mut());
        }

        // Build unified name registry — single source of truth for all top-level names.
        let registry =
            name_registry::NameRegistry::build(&self.ast, module_manager, &self.checker_builtins);

        // Pre-register ALL names with deterministic SymbolId assignment (BTreeMap order).
        let mut symbols = SymbolTable::new();
        let mut top_level_symbols = std::collections::HashMap::new();
        for (name, _kind) in registry.iter() {
            let sym = symbols.alloc(name.to_string());
            top_level_symbols.insert(name.to_string(), sym);
        }

        // Shared accumulator for `--fill-holes` default-fill failures.
        // Errors from every Transformer that runs below land here.
        let mut fill_hole_errors: Vec<error::CompilerError> = Vec::new();

        // Transform prelude module declarations (f32.pi, rand.init, etc.)
        // All names already registered — no separate first pass needed.
        let mut prelude_tlc_defs = Vec::new();
        for (module_name, elaborated) in module_manager.get_elaborated_modules() {
            let mut transformer = tlc::Transformer::with_namespace(
                &self.type_table,
                &mut symbols,
                &mut top_level_symbols,
                module_name,
                fill_holes,
                &mut fill_hole_errors,
            );
            for item in &elaborated.items {
                if let module_manager::ElaboratedItem::Decl(decl) = item {
                    if let Some(def) = transformer.transform_decl(decl) {
                        prelude_tlc_defs.push(def);
                    }
                }
            }
        }

        // Transform top-level prelude functions (unzip, all, any, etc.)
        {
            let mut transformer = tlc::Transformer::new(
                &self.type_table,
                &mut symbols,
                &mut top_level_symbols,
                fill_holes,
                &mut fill_hole_errors,
            );
            for decl in module_manager.get_prelude_function_declarations() {
                if let Some(def) = transformer.transform_decl(decl) {
                    prelude_tlc_defs.push(def);
                }
            }
        }

        // Transform user program to TLC
        let mut transformer = tlc::Transformer::new(
            &self.type_table,
            &mut symbols,
            &mut top_level_symbols,
            fill_holes,
            &mut fill_hole_errors,
        );
        let mut parts = transformer.transform_program(&self.ast);

        // Prepend prelude TLC defs
        let mut merged_defs = prelude_tlc_defs;
        merged_defs.extend(parts.defs);
        parts.defs = merged_defs;

        // Combine parts with the symbol table to create the final Program
        let tlc_program = parts.with_symbols(symbols, top_level_symbols);

        TlcTransformed(TlcEarlyInner {
            tlc: tlc_program,
            type_table: self.type_table,
            known_defs: registry.name_set(),
            schemes: schemes.clone(),
            fill_hole_errors,
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
    pub(crate) schemes: HashMap<String, types::TypeScheme>,
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
        inner.tlc.assert_flat_apps();
        inner.tlc = tlc::partial_eval::PartialEvaluator::partial_eval(inner.tlc);
        inner.tlc.assert_flat_apps();
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
    /// Defunctionalize: lift lambdas and flatten SOAC closure captures.
    pub fn defunctionalize(self) -> TlcDefunctionalized {
        let TlcEarlyInner {
            tlc,
            type_table,
            known_defs,
            schemes,
            fill_hole_errors: _,
        } = self.0;
        let defunc = tlc::defunctionalize::run(tlc, &known_defs);
        defunc.assert_flat_apps();
        tlc::closure_convert::verify_closure_converted(&defunc).unwrap_or_else(|e| {
            panic!(
                "closure-conversion verifier failed after defunctionalize: {:?}",
                e
            )
        });
        tlc::closure_calls_lower::verify_closure_calls_lowered(&defunc).unwrap_or_else(|e| {
            panic!(
                "closure-calls-lowered verifier failed after defunctionalize: {:?}",
                e
            )
        });
        TlcDefunctionalized {
            tlc: defunc,
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
    schemes: HashMap<String, types::TypeScheme>,
}

impl TlcDefunctionalized {
    /// Specialize polymorphic intrinsics and monomorphize user functions.
    pub fn monomorphize(self) -> TlcMonomorphized {
        // Specialize polymorphic intrinsics (sign → f32.sign, etc.)
        let specialized = tlc::specialize::run(self.tlc);

        // Convert string-keyed schemes to SymbolId-keyed for monomorphization
        let name_to_sym: HashMap<&str, SymbolId> =
            specialized.symbols.iter().map(|(&id, name)| (name.as_str(), id)).collect();
        let schemes_by_sym: HashMap<SymbolId, types::TypeScheme> = self
            .schemes
            .iter()
            .filter_map(|(name, scheme)| name_to_sym.get(name.as_str()).map(|&sym| (sym, scheme.clone())))
            .collect();

        // Monomorphize polymorphic user functions
        let monomorphized = tlc::monomorphize::run(specialized, &schemes_by_sym);
        monomorphized.assert_flat_apps();
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
        inner.tlc.assert_flat_apps();
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
        inner.tlc.assert_flat_apps();
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
        tlc::hof_specialize::verify_hof_specialized(&tlc).unwrap_or_else(|e| {
            panic!(
                "hof-specialization verifier failed after filter_reachable: {:?}",
                e
            )
        });
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
    pub fn parallelize_soacs(self, disable: bool) -> TlcParallelized {
        let TlcLateInner { tlc, type_table } = self.0;
        let result = tlc::parallelize::run(tlc, disable);
        TlcParallelized(TlcPipelineInner {
            tlc: result.program,
            pipeline: result.pipeline,
            type_table,
        })
    }

    /// Eliminate unreachable defs (dead code elimination at TLC level).
    pub fn filter_reachable(self) -> TlcReachable {
        let TlcLateInner { tlc, type_table } = self.0;
        let tlc = tlc::inline::run_reachable(tlc);
        tlc::hof_specialize::verify_hof_specialized(&tlc).unwrap_or_else(|e| {
            panic!(
                "hof-specialization verifier failed after filter_reachable: {:?}",
                e
            )
        });
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
        use egir::domtree::{DomTree, SkeletonCfgView};
        use ssa::types::{BlockId, EntryPoint, Function};
        use std::collections::HashMap as Map;

        let EgirSkelOptimized(inner) = self;

        let functions: Vec<Function> = inner
            .functions
            .into_iter()
            .map(|f| {
                let body =
                    elaborate_one_body(f.graph, &f.control_headers, &f.aliases, &f.params, f.return_ty);
                Function {
                    name: f.name,
                    body,
                    span: f.span,
                    linkage_name: f.linkage_name,
                }
            })
            .chain(inner.externs.into_iter())
            .collect();

        let entry_points: Vec<EntryPoint> = inner
            .entry_points
            .into_iter()
            .map(|e| {
                let body =
                    elaborate_one_body(e.graph, &e.control_headers, &e.aliases, &e.params, e.return_ty);
                EntryPoint {
                    name: e.name,
                    body,
                    execution_model: e.execution_model,
                    inputs: e.inputs,
                    outputs: e.outputs,
                    storage_bindings: e.storage_bindings,
                    span: e.span,
                }
            })
            .collect();

        fn elaborate_one_body(
            graph: egir::types::EGraph,
            control_headers: &Map<BlockId, ssa::types::ControlHeader>,
            aliases: &Map<egir::types::NodeId, egir::types::NodeId>,
            params: &[(polytype::Type<ast::TypeName>, String)],
            return_ty: polytype::Type<ast::TypeName>,
        ) -> ssa::types::FuncBody {
            let skel_domtree = DomTree::build(&SkeletonCfgView {
                skeleton: &graph.skeleton,
            });
            let identity_map: Map<BlockId, BlockId> =
                graph.skeleton.blocks.keys().map(|b| (b, b)).collect();
            egir::elaborate::run(
                &graph,
                &skel_domtree,
                params,
                return_ty,
                control_headers,
                &identity_map,
                aliases,
            )
        }

        SsaConverted {
            ssa: ssa::types::Program {
                functions,
                entry_points,
                constants: inner.constants,
                uniforms: inner.uniforms,
                storage: inner.storage,
            },
            pipeline: inner.pipeline,
        }
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

/// Create a FrontEnd using the cached prelude (test-only)
#[cfg(test)]
pub fn cached_frontend() -> FrontEnd {
    let (prelude, node_counter) = get_prelude_cache();
    FrontEnd::new_from_prelude(prelude.clone(), node_counter)
}
