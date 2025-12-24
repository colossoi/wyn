pub mod ast;
pub mod defun_analysis;
pub mod diags;
pub mod error;
pub mod flattening;
pub mod impl_source;
pub mod intrinsics;
pub mod lexer;
pub mod mir;
pub mod module_manager;
pub mod name_resolution;
pub mod parser;
pub mod pattern;
pub mod reachability;
pub mod scope;
pub mod types;
pub mod visitor;

// Re-export type_checker from its new location for backwards compatibility
pub use types::checker as type_checker;

pub mod alias_checker;
pub mod ast_const_fold;
pub mod desugar;
pub mod lowering_common;

pub mod binding_lifter;
pub mod constant_folding;
pub mod glsl;
pub mod materialize_hoisting;
pub mod monomorphization;
pub mod normalize;
pub mod partial_eval;
pub mod spirv;

#[cfg(test)]
mod alias_checker_tests;
#[cfg(test)]
mod ast_const_fold_tests;
#[cfg(test)]
mod desugar_tests;
#[cfg(test)]
mod pattern_tests;

#[cfg(test)]
mod binding_lifter_tests;
#[cfg(test)]
mod constant_folding_tests;
#[cfg(test)]
mod flattening_tests;
#[cfg(test)]
mod scope_tests;
// #[cfg(test)]
// mod monomorphization_tests;
// #[cfg(test)]
// mod normalize_tests;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

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
// FrontEnd Pipeline (AST -> MIR):
//   let parsed = Compiler::parse(source)?;
//   let mut frontend = FrontEnd::new(module_manager);
//     -> parsed.desugar(&mut node_counter)              -> Desugared
//       -> .resolve(&frontend.module_manager)           -> Resolved
//       -> .fold_ast_constants()                        -> AstConstFoldedEarly
//       -> .type_check(&frontend.module_manager)        -> TypeChecked
//       -> .alias_check()                               -> AliasChecked
//       -> .flatten(&frontend.module_manager)           -> Flattened
//
// BackEnd Pipeline (MIR -> output):
//   let mut backend = BackEnd::new(node_counter);
//     -> flattened.hoist_materializations()             -> MaterializationsHoisted
//       -> .normalize()                                 -> Normalized
//       -> .monomorphize()                              -> Monomorphized
//       -> .fold_constants() or .partial_eval()         -> Folded
//       -> .filter_reachable()                          -> Reachable
//       -> .lift_bindings()                             -> Lifted
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
    /// Polymorphic intrinsic function types
    pub intrinsics: intrinsics::IntrinsicSource,
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
        let module_manager = module_manager::ModuleManager::new(&mut node_counter);

        // Type-related state is populated during type_check()
        let context = Context::default();
        let intrinsics = intrinsics::IntrinsicSource::new(&mut Context::default());

        FrontEnd {
            node_counter,
            module_manager,
            context,
            type_table: HashMap::new(),
            intrinsics,
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
        Ok(Parsed { ast })
    }
}

// =============================================================================
// FrontEnd stages (AST-based)
// =============================================================================

/// Source has been parsed into an AST
pub struct Parsed {
    pub ast: ast::Program,
}

impl Parsed {
    /// Desugar range and slice expressions into map/iota constructs.
    /// Should be called before resolve() to ensure the generated identifiers
    /// (iota, map) get properly resolved.
    pub fn desugar(mut self, nc: &mut ast::NodeCounter) -> Result<Desugared> {
        desugar::desugar_program(&mut self.ast, nc)?;
        Ok(Desugared { ast: self.ast })
    }
}

/// Range and slice expressions have been desugared to map/iota
pub struct Desugared {
    pub ast: ast::Program,
}

impl Desugared {
    /// Resolve names: rewrite FieldAccess -> QualifiedName and load modules
    pub fn resolve(mut self, module_manager: &module_manager::ModuleManager) -> Result<Resolved> {
        name_resolution::resolve_program(&mut self.ast, module_manager)?;
        Ok(Resolved { ast: self.ast })
    }
}

/// Names have been resolved
pub struct Resolved {
    pub ast: ast::Program,
}

impl Resolved {
    /// Fold AST-level integer constants (required before type checking)
    pub fn fold_ast_constants(mut self) -> AstConstFoldedEarly {
        ast_const_fold::fold_ast_constants(&mut self.ast);
        AstConstFoldedEarly { ast: self.ast }
    }
}

/// AST integer constants have been folded (before type checking)
pub struct AstConstFoldedEarly {
    pub ast: ast::Program,
}

impl AstConstFoldedEarly {
    /// Type check the program
    pub fn type_check(
        self,
        module_manager: &module_manager::ModuleManager,
        schemes: &mut HashMap<String, TypeScheme<TypeName>>,
    ) -> Result<TypeChecked> {
        let mut checker = type_checker::TypeChecker::new(module_manager);
        checker.load_builtins()?;
        let type_table = checker.check_program(&self.ast)?;
        // Populate schemes with function type schemes from type checking
        *schemes = checker.get_function_schemes();
        let warnings: Vec<_> = checker.warnings().to_vec();
        let span_table = build_span_table(&self.ast);

        Ok(TypeChecked {
            ast: self.ast,
            type_table,
            span_table,
            warnings,
        })
    }
}

/// Program has been type checked
pub struct TypeChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub span_table: SpanTable,
    pub warnings: Vec<type_checker::TypeWarning>,
}

impl TypeChecked {
    /// Print warnings to stderr (convenience method)
    pub fn print_warnings(&self) {
        let mut nc = NodeCounter::new();
        let mm = module_manager::ModuleManager::new(&mut nc);
        let checker = type_checker::TypeChecker::new(&mm);
        for warning in &self.warnings {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| checker.format_type(t)),
                warning.span()
            );
        }
    }

    /// Run alias checking analysis on the program
    pub fn alias_check(self) -> Result<AliasChecked> {
        let checker = alias_checker::AliasChecker::new(&self.type_table, &self.span_table);
        let alias_result = checker.check_program(&self.ast)?;

        Ok(AliasChecked {
            ast: self.ast,
            type_table: self.type_table,
            span_table: self.span_table,
            warnings: self.warnings,
            alias_result,
        })
    }
}

/// Program has been alias checked
pub struct AliasChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub span_table: SpanTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub alias_result: alias_checker::AliasCheckResult,
}

impl AliasChecked {
    /// Print warnings to stderr (convenience method)
    pub fn print_warnings(&self) {
        let mut nc = NodeCounter::new();
        let mm = module_manager::ModuleManager::new(&mut nc);
        let checker = type_checker::TypeChecker::new(&mm);
        for warning in &self.warnings {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| checker.format_type(t)),
                warning.span()
            );
        }
    }

    /// Check if alias checking found any errors
    pub fn has_alias_errors(&self) -> bool {
        self.alias_result.has_errors()
    }

    /// Print alias errors to stderr
    pub fn print_alias_errors(&self) {
        self.alias_result.print_errors();
    }

    /// Flatten AST to MIR (with defunctionalization and desugaring).
    /// Returns the flattened MIR and a BackEnd for subsequent passes.
    pub fn flatten(
        self,
        module_manager: &module_manager::ModuleManager,
        schemes: &HashMap<String, TypeScheme<TypeName>>,
    ) -> Result<(Flattened, BackEnd)> {
        let type_table = self.type_table;

        let builtins = impl_source::ImplSource::default().all_names();
        // Collect prelude function declarations for defun analysis
        let prelude_decls: Vec<_> = module_manager.get_prelude_function_declarations();
        let defun_analysis =
            defun_analysis::analyze_program_with_decls(&self.ast, &prelude_decls, &type_table, &builtins);
        let mut flattener =
            flattening::Flattener::new(type_table, builtins, defun_analysis, schemes.clone());
        let mut mir = flattener.flatten_program(&self.ast)?;

        // Flatten module function declarations so they're available in SPIR-V
        for (module_name, decl) in module_manager.get_module_function_declarations() {
            let qualified_name = format!("{}.{}", module_name, decl.name);
            let defs = flattener.flatten_module_decl(decl, &qualified_name)?;
            mir.defs.extend(defs);
        }

        // Flatten prelude function declarations (top-level, auto-imported)
        for decl in module_manager.get_prelude_function_declarations() {
            let defs = flattener.flatten_module_decl(decl, &decl.name)?;
            mir.defs.extend(defs);
        }

        let node_counter = flattener.into_node_counter();
        Ok((Flattened { mir }, BackEnd::new(node_counter)))
    }
}

// =============================================================================
// BackEnd stages (MIR-based)
// =============================================================================

/// AST has been flattened to MIR
pub struct Flattened {
    pub mir: mir::Program,
}

impl Flattened {
    /// Hoist duplicate materializations from if branches.
    pub fn hoist_materializations(self) -> MaterializationsHoisted {
        let mir = materialize_hoisting::hoist_materializations(self.mir);
        MaterializationsHoisted { mir }
    }
}

/// Duplicate materializations have been hoisted (stub)
pub struct MaterializationsHoisted {
    pub mir: mir::Program,
}

impl MaterializationsHoisted {
    /// Normalize MIR to A-normal form.
    pub fn normalize(self) -> Normalized {
        let mir = normalize::normalize_program(self.mir);
        Normalized { mir }
    }
}

/// MIR has been normalized to A-normal form (stub)
pub struct Normalized {
    pub mir: mir::Program,
}

impl Normalized {
    /// Monomorphize polymorphic functions.
    pub fn monomorphize(self) -> Result<Monomorphized> {
        let mir = monomorphization::monomorphize(self.mir)?;
        Ok(Monomorphized { mir })
    }
}

/// Program has been monomorphized (stub)
pub struct Monomorphized {
    pub mir: mir::Program,
}

impl Monomorphized {
    /// Skip constant folding/partial evaluation entirely.
    /// Just passes through to the next stage.
    pub fn skip_folding(self) -> Folded {
        Folded { mir: self.mir }
    }

    /// Partially evaluate the MIR using normalization-by-evaluation.
    /// Evaluates function calls and loops when their arguments/bounds
    /// are known at compile time.
    pub fn partial_eval(self) -> Result<Folded> {
        let mir = partial_eval::partial_eval(self.mir)?;
        Ok(Folded { mir })
    }
}

/// Constants have been folded (stub)
pub struct Folded {
    pub mir: mir::Program,
}

impl Folded {
    /// Filter out unreachable definitions and order them topologically.
    pub fn filter_reachable(self) -> Reachable {
        let mir = reachability::filter_reachable(self.mir);
        Reachable { mir }
    }
}

/// Unreachable code has been filtered out (stub)
pub struct Reachable {
    pub mir: mir::Program,
}

impl Reachable {
    /// Hoist loop-invariant bindings out of loops.
    pub fn lift_bindings(self) -> Lifted {
        let mir = binding_lifter::lift_bindings(self.mir);
        Lifted { mir }
    }
}

/// Bindings have been lifted (loop-invariant code motion)
pub struct Lifted {
    pub mir: mir::Program,
}

impl Lifted {
    /// Lower MIR to SPIR-V
    pub fn lower(self) -> Result<Lowered> {
        let inplace_info = alias_checker::analyze_inplace(&self.mir);
        let spirv = spirv::lowering::lower(&self.mir, &inplace_info)?;
        Ok(Lowered { mir: self.mir, spirv })
    }

    /// Lower MIR to GLSL
    pub fn lower_glsl(self) -> Result<LoweredGlsl> {
        let glsl_output = glsl::lowering::lower(&self.mir)?;
        Ok(LoweredGlsl {
            mir: self.mir,
            glsl: GlslOutput {
                vertex: glsl_output.vertex,
                fragment: glsl_output.fragment,
            },
        })
    }

    /// Lower MIR to Shadertoy-compatible GLSL (fragment shader only)
    pub fn lower_shadertoy(self) -> Result<String> {
        glsl::lowering::lower_shadertoy(&self.mir)
    }
}

/// Final stage - contains MIR and SPIR-V bytecode
pub struct Lowered {
    pub mir: mir::Program,
    pub spirv: Vec<u32>,
}

/// GLSL output containing vertex and fragment shader sources
pub struct GlslOutput {
    pub vertex: Option<String>,
    pub fragment: Option<String>,
}

/// Final stage for GLSL - contains MIR and GLSL source strings
pub struct LoweredGlsl {
    pub mir: mir::Program,
    pub glsl: GlslOutput,
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
    (module_manager::ModuleManager::from_prelude(prelude), node_counter)
}

/// Create a FrontEnd using the cached prelude (test-only)
#[cfg(test)]
pub fn cached_frontend() -> FrontEnd {
    let (prelude, node_counter) = get_prelude_cache();
    let module_manager = module_manager::ModuleManager::from_prelude(prelude);
    let context = Context::default();
    let intrinsics = intrinsics::IntrinsicSource::new(&mut Context::default());

    FrontEnd {
        node_counter,
        module_manager,
        context,
        type_table: HashMap::new(),
        intrinsics,
        schemes: HashMap::new(),
        module_schemes: HashMap::new(),
    }
}
