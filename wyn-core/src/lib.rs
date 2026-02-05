pub mod ast;
pub mod diags;
pub mod error;
pub mod impl_source;
pub mod intrinsics;
pub mod lexer;
pub mod mir;
pub mod module_manager;
pub mod name_resolution;
pub mod parser;
pub mod pattern;
pub mod scope;
pub mod types;
pub mod visitor;

// Re-export type_checker from its new location for backwards compatibility
pub use types::checker as type_checker;

pub mod alias_checker;
pub mod ast_const_fold;
pub mod desugar;
pub mod lowering_common;
pub mod tlc;

pub mod glsl;
pub mod parallelization;
pub mod resolve_placeholders;
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
mod integration_tests;
#[cfg(test)]
mod scope_tests;

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

/// Build the set of known definition names that should not be captured as free variables.
/// This includes intrinsics, user declarations, and prelude functions.
pub fn build_known_defs(
    ast: &ast::Program,
    module_manager: &module_manager::ModuleManager,
) -> std::collections::HashSet<String> {
    let mut known_defs = impl_source::ImplSource::default().all_names();

    // Also add polymorphic intrinsics from IntrinsicSource (sign, abs, magnitude, etc.)
    let mut ctx = polytype::Context::<ast::TypeName>::default();
    known_defs.extend(intrinsics::IntrinsicSource::new(&mut ctx).all_names());

    // Add top-level function names from user program
    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(d) => {
                known_defs.insert(d.name.clone());
            }
            ast::Declaration::Entry(e) => {
                known_defs.insert(e.name.clone());
            }
            ast::Declaration::Uniform(u) => {
                known_defs.insert(u.name.clone());
            }
            ast::Declaration::Storage(s) => {
                known_defs.insert(s.name.clone());
            }
            _ => {}
        }
    }

    // Add prelude function names
    for decl in module_manager.get_prelude_function_declarations() {
        known_defs.insert(decl.name.clone());
    }

    known_defs
}

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
//       -> .alias_check()                               -> AliasChecked
//
// TLC Pipeline (AST -> SSA):
//       -> .to_tlc()                                    -> TlcTransformed
//       -> .partial_eval() or .skip_partial_eval()      -> TlcTransformed (optimized)
//       -> .defunctionalize()                           -> TlcDefunctionalized
//       -> .monomorphize()                              -> TlcMonomorphized
//       -> .to_ssa()                                    -> SsaConverted
//
// BackEnd Pipeline (SSA -> output):
//       -> .parallelize_soacs()                         -> SsaParallelized
//       -> .filter_reachable()                          -> SsaReachable
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

    /// Create a FrontEnd from a pre-elaborated prelude.
    /// This is faster than `new()` as it reuses an already-parsed prelude.
    pub fn new_from_prelude(
        prelude: module_manager::PreElaboratedPrelude,
        node_counter: NodeCounter,
    ) -> Self {
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
    /// Elaborate inline module declarations from the parsed program.
    /// This registers modules with the module_manager so they're available during resolution,
    /// then removes the Module declarations from the AST (they've been copied to module_manager).
    /// Should be called before desugar() if the program contains module definitions.
    pub fn elaborate_modules(mut self, module_manager: &mut module_manager::ModuleManager) -> Result<Self> {
        module_manager.elaborate_modules(&self.ast)?;
        // Remove Module and ModuleTypeBind declarations - they've been elaborated
        self.ast.declarations.retain(|decl| {
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
        mut self,
        module_manager: &mut module_manager::ModuleManager,
        schemes: &mut HashMap<String, TypeScheme<TypeName>>,
    ) -> Result<TypeChecked> {
        // Resolve type placeholders to type variables before type checking
        let mut resolver = resolve_placeholders::PlaceholderResolver::new();
        resolver.resolve(module_manager, &mut self.ast);
        let (context, spec_schemes) = resolver.into_parts();

        let mut checker =
            type_checker::TypeChecker::with_context_and_schemes(module_manager, context, spec_schemes);
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

    /// Transform AST to TLC (new pipeline path)
    /// `known_defs` contains names that should not be captured as free variables during lambda lifting
    /// `schemes` contains type schemes for all functions (populated during type_check)
    /// `module_manager` provides access to prelude declarations for TLC transformation
    pub fn to_tlc(
        self,
        known_defs: std::collections::HashSet<String>,
        schemes: &HashMap<String, types::TypeScheme>,
        module_manager: &module_manager::ModuleManager,
    ) -> TlcTransformed {
        // Create shared tables for all transformations
        let mut symbols = SymbolTable::new();
        let mut top_level_symbols = std::collections::HashMap::new();

        // Transform prelude to TLC using the same type_table (consistent type variables)
        let mut prelude_tlc_defs = Vec::new();

        // Transform module declarations (f32.pi, rand.init, etc.)
        for (module_name, elaborated) in module_manager.get_elaborated_modules() {
            let mut transformer = tlc::Transformer::with_namespace(
                &self.type_table,
                &mut symbols,
                &mut top_level_symbols,
                module_name,
            );
            for item in &elaborated.items {
                if let module_manager::ElaboratedItem::Decl(decl) = item {
                    // Transform all def declarations - wrapper functions that call intrinsics
                    // (like `zip` calling `_w_intrinsic_zip`) need to be in TLC so they can
                    // be called by other prelude functions (like `zip3`).
                    if let Some(def) = transformer.transform_decl(decl) {
                        prelude_tlc_defs.push(def);
                    }
                }
            }
        }

        // Transform top-level prelude functions (zip3, map2, etc.)
        {
            let mut transformer =
                tlc::Transformer::new(&self.type_table, &mut symbols, &mut top_level_symbols);
            for decl in module_manager.get_prelude_function_declarations() {
                // Transform all prelude functions - even those that wrap intrinsics
                if let Some(def) = transformer.transform_decl(decl) {
                    prelude_tlc_defs.push(def);
                }
            }
        }

        // Transform user program to TLC using the shared symbol table
        let mut transformer = tlc::Transformer::new(&self.type_table, &mut symbols, &mut top_level_symbols);
        let mut parts = transformer.transform_program(&self.ast);

        // Prepend prelude TLC defs
        let mut merged_defs = prelude_tlc_defs;
        merged_defs.extend(parts.defs);
        parts.defs = merged_defs;

        // Combine parts with the symbol table to create the final Program
        let tlc_program = parts.with_symbols(symbols);

        TlcTransformed {
            tlc: tlc_program,
            type_table: self.type_table,
            known_defs,
            schemes: schemes.clone(),
        }
    }
}

// =============================================================================
// TLC-based pipeline stages
// =============================================================================

/// AST has been transformed to TLC
pub struct TlcTransformed {
    pub tlc: tlc::Program,
    pub type_table: TypeTable,
    /// Built-in names that should not be captured as free variables
    known_defs: std::collections::HashSet<String>,
    /// Type schemes for functions (for monomorphization)
    schemes: HashMap<String, types::TypeScheme>,
}

impl TlcTransformed {
    /// Apply partial evaluation (constant folding, algebraic simplifications, etc.)
    pub fn partial_eval(self) -> TlcTransformed {
        let optimized = tlc::partial_eval::PartialEvaluator::partial_eval(self.tlc);
        TlcTransformed {
            tlc: optimized,
            type_table: self.type_table,
            known_defs: self.known_defs,
            schemes: self.schemes,
        }
    }

    /// Skip partial evaluation
    pub fn skip_partial_eval(self) -> TlcTransformed {
        self
    }

    /// Defunctionalize: lift lambdas and flatten SOAC closure captures.
    pub fn defunctionalize(self) -> TlcDefunctionalized {
        let defunc = tlc::defunctionalize::defunctionalize(self.tlc, &self.known_defs);
        TlcDefunctionalized {
            tlc: defunc,
            type_table: self.type_table,
            schemes: self.schemes,
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
        let specialized = tlc::specialize::specialize(self.tlc);

        // Convert string-keyed schemes to SymbolId-keyed for monomorphization
        let name_to_sym: HashMap<&str, SymbolId> =
            specialized.symbols.iter().map(|(&id, name)| (name.as_str(), id)).collect();
        let schemes_by_sym: HashMap<SymbolId, types::TypeScheme> = self
            .schemes
            .iter()
            .filter_map(|(name, scheme)| name_to_sym.get(name.as_str()).map(|&sym| (sym, scheme.clone())))
            .collect();

        // Monomorphize polymorphic user functions
        let monomorphized = tlc::monomorphize::monomorphize(specialized, &schemes_by_sym);
        TlcMonomorphized {
            tlc: monomorphized,
            type_table: self.type_table,
        }
    }
}

/// TLC with all functions monomorphized (no type variables remain)
pub struct TlcMonomorphized {
    pub tlc: tlc::Program,
    pub type_table: TypeTable,
}

impl TlcMonomorphized {
    /// Transform TLC directly to SSA.
    pub fn to_ssa(self) -> std::result::Result<SsaConverted, tlc::to_ssa::ConvertError> {
        let ssa = tlc::to_ssa::convert_program(&self.tlc)?;
        Ok(SsaConverted { ssa })
    }
}

/// TLC has been converted directly to SSA
pub struct SsaConverted {
    pub ssa: tlc::to_ssa::SsaProgram,
}

impl SsaConverted {
    /// Parallelize SOACs in compute shaders.
    pub fn parallelize_soacs(self) -> SsaParallelized {
        let ssa = parallelization::parallelize_soacs(self.ssa);
        SsaParallelized { ssa }
    }
}

/// SSA with parallelized SOACs for compute shaders
pub struct SsaParallelized {
    pub ssa: tlc::to_ssa::SsaProgram,
}

impl SsaParallelized {
    /// Eliminate dead functions (not reachable from any entry point).
    pub fn filter_reachable(self) -> SsaReachable {
        let ssa = mir::reachability::eliminate_dead_functions(self.ssa);
        SsaReachable { ssa }
    }
}

/// SSA with dead functions eliminated
pub struct SsaReachable {
    pub ssa: tlc::to_ssa::SsaProgram,
}

impl SsaReachable {
    /// Lower SSA to SPIR-V.
    pub fn lower(self) -> error::Result<Lowered> {
        let spirv = spirv::lower_ssa_program(&self.ssa)?;
        Ok(Lowered { spirv })
    }

    /// Lower SSA to GLSL.
    pub fn lower_glsl(self) -> error::Result<glsl::GlslOutput> {
        glsl::lower(&self.ssa)
    }

    /// Lower SSA to Shadertoy-compatible GLSL.
    pub fn lower_shadertoy(self) -> error::Result<String> {
        glsl::lower_shadertoy(&self.ssa)
    }
}

/// Final SPIR-V output
pub struct Lowered {
    pub spirv: Vec<u32>,
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
