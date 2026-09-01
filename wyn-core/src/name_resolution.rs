//! Name resolution pass
//!
//! Resolves module-qualified names by rewriting:
//!   `FieldAccess(Identifier(module), field)` → `Identifier([module], field)`
//! when `module` is a known module name.
//!
//! The same AST-walking machinery is reused by the module-elaboration path
//! (`module_manager::ModuleManager::resolve_names_in_expr`) — the walker is
//! generic over a `ResolveContext` that decides what a given identifier /
//! field-access means in the current mode. See `Resolver` below.
//!
//! `build_name_resolution` then populates a
//! `NameResolution` side table that classifies each `Identifier` NodeId as
//! `Builtin(BuiltinId)` (single catalog match) or
//! `OverloadedBuiltin(Vec<BuiltinId>)` (multiple catalog matches under
//! the same surface name). Identifiers not in the catalog (locals,
//! top-level defs, module values) are absent from the side table — the
//! type checker handles them via scope/module lookup.

use crate::ast;
use crate::elaborate_modules;
use crate::error;
use crate::module_manager;
use crate::resolve_opens;
use crate::resolve_placeholders;
use crate::LookupMap;
use crate::LookupSet;
use crate::{SymbolId, SymbolTable};

use crate::ast::{Declaration, ExprKind, Expression, NodeId, Program};
use crate::builtins::{BuiltinCatalog, BuiltinId};
use crate::module_manager::ModuleManager;
use crate::scope::{for_each_pattern_name, ScopeStack};
use wyn_module_graph::PackageId;

/// AST after module-qualified value names have been resolved.
#[derive(Debug, Clone, Copy)]
pub enum NamesResolvedTag {}
pub type NamesResolved =
    Program<NamesResolvedTag, elaborate_modules::ModulesElaboratedFamily, ModuleManager>;

/// Insert every name bound by `pattern` into `scope`.
fn collect_pattern_bindings<T, A>(pattern: &ast::Pattern<T, A>, scope: &mut ScopeStack<()>)
where
    T: ast::TreeFamily,
    A: Clone + std::fmt::Debug + PartialEq,
{
    for_each_pattern_name(pattern, &mut |name| {
        scope.insert(name.to_string(), ());
    });
}

// ---------------------------------------------------------------------------
// Visitor — shared AST traversal, mode-specific identifier / field-access
// rewrites plugged in via `ResolveContext`.
// ---------------------------------------------------------------------------

/// Policy interface for a resolver pass. Consumers implement it to inject
/// their mode-specific identifier- and field-access- rewrite logic. The
/// walker handles everything else (recursion, scope push/pop, pattern
/// binding collection).
pub trait ResolveContext {
    /// Called for each `ExprKind::Identifier(quals, name)` leaf. May mutate
    /// `quals` / `name` in place (e.g. to qualify an intra-module ref). The
    /// `scope` argument reflects locals visible at this expression — used
    /// so intra-module rewrites don't shadow a lambda param of the same
    /// name.
    fn resolve_identifier(&self, _quals: &mut Vec<String>, _name: &mut String, _scope: &ScopeStack<()>) {}

    /// Called for each `ExprKind::FieldAccess(obj, field)` when `obj` is an
    /// `Identifier`. For a chained access, it may be called again after the
    /// inner access has been resolved to an `Identifier`. Return `Some(ExprKind)`
    /// to replace the entire FieldAccess expression (typical case:
    /// `mod.name` collapses to `Identifier([mod], name)`); return `None`
    /// to leave the FieldAccess alone — the walker will then recurse into
    /// `obj` as a regular expression.
    fn resolve_field_access(
        &self,
        _obj_quals: &[String],
        _obj_name: &str,
        _field: &str,
        _scope: &ScopeStack<()>,
    ) -> Option<ExprKind> {
        None
    }
}

/// Walk an expression in place using this pass's established scope-sensitive
/// resolution rules.
pub fn walk_expr<C: ResolveContext>(
    expression: &mut Expression,
    context: &C,
    scope: &mut ScopeStack<()>,
) -> error::Result<()> {
    match &mut expression.kind {
        ExprKind::Identifier(identifier) => {
            context.resolve_identifier(&mut identifier.qualifiers, &mut identifier.name, scope);
        }
        ExprKind::FieldAccess(object, field) => {
            let object_was_identifier = matches!(object.kind, ExprKind::Identifier(_));
            let mut replacement = if let ExprKind::Identifier(identifier) = &object.kind {
                context.resolve_field_access(&identifier.qualifiers, &identifier.name, field, scope)
            } else {
                None
            };
            if replacement.is_none() {
                walk_expr(object, context, scope)?;
                if !object_was_identifier {
                    if let ExprKind::Identifier(identifier) = &object.kind {
                        replacement = context.resolve_field_access(
                            &identifier.qualifiers,
                            &identifier.name,
                            field,
                            scope,
                        );
                    }
                }
            }
            if let Some(kind) = replacement {
                expression.kind = kind;
            }
        }
        ExprKind::Application(function, arguments) => {
            walk_expr(function, context, scope)?;
            for argument in arguments {
                walk_expr(argument, context, scope)?;
            }
        }
        ExprKind::Lambda(lambda) => {
            scope.push_scope();
            for pattern in &lambda.params {
                collect_pattern_bindings(pattern, scope);
            }
            walk_expr(&mut lambda.body, context, scope)?;
            scope.pop_scope();
        }
        ExprKind::LetIn(let_in) => {
            walk_expr(&mut let_in.value, context, scope)?;
            scope.push_scope();
            collect_pattern_bindings(&let_in.pattern, scope);
            walk_expr(&mut let_in.body, context, scope)?;
            scope.pop_scope();
        }
        ExprKind::If(if_expr) => {
            walk_expr(&mut if_expr.condition, context, scope)?;
            walk_expr(&mut if_expr.then_branch, context, scope)?;
            walk_expr(&mut if_expr.else_branch, context, scope)?;
        }
        ExprKind::BinaryOp(_, left, right) => {
            walk_expr(left, context, scope)?;
            walk_expr(right, context, scope)?;
        }
        ExprKind::UnaryOp(_, value) => walk_expr(value, context, scope)?,
        ExprKind::Tuple(values) | ExprKind::ArrayLiteral(values) | ExprKind::VecMatLiteral(values) => {
            for value in values {
                walk_expr(value, context, scope)?;
            }
        }
        ExprKind::ArrayIndex(array, index) => {
            walk_expr(array, context, scope)?;
            walk_expr(index, context, scope)?;
        }
        ExprKind::ArrayWith { array, index, value } => {
            walk_expr(array, context, scope)?;
            walk_expr(index, context, scope)?;
            walk_expr(value, context, scope)?;
        }
        ExprKind::VecWith { target, value, .. } => {
            walk_expr(target, context, scope)?;
            walk_expr(value, context, scope)?;
        }
        ExprKind::RecordWith { record, value, .. } => {
            walk_expr(record, context, scope)?;
            walk_expr(value, context, scope)?;
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, value) in fields {
                walk_expr(value, context, scope)?;
            }
        }
        ExprKind::Loop(loop_expr) => {
            // The initializer and iteration domain are evaluated outside the
            // loop scope. This matters when a loop binder shadows an outer
            // value, as in `loop x = x ...`.
            if let Some(init) = &mut loop_expr.init {
                walk_expr(init, context, scope)?;
            }
            match &mut loop_expr.form {
                ast::LoopForm::For(_, bound) => walk_expr(bound, context, scope)?,
                ast::LoopForm::ForIn(_, iterable) => walk_expr(iterable, context, scope)?,
                ast::LoopForm::While(_) => {}
            }

            scope.push_scope();
            collect_pattern_bindings(&loop_expr.pattern, scope);
            match &mut loop_expr.form {
                ast::LoopForm::While(condition) => walk_expr(condition, context, scope)?,
                ast::LoopForm::For(pattern, _) | ast::LoopForm::ForIn(pattern, _) => {
                    collect_pattern_bindings(pattern, scope);
                }
            }
            walk_expr(&mut loop_expr.body, context, scope)?;
            scope.pop_scope();
        }
        ExprKind::Match(match_expr) => {
            walk_expr(&mut match_expr.scrutinee, context, scope)?;
            for case in &mut match_expr.cases {
                scope.push_scope();
                collect_pattern_bindings(&case.pattern, scope);
                walk_expr(&mut case.body, context, scope)?;
                scope.pop_scope();
            }
        }
        ExprKind::TypeAscription(value, _) | ExprKind::TypeCoercion(value, _) => {
            walk_expr(value, context, scope)?;
        }
        ExprKind::Range(range) => {
            walk_expr(&mut range.start, context, scope)?;
            walk_expr(&mut range.end, context, scope)?;
            if let Some(step) = &mut range.step {
                walk_expr(step, context, scope)?;
            }
        }
        ExprKind::Slice(slice) => {
            walk_expr(&mut slice.array, context, scope)?;
            if let Some(start) = &mut slice.start {
                walk_expr(start, context, scope)?;
            }
            if let Some(end) = &mut slice.end {
                walk_expr(end, context, scope)?;
            }
        }
        ExprKind::Constructor(_, arguments) => {
            for argument in arguments {
                walk_expr(argument, context, scope)?;
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole(_) => {}
    }
    Ok(())
}

pub fn rewrite_expr<C: ResolveContext>(
    mut expression: Expression,
    context: &C,
    mut scope: ScopeStack<()>,
) -> Expression {
    walk_expr(&mut expression, context, &mut scope).expect("name resolution visitor is infallible");
    expression
}

// ---------------------------------------------------------------------------
// Program-mode resolver (top-level pass over user code)
// ---------------------------------------------------------------------------

/// Context used by the program-level name-resolution pass: rewrites
/// `mod.name` to `Identifier([mod], name)` when `mod` is a registered
/// module.
struct ProgramResolver<'a> {
    known_modules: &'a LookupSet<String>,
}

impl<'a> ResolveContext for ProgramResolver<'a> {
    fn resolve_field_access(
        &self,
        obj_quals: &[String],
        obj_name: &str,
        field: &str,
        _scope: &ScopeStack<()>,
    ) -> Option<ExprKind> {
        let module_name = if obj_quals.is_empty() {
            obj_name.to_string()
        } else {
            format!("{}.{}", obj_quals.join("."), obj_name)
        };
        if self.known_modules.contains(&module_name) {
            Some(ExprKind::Identifier(ast::Identifier {
                qualifiers: module_name.split('.').map(str::to_string).collect(),
                name: field.to_string(),
            }))
        } else {
            None
        }
    }
}

/// Resolve qualified field accesses while consuming the old program stage.
pub fn resolve_names(mut program: elaborate_modules::ModulesElaborated) -> NamesResolved {
    {
        let context = ProgramResolver {
            known_modules: program.global_context.known_module_names(),
        };
        for declaration in &mut program.declarations {
            let mut scope = ScopeStack::new();
            match declaration {
                Declaration::Decl(decl) => {
                    walk_expr(&mut decl.body, &context, &mut scope)
                        .expect("name resolution visitor is infallible");
                }
                Declaration::Entry(entry) => {
                    walk_expr(&mut entry.body, &context, &mut scope)
                        .expect("name resolution visitor is infallible");
                }
                Declaration::Extern(_) | Declaration::Frontend(_) => {}
            }
        }
    }
    program.retag()
}

/// Resolve names in a single Decl (for prelude functions).
pub fn resolve_decl(mut decl: ast::Decl, known_modules: &LookupSet<String>) -> ast::Decl {
    let context = ProgramResolver { known_modules };
    let mut scope = ScopeStack::new();
    walk_expr(&mut decl.body, &context, &mut scope).expect("name resolution visitor is infallible");
    decl
}

// ---------------------------------------------------------------------------
// Side-table builder: classify each Identifier NodeId against the catalog
// ---------------------------------------------------------------------------

/// Which second-order array combinator a bare identifier denotes. SOACs
/// are not catalog surface names (and `filter`/`zip`/`reduce_by_index`
/// are not catalog builtins at all), so they carry their own structural
/// tag rather than a `BuiltinId`. Recorded by the resolver, so a user
/// `def map` — top-level or local — that shadows the builtin is never
/// mistaken for the SOAC by a downstream string match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoacKind {
    Replicate,
    Map,
    Reduce,
    Scan,
    Filter,
    Zip,
    ReduceByIndex,
    Scatter,
    BucketScatter(u8),
}

impl SoacKind {
    /// The SOAC denoted by an unqualified identifier, if it names one.
    pub fn from_name(name: &str) -> Option<SoacKind> {
        Some(match name {
            "replicate" => SoacKind::Replicate,
            "map" => SoacKind::Map,
            "reduce" => SoacKind::Reduce,
            "scan" => SoacKind::Scan,
            "filter" => SoacKind::Filter,
            "zip" | "zip2" | "zip3" | "zip4" | "zip5" => SoacKind::Zip,
            "reduce_by_index" => SoacKind::ReduceByIndex,
            "scatter" => SoacKind::Scatter,
            "bucket_scatter_1d" => SoacKind::BucketScatter(1),
            "bucket_scatter_2d" => SoacKind::BucketScatter(2),
            "bucket_scatter_3d" => SoacKind::BucketScatter(3),
            "bucket_scatter_4d" => SoacKind::BucketScatter(4),
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedValueRef {
    /// A lexical, top-level, module, or prelude value binding.
    Symbol(SymbolId),
    /// Catalog entry matching this identifier's surface name.
    /// `overload_idx` is the index into `BuiltinDef::overloads()` chosen
    /// by the type checker after resolving the call against actual
    /// argument types. Set at classification time for single-overload
    /// entries (`Some(0)`); `None` for multi-overload entries until the
    /// type checker resolves the call site (`resolve_overload`) and
    /// writes back `Some(idx)`. Consumers downstream of type-checking
    /// (TLC → backends) unwrap it; `None` at that stage is a bug.
    Builtin {
        id: BuiltinId,
        overload_idx: Option<usize>,
    },
    /// Constructor-style vec conversion (`vec2i32(v)`, `vec3f32(u)`, …).
    /// The type checker recognises the call shape and records the
    /// target name + arity + per-component target type. `to_tlc` desugars
    /// the call to a `VecLit` of componentwise scalar conversions:
    /// `vec2i32(v)` → `@[i32.(elem_of_v)(v.x), i32.(elem_of_v)(v.y)]`.
    /// The source component type comes from the typed arg at desugar
    /// time.
    VecConstructor {
        /// Arity: 2, 3, or 4.
        arity: usize,
        /// Structurally resolved per-component target type. Conversion
        /// dispatch never reconstructs a catalog name from this value.
        target_elem: ast::TypeName,
        /// Filled by type checking once the source component type is known.
        /// Materialization rejects `None`.
        component_conversion: Option<BuiltinId>,
    },
    /// A second-order array combinator (`map`/`reduce`/…) named by a
    /// bare, unshadowed identifier. Recorded only when the name resolves
    /// to the builtin, so TLC lowers a call as a SOAC exactly when the
    /// frontend resolved one — never when a same-named user `def`
    /// shadows it.
    Soac(SoacKind),
}

/// Program-wide context after all value binders and references have stable
/// identities. `source` retains module/resource/type environments; `symbols`
/// is the sole allocator and diagnostic-name table for source-level bindings.
#[derive(Debug)]
pub struct BindingsResolvedGlobal {
    pub source: resolve_placeholders::PlaceholdersResolvedGlobal,
    pub symbols: SymbolTable,
    pub support_definitions: Vec<ast::SupportDefinition<ast::NameResolvedDefinition, ast::ResolvedTree>>,
}

/// Side table populated by `build_name_resolution`. Maps Identifier
/// NodeIds to their catalog classification. Identifiers not in the
/// catalog (locals, top-level defs, module values) are absent.
#[derive(Debug, Clone)]
pub struct NameResolution {
    pub values: LookupMap<NodeId, ResolvedValueRef>,
    /// Sole source-level identity arena. Names are diagnostic metadata.
    pub symbols: SymbolTable,
    /// Transient resolution-boundary lookup consumed while rebuilding
    /// declarations into the ID-bearing typed AST.
    pub definitions: LookupMap<String, SymbolId>,
    /// Pattern-node/source-spelling to binder identity. This table is consumed
    /// when resolved bindings are materialized in-tree.
    pub bindings: LookupMap<(NodeId, String), SymbolId>,
    /// Definition-node identity, kept separate from lexical name lookup so a
    /// shadowed prelude definition never shares a `SymbolId` with user code.
    declarations: LookupMap<(String, ast::Span), ResolvedDeclaration>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedDeclaration {
    pub(crate) symbol: SymbolId,
    pub(crate) package: Option<PackageId>,
}

impl NameResolution {
    fn new() -> Self {
        Self {
            values: LookupMap::new(),
            symbols: SymbolTable::new(),
            definitions: LookupMap::new(),
            bindings: LookupMap::new(),
            declarations: LookupMap::new(),
        }
    }
    pub fn get(&self, id: NodeId) -> Option<&ResolvedValueRef> {
        self.values.get(&id)
    }

    pub(crate) fn take_declaration(&mut self, name: &str, span: ast::Span) -> Option<ResolvedDeclaration> {
        self.declarations.remove(&(name.to_owned(), span))
    }

    /// Record the type checker's choice of overload index for a Builtin
    /// resolution. No-op for entries not in `values`. Panics if the
    /// resolved entry isn't `Builtin` (the only variant with overloads).
    pub fn set_overload_idx(&mut self, id: NodeId, idx: usize) {
        if let Some(entry) = self.values.get_mut(&id) {
            match entry {
                ResolvedValueRef::Builtin { overload_idx, .. } => {
                    *overload_idx = Some(idx);
                }
                ResolvedValueRef::VecConstructor { .. } => {
                    // Vec constructors don't carry a catalog overload index.
                }
                ResolvedValueRef::Symbol(_) | ResolvedValueRef::Soac(_) => {
                    // SOACs aren't overloaded catalog entries. No-op.
                }
            }
        }
    }
}

/// Walk the program after module-qualification rewrite and build the
/// side table. Top-level def names are pushed into scope first, so
/// user code that shadows a builtin (e.g. `def length = ...`) is
/// classified as a non-builtin.
///
/// Walks all elaborated modules, including prelude modules. This is
/// safe because functor instantiation freshens NodeIds (via
/// `clone_expr_fresh_ids` / `clone_pattern_fresh_ids` in
/// `module_manager::elaborate_decl_signature`), so per-instance bodies
/// have their own NodeId space and the previous collision risk is gone.
pub fn build_name_resolution(
    program: &resolve_opens::OpensResolved,
    module_manager: &ModuleManager,
    catalog: &BuiltinCatalog,
) -> NameResolution {
    let catalog = EnabledBuiltinCatalog {
        catalog,
        graphics: module_manager.options().graphics,
    };
    let mut nr = NameResolution::new();
    let mut top_level: ScopeStack<()> = ScopeStack::new();
    collect_top_level_names(&program.declarations, &mut top_level);

    walk_decls(&program.declarations, &top_level, &catalog, &mut nr);

    // Collect which elaborated modules were declared in the user
    // `Program` (as opposed to prelude / system modules). User-
    // defined module bodies close over the enclosing file scope —
    // their `module_scope` seeds in the user file-scope names BEFORE
    // module siblings, so a bare reference inside a user module body
    // shadows a SOAC iff the user has a file-scope `def` of the same
    // name. Prelude modules stay invariant: their `module_scope`
    // contains only their own siblings.
    // `module_manager.user_module_names` is populated at elaboration
    // time with every module the user declared (as opposed to prelude
    // / system modules). Use it directly — by the time we reach name
    // resolution, the user's `module m = …` declaration has already
    // been moved out of `program.declarations` into
    // `elaborated_modules`, so we can't grep the source AST for it.
    let user_module_names = &module_manager.user_module_names;

    // Walk every elaborated module body — both user-source and
    // prelude. Functor instantiations now produce per-instance fresh
    // NodeIds (via `clone_expr_fresh_ids` in
    // `module_manager::elaborate_decl_signature`), so the previous
    // collision risk is gone and prelude bodies can safely be
    // covered by NameResolution.
    for (mod_name, elaborated) in module_manager.elaborated_modules.iter() {
        let mut module_scope: ScopeStack<()> = ScopeStack::new();
        if user_module_names.contains(mod_name) {
            collect_top_level_names(&program.declarations, &mut module_scope);
        }
        for item in &elaborated.items {
            if let module_manager::ElaboratedItem::Decl(d) = item {
                module_scope.insert(d.name.clone(), ());
            }
        }
        for item in &elaborated.items {
            if let module_manager::ElaboratedItem::Decl(d) = item {
                let mut scope = module_scope.clone();
                scope.push_scope();
                for p in &d.params {
                    collect_pattern_bindings(p, &mut scope);
                }
                walk_resolution(&d.body, &catalog, &mut scope, &mut nr);
                scope.pop_scope();
            }
        }
    }

    // Top-level prelude functions (`reverse`, `rotate`, `iota`, etc.) —
    // these live outside any module and aren't reached via
    // `elaborated_modules`, but their bodies reference catalog builtins
    // like `length` that must classify as `Builtin` to satisfy
    // `var_term_builtin_id`'s no-string-lookup invariant.
    let prelude_decls = module_manager.get_prelude_function_declarations();
    let mut prelude_scope: ScopeStack<()> = ScopeStack::new();
    for d in &prelude_decls {
        prelude_scope.insert(d.name.clone(), ());
    }
    for d in &prelude_decls {
        let mut scope = prelude_scope.clone();
        scope.push_scope();
        for p in &d.params {
            collect_pattern_bindings(p, &mut scope);
        }
        walk_resolution(&d.body, &catalog, &mut scope, &mut nr);
        scope.pop_scope();
    }

    assign_symbol_identities(program, module_manager, &mut nr);
    nr
}

struct EnabledBuiltinCatalog<'a> {
    catalog: &'a BuiltinCatalog,
    graphics: bool,
}

impl EnabledBuiltinCatalog<'_> {
    fn lookup_by_surface_name(&self, name: &str) -> Option<&crate::builtins::BuiltinDef> {
        self.catalog
            .lookup_by_surface_name(name)
            .filter(|def| self.graphics || !crate::builtins::is_graphics_builtin(def.id))
    }
}

fn intern_definition(nr: &mut NameResolution, name: String) -> SymbolId {
    if let Some(symbol) = nr.definitions.get(&name) {
        return *symbol;
    }
    let symbol = nr.symbols.alloc(name.clone());
    nr.definitions.insert(name, symbol);
    symbol
}

fn alloc_declaration(
    nr: &mut NameResolution,
    name: String,
    span: ast::Span,
    package: Option<PackageId>,
) -> SymbolId {
    let symbol = nr.symbols.alloc(name.clone());
    nr.declarations.insert((name, span), ResolvedDeclaration { symbol, package });
    symbol
}

fn source_package(program: &resolve_opens::OpensResolved, span: ast::Span) -> Option<PackageId> {
    span.module().and_then(|module| program.source_graph().package_of(module))
}

fn bind_symbol_pattern<T, A>(
    pattern: &ast::Pattern<T, A>,
    scope: &mut ScopeStack<SymbolId>,
    nr: &mut NameResolution,
) where
    T: ast::TreeFamily<
        Header = ast::Header,
        Identifier = ast::Identifier,
        Binding = String,
        TypeHole = ast::TypeHole,
    >,
    A: Clone + std::fmt::Debug + PartialEq,
{
    use crate::ast::{PatternKind, RecordPatternTarget};

    match &pattern.kind {
        PatternKind::Name(name) => {
            let symbol = nr.symbols.alloc(name.clone());
            nr.bindings.insert((pattern.h.id, name.clone()), symbol);
            scope.insert(name.clone(), symbol);
        }
        PatternKind::Tuple(patterns)
        | PatternKind::Vec(patterns)
        | PatternKind::Constructor(_, patterns) => {
            for pattern in patterns {
                bind_symbol_pattern(pattern, scope, nr);
            }
        }
        PatternKind::Record(fields) => {
            for field in fields {
                match &field.target {
                    RecordPatternTarget::Pattern(pattern) => bind_symbol_pattern(pattern, scope, nr),
                    RecordPatternTarget::Shorthand(name) => {
                        let symbol = nr.symbols.alloc(name.clone());
                        nr.bindings.insert((pattern.h.id, name.clone()), symbol);
                        scope.insert(name.clone(), symbol);
                    }
                }
            }
        }
        PatternKind::Typed(pattern, _) | PatternKind::Attributed(_, pattern) => {
            bind_symbol_pattern(pattern, scope, nr);
        }
        PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Unit => {}
    }
}

fn assign_symbol_identities(
    program: &resolve_opens::OpensResolved,
    module_manager: &ModuleManager,
    nr: &mut NameResolution,
) {
    for (module, definition) in module_manager.get_all_module_declarations() {
        let name = format!("{}.{}", module, definition.name);
        let package = if module_manager.user_module_names.contains(module) {
            source_package(program, definition.name_span)
        } else {
            None
        };
        let symbol = alloc_declaration(nr, name.clone(), definition.name_span, package);
        nr.definitions.entry(name).or_insert(symbol);
    }
    let prelude = module_manager.get_prelude_function_declarations();
    for definition in &prelude {
        let symbol = alloc_declaration(nr, definition.name.clone(), definition.name_span, None);
        nr.definitions.entry(definition.name.clone()).or_insert(symbol);
    }
    for declaration in &program.declarations {
        match declaration {
            Declaration::Decl(definition) => {
                let package = source_package(program, definition.name_span);
                let symbol = alloc_declaration(nr, definition.name.clone(), definition.name_span, package);
                nr.definitions.insert(definition.name.clone(), symbol);
            }
            Declaration::Entry(entry) => {
                let package = source_package(program, entry.name_span);
                let symbol = alloc_declaration(nr, entry.name.clone(), entry.name_span, package);
                nr.definitions.insert(entry.name.clone(), symbol);
            }
            Declaration::Extern(external) => {
                let package = source_package(program, external.data.span);
                let symbol = alloc_declaration(nr, external.name.clone(), external.data.span, package);
                nr.definitions.insert(external.name.clone(), symbol);
            }
            Declaration::Frontend(_) => {}
        }
    }

    for declaration in &program.declarations {
        match declaration {
            Declaration::Decl(definition) => {
                let mut scope = ScopeStack::new();
                scope.push_scope();
                for pattern in &definition.params {
                    bind_symbol_pattern(pattern, &mut scope, nr);
                }
                assign_expr_symbols(&definition.body, &mut scope, nr);
            }
            Declaration::Entry(entry) => {
                let mut scope = ScopeStack::new();
                scope.push_scope();
                for pattern in &entry.params {
                    bind_symbol_pattern(pattern, &mut scope, nr);
                }
                assign_expr_symbols(&entry.body, &mut scope, nr);
            }
            Declaration::Extern(_) | Declaration::Frontend(_) => {}
        }
    }

    for (module, elaborated) in module_manager.elaborated_modules.iter() {
        let mut module_scope = ScopeStack::new();
        for item in &elaborated.items {
            if let module_manager::ElaboratedItem::Decl(definition) = item {
                let symbol = intern_definition(nr, format!("{}.{}", module, definition.name));
                module_scope.insert(definition.name.clone(), symbol);
            }
        }
        for item in &elaborated.items {
            if let module_manager::ElaboratedItem::Decl(definition) = item {
                let mut scope = module_scope.clone();
                scope.push_scope();
                for pattern in &definition.params {
                    bind_symbol_pattern(pattern, &mut scope, nr);
                }
                assign_expr_symbols(&definition.body, &mut scope, nr);
            }
        }
    }

    let mut prelude_scope = ScopeStack::new();
    for definition in &prelude {
        let symbol = intern_definition(nr, definition.name.clone());
        prelude_scope.insert(definition.name.clone(), symbol);
    }
    for definition in &prelude {
        let mut scope = prelude_scope.clone();
        scope.push_scope();
        for pattern in &definition.params {
            bind_symbol_pattern(pattern, &mut scope, nr);
        }
        assign_expr_symbols(&definition.body, &mut scope, nr);
    }
}

fn assign_expr_symbols<T>(
    expression: &Expression<T>,
    scope: &mut ScopeStack<SymbolId>,
    nr: &mut NameResolution,
) where
    T: ast::TreeFamily<
        Header = ast::Header,
        Identifier = ast::Identifier,
        Binding = String,
        TypeHole = ast::TypeHole,
    >,
{
    use crate::ast::LoopForm;

    match &expression.kind {
        ExprKind::Identifier(identifier) => {
            if nr.values.contains_key(&expression.h.id) {
                return;
            }
            let full_name = if identifier.qualifiers.is_empty() {
                identifier.name.clone()
            } else {
                format!("{}.{}", identifier.qualifiers.join("."), identifier.name)
            };
            let symbol = if identifier.qualifiers.is_empty() {
                scope.lookup(&identifier.name).copied().or_else(|| nr.definitions.get(&full_name).copied())
            } else {
                nr.definitions.get(&full_name).copied()
            }
            .unwrap_or_else(|| intern_definition(nr, full_name));
            nr.values.insert(expression.h.id, ResolvedValueRef::Symbol(symbol));
        }
        ExprKind::Application(function, arguments) => {
            assign_expr_symbols(function, scope, nr);
            for argument in arguments {
                assign_expr_symbols(argument, scope, nr);
            }
        }
        ExprKind::Lambda(lambda) => {
            scope.push_scope();
            for pattern in &lambda.params {
                bind_symbol_pattern(pattern, scope, nr);
            }
            assign_expr_symbols(&lambda.body, scope, nr);
            scope.pop_scope();
        }
        ExprKind::LetIn(let_in) => {
            assign_expr_symbols(&let_in.value, scope, nr);
            scope.push_scope();
            bind_symbol_pattern(&let_in.pattern, scope, nr);
            assign_expr_symbols(&let_in.body, scope, nr);
            scope.pop_scope();
        }
        ExprKind::If(if_expression) => {
            assign_expr_symbols(&if_expression.condition, scope, nr);
            assign_expr_symbols(&if_expression.then_branch, scope, nr);
            assign_expr_symbols(&if_expression.else_branch, scope, nr);
        }
        ExprKind::FieldAccess(value, _) => assign_expr_symbols(value, scope, nr),
        ExprKind::BinaryOp(_, left, right) => {
            assign_expr_symbols(left, scope, nr);
            assign_expr_symbols(right, scope, nr);
        }
        ExprKind::UnaryOp(_, operand) => assign_expr_symbols(operand, scope, nr),
        ExprKind::Tuple(values) | ExprKind::ArrayLiteral(values) | ExprKind::VecMatLiteral(values) => {
            for value in values {
                assign_expr_symbols(value, scope, nr);
            }
        }
        ExprKind::ArrayIndex(array, index) => {
            assign_expr_symbols(array, scope, nr);
            assign_expr_symbols(index, scope, nr);
        }
        ExprKind::ArrayWith { array, index, value } => {
            assign_expr_symbols(array, scope, nr);
            assign_expr_symbols(index, scope, nr);
            assign_expr_symbols(value, scope, nr);
        }
        ExprKind::VecWith { target, value, .. } => {
            assign_expr_symbols(target, scope, nr);
            assign_expr_symbols(value, scope, nr);
        }
        ExprKind::RecordWith { record, value, .. } => {
            assign_expr_symbols(record, scope, nr);
            assign_expr_symbols(value, scope, nr);
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, value) in fields {
                assign_expr_symbols(value, scope, nr);
            }
        }
        ExprKind::Loop(loop_expression) => {
            if let Some(init) = &loop_expression.init {
                assign_expr_symbols(init, scope, nr);
            }
            match &loop_expression.form {
                LoopForm::For(_, bound) => assign_expr_symbols(bound, scope, nr),
                LoopForm::ForIn(_, iterable) => assign_expr_symbols(iterable, scope, nr),
                LoopForm::While(_) => {}
            }

            scope.push_scope();
            bind_symbol_pattern(&loop_expression.pattern, scope, nr);
            match &loop_expression.form {
                LoopForm::While(condition) => assign_expr_symbols(condition, scope, nr),
                LoopForm::For(pattern, _) | LoopForm::ForIn(pattern, _) => {
                    bind_symbol_pattern(pattern, scope, nr);
                }
            }
            assign_expr_symbols(&loop_expression.body, scope, nr);
            scope.pop_scope();
        }
        ExprKind::Match(match_expression) => {
            assign_expr_symbols(&match_expression.scrutinee, scope, nr);
            for case in &match_expression.cases {
                scope.push_scope();
                bind_symbol_pattern(&case.pattern, scope, nr);
                assign_expr_symbols(&case.body, scope, nr);
                scope.pop_scope();
            }
        }
        ExprKind::TypeAscription(value, _) | ExprKind::TypeCoercion(value, _) => {
            assign_expr_symbols(value, scope, nr);
        }
        ExprKind::Range(range) => {
            assign_expr_symbols(&range.start, scope, nr);
            if let Some(step) = &range.step {
                assign_expr_symbols(step, scope, nr);
            }
            assign_expr_symbols(&range.end, scope, nr);
        }
        ExprKind::Slice(slice) => {
            assign_expr_symbols(&slice.array, scope, nr);
            if let Some(start) = &slice.start {
                assign_expr_symbols(start, scope, nr);
            }
            if let Some(end) = &slice.end {
                assign_expr_symbols(end, scope, nr);
            }
        }
        ExprKind::Constructor(_, arguments) => {
            for argument in arguments {
                assign_expr_symbols(argument, scope, nr);
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole(_) => {}
    }
}
/// Walk a list of declarations (program-level or module-body), classifying
/// every catalog reference in their bodies. `outer_scope` provides the
/// shadowing context (top-level user names, or surrounding module's
/// scope plus its sibling decls).
fn walk_decls(
    decls: &[Declaration<resolve_opens::OpensResolvedFamily>],
    outer_scope: &ScopeStack<()>,
    catalog: &EnabledBuiltinCatalog<'_>,
    nr: &mut NameResolution,
) {
    // Build a sibling scope that includes the names of all decls at this
    // level — needed so a module-local `def length = ...` shadows the
    // catalog within its sibling decls' bodies.
    let mut sibling_scope = outer_scope.clone();
    collect_top_level_names(decls, &mut sibling_scope);

    for decl in decls {
        match decl {
            Declaration::Decl(d) => {
                let mut scope = sibling_scope.clone();
                scope.push_scope();
                for p in &d.params {
                    collect_pattern_bindings(p, &mut scope);
                }
                walk_resolution(&d.body, catalog, &mut scope, nr);
                scope.pop_scope();
            }
            Declaration::Entry(entry) => {
                let mut scope = sibling_scope.clone();
                scope.push_scope();
                for p in &entry.params {
                    collect_pattern_bindings(p, &mut scope);
                }
                walk_resolution(&entry.body, catalog, &mut scope, nr);
                scope.pop_scope();
            }
            _ => {}
        }
    }
}

fn collect_top_level_names(
    decls: &[Declaration<resolve_opens::OpensResolvedFamily>],
    scope: &mut ScopeStack<()>,
) {
    for decl in decls {
        match decl {
            Declaration::Decl(d) => {
                scope.insert(d.name.clone(), ());
            }
            Declaration::Entry(entry) => {
                scope.insert(entry.name.clone(), ());
            }
            _ => {}
        }
    }
}

/// Walk an expression tree, recording catalog classifications for each
/// `ExprKind::Identifier` whose surface name is not lexically shadowed.
fn walk_resolution<T>(
    expr: &Expression<T>,
    catalog: &EnabledBuiltinCatalog<'_>,
    scope: &mut ScopeStack<()>,
    nr: &mut NameResolution,
) where
    T: ast::TreeFamily<Header = ast::Header, Identifier = ast::Identifier, TypeHole = ast::TypeHole>,
{
    match &expr.kind {
        ExprKind::Identifier(identifier) => {
            // Unqualified name shadowed by a local? Skip — let the
            // checker resolve via scope.
            if identifier.qualifiers.is_empty() && scope.lookup(&identifier.name).is_some() {
                return;
            }
            let full_name = if identifier.qualifiers.is_empty() {
                identifier.name.clone()
            } else {
                format!("{}.{}", identifier.qualifiers.join("."), identifier.name)
            };
            if let Some(def) = catalog.lookup_by_surface_name(&full_name) {
                let overload_idx = if def.overloads().len() == 1 { Some(0) } else { None };
                nr.values.insert(
                    expr.h.id,
                    ResolvedValueRef::Builtin {
                        id: def.id,
                        overload_idx,
                    },
                );
            } else if identifier.qualifiers.is_empty() {
                // SOACs (`map`/`reduce`/…) are not catalog surface names.
                // Record the structural tag here, after the shadowing
                // check above — so a user `def map` is never tagged.
                if let Some(kind) = SoacKind::from_name(&identifier.name) {
                    nr.values.insert(expr.h.id, ResolvedValueRef::Soac(kind));
                }
            }
        }
        ExprKind::Application(func, args) => {
            walk_resolution(func, catalog, scope, nr);
            for arg in args {
                walk_resolution(arg, catalog, scope, nr);
            }
        }
        ExprKind::Lambda(lambda) => {
            scope.push_scope();
            for p in &lambda.params {
                collect_pattern_bindings(p, scope);
            }
            walk_resolution(&lambda.body, catalog, scope, nr);
            scope.pop_scope();
        }
        ExprKind::LetIn(let_in) => {
            walk_resolution(&let_in.value, catalog, scope, nr);
            scope.push_scope();
            collect_pattern_bindings(&let_in.pattern, scope);
            walk_resolution(&let_in.body, catalog, scope, nr);
            scope.pop_scope();
        }
        ExprKind::If(if_expr) => {
            walk_resolution(&if_expr.condition, catalog, scope, nr);
            walk_resolution(&if_expr.then_branch, catalog, scope, nr);
            walk_resolution(&if_expr.else_branch, catalog, scope, nr);
        }
        ExprKind::BinaryOp(_, lhs, rhs) => {
            walk_resolution(lhs, catalog, scope, nr);
            walk_resolution(rhs, catalog, scope, nr);
        }
        ExprKind::UnaryOp(_, operand) => walk_resolution(operand, catalog, scope, nr),
        ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
            for e in exprs {
                walk_resolution(e, catalog, scope, nr);
            }
        }
        ExprKind::ArrayIndex(arr, idx) => {
            walk_resolution(arr, catalog, scope, nr);
            walk_resolution(idx, catalog, scope, nr);
        }
        ExprKind::ArrayWith {
            array, index, value, ..
        } => {
            walk_resolution(array, catalog, scope, nr);
            walk_resolution(index, catalog, scope, nr);
            walk_resolution(value, catalog, scope, nr);
        }
        ExprKind::VecWith { target, value, .. } => {
            walk_resolution(target, catalog, scope, nr);
            walk_resolution(value, catalog, scope, nr);
        }
        ExprKind::RecordWith { record, value, .. } => {
            walk_resolution(record, catalog, scope, nr);
            walk_resolution(value, catalog, scope, nr);
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, e) in fields {
                walk_resolution(e, catalog, scope, nr);
            }
        }
        ExprKind::Loop(loop_expr) => {
            use crate::ast::LoopForm;
            if let Some(init) = &loop_expr.init {
                walk_resolution(init, catalog, scope, nr);
            }
            match &loop_expr.form {
                LoopForm::For(_, bound) => walk_resolution(bound, catalog, scope, nr),
                LoopForm::ForIn(_, iter) => walk_resolution(iter, catalog, scope, nr),
                LoopForm::While(_) => {}
            }

            scope.push_scope();
            collect_pattern_bindings(&loop_expr.pattern, scope);
            match &loop_expr.form {
                LoopForm::While(cond) => walk_resolution(cond, catalog, scope, nr),
                LoopForm::For(pattern, _) | LoopForm::ForIn(pattern, _) => {
                    collect_pattern_bindings(pattern, scope);
                }
            }
            walk_resolution(&loop_expr.body, catalog, scope, nr);
            scope.pop_scope();
        }
        ExprKind::Match(match_expr) => {
            walk_resolution(&match_expr.scrutinee, catalog, scope, nr);
            for case in &match_expr.cases {
                scope.push_scope();
                collect_pattern_bindings(&case.pattern, scope);
                walk_resolution(&case.body, catalog, scope, nr);
                scope.pop_scope();
            }
        }
        ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
            walk_resolution(e, catalog, scope, nr);
        }
        ExprKind::FieldAccess(obj, _) => {
            walk_resolution(obj, catalog, scope, nr);
        }
        ExprKind::Range(range) => {
            walk_resolution(&range.start, catalog, scope, nr);
            if let Some(step) = &range.step {
                walk_resolution(step, catalog, scope, nr);
            }
            walk_resolution(&range.end, catalog, scope, nr);
        }
        ExprKind::Slice(slice) => {
            walk_resolution(&slice.array, catalog, scope, nr);
            if let Some(start) = &slice.start {
                walk_resolution(start, catalog, scope, nr);
            }
            if let Some(end) = &slice.end {
                walk_resolution(end, catalog, scope, nr);
            }
        }
        ExprKind::Constructor(_, args) => {
            for a in args {
                walk_resolution(a, catalog, scope, nr);
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole(_) => {}
    }
}
