//! Module manager for lazy loading and caching module definitions

use crate::ast::{
    Decl, Declaration, ModuleExpression, ModuleTypeExpression, Node, NodeCounter, Pattern, PatternKind,
    Program, Spec, Type,
};
use crate::error::Result;
use crate::lexer;
use crate::parser::Parser;
use crate::scope::ScopeStack;
use crate::{bail_module, err_module, err_parse};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Name resolver for tracking opened modules and resolving unqualified names
/// TODO: Integrate with elaboration to handle `open` declarations
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct NameResolver {
    /// Modules currently opened (via `open` declarations)
    opened_modules: Vec<String>,
    /// Local definitions in scope (for shadowing)
    local_scope: ScopeStack<()>,
}

#[allow(dead_code)]
impl NameResolver {
    fn new() -> Self {
        NameResolver {
            opened_modules: Vec::new(),
            local_scope: ScopeStack::new(),
        }
    }

    /// Resolve an unqualified name by checking opened modules
    /// Returns None if the name can't be resolved, or Some(qualified_name) if found
    fn resolve_name(&self, name: &str, module_manager: &ModuleManager) -> Option<String> {
        // 1. Check if it's locally defined (shadows everything)
        if self.local_scope.is_defined(name) {
            return None; // Keep unqualified, it's a local binding
        }

        // 2. Try each opened module in reverse order (most recent first)
        for module_name in self.opened_modules.iter().rev() {
            // Check if this module has this function
            if let Some(elaborated) = module_manager.elaborated_modules.get(module_name) {
                for item in &elaborated.items {
                    let item_name = match item {
                        ElaboratedItem::Spec(Spec::Sig(n, _, _)) => Some(n.as_str()),
                        ElaboratedItem::Spec(Spec::SigOp(op, _)) => Some(op.as_str()),
                        ElaboratedItem::Decl(decl) => Some(decl.name.as_str()),
                        _ => None,
                    };

                    if item_name == Some(name) {
                        return Some(format!("{}.{}", module_name, name));
                    }
                }
            }
        }

        // 3. Not found in any opened module
        None
    }

    /// Open a module (bring its names into scope)
    fn open_module(&mut self, module_name: String) {
        self.opened_modules.push(module_name);
    }

    /// Close the most recently opened module
    fn close_module(&mut self) {
        self.opened_modules.pop();
    }

    /// Push a new scope for local definitions
    fn push_scope(&mut self) {
        self.local_scope.push_scope();
    }

    /// Pop the current scope
    fn pop_scope(&mut self) {
        self.local_scope.pop_scope();
    }

    /// Add a local definition (for shadowing)
    fn add_local(&mut self, name: String) {
        self.local_scope.insert(name, ());
    }
}

/// Represents a single item in an elaborated module
#[derive(Debug, Clone)]
pub enum ElaboratedItem {
    /// A signature spec (from module type) with substitutions applied
    Spec(Spec),
    /// A declaration (def/let) from module body with substitutions and resolved names
    Decl(Decl),
    /// A type alias from module body (e.g., `type state = f32`)
    TypeAlias(String, Type),
}

/// Represents a fully elaborated module with all includes expanded, type substitutions applied,
/// and names resolved. Contains both signature specs and body declarations in source order.
#[derive(Debug, Clone)]
pub struct ElaboratedModule {
    pub name: String,
    /// Items in source order (specs first, then body declarations)
    pub items: Vec<ElaboratedItem>,
}

/// Represents a parameterized module (functor) that hasn't been applied yet
#[derive(Debug, Clone)]
pub struct FunctorModule {
    /// The functor's name
    pub name: String,
    /// Parameters this functor takes
    pub params: Vec<crate::ast::ModuleParam>,
    /// The module signature (if any)
    pub signature: Option<ModuleTypeExpression>,
    /// The unevaluated body
    pub body: ModuleExpression,
}

/// Pre-elaborated prelude data that can be shared across compilations (for test performance)
#[derive(Clone)]
pub struct PreElaboratedPrelude {
    /// Module type registry: type name -> ModuleTypeExpression
    pub module_type_registry: HashMap<String, ModuleTypeExpression>,
    /// Elaborated modules: module_name -> ElaboratedModule
    pub elaborated_modules: HashMap<String, ElaboratedModule>,
    /// Set of known module names (for name resolution)
    pub known_modules: HashSet<String>,
    /// Type aliases from modules: "module.typename" -> underlying Type
    pub type_aliases: HashMap<String, Type>,
    /// Top-level prelude function declarations (auto-imported)
    /// Uses IndexMap to preserve insertion order (file order) for proper type-checking
    pub prelude_functions: IndexMap<String, Decl>,
}

/// Manages lazy loading of module files
pub struct ModuleManager {
    /// Module type registry: type name -> ModuleTypeExpression
    module_type_registry: HashMap<String, ModuleTypeExpression>,
    /// Elaborated modules: module_name -> ElaboratedModule
    pub(crate) elaborated_modules: HashMap<String, ElaboratedModule>,
    /// Functor modules: functor_name -> FunctorModule (unevaluated parameterized modules)
    functor_modules: HashMap<String, FunctorModule>,
    /// Set of known module names (for name resolution)
    known_modules: HashSet<String>,
    /// Type aliases from modules: "module.typename" -> underlying Type
    type_aliases: HashMap<String, Type>,
    /// Top-level prelude function declarations (auto-imported)
    /// Uses IndexMap to preserve insertion order (file order) for proper type-checking
    prelude_functions: IndexMap<String, Decl>,
}

impl ModuleManager {
    /// Create a new module manager with fully type-checked prelude
    pub fn new(node_counter: &mut NodeCounter) -> Self {
        match Self::create_prelude(node_counter) {
            Ok(prelude) => Self::from_prelude(&prelude),
            Err(e) => {
                eprintln!("ERROR creating prelude: {:?}", e);
                Self::new_empty()
            }
        }
    }

    /// Create an empty module manager without loading prelude (internal helper)
    fn new_empty() -> Self {
        let known_modules = [
            "f32",
            "f64",
            "f16",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "bool",
            "graphics32",
            "graphics64",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        ModuleManager {
            module_type_registry: HashMap::new(),
            elaborated_modules: HashMap::new(),
            functor_modules: HashMap::new(),
            known_modules,
            type_aliases: HashMap::new(),
            prelude_functions: IndexMap::new(),
        }
    }

    /// Load all prelude files automatically
    fn load_prelude_files(&mut self, node_counter: &mut NodeCounter) -> Result<()> {
        // Load all prelude files using include_str!
        self.load_str(include_str!("../../../prelude/math.wyn"), node_counter)?;
        self.load_str(include_str!("../../../prelude/graphics.wyn"), node_counter)?;
        self.load_str(include_str!("../../../prelude/rand.wyn"), node_counter)?;
        self.load_str(include_str!("../../../prelude/soacs.wyn"), node_counter)?;
        Ok(())
    }

    /// Create a pre-elaborated prelude by loading all prelude files
    /// This can be cached and reused across compilations
    /// Note: Type-checking happens separately in FrontEnd initialization
    pub fn create_prelude(node_counter: &mut NodeCounter) -> Result<PreElaboratedPrelude> {
        let mut manager = Self::new_empty();
        manager.load_prelude_files(node_counter)?;

        Ok(PreElaboratedPrelude {
            module_type_registry: manager.module_type_registry,
            elaborated_modules: manager.elaborated_modules,
            known_modules: manager.known_modules,
            type_aliases: manager.type_aliases,
            prelude_functions: manager.prelude_functions,
        })
    }

    /// Create a ModuleManager using a pre-elaborated prelude (avoids re-parsing)
    /// Advances the node_counter to start after all prelude NodeIds
    pub fn from_prelude(prelude: &PreElaboratedPrelude) -> Self {
        ModuleManager {
            module_type_registry: prelude.module_type_registry.clone(),
            elaborated_modules: prelude.elaborated_modules.clone(),
            functor_modules: HashMap::new(), // Prelude doesn't have functors
            known_modules: prelude.known_modules.clone(),
            type_aliases: prelude.type_aliases.clone(),
            prelude_functions: prelude.prelude_functions.clone(),
        }
    }

    /// Check if a name is a known module
    pub fn is_known_module(&self, name: &str) -> bool {
        self.known_modules.contains(name)
    }

    /// Resolve a qualified type alias (e.g., "rand.state" -> underlying type)
    pub fn resolve_type_alias(&self, qualified_name: &str) -> Option<&Type> {
        self.type_aliases.get(qualified_name)
    }

    /// Load and elaborate modules from a source string
    pub fn load_str(&mut self, source: &str, node_counter: &mut NodeCounter) -> Result<()> {
        // Parse the source
        let tokens = lexer::tokenize(source).map_err(|e| err_parse!("{}", e))?;
        let mut parser = Parser::new(tokens, node_counter);
        let program = parser.parse()?;

        // Register module types first
        self.register_module_types(&program)?;

        // Elaborate all modules from the program
        self.elaborate_all_modules(&program)?;

        Ok(())
    }

    /// Elaborate a single module binding
    fn elaborate_module_bind(&mut self, mb: &crate::ast::ModuleBind) -> Result<()> {
        if self.elaborated_modules.contains_key(&mb.name) {
            bail_module!("Module '{}' is already defined", mb.name);
        }
        if self.functor_modules.contains_key(&mb.name) {
            bail_module!("Module '{}' is already defined as a functor", mb.name);
        }

        // If this module has parameters, store it as a functor (unevaluated)
        if !mb.params.is_empty() {
            let functor = FunctorModule {
                name: mb.name.clone(),
                params: mb.params.clone(),
                signature: mb.signature.clone(),
                body: mb.body.clone(),
            };
            self.functor_modules.insert(mb.name.clone(), functor);
            // Register as a known module for name resolution
            self.known_modules.insert(mb.name.clone());
            return Ok(());
        }

        // Extract type substitutions from the signature
        let substitutions = if let Some(signature) = &mb.signature {
            self.extract_substitutions(signature)?
        } else {
            HashMap::new()
        };

        let mut items = Vec::new();

        // Elaborate the module signature if it exists
        if let Some(signature) = &mb.signature {
            let specs = self.elaborate_module_type(signature, &HashMap::new())?;
            // Wrap specs in ElaboratedItem::Spec
            items.extend(specs.into_iter().map(ElaboratedItem::Spec));
        }

        // Elaborate the module body
        let body_items = self.elaborate_module_body(&mb.body, &mb.name, &substitutions, &HashMap::new())?;
        items.extend(body_items);

        // Add type aliases from `with` substitutions in the signature
        // This ensures that types like `t` from `(my_numeric with t = f32)` are available
        // when the module is used as a functor argument
        for (type_name, underlying_type) in &substitutions {
            // Only add if not already present from the body
            let already_present = items
                .iter()
                .any(|item| matches!(item, ElaboratedItem::TypeAlias(name, _) if name == type_name));
            if !already_present {
                items.push(ElaboratedItem::TypeAlias(
                    type_name.clone(),
                    underlying_type.clone(),
                ));
            }
        }

        let elaborated = ElaboratedModule {
            name: mb.name.clone(),
            items,
        };

        // Register type aliases from this module
        for item in &elaborated.items {
            if let ElaboratedItem::TypeAlias(type_name, underlying_type) = item {
                let qualified_name = format!("{}.{}", mb.name, type_name);
                self.type_aliases.insert(qualified_name, underlying_type.clone());
            }
        }

        self.elaborated_modules.insert(mb.name.clone(), elaborated);
        // Also register as a known module for name resolution
        self.known_modules.insert(mb.name.clone());
        Ok(())
    }

    /// Elaborate all module bindings from a parsed program
    /// This is the public entry point for elaborating modules from any source
    pub fn elaborate_modules(&mut self, program: &Program) -> Result<()> {
        // Register module types first
        self.register_module_types(program)?;

        // Elaborate each module binding
        for decl in &program.declarations {
            if let Declaration::ModuleBind(mb) = decl {
                self.elaborate_module_bind(mb)?;
            }
        }
        Ok(())
    }

    /// Elaborate all module bindings and collect top-level declarations (for prelude files)
    fn elaborate_all_modules(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            // Collect top-level function declarations for prelude
            if let Declaration::Decl(d) = decl {
                self.prelude_functions.insert(d.name.clone(), d.clone());
                continue;
            }

            if let Declaration::ModuleBind(mb) = decl {
                self.elaborate_module_bind(mb)?;
            }
        }
        Ok(())
    }

    /// Extract type substitutions from a module signature
    /// e.g., (float with t = f32 with int_t = u32) -> {t: f32, int_t: u32}
    fn extract_substitutions(&self, mte: &ModuleTypeExpression) -> Result<HashMap<String, Type>> {
        let mut substitutions = HashMap::new();
        let mut current = mte;

        // Walk through nested With expressions
        while let ModuleTypeExpression::With(inner, type_name, _type_params, type_value) = current {
            substitutions.insert(type_name.clone(), type_value.clone());
            current = inner;
        }

        Ok(substitutions)
    }

    /// Register all module type definitions from a parsed program
    fn register_module_types(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            if let Declaration::ModuleTypeBind(mtb) = decl {
                if self.module_type_registry.contains_key(&mtb.name) {
                    bail_module!("Module type '{}' is already defined", mtb.name);
                }
                self.module_type_registry.insert(mtb.name.clone(), mtb.definition.clone());
            }
        }
        Ok(())
    }

    /// Elaborate a module type expression into a flat list of specs
    /// Recursively expands includes and applies type substitutions
    fn elaborate_module_type(
        &self,
        mte: &ModuleTypeExpression,
        substitutions: &HashMap<String, Type>,
    ) -> Result<Vec<Spec>> {
        match mte {
            ModuleTypeExpression::Name(name) => {
                // Look up the module type in the registry
                let definition = self
                    .module_type_registry
                    .get(name)
                    .ok_or_else(|| err_module!("Module type '{}' not found", name))?;
                // Recurse on the definition
                self.elaborate_module_type(definition, substitutions)
            }

            ModuleTypeExpression::Signature(specs) => {
                // Process each spec, expanding includes and applying substitutions
                let mut result = Vec::new();
                for spec in specs {
                    match spec {
                        Spec::Include(inner_mte) => {
                            // Recursively elaborate the included module type
                            let included_specs = self.elaborate_module_type(inner_mte, substitutions)?;
                            result.extend(included_specs);
                        }
                        _ => {
                            // Apply type substitutions to the spec and add it
                            let substituted_spec = self.substitute_in_spec(spec, substitutions);
                            result.push(substituted_spec);
                        }
                    }
                }
                Ok(result)
            }

            ModuleTypeExpression::With(inner, type_name, _type_params, type_value) => {
                // Add the type substitution and recurse on the inner expression
                let mut new_substitutions = substitutions.clone();
                new_substitutions.insert(type_name.clone(), type_value.clone());
                self.elaborate_module_type(inner, &new_substitutions)
            }

            ModuleTypeExpression::Arrow(_, _, _) | ModuleTypeExpression::FunctorType(_, _) => {
                // Functor types not yet supported
                Err(err_module!("Functor types are not yet supported"))
            }
        }
    }

    /// Apply type substitutions to a spec
    fn substitute_in_spec(&self, spec: &Spec, substitutions: &HashMap<String, Type>) -> Spec {
        match spec {
            Spec::Sig(name, type_params, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::Sig(name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::SigOp(op, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::SigOp(op.clone(), substituted_ty)
            }
            Spec::Type(kind, name, type_params, maybe_ty) => {
                let substituted_ty = maybe_ty.as_ref().map(|ty| self.substitute_in_type(ty, substitutions));
                Spec::Type(kind.clone(), name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::Module(name, mte) => {
                // Don't substitute in nested module signatures for now
                Spec::Module(name.clone(), mte.clone())
            }
            Spec::Include(_) => {
                // Includes should have been expanded by now
                spec.clone()
            }
        }
    }

    /// Apply type substitutions to a type
    fn substitute_in_type(&self, ty: &Type, substitutions: &HashMap<String, Type>) -> Type {
        use crate::ast::TypeName;

        match ty {
            Type::Constructed(name, args) => {
                // Check if this is a named type that should be substituted
                if let TypeName::Named(type_name) = name {
                    if args.is_empty() {
                        if let Some(replacement) = substitutions.get(type_name) {
                            return replacement.clone();
                        }
                    }
                }

                // Recursively substitute in type arguments
                let new_args: Vec<Type> =
                    args.iter().map(|arg| self.substitute_in_type(arg, substitutions)).collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Like substitute_in_type but also handles parameter type references (e.g., n.t)
    fn substitute_in_type_with_params(
        &self,
        ty: &Type,
        substitutions: &HashMap<String, Type>,
        param_bindings: &HashMap<String, ElaboratedModule>,
    ) -> Type {
        use crate::ast::TypeName;

        match ty {
            Type::Constructed(name, args) => {
                if let TypeName::Named(type_name) = name {
                    if args.is_empty() {
                        // Check direct substitutions first
                        if let Some(replacement) = substitutions.get(type_name) {
                            return replacement.clone();
                        }

                        // Check for param.type pattern (e.g., "n.t")
                        if let Some((param_name, field_name)) = type_name.split_once('.') {
                            if let Some(param_module) = param_bindings.get(param_name) {
                                // Look for a type alias in the parameter module
                                for item in &param_module.items {
                                    if let ElaboratedItem::TypeAlias(alias_name, underlying) = item {
                                        if alias_name == field_name {
                                            return underlying.clone();
                                        }
                                    }
                                }
                                // Also check the global type_aliases for the bound module
                                let qualified = format!("{}.{}", param_module.name, field_name);
                                if let Some(underlying) = self.type_aliases.get(&qualified) {
                                    return underlying.clone();
                                }
                            }
                        }
                    }
                }

                // Recursively substitute in type arguments
                let new_args: Vec<Type> = args
                    .iter()
                    .map(|arg| self.substitute_in_type_with_params(arg, substitutions, param_bindings))
                    .collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Builtin/intrinsic modules that shouldn't be type-checked
    /// (their implementations use internal __builtin_* functions)
    const BUILTIN_MODULES: &'static [&'static str] = &[
        "f32",
        "f64",
        "f16",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "bool",
        "graphics32",
        "graphics64",
    ];

    /// Check if a module is a builtin primitive module
    fn is_builtin_module(name: &str) -> bool {
        Self::BUILTIN_MODULES.contains(&name)
    }

    /// Get all module function declarations for type-checking
    /// Returns (module_name, decl) pairs for user-defined modules only
    /// (excludes builtin primitive modules like f32, i32, etc.)
    pub fn get_module_function_declarations(&self) -> Vec<(&str, &Decl)> {
        self.elaborated_modules
            .iter()
            .filter(|(name, _)| !Self::is_builtin_module(name))
            .flat_map(|(module_name, elaborated)| {
                elaborated.items.iter().filter_map(move |item| {
                    if let ElaboratedItem::Decl(decl) = item {
                        Some((module_name.as_str(), decl))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Get all module declarations for flattening (includes builtin module constants like f32.pi)
    /// Excludes intrinsic functions (those using __builtin_* in their body)
    pub fn get_all_module_declarations(&self) -> Vec<(&str, &Decl)> {
        self.elaborated_modules
            .iter()
            .flat_map(|(module_name, elaborated)| {
                elaborated.items.iter().filter_map(move |item| {
                    if let ElaboratedItem::Decl(decl) = item {
                        // Skip intrinsics (functions that use __builtin_* calls)
                        if Self::is_intrinsic_decl(decl) {
                            None
                        } else {
                            Some((module_name.as_str(), decl))
                        }
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Check if a declaration is an intrinsic (uses __builtin_* in its body)
    fn is_intrinsic_decl(decl: &Decl) -> bool {
        Self::expr_uses_builtin(&decl.body)
    }

    /// Recursively check if an expression uses __builtin_* functions
    fn expr_uses_builtin(expr: &crate::ast::Expression) -> bool {
        use crate::ast::ExprKind;
        match &expr.kind {
            ExprKind::Identifier(quals, name) => {
                name.starts_with("__builtin_") || (quals.is_empty() && name.starts_with("__builtin_"))
            }
            ExprKind::Application(func, args) => {
                Self::expr_uses_builtin(func) || args.iter().any(Self::expr_uses_builtin)
            }
            ExprKind::Lambda(lambda) => Self::expr_uses_builtin(&lambda.body),
            ExprKind::LetIn(let_in) => {
                Self::expr_uses_builtin(&let_in.value) || Self::expr_uses_builtin(&let_in.body)
            }
            ExprKind::If(if_expr) => {
                Self::expr_uses_builtin(&if_expr.condition)
                    || Self::expr_uses_builtin(&if_expr.then_branch)
                    || Self::expr_uses_builtin(&if_expr.else_branch)
            }
            ExprKind::BinaryOp(_, lhs, rhs) => Self::expr_uses_builtin(lhs) || Self::expr_uses_builtin(rhs),
            ExprKind::UnaryOp(_, operand) => Self::expr_uses_builtin(operand),
            ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
                exprs.iter().any(Self::expr_uses_builtin)
            }
            _ => false,
        }
    }

    /// Get all top-level prelude function declarations for flattening
    /// These are auto-imported functions from prelude files
    pub fn get_prelude_function_declarations(&self) -> Vec<&Decl> {
        self.prelude_functions.values().collect()
    }

    /// Check if a name is a qualified module reference (e.g., "f32.sum")
    pub fn is_qualified_name(name: &str) -> bool {
        name.contains('.')
    }

    /// Split a qualified name into (module, function) parts
    /// E.g., "f32.sum" -> Some(("f32", "sum"))
    pub fn split_qualified_name(name: &str) -> Option<(&str, &str)> {
        let parts: Vec<&str> = name.splitn(2, '.').collect();
        if parts.len() == 2 { Some((parts[0], parts[1])) } else { None }
    }

    /// Check if a name is a top-level prelude function (auto-imported)
    pub fn is_prelude_function(&self, name: &str) -> bool {
        self.prelude_functions.contains_key(name)
    }

    /// Get a top-level prelude function declaration
    pub fn get_prelude_function(&self, name: &str) -> Option<&Decl> {
        self.prelude_functions.get(name)
    }

    /// Get an elaborated module by name
    pub fn get_elaborated_module(&self, name: &str) -> Option<&ElaboratedModule> {
        self.elaborated_modules.get(name)
    }

    /// Elaborate a module body expression into a list of elaborated items
    /// Applies type substitutions to declaration signatures
    /// `param_bindings` maps parameter names to their resolved module's elaborated items
    fn elaborate_module_body(
        &self,
        module_expr: &ModuleExpression,
        module_name: &str,
        substitutions: &HashMap<String, Type>,
        param_bindings: &HashMap<String, ElaboratedModule>,
    ) -> Result<Vec<ElaboratedItem>> {
        match module_expr {
            ModuleExpression::Struct(declarations) => {
                // First pass: collect all function names in this module (for intra-module resolution)
                let mut module_functions: HashSet<String> = HashSet::new();
                for decl in declarations {
                    match decl {
                        Declaration::Decl(d) => {
                            module_functions.insert(d.name.clone());
                        }
                        Declaration::Sig(sig_decl) => {
                            module_functions.insert(sig_decl.name.clone());
                        }
                        _ => {}
                    }
                }

                let mut items = Vec::new();

                // Second pass: elaborate declarations with name resolution
                for decl in declarations {
                    match decl {
                        Declaration::Decl(d) => {
                            // Apply type substitutions and resolve names
                            let elaborated_decl = self.elaborate_decl_signature_with_params(
                                d,
                                module_name,
                                &module_functions,
                                substitutions,
                                param_bindings,
                            );
                            items.push(ElaboratedItem::Decl(elaborated_decl));
                        }
                        Declaration::Sig(sig_decl) => {
                            // Apply type substitutions to sig declarations
                            let substituted_ty = self.substitute_in_type(&sig_decl.ty, substitutions);

                            // Convert size_params and type_params to Vec<TypeParam>
                            use crate::ast::TypeParam;
                            let mut type_params_vec = Vec::new();
                            for size_param in &sig_decl.size_params {
                                type_params_vec.push(TypeParam::Size(size_param.clone()));
                            }
                            for type_param in &sig_decl.type_params {
                                type_params_vec.push(TypeParam::Type(type_param.clone()));
                            }

                            let spec = Spec::Sig(sig_decl.name.clone(), type_params_vec, substituted_ty);
                            items.push(ElaboratedItem::Spec(spec));
                        }
                        Declaration::Open(_) => {
                            // TODO: Handle open declarations for name resolution
                        }
                        Declaration::TypeBind(type_bind) => {
                            // Handle type aliases, including those referencing parameters
                            // e.g., `type t = n.t` where n is a parameter
                            let substituted_ty = self.substitute_in_type_with_params(
                                &type_bind.definition,
                                substitutions,
                                param_bindings,
                            );
                            items.push(ElaboratedItem::TypeAlias(type_bind.name.clone(), substituted_ty));
                        }
                        _ => {
                            // Skip other declaration types (ModuleTypeBind, etc.)
                        }
                    }
                }

                Ok(items)
            }
            ModuleExpression::Name(name) => {
                // Look up an existing elaborated module by name
                if let Some(param_module) = param_bindings.get(name) {
                    // It's a parameter reference - return its items
                    Ok(param_module.items.clone())
                } else if let Some(elaborated) = self.elaborated_modules.get(name) {
                    // It's a known elaborated module
                    Ok(elaborated.items.clone())
                } else {
                    Err(err_module!("Unknown module: '{}'", name))
                }
            }
            ModuleExpression::Application(functor_expr, arg_expr) => {
                // Apply a functor to an argument
                // 1. The functor must be a Name referencing a FunctorModule
                // 2. The argument must be a Name referencing an elaborated module
                let functor_name = match functor_expr.as_ref() {
                    ModuleExpression::Name(name) => name,
                    _ => return Err(err_module!("Functor must be a module name")),
                };

                let arg_name = match arg_expr.as_ref() {
                    ModuleExpression::Name(name) => name,
                    _ => return Err(err_module!("Functor argument must be a module name")),
                };

                // Look up the functor
                let functor = self.functor_modules.get(functor_name).ok_or_else(|| {
                    err_module!("'{}' is not a parameterized module (functor)", functor_name)
                })?;

                // Look up the argument module
                let arg_module = self
                    .elaborated_modules
                    .get(arg_name)
                    .ok_or_else(|| err_module!("Unknown module argument: '{}'", arg_name))?;

                // Create parameter binding: param_name -> arg_module
                if functor.params.len() != 1 {
                    return Err(err_module!(
                        "Functor '{}' expects {} parameters, got 1",
                        functor_name,
                        functor.params.len()
                    ));
                }

                let param_name = &functor.params[0].name;
                let mut new_param_bindings = param_bindings.clone();
                new_param_bindings.insert(param_name.clone(), arg_module.clone());

                // Extract type substitutions from the argument module
                // Find type aliases in the arg module to use as substitutions
                let mut new_substitutions = substitutions.clone();
                for item in &arg_module.items {
                    if let ElaboratedItem::TypeAlias(type_name, underlying_type) = item {
                        // Map param.type_name to the underlying type
                        let param_qualified = format!("{}.{}", param_name, type_name);
                        new_substitutions.insert(param_qualified, underlying_type.clone());
                    }
                }

                // Also look up type from the argument module's signature
                // For `my_f32_num : (my_numeric with t = f32)`, we need to know t = f32
                // This info is already in the arg_module's type aliases via `with` extraction
                // But we also need to handle direct type references like `n.t`
                // For now, add the arg module's type aliases with param prefix
                if let Some(arg_elaborated) = self.elaborated_modules.get(arg_name) {
                    for item in &arg_elaborated.items {
                        if let ElaboratedItem::TypeAlias(type_name, underlying_type) = item {
                            let param_qualified = format!("{}.{}", param_name, type_name);
                            new_substitutions.insert(param_qualified, underlying_type.clone());
                        }
                    }
                }

                // Look up type substitutions from the type_aliases map for the arg module
                for (qualified_name, underlying_type) in &self.type_aliases {
                    if qualified_name.starts_with(&format!("{}.", arg_name)) {
                        let type_name = qualified_name.strip_prefix(&format!("{}.", arg_name)).unwrap();
                        let param_qualified = format!("{}.{}", param_name, type_name);
                        new_substitutions.insert(param_qualified, underlying_type.clone());
                    }
                }

                // Elaborate the functor body with the parameter binding
                self.elaborate_module_body(
                    &functor.body,
                    module_name,
                    &new_substitutions,
                    &new_param_bindings,
                )
            }
            _ => {
                // For now, only handle struct, name, and application module expressions
                Err(err_module!("Unsupported module expression type"))
            }
        }
    }

    /// Elaborate a declaration's signature (params and return type) with type substitutions
    fn elaborate_decl_signature(
        &self,
        decl: &Decl,
        module_name: &str,
        module_functions: &HashSet<String>,
        substitutions: &HashMap<String, Type>,
    ) -> Decl {
        // Apply type substitutions to params
        let new_params: Vec<Pattern> =
            decl.params.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();

        // Apply type substitutions to return type
        let new_ty = decl.ty.as_ref().map(|ty| self.substitute_in_type(ty, substitutions));

        // Collect parameter names to avoid qualifying them as module functions
        let param_names: HashSet<String> =
            decl.params.iter().flat_map(|p| self.collect_pattern_names(p)).collect();

        // Resolve names in body (convert FieldAccess to QualifiedName, qualify intra-module refs)
        let mut new_body = decl.body.clone();
        self.resolve_names_in_expr(&mut new_body, module_name, module_functions, &param_names);

        Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
            params: new_params,
            ty: new_ty,
            body: new_body,
        }
    }

    /// Elaborate a declaration's signature with type substitutions and parameter bindings
    /// This handles functor parameter references in types and expressions
    fn elaborate_decl_signature_with_params(
        &self,
        decl: &Decl,
        module_name: &str,
        module_functions: &HashSet<String>,
        substitutions: &HashMap<String, Type>,
        param_bindings: &HashMap<String, ElaboratedModule>,
    ) -> Decl {
        // Apply type substitutions to params (including param.type references)
        let new_params: Vec<Pattern> = decl
            .params
            .iter()
            .map(|p| self.substitute_in_pattern_with_params(p, substitutions, param_bindings))
            .collect();

        // Apply type substitutions to return type
        let new_ty = decl
            .ty
            .as_ref()
            .map(|ty| self.substitute_in_type_with_params(ty, substitutions, param_bindings));

        // Collect parameter names to avoid qualifying them as module functions
        let param_names: HashSet<String> =
            decl.params.iter().flat_map(|p| self.collect_pattern_names(p)).collect();

        // Resolve names in body (convert FieldAccess to QualifiedName, qualify intra-module refs)
        // Also handle parameter module references (e.g., n.add -> my_f32_num.add)
        let mut new_body = decl.body.clone();
        self.resolve_names_in_expr_with_params(
            &mut new_body,
            module_name,
            module_functions,
            &param_names,
            param_bindings,
        );

        Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
            params: new_params,
            ty: new_ty,
            body: new_body,
        }
    }

    /// Collect all bound names from a pattern
    fn collect_pattern_names(&self, pattern: &Pattern) -> Vec<String> {
        use crate::ast::PatternKind;
        let mut names = Vec::new();
        match &pattern.kind {
            PatternKind::Name(name) => names.push(name.clone()),
            PatternKind::Typed(inner, _) => names.extend(self.collect_pattern_names(inner)),
            PatternKind::Tuple(pats) => {
                for p in pats {
                    names.extend(self.collect_pattern_names(p));
                }
            }
            PatternKind::Constructor(_, pats) => {
                for p in pats {
                    names.extend(self.collect_pattern_names(p));
                }
            }
            PatternKind::Record(fields) => {
                for f in fields {
                    if let Some(p) = &f.pattern {
                        names.extend(self.collect_pattern_names(p));
                    } else {
                        names.push(f.field.clone());
                    }
                }
            }
            PatternKind::Attributed(_, inner) => names.extend(self.collect_pattern_names(inner)),
            PatternKind::Wildcard | PatternKind::Unit | PatternKind::Literal(_) => {}
        }
        names
    }

    /// Resolve names in an expression within a module context
    /// - Converts FieldAccess to QualifiedName for external module references
    /// - Qualifies intra-module function references
    fn resolve_names_in_expr(
        &self,
        expr: &mut crate::ast::Expression,
        module_name: &str,
        module_functions: &HashSet<String>,
        local_bindings: &HashSet<String>,
    ) {
        use crate::ast::ExprKind;

        match &mut expr.kind {
            ExprKind::Identifier(quals, name) => {
                // Check if this is an intra-module function reference (not shadowed by local binding)
                if quals.is_empty() && !local_bindings.contains(name) && module_functions.contains(name) {
                    // Convert to qualified identifier: next -> rand.next
                    *quals = vec![module_name.to_string()];
                }
            }
            ExprKind::FieldAccess(obj, field) => {
                // Check if this is module.name pattern
                if let ExprKind::Identifier(quals, name) = &obj.kind {
                    if quals.is_empty() && self.known_modules.contains(name) {
                        // Convert to qualified Identifier
                        expr.kind = ExprKind::Identifier(vec![name.clone()], field.clone());
                        return;
                    }
                }
                // Otherwise recurse into object
                self.resolve_names_in_expr(obj, module_name, module_functions, local_bindings);
            }
            ExprKind::Application(func, args) => {
                self.resolve_names_in_expr(func, module_name, module_functions, local_bindings);
                for arg in args {
                    self.resolve_names_in_expr(arg, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::Lambda(lambda) => {
                // Collect lambda parameter names
                let mut inner_bindings = local_bindings.clone();
                for p in &lambda.params {
                    inner_bindings.extend(self.collect_pattern_names(p));
                }
                self.resolve_names_in_expr(
                    &mut lambda.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                );
            }
            ExprKind::LetIn(let_in) => {
                self.resolve_names_in_expr(
                    &mut let_in.value,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                // Collect let binding names
                let mut inner_bindings = local_bindings.clone();
                inner_bindings.extend(self.collect_pattern_names(&let_in.pattern));
                self.resolve_names_in_expr(
                    &mut let_in.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                );
            }
            ExprKind::If(if_expr) => {
                self.resolve_names_in_expr(
                    &mut if_expr.condition,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                self.resolve_names_in_expr(
                    &mut if_expr.then_branch,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                self.resolve_names_in_expr(
                    &mut if_expr.else_branch,
                    module_name,
                    module_functions,
                    local_bindings,
                );
            }
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.resolve_names_in_expr(lhs, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(rhs, module_name, module_functions, local_bindings);
            }
            ExprKind::UnaryOp(_, operand) => {
                self.resolve_names_in_expr(operand, module_name, module_functions, local_bindings);
            }
            ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
                for e in exprs {
                    self.resolve_names_in_expr(e, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.resolve_names_in_expr(arr, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(idx, module_name, module_functions, local_bindings);
            }
            ExprKind::ArrayWith { array, index, value } => {
                self.resolve_names_in_expr(array, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(index, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(value, module_name, module_functions, local_bindings);
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, e) in fields {
                    self.resolve_names_in_expr(e, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(ref mut init) = loop_expr.init {
                    self.resolve_names_in_expr(init, module_name, module_functions, local_bindings);
                }
                // Collect loop variable binding
                let mut inner_bindings = local_bindings.clone();
                inner_bindings.extend(self.collect_pattern_names(&loop_expr.pattern));
                match &mut loop_expr.form {
                    crate::ast::LoopForm::While(cond) => {
                        self.resolve_names_in_expr(cond, module_name, module_functions, &inner_bindings);
                    }
                    crate::ast::LoopForm::For(var, bound) => {
                        inner_bindings.insert(var.clone());
                        self.resolve_names_in_expr(bound, module_name, module_functions, local_bindings);
                    }
                    crate::ast::LoopForm::ForIn(var_pat, iter) => {
                        inner_bindings.extend(self.collect_pattern_names(var_pat));
                        self.resolve_names_in_expr(iter, module_name, module_functions, local_bindings);
                    }
                }
                self.resolve_names_in_expr(
                    &mut loop_expr.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                );
            }
            ExprKind::Match(match_expr) => {
                self.resolve_names_in_expr(
                    &mut match_expr.scrutinee,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                for case in &mut match_expr.cases {
                    let mut inner_bindings = local_bindings.clone();
                    inner_bindings.extend(self.collect_pattern_names(&case.pattern));
                    self.resolve_names_in_expr(
                        &mut case.body,
                        module_name,
                        module_functions,
                        &inner_bindings,
                    );
                }
            }
            ExprKind::Range(range_expr) => {
                self.resolve_names_in_expr(
                    &mut range_expr.start,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                self.resolve_names_in_expr(
                    &mut range_expr.end,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                if let Some(step) = &mut range_expr.step {
                    self.resolve_names_in_expr(step, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::Slice(slice_expr) => {
                self.resolve_names_in_expr(
                    &mut slice_expr.array,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                if let Some(start) = &mut slice_expr.start {
                    self.resolve_names_in_expr(start, module_name, module_functions, local_bindings);
                }
                if let Some(end) = &mut slice_expr.end {
                    self.resolve_names_in_expr(end, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.resolve_names_in_expr(inner, module_name, module_functions, local_bindings);
            }
            // Leaf expressions - nothing to resolve
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {}
        }
    }

    /// Apply type substitutions to a pattern
    fn substitute_in_pattern(&self, pattern: &Pattern, substitutions: &HashMap<String, Type>) -> Pattern {
        let new_kind = match &pattern.kind {
            PatternKind::Typed(inner, ty) => {
                let new_ty = self.substitute_in_type(ty, substitutions);
                PatternKind::Typed(inner.clone(), new_ty)
            }
            PatternKind::Tuple(pats) => {
                let new_pats: Vec<Pattern> =
                    pats.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();
                PatternKind::Tuple(new_pats)
            }
            PatternKind::Record(fields) => {
                let new_fields = fields
                    .iter()
                    .map(|field| crate::ast::RecordPatternField {
                        field: field.field.clone(),
                        pattern: field
                            .pattern
                            .as_ref()
                            .map(|p| self.substitute_in_pattern(p, substitutions)),
                    })
                    .collect();
                PatternKind::Record(new_fields)
            }
            PatternKind::Constructor(name, pats) => {
                let new_pats: Vec<Pattern> =
                    pats.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();
                PatternKind::Constructor(name.clone(), new_pats)
            }
            PatternKind::Attributed(attrs, inner) => {
                let new_inner = self.substitute_in_pattern(inner, substitutions);
                PatternKind::Attributed(attrs.clone(), Box::new(new_inner))
            }
            // Name, Wildcard, Literal, Unit don't contain types
            _ => pattern.kind.clone(),
        };

        Node {
            h: pattern.h.clone(),
            kind: new_kind,
        }
    }

    /// Like substitute_in_pattern but handles param.type references
    fn substitute_in_pattern_with_params(
        &self,
        pattern: &Pattern,
        substitutions: &HashMap<String, Type>,
        param_bindings: &HashMap<String, ElaboratedModule>,
    ) -> Pattern {
        let new_kind = match &pattern.kind {
            PatternKind::Typed(inner, ty) => {
                let new_ty = self.substitute_in_type_with_params(ty, substitutions, param_bindings);
                PatternKind::Typed(inner.clone(), new_ty)
            }
            PatternKind::Tuple(pats) => {
                let new_pats: Vec<Pattern> = pats
                    .iter()
                    .map(|p| self.substitute_in_pattern_with_params(p, substitutions, param_bindings))
                    .collect();
                PatternKind::Tuple(new_pats)
            }
            PatternKind::Record(fields) => {
                let new_fields = fields
                    .iter()
                    .map(|field| crate::ast::RecordPatternField {
                        field: field.field.clone(),
                        pattern: field.pattern.as_ref().map(|p| {
                            self.substitute_in_pattern_with_params(p, substitutions, param_bindings)
                        }),
                    })
                    .collect();
                PatternKind::Record(new_fields)
            }
            PatternKind::Constructor(name, pats) => {
                let new_pats: Vec<Pattern> = pats
                    .iter()
                    .map(|p| self.substitute_in_pattern_with_params(p, substitutions, param_bindings))
                    .collect();
                PatternKind::Constructor(name.clone(), new_pats)
            }
            PatternKind::Attributed(attrs, inner) => {
                let new_inner =
                    self.substitute_in_pattern_with_params(inner, substitutions, param_bindings);
                PatternKind::Attributed(attrs.clone(), Box::new(new_inner))
            }
            // Name, Wildcard, Literal, Unit don't contain types
            _ => pattern.kind.clone(),
        };

        Node {
            h: pattern.h.clone(),
            kind: new_kind,
        }
    }

    /// Like resolve_names_in_expr but also handles parameter module references
    /// E.g., n.add -> my_f32_num.add when n is bound to my_f32_num
    fn resolve_names_in_expr_with_params(
        &self,
        expr: &mut crate::ast::Expression,
        module_name: &str,
        module_functions: &HashSet<String>,
        local_bindings: &HashSet<String>,
        param_bindings: &HashMap<String, ElaboratedModule>,
    ) {
        use crate::ast::ExprKind;

        match &mut expr.kind {
            ExprKind::Identifier(quals, name) => {
                // Check if this is an intra-module function reference (not shadowed by local binding)
                if quals.is_empty() && !local_bindings.contains(name) && module_functions.contains(name) {
                    // Convert to qualified identifier: next -> rand.next
                    *quals = vec![module_name.to_string()];
                }
            }
            ExprKind::FieldAccess(obj, field) => {
                // Check if this is param.name pattern (parameter module reference)
                if let ExprKind::Identifier(quals, name) = &obj.kind {
                    if quals.is_empty() {
                        // Check if it's a parameter reference
                        if let Some(param_module) = param_bindings.get(name) {
                            // Convert n.add to my_f32_num.add
                            expr.kind =
                                ExprKind::Identifier(vec![param_module.name.clone()], field.clone());
                            return;
                        }
                        // Check if it's a known module
                        if self.known_modules.contains(name) {
                            // Convert to qualified Identifier
                            expr.kind = ExprKind::Identifier(vec![name.clone()], field.clone());
                            return;
                        }
                    }
                }
                // Otherwise recurse into object
                self.resolve_names_in_expr_with_params(
                    obj,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
            }
            ExprKind::Application(func, args) => {
                self.resolve_names_in_expr_with_params(
                    func,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                for arg in args {
                    self.resolve_names_in_expr_with_params(
                        arg,
                        module_name,
                        module_functions,
                        local_bindings,
                        param_bindings,
                    );
                }
            }
            ExprKind::Lambda(lambda) => {
                // Collect lambda parameter names
                let mut inner_bindings = local_bindings.clone();
                for p in &lambda.params {
                    inner_bindings.extend(self.collect_pattern_names(p));
                }
                self.resolve_names_in_expr_with_params(
                    &mut lambda.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                    param_bindings,
                );
            }
            ExprKind::LetIn(let_in) => {
                self.resolve_names_in_expr_with_params(
                    &mut let_in.value,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                // Collect let binding names
                let mut inner_bindings = local_bindings.clone();
                inner_bindings.extend(self.collect_pattern_names(&let_in.pattern));
                self.resolve_names_in_expr_with_params(
                    &mut let_in.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                    param_bindings,
                );
            }
            ExprKind::If(if_expr) => {
                self.resolve_names_in_expr_with_params(
                    &mut if_expr.condition,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                self.resolve_names_in_expr_with_params(
                    &mut if_expr.then_branch,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                self.resolve_names_in_expr_with_params(
                    &mut if_expr.else_branch,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
            }
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.resolve_names_in_expr_with_params(
                    lhs,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                self.resolve_names_in_expr_with_params(
                    rhs,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
            }
            ExprKind::UnaryOp(_, operand) => {
                self.resolve_names_in_expr_with_params(
                    operand,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
            }
            ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
                for e in exprs {
                    self.resolve_names_in_expr_with_params(
                        e,
                        module_name,
                        module_functions,
                        local_bindings,
                        param_bindings,
                    );
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.resolve_names_in_expr_with_params(
                    arr,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                self.resolve_names_in_expr_with_params(
                    idx,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
            }
            ExprKind::ArrayWith { array, index, value } => {
                self.resolve_names_in_expr_with_params(
                    array,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                self.resolve_names_in_expr_with_params(
                    index,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                self.resolve_names_in_expr_with_params(
                    value,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, e) in fields {
                    self.resolve_names_in_expr_with_params(
                        e,
                        module_name,
                        module_functions,
                        local_bindings,
                        param_bindings,
                    );
                }
            }
            ExprKind::Match(match_expr) => {
                self.resolve_names_in_expr_with_params(
                    &mut match_expr.scrutinee,
                    module_name,
                    module_functions,
                    local_bindings,
                    param_bindings,
                );
                for case in &mut match_expr.cases {
                    let mut inner_bindings = local_bindings.clone();
                    inner_bindings.extend(self.collect_pattern_names(&case.pattern));
                    self.resolve_names_in_expr_with_params(
                        &mut case.body,
                        module_name,
                        module_functions,
                        &inner_bindings,
                        param_bindings,
                    );
                }
            }
            // Literals and unit don't contain references
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit => {}
            // Other cases that might need handling
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
