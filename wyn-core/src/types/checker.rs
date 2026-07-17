use super::{SkolemId, Type, TypeExt, TypeName, TypeScheme};
use crate::ast::*;
use crate::builtins::{by_id, BuiltinId};
use crate::error::{CompilerError, Result};
use crate::interface::{AttrExt, Attribute};
use crate::module_manager::ModuleManager;
use crate::name_resolution::NameResolution;
use crate::scope::{IdentifierKind, ScopeEntry, ScopeStack};
use crate::{
    bail_type_at, err_module, err_type, err_type_at, err_undef_at, LookupMap, LookupSet, StableMap,
};
use log::debug;
use polytype::Context;
use std::collections::BTreeSet;

// Import type helper functions from parent module
use super::patterns::coverage::{check_match, format_cov_pat, CoverageError};
use super::{
    as_arrow, bool_type, f32, function, i32, mat, no_buffer, record, sized_array, tuple, unit, vec, Diet,
};

/// Render a single swizzle slot index as its `xyzw` letter. Used by
/// VecWith diagnostics so error messages match the user's source.
/// Map a primitive scalar type to its surface module name
/// (`Float(32)` → `"f32"`, `Bool` → `"bool"`, …). Used by the vec
/// constructor dispatch to compute the per-component target type name
/// it'll feed into the catalog lookup at to_tlc desugar time. Returns
/// `None` for any non-primitive type (composite arrays, tuples, etc.)
/// — the vec dispatch falls back to the standard undefined-name path
/// for those.
pub(crate) fn type_name_to_module(ty: &Type) -> Option<String> {
    match ty {
        Type::Constructed(TypeName::Bool, _) => Some("bool".to_string()),
        Type::Constructed(TypeName::Float(b), _) => Some(format!("f{}", b)),
        Type::Constructed(TypeName::Int(b), _) => Some(format!("i{}", b)),
        Type::Constructed(TypeName::UInt(b), _) => Some(format!("u{}", b)),
        _ => None,
    }
}

/// Canonical, variable-insensitive signature of a resource type, used to
/// decide whether two entries name the same buffer at a `(set, binding)`.
/// Unbound type/size variables render as `_` so that two entries each
/// declaring `[]f32` (whose array-length variables differ) compare equal,
/// while `[]f32` vs `[]vec4f32` compare distinct.
fn resource_signature(ty: &Type) -> String {
    match ty {
        Type::Variable(_) => "_".to_string(),
        Type::Constructed(name, args) if args.is_empty() => format!("{:?}", name),
        Type::Constructed(name, args) => {
            let inner: Vec<String> = args.iter().map(resource_signature).collect();
            format!("{:?}({})", name, inner.join(","))
        }
    }
}

/// The element-type signature of a storage/uniform buffer param, used to detect
/// destructive slot sharing. Peels the `*` uniqueness marker and one array
/// layer so the same buffer seen as `[]T` in one entry and `*[]T` (owned) in
/// another compares equal; only the leaf element type (the thing that fixes the
/// SPIR-V struct layout) matters.
fn buffer_element_signature(ty: &Type) -> String {
    use crate::types::TypeName;
    let stripped = ty.clone();
    match &stripped {
        Type::Constructed(TypeName::Array, args) if !args.is_empty() => resource_signature(&args[0]),
        other => resource_signature(other),
    }
}

fn swizzle_letter(idx: u8) -> char {
    match idx {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        3 => 'w',
        _ => '?',
    }
}

/// Render a swizzle component list (`[1, 2]`) as `yz`.
fn format_swizzle_str(components: &[u8]) -> String {
    components.iter().map(|&c| swizzle_letter(c)).collect()
}

/// Trait for generating fresh type variables
pub trait TypeVarGenerator {
    fn new_variable(&mut self) -> Type;
}

// Implement TypeVarGenerator for Context
impl TypeVarGenerator for Context<TypeName> {
    fn new_variable(&mut self) -> Type {
        Context::new_variable(self)
    }
}

/// A warning produced during type checking
#[derive(Debug, Clone)]
pub enum TypeWarning {
    /// A type hole was filled with an inferred type
    TypeHoleFilled {
        inferred_type: Type,
        span: Span,
    },
}

impl TypeWarning {
    /// Get the span for this warning
    pub fn span(&self) -> &Span {
        match self {
            TypeWarning::TypeHoleFilled { span, .. } => span,
        }
    }

    /// Format the warning as a display message
    pub fn message(&self, formatter: &dyn Fn(&Type) -> String) -> String {
        match self {
            TypeWarning::TypeHoleFilled { inferred_type, .. } => {
                format!("Hole of type {}", formatter(inferred_type))
            }
        }
    }
}

/// Per-check resolution context. Names what env stack a lookup sees
/// and at what precedence. Threaded through the checker as
/// `current_context` and set at each top-level entry point
/// (`check_program` → `UserFile`, `check_prelude_functions` →
/// `Prelude`, `check_decl_as_in_module(_, Some(m))` → `Module { m }`).
/// Phase 3 makes `resolve_value_name` consult this enum directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupContext {
    /// Checking a user file-scope `def` body. Sees: locals →
    /// user_file_defs → prelude → builtins.
    UserFile,
    /// Checking an elaborated-module function body. Sees: locals →
    /// module siblings (this module's entries in `module_schemes`) →
    /// user_file_defs → prelude → builtins.
    Module {
        name: String,
    },
    /// Checking a prelude function body. Sees: locals → prelude →
    /// builtins. Does NOT see user-file or user-module entries.
    Prelude,
}

impl LookupContext {
    /// Module name if this context is checking inside a module.
    /// Used as a backwards-compat accessor for `current_module`.
    pub fn module_name(&self) -> Option<&str> {
        match self {
            LookupContext::Module { name } => Some(name.as_str()),
            _ => None,
        }
    }
}

/// Globally-named environments, separated by namespace.
///
/// Phase 1 of the env-split: populated by dual-writes alongside the
/// existing `scope_stack` so we can validate the population logic
/// before any read path depends on it. Lookups still go through
/// `scope_stack` until Phase 3 swaps in the per-context precedence.
///
/// Each map is `StableMap` to preserve insertion order across runs —
/// same reason `module_manager.elaborated_modules` / `prelude_functions`
/// are `StableMap` (`module_manager/mod.rs:55-67`): deterministic
/// type-check order, stable diagnostic output, stable golden
/// downstream. Lookups don't care; iteration does.
#[derive(Debug, Default, Clone)]
pub struct GlobalEnv {
    /// Catalog builtins (`map`, `reduce`, `length`, `f32.sin`, …).
    /// Populated by `load_builtins`.
    pub builtins: StableMap<String, TypeScheme>,
    /// Top-level prelude defs (`unzip`, `rotate`, `iota`, …).
    /// Populated by `check_prelude_functions`.
    pub prelude_defs: StableMap<String, TypeScheme>,
    /// File-scope user `def`s and `entry`s. Populated by
    /// `forward_declare_ascribed_file_scope` (for ascribed defs)
    /// and the main `check_program` walk (for everything).
    pub user_file_defs: StableMap<String, TypeScheme>,
    /// Module function schemes keyed by qualified name
    /// (`"f32.sin"`, `"rand.init"`). Populated by
    /// `check_module_functions` and seeded with `Spec::Sig` /
    /// `Spec::SigOp` schemes from `resolve_placeholders`.
    pub module_schemes: StableMap<String, TypeScheme>,
}

pub struct TypeChecker<'a> {
    scope_stack: ScopeStack<ScopeEntry<TypeScheme>>,
    /// Phase-1 dual-write target — read path still uses
    /// `scope_stack` / `module_schemes` below. Populated by
    /// `define_builtin`, the forward-decl pass, the main loop, and
    /// `check_module_functions` / `check_prelude_functions`. Phase 3
    /// will switch `resolve_value_name` to query this directly.
    pub(super) globals: GlobalEnv,
    /// What context the checker is currently in. Replaces the
    /// short-lived Phase-1 `checking_prelude` bool and supplements
    /// the existing `current_module` field (which Phase 4 collapses
    /// into the `Module` variant). Threaded via save/restore at
    /// `check_decl_as_in_module`'s prologue/epilogue.
    pub(super) current_context: LookupContext,
    pub(super) context: Context<TypeName>, // Polytype unification context
    record_field_map: LookupMap<(String, String), Type>, // Map (type_name, field_name) -> field_type
    module_manager: &'a ModuleManager,     // Lazy module loading
    pub(super) type_table: LookupMap<NodeId, TypeScheme>, // Maps NodeId to type scheme
    warnings: Vec<TypeWarning>,            // Collected warnings
    type_holes: Vec<(NodeId, Span)>,       // Track type hole locations for warning emission
    arity_map: LookupMap<String, usize>,   // function name -> required arity (number of params)
    /// Names of top-level functions that consume an argument — a consuming
    /// function may not be passed as a value, so a call passing one is
    /// rejected.
    consuming_defs: crate::LookupSet<String>,
    /// ID source for generating unique skolem constants when opening existential types.
    skolem_ids: crate::IdSource<SkolemId>,
    /// Current module context for resolving unqualified type aliases in expressions.
    /// Set during check_decl_as_in_module for module function checking.
    pub(super) current_module: Option<String>,
    /// Side table: maps `Identifier` NodeIds to their builtin classification.
    /// Built once before type-check by `name_resolution::build_name_resolution`.
    /// Identifiers absent from this table are resolved via scope/module lookup.
    name_resolution: NameResolution,
    /// First type-alias cycle error encountered during alias resolution.
    /// Recorded under `&self` via interior mutability; surfaced as a fatal
    /// error at the next public entry point (check_program /
    /// check_module_functions / check_prelude_functions).
    pending_cycle_error: std::cell::RefCell<Option<CompilerError>>,
    /// Side table for constructor-call dispatch (`i32(x)`, `f32(true)`,
    /// …). Maps the callee identifier's NodeId to the list of catalog
    /// `BuiltinId`s that were registered as overload candidates, in the
    /// same order as the candidates returned by
    /// `resolve_callee_candidates`. After `resolve_overload` picks a
    /// winner index, the type checker writes the winning `BuiltinId`
    /// back into `name_resolution.values` so downstream consumers see a
    /// resolved per-type conversion catalog entry (e.g. `i32.f32`).
    constructor_call_catalog_ids: LookupMap<NodeId, Vec<BuiltinId>>,
}

/// Compute free type variables in a Type
fn fv_type(ty: &Type) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn go(t: &Type, acc: &mut BTreeSet<usize>) {
        match t {
            Type::Variable(n) => {
                acc.insert(*n);
            }
            Type::Constructed(_, args) => {
                for a in args {
                    go(a, acc);
                }
            }
        }
    }
    go(ty, &mut out);
    out
}

/// Free type variables, skipping any that occur in the size or variant
/// positions of Array types — so let-polymorphism cannot accidentally
/// abstract over representation-critical array metadata. Size and
/// variant are compile-time invariants that must be pinned by
/// unification at the definition, not re-instantiated at every use.
/// Without this, a sized array flowing through a let binding gets its
/// size generalized, every use instantiates a fresh size var,
/// unification pins the per-use var, and the original definition stays
/// unresolved all the way into monomorphization / SPIR-V lowering.
///
/// The **region** slot (the trailing arg) IS generalizable: a view
/// function is `∀r. View[…, r] → …`, polymorphic over which buffer it
/// reads, each call site instantiating `r` to a concrete region. This is
/// exactly the element-type axis's treatment — generalize so the function
/// works on views from any buffer; concrete regions get pinned by
/// unification at each call, and monomorphize specializes per region.
fn fv_type_generalizable(ty: &Type) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn go(t: &Type, acc: &mut BTreeSet<usize>) {
        match t {
            Type::Variable(n) => {
                acc.insert(*n);
            }
            Type::Constructed(TypeName::Array, _) => {
                // The variant tag and dim sizes are compile-time invariants
                // that must be concrete before monomorphization, so they are
                // NOT generalized; the elem type and the region are.
                if let Some(elem) = t.elem_type() {
                    go(elem, acc);
                }
                if let Some(region) = t.array_buffer() {
                    go(region, acc);
                }
            }
            Type::Constructed(_, args) => {
                for a in args {
                    go(a, acc);
                }
            }
        }
    }
    go(ty, &mut out);
    out
}

/// Free type variables for a named function scheme. Function declarations are
/// representation-polymorphic over fixed-size arrays: a helper taking `[4]T`
/// can be specialized for a storage-backed View slice and for a local
/// Composite literal at different call sites.
///
/// Array **sizes** are generalized, so a helper is `∀n. [n]T → …` and each
/// call site instantiates a fresh size var that unification pins to that
/// call's actual length. Without this, one call passing the same array for
/// two params would equate those params' sizes for every other caller too,
/// and the equation would spread through the call graph. Monomorphization
/// keys on the resulting substitution, so a call with `[4]T` and a call with
/// a runtime-sized `[]T` get separate specializations.
///
/// For size-polymorphic/unsized arrays the **variant** stays pinned, so
/// producer-specialization can resolve filter outputs without defaulting a
/// Skolem-sized value to Composite.
fn fv_type_generalizable_for_function(ty: &Type) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn go(t: &Type, acc: &mut BTreeSet<usize>) {
        match t {
            Type::Variable(n) => {
                acc.insert(*n);
            }
            Type::Constructed(TypeName::Array, _) => {
                if let Some(elem) = t.elem_type() {
                    go(elem, acc);
                }
                if matches!(t.array_size(), Some(Type::Constructed(TypeName::Size(_), _))) {
                    if let Some(variant) = t.array_variant() {
                        go(variant, acc);
                    }
                }
                if let Some(size) = t.array_size() {
                    go(size, acc);
                }
                if let Some(region) = t.array_buffer() {
                    go(region, acc);
                }
            }
            Type::Constructed(_, args) => {
                for a in args {
                    go(a, acc);
                }
            }
        }
    }
    go(ty, &mut out);
    out
}

/// Compute free type variables in a TypeScheme
fn fv_scheme(s: &TypeScheme) -> BTreeSet<usize> {
    match s {
        TypeScheme::Monotype(t) => fv_type(t),
        TypeScheme::Polytype { variable, body } => {
            let mut set = fv_scheme(body);
            set.remove(variable);
            set
        }
    }
}

/// Wrap a TypeScheme in nested Polytype quantifiers for the given variables
fn quantify(mut body: TypeScheme, vars: &BTreeSet<usize>) -> TypeScheme {
    // Quantify in descending order so the smallest variable ends up outermost
    for v in vars.iter().rev() {
        body = TypeScheme::Polytype {
            variable: *v,
            body: Box::new(body),
        };
    }
    body
}

/// Represents candidate function types for a callee expression.
/// For overloaded intrinsics, there may be multiple candidates.
struct CalleeCandidates {
    candidates: Vec<Candidate>,
    display_name: String,
}

/// A single candidate function type.
struct Candidate {
    /// Fresh instantiated monotype to unify against
    ty: Type,
}

/// Result of resolving a value name (identifier or qualified name).
/// Centralizes name resolution to avoid precedence drift between different call sites.
struct ResolvedValue {
    /// Display name for error messages
    display_name: String,
    /// Instantiated monotype (with fresh type variables)
    instantiated: Type,
    /// For overloaded intrinsics: all available schemes (for callee resolution)
    overloads: Option<Vec<TypeScheme>>,
}

/// Unified scheme lookup result. All scheme providers (scope, catalog,
/// modules) feed into this.
enum SchemeLookup {
    Single(TypeScheme),
    Overloaded(Vec<TypeScheme>),
}

impl<'a> TypeChecker<'a> {
    /// Try to extract a constant integer value from an expression.
    /// Returns None if the expression is not a constant.
    fn try_extract_const_int(expr: &Expression) -> Option<i32> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => i32::try_from(n).ok(),
            _ => None,
        }
    }

    /// Try to extract constant size from a slice (e.g., `arr[0..4]` → size 4).
    /// Returns None if end is missing or bounds aren't constant.
    // TODO: Currently AST constant folding does not cross function boundaries,
    // so slices with bounds from function parameters won't have constant sizes.
    fn try_extract_slice_size(slice: &SliceExpr) -> Option<usize> {
        let start = match &slice.start {
            Some(s) => Self::try_extract_const_int(s)?,
            None => 0,
        };
        let end = Self::try_extract_const_int(slice.end.as_ref()?)?;
        if end >= start {
            Some((end - start) as usize)
        } else {
            None
        }
    }

    /// Constrain the address space of an array type to Storage.
    /// Used for entry point parameters where []f32 means storage buffer.
    /// Sized arrays (e.g. [19]u32) stay Composite — only unsized arrays become View.
    fn constrain_array_to_storage(&mut self, ty: &Type) -> Result<()> {
        let resolved = ty.apply(&self.context);

        if !resolved.is_array() {
            return Ok(());
        }

        let size = resolved.array_size().expect("Array has size");
        let variant = resolved.array_variant().expect("Array has variant");
        let elem = resolved.elem_type().expect("Array has elem");

        // Check if this is a sized array (Size constant)
        let is_sized = matches!(size.apply(&self.context), Type::Constructed(TypeName::Size(_), _));

        if is_sized {
            // Sized arrays → Composite (actual data, passed as push constants)
            let composite = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
            self.context
                .unify(variant, &composite)
                .map_err(|_| err_type!("Sized entry point array must be Composite"))?;
        } else {
            // Unsized arrays → View (storage buffer)
            let storage = Type::Constructed(TypeName::ArrayVariantView, vec![]);
            self.context
                .unify(variant, &storage)
                .map_err(|_| err_type!("Entry point array parameter must have Storage address space"))?;
        }

        // Recursively constrain nested arrays in element type
        self.constrain_array_to_storage(elem)?;
        Ok(())
    }

    /// Format a type for error messages by applying current substitution
    pub fn format_type(&self, ty: &Type) -> String {
        let applied = ty.apply(&self.context);
        match &applied {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
                // Special case for arrow types
                format!("{} -> {}", self.format_type(&args[0]), self.format_type(&args[1]))
            }
            Type::Constructed(TypeName::Tuple(_), args) => {
                // Special case for tuple types
                let arg_strs: Vec<String> = args.iter().map(|a| self.format_type(a)).collect();
                format!("({})", arg_strs.join(", "))
            }
            Type::Constructed(name, args) if args.is_empty() => format!("{}", name),
            Type::Constructed(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.format_type(a)).collect();
                format!("{}[{}]", name, arg_strs.join(", "))
            }
            Type::Variable(id) => format!("?{}", id),
        }
    }

    /// Format a type scheme for error messages
    pub fn format_scheme(&self, scheme: &TypeScheme) -> String {
        match scheme {
            TypeScheme::Monotype(ty) => self.format_type(ty),
            TypeScheme::Polytype { variable, body } => {
                // For display, we can show quantified vars or just the body
                // For now, just show the body
                format!("∀{}. {}", variable, self.format_scheme(body))
            }
        }
    }

    /// Check if a type contains an existential quantifier.
    /// Existential types are only valid in function return types, not parameters.
    fn contains_existential(ty: &Type) -> bool {
        match ty {
            Type::Constructed(TypeName::Existential(_), _) => true,
            Type::Constructed(_, args) => args.iter().any(Self::contains_existential),
            Type::Variable(_) => false,
        }
    }

    /// Check if a type is numeric (Int, UInt, Float, or Vec with numeric elements).
    /// Returns None if the type is a variable (not yet resolved).
    fn is_numeric_type(ty: &Type) -> Option<bool> {
        match ty {
            Type::Constructed(TypeName::Int(_), _) => Some(true),
            Type::Constructed(TypeName::UInt(_), _) => Some(true),
            Type::Constructed(TypeName::Float(_), _) => Some(true),
            // Vec types are numeric if their element type is numeric
            _ if ty.is_vec() => Self::is_numeric_type(ty.elem_type().expect("Vec has elem")),
            Type::Variable(_) => None,
            _ => Some(false),
        }
    }

    /// Open an existential type: ?k. T becomes T[#n/k] where #n is a fresh skolem.
    /// This is called when binding an existential value to a variable.
    /// Skolems are rigid constants that only unify with themselves, enforcing opacity.
    fn open_existential(&mut self, ty: Type) -> Type {
        match ty {
            Type::Constructed(TypeName::Existential(vars), args) if !args.is_empty() => {
                let mut inner = args.into_iter().next().unwrap();
                // Substitute each bound size variable with a fresh skolem constant
                for var_name in vars {
                    let skolem_id = self.skolem_ids.next_id();
                    let skolem = Type::Constructed(TypeName::Skolem(skolem_id), vec![]);
                    inner = Self::substitute_size_var(&inner, &var_name, &skolem);
                }
                inner
            }
            _ => ty,
        }
    }

    /// Instantiate an existential for *introduction* (packing): `?k. T` becomes
    /// `T[?α/k]` where each bound size variable is replaced by a fresh
    /// unification variable. This is the dual of `open_existential`: opening
    /// (elimination, at a let-binding) uses rigid skolems so the size stays
    /// opaque, whereas packing (a return position) uses fresh variables that
    /// unification solves to the witness size the body actually produces.
    ///
    /// Returning a `filter` result that also flows through `length` is the
    /// motivating case: the body's array carries a rigid skolem from its
    /// let-opening, and packing the declared `?k.` return to a fresh variable
    /// lets that variable solve to the skolem. Non-existential types pass
    /// through unchanged.
    ///
    /// Recurses into type components, so an existential nested in a position
    /// other than the top — e.g. a per-component `(?k. [k]u32, [1]u32)` tuple
    /// return — is packed too. Each existential is instantiated independently,
    /// so distinct components reusing the same bound name (`?k`) get distinct
    /// fresh variables.
    fn instantiate_existential(&mut self, ty: Type) -> Type {
        match ty {
            Type::Constructed(TypeName::Existential(vars), args) if !args.is_empty() => {
                let mut inner = args.into_iter().next().unwrap();
                for var_name in vars {
                    let fresh = self.context.new_variable();
                    inner = Self::substitute_size_var(&inner, &var_name, &fresh);
                }
                // The instantiated body may itself hold further existentials.
                self.instantiate_existential(inner)
            }
            Type::Constructed(name, args) => Type::Constructed(
                name,
                args.into_iter().map(|a| self.instantiate_existential(a)).collect(),
            ),
            Type::Variable(_) => ty,
        }
    }

    /// Substitute a size variable name with a type in a type expression.
    fn substitute_size_var(ty: &Type, var_name: &str, replacement: &Type) -> Type {
        match ty {
            Type::Variable(v) => Type::Variable(*v),
            Type::Constructed(TypeName::SizeVar(name), _) if name == var_name => replacement.clone(),
            Type::Constructed(name, args) => {
                let new_args: Vec<Type> =
                    args.iter().map(|arg| Self::substitute_size_var(arg, var_name, replacement)).collect();
                Type::Constructed(name.clone(), new_args)
            }
        }
    }

    /// Resolve type aliases in a type with optional module context.
    /// - Qualified names (containing '.') are looked up as-is
    /// - Unqualified names are qualified with current_module if provided
    /// - Recursively resolves nested aliases
    fn resolve_type_aliases_scoped(&self, ty: &Type, current_module: Option<&str>) -> Type {
        let mut visited = Vec::new();
        self.resolve_type_aliases_impl(ty, current_module, &mut visited).unwrap_or_else(|cycle_err| {
            // Stash the first cycle error; the next public entry point
            // (check_program / check_module_functions /
            // check_prelude_functions) drains it and bails. Returning the
            // unresolved type lets local checking continue without
            // cascading panics.
            let mut slot = self.pending_cycle_error.borrow_mut();
            if slot.is_none() {
                *slot = Some(cycle_err);
            }
            ty.clone()
        })
    }

    /// Drain the pending type-alias-cycle error, if any. Public entry
    /// points must call this and surface the error before returning Ok.
    fn take_pending_cycle_error(&self) -> Option<CompilerError> {
        self.pending_cycle_error.borrow_mut().take()
    }

    fn resolve_type_aliases_impl(
        &self,
        ty: &Type,
        current_module: Option<&str>,
        visited: &mut Vec<String>,
    ) -> Result<Type> {
        match ty {
            Type::Constructed(TypeName::Named(name), args) => {
                // First resolve args recursively
                let resolved_args: Result<Vec<Type>> = args
                    .iter()
                    .map(|a| self.resolve_type_aliases_impl(a, current_module, visited))
                    .collect();
                let resolved_args = resolved_args?;

                // Build lookup keys based on qualification
                let mut keys = Vec::new();
                if name.contains('.') {
                    // Already qualified - try as-is
                    keys.push(name.clone());
                } else {
                    if let Some(m) = current_module {
                        keys.push(format!("{}.{}", m, name));
                    }
                    keys.push(name.clone());
                }

                for key in keys {
                    // Check for cycles before resolving
                    if let Some(cycle_err) = Self::check_alias_cycle(visited, &key) {
                        return Err(cycle_err);
                    }

                    if let Some(underlying) = self.module_manager.resolve_type_alias(&key) {
                        // Track this alias as visited
                        visited.push(key);
                        // Recursively resolve in same module context
                        let result = self.resolve_type_aliases_impl(underlying, current_module, visited);
                        visited.pop();
                        return result;
                    }
                }

                // Not an alias - keep as-is with resolved args
                Ok(Type::Constructed(TypeName::Named(name.clone()), resolved_args))
            }
            Type::Constructed(name, args) => {
                // Non-Named constructor - just resolve args
                let resolved_args: Result<Vec<Type>> = args
                    .iter()
                    .map(|a| self.resolve_type_aliases_impl(a, current_module, visited))
                    .collect();
                Ok(Type::Constructed(name.clone(), resolved_args?))
            }
            Type::Variable(id) => Ok(Type::Variable(*id)),
        }
    }

    /// Check if adding `key` to visited would create a cycle.
    /// Returns an error if a cycle is detected.
    fn check_alias_cycle(visited: &[String], key: &str) -> Option<CompilerError> {
        if visited.contains(&key.to_string()) {
            let mut cycle_path = visited.to_vec();
            cycle_path.push(key.to_string());
            Some(err_type_at!(
                Span::new(0, 0, 0, 0),
                "type alias cycle detected: {}",
                cycle_path.join(" -> ")
            ))
        } else {
            None
        }
    }

    /// Insert a name into the scope with a given identifier kind.
    pub(super) fn define(&mut self, name: String, kind: IdentifierKind, value: TypeScheme) {
        self.scope_stack.insert(name, ScopeEntry { kind, value });
    }

    /// Look up a variable in the scope stack (for testing)
    pub fn lookup(&self, name: &str) -> Option<TypeScheme> {
        self.scope_stack.lookup(name).map(|e| e.value.clone())
    }

    /// Get a reference to the context (for testing)
    pub fn context(&self) -> &Context<TypeName> {
        &self.context
    }

    /// Compute all free type variables in the current environment (scope stack)
    ///
    /// Each scheme is solved against the context first. `generalize` takes
    /// `fv(ty) \ fv(env)` on a `ty` that has already been solved, so an env
    /// scheme still spelled with a variable that unification has since merged
    /// away would name a different variable than the same slot in `ty`, and the
    /// set difference would fail to remove it. Generalizing a variable that is
    /// in fact still reachable from the environment makes every use of the
    /// binding instantiate a fresh copy of it.
    fn env_free_type_vars(&self) -> BTreeSet<usize> {
        let mut acc = BTreeSet::new();
        // Local bindings (lambda params, let/loop/match locals).
        self.scope_stack.for_each_binding(|_name, entry| {
            acc.extend(fv_scheme(&self.apply_scheme(&entry.value)));
        });
        // Globally-named environments — must be included so HM
        // generalization correctly excludes vars free in the
        // surrounding env. `builtins` schemes are closed
        // (`generalize_closed`) so they contribute nothing, but
        // `prelude_defs` / `user_file_defs` / `module_schemes`
        // were generalized at insertion time against a possibly-open
        // env and can still carry free vars.
        for scheme in self.globals.builtins.values() {
            acc.extend(fv_scheme(scheme));
        }
        for scheme in self.globals.prelude_defs.values() {
            acc.extend(fv_scheme(scheme));
        }
        for scheme in self.globals.user_file_defs.values() {
            acc.extend(fv_scheme(scheme));
        }
        for scheme in self.globals.module_schemes.values() {
            acc.extend(fv_scheme(scheme));
        }
        acc
    }

    /// HM-style generalization at let: ∀(fv(ty) \ fv(env) \ ascription_vars). ty
    /// Quantifies over type variables that are free in ty but not free in the environment
    /// and not in the set of ascription variables (which must remain monomorphic)
    pub(super) fn generalize(&self, ty: &Type) -> TypeScheme {
        // Always generalize the *solved* view
        let applied = ty.apply(&self.context);

        // Assert no unsubstituted UserVar/SizeVar remain - these should have been
        // substituted with type Variables before generalization
        debug_assert!(
            !Self::contains_user_or_size_var(&applied),
            "Type contains unsubstituted UserVar/SizeVar before generalization: {:?}",
            applied
        );

        // Free vars in type — but skip size / variant positions of Array
        // types. Those are compile-time invariants that must be pinned by
        // unification, not let-generalized (see `fv_type_generalizable`).
        let mut fv_ty = fv_type_generalizable(&applied);

        // Free vars in environment
        let fv_env = self.env_free_type_vars();

        // vars to quantify = fv(ty) \ fv(env)
        for v in fv_env {
            fv_ty.remove(&v);
        }

        // Wrap in nested Polytype quantifiers
        quantify(TypeScheme::Monotype(applied), &fv_ty)
    }

    fn generalize_function(&self, ty: &Type) -> TypeScheme {
        let applied = ty.apply(&self.context);

        debug_assert!(
            !Self::contains_user_or_size_var(&applied),
            "Type contains unsubstituted UserVar/SizeVar before function generalization: {:?}",
            applied
        );

        let mut fv_ty = fv_type_generalizable_for_function(&applied);
        let fv_env = self.env_free_type_vars();
        for v in fv_env {
            fv_ty.remove(&v);
        }

        quantify(TypeScheme::Monotype(applied), &fv_ty)
    }

    /// Check if a type contains any unsubstituted UserVar or SizeVar.
    /// These should be substituted with proper type Variables before generalization.
    /// Skips checking inside Existential types, as those intentionally contain SizeVar
    /// for existentially quantified sizes.
    fn contains_user_or_size_var(ty: &Type) -> bool {
        match ty {
            Type::Constructed(TypeName::UserVar(_), _) => true,
            Type::Constructed(TypeName::SizeVar(_), _) => true,
            // Skip checking inside Existential - those intentionally use SizeVar
            Type::Constructed(TypeName::Existential(_), _) => false,
            Type::Constructed(_, args) => args.iter().any(Self::contains_user_or_size_var),
            Type::Variable(_) => false,
        }
    }

    /// Unified name resolution for identifiers and qualified names.
    ///
    /// Precedence: locals/top-level shadow builtins (`def length = ...`
    /// wins over the catalog entry). The pre-typecheck pass that builds
    /// `name_resolution` already enforces this by skipping shadowed
    /// names; entries present in the side table are guaranteed to refer
    /// to a catalog builtin. Identifiers absent from the side table fall
    /// through to scope/module lookup.
    ///
    /// `node_id` is the AST id of the Identifier expression being
    /// resolved. After Phase 4, callers must always supply a real
    /// NodeId — the `Option<NodeId>` branch (and the synthetic
    /// FieldAccess recovery) is gone.
    fn resolve_value_name(
        &mut self,
        full_name: &str,
        is_qualified: bool,
        node_id: NodeId,
    ) -> Option<ResolvedValue> {
        // Path A: NameResolution side table covers every catalog
        // identifier (Phases 1 + 3.5 + the prelude-walk in
        // `name_resolution::build_name_resolution`).
        if let Some(crate::name_resolution::ResolvedValueRef::Builtin { id: builtin_id, .. }) =
            self.name_resolution.get(node_id)
        {
            let bid = *builtin_id;
            let lookup = self.scheme_lookup_for_builtin(bid);
            let def_name = by_id(bid).raw.surface_name;
            return Some(self.resolve_scheme_lookup(def_name, lookup));
        }

        // Path A also fires for SOAC-tagged identifiers — go straight
        // to the catalog's builtin scheme so a user-defined shadow at
        // file scope can't reach the prelude. Equivalent to the
        // previous `scope_stack.lookup_by_kind(_, Builtin)` hack, but
        // sourced from the canonical `globals.builtins` map.
        if let Some(crate::name_resolution::ResolvedValueRef::Soac(_)) = self.name_resolution.get(node_id) {
            if let Some(scheme) = self.globals.builtins.get(full_name) {
                return Some(self.resolve_scheme_lookup(full_name, SchemeLookup::Single(scheme.clone())));
            }
        }

        // Path B: non-catalog names.
        // Qualified names go directly to module_schemes.
        if is_qualified {
            return self.lookup_module_scheme(full_name).map(|s| self.resolve_scheme_lookup(full_name, s));
        }

        // Unqualified: locals (lambda/let/match/loop bindings) first,
        // then the per-context global precedence rules.
        if let Some(scheme) = self.scope_stack.lookup_by_kind(full_name, IdentifierKind::Local) {
            return Some(self.resolve_scheme_lookup(full_name, SchemeLookup::Single(scheme.clone())));
        }

        let scheme = self.globals_lookup(full_name)?;
        Some(self.resolve_scheme_lookup(full_name, SchemeLookup::Single(scheme)))
    }

    /// Per-context unqualified-name lookup over the global envs.
    /// Precedence per `LookupContext`:
    ///   * `UserFile`: user_file_defs → prelude → builtins.
    ///   * `Module { name }`: module siblings (`{name}.{n}` key in
    ///     `module_schemes`) → user_file_defs → prelude → builtins.
    ///     User-defined modules close over file scope.
    ///   * `Prelude`: prelude → builtins. Does NOT see user.
    fn globals_lookup(&self, name: &str) -> Option<TypeScheme> {
        match &self.current_context {
            LookupContext::UserFile => self
                .globals
                .user_file_defs
                .get(name)
                .or_else(|| self.globals.prelude_defs.get(name))
                .or_else(|| self.globals.builtins.get(name))
                .cloned(),
            LookupContext::Module { name: mod_name } => {
                let qualified = format!("{}.{}", mod_name, name);
                self.globals
                    .module_schemes
                    .get(&qualified)
                    .or_else(|| self.globals.user_file_defs.get(name))
                    .or_else(|| self.globals.prelude_defs.get(name))
                    .or_else(|| self.globals.builtins.get(name))
                    .cloned()
            }
            LookupContext::Prelude => {
                self.globals.prelude_defs.get(name).or_else(|| self.globals.builtins.get(name)).cloned()
            }
        }
    }

    /// Look up the prelude-module-supplied scheme for a non-catalog
    /// qualified name (e.g. `materials.pbrCookTorrance`). Used by
    /// `resolve_value_name` Path B.
    fn lookup_module_scheme(&self, qualified_name: &str) -> Option<SchemeLookup> {
        self.globals.module_schemes.get(qualified_name).cloned().map(SchemeLookup::Single)
    }

    /// Look up the prelude-module-supplied scheme for a catalog
    /// per-type op by `BuiltinId`. The catalog stores the canonical
    /// surface_name on the `BuiltinDef`; we route through that to
    /// `globals.module_schemes`. This is the BuiltinId-keyed
    /// counterpart to `lookup_module_scheme(qualified_name)`.
    fn lookup_module_scheme_by_id(&self, id: BuiltinId) -> Option<TypeScheme> {
        let surface_name = by_id(id).raw.surface_name;
        self.globals.module_schemes.get(surface_name).cloned()
    }

    /// Build a `SchemeLookup` from a `BuiltinId` by invoking each
    /// overload's scheme builder against the current context.
    ///
    /// Overloads whose scheme is `None` (per-type ops `f32.add`,
    /// `f32.i32`, …) get their schemes from prelude module signatures
    /// via `lookup_module_scheme_by_id`.
    fn scheme_lookup_for_builtin(&mut self, id: BuiltinId) -> SchemeLookup {
        let catalog = crate::builtins::catalog();
        let def = catalog.get(id);
        let schemes: Vec<TypeScheme> = (0..def.overloads().len())
            .map(|i| {
                catalog.build_scheme(id, i, &mut self.context).unwrap_or_else(|| {
                    self.lookup_module_scheme_by_id(id).unwrap_or_else(|| {
                        panic!(
                            "BUG: builtin {:?} (`{}`) has no scheme builder and no module_schemes entry",
                            id, def.raw.surface_name
                        )
                    })
                })
            })
            .collect();
        if schemes.len() == 1 {
            SchemeLookup::Single(schemes.into_iter().next().unwrap())
        } else {
            SchemeLookup::Overloaded(schemes)
        }
    }

    fn resolve_scheme_lookup(&mut self, name: &str, lookup: SchemeLookup) -> ResolvedValue {
        match lookup {
            SchemeLookup::Single(scheme) => {
                let ty = scheme.instantiate(&mut self.context);
                ResolvedValue {
                    display_name: name.to_string(),
                    instantiated: ty,
                    overloads: None,
                }
            }
            SchemeLookup::Overloaded(schemes) => {
                let ty = self.context.new_variable();
                ResolvedValue {
                    display_name: name.to_string(),
                    instantiated: ty,
                    overloads: Some(schemes),
                }
            }
        }
    }

    /// Create a new TypeChecker with a reference to a ModuleManager
    pub fn new(module_manager: &'a ModuleManager) -> Self {
        Self::with_type_table(module_manager, LookupMap::new())
    }

    /// Create a TypeChecker with an empty type table (for building prelude)
    pub fn new_empty(module_manager: &'a ModuleManager) -> Self {
        Self::with_type_table(module_manager, LookupMap::new())
    }

    /// Create a TypeChecker with an existing Context and spec_schemes (from resolve_placeholders pass).
    pub fn with_context_and_schemes(
        module_manager: &'a ModuleManager,
        context: Context<TypeName>,
        spec_schemes: LookupMap<String, TypeScheme>,
    ) -> Self {
        Self::with_context_and_type_table(module_manager, context, LookupMap::new(), spec_schemes)
    }

    /// Create a TypeChecker with a given initial type table
    fn with_type_table(
        module_manager: &'a ModuleManager,
        type_table: LookupMap<NodeId, TypeScheme>,
    ) -> Self {
        Self::with_context_and_type_table(module_manager, Context::default(), type_table, LookupMap::new())
    }

    /// Create a TypeChecker with both an existing Context and type table.
    fn with_context_and_type_table(
        module_manager: &'a ModuleManager,
        context: Context<TypeName>,
        type_table: LookupMap<NodeId, TypeScheme>,
        spec_schemes: LookupMap<String, TypeScheme>,
    ) -> Self {
        let mut globals = GlobalEnv::default();
        // Seed `globals.module_schemes` from the spec_schemes computed
        // by `resolve_placeholders`. Sort the LookupMap entries for
        // stable iteration order downstream.
        let mut spec_seeded: Vec<_> = spec_schemes.into_iter().collect();
        spec_seeded.sort_by(|a, b| a.0.cmp(&b.0));
        for (k, v) in spec_seeded {
            globals.module_schemes.insert(k, v);
        }
        TypeChecker {
            scope_stack: ScopeStack::new(),
            globals,
            current_context: LookupContext::UserFile,
            context,
            record_field_map: LookupMap::new(),
            module_manager,
            type_table,
            warnings: Vec::new(),
            type_holes: Vec::new(),
            arity_map: LookupMap::new(),
            consuming_defs: crate::LookupSet::new(),
            skolem_ids: crate::IdSource::new(),
            current_module: None,
            name_resolution: NameResolution::default(),
            pending_cycle_error: std::cell::RefCell::new(None),
            constructor_call_catalog_ids: LookupMap::new(),
        }
    }

    /// Inject the side table populated by `build_name_resolution`. Must
    /// be called between construction and `check_program` for any
    /// program that uses builtin identifiers.
    pub fn set_name_resolution(&mut self, nr: NameResolution) {
        self.name_resolution = nr;
    }

    /// Borrow the (post-inference) NameResolution. The checker writes
    /// overload-resolution results into it during inference; downstream
    /// IR (TLC) reads them.
    pub fn name_resolution(&self) -> &NameResolution {
        &self.name_resolution
    }

    /// Get all warnings collected during type checking
    pub fn warnings(&self) -> &[TypeWarning] {
        &self.warnings
    }

    // `fresh_type_for_pattern` and `bind_pattern` live in
    // `super::patterns::bind` (extends this type via an `impl` block
    // there). Pattern logic is its own subsystem.

    /// Check an expression against an expected type (bidirectional checking mode)
    /// Returns the actual type (which should unify with expected_type)
    fn check_expression(&mut self, expr: &Expression, expected_type: &Type) -> Result<Type> {
        match &expr.kind {
            // Integer literals can be checked against any integer type
            ExprKind::IntLiteral(_) => {
                // Accept the expected type if it's an integer type
                let applied = expected_type.apply(&self.context);
                match &applied {
                    Type::Constructed(TypeName::Int(_), _) | Type::Constructed(TypeName::UInt(_), _) => {
                        // Store the type in the type table
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(expected_type.clone()));
                        Ok(expected_type.clone())
                    }
                    _ => {
                        // Fall back to inference (will produce i32)
                        let inferred = self.infer_expression(expr)?;
                        self.context.unify(&inferred, expected_type).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "Expected {}, got {}",
                                self.format_type(expected_type),
                                self.format_type(&inferred)
                            )
                        })?;
                        Ok(inferred)
                    }
                }
            }
            // Float literals can be checked against any float type
            ExprKind::FloatLiteral(_) => {
                let applied = expected_type.apply(&self.context);
                match &applied {
                    Type::Constructed(TypeName::Float(_), _) => {
                        // Store the type in the type table
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(expected_type.clone()));
                        Ok(expected_type.clone())
                    }
                    _ => {
                        let inferred = self.infer_expression(expr)?;
                        self.context.unify(&inferred, expected_type).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "Expected {}, got {}",
                                self.format_type(expected_type),
                                self.format_type(&inferred)
                            )
                        })?;
                        Ok(inferred)
                    }
                }
            }
            ExprKind::Lambda(lambda) => self.type_lambda(lambda, Some(expected_type), expr),
            // Sum-type constructor application uses the expected type to
            // resolve which sum type the constructor belongs to.
            // Bare-inference (`infer_expression`) can't disambiguate.
            ExprKind::Constructor(name, args) => {
                let applied = expected_type.apply(&self.context);
                let variants = match &applied {
                    Type::Constructed(TypeName::Sum(variants), _) => variants,
                    _ => bail_type_at!(
                        expr.h.span,
                        "constructor `#{}` requires a sum type from context, but expected {}",
                        name,
                        self.format_type(&applied)
                    ),
                };
                let payload_types = match variants.iter().find(|(n, _)| n == name) {
                    Some((_, payload)) => payload,
                    None => bail_type_at!(
                        expr.h.span,
                        "constructor `#{}` not found in sum type {}",
                        name,
                        self.format_type(&applied)
                    ),
                };
                if args.len() != payload_types.len() {
                    bail_type_at!(
                        expr.h.span,
                        "constructor `#{}` expects {} payload value{}, got {}",
                        name,
                        payload_types.len(),
                        if payload_types.len() == 1 { "" } else { "s" },
                        args.len()
                    );
                }
                for (arg, expected_arg_ty) in args.iter().zip(payload_types.iter()) {
                    self.check_expression(arg, expected_arg_ty)?;
                }
                self.type_table.insert(expr.h.id, TypeScheme::Monotype(expected_type.clone()));
                Ok(expected_type.clone())
            }
            _ => {
                // For non-lambdas, infer and check the inferred value
                // against the expected contract.  This is directional:
                // an alias-free `*T` value may be observed as `T`, but an
                // observing `T` value cannot be promoted to `*T`.
                let actual_type = self.infer_expression(expr)?;
                self.unify_or_err_weakening(&actual_type, expected_type, expr.h.span, "Type mismatch")?;
                Ok(actual_type)
            }
        }
    }

    /// Resolve type aliases in a type annotation. SizeVar/UserVar
    /// substitution is handled upstream by `resolve_placeholders`.
    pub(super) fn normalize_annotation_type(&self, ty: &Type, module: Option<&str>) -> Type {
        self.resolve_type_aliases_scoped(ty, module)
    }

    /// Expand aliases in an annotation *and* instantiate it: replace the
    /// parser's residual `AddressPlaceholder` / `SizePlaceholder` sentinels
    /// (array variant / region / size slots) with fresh type variables, so the
    /// annotation is variant/region/size-polymorphic at this use site.
    ///
    /// `resolve_placeholders` already does this for inline `def`/`entry`
    /// annotations, but it skips `type`-alias bodies, so an alias field like
    /// `points: []vec2f32` keeps rigid `AddressPlaceholder`s that can't unify
    /// with a concrete `view` / `composite` / `no_buffer`. Freshening on each
    /// expansion treats the alias as a polymorphic scheme instantiated per use.
    ///
    /// Unlike `resolve_placeholders::resolve_type`, sizes are *not* linked here
    /// — each `[]` slot gets an independent var — because a record/tuple of
    /// arrays (`{ points: [], items: [] }`) generally spans distinct domains and
    /// must not be forced to a single length.
    pub(super) fn instantiate_annotation_type(&mut self, ty: &Type, module: Option<&str>) -> Type {
        let expanded = self.resolve_type_aliases_scoped(ty, module);
        self.freshen_placeholders(&expanded)
    }

    /// Structural walk replacing each residual array `AddressPlaceholder` /
    /// `SizePlaceholder` with an independent fresh variable. A no-op on
    /// annotations that already went through `resolve_placeholders` (they carry
    /// no placeholders), so it only rewrites alias-expanded slots.
    fn freshen_placeholders(&mut self, ty: &Type) -> Type {
        match ty {
            Type::Constructed(TypeName::AddressPlaceholder, _)
            | Type::Constructed(TypeName::SizePlaceholder, _) => self.context.new_variable(),
            // Sum carries its payload types inside the `TypeName`, not `args`.
            Type::Constructed(TypeName::Sum(variants), args) => {
                let variants = variants
                    .iter()
                    .map(|(ctor, payload)| {
                        (
                            ctor.clone(),
                            payload.iter().map(|p| self.freshen_placeholders(p)).collect(),
                        )
                    })
                    .collect();
                Type::Constructed(TypeName::Sum(variants), args.clone())
            }
            Type::Constructed(name, args) => Type::Constructed(
                name.clone(),
                args.iter().map(|a| self.freshen_placeholders(a)).collect(),
            ),
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Allocate a fresh type variable and return its ID.
    fn fresh_var(&mut self) -> polytype::Variable {
        match self.context.new_variable() {
            Type::Variable(id) => id,
            _ => unreachable!(),
        }
    }

    /// Convert a variable ID to a Type.
    fn var(id: polytype::Variable) -> Type {
        Type::Variable(id)
    }

    /// Build a chain of arrows: p1 -> p2 -> ... -> ret
    fn arrow_chain(params: &[Type], ret: Type) -> Type {
        params.iter().rev().fold(ret, |acc, param| Type::arrow(param.clone(), acc))
    }

    /// Wrap a type in nested ∀ quantifiers.
    fn forall(ids: &[polytype::Variable], body: Type) -> TypeScheme {
        ids.iter().rev().fold(TypeScheme::Monotype(body), |acc, &id| TypeScheme::Polytype {
            variable: id,
            body: Box::new(acc),
        })
    }

    /// Apply the checker's substitution context to the innermost
    /// `Monotype` of a scheme, preserving the surrounding `Polytype`
    /// shells. `forall` builds one shell per quantified variable, so a
    /// scheme with N type variables walks through N `Polytype` cases
    /// before reaching the `Monotype`.
    fn apply_scheme(&self, scheme: &TypeScheme) -> TypeScheme {
        match scheme {
            TypeScheme::Monotype(ty) => TypeScheme::Monotype(ty.apply(&self.context)),
            TypeScheme::Polytype { variable, body } => TypeScheme::Polytype {
                variable: *variable,
                body: Box::new(self.apply_scheme(body)),
            },
        }
    }

    /// Walk through any number of `Polytype` shells to the innermost
    /// `Monotype` and return it with `self.context` applied.
    fn scheme_inner(&self, scheme: &TypeScheme) -> Type {
        match scheme {
            TypeScheme::Monotype(ty) => ty.apply(&self.context),
            TypeScheme::Polytype { body, .. } => self.scheme_inner(body),
        }
    }

    /// Build an array type `Array[elem, variant, size, region]` for a builtin
    /// scheme, allocating a fresh buffer variable so the array is polymorphic
    /// over *which buffer* it views — a view function is `∀r. View[…,r] → …`,
    /// each call site instantiating `r` to a concrete region. The variant slot
    /// is likewise a variable (the array may be a view or a composite); the
    /// region var is generalized alongside it (see `generalize_closed`).
    fn array_ty(&mut self, elem: Type, addrspace: polytype::Variable, size: polytype::Variable) -> Type {
        let region = self.fresh_var();
        Type::Constructed(
            TypeName::Array,
            vec![elem, Self::var(addrspace), Self::var(size), Self::var(region)],
        )
    }

    /// Quantify a closed builtin body over *all* its free type variables.
    /// Every variable in a builtin scheme is meant to be polymorphic, so this
    /// equals listing them explicitly — and, unlike a hand-written `forall`
    /// list, it cannot forget the buffer variable `array_ty` introduces for
    /// each array.
    fn generalize_closed(body: Type) -> TypeScheme {
        let vars = fv_type(&body);
        quantify(TypeScheme::Monotype(body), &vars)
    }

    /// Register zipN: ∀n a1..aN s. [a1,s,n] -> ... -> [aN,s,n] -> [(a1,...,aN),s,n]
    fn register_zip_n(&mut self, arity: usize) {
        let n = self.fresh_var();
        let s = self.fresh_var();
        let elem_vars: Vec<polytype::Variable> = (0..arity).map(|_| self.fresh_var()).collect();

        let params: Vec<Type> = elem_vars.iter().map(|&v| self.array_ty(Self::var(v), s, n)).collect();
        let tuple_ty = Type::Constructed(
            TypeName::Tuple(arity),
            elem_vars.iter().map(|&v| Self::var(v)).collect(),
        );
        let ret = self.array_ty(tuple_ty, s, n);
        let body = Self::arrow_chain(&params, ret);

        let name = format!("zip{}", arity);
        let scheme = Self::generalize_closed(body);
        // Route through `define_builtin` so the dual-write hits
        // `globals.builtins` — direct `define` here skipped Phase 1's
        // mirror and made `zipN` lookups in Path A's Soac branch
        // miss after Phase 3's switch to `globals.builtins`.
        self.define_builtin(&name, scheme);
    }

    /// Build a Vec type: Vec[elem, size]
    fn vec_ty(n: polytype::Variable, elem: Type) -> Type {
        Type::Constructed(TypeName::Vec, vec![elem, Self::var(n)])
    }

    /// Type a lambda expression, optionally with an expected type for bidirectional checking.
    fn type_lambda(
        &mut self,
        lambda: &crate::ast::LambdaExpr,
        expected: Option<&Type>,
        expr: &crate::ast::Expression,
    ) -> Result<Type> {
        // Extract expected parameter types if we have an expected function type
        let expected_param_types: Option<Vec<Type>> = expected.and_then(|exp| {
            let mut result = Vec::new();
            let mut ty = exp.clone();
            for _ in 0..lambda.params.len() {
                let applied = ty.apply(&self.context);
                if let Some((param_type, result_type)) = as_arrow(&applied) {
                    result.push(param_type.clone());
                    ty = result_type.clone();
                } else {
                    return None; // Expected type doesn't match lambda structure
                }
            }
            Some(result)
        });

        self.scope_stack.push_scope();

        // Determine parameter types and bind them
        let mut param_types = Vec::new();
        for (i, param) in lambda.params.iter().enumerate() {
            let param_type = if let Some(annotated_type) = param.pattern_type() {
                // Explicit annotation takes precedence
                self.normalize_annotation_type(annotated_type, self.current_module.as_deref())
            } else if let Some(ref expected_params) = expected_param_types {
                // Use expected type from bidirectional checking
                expected_params[i].clone()
            } else {
                // No annotation and no expected type - create fresh type variable
                self.fresh_type_for_pattern(param)
            };

            param_types.push(param_type.clone());
            // Bind the parameter as an observation (no `*`); the lambda's
            // own arrow type keeps the consumption contract.
            self.bind_irrefutable_pattern(param, &param_type, false)?;
        }

        // Type check the body
        let body_type = self.infer_expression(&lambda.body)?;

        self.scope_stack.pop_scope();

        // Build the function type
        let func_type = Self::arrow_chain(&param_types, body_type);

        // If we had an expected type, unify with it
        if let Some(exp) = expected {
            self.unify_or_err_weakening(&func_type, exp, expr.h.span, "Lambda type mismatch")?;
            // Store in type table when checking against expected type
            self.type_table.insert(expr.h.id, TypeScheme::Monotype(func_type.clone()));
        }

        Ok(func_type)
    }

    fn define_builtin(&mut self, name: &str, scheme: TypeScheme) {
        self.globals.builtins.insert(name.to_string(), scheme);
    }

    pub fn load_builtins(&mut self) -> Result<()> {
        // length: ∀n a s. Array[a, s, n] -> i32
        let (n, a, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[self.array_ty(Self::var(a), s, n)], i32());
        self.define_builtin("length", Self::generalize_closed(body));

        // map: ∀a b n s. (a -> b) -> Array[a, s, n] -> Array[b, n, Composite]
        //
        // The output variant is pinned to Composite because `egir::soac_expand`
        // always materializes the map result via `_w_intrinsic_uninit` +
        // `_w_intrinsic_array_with_inplace`. Preserving the input variant `s`
        // in the output type would be a lie: post-expand the representation
        // is always a composite buffer. The lie leaked into loop back-edges
        // — a loop carrying `map(…, iota(N))` ends up with a Virtual-variant
        // block parameter that SPIR-V `ArrayWith` can't write to.
        let (a, b, n, s) = (
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
        );
        let composite = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
        let body = Self::arrow_chain(
            &[
                Type::arrow(Self::var(a), Self::var(b)),
                self.array_ty(Self::var(a), s, n),
            ],
            Type::Constructed(
                TypeName::Array,
                vec![Self::var(b), composite, Self::var(n), no_buffer()],
            ),
        );
        self.define_builtin("map", Self::generalize_closed(body));

        // zip: ∀n a b s. Array[a, s, n] -> Array[b, s, n] -> Array[(a, b), s, n]
        let (n, a, b, s) = (
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
        );
        let body = Self::arrow_chain(
            &[
                self.array_ty(Self::var(a), s, n),
                self.array_ty(Self::var(b), s, n),
            ],
            self.array_ty(tuple(vec![Self::var(a), Self::var(b)]), s, n),
        );
        self.define_builtin("zip", Self::generalize_closed(body));

        // zip3..zip5: ∀n a1..aN s. [a1,s,n] -> ... -> [aN,s,n] -> [(a1,...,aN),s,n]
        for arity in 3..=5 {
            self.register_zip_n(arity);
        }

        // to_vec: ∀n a s. Array[a, s, n] -> Vec(n, a)
        let (n, a, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(
            &[self.array_ty(Self::var(a), s, n)],
            Self::vec_ty(n, Self::var(a)),
        );
        self.define_builtin("to_vec", Self::generalize_closed(body));

        // replicate: ∀size a s. i32 -> a -> Array[a, s, size]
        let (size, a, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[i32(), Self::var(a)], self.array_ty(Self::var(a), s, size));
        self.define_builtin("replicate", Self::generalize_closed(body));

        // reduce: ∀a n s. (a -> a -> a) -> a -> Array[a, s, n] -> a
        let (a, n, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let op_ty = Type::arrow(Self::var(a), Type::arrow(Self::var(a), Self::var(a)));
        let body = Self::arrow_chain(
            &[op_ty, Self::var(a), self.array_ty(Self::var(a), s, n)],
            Self::var(a),
        );
        self.define_builtin("reduce", Self::generalize_closed(body));

        // scan: ∀a n s. (a -> a -> a) -> a -> Array[a, s, n] -> Array[a, n, Composite]
        // Output variant pinned to Composite for the same reason as `map` above.
        let (a, n, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let op_ty = Type::arrow(Self::var(a), Type::arrow(Self::var(a), Self::var(a)));
        let composite = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
        let body = Self::arrow_chain(
            &[op_ty, Self::var(a), self.array_ty(Self::var(a), s, n)],
            Type::Constructed(
                TypeName::Array,
                vec![Self::var(a), composite, Self::var(n), no_buffer()],
            ),
        );
        self.define_builtin("scan", Self::generalize_closed(body));

        // filter: ∀a n s. (a -> bool) -> Array[a, s, n] -> ?k. Array[a, Abstract, k, no_buffer]
        //
        // Output variant is `Abstract` — representation-polymorphic at the
        // TLC level. The concrete runtime variant (Bounded for static-
        // capacity inputs, View for runtime-sized) is chosen by the
        // producer's EGIR lowering in `egir/from_tlc.rs`. Pinning
        // Composite here would freeze the consumer's signature before
        // the producer's representation exists — see
        // `egir::verify_no_abstract` for the backend-boundary invariant.
        let (a, n, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let pred_ty = Type::arrow(Self::var(a), bool_ty);
        let array_a = self.array_ty(Self::var(a), s, n);
        let k = "k".to_string();
        let k_var = Type::Constructed(TypeName::SizeVar(k.clone()), vec![]);
        let abstract_variant = Type::Constructed(TypeName::ArrayVariantAbstract, vec![]);
        let result_array = Type::Constructed(
            TypeName::Array,
            vec![Self::var(a), abstract_variant, k_var, no_buffer()],
        );
        let existential_result = Type::Constructed(TypeName::Existential(vec![k]), vec![result_array]);
        let body = Self::arrow_chain(&[pred_ty, array_a], existential_result);
        self.define_builtin("filter", Self::generalize_closed(body));

        // scatter: ∀a n m s1 s2 s3. Array[a, s1, n] -> Array[i32, s2, m] -> Array[a, s3, m] -> Array[a, s1, n]
        let (a, n, m) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let (s1, s2, s3) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let dest_array = self.array_ty(Self::var(a), s1, n);
        let indices_array = self.array_ty(i32(), s2, m);
        let values_array = self.array_ty(Self::var(a), s3, m);
        let body = Self::arrow_chain(&[dest_array.clone(), indices_array, values_array], dest_array);
        self.define_builtin("scatter", Self::generalize_closed(body));

        // reduce_by_index: ∀a n m s1 s2 s3. Array[a, s1, n] -> (a -> a -> a) -> a -> Array[i32, s2, m] -> Array[a, s3, m] -> Array[a, s1, n]
        let (a, n, m) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let (s1, s2, s3) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let op_ty = Type::arrow(Self::var(a), Type::arrow(Self::var(a), Self::var(a)));
        let dest_array = self.array_ty(Self::var(a), s1, n);
        let indices_array = self.array_ty(i32(), s2, m);
        let values_array = self.array_ty(Self::var(a), s3, m);
        let body = Self::arrow_chain(
            &[
                dest_array.clone(),
                op_ty,
                Self::var(a),
                indices_array,
                values_array,
            ],
            dest_array,
        );
        self.define_builtin("reduce_by_index", Self::generalize_closed(body));

        // _w_alloc_array: ∀n t s. i32 -> Array[t, s, n]
        let (n, t, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[i32()], self.array_ty(Self::var(t), s, n));
        self.define_builtin("_w_alloc_array", Self::generalize_closed(body));

        // dot: ∀n t. Vec(n, t) -> Vec(n, t) -> t
        let (n, t) = (self.fresh_var(), self.fresh_var());
        let vec = Self::vec_ty(n, Self::var(t));
        let body = Self::arrow_chain(&[vec.clone(), vec], Self::var(t));
        self.define_builtin("dot", Self::forall(&[n, t], body));

        // Math functions: ∀t. t -> t (works on f32, vec2f32, vec3f32, vec4f32).
        let t = self.fresh_var();
        let math_unary = Self::forall(&[t], Type::arrow(Self::var(t), Self::var(t)));
        for name in &["abs", "floor", "ceil", "fract"] {
            self.define_builtin(name, math_unary.clone());
        }
        // sin/cos/tan/asin/acos/atan/sqrt/exp/log/radians/degrees and the
        // binary `pow`/`atan2`/`mod` resolve via `open f32` (scalar) or
        // `open vec` (vector). The schemes for `f32.cos`/`vec.cos`/etc.
        // come from prelude module sigs and the catalog's vec.* entries.

        // Register vector field mappings
        self.register_vector_fields();

        // Note: Prelude files are automatically loaded when ModuleManager is created

        Ok(())
    }

    /// Names registered by `load_builtins`. Sourced from
    /// `globals.builtins`'s `StableMap` for deterministic order.
    pub fn builtin_names(&self) -> Vec<String> {
        self.globals.builtins.keys().cloned().collect()
    }

    fn register_vector_fields(&mut self) {
        // Vector field access is now handled directly in the FieldAccess case
        // Vec(size, element_type) fields (x, y, z, w) return element_type
    }

    /// Register a record type with its field mappings
    pub fn register_record_type(&mut self, type_name: &str, fields: Vec<(String, Type)>) {
        for (field_name, field_type) in fields {
            self.record_field_map.insert((type_name.to_string(), field_name), field_type);
        }
    }

    pub fn check_program(&mut self, program: &Program) -> Result<LookupMap<NodeId, TypeScheme>> {
        // Forward-declare ascribed file-scope `def`s into
        // `globals.user_file_defs` so module function bodies can close
        // over them. No `scope_stack.push_scope()` needed — `GlobalEnv`
        // is keyed independently of the local-scope stack.
        self.forward_declare_ascribed_file_scope(program);

        // Type-check module functions first to populate the module_schemes cache.
        // This must happen before prelude functions since they may reference module functions.
        self.check_module_functions()?;

        // Type-check prelude functions (needed so they're in scope for user code)
        self.check_prelude_functions()?;

        // Process user declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        if let Some(err) = self.take_pending_cycle_error() {
            return Err(err);
        }

        self.check_resource_binding_consistency(program)?;

        // Emit warnings for all type holes now that types are fully inferred
        self.emit_hole_warnings();

        // Apply the context to all types in the type table to resolve type variables
        let resolved_table: LookupMap<NodeId, TypeScheme> =
            self.type_table.iter().map(|(node_id, scheme)| (*node_id, self.apply_scheme(scheme))).collect();

        Ok(resolved_table)
    }

    /// A `(set, binding)` slot is fine to share across entry points — different
    /// pipelines have independent descriptor layouts, and the runtime can bind
    /// one resource through several views (an image written as a `storage_image`
    /// by one entry and sampled as a `texture` by another; a buffer read by
    /// two). The one thing that breaks is two `storage`/`uniform` **buffers**
    /// at the same slot with **different element types** (`[]f32` vs
    /// `[]vec4f32`): codegen coalesces them to one module-global of one element
    /// type that the other entry then indexes as another — invalid SPIR-V both
    /// spirv-val and naga reject. Reject exactly that; the element comparison
    /// peels `*[…]` / array-view wrappers so the same buffer seen as a view in
    /// one entry and a pointer-to-array in another is not a conflict.
    fn check_resource_binding_consistency(&self, program: &Program) -> Result<()> {
        fn param_attrs(p: &Pattern) -> &[Attribute] {
            match &p.kind {
                PatternKind::Attributed(attrs, _) => attrs,
                PatternKind::Typed(inner, _) => param_attrs(inner),
                _ => &[],
            }
        }

        struct BufferUse {
            kind: &'static str,
            elem: String,
            display: String,
            entry: String,
        }
        let mut seen: LookupMap<(u32, u32), BufferUse> = LookupMap::new();

        for decl in &program.declarations {
            let Declaration::Entry(entry) = decl else {
                continue;
            };
            for param in &entry.params {
                // Only buffers can coalesce destructively. Textures / samplers /
                // storage images get a separate global per entry, so cross-kind
                // reuse of a slot is valid aliasing, not a conflict.
                let Some((set, binding, kind)) = param_attrs(param).iter().find_map(|a| match a {
                    Attribute::Storage { set, binding, .. } => Some((*set, *binding, "storage")),
                    Attribute::Uniform { set, binding } => Some((*set, *binding, "uniform")),
                    _ => None,
                }) else {
                    continue;
                };

                let ty = param.pattern_type().map(|t| self.normalize_annotation_type(t, None));

                // A uniform must have a block layout: a 32-bit scalar, a
                // vec2/3/4 of them, or a flat record/tuple of those
                // (std140). Reject the rest here with a source span —
                // the backends assume the layout exists.
                if kind == "uniform" {
                    if let Some(t) = &ty {
                        if crate::ssa::layout::block_layout(t, crate::interface::StorageLayout::Std140)
                            .is_none()
                        {
                            bail_type_at!(
                                param.h.span,
                                "uniform (set={}, binding={}) has type `{}`, which cannot be a 
                                 uniform block: supported are 32-bit scalars (f32/i32/u32), 
                                 vec2/3/4 of them, and flat records/tuples of those (std140). 
                                 bools, matrices, arrays, and nested records are not supported",
                                set,
                                binding,
                                self.format_type(t)
                            );
                        }
                    }
                }

                let elem = ty.as_ref().map(buffer_element_signature).unwrap_or_default();
                let display = match &ty {
                    Some(t) => format!("{} {}", kind, self.format_type(t)),
                    None => kind.to_string(),
                };

                match seen.get(&(set, binding)) {
                    Some(prev) if prev.kind == kind && prev.elem != elem => {
                        bail_type_at!(
                            param.h.span,
                            "{} buffer (set={}, binding={}) is declared as `{}` in entry `{}`, \
                             but as `{}` in entry `{}`; a buffer slot shared across entries must \
                             use the same element type",
                            kind,
                            set,
                            binding,
                            display,
                            entry.name,
                            prev.display,
                            prev.entry
                        );
                    }
                    Some(_) => {}
                    None => {
                        seen.insert(
                            (set, binding),
                            BufferUse {
                                kind,
                                elem,
                                display,
                                entry: entry.name.clone(),
                            },
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Emit warnings for all type holes showing their inferred types
    fn emit_hole_warnings(&mut self) {
        // Clone the holes list to avoid borrow checker issues
        let holes = self.type_holes.clone();
        for (node_id, span) in holes {
            if let Some(hole_scheme) = self.type_table.get(&node_id) {
                self.warnings.push(TypeWarning::TypeHoleFilled {
                    inferred_type: self.scheme_inner(hole_scheme),
                    span,
                });
            }
        }
    }

    /// Helper to type check a function body with parameters in scope
    /// Returns (param_types, body_type)
    fn check_function_with_params(
        &mut self,
        params: &[Pattern],
        body: &Expression,
        module_name: Option<&str>,
    ) -> Result<(Vec<Type>, Type)> {
        self.check_function_with_params_inner(params, body, module_name, false, None)
    }

    fn check_entry_with_params(
        &mut self,
        params: &[Pattern],
        body: &Expression,
        entry_type: &Attribute,
    ) -> Result<(Vec<Type>, Type)> {
        self.check_function_with_params_inner(params, body, None, true, Some(entry_type))
    }

    fn check_function_with_params_inner(
        &mut self,
        params: &[Pattern],
        body: &Expression,
        module_name: Option<&str>,
        is_entry: bool,
        // The entry's stage attribute (`Vertex` / `Fragment` / `Compute`)
        // when `is_entry`; `None` for ordinary functions. Drives
        // stage-specific parameter validation.
        entry_stage: Option<&Attribute>,
    ) -> Result<(Vec<Type>, Type)> {
        // Create type variables or use explicit types for parameters
        let mut param_types: Vec<Type> = Vec::with_capacity(params.len());
        for p in params {
            let ty = p.pattern_type().cloned().unwrap_or_else(|| self.context.new_variable());
            param_types.push(self.instantiate_annotation_type(&ty, module_name));
        }

        // For entry point parameters, constrain array address spaces to Storage
        if is_entry {
            for param_type in &param_types {
                self.constrain_array_to_storage(param_type)?;
            }
        }

        // Validate that no parameter types contain existential quantifiers
        // Existential types are only valid in return types, not parameters
        for (param, param_type) in params.iter().zip(param_types.iter()) {
            if Self::contains_existential(param_type) {
                bail_type_at!(
                    param.h.span,
                    "Existential types (?k. ...) are only allowed in return types, not parameter types"
                );
            }
        }

        // Vertex entries: validate `#[vertex_slot(n)]` input parameters.
        // Each must carry a GPU vertex-buffer format (32-bit scalar or
        // 2-4 wide vector of f32/i32/u32), slots must be unique, and
        // every non-builtin vertex param must have an explicit slot —
        // the pipeline descriptor needs a stable slot per attribute.
        // Fragment inputs are `#[varying(n)]` interpolants, validated below.
        if matches!(entry_stage, Some(Attribute::Vertex)) {
            // `parse_entry_params` builds `Typed(Attributed([attrs], Name), ty)`
            // — `Typed` outermost — so peel through `Typed` to reach the attrs.
            fn param_attrs(p: &Pattern) -> &[Attribute] {
                match &p.kind {
                    PatternKind::Attributed(attrs, _) => attrs,
                    PatternKind::Typed(inner, _) => param_attrs(inner),
                    _ => &[],
                }
            }
            let mut seen_locations = LookupSet::new();
            for (param, param_type) in params.iter().zip(param_types.iter()) {
                if matches!(param.kind, PatternKind::Unit) {
                    continue;
                }
                let attrs = param_attrs(param);
                // `#[uniform]` / `#[storage]` / `#[texture]` / `#[sampler]`
                // params are resource bindings, not vertex attributes — a
                // vertex shader reading a uniform or sampling a texture is
                // standard. Skip them here; they're validated as bindings
                // elsewhere (and surfaced into the pipeline descriptor).
                if attrs.iter().any(|a| {
                    matches!(
                        a,
                        Attribute::Uniform { .. }
                            | Attribute::Storage { .. }
                            | Attribute::Texture { .. }
                            | Attribute::Sampler { .. }
                    )
                }) {
                    continue;
                }
                if let Some(bad) = attrs.iter().find_map(|a| match a {
                    Attribute::Varying(_) => Some("#[varying(n)]"),
                    Attribute::Target(_) => Some("#[target(name)]"),
                    _ => None,
                }) {
                    bail_type_at!(
                        param.h.span,
                        "{} is not valid on a vertex input; vertex-buffer inputs use #[vertex_slot(n)]",
                        bad
                    );
                }
                let slot = attrs.first_vertex_slot();
                let is_builtin = attrs.iter().any(|a| matches!(a, Attribute::BuiltIn(_)));
                match slot {
                    Some(slot) => {
                        if crate::ssa::layout::vertex_format(param_type).is_none() {
                            bail_type_at!(
                                param.h.span,
                                "vertex shader #[vertex_slot({})] input must be an explicitly-typed \
                                 32-bit scalar or vec2/3/4 of f32/i32/u32; got {:?}",
                                slot,
                                param_type
                            );
                        }
                        if !seen_locations.insert(slot) {
                            bail_type_at!(
                                param.h.span,
                                "duplicate vertex shader input #[vertex_slot({})]",
                                slot
                            );
                        }
                    }
                    None if is_builtin => {}
                    None => {
                        bail_type_at!(
                            param.h.span,
                            "vertex shader input parameter must have #[vertex_slot(n)] or #[builtin(...)]"
                        );
                    }
                }
            }
        }

        // Push new scope for function parameters
        self.scope_stack.push_scope();

        // Add parameters to scope
        for (param, param_type) in params.iter().zip(param_types.iter()) {
            // Skip unit patterns (no parameters)
            if matches!(param.kind, PatternKind::Unit) {
                continue;
            }

            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    err_type_at!(
                        param.h.span,
                        "Complex patterns in function parameters not yet supported"
                    )
                })?
                .to_string();
            let resolved_param_type = param_type.apply(&self.context);
            self.type_table.insert(param.h.id, TypeScheme::Monotype(resolved_param_type.clone()));
            let type_scheme = TypeScheme::Monotype(resolved_param_type);

            debug!(
                "Adding parameter '{}' to scope with type: {:?}",
                param_name, param_type
            );
            self.define(param_name, IdentifierKind::Local, type_scheme);
        }

        // Infer body type
        let body_type = self.infer_expression(body)?;

        // Pop parameter scope
        self.scope_stack.pop_scope();

        Ok((param_types, body_type))
    }

    /// Insert each ascribed file-scope `def` into
    /// `globals.user_file_defs`. Idempotent with the main
    /// `check_program` walk, which re-inserts the same scheme. Defs
    /// without full ascription are deferred to the main loop.
    fn forward_declare_ascribed_file_scope(&mut self, program: &Program) {
        for decl in &program.declarations {
            if let Declaration::Decl(d) = decl {
                if let Some(scheme) = self.ascription_to_scheme(d) {
                    self.globals.user_file_defs.insert(d.name.clone(), scheme);
                }
            }
        }
    }

    /// Phase-1 dual-write hook, Phase-2 reading `current_context`.
    /// Routes a top-level `Decl` scheme into the right `GlobalEnv` slot:
    ///   * `module_name = Some(_)` → `globals.module_schemes` (key
    ///     is the already-qualified `scope_name`).
    ///   * `current_context == Prelude` → `globals.prelude_defs`.
    ///   * Otherwise → `globals.user_file_defs`.
    ///
    /// Removed when Phase 4 migrates the `define` insertion sites
    /// directly.
    fn globals_dual_write_decl(
        &mut self,
        scope_name: &str,
        scheme: &TypeScheme,
        module_name: Option<&str>,
    ) {
        if module_name.is_some() {
            self.globals.module_schemes.insert(scope_name.to_string(), scheme.clone());
        } else if self.current_context == LookupContext::Prelude {
            self.globals.prelude_defs.insert(scope_name.to_string(), scheme.clone());
        } else {
            self.globals.user_file_defs.insert(scope_name.to_string(), scheme.clone());
        }
    }

    /// Build a `TypeScheme` from a `Decl`'s ascription alone, without
    /// inspecting the body. Returns `None` if the return type or any
    /// parameter type isn't statically determined (parameter without
    /// `Pattern::Typed`, missing return-type annotation, etc.) — the
    /// main type-check loop handles those.
    fn ascription_to_scheme(&self, decl: &Decl) -> Option<TypeScheme> {
        let return_ty = decl.ty.as_ref()?;
        let resolved_return = self.resolve_type_aliases_scoped(return_ty, None);
        let mut param_types = Vec::with_capacity(decl.params.len());
        for param in &decl.params {
            let ty = match &param.kind {
                PatternKind::Typed(_, ty) => ty.clone(),
                _ => return None,
            };
            param_types.push(self.resolve_type_aliases_scoped(&ty, None));
        }
        let func_ty =
            param_types.into_iter().rev().fold(resolved_return, |acc, p| crate::types::function(p, acc));
        Some(self.generalize(&func_ty))
    }

    /// Type-check function bodies from modules (e.g., rand.init, rand.int, f32.pi)
    /// This populates the type table so these functions can be flattened to MIR.
    /// Note: module_schemes for Spec::Sig items are pre-built by resolve_placeholders
    /// and passed to the constructor.
    pub fn check_module_functions(&mut self) -> Result<()> {
        // Collect all module declarations that need flattening (includes constants like f32.pi)
        let module_functions: Vec<(String, crate::ast::Decl)> = self
            .module_manager
            .get_all_module_declarations()
            .into_iter()
            .map(|(module_name, decl)| (module_name.to_string(), decl.clone()))
            .collect();

        // Type-check each module function with module context for
        // alias resolution. `globals_dual_write_decl` inside
        // `check_decl_as_in_module` is the single source of truth
        // for `globals.module_schemes`.
        for (module_name, decl) in module_functions {
            let qualified_name = format!("{}.{}", module_name, decl.name);
            debug!("Type-checking module function: {}", qualified_name);
            self.check_decl_as_in_module(&decl, &qualified_name, Some(&module_name))?;
        }

        if let Some(err) = self.take_pending_cycle_error() {
            return Err(err);
        }
        Ok(())
    }

    /// Type-check all prelude functions.
    /// Called during prelude creation to populate the type table for prelude function bodies.
    pub fn check_prelude_functions(&mut self) -> Result<()> {
        // Collect all prelude function declarations to avoid borrowing issues
        let prelude_functions: Vec<crate::ast::Decl> =
            self.module_manager.get_prelude_function_declarations().into_iter().cloned().collect();

        let saved_context = std::mem::replace(&mut self.current_context, LookupContext::Prelude);
        let result: Result<()> = (|| {
            for decl in prelude_functions {
                debug!("Type-checking prelude function: {}", decl.name);
                self.check_decl(&decl)?;
            }
            Ok(())
        })();
        self.current_context = saved_context;
        result?;

        if let Some(err) = self.take_pending_cycle_error() {
            return Err(err);
        }
        Ok(())
    }

    /// Consume the type checker and return the type table.
    /// Extracts the prelude type table after type-checking prelude functions.
    pub fn into_type_table(self) -> LookupMap<NodeId, TypeScheme> {
        self.type_table
    }

    /// Get all function type schemes from the `GlobalEnv`. Extracts
    /// canonical schemes for prelude/user/module functions during
    /// prelude creation and downstream lowering, so
    /// monomorphization has consistent type-variable IDs across
    /// params/return. Walks the four global namespaces in a
    /// deterministic order; ties broken last-write-wins per the
    /// insert order.
    pub fn get_function_schemes(&self) -> LookupMap<String, TypeScheme> {
        let mut schemes = LookupMap::new();
        let mut insert = |name: &str, scheme: &TypeScheme| {
            schemes.insert(name.to_string(), self.apply_context_to_scheme(scheme));
        };
        for (name, scheme) in &self.globals.builtins {
            insert(name, scheme);
        }
        for (name, scheme) in &self.globals.module_schemes {
            insert(name, scheme);
        }
        for (name, scheme) in &self.globals.prelude_defs {
            insert(name, scheme);
        }
        for (name, scheme) in &self.globals.user_file_defs {
            insert(name, scheme);
        }
        schemes
    }

    /// Apply the unification context to a TypeScheme, resolving type variables.
    fn apply_context_to_scheme(&self, scheme: &TypeScheme) -> TypeScheme {
        match scheme {
            TypeScheme::Monotype(ty) => TypeScheme::Monotype(ty.apply(&self.context)),
            TypeScheme::Polytype { variable, body } => {
                // Check if this variable was resolved to a concrete type
                let var_ty = Type::Variable(*variable);
                let resolved = var_ty.apply(&self.context);
                if let Type::Variable(v) = resolved {
                    if v == *variable {
                        // Variable is still free, keep the quantifier
                        TypeScheme::Polytype {
                            variable: *variable,
                            body: Box::new(self.apply_context_to_scheme(body)),
                        }
                    } else {
                        // Variable was unified with another variable, use the resolved ID
                        TypeScheme::Polytype {
                            variable: v,
                            body: Box::new(self.apply_context_to_scheme(body)),
                        }
                    }
                } else {
                    // Variable was resolved to a concrete type, remove the quantifier
                    // and substitute in the body
                    self.apply_context_to_scheme(body)
                }
            }
        }
    }

    /// Get a module function scheme from `globals.module_schemes`.
    /// Requires `check_module_functions()` to have been called first.
    pub fn get_module_scheme(&self, qualified_name: &str) -> Option<&TypeScheme> {
        self.globals.module_schemes.get(qualified_name)
    }

    fn check_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(decl_node) => {
                debug!("Checking {} declaration: {}", decl_node.keyword, decl_node.name);
                self.check_decl(decl_node)
            }
            Declaration::Entry(entry) => {
                debug!("Checking entry point: {}", entry.name);
                let (_param_types, body_type) =
                    self.check_entry_with_params(&entry.params, &entry.body, &entry.entry_type)?;
                debug!("Entry point '{}' body type: {:?}", entry.name, body_type);

                // Build expected return type from declared outputs
                let expected_type = match entry.outputs.len() {
                    0 => Type::Constructed(TypeName::Unit, vec![]),
                    1 => entry.outputs[0].ty.clone(),
                    n => Type::Constructed(
                        TypeName::Tuple(n),
                        entry.outputs.iter().map(|o| o.ty.clone()).collect(),
                    ),
                };
                // Resolve type aliases (e.g., rand.state -> f32) and instantiate
                // residual array placeholders so an alias return like `world`
                // unifies against the body's concrete `view`/`composite` arrays.
                let expected_type = self.instantiate_annotation_type(&expected_type, None);

                // An existential return (`?k. T`) is *packed*: instantiate the
                // declared bound size vars — and any existential the body
                // itself still carries — as fresh unification variables, then
                // unify the inner types. The fresh var solves to the witness
                // size the body produces (e.g. the runtime length of a `filter`
                // result that also feeds `length`). Unifying the raw `?k.`
                // wrapper instead only matched a body that happened to spell its
                // existential with the same variable name.
                let expected_inner = self.instantiate_existential(expected_type.clone());
                let body_inner = self.instantiate_existential(body_type.clone());

                // Validate the body's value shape against the declared
                // outputs; any `*` on the output contract is a signature
                // property forgotten for this shape check.
                let storage_image_tail_sinks_to_unit = entry.outputs.is_empty()
                    && matches!(expected_inner, Type::Constructed(TypeName::Unit, _))
                    && matches!(
                        body_inner.apply(&self.context),
                        Type::Constructed(TypeName::StorageTexture, _)
                    );
                if !storage_image_tail_sinks_to_unit {
                    self.unify_or_err_weakening(
                        &body_inner,
                        &expected_inner,
                        entry.body.h.span,
                        &format!("Entry point '{}' return type mismatch", entry.name),
                    )?;
                }

                Ok(())
            }
            Declaration::Sig(sig_decl) => {
                debug!("Checking Sig declaration: {}", sig_decl.name);
                self.check_sig_decl(sig_decl)
            }
            Declaration::TypeBind(type_bind) => {
                debug!("Processing TypeBind: {}", type_bind.name);
                // Type bindings are registered in the environment during elaboration
                // For now, just skip them in type checking
                Ok(())
            }
            Declaration::Module(_) => {
                // Module/functor declarations should be elaborated away before type checking
                // If we encounter one here, it means elaboration wasn't run or failed
                Err(err_module!(
                    "Module declarations should be elaborated before type checking"
                ))
            }
            Declaration::ModuleTypeBind(_) => {
                // Module type bindings are erased during elaboration
                // If we see one, elaboration wasn't run
                Ok(())
            }
            Declaration::Open(_) => {
                // Open declarations should be elaborated away
                Ok(())
            }
            Declaration::Import(_) => {
                // Import declarations should be resolved during elaboration
                Ok(())
            }
            Declaration::Extern(extern_decl) => {
                debug!("Checking Extern declaration: {}", extern_decl.name);
                self.check_extern_decl(extern_decl)
            }
            Declaration::Resource(_) => {
                // Resources carry no body to check; views were rewritten to
                // concrete binding attributes before type checking.
                Ok(())
            }
        }
    }

    fn check_decl(&mut self, decl: &Decl) -> Result<()> {
        self.check_decl_as_in_module(decl, &decl.name, None)
    }

    fn check_decl_as_in_module(
        &mut self,
        decl: &Decl,
        scope_name: &str,
        module_name: Option<&str>,
    ) -> Result<()> {
        // Set current module context for alias resolution in nested expressions
        let saved_module = self.current_module.take();
        self.current_module = module_name.map(|s| s.to_string());
        // Mirror the legacy `current_module` field into `current_context`.
        // Inside `Prelude` (set by `check_prelude_functions`) we KEEP the
        // Prelude context; module_name will be None there. Otherwise:
        //   Some(m) → Module { name: m }
        //   None    → preserve the surrounding context (UserFile, or
        //             whatever else the outer driver chose).
        let saved_context = self.current_context.clone();
        if let Some(m) = module_name {
            self.current_context = LookupContext::Module { name: m.to_string() };
        }

        // Note: SizeVar/UserVar substitution is now handled by resolve_placeholders pass
        // before type checking. Type parameter names are already converted to type variables.

        if decl.params.is_empty() {
            // Variable or entry point declaration: let/def name: type = value or let/def name = value
            // Resolve type aliases in declared type (e.g., rand.state -> f32)
            let resolved_declared_type =
                decl.ty.as_ref().map(|ty| self.resolve_type_aliases_scoped(ty, module_name));

            let expr_type = if let Some(ref declared_type) = resolved_declared_type {
                // Use bidirectional checking when type annotation is present
                self.check_expression(&decl.body, declared_type)?
            } else {
                // No type annotation, infer the type
                self.infer_expression(&decl.body)?
            };

            if let Some(ref declared_type) = resolved_declared_type {
                self.unify_or_err_weakening(&expr_type, declared_type, decl.body.h.span, "Type mismatch")?;
            }

            // Add to GlobalEnv (no scope_stack write — that's the
            // local-only stack now).
            let stored_type = resolved_declared_type.unwrap_or(expr_type.clone());
            let type_scheme = self.generalize(&stored_type);
            debug!("Inserting variable '{}' into globals", scope_name);
            self.globals_dual_write_decl(scope_name, &type_scheme, module_name);
            debug!("Inferred type for {}: {}", scope_name, stored_type);
        } else {
            // Function declaration: let/def name param1 param2 = body

            // A consuming function may not be used as a value: a parameter
            // whose diet mentions a `*` behind an arrow (e.g. `f: *T -> U`)
            // is rejected.
            for (param, diet) in decl.params.iter().zip(&decl.param_diets) {
                if diet.mentions_consuming_function() {
                    bail_type_at!(
                        param.h.span,
                        "a consuming function may not be used as a value: \
                         a parameter's type may not be a function that consumes its argument"
                    );
                }
            }
            // Record whether this def is itself a consuming function, so a
            // call site can reject passing it as a value.
            if decl.param_diets.iter().any(Diet::is_consuming) {
                self.consuming_defs.insert(decl.name.clone());
            }

            let (param_types, body_type) =
                self.check_function_with_params(&decl.params, &decl.body, module_name)?;
            debug!(
                "Successfully inferred body type for '{}': {:?}",
                decl.name, body_type
            );

            // Uniqueness lives on the signature diet, never in the type, so
            // the body's value type and the declared return shape-check
            // directly. Whether the body satisfies a `*` return is verified
            // by tlc::ownership from the diet.
            let return_type = if let Some(declared_type) = &decl.ty {
                let normalized_return_type = self.instantiate_annotation_type(declared_type, module_name);
                let ctx = if !decl.params.is_empty() {
                    format!("Function return type mismatch for '{}'", decl.name)
                } else {
                    format!("Type mismatch for '{}'", decl.name)
                };
                self.unify_or_err(&body_type, &normalized_return_type, decl.body.h.span, &ctx)?;
                normalized_return_type
            } else {
                body_type.clone()
            };

            // Build function type: param1 -> param2 -> ... -> return_type
            let func_type =
                param_types.into_iter().rev().fold(return_type, |acc, param_ty| function(param_ty, acc));

            // Entry points go through `Declaration::Entry`; `Decl` has
            // no attributed return types.

            // Update GlobalEnv with inferred type (no scope_stack
            // write — locals only).
            let type_scheme = self.generalize_function(&func_type);
            self.globals_dual_write_decl(scope_name, &type_scheme, module_name);

            // Track arity for partial application checking
            self.arity_map.insert(scope_name.to_string(), decl.params.len());

            debug!("Inferred type for {}: {}", scope_name, func_type);
        }

        // Restore previous module context
        self.current_module = saved_module;
        self.current_context = saved_context;

        Ok(())
    }

    fn check_sig_decl(&mut self, decl: &SigDecl) -> Result<()> {
        // Sig declarations are just type signatures - register them in
        // GlobalEnv per current context.
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.globals_dual_write_decl(&decl.name, &type_scheme, None);
        Ok(())
    }

    fn check_extern_decl(&mut self, decl: &ExternDecl) -> Result<()> {
        // Extern declarations register a type signature for a linked
        // SPIR-V function — same routing as Sig / Decl.
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.globals_dual_write_decl(&decl.name, &type_scheme, None);
        debug!(
            "Registered extern function '{}' with linkage '{}'",
            decl.name, decl.linkage_name
        );
        Ok(())
    }

    fn infer_expression(&mut self, expr: &Expression) -> Result<Type> {
        let ty = match &expr.kind {
            ExprKind::RecordLiteral(fields) => {
                let mut field_types = Vec::new();
                for (field_name, field_expr) in fields {
                    let field_ty = self.infer_expression(field_expr)?;
                    field_types.push((field_name.clone(), field_ty));
                }
                Ok(record(field_types))
            }
            ExprKind::TypeHole => {
                // Record this hole for warning emission after type inference completes
                self.type_holes.push((expr.h.id, expr.h.span));
                Ok(self.context.new_variable())
            }
            ExprKind::IntLiteral(_) => Ok(i32()),
            ExprKind::FloatLiteral(_) => Ok(f32()),
            ExprKind::BoolLiteral(_) => Ok(bool_type()),
            ExprKind::Unit => Ok(unit()),
            ExprKind::Identifier(quals, name) => {
                let full_name = if quals.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", quals.join("."), name)
                };
                let is_qualified = !quals.is_empty();

                debug!("Looking up identifier '{}'", full_name);

                if let Some(resolved) = self.resolve_value_name(&full_name, is_qualified, expr.h.id) {
                    // Store instantiated type with substitutions applied
                    let applied = resolved.instantiated.apply(&self.context);
                    self.type_table
                        .insert(expr.h.id, TypeScheme::Monotype(applied));
                    return Ok(resolved.instantiated);
                }

                debug!("Variable lookup failed for '{}' - not in scope, intrinsics, or prelude", full_name);
                return Err(err_undef_at!(expr.h.span, "{}", full_name));
            }
            ExprKind::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    Err(err_type_at!(expr.h.span, "Cannot infer type of empty array"))
                } else {
                    let first_type = self.infer_expression(&elements[0])?;

                    // Futhark restriction: arrays of functions are not permitted
                    // Strip uniqueness to catch Unique(Arrow(...)) as well
                    let resolved_first = first_type.apply(&self.context);
                    if as_arrow(&resolved_first).is_some() {
                        bail_type_at!(
                            expr.h.span,
                            "Arrays of functions are not permitted"
                        );
                    }

                    for elem in &elements[1..] {
                        let elem_type = self.infer_expression(elem)?;
                        self.context.unify(&elem_type, &first_type).map_err(|_| {
                            err_type_at!(
                                elem.h.span,
                                "Array elements must have the same type, expected {}, got {}",
                                self.format_type(&first_type),
                                self.format_type(&elem_type)
                            )
                        })?;
                    }

                    // Array literals have concrete sizes: [1, 2, 3] has type [3]i32
                    // Variable sizes require explicit type parameters: def f[n]: [n]i32 = ...
                    Ok(sized_array(elements.len(), first_type))
                }
            }
            ExprKind::VecMatLiteral(elements) => {
                // @[...] vector/matrix literal - type inferred from context
                // Vector: @[1.0, 2.0, 3.0] - elements are scalars
                // Matrix: @[[1,2,3], [4,5,6]] - elements are array literals (rows)
                if elements.is_empty() {
                    bail_type_at!(
                        expr.h.span,
                        "Cannot infer type of empty vector/matrix literal"
                    );
                }

                // Infer type of first element to determine if vector or matrix
                let first_type = self.infer_expression(&elements[0])?;

                // Check if first element is an array (matrix) or scalar (vector)
                let is_matrix = matches!(
                    first_type.apply(&self.context),
                    Type::Constructed(TypeName::Array, _)
                );

                // Unify all element types
                for elem in &elements[1..] {
                    let elem_type = self.infer_expression(elem)?;
                    self.context.unify(&elem_type, &first_type).map_err(|_| {
                        err_type_at!(
                            elem.h.span,
                            "All elements must have the same type, expected {}, got {}",
                            self.format_type(&first_type),
                            self.format_type(&elem_type)
                        )
                    })?;
                }

                if is_matrix {
                    // Matrix: extract row size and element type from the array type
                    // Array[elem_type, variant, dim_0, ...]
                    let resolved = first_type.apply(&self.context);
                    if resolved.is_array() {
                        if let Type::Constructed(TypeName::Size(cols), _) = resolved.array_size().expect("Array has size") {
                            let rows = elements.len();
                            let elem_type = resolved.elem_type().expect("Array has elem").clone();
                            Ok(mat(rows, *cols, elem_type))
                        } else {
                            Err(err_type_at!(
                                expr.h.span,
                                "Matrix rows must be fixed-size arrays"
                            ))
                        }
                    } else {
                        Err(err_type_at!(
                            expr.h.span,
                            "Matrix rows must be fixed-size arrays"
                        ))
                    }
                } else {
                    // Vector literal
                    let size = elements.len();
                    if !(2..=4).contains(&size) {
                        Err(err_type_at!(
                            expr.h.span,
                            "Vector size must be 2, 3, or 4, got {}",
                            size
                        ))
                    } else {
                        Ok(vec(size, first_type))
                    }
                }
            }
            ExprKind::ArrayIndex(array_expr, index_expr) => {
                let array_type = self.infer_expression(array_expr)?;
                let index_type = self.infer_expression(index_expr)?;

                // Unify index type with i32
                self.context.unify(&index_type, &i32()).map_err(|_| {
                    err_type_at!(
                        index_expr.h.span,
                        "Array index must be an integer type, got {}",
                        self.format_type(&index_type.apply(&self.context))
                    )
                })?;

                // Constrain array type - strip uniqueness (indexing *[n]T works like [n]T)
                let array_type_stripped = array_type.clone();
                let (elem_var, _, _, _) = self.constrain_array_type(
                    &array_type_stripped,
                    &array_expr.h.span,
                    "Cannot index non-array type",
                )?;

                Ok(elem_var.apply(&self.context))
            }
            ExprKind::ArrayWith { array, index, value, .. } => {
                // Type check either an array update (`arr with [i] = v`) or
                // a linear storage-image update (`img with [xy] = rgba`).
                let array_type = self.infer_expression(array)?;
                let index_type = self.infer_expression(index)?;
                let value_type = self.infer_expression(value)?;

                let resolved_target = array_type.apply(&self.context);
                if matches!(resolved_target, Type::Constructed(TypeName::StorageTexture, _)) {
                    let coord_ty = vec(2, i32());
                    self.context.unify(&index_type, &coord_ty).map_err(|_| {
                        err_type_at!(
                            index.h.span,
                            "Storage image update index must be vec2i32, got {}",
                            self.format_type(&index_type.apply(&self.context))
                        )
                    })?;

                    let texel_ty = vec(4, f32());
                    self.context.unify(&value_type, &texel_ty).map_err(|_| {
                        err_type_at!(
                            value.h.span,
                            "Storage image update value must be vec4f32, got {}",
                            self.format_type(&value_type.apply(&self.context))
                        )
                    })?;

                    Ok(resolved_target)
                } else {
                    // Unify index type with i32
                    self.context.unify(&index_type, &i32()).map_err(|_| {
                        err_type_at!(
                            index.h.span,
                            "Array index must be an integer type, got {}",
                            self.format_type(&index_type.apply(&self.context))
                        )
                    })?;

                    // Constrain array type - strip uniqueness
                    let array_type_stripped = array_type.clone();
                    let (elem_var, _, _, _) = self.constrain_array_type(
                        &array_type_stripped,
                        &array.h.span,
                        "Cannot update non-array type",
                    )?;

                    // Unify value type with element type
                    self.context.unify(&value_type, &elem_var).map_err(|_| {
                        err_type_at!(
                            value.h.span,
                            "Array element type mismatch: expected {}, got {}",
                            self.format_type(&elem_var.apply(&self.context)),
                            self.format_type(&value_type.apply(&self.context))
                        )
                    })?;

                    // Return the array type (same type as input)
                    Ok(array_type.apply(&self.context))
                }
            }
            ExprKind::RecordWith { record, path, value } => {
                // Walk `path` segment-by-segment through nested records
                // to find the inner field's type, unify the RHS against
                // it, and return the outer record's type unchanged.
                let outer_ty = self.infer_expression(record)?;
                let mut current_ty = outer_ty.clone();
                for segment in path {
                    let resolved = current_ty.apply(&self.context);
                    let stripped = resolved.clone();
                    let (fields, field_types) = match &stripped {
                        Type::Constructed(TypeName::Record(fs), tys) => (fs, tys),
                        _ => bail_type_at!(
                            expr.h.span,
                            "`with` field path requires a record type, got {}",
                            self.format_type(&resolved)
                        ),
                    };
                    let Some(field_index) = fields.get_index(segment) else {
                        bail_type_at!(
                            expr.h.span,
                            "Record type has no field '{}'. Available: {}",
                            segment,
                            fields.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                        );
                    };
                    current_ty = field_types[field_index].clone();
                }

                let value_ty = self.infer_expression(value)?;
                self.context.unify(&value_ty, &current_ty).map_err(|_| {
                    err_type_at!(
                        value.h.span,
                        "Record field type mismatch: expected {}, got {}",
                        self.format_type(&current_ty.apply(&self.context)),
                        self.format_type(&value_ty.apply(&self.context))
                    )
                })?;

                Ok(outer_ty.apply(&self.context))
            }
            ExprKind::VecWith {
                target,
                components,
                op,
                value,
            } => {
                // Target must resolve to a Vec<elem, size>.
                let target_type = self.infer_expression(target)?;
                let elem_var = self.context.new_variable();
                let size_var = self.context.new_variable();
                let want_vec =
                    Type::Constructed(TypeName::Vec, vec![elem_var.clone(), size_var.clone()]);
                self.context.unify(&target_type, &want_vec).map_err(|_| {
                    err_type_at!(
                        target.h.span,
                        "`with .swizzle` requires a vector target, got {}",
                        self.format_type(&target_type.apply(&self.context))
                    )
                })?;

                // Range-check components against the resolved vec size,
                // when known. (Parser already enforced distinctness.)
                let resolved_target = target_type.apply(&self.context);
                if let Type::Constructed(TypeName::Vec, args) = &resolved_target {
                    if let Some(Type::Constructed(TypeName::Size(n), _)) = args.get(1) {
                        for &c in components {
                            if (c as usize) >= *n {
                                bail_type_at!(
                                    expr.h.span,
                                    "swizzle component `{}` is out of range for vec{}",
                                    swizzle_letter(c),
                                    n
                                );
                            }
                        }
                    }
                }

                // What type does `target.swizzle` have? Single-component
                // is the elem type; multi-component is a vec of swizzle
                // length. This is what the lhs slot reads/writes.
                let elem_ty = elem_var.apply(&self.context);
                let swizzle_ty = if components.len() == 1 {
                    elem_ty.clone()
                } else {
                    Type::Constructed(
                        TypeName::Vec,
                        vec![
                            elem_ty.clone(),
                            Type::Constructed(TypeName::Size(components.len()), vec![]),
                        ],
                    )
                };

                let rhs_type = self.infer_expression(value)?;

                match op {
                    None => {
                        // Plain `=`: rhs must match the swizzle slot's type.
                        self.context.unify(&rhs_type, &swizzle_ty).map_err(|_| {
                            err_type_at!(
                                value.h.span,
                                "`with .{}` expects {}, got {}",
                                format_swizzle_str(components),
                                self.format_type(&swizzle_ty),
                                self.format_type(&rhs_type.apply(&self.context))
                            )
                        })?;
                    }
                    Some(binop) => {
                        // Compound `op=`: type `swizzle_ty <op> rhs` and
                        // require the result to equal swizzle_ty.
                        let combined = self.infer_binop_result(
                            binop,
                            swizzle_ty.clone(),
                            rhs_type,
                            expr.h.span,
                        )?;
                        self.context.unify(&combined, &swizzle_ty).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "compound `with .{} {}=` must produce {}, got {}",
                                format_swizzle_str(components),
                                binop,
                                self.format_type(&swizzle_ty),
                                self.format_type(&combined.apply(&self.context))
                            )
                        })?;
                    }
                }

                Ok(target_type.apply(&self.context))
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;
                self.infer_binop_result(&op.op, left_type, right_type, expr.h.span)
            }
            ExprKind::Tuple(elements) => {
                let elem_types: Result<Vec<Type>> =
                    elements.iter().map(|e| self.infer_expression(e)).collect();

                Ok(tuple(elem_types?))
            }
            ExprKind::Lambda(lambda) => self.type_lambda(lambda, None, expr),
            ExprKind::LetIn(let_in) => {
                // Infer type of the value expression
                let value_type = self.infer_expression(&let_in.value)?;

                let resolved_annotation = let_in
                    .ty
                    .as_ref()
                    .map(|ty| self.normalize_annotation_type(ty, self.current_module.as_deref()));
                if let Some(declared_type) = &resolved_annotation {
                    self.unify_or_err_weakening(
                        &value_type,
                        declared_type,
                        let_in.value.h.span,
                        "Type mismatch in let binding",
                    )?;
                }

                // Push new scope and bind pattern
                self.scope_stack.push_scope();
                let bound_type = resolved_annotation.unwrap_or_else(|| value_type.clone());

                // Open existential types: ?k. T becomes T[k'/k] where k' is fresh
                let bound_type = self.open_existential(bound_type);

                // Bind all names in the pattern.
                // Let bindings should be generalized for polymorphism.
                // Refutability is enforced — refutable patterns must use
                // `match` instead.
                self.bind_irrefutable_pattern(&let_in.pattern, &bound_type, true)?;

                // Infer type of body expression
                let body_type = self.infer_expression(&let_in.body)?;

                // Pop scope
                self.scope_stack.pop_scope();

                Ok(body_type)
            }
            ExprKind::Application(func, args) => {
                // Resolve callee to candidate function types
                let callee = self.resolve_callee_candidates(func)?;

                // For single candidate, use apply_two_pass (handles lambdas correctly)
                // For multiple candidates (overloads), use inference-first approach
                if callee.candidates.len() == 1 {
                    let cand = callee.candidates.into_iter().next().unwrap();

                    // Use two-pass for bidirectional lambda checking
                    let result_ty = self.apply_two_pass(cand.ty.clone(), args)?;

                    // Check for partial application AFTER application (when types are resolved)
                    self.ensure_not_partial(&result_ty, &expr.h.span)?;

                    // Store resolved type in type table (apply substitutions)
                    let resolved = cand.ty.apply(&self.context);
                    self.type_table
                        .insert(func.h.id, TypeScheme::Monotype(resolved));
                    self.type_table
                        .insert(expr.h.id, TypeScheme::Monotype(result_ty.clone()));
                    Ok(result_ty)
                } else {
                    // Multiple candidates: infer argument types first, then try overloads
                    let mut arg_types = Vec::new();
                    for arg in args {
                        arg_types.push(self.infer_expression(arg)?);
                    }

                    let candidate_tys: Vec<Type> =
                        callee.candidates.iter().map(|c| c.ty.clone()).collect();
                    let span = expr.h.span;
                    let resolved = crate::builtins::overload::resolve_overload(
                        &candidate_tys,
                        &arg_types,
                        &mut self.context,
                    );
                    match resolved {
                        Ok(r) => {
                            self.ensure_not_partial(&r.return_type, &span)?;
                            let resolved_func_ty = candidate_tys[r.winner_index].apply(&self.context);
                            self.type_table
                                .insert(func.h.id, TypeScheme::Monotype(resolved_func_ty));
                            self.type_table
                                .insert(expr.h.id, TypeScheme::Monotype(r.return_type.clone()));
                            // Record the chosen overload so downstream IR
                            // (TLC `Var(Builtin)`, `PureOp::Intrinsic`, ...) can
                            // dispatch on the right `BuiltinDef::overloads()` entry.
                            self.name_resolution.set_overload_idx(func.h.id, r.winner_index);
                            // If this was a scalar constructor call
                            // (`i32(x)`, etc.), record the chosen
                            // per-type conversion catalog entry.
                            self.record_constructor_dispatch(func.h.id, r.winner_index);
                            return Ok(r.return_type);
                        }
                        Err(_) => bail_type_at!(
                            expr.h.span,
                            "No matching overload for '{}' with argument types: {}",
                            callee.display_name,
                            arg_types
                                .iter()
                                .map(|t| self.format_type(t))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    }
                }
            }
            ExprKind::FieldAccess(inner_expr, field) => {
                // `name_resolution::run` rewrites every module-style
                // `mod.name` chain into `Identifier([mod], name)` before
                // type-check, so by the time we see `FieldAccess` here
                // it's a genuine record field access. (The previous
                // `try_extract_qual_name` recovery path was a defensive
                // shim from before that rewrite; it's now unreachable.)
                let base_type = self.infer_expression(inner_expr)?;

                // Apply context and strip uniqueness
                let base_type = base_type.apply(&self.context);

                // Use unified field access helper
                let field_type = self.infer_field_access(&base_type, field, &expr.h.span)?;
                self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                Ok(field_type)
            }
            ExprKind::If(if_expr) => {
                // Infer condition type - should be bool
                let condition_ty = self.infer_expression(&if_expr.condition)?;
                let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

                // Unify condition with bool type
                self.context.unify(&condition_ty, &bool_ty).map_err(|_| {
                    err_type_at!(
                        if_expr.condition.h.span,
                        "If condition must be boolean, got: {}",
                        self.format_type(&condition_ty)
                    )
                })?;

                // Infer then and else branch types - they must be the same.
                // Branch values carry no `*` (uniqueness is signature-only),
                // so they unify directly.
                let then_ty = self.infer_expression(&if_expr.then_branch)?;
                let else_ty = self.infer_expression(&if_expr.else_branch)?;

                self.unify_or_err(
                    &then_ty,
                    &else_ty,
                    if_expr.else_branch.h.span,
                    "If branches have incompatible types",
                )?;
                let result_ty = then_ty.apply(&self.context);

                // Futhark restriction: functions cannot be returned from if expressions
                let resolved_result = result_ty.apply(&self.context);
                if as_arrow(&resolved_result).is_some() {
                    bail_type_at!(
                        expr.h.span,
                        "Functions cannot be returned from if expressions"
                    );
                }

                Ok(result_ty)
            }

            ExprKind::UnaryOp(op, operand) => {
                let operand_type = self.infer_expression(operand)?;
                let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
                match op.op.as_str() {
                    "-" => {
                        // Numeric negation - operand must be numeric
                        let resolved = operand_type.apply(&self.context);
                        if let Some(false) = Self::is_numeric_type(&resolved) {
                            return Err(err_type_at!(
                                operand.h.span,
                                "Unary minus requires numeric operand, got {}",
                                self.format_type(&resolved)
                            ));
                        }
                        Ok(resolved)
                    }
                    "!" => {
                        // Logical not - operand must be bool, returns bool
                        self.context.unify(&operand_type, &bool_ty).map_err(|_| {
                            err_type_at!(
                                operand.h.span,
                                "Logical not (!) requires bool operand, got {:?}",
                                operand_type
                            )
                        })?;
                        Ok(bool_ty)
                    }
                    _ => Err(err_type_at!(expr.h.span, "Unknown unary operator: {}", op.op)),
                }
            }

            ExprKind::Loop(loop_expr) => {
                // Push a new scope for loop variables
                self.scope_stack.push_scope();

                // Get or infer the type of the loop variable from init
                let loop_var_type = if let Some(init) = &loop_expr.init {
                    self.infer_expression(init)?
                } else {
                    // No init - create a fresh type variable
                    self.context.new_variable()
                };

                // Futhark restriction: loop parameters cannot be functions
                let resolved_loop_var = loop_var_type.apply(&self.context);
                if as_arrow(&resolved_loop_var).is_some() {
                    bail_type_at!(
                        expr.h.span,
                        "Loop parameters cannot be functions"
                    );
                }

                // Bind pattern to the loop variable type.
                self.bind_irrefutable_pattern(&loop_expr.pattern, &loop_var_type, false)?;

                // Type check the loop form
                match &loop_expr.form {
                    LoopForm::While(cond) => {
                        // Condition must be bool
                        let cond_type = self.infer_expression(cond)?;
                        self.context.unify(&cond_type, &bool_type()).map_err(|_| {
                            err_type_at!(
                                cond.h.span,
                                "While condition must be bool, got {}",
                                self.format_type(&cond_type)
                            )
                        })?;
                    }
                    LoopForm::For(var_name, bound) => {
                        // Iteration variable is i32
                        self.define(var_name.clone(), IdentifierKind::Local, TypeScheme::Monotype(i32()));

                        // Bound must be integer
                        let bound_type = self.infer_expression(bound)?;
                        self.context.unify(&bound_type, &i32()).map_err(|_| {
                            err_type_at!(
                                bound.h.span,
                                "Loop bound must be i32, got {}",
                                self.format_type(&bound_type)
                            )
                        })?;
                    }
                    LoopForm::ForIn(pat, arr) => {
                        // Array must be an array type: Array[elem, addrspace, size]
                        let arr_type = self.infer_expression(arr)?;
                        let (elem_type, _, _, _) =
                            self.constrain_array_type(&arr_type, &arr.h.span, "for-in requires an array")?;

                        // Bind pattern to element type.
                        self.bind_irrefutable_pattern(pat, &elem_type, false)?;
                    }
                }

                // Type check the body - its type must match the loop variable type
                let body_type = self.infer_expression(&loop_expr.body)?;
                self.context.unify(&body_type, &loop_var_type).map_err(|e| {
                    err_type_at!(
                        loop_expr.body.h.span,
                        "Loop body type must match loop variable type: {:?}",
                        e
                    )
                })?;

                // Pop the scope
                self.scope_stack.pop_scope();

                // The loop returns the loop variable type
                Ok(loop_var_type)
            }

            ExprKind::Match(match_expr) => {
                // Type the scrutinee. Any type is permitted — coverage
                // analysis (Maranget) handles the per-type universe.
                let scrutinee_ty = self.infer_expression(&match_expr.scrutinee)?;

                if match_expr.cases.is_empty() {
                    bail_type_at!(expr.h.span, "match expression must have at least one arm");
                }

                // All arms produce a common result type. Check each arm
                // body against a shared variable so an earlier arm's type
                // flows into later arms — a bare constructor or an
                // ambiguous literal arm resolves from its siblings.
                let result_var = self.context.new_variable();
                for case in &match_expr.cases {
                    self.scope_stack.push_scope();
                    self.bind_pattern(&case.pattern, &scrutinee_ty, false)?;
                    self.check_expression(&case.body, &result_var)?;
                    self.scope_stack.pop_scope();
                }
                let result_ty = result_var;

                // Coverage analysis: exhaustiveness + redundancy.
                let arms: Vec<(Pattern, Span)> = match_expr
                    .cases
                    .iter()
                    .map(|c| (c.pattern.clone(), c.pattern.h.span))
                    .collect();
                let scrutinee_applied = scrutinee_ty.apply(&self.context);
                match check_match(&scrutinee_applied, &arms, expr.h.span) {
                    Ok(()) => {}
                    Err(CoverageError::NonExhaustive { missing, .. }) => bail_type_at!(
                        expr.h.span,
                        "non-exhaustive match: missing case {}",
                        format_cov_pat(&missing)
                    ),
                    Err(CoverageError::Redundant { arm_span, .. }) => {
                        bail_type_at!(arm_span, "unreachable match arm")
                    }
                }

                // Futhark restriction: functions cannot be returned from
                // match expressions (as for `if`).
                let result = result_ty.apply(&self.context);
                if as_arrow(&result).is_some() {
                    bail_type_at!(expr.h.span, "Functions cannot be returned from match expressions");
                }

                Ok(result)
            }

            // A bare `#name(args)` doesn't pin down which sum type it
            // belongs to — `#some(3)` is a value of any sum type that
            // declares a `#some` variant carrying an `i32` payload.
            // Use a type ascription, an annotated `let`, or context
            // (function argument, return type) to disambiguate, which
            // routes through `check_expression`'s Constructor arm.
            ExprKind::Constructor(name, _) => Err(err_type_at!(
                expr.h.span,
                "ambiguous constructor `#{}`: cannot determine which sum type it belongs to. \
                 Add a type annotation or use it where a sum type is expected.",
                name
            )),

            ExprKind::Range(range) => {
                // Range expressions produce an array of integers
                // All operands must be the same integer type
                let start_type = self.infer_expression(&range.start)?;
                let end_type = self.infer_expression(&range.end)?;

                // Unify start and end types
                self.context.unify(&start_type, &end_type).map_err(|_| {
                    err_type_at!(
                        expr.h.span,
                        "Range start and end must have the same type: {} vs {}",
                        self.format_type(&start_type.apply(&self.context)),
                        self.format_type(&end_type.apply(&self.context))
                    )
                })?;

                // Check start is an integer type (i32, u32, etc.)
                let resolved_start = start_type.apply(&self.context);
                if !crate::types::is_integer_type(&resolved_start) {
                    // If still a type variable, default to i32
                    if matches!(resolved_start, Type::Variable(_)) {
                        self.context.unify(&start_type, &i32()).map_err(|_| {
                            err_type_at!(
                                range.start.h.span,
                                "Range operands must be integer types, got {}",
                                self.format_type(&start_type.apply(&self.context))
                            )
                        })?;
                    } else {
                        return Err(err_type_at!(
                            range.start.h.span,
                            "Range operands must be integer types, got {}",
                            self.format_type(&resolved_start)
                        ));
                    }
                }

                // Check step type if present
                let step_val = if let Some(step) = &range.step {
                    let step_type = self.infer_expression(step)?;
                    self.context.unify(&step_type, &start_type).map_err(|_| {
                        err_type_at!(
                            step.h.span,
                            "Range step must match start/end type: expected {}, got {}",
                            self.format_type(&start_type.apply(&self.context)),
                            self.format_type(&step_type.apply(&self.context))
                        )
                    })?;
                    Self::try_extract_const_int(step)
                } else {
                    Some(1) // Default step is 1
                };

                // Try to compute a concrete size if bounds are constant
                let start_val = Self::try_extract_const_int(&range.start);
                let end_val = Self::try_extract_const_int(&range.end);

                let size_type = match (start_val, end_val, step_val) {
                    (Some(start), Some(end), Some(step)) if step != 0 => {
                        // Calculate size based on range kind
                        let size = match range.kind {
                            RangeKind::Inclusive => (end - start) / step + 1,
                            RangeKind::Exclusive | RangeKind::ExclusiveLt => (end - start) / step,
                            RangeKind::ExclusiveGt => (start - end) / step,
                        };
                        if size > 0 {
                            Type::Constructed(TypeName::Size(size as usize), vec![])
                        } else {
                            self.context.new_variable()
                        }
                    }
                    _ => self.context.new_variable(),
                };

                let elem_type = start_type.apply(&self.context);
                // Range literals produce virtual arrays (struct {start, step, len})
                let addrspace = Type::Constructed(TypeName::ArrayVariantVirtual, vec![]);
                Ok(Type::Constructed(
                    TypeName::Array,
                    vec![elem_type, addrspace, size_type, no_buffer()],
                ))
            }

            ExprKind::Slice(slice) => {
                // Slice expression: array[start:end]
                // - array must be Array[elem, addrspace, size]
                // - start/end (if present) must be integers
                // - result is Array[elem, addrspace, size'] where size' = end - start

                let array_type = self.infer_expression(&slice.array)?;
                let array_type_stripped = array_type.clone();

                // Constrain array to be Array[elem, addrspace, size, region]
                let (elem_var, addrspace_var, _, buffer_var) = self.constrain_array_type(
                    &array_type_stripped,
                    &slice.array.h.span,
                    "Cannot slice non-array type",
                )?;

                // Type-check start bound if present
                if let Some(start) = &slice.start {
                    let start_type = self.infer_expression(start)?;
                    self.context.unify(&start_type, &i32()).map_err(|_| {
                        err_type_at!(
                            start.h.span,
                            "Slice start must be an integer, got {}",
                            self.format_type(&start_type.apply(&self.context))
                        )
                    })?;
                }

                // Type-check end bound if present
                if let Some(end) = &slice.end {
                    let end_type = self.infer_expression(end)?;
                    self.context.unify(&end_type, &i32()).map_err(|_| {
                        err_type_at!(
                            end.h.span,
                            "Slice end must be an integer, got {}",
                            self.format_type(&end_type.apply(&self.context))
                        )
                    })?;
                }

                // Compute result size from constant bounds, or use fresh variable
                let result_size = match Self::try_extract_slice_size(slice) {
                    Some(size) => Type::Constructed(TypeName::Size(size), vec![]),
                    None => self.context.new_variable(),
                };

                let elem_type = elem_var.apply(&self.context);
                let addrspace = addrspace_var.apply(&self.context);
                // Slices preserve representation: a storage-backed view slice
                // is another view with adjusted offset/len, not an eager
                // materialized Composite value. The size slot still records a
                // constant length when both bounds are literal.
                let result_buffer = array_type_stripped
                    .array_buffer()
                    .cloned()
                    .or_else(|| array_type_stripped.apply(&self.context).array_buffer().cloned())
                    .unwrap_or_else(|| buffer_var.apply(&self.context));
                Ok(Type::Constructed(
                    TypeName::Array,
                    vec![elem_type, addrspace, result_size, result_buffer],
                ))
            }

            ExprKind::TypeAscription(inner, ascribed_ty) => {
                // Type ascription: check the inner expression against the ascribed type
                // This allows integer literals to take on the ascribed type (e.g., 42u32)
                let normalized =
                    self.normalize_annotation_type(ascribed_ty, self.current_module.as_deref());
                self.check_expression(inner, &normalized)?;
                Ok(normalized)
            }

            ExprKind::TypeCoercion(_, _) => {
                Err(err_type_at!(
                    expr.h.span,
                    "type coercion (:>) is not yet supported"
                ))
            }
        } // NEWCASESHERE - add new cases before this closing brace
        ?;

        // Store the inferred type in the type table
        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
        Ok(ty)
    }

    // Removed: fresh_var - now using polytype's context.new_variable()

    /// Resolve a callee expression to a set of candidate function types.
    /// For overloaded intrinsics, returns multiple candidates.
    /// For normal functions, returns a single candidate.
    fn resolve_callee_candidates(&mut self, func: &Expression) -> Result<CalleeCandidates> {
        match &func.kind {
            ExprKind::Identifier(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                let is_qualified = !quals.is_empty();

                if let Some(resolved) = self.resolve_value_name(&full_name, is_qualified, func.h.id) {
                    // For overloaded intrinsics, expand into multiple candidates
                    let candidates = if let Some(overloads) = resolved.overloads {
                        overloads
                            .into_iter()
                            .map(|scheme| {
                                let ty = scheme.instantiate(&mut self.context);
                                Candidate { ty }
                            })
                            .collect()
                    } else {
                        vec![Candidate {
                            ty: resolved.instantiated,
                        }]
                    };

                    return Ok(CalleeCandidates {
                        candidates,
                        display_name: resolved.display_name,
                    });
                }

                // Constructor-style conversion: `i32(x)`, `vec2i32(v)`, etc.
                // The parser already accepts `Application(Identifier([], T), [v])`;
                // we intercept here when the name isn't a known value but
                // matches a type-constructor pattern.
                if quals.is_empty() {
                    if let Some(candidates) = self.try_resolve_constructor_call(name, func.h.id) {
                        return Ok(candidates);
                    }
                }

                Err(err_undef_at!(func.h.span, "{}", full_name))
            }

            // Any non-identifier callee: infer its type as single candidate
            _ => {
                let ty = self.infer_expression(func)?;
                Ok(CalleeCandidates {
                    candidates: vec![Candidate { ty }],
                    display_name: "<expr>".to_string(),
                })
            }
        }
    }

    /// Constructor-style conversion dispatch (`i32(x)`, `vec2i32(v)`, …).
    ///
    /// The parser-level surface already accepts `T(value)` as a normal
    /// `Application` node; the standard identifier-resolution path
    /// returns `None` because `T` isn't a value name (it's a type
    /// module or a vec shorthand). This hook intercepts that failure
    /// and routes the call to one of two dispatch paths:
    ///
    /// **Scalar.** If `name` is a known type module (`i32`, `f32`,
    /// `bool`, …), enumerate every catalog entry whose surface name
    /// is `<name>.<source>` via
    /// `catalog::lookup_by_surface_prefix(name)`. Build a
    /// multi-candidate result so the existing `resolve_overload`
    /// machinery picks the right `<name>.<source>` based on the
    /// argument's inferred type. After `infer_application` records
    /// the winning index, `record_constructor_dispatch` writes the
    /// chosen `BuiltinId` back into `name_resolution.values` so
    /// downstream IR sees the resolved per-type conversion catalog
    /// entry.
    ///
    /// **Vec.** If `name` matches a parser-level vec shorthand
    /// (`vec2i32`, `vec3f32`, `vec4u32`, …), build a single-candidate
    /// `vec<a, n> -> vec<target_elem, n>` scheme and record a
    /// `ResolvedValueRef::VecConstructor` in `name_resolution.values`
    /// for the callee. `to_tlc::transform_application` reads that
    /// record and desugars to a `VecLit` of componentwise scalar
    /// conversion calls.
    ///
    /// Returns `None` if the name doesn't match either pattern — the
    /// caller falls through to the existing "undefined" error path.
    fn try_resolve_constructor_call(
        &mut self,
        name: &str,
        callee_node_id: NodeId,
    ) -> Option<CalleeCandidates> {
        // Scalar dispatch.
        if self.module_manager.is_known_module(name) {
            let catalog = crate::builtins::catalog();
            let entries = catalog.lookup_by_surface_prefix(name);
            let mut candidates: Vec<Candidate> = Vec::with_capacity(entries.len());
            let mut catalog_ids: Vec<BuiltinId> = Vec::with_capacity(entries.len());
            for def in &entries {
                // Only `per_type_conv` entries are valid `T(value)`
                // dispatch targets. The same prefix scan also turns up
                // operators (`T.+`, `T.abs`, …) — those are kind
                // `Operator`/other; including them would let overload
                // resolution pick a 2-arg `T.+` and report the result
                // as a partial application.
                if !matches!(def.raw.kind, crate::builtins::catalog::BuiltinKind::ModuleBuiltin) {
                    continue;
                }
                // Per-type conversion entries (`per_type_conv` in
                // `builtins/defs.rs`) carry `scheme: None`; their
                // schemes come from prelude module signatures via
                // `lookup_module_scheme_by_id`.
                if let Some(scheme) = self.lookup_module_scheme_by_id(def.id) {
                    let ty = scheme.instantiate(&mut self.context);
                    candidates.push(Candidate { ty });
                    catalog_ids.push(def.id);
                }
            }
            if !candidates.is_empty() {
                self.constructor_call_catalog_ids.insert(callee_node_id, catalog_ids);
                return Some(CalleeCandidates {
                    candidates,
                    display_name: format!("{}(...)", name),
                });
            }
            // Name matched a module but no `<name>.<X>` entries exist
            // — fall through; the caller emits an undefined-name
            // error which is the right diagnostic.
            return None;
        }

        // Vec dispatch.
        let vec_constructors = crate::types::vector_type_constructors();
        if let Some(target_ty) = vec_constructors.get(name) {
            // Parse arity and component-type-name from the target ty.
            let (arity, target_elem) = match target_ty {
                Type::Constructed(TypeName::Vec, args) if args.len() == 2 => {
                    let arity = match &args[1] {
                        Type::Constructed(TypeName::Size(n), _) => *n,
                        _ => return None,
                    };
                    let target_elem = type_name_to_module(&args[0])?;
                    (arity, target_elem)
                }
                _ => return None,
            };

            // Build a synthetic scheme: `vec<a, arity> -> target_ty`.
            // The arg's element type `a` is left free so the type
            // checker accepts any vec-of-arity-n on the arg side; the
            // componentwise desugar at to_tlc time looks up the
            // per-component conversion catalog entry and surfaces a
            // clean error if the source-elem can't be converted.
            let arg_elem = self.context.new_variable();
            let arg_vec = Type::Constructed(
                TypeName::Vec,
                vec![arg_elem, Type::Constructed(TypeName::Size(arity), vec![])],
            );
            let func_ty = Type::Constructed(TypeName::Arrow, vec![arg_vec, target_ty.clone()]);

            // Record the dispatch so `to_tlc` can desugar.
            self.name_resolution.values.insert(
                callee_node_id,
                crate::name_resolution::ResolvedValueRef::VecConstructor {
                    target_name: name.to_string(),
                    arity,
                    target_elem,
                },
            );

            return Some(CalleeCandidates {
                candidates: vec![Candidate { ty: func_ty }],
                display_name: name.to_string(),
            });
        }

        None
    }

    /// After `resolve_overload` picks a winning candidate in
    /// `infer_application`, record the corresponding catalog
    /// `BuiltinId` in `name_resolution.values` so downstream IR
    /// (TLC `Var(Builtin)`, the SPIR-V dispatcher, …) can dispatch
    /// on the right per-type conversion entry. No-op when the
    /// callee wasn't a scalar constructor call.
    fn record_constructor_dispatch(&mut self, callee_node_id: NodeId, winner_index: usize) {
        if let Some(catalog_ids) = self.constructor_call_catalog_ids.remove(&callee_node_id) {
            if let Some(&chosen_id) = catalog_ids.get(winner_index) {
                self.name_resolution.values.insert(
                    callee_node_id,
                    crate::name_resolution::ResolvedValueRef::Builtin {
                        id: chosen_id,
                        // `per_type_conv` builtins have exactly one
                        // overload each — the catalog overload index
                        // within a single `BuiltinDef` is always 0.
                        overload_idx: Some(0),
                    },
                );
            }
        }
    }

    /// Ensure the result type is not a function (no partial application).
    /// Note: () -> T is allowed (unit functions are not considered partial application).
    fn ensure_not_partial(&self, result_ty: &Type, call_span: &Span) -> Result<()> {
        let r = result_ty.apply(&self.context);
        if let Some((param, _)) = as_arrow(&r) {
            // Unit parameter doesn't count as partial application
            if !matches!(param, Type::Constructed(TypeName::Unit, _)) {
                bail_type_at!(
                    *call_span,
                    "Partial application not allowed: result is function type {}",
                    self.format_type(&r)
                );
            }
        }
        Ok(())
    }

    /// Unify two types or produce a "<context>, got A and B" type error.
    fn unify_or_err(&mut self, a: &Type, b: &Type, span: Span, ctx: &str) -> Result<()> {
        self.context.unify(a, b).map_err(|_| {
            err_type_at!(
                span,
                "{}, got {} and {}",
                ctx,
                self.format_type(&a.apply(&self.context)),
                self.format_type(&b.apply(&self.context))
            )
        })
    }

    /// Shape-check a value against a contract that may carry `*`.
    /// Uniqueness is a signature-level property, absent from expression
    /// types, so forget any `*` in the expected contract's value
    /// positions and unify. Whether an alias-free (`*`) result is
    /// actually supplied is a provenance question left to `tlc::ownership`.
    fn unify_or_err_weakening(
        &mut self,
        actual: &Type,
        expected: &Type,
        span: Span,
        ctx: &str,
    ) -> Result<()> {
        let expected = expected.apply(&self.context);
        self.unify_or_err(actual, &expected, span, ctx)
    }

    /// Destructure a `Mat` type into `(elem, cols, rows)`. None for non-Mat.
    fn mat_parts(t: &Type) -> Option<(&Type, &Type, &Type)> {
        match t {
            Type::Constructed(TypeName::Mat, args) if args.len() == 3 => {
                Some((&args[0], &args[1], &args[2]))
            }
            _ => None,
        }
    }

    /// Destructure a `Vec` type into `(elem, size)`. None for non-Vec.
    fn vec_parts(t: &Type) -> Option<(&Type, &Type)> {
        match t {
            Type::Constructed(TypeName::Vec, args) if args.len() == 2 => Some((&args[0], &args[1])),
            _ => None,
        }
    }

    /// Compute the result type of a binary operator given pre-inferred
    /// operand types. Shared between `ExprKind::BinaryOp` and the
    /// compound-form arm of `ExprKind::VecWith`, which needs to know
    /// the type of `target.swizzle <op> rhs` without rebuilding an AST.
    fn infer_binop_result(
        &mut self,
        op: &str,
        left_type: Type,
        right_type: Type,
        span: Span,
    ) -> Result<Type> {
        let bool_ty = || Type::Constructed(TypeName::Bool, vec![]);

        match op {
            "==" | "!=" | "<" | ">" | "<=" | ">=" => {
                self.unify_or_err(
                    &left_type,
                    &right_type,
                    span,
                    &format!("Operator '{}' requires same-typed operands", op),
                )?;
                Ok(bool_ty())
            }
            "&&" | "||" => {
                let bt = bool_ty();
                let ctx = format!("Logical operator '{}' requires bool operands", op);
                self.unify_or_err(&left_type, &bt, span, &ctx)?;
                self.unify_or_err(&right_type, &bt, span, &ctx)?;
                Ok(bt)
            }
            "+" | "-" | "*" | "/" | "%" | "**" => {
                self.infer_arith_op_result(op, left_type, right_type, span)
            }
            "&" | "|" | "^" | "<<" | ">>" => {
                self.unify_or_err(
                    &left_type,
                    &right_type,
                    span,
                    &format!("Bitwise operator '{}' requires same-typed operands", op),
                )?;
                let t = left_type.apply(&self.context);
                match t {
                    Type::Constructed(TypeName::Int(_), _) | Type::Constructed(TypeName::UInt(_), _) => {
                        Ok(t)
                    }
                    _ => Err(err_type_at!(
                        span,
                        "Bitwise operator '{}' requires integer operands, got {}",
                        op,
                        self.format_type(&t)
                    )),
                }
            }
            _ => Err(err_type_at!(span, "Unknown binary operator: {}", op)),
        }
    }

    /// Result type for arithmetic operators (`+ - * / % **`). Handles
    /// matrix-mul (only for `*`), vec-vec / vec-scalar / scalar-scalar
    /// dispatch, and the numeric-elem check.
    fn infer_arith_op_result(
        &mut self,
        op: &str,
        left_type: Type,
        right_type: Type,
        span: Span,
    ) -> Result<Type> {
        let l = left_type.apply(&self.context);
        let r = right_type.apply(&self.context);

        // `*` covers matrix products that don't reduce to plain
        // component-wise arithmetic. Try those first.
        if op == "*" {
            if let Some(ty) = self.try_mat_mul(&l, &r, span)? {
                return Ok(ty);
            }
        }

        // Vec-vec, vec-scalar, scalar-vec, or scalar-scalar.
        let result = match (Self::vec_parts(&l), Self::vec_parts(&r)) {
            (Some(_), Some(_)) => {
                self.unify_or_err(
                    &left_type,
                    &right_type,
                    span,
                    "Vector arithmetic requires matching vector types",
                )?;
                l.clone()
            }
            (Some((le, _)), None) => {
                self.unify_or_err(le, &r, span, "Vector-scalar element type must match scalar")?;
                l.clone()
            }
            (None, Some((re, _))) => {
                self.unify_or_err(&l, re, span, "Scalar-vector scalar must match element type")?;
                r.clone()
            }
            (None, None) => {
                // `**` exception: a float base may take an integer exponent of
                // any width. Result is the base's float type. See
                // SPECIFICATION.md `x binop y`. The (Int base, Float exp) shape
                // stays a same-typed error, matching the spec's wording.
                if op == "**"
                    && matches!(l, Type::Constructed(TypeName::Float(_), _))
                    && matches!(
                        r,
                        Type::Constructed(TypeName::Int(_), _) | Type::Constructed(TypeName::UInt(_), _)
                    )
                {
                    l.clone()
                } else {
                    self.unify_or_err(
                        &left_type,
                        &right_type,
                        span,
                        &format!("Operator '{}' requires same-typed operands", op),
                    )?;
                    l.clone()
                }
            }
        };

        // Operands (or vec elements) must be numeric.
        let check = Self::vec_parts(&result).map(|(e, _)| e).unwrap_or(&result);
        if let Some(false) = Self::is_numeric_type(check) {
            return Err(err_type_at!(
                span,
                "Arithmetic operator '{}' requires numeric operands, got {}",
                op,
                self.format_type(check)
            ));
        }

        Ok(result)
    }

    /// Try the matrix-product family for `*`. Returns `Ok(Some(_))` if
    /// the operands match a Mat-Mat / Mat-Vec / Vec-Mat / Mat-scalar /
    /// scalar-Mat case (with unification performed); `Ok(None)` if
    /// neither operand is a Mat, leaving the caller to dispatch to
    /// component-wise arithmetic.
    fn try_mat_mul(&mut self, l: &Type, r: &Type, span: Span) -> Result<Option<Type>> {
        let lm = Self::mat_parts(l);
        let rm = Self::mat_parts(r);
        let lv = Self::vec_parts(l);
        let rv = Self::vec_parts(r);

        let result = match (lm, rm, lv, rv) {
            // Mat × Mat: inner dims must match (left rows = right cols).
            (Some((le, lc, lr)), Some((re, rc, rr)), _, _) => {
                self.unify_or_err(lr, rc, span, "Matrix multiply inner dimensions must match")?;
                self.unify_or_err(le, re, span, "Matrix multiply element types must match")?;
                Some(Type::Constructed(
                    TypeName::Mat,
                    vec![le.clone(), lc.clone(), rr.clone()],
                ))
            }
            // Mat × Vec: matrix rows must equal vec size; result is Vec[cols].
            (Some((le, lc, lr)), _, _, Some((re, rs))) => {
                self.unify_or_err(
                    lr,
                    rs,
                    span,
                    "Matrix-vector multiply: matrix rows must match vector size",
                )?;
                self.unify_or_err(le, re, span, "Matrix-vector multiply element types must match")?;
                Some(Type::Constructed(TypeName::Vec, vec![le.clone(), lc.clone()]))
            }
            // Vec × Mat: vec size must equal matrix cols; result is Vec[rows].
            (_, Some((re, rc, rr)), Some((le, ls)), _) => {
                self.unify_or_err(
                    ls,
                    rc,
                    span,
                    "Vector-matrix multiply: vector size must match matrix cols",
                )?;
                self.unify_or_err(le, re, span, "Vector-matrix multiply element types must match")?;
                Some(Type::Constructed(TypeName::Vec, vec![le.clone(), rr.clone()]))
            }
            // Mat × scalar / scalar × Mat: element types must match.
            (Some((le, _, _)), None, _, None) => {
                self.unify_or_err(le, r, span, "Matrix-scalar element type must match scalar")?;
                Some(l.clone())
            }
            (None, Some((re, _, _)), None, _) => {
                self.unify_or_err(l, re, span, "Scalar-matrix scalar must match element type")?;
                Some(r.clone())
            }
            _ => None,
        };

        Ok(result)
    }

    /// Peel one arrow from a function type and unify an argument with the expected param.
    ///
    /// Returns the result type after application.
    fn unify_apply_arg(
        &mut self,
        func_ty: &Type,
        arg_ty: &Type,
        arg: &Expression,
        arg_index: usize,
    ) -> Result<Type> {
        // Create fresh variables for param and result
        let param_var = self.context.new_variable();
        let result_var = self.context.new_variable();
        let arrow_type = Type::arrow(param_var.clone(), result_var.clone());

        // Unify function type with arrow
        self.context.unify(func_ty, &arrow_type).map_err(|_| {
            err_type_at!(
                arg.h.span,
                "Cannot apply {} as a function",
                self.format_type(func_ty)
            )
        })?;

        // Get expected param type after unification
        let expected_param = param_var.apply(&self.context);

        let arg_applied = arg_ty.apply(&self.context);
        let param_applied = expected_param.apply(&self.context);

        // Open the argument's existential type at the use site if the
        // expected parameter is not itself existential. Existential
        // elimination fires both here, at function-application argument
        // positions, and at `let` binders, so `reduce(_, _, filter(...))`
        // inline unifies the same way `let kept = filter(...) in
        // reduce(_, _, kept)` does. Don't open when the param is itself
        // existential — the existential type can flow through unchanged
        // in that case.
        let is_existential = |ty: &Type| matches!(ty, Type::Constructed(TypeName::Existential(_), _));
        let arg_for_check = if is_existential(&arg_applied) && !is_existential(&param_applied) {
            self.open_existential(arg_applied)
        } else {
            arg_applied
        };

        // Compare value shapes. Whether the argument is actually legal to
        // consume is an alias/provenance question checked by `tlc::ownership`
        // from the callee's diet: a fresh local array is consumable, an
        // observing parameter is not, though both have the same value type.
        let checkpoint = self.context.len();
        if self.context.unify(&arg_for_check, &param_applied).is_err() {
            self.context.rollback(checkpoint);
            let base = format!(
                "Function argument type mismatch at argument {}: expected {}, got {}",
                arg_index + 1,
                self.format_type(&param_applied),
                self.format_type(&arg_for_check),
            );
            let error_msg = if arg.h.span.is_generated() {
                format!("{base}\nGenerated expression: {:#?}", arg)
            } else {
                base
            };
            return Err(err_type_at!(arg.h.span, "{}", error_msg));
        }

        Ok(result_var)
    }

    /// Constrain a type to be an Array and return its components.
    ///
    /// Creates fresh type variables for elem, addrspace, and size, then unifies
    /// the given type with Array[elem, addrspace, size]. Returns the three
    /// component type variables.
    ///
    /// Caller should strip uniqueness before calling if needed.
    /// Returns `(elem, variant, size, region)` vars after unifying `array_ty`
    /// with a fresh `Array[_, _, _, _]`. The region var lets a view's buffer
    /// flow to slice/for-in results without being fabricated.
    fn constrain_array_type(
        &mut self,
        array_ty: &Type,
        span: &Span,
        error_context: &str,
    ) -> Result<(Type, Type, Type, Type)> {
        let elem_var = self.context.new_variable();
        let addrspace_var = self.context.new_variable();
        let size_var = self.context.new_variable();
        let buffer_var = self.context.new_variable();
        let want_array = Type::Constructed(
            TypeName::Array,
            vec![
                elem_var.clone(),
                addrspace_var.clone(),
                size_var.clone(),
                buffer_var.clone(),
            ],
        );

        self.context.unify(array_ty, &want_array).map_err(|_| {
            err_type_at!(
                *span,
                "{}: got {}",
                error_context,
                self.format_type(&array_ty.apply(&self.context))
            )
        })?;

        Ok((elem_var, addrspace_var, size_var, buffer_var))
    }

    /// Infer the type of a field access on a base type.
    ///
    /// Precedence:
    /// 1. Record fields (by TypeName::Record field map)
    /// 2. Tuple numeric indices (.0, .1, etc.)
    /// 3. Vec swizzle (x/y/z/w) - constrains to Vec if type is unknown
    /// 4. Known type fields via impl_source / record_field_map
    ///
    /// The base_ty should already be applied and uniqueness-stripped.
    fn infer_field_access(&mut self, base_ty: &Type, field: &str, span: &Span) -> Result<Type> {
        // 1. Record field access
        if let Type::Constructed(TypeName::Record(fields), field_types) = base_ty {
            if let Some(field_index) = fields.get_index(field) {
                if field_index < field_types.len() {
                    return Ok(field_types[field_index].clone());
                }
            }
            bail_type_at!(
                *span,
                "Record type has no field '{}'. Available: {}",
                field,
                fields.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            );
        }

        // 2. Tuple numeric index (.0, .1, etc.)
        if let Ok(index) = field.parse::<usize>() {
            if let Type::Constructed(TypeName::Tuple(_), elem_types) = base_ty {
                if index < elem_types.len() {
                    return Ok(elem_types[index].clone());
                }
                bail_type_at!(
                    *span,
                    "Tuple index {} out of bounds (tuple has {} elements)",
                    index,
                    elem_types.len()
                );
            }
            bail_type_at!(
                *span,
                "Numeric field access '.{}' requires a tuple type, got {}",
                index,
                self.format_type(base_ty)
            );
        }

        // 3. Vec swizzle: single-letter returns a scalar, 2–4-letter
        //    returns a vec of that length. Each letter is drawn from
        //    ONE of the two swizzle sets — `xyzw` (position) or `rgba`
        //    (color, aliased to position: r=x, g=y, b=z, a=w). Mixing
        //    sets (`.xg`) is rejected at the predicate level. Each
        //    letter must also be in range for the source vec's length
        //    (`.z` on a vec2 is a range error, caught here when the
        //    size is known, otherwise only that the base is a vec).
        if super::is_swizzle_field(field) {
            // Constrain base to some Vec<_, _>.
            let elem_var = self.context.new_variable();
            let size_var = self.context.new_variable();
            let want_vec = Type::Constructed(TypeName::Vec, vec![elem_var.clone(), size_var.clone()]);
            self.context.unify(base_ty, &want_vec).map_err(|_| {
                err_type_at!(
                    *span,
                    "Field '{}' requires a vector type, got {}",
                    field,
                    self.format_type(&base_ty.apply(&self.context))
                )
            })?;

            // If the size is resolved, bounds-check the swizzle letters.
            let resolved_base = base_ty.apply(&self.context);
            if let Type::Constructed(TypeName::Vec, args) = &resolved_base {
                if let Some(Type::Constructed(TypeName::Size(n), _)) = args.get(1) {
                    for c in field.chars() {
                        let idx = super::swizzle_component_index(c)
                            .expect("is_swizzle_field already accepted this letter");
                        if idx >= *n {
                            bail_type_at!(
                                *span,
                                "Swizzle '.{}' uses component '{}' (index {}) but source is vec{}",
                                field,
                                c,
                                idx,
                                n
                            );
                        }
                    }
                }
            }

            let elem = elem_var.apply(&self.context);
            if field.len() == 1 {
                return Ok(elem);
            }
            // Multi-letter swizzle → a new vec of that length.
            return Ok(Type::Constructed(
                TypeName::Vec,
                vec![elem, Type::Constructed(TypeName::Size(field.len()), vec![])],
            ));
        }

        // 4. Known type fields via impl_source / record_field_map
        if let Type::Constructed(type_name, _) = base_ty {
            let type_name_str = match type_name {
                TypeName::Bool => "bool".to_string(),
                TypeName::Float(bits) => format!("f{}", bits),
                TypeName::UInt(bits) => format!("u{}", bits),
                TypeName::Int(bits) => format!("i{}", bits),
                TypeName::Vec => "vec".to_string(),
                TypeName::Mat => "mat".to_string(),
                TypeName::Array => "array".to_string(),
                TypeName::Named(name) => name.clone(),
                _ => {
                    bail_type_at!(
                        *span,
                        "Type {} has no field '{}'",
                        self.format_type(base_ty),
                        field
                    );
                }
            };

            if let Some(field_type) = crate::types::vec_field_type(&type_name_str, field) {
                return Ok(field_type);
            }
            if let Some(field_type) = self.record_field_map.get(&(type_name_str.clone(), field.to_string()))
            {
                return Ok(field_type.clone());
            }

            bail_type_at!(*span, "Type '{}' has no field '{}'", type_name_str, field);
        }

        bail_type_at!(
            *span,
            "Field access '{}' not supported on type {}",
            field,
            self.format_type(base_ty)
        )
    }

    /// Two-pass function application for better lambda inference
    ///
    /// Pass 1: Process non-lambda arguments to constrain type variables
    /// Pass 2: Process lambda arguments with bidirectionally checked expected types
    ///
    /// This allows map (\x -> ...) arr to infer properly regardless of argument order
    fn apply_two_pass(&mut self, mut func_type: Type, args: &[Expression]) -> Result<Type> {
        // Track expected types for lambda arguments (for second pass)
        let mut lambda_expected_types: Vec<Option<Type>> = vec![None; args.len()];

        // First pass: process arguments to constrain type variables
        for (i, arg) in args.iter().enumerate() {
            // A consuming function may not be passed as a value: reject an
            // argument that names a top-level consuming function.
            if let ExprKind::Identifier(path, name) = &arg.kind {
                let qualified =
                    if path.is_empty() { name.clone() } else { format!("{}.{}", path.join("."), name) };
                if self.consuming_defs.contains(name) || self.consuming_defs.contains(&qualified) {
                    bail_type_at!(
                        arg.h.span,
                        "a consuming function may not be passed as a value: `{}` consumes its argument",
                        name
                    );
                }
            }
            if let ExprKind::Lambda(lambda) = &arg.kind {
                // For lambdas with k params: build k arrows for the expected lambda type
                // This ensures check_expression can extract all k parameter types
                let num_params = lambda.params.len();
                let final_result = self.context.new_variable();

                // Build expected lambda type: α1 -> α2 -> ... -> αk -> β
                let mut expected_lambda_type = final_result;
                for _ in 0..num_params {
                    let param_var = self.context.new_variable();
                    expected_lambda_type = Type::arrow(param_var, expected_lambda_type);
                }

                // Peel one arrow from func_type: (expected_lambda_type -> result)
                let result_type = self.context.new_variable();
                let expected_func_type = Type::arrow(expected_lambda_type.clone(), result_type.clone());

                self.context.unify(&func_type, &expected_func_type).map_err(|_| {
                    err_type_at!(
                        arg.h.span,
                        "Cannot apply {} as a function",
                        self.format_type(&func_type)
                    )
                })?;

                // Store the whole expected lambda type (not just first param)
                lambda_expected_types[i] = Some(expected_lambda_type);

                func_type = result_type;
            } else if matches!(&arg.kind, ExprKind::Constructor(_, _)) {
                // Constructor expressions need bidirectional checking against
                // the expected parameter type to disambiguate which sum type
                // they belong to — `infer_expression` produces an "ambiguous
                // constructor" error.
                let param_var = self.context.new_variable();
                let result_var = self.context.new_variable();
                let arrow_type = Type::arrow(param_var.clone(), result_var.clone());
                self.context.unify(&func_type, &arrow_type).map_err(|_| {
                    err_type_at!(
                        arg.h.span,
                        "Cannot apply {} as a function",
                        self.format_type(&func_type)
                    )
                })?;
                let expected_param = param_var.apply(&self.context);
                self.check_expression(arg, &expected_param)?;
                func_type = result_var;
            } else {
                // For non-lambda argument: infer type and unify with expected param
                let arg_type = self.infer_expression(arg)?;
                func_type = self.unify_apply_arg(&func_type, &arg_type, arg, i)?;
            }
        }

        // Second pass: process lambda arguments with bidirectional checking
        for (i, arg) in args.iter().enumerate() {
            if !matches!(&arg.kind, ExprKind::Lambda(_)) {
                continue;
            }

            // Get the expected type from first pass
            let expected_param_type = lambda_expected_types[i].as_ref().map(|t| t.apply(&self.context));

            // Use bidirectional checking for lambdas
            if let Some(ref expected) = expected_param_type {
                self.check_expression(arg, expected)?;
            } else {
                self.infer_expression(arg)?;
            }
        }

        // A `*` return is a signature-level contract; the applied value
        // is an ordinary observation, so no expression type carries it.
        Ok(func_type.apply(&self.context))
    }
}

#[cfg(test)]
#[path = "checker_tests.rs"]
mod checker_tests;
