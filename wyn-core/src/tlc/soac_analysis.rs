//! TLC-level SOAC (Second-Order Array Combinator) analysis.
//!
//! This module analyzes TLC programs to identify SOAC patterns and their
//! symbolic sizes. The analysis runs before defunctionalization (when lambdas
//! are still inline) to capture the full SOAC structure.
//!
//! Key features:
//! - Identifies all SOAC intrinsic applications
//! - Produces symbolic size expressions for input/output arrays
//! - Tracks nesting depth for nested SOACs
//! - Propagates size hints from `#[size_hint(N)]` attributes
//! - Interprocedural analysis starting from compute entry points

use std::collections::{HashMap, HashSet};

use crate::ast::{self, Attribute, PatternKind, Span};
use crate::scope::ScopeStack;
use crate::types::TypeName;
use polytype::Type;

use super::{Def, DefMeta, Program, Term, TermId, TermKind};

// =============================================================================
// Symbolic Size Expressions
// =============================================================================

/// Symbolic representation of array sizes.
///
/// Used to track and propagate size information through the analysis
/// without requiring concrete values at compile time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SizeExpr {
    /// Known constant size.
    Const(usize),
    /// Named size variable (from type system: `[n]T`).
    Var(String),
    /// Size hint from `#[size_hint(N)]` attribute.
    Hint(u32),
    /// Unknown/runtime size.
    Unknown,
    /// Product of sizes (for flattening nested arrays).
    Product(Box<SizeExpr>, Box<SizeExpr>),
    /// Bounded by another size (for filter results, scatter).
    BoundedBy(Box<SizeExpr>),
}

impl std::fmt::Display for SizeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizeExpr::Const(n) => write!(f, "{}", n),
            SizeExpr::Var(name) => write!(f, "{}", name),
            SizeExpr::Hint(n) => write!(f, "hint({})", n),
            SizeExpr::Unknown => write!(f, "?"),
            SizeExpr::Product(a, b) => write!(f, "({} * {})", a, b),
            SizeExpr::BoundedBy(inner) => write!(f, "≤{}", inner),
        }
    }
}

// =============================================================================
// SOAC Kind Classification
// =============================================================================

/// Classification of SOAC intrinsics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoacKind {
    /// `map(f, arr)` - applies f to each element, preserves size.
    Map,
    /// `inplace_map(f, arr)` - in-place map variant, preserves size.
    InplaceMap,
    /// `reduce(op, ne, arr)` - reduces array to single value.
    Reduce,
    /// `scan(op, ne, arr)` - prefix scan, preserves size.
    Scan,
    /// `filter(pred, arr)` - filters elements, output size bounded by input.
    Filter,
    /// `scatter(dest, indices, values)` - scattered writes.
    Scatter,
    /// `hist_1d(dest, op, ne, indices, values)` - histogram/reduce-by-index.
    Hist1D,
    /// `zip(arr1, arr2)` - zips two arrays.
    Zip,
}

impl SoacKind {
    /// Parse a SOAC kind from an intrinsic name.
    pub fn from_intrinsic_name(name: &str) -> Option<Self> {
        match name {
            // Intrinsic names
            "_w_intrinsic_map" | "map" => Some(SoacKind::Map),
            "_w_intrinsic_inplace_map" | "inplace_map" => Some(SoacKind::InplaceMap),
            "_w_intrinsic_reduce" | "reduce" => Some(SoacKind::Reduce),
            "_w_intrinsic_scan" | "scan" => Some(SoacKind::Scan),
            "_w_intrinsic_filter" | "filter" => Some(SoacKind::Filter),
            "_w_intrinsic_scatter" | "scatter" => Some(SoacKind::Scatter),
            "_w_intrinsic_hist_1d" | "hist_1d" => Some(SoacKind::Hist1D),
            "_w_intrinsic_zip" | "zip" => Some(SoacKind::Zip),
            _ => None,
        }
    }

    /// Returns true if this SOAC can be parallelized independently.
    pub fn is_independent(&self) -> bool {
        matches!(self, SoacKind::Map | SoacKind::InplaceMap | SoacKind::Zip)
    }

    /// Returns the expected number of arguments for a SOAC intrinsic application.
    pub fn expected_args(&self) -> usize {
        match self {
            SoacKind::Map | SoacKind::InplaceMap => 2, // (f, arr)
            SoacKind::Reduce | SoacKind::Scan => 3,    // (op, ne, arr)
            SoacKind::Filter => 2,                     // (pred, arr)
            SoacKind::Scatter => 3,                    // (dest, indices, values)
            SoacKind::Hist1D => 5,                     // (dest, op, ne, indices, values)
            SoacKind::Zip => 2,                        // (arr1, arr2)
        }
    }
}

impl std::fmt::Display for SoacKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            SoacKind::Map => "map",
            SoacKind::InplaceMap => "inplace_map",
            SoacKind::Reduce => "reduce",
            SoacKind::Scan => "scan",
            SoacKind::Filter => "filter",
            SoacKind::Scatter => "scatter",
            SoacKind::Hist1D => "hist_1d",
            SoacKind::Zip => "zip",
        };
        write!(f, "{}", name)
    }
}

// =============================================================================
// SOAC Information
// =============================================================================

/// Information about a SOAC in TLC.
#[derive(Debug, Clone)]
pub struct TlcSoacInfo {
    /// Source location for error messages.
    pub span: Span,
    /// The term ID of the SOAC application.
    pub term_id: TermId,
    /// Which SOAC intrinsic.
    pub kind: SoacKind,
    /// Input array size expression.
    pub input_size: SizeExpr,
    /// Output size expression (same for map, bounded for filter, 1 for reduce).
    pub output_size: SizeExpr,
    /// Nesting depth (0 = top-level, 1 = inside one SOAC, etc.).
    pub nesting_depth: u32,
    /// Parent SOAC if nested.
    pub parent: Option<TermId>,
    /// True if the input array traces back to an entry parameter.
    /// This is used by the parallelization pass to determine if the SOAC
    /// can be chunked across threads.
    pub maps_entry_input: bool,
}

// =============================================================================
// Analysis Result
// =============================================================================

/// Result of SOAC analysis on a TLC program.
#[derive(Debug)]
pub struct SoacAnalysis {
    /// SOACs reachable from each compute entry point.
    /// Key: entry point name, Value: all SOACs found (including in called functions).
    pub by_entry: HashMap<String, Vec<TlcSoacInfo>>,
}

impl SoacAnalysis {
    /// Create an empty analysis result.
    pub fn new() -> Self {
        Self {
            by_entry: HashMap::new(),
        }
    }

    /// Returns true if any compute entry point has SOACs.
    pub fn has_soacs(&self) -> bool {
        self.by_entry.values().any(|soacs| !soacs.is_empty())
    }

    /// Get all SOACs for a specific entry point.
    pub fn get_entry_soacs(&self, entry_name: &str) -> Option<&[TlcSoacInfo]> {
        self.by_entry.get(entry_name).map(|v| v.as_slice())
    }

    /// Get total count of SOACs across all entry points.
    pub fn total_soac_count(&self) -> usize {
        self.by_entry.values().map(|v| v.len()).sum()
    }
}

impl Default for SoacAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Analyzer
// =============================================================================

/// Analyzer state for SOAC detection.
pub struct SoacAnalyzer<'a> {
    /// All definitions in the program (for resolving calls).
    all_defs: HashMap<&'a str, &'a Def>,
    /// Size environment: variable name -> size expression.
    pub(crate) env: ScopeStack<SizeExpr>,
    /// Entry parameter provenance: variable name -> is from entry param.
    /// Variables that trace back to entry parameters are candidates for parallelization.
    pub(crate) is_entry_param: ScopeStack<bool>,
    /// Collected SOAC info for the current entry point.
    pub(crate) results: Vec<TlcSoacInfo>,
    /// Current nesting depth.
    nesting_depth: u32,
    /// Parent SOAC (if inside one).
    parent_soac: Option<TermId>,
    /// Functions currently being analyzed (for recursion detection).
    analyzing: HashSet<String>,
}

impl<'a> SoacAnalyzer<'a> {
    pub fn new(all_defs: HashMap<&'a str, &'a Def>) -> Self {
        Self {
            all_defs,
            env: ScopeStack::new(),
            is_entry_param: ScopeStack::new(),
            results: Vec::new(),
            nesting_depth: 0,
            parent_soac: None,
            analyzing: HashSet::new(),
        }
    }

    /// Reset the analyzer for a new entry point.
    fn reset(&mut self) {
        self.env = ScopeStack::new();
        self.is_entry_param = ScopeStack::new();
        self.results.clear();
        self.nesting_depth = 0;
        self.parent_soac = None;
        self.analyzing.clear();
    }

    /// Analyze a term for SOAC patterns.
    pub(crate) fn analyze_term(&mut self, term: &Term) {
        match &term.kind {
            TermKind::App { func, arg } => {
                // Check if this is a SOAC application
                if let Some((soac_kind, args)) = self.parse_soac_call(term) {
                    self.handle_soac(term, soac_kind, &args);
                    return;
                }

                // Check if this is a call to a user function
                if let Some((fn_name, args)) = self.uncurry_call(term) {
                    // Avoid infinite recursion
                    if !self.analyzing.contains(&fn_name) {
                        if let Some(def) = self.all_defs.get(fn_name.as_str()).copied() {
                            self.analyze_function_call(def, &args);
                        }
                    }
                }

                // Also analyze the function and argument
                self.analyze_term(func);
                self.analyze_term(arg);
            }

            TermKind::Let { name, rhs, body, .. } => {
                // Analyze RHS
                self.analyze_term(rhs);

                // Extract size from RHS and bind
                let rhs_size = self.size_of_term(rhs);
                // Track whether RHS traces back to entry param
                let is_entry = self.is_from_entry_param(rhs);
                self.env.push_scope();
                self.is_entry_param.push_scope();
                self.env.insert(name.clone(), rhs_size);
                self.is_entry_param.insert(name.clone(), is_entry);
                self.analyze_term(body);
                self.is_entry_param.pop_scope();
                self.env.pop_scope();
            }

            TermKind::Lam { param, body, .. } => {
                self.env.push_scope();
                self.is_entry_param.push_scope();
                self.env.insert(param.clone(), SizeExpr::Unknown);
                // Lambda parameters are not entry params
                self.is_entry_param.insert(param.clone(), false);
                self.analyze_term(body);
                self.is_entry_param.pop_scope();
                self.env.pop_scope();
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.analyze_term(cond);
                self.analyze_term(then_branch);
                self.analyze_term(else_branch);
            }

            TermKind::Loop {
                init,
                body,
                kind,
                init_bindings,
                ..
            } => {
                self.analyze_term(init);
                for (_, _, binding_term) in init_bindings {
                    self.analyze_term(binding_term);
                }

                match kind {
                    super::LoopKind::For { iter, .. } => self.analyze_term(iter),
                    super::LoopKind::ForRange { bound, .. } => self.analyze_term(bound),
                    super::LoopKind::While { cond } => self.analyze_term(cond),
                }

                self.analyze_term(body);
            }

            // Literals and variables don't contain SOACs
            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_) => {}
        }
    }

    /// Handle a detected SOAC call.
    fn handle_soac(&mut self, term: &Term, kind: SoacKind, args: &[&Term]) {
        // Determine which argument is the "main" array for size and entry param check
        let (input_size, input_array, lambda_to_analyze) = match kind {
            SoacKind::Map | SoacKind::InplaceMap => {
                // map(f, arr) - arr is args[1]
                let arr = args.get(1).copied();
                let arr_size = arr.map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown);
                let lambda = args.first().copied();
                (arr_size, arr, lambda)
            }
            SoacKind::Reduce | SoacKind::Scan => {
                // reduce(op, ne, arr) - arr is args[2]
                let arr = args.get(2).copied();
                let arr_size = arr.map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown);
                let lambda = args.first().copied();
                (arr_size, arr, lambda)
            }
            SoacKind::Filter => {
                // filter(pred, arr) - arr is args[1]
                let arr = args.get(1).copied();
                let arr_size = arr.map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown);
                let lambda = args.first().copied();
                (arr_size, arr, lambda)
            }
            SoacKind::Scatter => {
                // scatter(dest, indices, values) - indices determines iteration count
                let arr = args.get(1).copied();
                let idx_size = arr.map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown);
                (idx_size, arr, None)
            }
            SoacKind::Hist1D => {
                // hist_1d(dest, op, ne, indices, values) - indices determines iteration count
                let arr = args.get(3).copied();
                let idx_size = arr.map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown);
                let lambda = args.get(1).copied();
                (idx_size, arr, lambda)
            }
            SoacKind::Zip => {
                // zip(arr1, arr2) - both should be same size
                let arr = args.first().copied();
                let arr_size = arr.map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown);
                (arr_size, arr, None)
            }
        };

        // Compute output size
        let output_size = self.output_size_for_soac(kind, &input_size);

        // Check if the input array traces back to an entry parameter
        let maps_entry_input = input_array.map(|a| self.is_from_entry_param(a)).unwrap_or(false);

        // Record this SOAC
        self.results.push(TlcSoacInfo {
            span: term.span,
            term_id: term.id,
            kind,
            input_size,
            output_size,
            nesting_depth: self.nesting_depth,
            parent: self.parent_soac,
            maps_entry_input,
        });

        // Analyze lambda body at increased nesting depth
        if let Some(lambda_term) = lambda_to_analyze {
            let old_depth = self.nesting_depth;
            let old_parent = self.parent_soac;
            self.nesting_depth += 1;
            self.parent_soac = Some(term.id);

            self.analyze_term(lambda_term);

            self.nesting_depth = old_depth;
            self.parent_soac = old_parent;
        }
    }

    /// Analyze a function call, propagating size info and entry param status to the callee.
    fn analyze_function_call(&mut self, def: &Def, args: &[&Term]) {
        // Mark as analyzing to prevent recursion
        self.analyzing.insert(def.name.clone());

        // Extract parameter names from curried lambdas
        let param_names = extract_param_names(&def.body);

        // Push scope and bind params to arg sizes and entry param status
        self.env.push_scope();
        self.is_entry_param.push_scope();
        for (param, arg) in param_names.iter().zip(args.iter()) {
            let arg_size = self.size_of_term(arg);
            let is_entry = self.is_from_entry_param(arg);
            self.env.insert(param.clone(), arg_size);
            self.is_entry_param.insert(param.clone(), is_entry);
        }

        // Analyze callee body, skipping outer lambdas (params already bound)
        let inner_body = skip_lambdas(&def.body, param_names.len());
        self.analyze_term(inner_body);

        self.is_entry_param.pop_scope();
        self.env.pop_scope();
        self.analyzing.remove(&def.name);
    }

    /// Try to parse a SOAC call from a term.
    ///
    /// Returns the SOAC kind and collected arguments if this is a SOAC application.
    fn parse_soac_call<'t>(&self, term: &'t Term) -> Option<(SoacKind, Vec<&'t Term>)> {
        let (fn_name, args) = self.uncurry_call(term)?;
        let kind = SoacKind::from_intrinsic_name(&fn_name)?;

        // Verify we have the expected number of arguments
        if args.len() == kind.expected_args() { Some((kind, args)) } else { None }
    }

    /// Uncurry a call: turn `App(App(App(f, a), b), c)` into `(f, [a, b, c])`.
    fn uncurry_call<'t>(&self, term: &'t Term) -> Option<(String, Vec<&'t Term>)> {
        let mut args = Vec::new();
        let mut current = term;

        while let TermKind::App { func, arg } = &current.kind {
            args.push(arg.as_ref());
            current = func.as_ref();
        }

        if let TermKind::Var(name) = &current.kind {
            args.reverse();
            Some((name.clone(), args))
        } else {
            None
        }
    }

    /// Compute the size of a term's result.
    fn size_of_term(&self, term: &Term) -> SizeExpr {
        match &term.kind {
            TermKind::Var(name) => {
                // Look up in environment
                self.env.lookup(name).cloned().unwrap_or_else(|| {
                    // Try to extract from the term's type
                    extract_size_from_type(&term.ty)
                })
            }

            TermKind::App { .. } => {
                // Check if this is a SOAC call
                if let Some((soac_kind, args)) = self.parse_soac_call(term) {
                    let input_size = match soac_kind {
                        SoacKind::Map | SoacKind::InplaceMap | SoacKind::Filter => {
                            args.get(1).map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown)
                        }
                        SoacKind::Reduce | SoacKind::Scan => {
                            args.get(2).map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown)
                        }
                        SoacKind::Scatter => {
                            args.first().map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown)
                        }
                        SoacKind::Hist1D => {
                            args.first().map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown)
                        }
                        SoacKind::Zip => {
                            args.first().map(|a| self.size_of_term(a)).unwrap_or(SizeExpr::Unknown)
                        }
                    };
                    return self.output_size_for_soac(soac_kind, &input_size);
                }

                // Not a SOAC - extract from type
                extract_size_from_type(&term.ty)
            }

            // For other terms, extract from type
            _ => extract_size_from_type(&term.ty),
        }
    }

    /// Compute output size for a given SOAC kind.
    pub fn output_size_for_soac(&self, kind: SoacKind, input: &SizeExpr) -> SizeExpr {
        match kind {
            // These preserve size
            SoacKind::Map | SoacKind::InplaceMap | SoacKind::Scan | SoacKind::Zip => input.clone(),
            // Reduce produces a scalar
            SoacKind::Reduce => SizeExpr::Const(1),
            // Filter/Scatter output size is bounded by input
            SoacKind::Filter | SoacKind::Scatter => SizeExpr::BoundedBy(Box::new(input.clone())),
            // Hist1D output size is the dest size (first arg), not indices
            SoacKind::Hist1D => SizeExpr::Unknown,
        }
    }

    /// Check if a term traces back to an entry parameter.
    ///
    /// This is used to determine if a SOAC's input array comes from an entry
    /// parameter, which means it can be parallelized by chunking.
    fn is_from_entry_param(&self, term: &Term) -> bool {
        match &term.kind {
            TermKind::Var(name) => {
                // Look up in entry param tracking environment
                self.is_entry_param.lookup(name).copied().unwrap_or(false)
            }
            // For other terms, we conservatively return false.
            // In the future, we could trace through let bindings in the term itself,
            // but for now we rely on the scope tracking done during analyze_term.
            _ => false,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Extract size info from an array type.
pub fn extract_size_from_type(ty: &Type<TypeName>) -> SizeExpr {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            // Array[elem, addrspace, size]
            match &args[2] {
                Type::Constructed(TypeName::Size(n), _) => SizeExpr::Const(*n),
                Type::Constructed(TypeName::SizeVar(name), _) => SizeExpr::Var(name.clone()),
                Type::Variable(_) => SizeExpr::Unknown, // Size determined by parallelization
                _ => SizeExpr::Unknown,
            }
        }
        _ => SizeExpr::Unknown,
    }
}

/// Extract parameter names from nested Lam structure.
fn extract_param_names(term: &Term) -> Vec<String> {
    let mut params = Vec::new();
    let mut current = term;
    while let TermKind::Lam { param, body, .. } = &current.kind {
        params.push(param.clone());
        current = body.as_ref();
    }
    params
}

/// Skip N outer lambda wrappers and return the inner body.
fn skip_lambdas(term: &Term, count: usize) -> &Term {
    let mut current = term;
    for _ in 0..count {
        if let TermKind::Lam { body, .. } = &current.kind {
            current = body.as_ref();
        } else {
            break;
        }
    }
    current
}

/// Extract size hints from entry point parameters.
fn extract_size_hints(entry: &ast::EntryDecl) -> HashMap<String, u32> {
    let mut hints = HashMap::new();
    for param in &entry.params {
        if let Some((name, hint)) = extract_param_size_hint(param) {
            hints.insert(name, hint);
        }
    }
    hints
}

/// Extract a size hint from a single parameter pattern.
fn extract_param_size_hint(pattern: &ast::Pattern) -> Option<(String, u32)> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            let hint = attrs.iter().find_map(|a| match a {
                Attribute::SizeHint(n) => Some(*n),
                _ => None,
            });
            if let Some(h) = hint {
                inner.simple_name().map(|name| (name.to_string(), h))
            } else {
                extract_param_size_hint(inner)
            }
        }
        PatternKind::Typed(inner, _) => extract_param_size_hint(inner),
        _ => None,
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Analyze a TLC program for SOAC patterns.
///
/// This finds all `#[compute]` entry points and traces through the call graph
/// to find all reachable SOACs. Size hints from parameters are propagated
/// through function calls.
pub fn analyze_program(program: &Program) -> SoacAnalysis {
    // Build def map
    let all_defs: HashMap<&str, &Def> = program.defs.iter().map(|d| (d.name.as_str(), d)).collect();

    let mut analysis = SoacAnalysis::new();

    // Find compute entry points
    for def in &program.defs {
        if let DefMeta::EntryPoint(entry) = &def.meta {
            if entry.entry_type.is_compute() {
                let mut analyzer = SoacAnalyzer::new(all_defs.clone());
                analyzer.reset();

                // Extract size hints from entry params
                let hints = extract_size_hints(entry);
                let param_names = extract_param_names(&def.body);

                // Initialize environment with hinted sizes and mark as entry params
                for param_name in &param_names {
                    let size =
                        hints.get(param_name).map(|h| SizeExpr::Hint(*h)).unwrap_or(SizeExpr::Unknown);
                    analyzer.env.insert(param_name.clone(), size);
                    // All entry point parameters are entry params (can be parallelized)
                    analyzer.is_entry_param.insert(param_name.clone(), true);
                }

                // Analyze the entry body, skipping the outer lambdas (params already bound)
                let inner_body = skip_lambdas(&def.body, param_names.len());
                analyzer.analyze_term(inner_body);

                analysis.by_entry.insert(def.name.clone(), analyzer.results);
            }
        }
    }

    analysis
}
