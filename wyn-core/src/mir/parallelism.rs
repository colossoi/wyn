//! Parallelism analysis for SOAC (Second-Order Array Combinator) operations.
//!
//! This module analyzes MIR to identify SOAC intrinsics and classify them
//! based on their parallelization characteristics. It also provides detection
//! for simple compute shader patterns that can be parallelized.

use super::{Body, Def, EntryInput, ExecutionModel, Expr, ExprId, LocalId};
use crate::types::TypeName;
use polytype::Type;

/// Classification of a SOAC's parallelization characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismKind {
    /// Each output element can be computed independently.
    /// Examples: map, zip
    Independent,
    /// Output elements have data dependencies on each other.
    /// Examples: reduce, scan, filter, scatter, hist_1d
    Dependent,
}

/// Information about a SOAC found in the MIR.
#[derive(Debug, Clone)]
pub struct SoacInfo {
    /// The expression ID of the SOAC call.
    pub expr_id: ExprId,
    /// The intrinsic name (e.g., "_w_intrinsic_map").
    pub soac_name: String,
    /// The parallelism classification.
    pub parallelism: ParallelismKind,
}

/// Result of parallelism analysis on a MIR body.
#[derive(Debug)]
pub struct ParallelismAnalysis {
    /// All SOACs found in the body.
    pub soacs: Vec<SoacInfo>,
}

impl ParallelismAnalysis {
    /// Analyze a MIR body to find and classify all SOAC intrinsics.
    pub fn analyze(body: &Body) -> Self {
        let mut soacs = Vec::new();

        for (idx, expr) in body.exprs.iter().enumerate() {
            if let Expr::Call { func, .. } = expr {
                if let Some(parallelism) = classify_soac(func) {
                    soacs.push(SoacInfo {
                        expr_id: ExprId(idx as u32),
                        soac_name: func.clone(),
                        parallelism,
                    });
                }
            }
        }

        ParallelismAnalysis { soacs }
    }

    /// Returns true if all SOACs in the body are independent (embarrassingly parallel).
    pub fn all_independent(&self) -> bool {
        self.soacs.iter().all(|s| s.parallelism == ParallelismKind::Independent)
    }

    /// Returns the count of independent vs dependent SOACs.
    pub fn counts(&self) -> (usize, usize) {
        let independent =
            self.soacs.iter().filter(|s| s.parallelism == ParallelismKind::Independent).count();
        let dependent = self.soacs.len() - independent;
        (independent, dependent)
    }
}

// =============================================================================
// Simple Compute Map Detection
// =============================================================================

/// Information about a detected simple compute map pattern.
///
/// This pattern is:
/// - A `#[compute]` entry point
/// - With one or more slice input parameters
/// - Whose body is sequential setup (Let bindings) followed by a single `map` call
#[derive(Debug, Clone)]
pub struct SimpleComputeMap {
    /// The workgroup size.
    pub local_size: (u32, u32, u32),
    /// All input parameters (slices). First one is the primary input for the map.
    pub inputs: Vec<InputSliceInfo>,
    /// The ExprId of the map call.
    pub map_expr: ExprId,
    /// The closure/lambda being mapped.
    pub map_closure: ExprId,
    /// The array argument to map.
    pub map_array: ExprId,
}

/// Information about an input slice parameter.
#[derive(Debug, Clone)]
pub struct InputSliceInfo {
    /// LocalId of the input parameter.
    pub local: LocalId,
    /// Name of the parameter.
    pub name: String,
    /// Element type of the slice.
    pub element_type: Type<TypeName>,
    /// Size hint if provided (from `#[size_hint(N)]`).
    pub size_hint: Option<u32>,
}

/// Detect if a definition is a simple compute shader with a map.
///
/// Returns `Some(SimpleComputeMap)` if the definition matches the pattern:
/// - `#[compute]` entry point
/// - One or more slice input parameters
/// - Body is Let bindings followed by a single `_w_intrinsic_map` call
pub fn detect_simple_compute_map(def: &Def) -> Option<SimpleComputeMap> {
    // Must be an EntryPoint with Compute execution model
    let (execution_model, inputs, body) = match def {
        Def::EntryPoint {
            execution_model,
            inputs,
            body,
            ..
        } => (execution_model, inputs, body),
        _ => return None,
    };

    let local_size = match execution_model {
        ExecutionModel::Compute { local_size } => *local_size,
        _ => return None,
    };

    // Must have at least one input that is a slice
    if inputs.is_empty() {
        return None;
    }

    // Collect all slice inputs
    let mut input_infos = Vec::new();
    for input in inputs {
        if let Some(info) = analyze_input_slice(input) {
            input_infos.push(info);
        }
    }

    // Must have at least one slice input
    if input_infos.is_empty() {
        return None;
    }

    // Analyze body structure: should be sequential Let bindings + single map
    let (map_expr, map_closure, map_array) = find_single_map_at_root(body)?;

    Some(SimpleComputeMap {
        local_size,
        inputs: input_infos,
        map_expr,
        map_closure,
        map_array,
    })
}

/// Check if an input is a slice/array type suitable for compute shaders and extract its info.
fn analyze_input_slice(input: &EntryInput) -> Option<InputSliceInfo> {
    // Array[elem, addrspace, size] - check if size is Unsized (unsized array)
    let element_type = match &input.ty {
        Type::Constructed(TypeName::Array, args) if args.len() >= 3 => {
            // Array[elem, addrspace, size] - check if size is Unsized
            match &args[2] {
                Type::Constructed(TypeName::Unsized, _) => args[0].clone(),
                _ => return None, // Fixed-size arrays not supported for compute
            }
        }
        // Legacy 2-arg format: Array[size, elem]
        Type::Constructed(TypeName::Array, args) if args.len() == 2 => {
            match &args[0] {
                Type::Constructed(TypeName::Unsized, _) => args[1].clone(),
                _ => return None,
            }
        }
        _ => return None,
    };

    // TODO: Extract size_hint from attributes if present
    // For now, we don't have attribute info on EntryInput

    Some(InputSliceInfo {
        local: input.local,
        name: input.name.clone(),
        element_type,
        size_hint: None,
    })
}

/// Walk the body from root to find a single map call at the "end" of setup.
///
/// The allowed pattern is:
/// ```text
/// let a = ... in
/// let b = ... in
/// ...
/// _w_intrinsic_map(closure, array)
/// ```
///
/// Returns (map_expr_id, closure_arg, array_arg) if found.
fn find_single_map_at_root(body: &Body) -> Option<(ExprId, ExprId, ExprId)> {
    let mut current = body.root;

    // Walk through Let bindings
    loop {
        match body.get_expr(current) {
            Expr::Let { body: inner, .. } => {
                current = *inner;
            }
            Expr::Call { func, args }
                if func == "_w_intrinsic_map" || func == "_w_intrinsic_inplace_map" =>
            {
                // Found the map! Should have 2 args: closure, array
                if args.len() == 2 {
                    return Some((current, args[0], args[1]));
                }
                return None;
            }
            // Any other expression at root level means this isn't a simple pattern
            _ => return None,
        }
    }
}

// =============================================================================
// SOAC Classification
// =============================================================================

/// Classify a function name as a SOAC intrinsic and determine its parallelism kind.
/// Returns None if the function is not a SOAC intrinsic.
fn classify_soac(name: &str) -> Option<ParallelismKind> {
    use ParallelismKind::*;

    match name {
        // Independent: each output element can be computed without knowledge of others
        "_w_intrinsic_map" | "_w_intrinsic_inplace_map" => Some(Independent),
        "_w_intrinsic_zip" => Some(Independent),

        // Dependent: output elements depend on other elements or require coordination
        "_w_intrinsic_reduce" => Some(Dependent), // accumulation to single value
        "_w_intrinsic_scan" => Some(Dependent),   // prefix sum, each depends on prior
        "_w_intrinsic_filter" => Some(Dependent), // output size is data-dependent
        "_w_intrinsic_scatter" => Some(Dependent), // multiple writes may target same index
        "_w_intrinsic_hist_1d" => Some(Dependent), // reduce-by-index, same bin conflicts

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_soac() {
        assert_eq!(
            classify_soac("_w_intrinsic_map"),
            Some(ParallelismKind::Independent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_zip"),
            Some(ParallelismKind::Independent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_reduce"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_scan"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_filter"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_scatter"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_hist_1d"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(classify_soac("some_other_function"), None);
    }
}
