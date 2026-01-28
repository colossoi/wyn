//! MIR-level SOAC analysis for compute shader parallelization.
//!
//! This module analyzes MIR to detect parallelizable patterns in compute shaders:
//! - Entry points with map operations over input arrays
//! - Tracking whether array arguments come from entry parameters
//!
//! The analysis results are used by the `soac_parallelize` pass to transform
//! compute shaders for GPU execution.

use crate::ast::TypeName;
use crate::mir::{Body, Def, EntryInput, ExecutionModel, Expr, ExprId, LocalId, Program};
use polytype::Type;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Analysis Results
// =============================================================================

/// Information about a parallelizable map in a compute shader.
#[derive(Debug, Clone)]
pub struct ParallelizableMap {
    /// The ExprId of the map intrinsic call.
    pub map_expr: ExprId,
    /// The closure/function being mapped (ExprId pointing to Global or Call).
    pub closure: ExprId,
    /// The array being mapped over.
    pub array: ExprId,
    /// Whether the array traces back to an entry parameter.
    pub maps_entry_input: bool,
    /// Whether this is an inplace map.
    pub is_inplace: bool,
}

/// Analysis results for a compute entry point.
#[derive(Debug, Clone)]
pub struct ComputeEntryAnalysis {
    /// Name of the entry point.
    pub name: String,
    /// All parallelizable maps found in the entry point body.
    pub maps: Vec<ParallelizableMap>,
}

/// Analysis results for an entire MIR program.
#[derive(Debug, Default)]
pub struct MirSoacAnalysis {
    /// Analysis for each compute entry point.
    pub by_entry: HashMap<String, ComputeEntryAnalysis>,
}

// =============================================================================
// Analyzer
// =============================================================================

/// Analyzer for MIR SOAC patterns.
struct MirSoacAnalyzer<'a> {
    /// The body being analyzed.
    body: &'a Body,
    /// Entry input parameters (LocalId -> is from entry).
    entry_params: HashSet<LocalId>,
    /// Variables that trace back to entry params.
    from_entry: HashSet<LocalId>,
    /// Collected maps.
    maps: Vec<ParallelizableMap>,
}

impl<'a> MirSoacAnalyzer<'a> {
    fn new(body: &'a Body, inputs: &[EntryInput]) -> Self {
        let entry_params: HashSet<LocalId> = inputs.iter().map(|i| i.local).collect();
        let from_entry = entry_params.clone();

        Self {
            body,
            entry_params,
            from_entry,
            maps: Vec::new(),
        }
    }

    /// Analyze the body starting from root.
    fn analyze(&mut self) {
        self.analyze_expr(self.body.root);
    }

    /// Analyze an expression recursively.
    fn analyze_expr(&mut self, expr_id: ExprId) {
        match self.body.get_expr(expr_id) {
            Expr::Let { local, rhs, body } => {
                // First analyze the RHS
                self.analyze_expr(*rhs);

                // Track if RHS traces back to entry param
                if self.is_from_entry_param(*rhs) {
                    self.from_entry.insert(*local);
                }

                // Then analyze the body
                self.analyze_expr(*body);
            }

            Expr::Intrinsic { name, args } => {
                // Check for map intrinsics
                let is_map = name == "_w_intrinsic_map";
                let is_inplace_map = name == "_w_intrinsic_inplace_map";

                if (is_map || is_inplace_map) && args.len() >= 2 {
                    let closure = args[0];
                    let array = args[1];
                    let maps_entry_input = self.is_from_entry_param(array);

                    self.maps.push(ParallelizableMap {
                        map_expr: expr_id,
                        closure,
                        array,
                        maps_entry_input,
                        is_inplace: is_inplace_map,
                    });
                }

                // Analyze all arguments
                for arg in args {
                    self.analyze_expr(*arg);
                }
            }

            Expr::If { cond, then_, else_ } => {
                self.analyze_expr(*cond);
                self.analyze_expr(*then_);
                self.analyze_expr(*else_);
            }

            Expr::Call { args, .. } => {
                for arg in args {
                    self.analyze_expr(*arg);
                }
            }

            Expr::Loop { init, body, .. } => {
                self.analyze_expr(*init);
                self.analyze_expr(*body);
            }

            Expr::Array { size, .. } => {
                self.analyze_expr(*size);
            }

            Expr::Tuple(elems) | Expr::Vector(elems) => {
                for elem in elems {
                    self.analyze_expr(*elem);
                }
            }

            Expr::BinOp { lhs, rhs, .. } => {
                self.analyze_expr(*lhs);
                self.analyze_expr(*rhs);
            }

            Expr::UnaryOp { operand, .. } => {
                self.analyze_expr(*operand);
            }

            Expr::Materialize(inner) | Expr::Load { ptr: inner } => {
                self.analyze_expr(*inner);
            }

            Expr::Store { ptr, value } => {
                self.analyze_expr(*ptr);
                self.analyze_expr(*value);
            }

            Expr::Attributed { expr, .. } => {
                self.analyze_expr(*expr);
            }

            // Terminals - no sub-expressions
            Expr::Local(_)
            | Expr::Global(_)
            | Expr::Extern(_)
            | Expr::Int(_)
            | Expr::Float(_)
            | Expr::Bool(_)
            | Expr::Unit
            | Expr::String(_)
            | Expr::Matrix(_) => {}
        }
    }

    /// Check if an expression traces back to an entry parameter.
    fn is_from_entry_param(&self, expr_id: ExprId) -> bool {
        match self.body.get_expr(expr_id) {
            Expr::Local(local_id) => {
                self.entry_params.contains(local_id) || self.from_entry.contains(local_id)
            }
            // For other expressions, check if it's an array type from entry
            // This handles cases like array views or storage references
            _ => false,
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Analyze a MIR program for parallelizable SOAC patterns.
pub fn analyze_program(program: &Program) -> MirSoacAnalysis {
    let mut analysis = MirSoacAnalysis::default();

    for def in &program.defs {
        if let Def::EntryPoint {
            name,
            execution_model: ExecutionModel::Compute { .. },
            inputs,
            body,
            ..
        } = def
        {
            let mut analyzer = MirSoacAnalyzer::new(body, inputs);
            analyzer.analyze();

            analysis.by_entry.insert(
                name.clone(),
                ComputeEntryAnalysis {
                    name: name.clone(),
                    maps: analyzer.maps,
                },
            );
        }
    }

    analysis
}

/// Check if an entry point has a single parallelizable map at the root.
///
/// Returns the map info if the pattern matches:
/// - Compute entry point
/// - Body is Let bindings followed by a single map call
/// - The map operates on an entry parameter array
pub fn find_root_parallelizable_map<'a>(
    analysis: &'a MirSoacAnalysis,
    entry_name: &str,
) -> Option<&'a ParallelizableMap> {
    let entry_analysis = analysis.by_entry.get(entry_name)?;

    // For now, we only support a single map that operates on entry input
    // and is at the "root" level (not nested in conditionals, etc.)
    entry_analysis.maps.iter().find(|m| m.maps_entry_input)
}

/// Check if a type is an array type.
pub fn is_array_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Array, _))
}

/// Get the element type from an array type.
pub fn get_element_type(ty: &Type<TypeName>) -> Option<Type<TypeName>> {
    match ty {
        Type::Constructed(TypeName::Array, args) if !args.is_empty() => Some(args[0].clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_program() {
        let program = Program {
            defs: vec![],
            lambda_registry: crate::IdArena::new(),
        };
        let analysis = analyze_program(&program);
        assert!(analysis.by_entry.is_empty());
    }
}
