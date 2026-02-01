//! MIR-level SOAC analysis for compute shader parallelization.
//!
//! This module analyzes MIR to detect parallelizable patterns in compute shaders.
//! It performs provenance tracking to trace array arguments back to their origins,
//! enabling the parallelization pass to correctly chunk different array sources.
//!
//! The analysis walks the call graph, tracking array provenance through:
//! - Entry input parameters (storage buffers)
//! - Range/iota expressions
//! - Function calls and parameter passing

use crate::ast::TypeName;
use crate::mir::{
    ArrayBacking, Body, Def, EntryInput, ExecutionModel, Expr, ExprId, LocalId, Program, RangeKind,
};
use polytype::Type;
use std::collections::HashMap;

// =============================================================================
// Array Provenance
// =============================================================================

/// Tracks where an array value ultimately comes from.
/// Used to determine how to chunk/slice arrays for parallelization.
#[derive(Debug, Clone)]
pub enum ArrayProvenance {
    /// Entry input storage buffer.
    EntryStorage {
        /// Name of the storage buffer parameter.
        name: String,
        /// LocalId in the entry body.
        local: LocalId,
    },

    /// Range/iota expression that can be chunked by adjusting bounds.
    Range {
        /// Start expression (in the body where the range is defined).
        start: ExprId,
        /// End/size expression.
        end: ExprId,
        /// Step expression (None means 1).
        step: Option<ExprId>,
        /// Range kind (inclusive/exclusive).
        kind: RangeKind,
        /// Call path to inline to expose this range.
        /// Empty if the range is directly in the entry body.
        inline_path: Vec<String>,
    },

    /// Unknown provenance - cannot parallelize.
    Unknown,
}

// =============================================================================
// Analysis Results
// =============================================================================

/// Information about a parallelizable map in a compute shader.
#[derive(Debug, Clone)]
pub struct ParallelizableMap {
    /// Provenance of the array being mapped over.
    pub source: ArrayProvenance,

    /// The closure/function being mapped.
    /// This is always relative to where the map is found.
    pub closure_name: String,

    /// For interprocedural maps: the call expression in the entry body
    /// that leads to this map. The parallelization pass should transform
    /// the argument to this call.
    pub entry_call: Option<CallInfo>,

    /// Whether this is an inplace map.
    pub is_inplace: bool,
}

/// Information about a call in the entry body that leads to a map.
#[derive(Debug, Clone)]
pub struct CallInfo {
    /// The ExprId of the Call expression in the entry body.
    pub call_expr: ExprId,
    /// The function being called.
    pub func_name: String,
    /// Which argument index receives the array that flows to the map.
    pub array_arg_index: usize,
}

/// Analysis results for a compute entry point.
#[derive(Debug, Clone)]
pub struct ComputeEntryAnalysis {
    /// Name of the entry point.
    pub name: String,
    /// The first parallelizable map found (if any).
    pub root_map: Option<ParallelizableMap>,
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

/// Analyzer that tracks array provenance through the call graph.
struct ProvenanceAnalyzer<'a> {
    /// The program being analyzed.
    program: &'a Program,
    /// Current body being analyzed.
    body: &'a Body,
    /// Provenance of each local variable in scope.
    provenance: HashMap<LocalId, ArrayProvenance>,
    /// Current call path (for tracking inline_path in Range provenance).
    call_path: Vec<String>,
    /// The call in the entry body that we're currently inside (if any).
    entry_call: Option<CallInfo>,
}

impl<'a> ProvenanceAnalyzer<'a> {
    fn new(program: &'a Program, body: &'a Body) -> Self {
        Self {
            program,
            body,
            provenance: HashMap::new(),
            call_path: Vec::new(),
            entry_call: None,
        }
    }

    /// Initialize provenance for entry inputs.
    fn init_entry_inputs(&mut self, inputs: &[EntryInput]) {
        for input in inputs {
            if is_array_type(&input.ty) {
                self.provenance.insert(
                    input.local,
                    ArrayProvenance::EntryStorage {
                        name: input.name.clone(),
                        local: input.local,
                    },
                );
            }
        }
    }

    /// Find the first parallelizable map in the body.
    fn find_first_map(&mut self) -> Option<ParallelizableMap> {
        self.find_map_in_expr(self.body.root)
    }

    /// Recursively search for a map, tracking provenance along the way.
    fn find_map_in_expr(&mut self, expr_id: ExprId) -> Option<ParallelizableMap> {
        match self.body.get_expr(expr_id).clone() {
            Expr::Let { local, rhs, body } => {
                // First check if RHS contains a map
                if let Some(map) = self.find_map_in_expr(rhs) {
                    return Some(map);
                }

                // Track provenance for this binding
                let prov = self.compute_provenance(rhs);
                self.provenance.insert(local, prov);

                // Continue in the body
                self.find_map_in_expr(body)
            }

            Expr::Intrinsic { name, args } => {
                let is_map = name == "_w_intrinsic_map";
                let is_inplace_map = name == "_w_intrinsic_inplace_map";

                if (is_map || is_inplace_map) && args.len() >= 2 {
                    let closure_expr = args[0];
                    let array_expr = args[1];

                    // Get the closure name
                    let closure_name = self.extract_closure_name(closure_expr);

                    // Get provenance of the array argument
                    let source = self.compute_provenance(array_expr);

                    return Some(ParallelizableMap {
                        source,
                        closure_name,
                        entry_call: self.entry_call.clone(),
                        is_inplace: is_inplace_map,
                    });
                }

                // Check arguments for nested maps
                for arg in args {
                    if let Some(map) = self.find_map_in_expr(arg) {
                        return Some(map);
                    }
                }
                None
            }

            Expr::Call { func, args } => {
                // Check if any argument contains a map
                for arg in &args {
                    if let Some(map) = self.find_map_in_expr(*arg) {
                        return Some(map);
                    }
                }

                // Follow into the called function
                if let Some(callee) = self.find_function(&func) {
                    return self.analyze_call(expr_id, &func, &args, callee);
                }
                None
            }

            Expr::If { cond, then_, else_ } => {
                if let Some(map) = self.find_map_in_expr(cond) {
                    return Some(map);
                }
                if let Some(map) = self.find_map_in_expr(then_) {
                    return Some(map);
                }
                self.find_map_in_expr(else_)
            }

            Expr::Array { backing, size } => {
                // Check size expression
                if let Some(map) = self.find_map_in_expr(size) {
                    return Some(map);
                }
                // Check backing expressions
                match backing {
                    ArrayBacking::View { base, offset } => {
                        if let Some(map) = self.find_map_in_expr(base) {
                            return Some(map);
                        }
                        self.find_map_in_expr(offset)
                    }
                    ArrayBacking::Range { start, step, .. } => {
                        if let Some(map) = self.find_map_in_expr(start) {
                            return Some(map);
                        }
                        if let Some(s) = step { self.find_map_in_expr(s) } else { None }
                    }
                    ArrayBacking::IndexFn { index_fn } => self.find_map_in_expr(index_fn),
                    ArrayBacking::Literal(elems) => {
                        for e in elems {
                            if let Some(map) = self.find_map_in_expr(e) {
                                return Some(map);
                            }
                        }
                        None
                    }
                    ArrayBacking::Owned { data } => self.find_map_in_expr(data),
                }
            }

            Expr::BinOp { lhs, rhs, .. } => {
                if let Some(map) = self.find_map_in_expr(lhs) {
                    return Some(map);
                }
                self.find_map_in_expr(rhs)
            }

            Expr::UnaryOp { operand, .. } => self.find_map_in_expr(operand),

            Expr::Tuple(elems) | Expr::Vector(elems) => {
                for e in elems {
                    if let Some(map) = self.find_map_in_expr(e) {
                        return Some(map);
                    }
                }
                None
            }

            Expr::Loop { init, body, .. } => {
                if let Some(map) = self.find_map_in_expr(init) {
                    return Some(map);
                }
                self.find_map_in_expr(body)
            }

            Expr::Materialize(inner) | Expr::Load { ptr: inner } => self.find_map_in_expr(inner),

            Expr::Store { ptr, value } => {
                if let Some(map) = self.find_map_in_expr(ptr) {
                    return Some(map);
                }
                self.find_map_in_expr(value)
            }

            Expr::Attributed { expr, .. } => self.find_map_in_expr(expr),

            // Terminals
            Expr::Local(_)
            | Expr::Global(_)
            | Expr::Extern(_)
            | Expr::Int(_)
            | Expr::Float(_)
            | Expr::Bool(_)
            | Expr::Unit
            | Expr::String(_)
            | Expr::Matrix(_) => None,
        }
    }

    /// Analyze a function call, propagating provenance to parameters.
    fn analyze_call(
        &mut self,
        call_expr: ExprId,
        func_name: &str,
        args: &[ExprId],
        callee: &'a Def,
    ) -> Option<ParallelizableMap> {
        let (params, callee_body) = match callee {
            Def::Function { params, body, .. } => (params, body),
            _ => return None,
        };

        // Build provenance map for callee's parameters
        let mut callee_provenance = HashMap::new();
        let mut array_arg_index = None;

        for (i, (&param_local, &arg_expr)) in params.iter().zip(args.iter()).enumerate() {
            let arg_prov = self.compute_provenance(arg_expr);

            // Track which argument has array provenance (for CallInfo)
            if !matches!(arg_prov, ArrayProvenance::Unknown) {
                array_arg_index = Some(i);
            }

            callee_provenance.insert(param_local, arg_prov);
        }

        // Save state and switch to callee
        let saved_body = self.body;
        let saved_provenance = std::mem::replace(&mut self.provenance, callee_provenance);
        let saved_entry_call = self.entry_call.clone();

        self.body = callee_body;
        self.call_path.push(func_name.to_string());

        // Set entry_call if we're at the top level (call_path was empty)
        if self.call_path.len() == 1 {
            self.entry_call = Some(CallInfo {
                call_expr,
                func_name: func_name.to_string(),
                array_arg_index: array_arg_index.unwrap_or(0),
            });
        }

        // Analyze callee
        let result = self.find_map_in_expr(callee_body.root);

        // Restore state
        self.call_path.pop();
        self.body = saved_body;
        self.provenance = saved_provenance;
        self.entry_call = saved_entry_call;

        result
    }

    /// Compute the provenance of an expression.
    fn compute_provenance(&self, expr_id: ExprId) -> ArrayProvenance {
        match self.body.get_expr(expr_id) {
            Expr::Local(local_id) => {
                self.provenance.get(local_id).cloned().unwrap_or(ArrayProvenance::Unknown)
            }

            Expr::Array { backing, size } => match backing {
                ArrayBacking::Range { start, step, kind } => ArrayProvenance::Range {
                    start: *start,
                    end: *size,
                    step: *step,
                    kind: *kind,
                    inline_path: self.call_path.clone(),
                },

                ArrayBacking::View { base, .. } => {
                    // Provenance flows through views
                    self.compute_provenance(*base)
                }

                _ => ArrayProvenance::Unknown,
            },

            // For other expressions, provenance is unknown
            _ => ArrayProvenance::Unknown,
        }
    }

    /// Extract the closure/function name from a closure expression.
    fn extract_closure_name(&self, expr_id: ExprId) -> String {
        match self.body.get_expr(expr_id) {
            Expr::Global(name) => name.clone(),
            Expr::Local(local_id) => self.body.get_local(*local_id).name.clone(),
            _ => "<unknown>".to_string(),
        }
    }

    /// Find a function definition by name.
    fn find_function(&self, name: &str) -> Option<&'a Def> {
        self.program.defs.iter().find(|def| matches!(def, Def::Function { name: n, .. } if n == name))
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
            let mut analyzer = ProvenanceAnalyzer::new(program, body);
            analyzer.init_entry_inputs(inputs);
            let root_map = analyzer.find_first_map();

            analysis.by_entry.insert(
                name.clone(),
                ComputeEntryAnalysis {
                    name: name.clone(),
                    root_map,
                },
            );
        }
    }

    analysis
}

/// Get the root parallelizable map for an entry point (if any).
pub fn find_root_parallelizable_map<'a>(
    analysis: &'a MirSoacAnalysis,
    entry_name: &str,
) -> Option<&'a ParallelizableMap> {
    analysis.by_entry.get(entry_name)?.root_map.as_ref()
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
