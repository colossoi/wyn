//! SIR Parallelization Pass
//!
//! Transforms SIR for GPU parallelization. This pass runs after fusion and
//! before flattening to MIR.
//!
//! Key responsibilities:
//! - Transform compute entry points with top-level maps on unsized arrays
//! - Add thread_id lookup and slice calculation thunk
//! - For multi-kernel operations (scan, reduce): split into multiple kernel phases
//!
//! For a top-level map on unsized storage array:
//! ```text
//! entry foo(arr: [?]T) -> [?]U = map(f, arr)
//! ```
//! Transforms to:
//! ```text
//! entry foo(arr: [?]T) -> [?]U =
//!   let tid = intrinsic("_w_global_invocation_id")
//!   let idx = tid.x
//!   let chunk = slice(arr, idx, 1)
//!   let result = map(f, chunk)  // now operates on size-1 slice
//!   in result
//! ```

use std::collections::HashMap;

use crate::ast::TypeName;
use crate::IdSource;
use polytype::Type;

use super::{
    Body, Def, EntryInput, ExecutionModel, Exp, Lambda, Map, Op, ParallelizationConfig, Pat,
    PatElem, Prim, Program, Size, SizeVar, Soac, Statement, StatementId, VarId,
};

// =============================================================================
// Execution Strategies
// =============================================================================

/// GPU execution strategy for a SOAC operation.
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Single kernel execution - one thread per element.
    /// Used for embarrassingly parallel operations like map.
    SingleKernel {
        /// Workgroup (local) size for the kernel.
        workgroup_size: (u32, u32, u32),
    },

    /// Multi-phase execution requiring multiple kernel dispatches.
    /// Used for operations that need global synchronization (scan, reduce).
    MultiKernel(Vec<KernelPhase>),

    /// Sequential fallback for small arrays or unsupported patterns.
    /// Falls back to loop-based execution.
    Sequential,
}

/// A single phase in a multi-kernel operation.
#[derive(Debug, Clone)]
pub struct KernelPhase {
    /// Name for debugging/profiling.
    pub name: String,
    /// Workgroup size for this phase.
    pub workgroup_size: (u32, u32, u32),
    /// Input buffer bindings.
    pub inputs: Vec<BufferBinding>,
    /// Output buffer bindings.
    pub outputs: Vec<BufferBinding>,
    /// The kernel operation to perform.
    pub body: KernelBody,
}

/// Buffer binding information for kernel I/O.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    /// Variable ID this buffer corresponds to.
    pub var: VarId,
    /// Descriptor set number.
    pub set: u32,
    /// Binding number within the set.
    pub binding: u32,
    /// Whether this is an intermediate buffer (needs allocation).
    pub is_intermediate: bool,
}

/// The operation a kernel phase performs.
#[derive(Debug, Clone)]
pub enum KernelBody {
    /// Simple element-wise map operation.
    Map {
        /// The lambda to apply to each element.
        lambda: Lambda,
    },

    /// Per-workgroup reduction using shared memory.
    WorkgroupReduce {
        /// Reduction operation (binary).
        op: Lambda,
        /// Neutral element for the reduction.
        neutral: VarId,
    },

    /// Block-level scan (prefix sum within each workgroup).
    BlockScan {
        /// Scan operation (binary, associative).
        op: Lambda,
        /// Neutral element.
        neutral: VarId,
        /// Whether to output block sums for next phase.
        output_block_sums: bool,
    },

    /// Add block offsets back to elements (scan phase 3).
    AddOffsets,

    /// Stream compaction for filter operations.
    Compact,

    /// Scatter with atomic operations for conflict handling.
    ScatterAtomic,

    /// Segmented reduction within segment boundaries.
    SegmentedReduce {
        /// Segment descriptor variable.
        segments: VarId,
        /// Reduction operation.
        op: Lambda,
    },
}

// =============================================================================
// Parallelization Analysis
// =============================================================================

/// Result of parallelization analysis for a SIR program.
#[derive(Debug, Clone)]
pub struct ParallelizationPlan {
    /// Execution strategy for each SOAC statement.
    pub strategies: HashMap<StatementId, ExecutionStrategy>,
    /// Intermediate buffers that need allocation.
    pub intermediate_buffers: Vec<IntermediateBuffer>,
    /// Size hints collected from entry parameters.
    pub size_hints: HashMap<SizeVar, u32>,
}

/// An intermediate buffer needed for multi-phase operations.
#[derive(Debug, Clone)]
pub struct IntermediateBuffer {
    /// Unique identifier for this buffer.
    pub id: u32,
    /// Element type (as string for now).
    pub element_type: String,
    /// Size expression.
    pub size: Size,
    /// Which kernel phases use this buffer.
    pub used_by_phases: Vec<String>,
}

impl ParallelizationPlan {
    /// Create an empty plan.
    pub fn new() -> Self {
        ParallelizationPlan {
            strategies: HashMap::new(),
            intermediate_buffers: Vec::new(),
            size_hints: HashMap::new(),
        }
    }
}

impl Default for ParallelizationPlan {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Parallelizer
// =============================================================================

/// The parallelization pass.
pub struct Parallelizer {
    /// Configuration for parallelization decisions.
    config: ParallelizationConfig,
    /// Collected size hints from entry parameters.
    size_hints: HashMap<SizeVar, u32>,
    /// Counter for intermediate buffer IDs.
    #[allow(dead_code)]
    next_buffer_id: u32,
    /// ID source for fresh variables.
    var_source: IdSource<VarId>,
    /// ID source for fresh statements.
    stm_source: IdSource<StatementId>,
}

impl Parallelizer {
    /// Create a new parallelizer with default configuration.
    pub fn new() -> Self {
        Parallelizer {
            config: ParallelizationConfig::default(),
            size_hints: HashMap::new(),
            next_buffer_id: 0,
            var_source: IdSource::new(),
            stm_source: IdSource::new(),
        }
    }

    /// Create a parallelizer with custom configuration.
    pub fn with_config(config: ParallelizationConfig) -> Self {
        Parallelizer {
            config,
            size_hints: HashMap::new(),
            next_buffer_id: 0,
            var_source: IdSource::new(),
            stm_source: IdSource::new(),
        }
    }

    /// Initialize ID sources from existing program to avoid conflicts.
    fn init_id_sources(&mut self, program: &Program) {
        let mut max_var = 0u32;
        let mut max_stm = 0u32;

        for def in &program.defs {
            self.scan_def_for_max_ids(def, &mut max_var, &mut max_stm);
        }

        self.var_source = IdSource::starting_from(max_var + 1);
        self.stm_source = IdSource::starting_from(max_stm + 1);
    }

    fn scan_def_for_max_ids(&self, def: &Def, max_var: &mut u32, max_stm: &mut u32) {
        match def {
            Def::Function { params, body, .. } => {
                for p in params {
                    *max_var = (*max_var).max(p.var.0);
                }
                self.scan_body_for_max_ids(body, max_var, max_stm);
            }
            Def::EntryPoint { inputs, body, .. } => {
                for inp in inputs {
                    *max_var = (*max_var).max(inp.var.0);
                }
                self.scan_body_for_max_ids(body, max_var, max_stm);
            }
            Def::Constant { body, .. } => {
                self.scan_body_for_max_ids(body, max_var, max_stm);
            }
            Def::Uniform { .. } | Def::Storage { .. } => {}
        }
    }

    fn scan_body_for_max_ids(&self, body: &Body, max_var: &mut u32, max_stm: &mut u32) {
        for stmt in &body.statements {
            *max_stm = (*max_stm).max(stmt.id.0);
            for bind in &stmt.pat.binds {
                *max_var = (*max_var).max(bind.var.0);
            }
        }
    }

    /// Transform a SIR program for GPU parallelization.
    ///
    /// This pass transforms compute entry points:
    /// - For top-level maps on unsized arrays: add thread_id thunk and slice calculation
    /// - For multi-kernel operations (scan, reduce): split into multiple kernel phases
    ///
    /// Returns the transformed program and parallelization plan.
    pub fn transform(&mut self, program: Program) -> (Program, ParallelizationPlan) {
        self.init_id_sources(&program);

        let mut strategies = HashMap::new();
        let mut transformed_defs = Vec::new();

        for def in program.defs {
            match def {
                Def::EntryPoint {
                    id,
                    name,
                    execution_model: ExecutionModel::Compute { local_size },
                    inputs,
                    outputs,
                    body,
                    span,
                } => {
                    // Check if this has a top-level map on unsized storage array
                    if let Some((transformed_body, strategy)) =
                        self.transform_compute_body(&body, &inputs, local_size)
                    {
                        // Record strategies for any SOACs in the transformed body
                        for stmt in &transformed_body.statements {
                            if let Exp::Op(Op::Soac(_)) = &stmt.exp {
                                strategies.insert(stmt.id, strategy.clone());
                            }
                        }

                        transformed_defs.push(Def::EntryPoint {
                            id,
                            name,
                            execution_model: ExecutionModel::Compute { local_size },
                            inputs,
                            outputs,
                            body: transformed_body,
                            span,
                        });
                    } else {
                        // No transformation needed, analyze as before
                        self.analyze_compute_body(&body, local_size, &mut strategies);
                        transformed_defs.push(Def::EntryPoint {
                            id,
                            name,
                            execution_model: ExecutionModel::Compute { local_size },
                            inputs,
                            outputs,
                            body,
                            span,
                        });
                    }
                }
                other => {
                    // Non-compute defs pass through unchanged
                    transformed_defs.push(other);
                }
            }
        }

        let plan = ParallelizationPlan {
            strategies,
            intermediate_buffers: Vec::new(),
            size_hints: self.size_hints.clone(),
        };

        (Program { defs: transformed_defs, lambdas: program.lambdas }, plan)
    }

    /// Transform a compute shader body if it has a top-level map on unsized storage array.
    /// Returns the transformed body and execution strategy, or None if no transform needed.
    fn transform_compute_body(
        &mut self,
        body: &Body,
        inputs: &[EntryInput],
        local_size: (u32, u32, u32),
    ) -> Option<(Body, ExecutionStrategy)> {
        // Look for pattern: body has statements ending with a map, and the map's input
        // is an entry parameter with unsized storage array type

        // Find the last statement that produces the result
        let result_stmt = body.statements.last()?;

        // Check if it's a map operation
        let map = match &result_stmt.exp {
            Exp::Op(Op::Soac(Soac::Map(map))) => map,
            _ => return None,
        };

        // Check if the map's input is an unsized storage array
        // (i.e., one of the entry inputs with Unsized size)
        let arr_var = map.arrs.first()?;
        let input = inputs.iter().find(|inp| inp.var == *arr_var)?;

        if !Self::is_unsized_storage_array(&input.ty) {
            return None;
        }

        // Build the transformed body with thread_id thunk
        let transformed = self.build_parallelized_map_body(result_stmt, map, *arr_var, &input.ty);

        let strategy = ExecutionStrategy::SingleKernel {
            workgroup_size: local_size,
        };

        Some((transformed, strategy))
    }

    /// Check if a type is an unsized storage array.
    fn is_unsized_storage_array(ty: &Type<TypeName>) -> bool {
        if let Type::Constructed(TypeName::Array, args) = ty {
            if args.len() == 3 {
                let is_storage = matches!(&args[1], Type::Constructed(TypeName::AddressStorage, _));
                let is_unsized = matches!(&args[2], Type::Constructed(TypeName::Unsized, _));
                return is_storage && is_unsized;
            }
        }
        false
    }

    /// Build a transformed body for a map on unsized storage array.
    /// Adds: thread_id lookup, index extraction, slice creation, then the original map on the slice.
    fn build_parallelized_map_body(
        &mut self,
        map_stmt: &Statement,
        map: &Map,
        arr_var: VarId,
        arr_ty: &Type<TypeName>,
    ) -> Body {
        let mut statements = Vec::new();
        // Use the original map statement's span for all generated statements
        let span = map_stmt.span;

        // 1. let tid = intrinsic("_w_global_invocation_id") : uvec3
        let tid_var = self.var_source.next_id();
        let tid_ty = Type::Constructed(TypeName::Vec, vec![
            Type::Constructed(TypeName::Size(3), vec![]),
            Type::Constructed(TypeName::UInt(32), vec![]),
        ]);
        statements.push(Statement {
            id: self.stm_source.next_id(),
            pat: Pat::single(tid_var, tid_ty.clone(), "tid".to_string()),
            exp: Exp::Prim(Prim::Intrinsic {
                name: "_w_global_invocation_id".to_string(),
                args: vec![],
            }),
            ty: tid_ty,
            span,
        });

        // 2. let idx = tid.x : u32 (extract x component, index 0)
        let idx_var = self.var_source.next_id();
        let idx_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        statements.push(Statement {
            id: self.stm_source.next_id(),
            pat: Pat::single(idx_var, idx_ty.clone(), "idx".to_string()),
            exp: Exp::TupleProj {
                tuple: tid_var,
                index: 0,
            },
            ty: idx_ty,
            span,
        });

        // 3. let one = 1u32
        let one_var = self.var_source.next_id();
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        statements.push(Statement {
            id: self.stm_source.next_id(),
            pat: Pat::single(one_var, u32_ty.clone(), "one".to_string()),
            exp: Exp::Prim(Prim::ConstU32(1)),
            ty: u32_ty.clone(),
            span,
        });

        // 4. let end_idx = idx + 1 : u32
        let end_idx_var = self.var_source.next_id();
        statements.push(Statement {
            id: self.stm_source.next_id(),
            pat: Pat::single(end_idx_var, u32_ty.clone(), "end_idx".to_string()),
            exp: Exp::Prim(Prim::Add(idx_var, one_var)),
            ty: u32_ty,
            span,
        });

        // 4. let chunk = arr[idx..end_idx] : [1]T (using __slice_range)
        let chunk_var = self.var_source.next_id();
        // Build [1]T type from arr_ty which is [?]T
        let chunk_ty = if let Type::Constructed(TypeName::Array, args) = arr_ty {
            Type::Constructed(TypeName::Array, vec![
                args[0].clone(),  // element type
                args[1].clone(),  // address space (Storage)
                Type::Constructed(TypeName::Size(1), vec![]),  // Size 1
            ])
        } else {
            arr_ty.clone()
        };
        statements.push(Statement {
            id: self.stm_source.next_id(),
            pat: Pat::single(chunk_var, chunk_ty.clone(), "chunk".to_string()),
            exp: Exp::Prim(Prim::Intrinsic {
                name: "__slice_range".to_string(),
                args: vec![arr_var, idx_var, end_idx_var],
            }),
            ty: chunk_ty,
            span,
        });

        // 4. let result = map(f, chunk) : [1]U
        // Clone the original map but with chunk_var as input and Size::Const(1)
        let result_var = self.var_source.next_id();
        let new_map = Map {
            w: Size::Const(1),
            f: map.f.clone(),
            arrs: vec![chunk_var],
        };
        statements.push(Statement {
            id: self.stm_source.next_id(),
            pat: Pat::single(result_var, map_stmt.ty.clone(), "result".to_string()),
            exp: Exp::Op(Op::Soac(Soac::Map(new_map))),
            ty: map_stmt.ty.clone(),
            span,
        });

        Body {
            statements,
            result: vec![result_var],
        }
    }

    /// Analyze a compute shader body and determine execution strategies for SOACs.
    fn analyze_compute_body(
        &self,
        body: &Body,
        local_size: (u32, u32, u32),
        strategies: &mut HashMap<StatementId, ExecutionStrategy>,
    ) {
        for stmt in &body.statements {
            if let Exp::Op(Op::Soac(soac)) = &stmt.exp {
                let strategy = self.determine_strategy(soac, local_size);
                strategies.insert(stmt.id, strategy);
            }
        }
    }

    /// Determine the execution strategy for a SOAC.
    fn determine_strategy(&self, soac: &Soac, local_size: (u32, u32, u32)) -> ExecutionStrategy {
        match soac {
            Soac::Map(_) => {
                // Maps execute as single kernel - one thread per element (or chunk)
                // Lowering handles thread_id → slice bounds calculation
                ExecutionStrategy::SingleKernel {
                    workgroup_size: local_size,
                }
            }
            Soac::Reduce(_) | Soac::Scan(_) => {
                // Reductions and scans need multi-phase execution
                // TODO: Implement proper multi-kernel splitting
                ExecutionStrategy::MultiKernel(vec![])
            }
            Soac::SegMap(_) | Soac::SegReduce(_) | Soac::SegScan(_) => {
                // Segmented operations - complex scheduling
                ExecutionStrategy::Sequential
            }
            Soac::Iota { .. } | Soac::Replicate { .. } | Soac::Reshape { .. } => {
                // These are typically fused away or trivial
                ExecutionStrategy::SingleKernel {
                    workgroup_size: local_size,
                }
            }
        }
    }

    #[allow(dead_code)]
    fn substitute_statement(&mut self, stmt: &Statement, subst: &mut HashMap<VarId, VarId>) -> Statement {
        // Create fresh variables for pattern bindings
        let new_binds: Vec<PatElem> = stmt
            .pat
            .binds
            .iter()
            .map(|bind| {
                let fresh_var = self.var_source.next_id();
                subst.insert(bind.var, fresh_var);
                PatElem {
                    var: fresh_var,
                    ty: bind.ty.clone(),
                    name_hint: bind.name_hint.clone(),
                }
            })
            .collect();

        let new_exp = self.substitute_exp(&stmt.exp, subst);

        Statement {
            id: self.stm_source.next_id(),
            pat: Pat { binds: new_binds },
            exp: new_exp,
            ty: stmt.ty.clone(),
            span: stmt.span,
        }
    }

    #[allow(dead_code)]
    fn substitute_exp(&self, exp: &Exp, subst: &HashMap<VarId, VarId>) -> Exp {
        let get = |v: &VarId| *subst.get(v).unwrap_or(v);

        match exp {
            Exp::Var(v) => Exp::Var(get(v)),
            Exp::Prim(prim) => Exp::Prim(self.substitute_prim(prim, subst)),
            Exp::Tuple(vars) => Exp::Tuple(vars.iter().map(get).collect()),
            Exp::TupleProj { tuple, index } => Exp::TupleProj {
                tuple: get(tuple),
                index: *index,
            },
            Exp::Apply { func, args } => Exp::Apply {
                func: func.clone(),
                args: args.iter().map(get).collect(),
            },
            Exp::If {
                cond,
                then_body,
                else_body,
            } => Exp::If {
                cond: get(cond),
                then_body: then_body.clone(),
                else_body: else_body.clone(),
            },
            Exp::Loop { params, init, body } => Exp::Loop {
                params: params.clone(),
                init: init.iter().map(get).collect(),
                body: body.clone(), // TODO: substitute in nested bodies
            },
            Exp::Op(op) => Exp::Op(op.clone()), // TODO: substitute in SOACs if needed
        }
    }

    /// Substitute variables in a primitive expression.
    fn substitute_prim(&self, prim: &Prim, subst: &HashMap<VarId, VarId>) -> Prim {
        let get = |v: &VarId| *subst.get(v).unwrap_or(v);

        match prim {
            Prim::ConstBool(b) => Prim::ConstBool(*b),
            Prim::ConstI32(n) => Prim::ConstI32(*n),
            Prim::ConstI64(n) => Prim::ConstI64(*n),
            Prim::ConstU32(n) => Prim::ConstU32(*n),
            Prim::ConstU64(n) => Prim::ConstU64(*n),
            Prim::ConstF32(n) => Prim::ConstF32(*n),
            Prim::ConstF64(n) => Prim::ConstF64(*n),
            Prim::Add(a, b) => Prim::Add(get(a), get(b)),
            Prim::Sub(a, b) => Prim::Sub(get(a), get(b)),
            Prim::Mul(a, b) => Prim::Mul(get(a), get(b)),
            Prim::Div(a, b) => Prim::Div(get(a), get(b)),
            Prim::Mod(a, b) => Prim::Mod(get(a), get(b)),
            Prim::Eq(a, b) => Prim::Eq(get(a), get(b)),
            Prim::Ne(a, b) => Prim::Ne(get(a), get(b)),
            Prim::Lt(a, b) => Prim::Lt(get(a), get(b)),
            Prim::Le(a, b) => Prim::Le(get(a), get(b)),
            Prim::Gt(a, b) => Prim::Gt(get(a), get(b)),
            Prim::Ge(a, b) => Prim::Ge(get(a), get(b)),
            Prim::And(a, b) => Prim::And(get(a), get(b)),
            Prim::Or(a, b) => Prim::Or(get(a), get(b)),
            Prim::Neg(v) => Prim::Neg(get(v)),
            Prim::Not(v) => Prim::Not(get(v)),
            Prim::Index { arr, idx } => Prim::Index {
                arr: get(arr),
                idx: get(idx),
            },
            Prim::Intrinsic { name, args } => Prim::Intrinsic {
                name: name.clone(),
                args: args.iter().map(get).collect(),
            },
        }
    }

    /// Get the size hint for a Size expression if known.
    pub fn get_size_hint(&self, size: &Size) -> Option<u32> {
        match size {
            Size::Const(n) => Some(*n as u32),
            Size::Sym(var) => self.size_hints.get(var).copied(),
            _ => None,
        }
    }

    /// Determine execution strategy for a map operation.
    #[allow(dead_code)]
    fn parallelize_map(&self, width: &Size, _lambda: Lambda) -> ExecutionStrategy {
        let size_hint = self.get_size_hint(width);
        let workgroup_size = self.config.derive_workgroup_size(size_hint);

        ExecutionStrategy::SingleKernel { workgroup_size }
    }

    /// Determine execution strategy for a reduce operation.
    #[allow(dead_code)]
    fn parallelize_reduce(&mut self, width: &Size, op: Lambda, neutral: VarId) -> ExecutionStrategy {
        let size_hint = self.get_size_hint(width);

        if self.config.fits_single_workgroup(size_hint) {
            // Small enough for single workgroup reduction
            let workgroup_size = self.config.derive_workgroup_size(size_hint);
            ExecutionStrategy::SingleKernel { workgroup_size }
        } else {
            // Two-phase reduction for large arrays
            let workgroup_size = (256, 1, 1);

            ExecutionStrategy::MultiKernel(vec![
                KernelPhase {
                    name: "reduce_per_workgroup".into(),
                    workgroup_size,
                    inputs: vec![], // Will be filled during lowering
                    outputs: vec![],
                    body: KernelBody::WorkgroupReduce {
                        op: op.clone(),
                        neutral,
                    },
                },
                KernelPhase {
                    name: "reduce_final".into(),
                    workgroup_size: (256, 1, 1), // Reduce partial results
                    inputs: vec![],
                    outputs: vec![],
                    body: KernelBody::WorkgroupReduce { op, neutral },
                },
            ])
        }
    }

    /// Determine execution strategy for a scan operation.
    #[allow(dead_code)]
    fn parallelize_scan(&mut self, width: &Size, op: Lambda, neutral: VarId) -> ExecutionStrategy {
        let size_hint = self.get_size_hint(width);

        if self.config.fits_single_workgroup(size_hint) {
            // Small enough for single workgroup scan
            let workgroup_size = self.config.derive_workgroup_size(size_hint);
            ExecutionStrategy::SingleKernel { workgroup_size }
        } else {
            // Three-phase scan for large arrays
            let workgroup_size = (256, 1, 1);

            ExecutionStrategy::MultiKernel(vec![
                // Phase 1: Per-block scan, extract block sums
                KernelPhase {
                    name: "scan_blocks".into(),
                    workgroup_size,
                    inputs: vec![],
                    outputs: vec![],
                    body: KernelBody::BlockScan {
                        op: op.clone(),
                        neutral,
                        output_block_sums: true,
                    },
                },
                // Phase 2: Scan block sums
                KernelPhase {
                    name: "scan_block_sums".into(),
                    workgroup_size: (256, 1, 1),
                    inputs: vec![],
                    outputs: vec![],
                    body: KernelBody::BlockScan {
                        op: op.clone(),
                        neutral,
                        output_block_sums: false,
                    },
                },
                // Phase 3: Add offsets back
                KernelPhase {
                    name: "add_block_offsets".into(),
                    workgroup_size,
                    inputs: vec![],
                    outputs: vec![],
                    body: KernelBody::AddOffsets,
                },
            ])
        }
    }
}

impl Default for Parallelizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelizer_creation() {
        let p = Parallelizer::new();
        assert_eq!(p.config.default_workgroup_size, (64, 1, 1));
    }

    #[test]
    fn test_size_hint_lookup_const() {
        let p = Parallelizer::new();
        let size = Size::Const(1024);
        assert_eq!(p.get_size_hint(&size), Some(1024));
    }

    #[test]
    fn test_empty_parallelization_plan() {
        let plan = ParallelizationPlan::new();
        assert!(plan.strategies.is_empty());
        assert!(plan.intermediate_buffers.is_empty());
    }
}
