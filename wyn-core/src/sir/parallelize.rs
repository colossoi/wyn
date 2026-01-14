//! SIR Parallelization Pass
//!
//! Analyzes SOACs in SIR and determines execution strategies for GPU parallelization.
//! This pass runs after SIR transformation and before flattening to MIR.
//!
//! Key responsibilities:
//! - Classify SOACs by parallelism potential (map, reduce, scan, etc.)
//! - Determine execution strategy based on size hints and operation type
//! - Generate multi-kernel pipelines for complex operations (scan, reduce)
//! - Track intermediate buffers needed for multi-phase algorithms

use std::collections::HashMap;

use super::{Lambda, ParallelizationConfig, Program, Size, SizeVar, StatementId, VarId};

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
}

impl Parallelizer {
    /// Create a new parallelizer with default configuration.
    pub fn new() -> Self {
        Parallelizer {
            config: ParallelizationConfig::default(),
            size_hints: HashMap::new(),
            next_buffer_id: 0,
        }
    }

    /// Create a parallelizer with custom configuration.
    pub fn with_config(config: ParallelizationConfig) -> Self {
        Parallelizer {
            config,
            size_hints: HashMap::new(),
            next_buffer_id: 0,
        }
    }

    /// Analyze a SIR program and produce a parallelization plan.
    ///
    /// This is currently a stub that returns all Sequential strategies.
    /// TODO: Implement actual SOAC analysis and strategy selection.
    pub fn analyze(&mut self, _program: &Program) -> ParallelizationPlan {
        // For now, return an empty plan (all operations will be sequential by default)
        // This will be filled in as we implement each SOAC parallelization strategy
        ParallelizationPlan {
            strategies: HashMap::new(),
            intermediate_buffers: Vec::new(),
            size_hints: self.size_hints.clone(),
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
