//! SSA-level SOAC analysis for compute shader parallelization.
//!
//! This module analyzes SSA to detect parallelizable loop patterns.
//! It identifies map loops and tracks array provenance to determine
//! if they can be chunked across threads.
//!
//! The analysis looks for the standard map loop pattern:
//! ```text
//! header(acc: array, index: i32):
//!     cond = index < len
//!     br_if cond, body(), exit(acc)
//!
//! body():
//!     elem = arr[index]
//!     result = f(elem)
//!     new_arr = array_with(acc, index, result)
//!     next_i = index + 1
//!     br header(new_arr, next_i)
//!
//! exit(result: array):
//!     // final array
//! ```

use crate::ast::TypeName;
use crate::tlc::to_ssa::{EntryInput, ExecutionModel, SsaEntryPoint, SsaProgram};
use polytype::Type;
use std::collections::{HashMap, HashSet};

use super::ssa::{BlockId, FuncBody, InstKind, Terminator, ValueId};

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
        /// Parameter index in the entry point.
        param_index: usize,
        /// Set and binding for the storage buffer.
        storage_binding: (u32, u32),
    },

    /// Range/iota that can be chunked by adjusting bounds.
    Range {
        /// The ValueId producing this range.
        value: ValueId,
    },

    /// Unknown provenance - cannot parallelize.
    Unknown,
}

// =============================================================================
// Loop Detection
// =============================================================================

/// Information about a detected loop in the CFG.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// The loop header block (has back-edge target).
    pub header: BlockId,
    /// The loop body block(s).
    pub body_blocks: Vec<BlockId>,
    /// The loop exit block.
    pub exit: BlockId,
    /// The back-edge source block.
    pub latch: BlockId,
}

/// Detect loops in a function body by finding back-edges.
fn detect_loops(body: &FuncBody) -> Vec<LoopInfo> {
    let mut loops = Vec::new();
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();

    // Simple DFS to find back-edges
    fn dfs(
        body: &FuncBody,
        block: BlockId,
        visited: &mut HashSet<BlockId>,
        in_stack: &mut HashSet<BlockId>,
        back_edges: &mut Vec<(BlockId, BlockId)>, // (from, to)
    ) {
        if visited.contains(&block) {
            return;
        }
        visited.insert(block);
        in_stack.insert(block);

        let blk = &body.blocks[block.index()];
        if let Some(ref term) = blk.terminator {
            let successors = match term {
                Terminator::Branch { target, .. } => vec![*target],
                Terminator::CondBranch {
                    then_target,
                    else_target,
                    ..
                } => vec![*then_target, *else_target],
                Terminator::Return(_) | Terminator::ReturnUnit | Terminator::Unreachable => vec![],
            };

            for succ in successors {
                if in_stack.contains(&succ) {
                    // Back-edge found
                    back_edges.push((block, succ));
                } else {
                    dfs(body, succ, visited, in_stack, back_edges);
                }
            }
        }

        in_stack.remove(&block);
    }

    let mut back_edges = Vec::new();
    dfs(body, BlockId::ENTRY, &mut visited, &mut in_stack, &mut back_edges);

    // For each back-edge, construct loop info
    for (latch, header) in back_edges {
        // Find body blocks (blocks dominated by header that reach latch)
        // For now, simple heuristic: blocks between header and latch
        let mut body_blocks = Vec::new();

        // Simple approach: the block after header in a conditional branch is the body
        let header_blk = &body.blocks[header.index()];
        if let Some(Terminator::CondBranch {
            then_target,
            else_target,
            ..
        }) = &header_blk.terminator
        {
            // then_target is typically the body, else_target is the exit
            body_blocks.push(*then_target);
            loops.push(LoopInfo {
                header,
                body_blocks,
                exit: *else_target,
                latch,
            });
        }
    }

    loops
}

// =============================================================================
// Map Pattern Detection
// =============================================================================

/// Information about a detected map loop.
#[derive(Debug, Clone)]
pub struct MapLoopInfo {
    /// The loop structure.
    pub loop_info: LoopInfo,
    /// The array being iterated over.
    pub input_array: ValueId,
    /// The function being called on each element.
    pub map_function: String,
    /// The index value in the header.
    pub index_value: ValueId,
    /// The accumulator value in the header.
    pub acc_value: ValueId,
    /// The length value used in the condition.
    pub length_value: ValueId,
}

/// Check if a loop matches the map pattern and extract info.
fn analyze_map_loop(body: &FuncBody, loop_info: &LoopInfo) -> Option<MapLoopInfo> {
    let header = &body.blocks[loop_info.header.index()];

    // Header should have 2 params: accumulator and index
    if header.params.len() != 2 {
        return None;
    }

    let acc_value = header.params[0].value;
    let index_value = header.params[1].value;

    // Header should have a comparison instruction and conditional branch
    // Find the comparison: index < length
    let mut length_value = None;
    for &inst_id in &header.insts {
        let inst = &body.insts[inst_id.index()];
        if let InstKind::BinOp { op, lhs, rhs } = &inst.kind {
            if op == "<" && *lhs == index_value {
                length_value = Some(*rhs);
                break;
            }
        }
    }
    let length_value = length_value?;

    // Analyze the body block to find: arr[index], f(elem), array_with(acc, index, result)
    if loop_info.body_blocks.is_empty() {
        return None;
    }
    let body_block = &body.blocks[loop_info.body_blocks[0].index()];

    let mut input_array = None;
    let mut map_function = None;

    for &inst_id in &body_block.insts {
        let inst = &body.insts[inst_id.index()];
        match &inst.kind {
            InstKind::Index { base, index } if *index == index_value => {
                input_array = Some(*base);
            }
            InstKind::Call { func, args } => {
                // Check if this is the map function call (takes the indexed element)
                // or the array_with call
                if func == "_w_intrinsic_array_with" {
                    // This is the accumulator update, skip
                } else if args.len() == 1 {
                    // Single-argument call is likely the map function
                    map_function = Some(func.clone());
                }
            }
            _ => {}
        }
    }

    Some(MapLoopInfo {
        loop_info: loop_info.clone(),
        input_array: input_array?,
        map_function: map_function?,
        index_value,
        acc_value,
        length_value,
    })
}

// =============================================================================
// Provenance Tracking
// =============================================================================

/// Track the provenance of a value back to its source.
fn track_provenance(
    body: &FuncBody,
    value: ValueId,
    inputs: &[EntryInput],
) -> ArrayProvenance {
    // Check if it's a parameter
    for (i, (param_value, _, _)) in body.params.iter().enumerate() {
        if *param_value == value {
            // Check if this parameter corresponds to a storage buffer input
            if let Some(input) = inputs.get(i) {
                if let Some(binding) = input.storage_binding {
                    return ArrayProvenance::EntryStorage {
                        name: input.name.clone(),
                        param_index: i,
                        storage_binding: binding,
                    };
                }
            }
            return ArrayProvenance::Unknown;
        }
    }

    // Check if it's produced by an instruction
    for inst in &body.insts {
        if inst.result == Some(value) {
            match &inst.kind {
                InstKind::ArrayRange { .. } => {
                    return ArrayProvenance::Range { value };
                }
                // Could trace through other instructions here
                _ => {}
            }
        }
    }

    ArrayProvenance::Unknown
}

// =============================================================================
// Analysis Results
// =============================================================================

/// Information about a parallelizable map in a compute shader.
#[derive(Debug, Clone)]
pub struct ParallelizableMap {
    /// Provenance of the array being mapped over.
    pub source: ArrayProvenance,
    /// The function being mapped.
    pub map_function: String,
    /// The map loop information.
    pub map_loop: MapLoopInfo,
}

/// Analysis results for a compute entry point.
#[derive(Debug, Clone)]
pub struct ComputeEntryAnalysis {
    /// Name of the entry point.
    pub name: String,
    /// Local size for the compute shader.
    pub local_size: (u32, u32, u32),
    /// The first parallelizable map found (if any).
    pub parallelizable_map: Option<ParallelizableMap>,
}

/// Analysis results for an SSA program.
#[derive(Debug, Default)]
pub struct SsaSoacAnalysis {
    /// Analysis for each compute entry point.
    pub by_entry: HashMap<String, ComputeEntryAnalysis>,
}

// =============================================================================
// Public API
// =============================================================================

/// Analyze an SSA program for parallelizable SOAC patterns.
pub fn analyze_program(program: &SsaProgram) -> SsaSoacAnalysis {
    let mut analysis = SsaSoacAnalysis::default();

    for entry in &program.entry_points {
        if let ExecutionModel::Compute { local_size } = entry.execution_model {
            let entry_analysis = analyze_compute_entry(entry, local_size);
            analysis.by_entry.insert(entry.name.clone(), entry_analysis);
        }
    }

    analysis
}

/// Analyze a single compute entry point.
fn analyze_compute_entry(entry: &SsaEntryPoint, local_size: (u32, u32, u32)) -> ComputeEntryAnalysis {
    let body = &entry.body;

    // Detect loops
    let loops = detect_loops(body);

    // Find the first map loop that's parallelizable
    let mut parallelizable_map = None;

    for loop_info in &loops {
        if let Some(map_loop) = analyze_map_loop(body, loop_info) {
            // Track provenance of the input array
            let provenance = track_provenance(body, map_loop.input_array, &entry.inputs);

            // Only parallelize if we have storage buffer provenance
            if matches!(provenance, ArrayProvenance::EntryStorage { .. }) {
                parallelizable_map = Some(ParallelizableMap {
                    source: provenance,
                    map_function: map_loop.map_function.clone(),
                    map_loop,
                });
                break;
            }
        }
    }

    ComputeEntryAnalysis {
        name: entry.name.clone(),
        local_size,
        parallelizable_map,
    }
}

/// Check if a type is an array type.
fn is_array_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Array, _))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_array_type() {
        let arr_ty = Type::Constructed(TypeName::Array, vec![Type::Constructed(TypeName::Float(32), vec![])]);
        assert!(is_array_type(&arr_ty));

        let int_ty = Type::Constructed(TypeName::Int(32), vec![]);
        assert!(!is_array_type(&int_ty));
    }
}
