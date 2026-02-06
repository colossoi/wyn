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
use crate::tlc::to_ssa::{EntryInput, ExecutionModel, SsaEntryPoint, SsaFunction, SsaProgram};
use crate::types::is_virtual_array;
use polytype::Type;
use std::collections::HashMap;

use super::ssa::{BlockId, ControlHeader, FuncBody, InstKind, Terminator, ValueId};

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
///
/// Derived from `ControlHeader::Loop` metadata set by the SSA builder,
/// not from CFG heuristics. This means we only detect loops that were
/// explicitly created via `create_for_range_loop` / `create_while_loop`.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// The loop header block (tagged with `ControlHeader::Loop`).
    pub header: BlockId,
    /// The continue block (branches back to header).
    pub continue_block: BlockId,
    /// The merge block (loop exit).
    pub exit: BlockId,
}

/// Detect loops by reading `ControlHeader::Loop` metadata on blocks.
///
/// This relies on the SSA builder tagging loop headers during construction
/// rather than attempting CFG analysis, so it works regardless of loop shape
/// (rotated loops, multi-block bodies, if-structured loops, etc.).
fn detect_loops(body: &FuncBody) -> Vec<LoopInfo> {
    body.blocks
        .iter()
        .enumerate()
        .filter_map(|(i, block)| {
            if let Some(ControlHeader::Loop {
                merge,
                continue_block,
            }) = &block.control
            {
                Some(LoopInfo {
                    header: BlockId(i as u32),
                    continue_block: *continue_block,
                    exit: *merge,
                })
            } else {
                None
            }
        })
        .collect()
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
    /// All arguments to the map function call (including captured variables).
    pub map_call_args: Vec<ValueId>,
    /// Which argument index in `map_call_args` is the loop element.
    pub element_arg_index: usize,
    /// The return type of the map function call.
    pub map_result_ty: Type<TypeName>,
    /// The index value in the header.
    pub index_value: ValueId,
    /// The accumulator value in the header.
    pub acc_value: ValueId,
    /// The length value used in the condition.
    pub length_value: ValueId,
}

/// Check if a loop matches the canonical map pattern from `create_for_range_loop`.
///
/// Expects the canonical form emitted by `to_ssa`:
/// - Header has exactly 2 params: `(acc, index)`
/// - Header contains a single `index < length` comparison
/// - Continue block contains: `arr[index]`, `f(elem, ...)`, `array_with(...)`
///
/// Returns `None` for loops that don't match this shape.
fn analyze_map_loop(body: &FuncBody, loop_info: &LoopInfo) -> Option<MapLoopInfo> {
    let header = &body.blocks[loop_info.header.index()];

    if header.params.len() != 2 {
        return None;
    }

    let acc_value = header.params[0].value;
    let index_value = header.params[1].value;

    // Find the canonical `index < length` comparison in the header.
    // to_ssa always emits this as: push_binop("<", index, len, bool_ty)
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

    // Analyze the continue block end-to-end:
    //   1. Find _w_intrinsic_array_with(acc, index, result) — the accumulator update
    //   2. Trace `result` back to a function call that consumes an indexed element
    //   3. Verify the updated array is loop-carried back to the header
    let body_block = &body.blocks[loop_info.continue_block.index()];

    // Step 1: Find the array_with call and extract its arguments.
    let mut array_with_acc = None;
    let mut array_with_index = None;
    let mut array_with_result_val = None;
    let mut array_with_output = None;
    for &inst_id in &body_block.insts {
        let inst = &body.insts[inst_id.index()];
        if let InstKind::Call { func, args } = &inst.kind {
            if func == "_w_intrinsic_array_with" && args.len() == 3 {
                array_with_acc = Some(args[0]);
                array_with_index = Some(args[1]);
                array_with_result_val = Some(args[2]);
                array_with_output = inst.result;
                break;
            }
        }
    }
    let aw_acc = array_with_acc?;
    let aw_index = array_with_index?;
    let aw_result = array_with_result_val?;
    let aw_output = array_with_output?;

    // The array_with must update the accumulator at the loop index.
    if aw_acc != acc_value || aw_index != index_value {
        return None;
    }

    // Step 2: Verify the branch carries the updated array back as the accumulator.
    match &body_block.terminator {
        Some(Terminator::Branch { target, args }) if *target == loop_info.header => {
            // First branch arg is the new accumulator — must be the array_with result.
            if args.first() != Some(&aw_output) {
                return None;
            }
        }
        _ => return None,
    }

    // Step 3: Find the map function call whose result feeds into array_with.
    let mut map_function = None;
    let mut map_call_args = Vec::new();
    let mut element_arg_index = 0;
    let mut map_result_ty: Option<Type<TypeName>> = None;
    let mut indexed_elem = None;

    // First, find arr[index] to know the indexed element.
    let mut input_array = None;
    for &inst_id in &body_block.insts {
        let inst = &body.insts[inst_id.index()];
        if let InstKind::Index { base, index } = &inst.kind {
            if *index == index_value {
                input_array = Some(*base);
                indexed_elem = inst.result;
                break;
            }
        }
    }

    // Then find the call whose result is aw_result and that uses the indexed element.
    for &inst_id in &body_block.insts {
        let inst = &body.insts[inst_id.index()];
        if inst.result != Some(aw_result) {
            continue;
        }
        if let InstKind::Call { func, args } = &inst.kind {
            if let Some(elem) = indexed_elem {
                if let Some(pos) = args.iter().position(|a| *a == elem) {
                    map_function = Some(func.clone());
                    map_call_args = args.clone();
                    element_arg_index = pos;
                    map_result_ty = Some(inst.result_ty.clone());
                    break;
                }
            }
        }
    }

    Some(MapLoopInfo {
        loop_info: loop_info.clone(),
        input_array: input_array?,
        map_function: map_function?,
        map_call_args,
        element_arg_index,
        map_result_ty: map_result_ty?,
        index_value,
        acc_value,
        length_value,
    })
}

// =============================================================================
// Provenance Tracking
// =============================================================================

/// Track the provenance of a value back to its source.
fn track_provenance(body: &FuncBody, value: ValueId, inputs: &[EntryInput]) -> ArrayProvenance {
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
            // Check if result type is a virtual array (range)
            if is_virtual_array(&inst.result_ty) {
                return ArrayProvenance::Range { value };
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
            let entry_analysis = analyze_compute_entry(entry, local_size, program);
            analysis.by_entry.insert(entry.name.clone(), entry_analysis);
        }
    }

    analysis
}

/// Find a function by name in the program.
fn find_function<'a>(program: &'a SsaProgram, name: &str) -> Option<&'a SsaFunction> {
    program.functions.iter().find(|f| f.name == name)
}

/// Maximum call depth for traversing through function calls.
const MAX_CALL_DEPTH: usize = 10;

/// Analyze a single compute entry point.
fn analyze_compute_entry(
    entry: &SsaEntryPoint,
    local_size: (u32, u32, u32),
    program: &SsaProgram,
) -> ComputeEntryAnalysis {
    let par_map = find_highest_map(entry, program);
    ComputeEntryAnalysis {
        name: entry.name.clone(),
        local_size,
        parallelizable_map: par_map,
    }
}

/// Find the highest (closest to entry point) parallelizable map in the call tree.
/// Uses breadth-first search: checks each level before recursing into called functions.
fn find_highest_map(entry: &SsaEntryPoint, program: &SsaProgram) -> Option<ParallelizableMap> {
    // Entry params map to themselves for provenance tracking
    let initial_mapping: HashMap<usize, ValueId> =
        entry.body.params.iter().enumerate().map(|(i, (v, _, _))| (i, *v)).collect();

    find_map_in_body(entry, &entry.body, &initial_mapping, program, 0)
}

/// Search for a parallelizable map in a function body, recursing into called functions.
fn find_map_in_body(
    entry: &SsaEntryPoint,
    body: &FuncBody,
    param_to_entry_arg: &HashMap<usize, ValueId>,
    program: &SsaProgram,
    depth: usize,
) -> Option<ParallelizableMap> {
    if depth > MAX_CALL_DEPTH {
        return None;
    }

    // Breadth-first: check for map loops at THIS level first
    let loops = detect_loops(body);

    for loop_info in &loops {
        if let Some(map_loop) = analyze_map_loop(body, loop_info) {
            let provenance =
                track_provenance_unified(entry, body, map_loop.input_array, param_to_entry_arg, depth == 0);
            if matches!(
                provenance,
                ArrayProvenance::EntryStorage { .. } | ArrayProvenance::Range { .. }
            ) {
                return Some(ParallelizableMap {
                    source: provenance,
                    map_function: map_loop.map_function.clone(),
                    map_loop,
                });
            }
        }
    }

    // No map at this level - recurse into called functions
    for inst in &body.insts {
        if let InstKind::Call { func, args } = &inst.kind {
            // Skip intrinsic calls
            if func.starts_with("_w_") {
                continue;
            }
            if let Some(called_func) = find_function(program, func) {
                let new_mapping = build_param_mapping(body, param_to_entry_arg, args, entry);

                if let Some(par_map) =
                    find_map_in_body(entry, &called_func.body, &new_mapping, program, depth + 1)
                {
                    return Some(par_map);
                }
            }
        }
    }

    None
}

/// Build a parameter mapping for a called function.
/// Maps the called function's parameter indices to entry-level ValueIds.
fn build_param_mapping(
    caller_body: &FuncBody,
    caller_param_to_entry: &HashMap<usize, ValueId>,
    call_args: &[ValueId],
    entry: &SsaEntryPoint,
) -> HashMap<usize, ValueId> {
    call_args
        .iter()
        .enumerate()
        .filter_map(|(i, &arg)| {
            // If arg is a parameter of the caller, map through to entry
            for (caller_param_idx, (param_val, _, _)) in caller_body.params.iter().enumerate() {
                if *param_val == arg {
                    if let Some(&entry_val) = caller_param_to_entry.get(&caller_param_idx) {
                        return Some((i, entry_val));
                    }
                }
            }
            // If arg is a StorageView, match by (set, binding) to an entry input.
            // to_ssa wraps entry inputs in StorageView instructions, so the call arg
            // may be a view derived from a param rather than the param itself.
            for inst in &caller_body.insts {
                if inst.result == Some(arg) {
                    if let InstKind::StorageView { set, binding, .. } = &inst.kind {
                        for (input_idx, input) in entry.inputs.iter().enumerate() {
                            if input.storage_binding == Some((*set, *binding)) {
                                let entry_val = entry.body.params[input_idx].0;
                                return Some((i, entry_val));
                            }
                        }
                    }
                }
            }
            None
        })
        .collect()
}

/// Track provenance of a value, handling values at any call depth.
/// Uses the param_to_entry_arg mapping to trace back to entry-level values.
///
/// `at_entry` must be true only when `body` is the entry body itself.
/// Ranges constructed in called functions have bounds that aren't accessible
/// at entry level, so we only report `Range` provenance at entry depth.
fn track_provenance_unified(
    entry: &SsaEntryPoint,
    body: &FuncBody,
    value: ValueId,
    param_to_entry_arg: &HashMap<usize, ValueId>,
    at_entry: bool,
) -> ArrayProvenance {
    // Check if value is a function parameter
    for (i, (param_value, _, _)) in body.params.iter().enumerate() {
        if *param_value == value {
            if let Some(&entry_arg) = param_to_entry_arg.get(&i) {
                // Map back to entry level and use existing track_provenance
                return track_provenance(&entry.body, entry_arg, &entry.inputs);
            }
            return ArrayProvenance::Unknown;
        }
    }

    // Only allow local range provenance at entry level — ranges constructed
    // in called functions have bounds that aren't accessible at entry level.
    if at_entry {
        for inst in &body.insts {
            if inst.result == Some(value) && is_virtual_array(&inst.result_ty) {
                return ArrayProvenance::Range { value };
            }
        }
    }

    ArrayProvenance::Unknown
}
