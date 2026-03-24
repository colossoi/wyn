//! SSA-level SOAC analysis for compute shader parallelization.
//!
//! This module analyzes SSA to detect parallelizable SOAC patterns.
//! It recognizes first-class `InstKind::Soac(SsaSoac::Map { .. })` instructions
//! and tracks array provenance to determine if they can be chunked across threads.

use crate::ast::TypeName;
use crate::ssa::types::{EntryInput, EntryPoint, ExecutionModel, Function, Program};
use crate::types::is_virtual_array;
use polytype::Type;
use std::collections::HashMap;

use super::types::{FuncBody, InstKind, Soac, ValueId, ViewSource};

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
// Provenance Tracking
// =============================================================================

/// If `value` is produced by a `StorageView` instruction in `body`,
/// return the matching entry input's (param_index, storage_binding).
fn resolve_storage_view(
    body: &FuncBody,
    value: ValueId,
    entry: &EntryPoint,
) -> Option<(usize, (u32, u32))> {
    for inst in &body.insts {
        if inst.result == Some(value) {
            if let InstKind::StorageView {
                source: ViewSource::Storage { set, binding },
                ..
            } = &inst.kind
            {
                for (i, input) in entry.inputs.iter().enumerate() {
                    if input.storage_binding == Some((*set, *binding)) {
                        return Some((i, (*set, *binding)));
                    }
                }
            }
        }
    }
    None
}

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

/// Track provenance of a value, handling values at any call depth.
/// Uses the param_to_entry_arg mapping to trace back to entry-level values.
///
/// `at_entry` must be true only when `body` is the entry body itself.
/// Ranges constructed in called functions have bounds that aren't accessible
/// at entry level, so we only report `Range` provenance at entry depth.
fn track_provenance_unified(
    entry: &EntryPoint,
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

    // Check if value is a StorageView — trace to entry input
    if let Some((param_index, storage_binding)) = resolve_storage_view(body, value, entry) {
        return ArrayProvenance::EntryStorage {
            name: entry.inputs[param_index].name.clone(),
            param_index,
            storage_binding,
        };
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

// =============================================================================
// Analysis Results
// =============================================================================

/// A parallelizable SOAC found in a compute shader.
#[derive(Debug, Clone)]
pub enum ParallelizableSoac {
    /// `map f inputs` — element-wise transformation, trivially parallel.
    Map {
        source: ArrayProvenance,
        map_function: String,
        captures: Vec<ValueId>,
        output_elem_type: Type<TypeName>,
    },
    /// `reduce f init input` — fold with associative operator.
    Reduce {
        source: ArrayProvenance,
        reduce_function: String,
        init: ValueId,
        captures: Vec<ValueId>,
        elem_type: Type<TypeName>,
    },
    /// `scan f init input` — prefix scan with associative operator.
    Scan {
        source: ArrayProvenance,
        scan_function: String,
        init: ValueId,
        captures: Vec<ValueId>,
        elem_type: Type<TypeName>,
    },
}

impl ParallelizableSoac {
    /// The provenance of the input array.
    pub fn source(&self) -> &ArrayProvenance {
        match self {
            ParallelizableSoac::Map { source, .. }
            | ParallelizableSoac::Reduce { source, .. }
            | ParallelizableSoac::Scan { source, .. } => source,
        }
    }

    /// Captured variables.
    pub fn captures(&self) -> &[ValueId] {
        match self {
            ParallelizableSoac::Map { captures, .. }
            | ParallelizableSoac::Reduce { captures, .. }
            | ParallelizableSoac::Scan { captures, .. } => captures,
        }
    }
}

/// Analysis results for a compute entry point.
#[derive(Debug, Clone)]
pub struct ComputeEntryAnalysis {
    /// Name of the entry point.
    pub name: String,
    /// Local size for the compute shader.
    pub local_size: (u32, u32, u32),
    /// The first parallelizable SOAC found (if any).
    pub parallelizable_soac: Option<ParallelizableSoac>,
}

/// Analysis results for an SSA program.
#[derive(Debug, Default)]
pub struct SoacAnalysis {
    /// Analysis for each compute entry point.
    pub by_entry: HashMap<String, ComputeEntryAnalysis>,
}

// =============================================================================
// Public API
// =============================================================================

/// Analyze an SSA program for parallelizable SOAC patterns.
pub fn analyze_program(program: &Program) -> SoacAnalysis {
    let mut analysis = SoacAnalysis::default();

    for entry in &program.entry_points {
        if let ExecutionModel::Compute { local_size } = entry.execution_model {
            let entry_analysis = analyze_compute_entry(entry, local_size, program);
            analysis.by_entry.insert(entry.name.clone(), entry_analysis);
        }
    }

    analysis
}

/// Find a function by name in the program.
fn find_function<'a>(program: &'a Program, name: &str) -> Option<&'a Function> {
    program.functions.iter().find(|f| f.name == name)
}

/// Maximum call depth for traversing through function calls.
const MAX_CALL_DEPTH: usize = 10;

/// Analyze a single compute entry point.
fn analyze_compute_entry(
    entry: &EntryPoint,
    local_size: (u32, u32, u32),
    program: &Program,
) -> ComputeEntryAnalysis {
    let par_soac = find_highest_soac(entry, program);
    ComputeEntryAnalysis {
        name: entry.name.clone(),
        local_size,
        parallelizable_soac: par_soac,
    }
}

/// Find the highest (closest to entry point) parallelizable SOAC in the call tree.
/// Uses breadth-first search: checks each level before recursing into called functions.
fn find_highest_soac(entry: &EntryPoint, program: &Program) -> Option<ParallelizableSoac> {
    // Entry params map to themselves for provenance tracking
    let initial_mapping: HashMap<usize, ValueId> =
        entry.body.params.iter().enumerate().map(|(i, (v, _, _))| (i, *v)).collect();

    find_soac_in_body(entry, &entry.body, &initial_mapping, program, 0)
}

/// Search for a parallelizable SOAC in a function body, recursing into called functions.
fn find_soac_in_body(
    entry: &EntryPoint,
    body: &FuncBody,
    param_to_entry_arg: &HashMap<usize, ValueId>,
    program: &Program,
    depth: usize,
) -> Option<ParallelizableSoac> {
    if depth > MAX_CALL_DEPTH {
        return None;
    }

    // Breadth-first: check for SOAC instructions at THIS level first
    for inst in &body.insts {
        match &inst.kind {
            InstKind::Soac(Soac::Map {
                func,
                inputs,
                captures,
                output_elem_type,
                ..
            }) => {
                if let Some(input_value) = inputs.first() {
                    let provenance =
                        track_provenance_unified(entry, body, *input_value, param_to_entry_arg, depth == 0);
                    if matches!(
                        provenance,
                        ArrayProvenance::EntryStorage { .. } | ArrayProvenance::Range { .. }
                    ) {
                        return Some(ParallelizableSoac::Map {
                            source: provenance,
                            map_function: func.clone(),
                            captures: captures.clone(),
                            output_elem_type: output_elem_type.clone(),
                        });
                    }
                }
            }
            InstKind::Soac(Soac::Reduce {
                func,
                input,
                init,
                captures,
                input_elem_type,
                ..
            }) => {
                let provenance =
                    track_provenance_unified(entry, body, *input, param_to_entry_arg, depth == 0);
                if matches!(
                    provenance,
                    ArrayProvenance::EntryStorage { .. } | ArrayProvenance::Range { .. }
                ) {
                    return Some(ParallelizableSoac::Reduce {
                        source: provenance,
                        reduce_function: func.clone(),
                        init: *init,
                        captures: captures.clone(),
                        elem_type: input_elem_type.clone(),
                    });
                }
            }
            InstKind::Soac(Soac::Scan {
                func,
                input,
                init,
                captures,
                input_elem_type,
                ..
            }) => {
                let provenance =
                    track_provenance_unified(entry, body, *input, param_to_entry_arg, depth == 0);
                if matches!(
                    provenance,
                    ArrayProvenance::EntryStorage { .. } | ArrayProvenance::Range { .. }
                ) {
                    return Some(ParallelizableSoac::Scan {
                        source: provenance,
                        scan_function: func.clone(),
                        init: *init,
                        captures: captures.clone(),
                        elem_type: input_elem_type.clone(),
                    });
                }
            }
            _ => {}
        }
    }

    // No SOAC at this level - recurse into called functions
    for inst in &body.insts {
        if let InstKind::Call { func, args } = &inst.kind {
            // Skip intrinsic calls
            if func.starts_with("_w_") {
                continue;
            }
            if let Some(called_func) = find_function(program, func) {
                let new_mapping = build_param_mapping(body, param_to_entry_arg, args, entry);

                if let Some(par_soac) =
                    find_soac_in_body(entry, &called_func.body, &new_mapping, program, depth + 1)
                {
                    return Some(par_soac);
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
    entry: &EntryPoint,
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
            if let Some((input_idx, _)) = resolve_storage_view(caller_body, arg, entry) {
                let entry_val = entry.body.params[input_idx].0;
                return Some((i, entry_val));
            }
            None
        })
        .collect()
}
