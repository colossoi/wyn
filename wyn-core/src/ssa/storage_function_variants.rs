//! Plan backend function emission when storage access differs by entry.
//!
//! Storage bindings are statically embedded in view types, so a helper that
//! touches one cannot select a different module global at runtime. Most
//! helpers either touch no storage or are reached under one access signature;
//! only helpers reached under conflicting signatures receive variants. The
//! plan contains indices and emitted names only and never owns or rewrites IR.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::op::OpTag;
use crate::ssa::types::{FuncBody, InstKind, Program};
use crate::{BindingRef, LookupMap, ResourceAccess};

#[derive(Clone, Debug)]
pub(crate) struct FunctionEmission {
    pub(crate) function: usize,
    pub(crate) name: String,
    pub(crate) entry_context: Option<usize>,
}

pub(crate) struct StorageFunctionVariants {
    emissions: Vec<FunctionEmission>,
    entry_names: Vec<LookupMap<String, String>>,
    module_accesses: LookupMap<BindingRef, ResourceAccess>,
}

impl StorageFunctionVariants {
    pub(crate) fn new(program: &Program) -> Self {
        let function_indices = program
            .functions
            .iter()
            .enumerate()
            .map(|(index, function)| (function.name.clone(), index))
            .collect::<HashMap<_, _>>();
        let calls = program
            .functions
            .iter()
            .map(|function| body_calls(&function.body, &function_indices))
            .collect::<Vec<_>>();
        let mut dependencies = program
            .functions
            .iter()
            .map(|function| direct_storage_bindings(&function.body))
            .collect::<Vec<_>>();
        loop {
            let previous = dependencies.clone();
            let mut changed = false;
            for (function, callees) in calls.iter().enumerate() {
                for &callee in callees {
                    let old_len = dependencies[function].len();
                    dependencies[function].extend(previous[callee].iter().copied());
                    changed |= dependencies[function].len() != old_len;
                }
            }
            if !changed {
                break;
            }
        }

        let entry_accesses =
            program.entry_points.iter().map(|entry| entry.shader_storage_accesses()).collect::<Vec<_>>();
        let mut module_accesses = LookupMap::new();
        for accesses in &entry_accesses {
            for (&binding, &access) in accesses {
                module_accesses
                    .entry(binding)
                    .and_modify(|current: &mut ResourceAccess| *current = current.merge(access))
                    .or_insert(access);
            }
        }

        let reachable = program
            .entry_points
            .iter()
            .map(|entry| reachable_functions(&entry.body, &calls, &function_indices))
            .collect::<Vec<_>>();
        let mut emissions = Vec::new();
        let mut entry_names = vec![LookupMap::new(); program.entry_points.len()];
        for (function_index, function) in program.functions.iter().enumerate() {
            let mut contexts = BTreeMap::<Vec<(u32, u32, bool)>, Vec<usize>>::new();
            for (entry_index, functions) in reachable.iter().enumerate() {
                if functions.contains(&function_index) {
                    let mut signature = dependencies[function_index]
                        .iter()
                        .map(|&binding| {
                            let writable = entry_accesses[entry_index]
                                .get(&binding)
                                .is_none_or(|access| access.writes());
                            (binding.set, binding.binding, writable)
                        })
                        .collect::<Vec<_>>();
                    signature.sort_unstable();
                    contexts.entry(signature).or_default().push(entry_index);
                }
            }

            if contexts.is_empty() {
                emissions.push(FunctionEmission {
                    function: function_index,
                    name: function.name.clone(),
                    entry_context: None,
                });
                continue;
            }

            let needs_variants = contexts.len() > 1;
            for (variant_index, entries) in contexts.values().enumerate() {
                let name = if needs_variants {
                    format!("_w_storage_{function_index}_{variant_index}_{}", function.name)
                } else {
                    function.name.clone()
                };
                for &entry in entries {
                    entry_names[entry].insert(function.name.clone(), name.clone());
                }
                emissions.push(FunctionEmission {
                    function: function_index,
                    name,
                    entry_context: entries.first().copied(),
                });
            }
        }

        Self {
            emissions,
            entry_names,
            module_accesses,
        }
    }

    pub(crate) fn emissions(&self) -> &[FunctionEmission] {
        &self.emissions
    }

    pub(crate) fn names_for_entry(&self, entry: usize) -> &LookupMap<String, String> {
        &self.entry_names[entry]
    }

    pub(crate) fn accesses_for<'a>(
        &'a self,
        program: &'a Program,
        entry: Option<usize>,
    ) -> LookupMap<BindingRef, ResourceAccess> {
        entry
            .map(|index| program.entry_points[index].shader_storage_accesses())
            .unwrap_or_else(|| self.module_accesses.clone())
    }
}

fn body_calls(body: &FuncBody, indices: &HashMap<String, usize>) -> Vec<usize> {
    body.inner
        .insts
        .iter()
        .filter_map(|(_, instruction)| match &instruction.data {
            InstKind::Op {
                tag: OpTag::Call(callee),
                ..
            } => indices.get(callee).copied(),
            _ => None,
        })
        .collect()
}

fn direct_storage_bindings(body: &FuncBody) -> HashSet<BindingRef> {
    body.inner
        .values
        .iter()
        .filter_map(|(_, value)| crate::types::array_view_buffer(&value.ty))
        .collect()
}

fn reachable_functions(
    body: &FuncBody,
    calls: &[Vec<usize>],
    indices: &HashMap<String, usize>,
) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    let mut pending = VecDeque::from(body_calls(body, indices));
    while let Some(function) = pending.pop_front() {
        if reachable.insert(function) {
            pending.extend(calls[function].iter().copied());
        }
    }
    reachable
}
