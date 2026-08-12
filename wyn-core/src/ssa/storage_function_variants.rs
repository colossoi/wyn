//! Plan backend function emission when storage access differs by entry.
//!
//! Storage bindings are statically embedded in view types, so a helper that
//! touches one cannot select a different module global at runtime. Most
//! helpers either touch no storage or are reached under one access signature;
//! only helpers reached under conflicting signatures receive variants. The
//! plan contains indices and emitted names only and never owns or rewrites IR.

use std::collections::{BTreeMap, HashSet, VecDeque};

use crate::op::OpTag;
use crate::ssa::types::{FuncBody, InstKind, Program};
use crate::{BindingRef, LookupMap, ResourceAccess};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]

pub(crate) struct FunctionEmissionId(pub(crate) usize);

#[derive(Clone, Debug)]
pub(crate) struct FunctionEmission {
    /// Structural identity for this concrete storage-specialized emission.
    pub(crate) id: FunctionEmissionId,
    pub(crate) function: crate::FunctionId,
    pub(crate) name: String,
    pub(crate) entry_context: Option<crate::EntryId>,
}

pub(crate) struct StorageFunctionVariants {
    emissions: Vec<FunctionEmission>,
    /// Deterministic fallback emission for every local function. Entry-specific
    /// selections override it before lowering an entry or its helper variants.
    fallback_emissions: LookupMap<crate::FunctionId, FunctionEmissionId>,
    entry_emissions: LookupMap<crate::EntryId, LookupMap<crate::FunctionId, FunctionEmissionId>>,
    entry_names: LookupMap<crate::EntryId, LookupMap<crate::FunctionId, String>>,
    module_accesses: LookupMap<BindingRef, ResourceAccess>,
}

impl StorageFunctionVariants {
    pub(crate) fn new<Tag, GlobalContext>(program: &Program<Tag, GlobalContext>) -> Self {
        let function_indices = program
            .functions
            .iter()
            .enumerate()
            .map(|(index, function)| (function.id, index))
            .collect::<LookupMap<_, _>>();
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

        let entry_accesses = program
            .entry_points
            .iter()
            .map(|entry| (entry.id, entry.shader_storage_accesses()))
            .collect::<LookupMap<_, _>>();
        let mut module_accesses = LookupMap::new();
        for accesses in entry_accesses.values() {
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
            .map(|entry| {
                (
                    entry.id,
                    reachable_functions(&entry.body, &calls, &function_indices),
                )
            })
            .collect::<Vec<_>>();
        let mut emissions = Vec::new();
        let mut fallback_emissions = LookupMap::new();
        let mut entry_names = program
            .entry_points
            .iter()
            .map(|entry| (entry.id, LookupMap::new()))
            .collect::<LookupMap<_, _>>();
        let mut entry_emissions = program
            .entry_points
            .iter()
            .map(|entry| (entry.id, LookupMap::new()))
            .collect::<LookupMap<_, _>>();
        for (function_index, function) in program.functions.iter().enumerate() {
            let mut contexts = BTreeMap::<Vec<(u32, u32, bool)>, Vec<crate::EntryId>>::new();
            for (entry_id, functions) in &reachable {
                if functions.contains(&function_index) {
                    let mut signature = dependencies[function_index]
                        .iter()
                        .map(|&binding| {
                            let writable =
                                entry_accesses[entry_id].get(&binding).is_none_or(|access| access.writes());
                            (binding.set, binding.binding, writable)
                        })
                        .collect::<Vec<_>>();
                    signature.sort_unstable();
                    contexts.entry(signature).or_default().push(*entry_id);
                }
            }

            if contexts.is_empty() {
                let id = FunctionEmissionId(emissions.len());
                fallback_emissions.insert(function.id, id);
                emissions.push(FunctionEmission {
                    id,
                    function: function.id,
                    name: function.name.clone(),
                    entry_context: None,
                });
                continue;
            }

            let needs_variants = contexts.len() > 1;
            for (variant_index, entries) in contexts.values().enumerate() {
                let name = if needs_variants {
                    format!("_w_storage_{}_{}_{}", function.id, variant_index, function.name)
                } else {
                    function.name.clone()
                };
                for &entry in entries {
                    entry_emissions
                        .get_mut(&entry)
                        .expect("entry emission map must exist for every entry")
                        .insert(function.id, FunctionEmissionId(emissions.len()));
                    entry_names
                        .get_mut(&entry)
                        .expect("entry name map must exist for every entry")
                        .insert(function.id, name.clone());
                }
                emissions.push(FunctionEmission {
                    id: FunctionEmissionId(emissions.len()),
                    function: function.id,
                    name,
                    entry_context: entries.first().copied(),
                });
            }
        }

        // Every direct `OpTag::Call` must remain resolvable while lowering
        // helpers reached through module-level constant bodies. The selected
        // entry map below overrides this deterministic fallback for every
        // direct live entry path.
        for emission in &emissions {
            fallback_emissions.entry(emission.function).or_insert(emission.id);
        }

        Self {
            emissions,
            fallback_emissions,
            entry_emissions,
            entry_names,
            module_accesses,
        }
    }

    pub(crate) fn emissions(&self) -> &[FunctionEmission] {
        &self.emissions
    }

    pub(crate) fn names_for_entry(&self, entry: crate::EntryId) -> &LookupMap<crate::FunctionId, String> {
        &self.entry_names[&entry]
    }

    pub(crate) fn emissions_for_context(
        &self,
        entry: Option<crate::EntryId>,
    ) -> &LookupMap<crate::FunctionId, FunctionEmissionId> {
        match entry {
            Some(entry) => &self.entry_emissions[&entry],
            None => &self.fallback_emissions,
        }
    }

    pub(crate) fn emissions_for_entry(
        &self,
        entry: crate::EntryId,
    ) -> &LookupMap<crate::FunctionId, FunctionEmissionId> {
        self.emissions_for_context(Some(entry))
    }

    pub(crate) fn accesses_for<'a, Tag, GlobalContext>(
        &'a self,
        program: &'a Program<Tag, GlobalContext>,
        entry: Option<crate::EntryId>,
    ) -> LookupMap<BindingRef, ResourceAccess> {
        entry
            .map(|entry_id| {
                program
                    .entry_points
                    .iter()
                    .find(|entry| entry.id == entry_id)
                    .expect("emission entry context must be in the program")
                    .shader_storage_accesses()
            })
            .unwrap_or_else(|| self.module_accesses.clone())
    }
}

fn body_calls(body: &FuncBody, indices: &LookupMap<crate::FunctionId, usize>) -> Vec<usize> {
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
    indices: &LookupMap<crate::FunctionId, usize>,
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
