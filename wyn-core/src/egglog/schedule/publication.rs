//! Assign backend identities to the completed stage and resource plan.
use super::{error, FunctionKind, OptimizeError};
use crate::egglog::abi::Abi;
use crate::egglog::blocks::{BlockData, DispatchData};
use crate::egglog::data::{BlockId, DispatchId, Ir};
use crate::egglog::names;
use crate::egglog::planning::rows;
use crate::host::DispatchSize;
use crate::interface::EntryKind;
use crate::kernel_graph::{KernelDomain, KernelId, PhysicalKernel, PhysicalKernelGraph};
use crate::{EntryId, ResourceId, ResourceUse};
use egglog_engine::{EGraph, Value as EggValue};
use std::collections::{BTreeMap, HashMap};
use wyn_base::IdArena;

pub(super) fn build_physical_kernel_graph(
    graph: &EGraph,
    roots: &HashMap<EggValue, BlockId>,
    order: &[DispatchId],
    abi: &mut Abi,
    dispatches: &IdArena<DispatchId, DispatchData>,
    ir: &Ir,
    blocks: &IdArena<BlockId, BlockData>,
) -> Result<PhysicalKernelGraph, OptimizeError> {
    let entries = &ir.entries;
    for &id in order {
        let d = &dispatches[id];
        abi.entry_roots.entry(d.owner).or_default().push(d.kernel);
    }
    for &(root, owner, _, finish) in &abi.roots {
        if finish {
            abi.entry_roots.entry(owner).or_default().push(root);
        }
    }
    abi.entry_names = names::entry_points(abi, ir, blocks)?;
    let by_binding: BTreeMap<_, _> =
        abi.buffer_bindings.iter().map(|(&id, &binding)| (binding, id)).collect();
    let mut kernels = vec![];
    let mut identities = BTreeMap::new();
    for (&owner, roots) in &abi.entry_roots {
        if entries[owner].declaration.entry_kind != EntryKind::Compute {
            continue;
        }
        for &root in roots {
            let id = KernelId::from(kernels.len() as u32);
            identities.insert(root, id);
            let Some(function) = &blocks[root].interface else {
                return Err(error("shader root has no interface"));
            };
            let size = match function.kind {
                FunctionKind::Kernel([x, y, z]) => (x, y, z),
                _ => (1, 1, 1),
            };
            let domain = match &abi.dispatch_sizes[&root] {
                DispatchSize::Fixed { x, y, z, .. } => KernelDomain::Fixed { x: *x, y: *y, z: *z },
                DispatchSize::DerivedFrom { len, workgroup_size } if *workgroup_size == size.0 => {
                    KernelDomain::Elements(len.clone())
                }
                DispatchSize::DerivedFrom { len, workgroup_size }
                    if *workgroup_size > 0 && *workgroup_size % size.0 == 0 =>
                {
                    KernelDomain::ChunkedElements {
                        len: len.clone(),
                        chunk_size: *workgroup_size / size.0,
                    }
                }
                DispatchSize::DerivedFrom { .. } => {
                    return Err(error("launch divisor is not a whole number of workgroups"))
                }
            };
            kernels.push(PhysicalKernel {
                id,
                entry: EntryId::from(root.as_u32()),
                entry_point: abi.entry_names[&root].clone(),
                label: function.name.clone(),
                source_entry: Some(EntryId::from(owner.as_u32())),
                output_routes: vec![],
                workgroup_size: size,
                domain,
                resources: abi.root_accesses[&root]
                    .iter()
                    .filter_map(|(binding, &access)| {
                        by_binding.get(binding).map(|buffer| ResourceUse {
                            resource: ResourceId::from_egglog_buffer(buffer.as_u32()),
                            access,
                        })
                    })
                    .collect(),
                dependencies: vec![],
            });
        }
    }
    rows(graph, "AbiRootDependency", |a| {
        let before = roots[&a[0]];
        let after = roots[&a[1]];
        let (Some(&before), Some(&after)) = (identities.get(&before), identities.get(&after)) else {
            return Err(error("dependency without a compute root"));
        };
        kernels[after.index()].dependencies.push(before);
        Ok(())
    })?;
    for kernel in &mut kernels {
        kernel.dependencies.sort_unstable();
        kernel.resources.sort_by_key(|r| r.resource);
    }
    PhysicalKernelGraph::from_ordered(kernels).map_err(|e| error(&e))
}
