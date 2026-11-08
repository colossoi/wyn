//! Physical ranges for fresh scalar output allocations. Logical allocation keys
//! and scheduling dependencies stay intact; all ABI consumers use this plan.
use crate::egglog::{output_error as error, OptimizeError};
use crate::ssa::layout::storage_elem_stride;
use crate::types::{Type, TypeExt, TypeName};
use crate::{BindingRef, LookupMap};
use egglog_engine::Value;

#[derive(Clone)]
pub(super) struct Output {
    pub writer: Value,
    pub binding: BindingRef,
    pub element: Type,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::egglog) struct Placement {
    pub binding: BindingRef,
    pub element_offset: u32,
    pub stride: u32,
    pub allocation_bytes: u32,
    pub name: String,
}
impl Placement {
    pub fn range(&self) -> (u32, u32) {
        (self.element_offset * self.stride, self.stride)
    }
}

pub(super) fn plan(outputs: Vec<Output>) -> Result<LookupMap<BindingRef, Placement>, OptimizeError> {
    let mut groups: Vec<Vec<Output>> = Vec::new();
    let mut widths = LookupMap::<Value, usize>::default();
    for output in outputs {
        let scalar =
            if output.element.is_vec() { output.element.elem_type().unwrap() } else { &output.element };
        // Set zero is compiler-owned. Four-byte components keep shader storage
        // and WHL GPU-copy alignment compatible without representation changes.
        if output.binding.set != 0
            || !matches!(
                scalar,
                Type::Constructed(TypeName::Float(32) | TypeName::Int(32) | TypeName::UInt(32), _)
            )
        {
            continue;
        }
        *widths.entry(output.writer).or_default() += 1;
        if let Some(group) =
            groups.iter_mut().find(|g| g[0].writer == output.writer && g[0].element == output.element)
        {
            group.push(output);
        } else {
            groups.push(vec![output]);
        }
    }
    let mut result = LookupMap::default();
    for group in groups {
        // Small records do not warrant extra result materialization. Fields
        // sharing a producer have the same lifetime start and cannot add cycles.
        if group.len() < 2 || widths[&group[0].writer] < 8 {
            continue;
        }
        let stride =
            storage_elem_stride(&group[0].element).ok_or_else(|| error("packed output stride missing"))?;
        let allocation_bytes = u32::try_from(group.len())
            .ok()
            .and_then(|n| n.checked_mul(stride))
            .ok_or_else(|| error("packed output capacity overflow"))?;
        let binding = group[0].binding;
        let name = group[0].name.clone();
        for (index, field) in group.into_iter().enumerate() {
            result.insert(
                field.binding,
                Placement {
                    binding,
                    element_offset: index as u32,
                    stride,
                    allocation_bytes,
                    name: name.clone(),
                },
            );
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "packing_tests.rs"]
mod tests;
