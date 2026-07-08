//! Reconcile Seg region parameter types with their captures.
//!
//! Output realization can rewrite a producer's result node from an in-register
//! `Composite` array to a storage `View` — a `map` whose result is a declared
//! entry output has its lane store retargeted into the output buffer, and the
//! result node is retyped to that view (`dispatch::compute_slot_source`). When
//! that same result is *also* captured by another Seg body (e.g. a second `map`
//! that reads `occ[j % 4]`), the capturing region's parameter still carries the
//! pre-retarget `Composite` type. A runtime-sized `Composite` array has no
//! SPIR-V form, so the region body would panic in lowering.
//!
//! A region parameter is bound to its capture argument at the call site, so the
//! two types must agree. This pass restores that invariant: for every Seg body
//! capture whose node type has drifted to a storage view, retype the callee's
//! parameter — both the semantic `EgirRegion` and the lowered `EgirFunc`, keyed
//! by the shared region name — and the region body's `FuncParam` node. An
//! `Index` over a view parameter then lowers as a storage load, variant-generic
//! like every other view read.

use polytype::Type;

use super::super::from_tlc::ConvertError;
use super::super::program::EgirInner;
use super::super::types::{ENode, RegionId, SideEffectKind};
use crate::ast::TypeName;
use crate::types::TypeExt;

/// One parameter retype: the callee `region`/function `name`, the parameter
/// index, and the view type the capture drifted to.
struct Retype {
    region: RegionId,
    name: String,
    param_index: usize,
    view_ty: Type<TypeName>,
}

pub fn run(inner: &mut EgirInner) -> Result<(), ConvertError> {
    let EgirInner {
        entry_points,
        regions,
        functions,
        ..
    } = inner;

    // Phase 1: collect capture→parameter type drifts (Composite → view).
    let mut retypes: Vec<Retype> = Vec::new();
    for entry in entry_points.iter() {
        let graph = &entry.graph;
        for (_, block) in &graph.skeleton.blocks {
            for se in &block.side_effects {
                let SideEffectKind::Soac(soac) = &se.kind else {
                    continue;
                };
                for body in soac.seg_bodies() {
                    let Some(region) = regions.get(&body.region) else {
                        continue;
                    };
                    let n_caps = body.captures.len();
                    // Captures are the trailing parameters in every call
                    // convention (`[positional.., captures..]`), so capture `i`
                    // binds parameter `n_params - n_caps + i`.
                    let base = region.params.len().saturating_sub(n_caps);
                    for (i, &capture) in body.captures.iter().enumerate() {
                        let param_index = base + i;
                        let Some(cap_ty) = graph.types.get(&capture) else {
                            continue;
                        };
                        let Some(param_ty) = region.params.get(param_index).map(|(t, _)| t) else {
                            continue;
                        };
                        if is_view_array(cap_ty) && param_ty != cap_ty {
                            retypes.push(Retype {
                                region: body.region,
                                name: region.name.clone(),
                                param_index,
                                view_ty: cap_ty.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    // A region shared by two callers that disagree on a parameter type cannot be
    // retyped in place. This arises only when the same monomorphized lambda is a
    // Seg body in two places and exactly one has its capture output-retargeted —
    // exotic, and today it reaches lowering as a hard panic. Reject it cleanly
    // instead; a dedicated per-caller clone can follow if a real program needs it.
    for a in &retypes {
        for b in &retypes {
            if a.region == b.region && a.param_index == b.param_index && a.view_ty != b.view_ty {
                return Err(ConvertError::Unsupported(format!(
                    "region `{}` is captured by callers that disagree on parameter \
                     #{} after output retargeting; cloning the shared region per \
                     caller is not yet supported",
                    a.name, a.param_index
                )));
            }
        }
    }

    // Phase 2: apply. Retype the semantic region and the lowered function that
    // share the name, plus each body's `FuncParam` node.
    for Retype {
        region,
        name,
        param_index,
        view_ty,
    } in retypes
    {
        if let Some(region) = regions.get_mut(&region) {
            if let Some(slot) = region.params.get_mut(param_index) {
                slot.0 = view_ty.clone();
            }
            retype_func_param(&mut region.graph, param_index, &view_ty);
        }
        if let Some(func) = functions.iter_mut().find(|f| f.name == name) {
            if let Some(slot) = func.params.get_mut(param_index) {
                slot.0 = view_ty.clone();
            }
            retype_func_param(&mut func.graph, param_index, &view_ty);
        }
    }

    Ok(())
}

/// True iff `ty` is a storage-view array — the shape output retargeting rewrites
/// a captured `Composite` result into.
fn is_view_array(ty: &Type<TypeName>) -> bool {
    ty.is_array()
        && matches!(
            ty.array_variant(),
            Some(Type::Constructed(TypeName::ArrayVariantView, _))
        )
}

/// Retype the `FuncParam { index }` node in a region/function body graph.
fn retype_func_param(graph: &mut super::super::types::EGraph, index: usize, view_ty: &Type<TypeName>) {
    let target = graph.nodes.iter().find_map(|(nid, node)| match node {
        ENode::FuncParam { index: i } if *i == index => Some(nid),
        _ => None,
    });
    if let Some(nid) = target {
        graph.types.insert(nid, view_ty.clone());
    }
}
