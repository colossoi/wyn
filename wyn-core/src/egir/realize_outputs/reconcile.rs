//! Propagate storage-view representation drift after output realization.
//!
//! Output realization can rewrite a producer's result node from an in-register
//! `Composite` array to a storage `View` — a `map` whose result is a declared
//! entry output has its lane store retargeted into the output buffer, and the
//! result node is retyped to that view (`dispatch::compute_slot_source`). Only
//! the producing node is retyped; anything that *forwards* that value keeps its
//! stale `Composite` type. A runtime-sized `Composite` array has no SPIR-V form,
//! so it panics in lowering when it reaches codegen through one of those forwards.
//!
//! Two forwards occur in practice, both handled here as one representation-drift
//! propagation:
//!
//!   * **Aggregates.** `let w = { points = p }` interns a record whose field
//!     type was fixed before `p` drifted. Recompute `Tuple`/`Project` node types
//!     from their operands so the aggregate reflects the view (`Record([view])`).
//!   * **Call boundaries.** A `Seg` body capture — the record `w` passed to a
//!     downstream `map`, or a bare array read by another `map` — binds a callee
//!     parameter. Ordinary named-function arguments are call boundaries too.
//!     The parameter type must equal the argument type, so retype the
//!     callee (its callable `Func` and the body's `FuncParam` node) and
//!     re-propagate inside the callee. An `Index` over a
//!     view — bare or projected out of a record parameter — then lowers as a
//!     storage load, variant-generic like every other view read.

use polytype::Type;

use super::super::from_tlc::ConvertError;
use super::super::ir::{Body, BodySite, ProgramShape};
use super::super::program::{Func, Program};
use super::super::types::{
    EGraph, EffectOp, PureOp, RegionId, SideEffectKind, SoacEffect, ValueId, ValueKind, WynSoacPhase,
};
use crate::ast::TypeName;

/// One parameter retype: the callee `region`/function `name`, the parameter
/// index, and the type its capture argument drifted to.
struct Retype {
    region: RegionId,
    name: String,
    param_index: usize,
    arg_ty: Type<TypeName>,
}

pub fn reconcile_representation_drift<Tag, Shape, GlobalContext>(
    program: Program<Tag, Shape, GlobalContext>,
) -> Result<Program<Tag, Shape, GlobalContext>, ConvertError>
where
    Shape: ProgramShape,
    Shape::Family: WynSoacPhase,
{
    // Phase A: propagate drift through aggregate node types in every body, so a
    // record/tuple holding a retargeted array reflects the view at its field.
    let mut program = program.map_graphs(|_, mut graph| {
        recompute_aggregate_types(&mut graph);
        graph
    });

    // Phase B: reconcile callee parameters to their capture arguments. Retyping a
    // callee parameter re-propagates inside that callee (`recompute_aggregate_types`
    // in `apply`), which can surface a fresh drift one level deeper — so iterate
    // to a fixpoint.
    // TODO: Replace this bounded loop with a real worklist/fixed-point algorithm;
    // report an error if progress becomes non-monotone instead of silently
    // succeeding after 64 iterations with representation drift still present.
    for _ in 0..64 {
        let mut drifts: Vec<Retype> = Vec::new();
        collect_drifts(
            program.entry_points.iter().map(|entry| &entry.graph),
            &program.functions,
            &mut drifts,
        );
        collect_drifts(
            program.functions.iter().map(|function| &function.graph),
            &program.functions,
            &mut drifts,
        );
        if drifts.is_empty() {
            break;
        }
        reject_shared_conflicts(&drifts)?;
        for drift in drifts {
            program = apply(program, drift);
        }
    }

    Ok(program)
}

/// Scan every `Seg` body and direct named call in `graphs` for an argument
/// whose type has drifted view-ward from the callee parameter it binds,
/// pushing a retype for each.
fn collect_drifts<'a, P: WynSoacPhase + 'a>(
    graphs: impl Iterator<Item = &'a EGraph<P>>,
    functions: &[Func<P>],
    out: &mut Vec<Retype>,
) {
    for graph in graphs {
        for (_, node) in &graph.nodes {
            let ValueKind::Pure {
                op: PureOp::Call(region),
                operands,
            } = &node.kind
            else {
                continue;
            };
            collect_call_drifts(graph, functions, *region, operands, out);
        }

        for (_, block) in &graph.skeleton.blocks {
            for se in &block.side_effects {
                if let SideEffectKind::Effect(EffectOp::Op {
                    tag: crate::op::OpTag::Call(region),
                }) = &se.kind
                {
                    collect_call_drifts(graph, functions, *region, &se.operand_nodes, out);
                }

                let SideEffectKind::Soac(SoacEffect(_, soac)) = &se.kind else {
                    continue;
                };
                for body in soac.seg_bodies() {
                    let Some(region) = functions.iter().find(|function| function.region == body.region)
                    else {
                        continue;
                    };
                    let n_caps = body.captures.len();
                    // Captures are the trailing parameters in every call
                    // convention (`[positional.., captures..]`), so capture `i`
                    // binds parameter `n_params - n_caps + i`.
                    let base = region.params.len().saturating_sub(n_caps);
                    for (i, &capture) in body.captures.iter().enumerate() {
                        let param_index = base + i;
                        let Some(cap_ty) = graph.nodes.get(capture).map(|node| &node.ty) else {
                            continue;
                        };
                        let Some(param_ty) = region.params.get(param_index).map(|(t, _)| t) else {
                            continue;
                        };
                        if is_view_ward_drift(param_ty, cap_ty) {
                            out.push(Retype {
                                region: body.region,
                                name: region.name.clone(),
                                param_index,
                                arg_ty: cap_ty.clone(),
                            });
                        }
                    }
                }
            }
        }
    }
}

fn collect_call_drifts<P: WynSoacPhase>(
    graph: &EGraph<P>,
    functions: &[Func<P>],
    region: RegionId,
    arguments: &[ValueId],
    out: &mut Vec<Retype>,
) {
    let Some(callee) = functions.iter().find(|function| function.region == region) else {
        return;
    };
    for (param_index, &argument) in arguments.iter().enumerate() {
        let Some(arg_ty) = graph.nodes.get(argument).map(|node| &node.ty) else {
            continue;
        };
        let Some(param_ty) = callee.params.get(param_index).map(|(ty, _)| ty) else {
            continue;
        };
        if is_view_ward_drift(param_ty, arg_ty) {
            out.push(Retype {
                region,
                name: callee.name.clone(),
                param_index,
                arg_ty: arg_ty.clone(),
            });
        }
    }
}

/// A region shared by two callers that disagree on a parameter type cannot be
/// retyped in place. This arises only when the same monomorphized callee is a
/// `Seg` body in two places and exactly one has its capture output-retargeted —
/// exotic, and today it reaches lowering as a hard panic. Reject it cleanly
/// instead; a dedicated per-caller clone can follow if a real program needs it.
fn reject_shared_conflicts(drifts: &[Retype]) -> Result<(), ConvertError> {
    for a in drifts {
        for b in drifts {
            if a.region == b.region && a.param_index == b.param_index && a.arg_ty != b.arg_ty {
                return Err(ConvertError::Unsupported(format!(
                    "region `{}` is captured by callers that disagree on parameter \
                     #{} after output retargeting; cloning the shared region per \
                     caller is not yet supported",
                    a.name, a.param_index
                )));
            }
        }
    }
    Ok(())
}

/// Retype the callee parameter — the semantic region and the lowered function
/// that share the name, plus each body's `FuncParam` node — then re-propagate
/// aggregate types inside the callee so a projection out of the parameter sees
/// the view.
fn apply<Tag, Shape, GlobalContext>(
    program: Program<Tag, Shape, GlobalContext>,
    Retype {
        region,
        name: _,
        param_index,
        arg_ty,
    }: Retype,
) -> Program<Tag, Shape, GlobalContext>
where
    Shape: ProgramShape,
    Shape::Family: WynSoacPhase,
{
    program.rewrite_body(BodySite::Function(region), |body| {
        let Body::Function(mut function) = body else {
            unreachable!()
        };
        if let Some(slot) = function.params.get_mut(param_index) {
            slot.0 = arg_ty.clone();
        }
        retype_func_param(&mut function.graph, param_index, &arg_ty);
        recompute_aggregate_types(&mut function.graph);
        Body::Function(function)
    })
}

/// Propagate storage-view representation *view-ward only* through `Tuple` and
/// `Project` node types, to a fixpoint. A `Tuple` (record or tuple construction)
/// inherits its operands' current types; a `Project` picks the selected field
/// out of its operand's aggregate type. Every other node's type is independent
/// of representation drift in its inputs (an `Index` yields the element type
/// whether the array is composite or a view), so only these two carry a view
/// outward through the graph.
///
/// Updates are gated on `is_view_ward_drift`: a node's type only ever moves
/// Composite → View, never back. Output realization deliberately retypes a
/// projection that streams into an output view (`w.points`) ahead of its
/// operand tuple; recomputing that projection from the still-composite tuple
/// field would undo the retarget and oscillate. The view-ward gate keeps the
/// retarget and makes the fixpoint monotone.
fn recompute_aggregate_types<P: WynSoacPhase>(graph: &mut EGraph<P>) {
    loop {
        let mut updates: Vec<(ValueId, Type<TypeName>)> = Vec::new();
        for (nid, node) in &graph.nodes {
            let ValueKind::Pure { op, operands } = &node.kind else {
                continue;
            };
            let cur = &node.ty;
            let recomputed = match op {
                PureOp::Tuple(_) => {
                    let Type::Constructed(name, _) = cur else {
                        continue;
                    };
                    let mut args = Vec::with_capacity(operands.len());
                    let mut ready = true;
                    for &operand in operands.iter() {
                        match graph.nodes.get(operand) {
                            Some(node) => args.push(node.ty.clone()),
                            None => {
                                ready = false;
                                break;
                            }
                        }
                    }
                    if !ready {
                        continue;
                    }
                    Type::Constructed(name.clone(), args)
                }
                PureOp::Project { index } => {
                    let Some(&base) = operands.first() else {
                        continue;
                    };
                    let Some(Type::Constructed(_, args)) = graph.nodes.get(base).map(|node| &node.ty)
                    else {
                        continue;
                    };
                    let Some(field) = args.get(*index as usize) else {
                        continue;
                    };
                    field.clone()
                }
                _ => continue,
            };
            if is_view_ward_drift(cur, &recomputed) {
                updates.push((nid, recomputed));
            }
        }
        if updates.is_empty() {
            break;
        }
        for (nid, ty) in updates {
            graph.retype_node(nid, ty);
        }
    }
}

/// True iff `cap` differs from `param` by an array that became a storage view —
/// the exact shape output retargeting introduces, possibly nested inside a
/// record/tuple. A view-ward-only test ignores benign type-variable renaming so
/// the pass never churns a parameter that has not actually drifted.
fn is_view_ward_drift(param: &Type<TypeName>, cap: &Type<TypeName>) -> bool {
    match (param, cap) {
        (Type::Constructed(TypeName::Array, pa), Type::Constructed(TypeName::Array, ca)) => {
            let p_view = matches!(pa.get(1), Some(Type::Constructed(TypeName::ArrayVariantView, _)));
            let c_view = matches!(ca.get(1), Some(Type::Constructed(TypeName::ArrayVariantView, _)));
            (!p_view && c_view) || pa.iter().zip(ca).any(|(p, c)| is_view_ward_drift(p, c))
        }
        (Type::Constructed(pn, pa), Type::Constructed(cn, ca)) if pn == cn && pa.len() == ca.len() => {
            pa.iter().zip(ca).any(|(p, c)| is_view_ward_drift(p, c))
        }
        _ => false,
    }
}

/// Retype the `FuncParam { index }` node in a region/function body graph.
fn retype_func_param<P: WynSoacPhase>(graph: &mut EGraph<P>, index: usize, view_ty: &Type<TypeName>) {
    let target = graph.nodes.iter().find_map(|(nid, node)| match &node.kind {
        ValueKind::FuncParam { index: i } if *i == index => Some(nid),
        _ => None,
    });
    if let Some(nid) = target {
        graph.retype_node(nid, view_ty.clone());
    }
}
