//! Output actions share bounds checking and serial/atomic accumulation.
use super::super::{builder_error, error, Body, OptimizeError, Typed};
use super::contribution::Contribution;
use super::{coordinates, invocation};
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::{AtomicOp, InstKind};
use crate::types;
use crate::LookupMap;
use egglog_engine::Value;

#[derive(Clone, Copy)]
enum Update {
    Store,
    Add,
    Combine(Value),
    Atomic(AtomicOp),
    AtomicCombine(Value),
}

/// Add/combine return the previous value, allowing the same primitive to reserve
/// a slot. Store returns its argument and never reads uninitialized storage.
fn apply(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    output: Typed,
    key: Typed,
    value: Typed,
    update: Update,
) -> Result<Typed, OptimizeError> {
    match update {
        Update::Store => {
            body.store(output, key, value.clone())?;
            Ok(value)
        }
        Update::Add | Update::Combine(_) => {
            let old = body.index(output.clone(), key.clone())?;
            let next = match update {
                Update::Combine(operation) => {
                    body.apply_region(scope, operation, vec![old.clone(), value])?
                }
                _ => body.binary(BinaryOperator::Add, old.clone(), value)?,
            };
            body.store(output, key, next)?;
            Ok(old)
        }
        Update::Atomic(op) => {
            let (place, ty) = body.index_place(output, key)?;
            let value = body.cast(value, &ty)?;
            let result = body
                .builder
                .push_inst(
                    InstKind::Atomic {
                        place,
                        op,
                        values: vec![value.value],
                    },
                    ty.clone(),
                )
                .map_err(builder_error)?;
            Ok(Typed {
                value: result.into(),
                ty,
            })
        }
        Update::AtomicCombine(operation) => {
            let (place, ty) = body.index_place(output, key)?;
            let value = body.cast(value, &ty)?;
            let old = body
                .builder
                .push_inst(
                    InstKind::Atomic {
                        place,
                        op: AtomicOp::Load,
                        values: vec![],
                    },
                    ty.clone(),
                )
                .map_err(builder_error)?;
            let old = Typed {
                value: old.into(),
                ty: ty.clone(),
            };
            let state_ty = types::tuple(vec![ty, types::bool_type()]);
            let done = body.op(OpTag::Bool(false), vec![], types::bool_type())?;
            let initial = body.tuple(vec![old, done])?;
            let state = body.retry(initial, |body, state| {
                let old = body.field(state, 0)?;
                let next = body.apply_region(scope, operation, vec![old.clone(), value.clone()])?;
                let result = body
                    .builder
                    .push_inst(
                        InstKind::Atomic {
                            place,
                            op: AtomicOp::CompareExchange,
                            values: vec![old.value, next.value],
                        },
                        state_ty.clone(),
                    )
                    .map_err(builder_error)?;
                Ok(Typed {
                    value: result.into(),
                    ty: state_ty.clone(),
                })
            })?;
            body.field(state, 0)
        }
    }
}

fn iteration(
    body: &mut Body<'_, '_, '_>,
    phase: &str,
    width: u32,
) -> Result<(Typed, Typed), OptimizeError> {
    match phase {
        "ordered" => Ok((Body::number(0), Body::number(1))),
        "local" => Ok((body.local_id()?, Body::number(width))),
        _ => invocation(body, width),
    }
}

pub(super) fn initialize(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    width: u32,
) -> Result<(), OptimizeError> {
    let arrays = body.compiler.facts.initialized_arrays(owner)?;
    if arrays.is_empty() {
        return Ok(());
    }
    for (resource, source) in arrays {
        let output = body.resource(scope, resource, 2)?;
        let n = body.length(output.clone())?;
        let (start, step) = iteration(body, phase, width)?;
        if let Some(copy) = body.compiler.facts.enode("CopyFrom", source) {
            let initial = body.value(scope, copy[0])?;
            body.copy_array_strided((output, &[]), initial, n, &[], start, step)?;
        } else if body.compiler.facts.enode("ZeroContents", source).is_some() {
            body.counted(start, n, step, vec![], |body, index, _| {
                body.store(output.clone(), index, Body::number(0))?;
                Ok(vec![])
            })?;
        } else {
            return Err(error("invalid result initializer"));
        }
    }
    if phase == "local" {
        body.workgroup_barrier()?;
    }
    Ok(())
}

/// A capability is available only while interpreting its declaring scope.
struct Capabilities(LookupMap<Value, (Typed, Update)>);

impl Capabilities {
    fn emit(
        &self,
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        program: Value,
        input: &Typed,
        bound: &[Typed],
    ) -> Result<(), OptimizeError> {
        if body.compiler.facts.enode("UpdatesDone", program).is_some() {
            return Ok(());
        }
        if let Some(fields) = body.compiler.facts.enode("ChooseUpdates", program) {
            let condition = body.formula(scope, fields[0], input, bound)?;
            body.branch(
                scope,
                condition,
                |body| {
                    self.emit(body, scope, fields[1], input, bound)?;
                    body.tuple(vec![])
                },
                |body| {
                    self.emit(body, scope, fields[2], input, bound)?;
                    body.tuple(vec![])
                },
                None,
            )?;
            return Ok(());
        }
        let Some(fields) = body.compiler.facts.enode("BindUpdate", program) else {
            return Err(error("invalid update program"));
        };
        let Some((output, mode)) = self.0.get(&fields[0]) else {
            return Err(error("update handle is outside its declaring scope"));
        };
        let coordinates = body
            .compiler
            .facts
            .vector(fields[1])?
            .into_iter()
            .map(|value| body.formula(scope, value, input, bound))
            .collect::<Result<Vec<_>, _>>()?;
        let value = body.formula(scope, fields[2], input, bound)?;
        let result = if matches!(mode, Update::Store) {
            body.store_at(output.clone(), &coordinates, value.clone())?;
            value
        } else {
            let Some((key, prefix)) = coordinates.split_last() else {
                return Err(error("accumulator update requires coordinates"));
            };
            let mut target = output.clone();
            for index in prefix {
                target = body.index(target, index.clone())?;
            }
            apply(body, scope, target, key.clone(), value, *mode)?
        };
        let mut bound = bound.to_vec();
        bound.push(result);
        self.emit(body, scope, fields[3], input, &bound)
    }
}

fn capabilities(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    plan: Value,
    phase: &str,
    declarations: Value,
) -> Result<Capabilities, OptimizeError> {
    let mut handles = LookupMap::default();
    for declaration in body.compiler.facts.vector(declarations)? {
        let Some(fields) = body.compiler.facts.enode("UpdateArray", declaration) else {
            return Err(error("invalid update declaration"));
        };
        let Some(handle) = body.compiler.facts.enode("ScopedUpdate", fields[0]) else {
            return Err(error("invalid update handle"));
        };
        if handle[0] != plan {
            return Err(error("update declaration belongs to another scope"));
        }
        let Some(base) = body.compiler.facts.lookup("DestinationSlot", (owner,)) else {
            return Err(error("update scope has no result slots"));
        };
        let slot = body.compiler.facts.integer(base) + body.compiler.facts.integer(handle[1]);
        let Some(resource) = body.compiler.facts.constructor("Result", (owner, slot)) else {
            return Err(error("update resource missing"));
        };
        let existing = body
            .compiler
            .facts
            .lookup("StorageDeclaration", (resource,))
            .and_then(|declaration| body.compiler.facts.enode("ExistingResult", declaration));
        let output = if let Some(initial) = existing {
            body.value(scope, initial[0])?
        } else {
            body.resource(scope, resource, 3)?
        };
        let mode = if body.compiler.facts.enode("Overwrite", fields[2]).is_some() {
            Update::Store
        } else if body.compiler.facts.enode("Adding", fields[2]).is_some() {
            if phase == "ordered" {
                Update::Add
            } else {
                Update::Atomic(AtomicOp::Add)
            }
        } else if body.compiler.facts.enode("IdenticalWrites", fields[2]).is_some() {
            if phase == "ordered" {
                Update::Store
            } else {
                Update::Atomic(AtomicOp::Exchange)
            }
        } else {
            let Some(combine) = body.compiler.facts.enode("Combining", fields[2]) else {
                return Err(error("invalid accumulator mode"));
            };
            if phase == "updates" {
                let Some(op) = body.compiler.facts.atomic(combine[0]) else {
                    return Err(error("atomic update missing"));
                };
                if op == AtomicOp::CompareExchange {
                    Update::AtomicCombine(combine[0])
                } else {
                    Update::Atomic(op)
                }
            } else {
                Update::Combine(combine[0])
            }
        };
        if handles.insert(fields[0], (output, mode)).is_some() {
            return Err(error("duplicate update capability"));
        }
    }
    Ok(Capabilities(handles))
}

pub(super) fn update(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    plan: Value,
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let Some(fields) = body.compiler.facts.record("PlanUpdates", "WithUpdates", (plan,)) else {
        return Err(error("parallel loop has no update scope"));
    };
    let handles = capabilities(body, scope, owner, plan, phase, fields[0])?;
    for block in body.compiler.facts.vector(fields[1])? {
        let Some(block) = body.compiler.facts.enode("EvaluateUpdates", block) else {
            return Err(error("invalid update block"));
        };
        Contribution {
            scope,
            plan,
            indices,
            skipped: &|body, _| body.tuple(vec![]),
        }
        .emit(body, block[0], cache, &mut |body, _, input| {
            handles.emit(body, scope, block[1], &input, &[])?;
            body.tuple(vec![])
        })?;
    }
    Ok(())
}

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    width: u32,
    plan: Value,
) -> Result<(), OptimizeError> {
    let dimensions = body
        .compiler
        .facts
        .iteration_dimensions(owner)?
        .into_iter()
        .map(|extent| body.extent(scope, extent))
        .collect::<Result<Vec<_>, _>>()?;
    let n = coordinates::length(body, &dimensions)?;
    let (start, step) = iteration(body, phase, width)?;
    body.counted(start, n, step, vec![], |body, index, _| {
        let coordinates = coordinates::decode(body, &dimensions, index)?;
        update(
            body,
            scope,
            owner,
            phase,
            plan,
            &coordinates,
            &mut LookupMap::default(),
        )?;
        Ok(vec![])
    })?;
    Ok(())
}
