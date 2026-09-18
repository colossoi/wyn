use super::{array_value, error, length, singleton, Instruction, OptimizeError, Planner, Value};
use crate::egglog::data::{BlockId, OperationId, OperationKind};
use crate::ssa::types::AtomicOp;

impl Planner<'_> {
    pub(super) fn indexed(
        &mut self,
        op: OperationId,
        entry: BlockId,
        local: bool,
        atomic: Option<AtomicOp>,
    ) -> Result<(BlockId, Value), OptimizeError> {
        let owner = self.data.state.blocks[entry].function;
        match self.data.operations[op].kind.clone() {
            OperationKind::Scatter {
                destination,
                initialize,
                body,
                inputs,
            } => {
                let source = Value::Source(destination.value);
                let (entry, dest) = if initialize {
                    let dest = self.slot(op, "output", 0, local);
                    let copy = self.start_loop(
                        entry,
                        Value::Int(0),
                        Value::op("length", [source.clone()]),
                        vec![],
                    );
                    let value = self.load(copy.body, source, copy.index.clone(), "initial");
                    self.store(copy.body, &dest, copy.index.clone(), value);
                    self.finish_loop(&copy, copy.body, vec![]);
                    (copy.done, dest)
                } else {
                    (entry, source)
                };
                let loop_ = self.start_loop(entry, Value::Int(0), length(&inputs), vec![]);
                let args = self.read_inputs(loop_.body, &inputs, loop_.index.clone());
                let values = self.invoke_body(loop_.body, &body, args, "item")?;
                let [pair] = values.as_slice() else {
                    return Err(error("indexed write body must return a pair"));
                };
                let key = pair.clone().field(0);
                let write = self.block(owner, vec![]);
                let next = self.block(owner, vec![]);
                self.branch(
                    loop_.body,
                    in_bounds(key.clone(), Value::op("length", [dest.clone()])),
                    write,
                    next,
                    Some(next),
                );
                self.emit(
                    write,
                    Instruction::Store {
                        buffer: dest.clone(),
                        index: key,
                        value: pair.clone().field(1),
                    },
                );
                self.jump(write, next, vec![]);
                self.finish_loop(&loop_, next, vec![]);
                Ok((loop_.done, dest))
            }
            OperationKind::ReduceByIndex {
                destination,
                map,
                body,
                inputs,
                ..
            } => {
                // The supplied destination already contains the initial bins.
                // The neutral is an algebraic identity, not a request to clear it.
                let dest = Value::Source(destination.value);
                let n = length(&inputs);
                let loop_ = if atomic.is_some() {
                    self.invocations(entry, n)
                } else {
                    self.start_loop(entry, Value::Int(0), n, vec![])
                };
                let args = self.read_inputs(loop_.body, &inputs, loop_.index.clone());
                let values = self.invoke_body(loop_.body, &map, args, "item")?;
                let [key, value] = values.as_slice() else {
                    return Err(error("indexed mapping must return an index and a value"));
                };
                let update = self.block(owner, vec![]);
                let next = self.block(owner, vec![]);
                self.branch(
                    loop_.body,
                    in_bounds(key.clone(), Value::op("length", [dest.clone()])),
                    update,
                    next,
                    Some(next),
                );
                let mut end = update;
                if let Some(atomic) = atomic {
                    if atomic == AtomicOp::CompareExchange {
                        let old = self.atomic(update, &dest, key.clone(), AtomicOp::Load, vec![]);
                        let mut retry = self.start_loop(update, Value::Int(0), Value::Int(1), vec![old]);
                        let values = self.invoke_body(
                            retry.body,
                            &body,
                            vec![retry.state[0].clone(), value.clone()],
                            "updated",
                        )?;
                        let [value] = values.as_slice() else {
                            return Err(error("indexed reducer must return one logical value"));
                        };
                        let exchanged = self.atomic(
                            retry.body,
                            &dest,
                            key.clone(),
                            atomic,
                            vec![retry.state[0].clone(), value.clone()],
                        );
                        retry.step = Value::op("bool_to_u32", [exchanged.clone().field(1)]);
                        self.finish_loop(&retry, retry.body, vec![exchanged.field(0)]);
                        end = retry.done;
                    } else {
                        self.atomic(update, &dest, key.clone(), atomic, vec![value.clone()]);
                    }
                } else {
                    let old = self.load(update, dest.clone(), key.clone(), "old");
                    let result = self.invoke_body(update, &body, vec![old, value.clone()], "updated")?;
                    let [value] = result.as_slice() else {
                        return Err(error("indexed reducer must return one logical value"));
                    };
                    self.store(update, &dest, key.clone(), value.clone());
                }
                self.jump(end, next, vec![]);
                self.finish_loop(&loop_, next, vec![]);
                Ok((loop_.done, dest))
            }
            OperationKind::BucketScatter {
                destination,
                body,
                inputs,
                shape,
            } => {
                let shape = self.data.bucket_shapes[shape].clone();
                if inputs.len() != shape.input_dimensions.len() || shape.domain_rank == 0 {
                    return Err(error("invalid ranked input shape"));
                }
                let mut dimensions = vec![None; usize::from(shape.domain_rank)];
                for (input, axes) in inputs.iter().zip(&shape.input_dimensions) {
                    for (input_axis, &domain_axis) in axes.iter().enumerate() {
                        let Some(slot) = dimensions.get_mut(usize::from(domain_axis)) else {
                            return Err(error("invalid input dimension"));
                        };
                        *slot = Some(Value::op(
                            "dimension",
                            [array_value(input), Value::Int(input_axis as u32)],
                        ));
                    }
                }
                let dimensions: Vec<_> = dimensions
                    .into_iter()
                    .map(|d| {
                        let Some(d) = d else {
                            return Err(error("unbound ranked domain dimension"));
                        };
                        Ok(d)
                    })
                    .collect::<Result<_, _>>()?;
                let n = dimensions.iter().cloned().fold(Value::Int(1), |a, b| Value::op("mul", [a, b]));
                let dest = Value::Source(destination.value);
                let buckets = Value::op("length", [dest.clone()]);
                let capacity = Value::op("dimension", [dest.clone(), Value::Int(1)]);
                let counts = self.slot(op, "counts", 0, local);
                let overflow = self.slot(op, "overflow", 0, local);
                let loop_ = if atomic.is_some() {
                    self.invocations(entry, n)
                } else {
                    self.store(entry, &overflow, Value::Int(0), Value::Int(0));
                    let clear = self.start_loop(entry, Value::Int(0), buckets.clone(), vec![]);
                    self.store(clear.body, &counts, clear.index.clone(), Value::Int(0));
                    self.finish_loop(&clear, clear.body, vec![]);
                    self.start_loop(clear.done, Value::Int(0), n, vec![])
                };
                let mut coordinates = vec![Value::Int(0); dimensions.len()];
                let mut remainder = loop_.index.clone();
                for i in (0..dimensions.len()).rev() {
                    coordinates[i] = Value::op("rem", [remainder.clone(), dimensions[i].clone()]);
                    remainder = Value::op("div", [remainder, dimensions[i].clone()]);
                }
                let mut args = vec![];
                for (input, axes) in inputs.iter().zip(&shape.input_dimensions) {
                    let mut value = array_value(input);
                    for &axis in axes {
                        value =
                            self.load(loop_.body, value, coordinates[usize::from(axis)].clone(), "item");
                    }
                    args.push(value);
                }
                let values = self.invoke_body(loop_.body, &body, args, "emission")?;
                let [emission] = values.as_slice() else {
                    return Err(error("bucket body must return an emission tuple"));
                };
                let active = emission.clone().field(0);
                let key = emission.clone().field(1);
                let check = self.block(owner, vec![]);
                let reserve = self.block(owner, vec![]);
                let full = self.block(owner, vec![]);
                let write = self.block(owner, vec![]);
                let next = self.block(owner, vec![]);
                let invalid = self.block(owner, vec![]);
                let reserved = self.block(owner, vec![]);
                let checked = self.block(owner, vec![]);
                self.branch(loop_.body, active, check, next, Some(next));
                self.branch(
                    check,
                    in_bounds(key.clone(), buckets),
                    reserve,
                    invalid,
                    Some(checked),
                );
                let slot = if atomic.is_some() {
                    self.atomic(reserve, &counts, key.clone(), AtomicOp::Add, vec![Value::Int(1)])
                } else {
                    let slot = self.load(reserve, counts.clone(), key.clone(), "slot");
                    self.store(
                        reserve,
                        &counts,
                        key.clone(),
                        Value::op("add", [slot.clone(), Value::Int(1)]),
                    );
                    slot
                };
                self.branch(
                    reserve,
                    Value::op("lt", [slot.clone(), capacity]),
                    write,
                    full,
                    Some(reserved),
                );
                let row = Value::op("index", [dest.clone(), key]);
                self.emit(
                    write,
                    Instruction::Store {
                        buffer: row,
                        index: slot,
                        value: emission.clone().field(2),
                    },
                );
                self.jump(write, reserved, vec![]);
                if atomic.is_some() {
                    self.atomic(
                        full,
                        &overflow,
                        Value::Int(0),
                        AtomicOp::Exchange,
                        vec![Value::Int(1)],
                    );
                } else {
                    self.store(full, &overflow, Value::Int(0), Value::Int(1));
                }
                self.jump(full, reserved, vec![]);
                self.jump(reserved, checked, vec![]);
                if atomic.is_some() {
                    self.atomic(
                        invalid,
                        &overflow,
                        Value::Int(0),
                        AtomicOp::Exchange,
                        vec![Value::Int(1)],
                    );
                } else {
                    self.store(invalid, &overflow, Value::Int(0), Value::Int(1));
                }
                self.jump(invalid, checked, vec![]);
                self.jump(checked, next, vec![]);
                self.finish_loop(&loop_, next, vec![]);
                Ok((loop_.done, Value::Tuple(vec![dest, counts, singleton(overflow)])))
            }
            _ => Err(error("expected indexed memory operation")),
        }
    }

    fn atomic(
        &mut self,
        block: BlockId,
        buffer: &Value,
        index: Value,
        op: AtomicOp,
        values: Vec<Value>,
    ) -> Value {
        let name = format!(
            "atomic{}_{}",
            block.as_u32(),
            self.data.state.bodies[self.data.state.blocks[block].body].instructions.len()
        );
        self.emit(
            block,
            Instruction::Atomic {
                result: name.clone(),
                buffer: buffer.clone(),
                index,
                op,
                values,
            },
        );
        Value::Local(name)
    }
}

fn in_bounds(index: Value, bound: Value) -> Value {
    Value::op(
        "and",
        [
            Value::op("ge", [index.clone(), Value::Int(0)]),
            Value::op("lt", [index, bound]),
        ],
    )
}
