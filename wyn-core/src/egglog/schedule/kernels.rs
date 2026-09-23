//! Instantiate the selected GPU recipe with ordinary calls, loads, stores and
//! control edges. No source SOAC survives as an executable instruction.
use super::super::data::Array;
use super::super::visit::Operand;
use super::{
    array_value, error, length, Control, FunctionKind, Instruction, OptimizeError, Planner, Recipe, Value,
    WIDTH,
};
use crate::egglog::data::{BlockId, DispatchId, ExprId, OperationId, OperationKind};
use crate::ssa::types::AtomicOp;
use std::collections::{BTreeMap, BTreeSet};

mod filter;
mod indexed;
mod screma;

struct Loop {
    continuing: BlockId,
    body: BlockId,
    done: BlockId,
    index: Value,
    state: Vec<Value>,
    step: Value,
}

impl Planner<'_> {
    pub(super) fn output_copies(&mut self) -> Result<(), OptimizeError> {
        let outputs: Vec<_> = self
            .data
            .state
            .outputs
            .iter()
            .filter(|(_, o)| o.copy)
            .map(|(&id, o)| (id, o.clone()))
            .collect();
        let roots: Vec<_> = self
            .data
            .state
            .blocks
            .iter()
            .filter_map(|(&id, b)| match b.interface.as_ref().map(|f| &f.kind) {
                Some(FunctionKind::Entry(entry))
                    if self.data.entries[*entry].declaration.entry_kind
                        == crate::interface::EntryKind::Compute =>
                {
                    Some((id, *entry))
                }
                _ => None,
            })
            .collect();
        let mut copies = BTreeMap::<_, Vec<_>>::new();
        for (id, output) in outputs {
            let root = match self.resources.output_writers.get(&id) {
                Some(&root) => root,
                None => {
                    let Some(&(root, _)) = roots.iter().find(|(_, entry)| *entry == output.entry) else {
                        continue;
                    };
                    root
                }
            };
            copies.entry(root).or_default().push(output);
        }
        for (root, outputs) in copies {
            let returns: Vec<_> = self
                .data
                .state
                .blocks
                .iter()
                .filter_map(|(&id, b)| {
                    (b.function == root && matches!(b.exit, super::Exit::Return(_))).then_some(id)
                })
                .collect();
            for mut end in returns {
                let super::Exit::Return(original) = self.data.state.blocks[end].exit else {
                    unreachable!()
                };
                let values = self.data.state.bodies[original].results.clone();
                // EpilogueValue only exposes results already available to lane
                // zero. Publish attached outputs once, even in a workgroup kernel.
                let merge = if matches!(
                    self.data.state.blocks[root].interface.as_ref().map(|f| &f.kind),
                    Some(FunctionKind::Kernel(size)) if *size != [1, 1, 1]
                ) {
                    let write = self.block(root, vec![]);
                    let done = self.block(root, vec![]);
                    self.branch(
                        end,
                        Value::op("eq", [Value::op("local_id", []), Value::Int(0)]),
                        write,
                        done,
                        Some(done),
                    );
                    end = write;
                    Some(done)
                } else {
                    None
                };
                for output in &outputs {
                    let Some(buffer) = output.buffer else {
                        return Err(error("copy output has no backing"));
                    };
                    self.emit(end, Instruction::Allocate(buffer));
                    let source = Value::Source(output.expression);
                    if crate::types::TypeExt::is_array(crate::types::strip_existentials(
                        &self.data.types[self.data.expressions[output.expression].ty].ty,
                    )) {
                        let loop_ = self.start_loop(
                            end,
                            Value::Int(0),
                            Value::op("length", [source.clone()]),
                            vec![],
                        );
                        let value = self.load(loop_.body, source, loop_.index.clone(), "output");
                        self.store(loop_.body, &Value::Buffer(buffer), loop_.index.clone(), value);
                        self.finish_loop(&loop_, loop_.body, vec![]);
                        end = loop_.done;
                    } else {
                        self.store(end, &Value::Buffer(buffer), Value::Int(0), source);
                    }
                }
                if let Some(done) = merge {
                    self.jump(end, done, vec![]);
                    end = done;
                }
                self.returns(end, values);
            }
        }
        Ok(())
    }

    pub(super) fn scalar_dispatch(
        &mut self,
        op: OperationId,
        host: BlockId,
    ) -> Result<BlockId, OptimizeError> {
        let kernel = self.kernel(op, "scalar");
        let members = self.data.state.execution.groups.get(&op).cloned().unwrap_or_else(|| vec![op]);
        let mut end = kernel;
        let mut results = vec![];
        for member in members {
            self.emit_placements(member, end);
            end = self.operation(member, end, true)?;
            let Some(&result) = self.operation_values.get(&member) else {
                return Err(error("missing scalar result expression"));
            };
            self.allocate_slots(member, host, false);
            let destination = self.slot(member, "scalar", 0, false);
            self.store(end, &destination, Value::Int(0), Value::Source(result));
            if !matches!(destination, Value::Discarded) {
                results.push((member, singleton(destination)));
            }
        }
        self.returns(end, vec![]);
        self.dispatch(op, host, kernel);
        for (member, value) in results {
            self.emit(host, Instruction::BindResult(member, value));
        }
        Ok(host)
    }

    pub(super) fn parallel(&mut self, op: OperationId, host: BlockId) -> Result<BlockId, OptimizeError> {
        let Some(recipe) = self.resources.recipes.get(&op).copied() else {
            return Err(error("missing execution recipe"));
        };
        self.allocate_slots(op, host, false);
        let value = match recipe {
            Recipe::Elements | Recipe::Totals | Recipe::Prefixes => self.parallel_screma(op, host)?,
            Recipe::Compact => self.parallel_filter(op, host)?,
            Recipe::Buckets => {
                let clear = self.kernel(op, "clear");
                let OperationKind::BucketScatter { ref destination, .. } = self.data.operations[op].kind
                else {
                    return Err(error("bucket recipe requires a bucket operation"));
                };
                let n = Value::op("length", [Value::Source(destination.value)]);
                let counts = self.slot(op, "counts", 0, false);
                let overflow = self.slot(op, "overflow", 0, false);
                let zero = self.invocations(clear, Value::Int(1));
                self.store(zero.body, &overflow, Value::Int(0), Value::Int(0));
                self.finish_loop(&zero, zero.body, vec![]);
                let bins = self.invocations(zero.done, n);
                self.store(bins.body, &counts, bins.index.clone(), Value::Int(0));
                self.finish_loop(&bins, bins.body, vec![]);
                self.returns(bins.done, vec![]);
                self.dispatch(op, host, clear);
                let kernel = self.kernel(op, "buckets");
                let (end, value) = self.indexed(op, kernel, false, Some(AtomicOp::Add))?;
                self.returns(end, vec![]);
                self.dispatch(op, host, kernel);
                value
            }
            Recipe::Atomic(update) => {
                let kernel = self.kernel(op, "atomic");
                let (end, value) = self.indexed(op, kernel, false, Some(update))?;
                self.returns(end, vec![]);
                self.dispatch(op, host, kernel);
                value
            }
            Recipe::Serial => {
                let kernel = self.kernel(op, "ordered");
                let (end, value) = self.serial_body(op, kernel, false)?;
                self.returns(end, vec![]);
                self.dispatch(op, host, kernel);
                value
            }
        };
        self.emit(host, Instruction::BindResult(op, value));
        Ok(host)
    }

    pub(super) fn serial(&mut self, op: OperationId, block: BlockId) -> Result<BlockId, OptimizeError> {
        // A nested SOAC is local work in the calling invocation. Isolate its
        // counter/accumulator bindings in a normal device function, never a launch.
        let captures = self.captures(op);
        let names = (0..captures.len()).map(|i| format!("c{i}")).collect();
        let function = self.function("local".into(), FunctionKind::Device, names, 1);
        self.bind_captures(function, &captures);
        self.allocate_slots(op, function, true);
        let (end, value) = self.serial_body(op, function, true)?;
        self.returns(end, vec![value]);
        let name = format!("local{}", op.as_u32());
        self.emit(
            block,
            Instruction::Call {
                function,
                arguments: captures.into_iter().map(Value::Source).collect(),
                results: vec![name.clone()],
            },
        );
        self.emit(block, Instruction::BindResult(op, Value::Local(name)));
        Ok(block)
    }

    fn serial_body(
        &mut self,
        op: OperationId,
        block: BlockId,
        local: bool,
    ) -> Result<(BlockId, Value), OptimizeError> {
        match self.data.operations[op].kind.clone() {
            OperationKind::Screma { form, inputs, .. } => {
                self.serial_screma(op, block, local, &form, &inputs)
            }
            OperationKind::Filter {
                map,
                body,
                post,
                inputs,
                ..
            } => self.serial_filter(op, block, local, &map, &body, &post, &inputs),
            OperationKind::Scatter { .. }
            | OperationKind::BucketScatter { .. }
            | OperationKind::ReduceByIndex { .. } => self.indexed(op, block, local, None),
            _ => Err(error("expected an array operation")),
        }
    }

    fn kernel(&self, op: OperationId, name: &str) -> BlockId {
        let dispatch = self.resources.stages[&(op, name.into())];
        self.data.state.dispatches[dispatch].kernel
    }

    fn bind_captures(&mut self, entry: BlockId, captures: &[ExprId]) {
        for (i, &id) in captures.iter().enumerate() {
            self.emit(
                entry,
                Instruction::BindExpression(id, Value::Local(format!("c{i}"))),
            );
        }
    }

    fn invocations(&mut self, kernel: BlockId, count: Value) -> Loop {
        // Grid-stride iteration keeps dynamic domains valid even beyond the
        // portable 65,535-workgroup dispatch limit. Disjoint indices stay owned
        // by one invocation throughout a dispatch.
        let mut loop_ = self.start_loop(kernel, Value::op("global_id", [Value::Int(0)]), count, vec![]);
        loop_.step = Value::op("global_size", [Value::Int(0)]);
        self.returns(loop_.done, vec![]);
        loop_
    }

    fn start_loop(&mut self, from: BlockId, start: Value, bound: Value, state: Vec<Value>) -> Loop {
        let owner = self.data.state.blocks[from].function;
        let prefix = format!("loop{}", self.data.state.blocks.len());
        let index = Value::Local(format!("{prefix}_i"));
        let mut names = vec![format!("{prefix}_i")];
        names.extend((0..state.len()).map(|i| format!("{prefix}_s{i}")));
        let values: Vec<_> = names.iter().skip(1).cloned().map(Value::Local).collect();
        let next_names: Vec<_> = names.iter().map(|n| format!("{n}_next")).collect();
        let header = self.block(owner, names);
        let test = self.block(owner, vec![]);
        let continuing = self.block(owner, next_names.clone());
        self.jump(
            continuing,
            header,
            next_names.into_iter().map(Value::Local).collect(),
        );
        self.jump(header, test, vec![]);
        let body = self.block(owner, vec![]);
        let done = self.block(owner, vec![]);
        self.data.state.blocks[header].control = Some(Control::Loop {
            merge: done,
            continuing,
        });
        self.jump(from, header, std::iter::once(start).chain(state).collect());
        self.branch(test, Value::op("lt", [index.clone(), bound]), body, done, None);
        Loop {
            continuing,
            body,
            done,
            index,
            state: values,
            step: Value::Int(1),
        }
    }

    fn finish_loop(&mut self, loop_: &Loop, end: BlockId, state: Vec<Value>) {
        let next = Value::op("add", [loop_.index.clone(), loop_.step.clone()]);
        self.jump(
            end,
            loop_.continuing,
            std::iter::once(next).chain(state).collect(),
        );
    }

    fn allocate_slots(&mut self, op: OperationId, block: BlockId, local: bool) {
        let body = self.data.state.blocks[block].body;
        let instructions = &mut self.data.state.bodies[body].instructions;
        if local {
            instructions.extend(
                self.resources
                    .local_slots
                    .range((op, String::new(), 0)..)
                    .take_while(|(key, _)| key.0 == op)
                    .map(|(_, &buffer)| Instruction::Allocate(buffer)),
            );
        } else {
            instructions.extend(
                self.resources
                    .slots
                    .range((op, String::new(), 0)..)
                    .take_while(|(key, _)| key.0 == op)
                    .filter_map(|(_, value)| match value {
                        Value::Buffer(buffer) => Some(Instruction::Allocate(*buffer)),
                        _ => None,
                    }),
            );
        }
    }

    fn slot(&self, op: OperationId, name: &str, index: usize, local: bool) -> Value {
        let key = (op, name.into(), index as u32);
        if local {
            Value::Buffer(self.resources.local_slots[&key])
        } else {
            self.resources.slots[&key].clone()
        }
    }

    fn load(&mut self, block: BlockId, buffer: Value, index: Value, prefix: &str) -> Value {
        let name = format!(
            "{prefix}_{}_{}",
            block.as_u32(),
            self.data.state.bodies[self.data.state.blocks[block].body].instructions.len()
        );
        self.emit(
            block,
            Instruction::Load {
                result: name.clone(),
                buffer,
                index,
            },
        );
        Value::Local(name)
    }

    fn store(&mut self, block: BlockId, destination: &Value, index: Value, value: Value) {
        if matches!(destination, Value::Discarded) {
            return;
        }
        self.emit(
            block,
            Instruction::Store {
                buffer: destination.clone(),
                index,
                value,
            },
        );
    }

    fn read_inputs(&mut self, block: BlockId, inputs: &[Array], index: Value) -> Vec<Value> {
        inputs.iter().map(|array| self.load(block, array_value(array), index.clone(), "element")).collect()
    }

    fn dispatch(&mut self, op: OperationId, host: BlockId, kernel: BlockId) -> DispatchId {
        let Some(interface) = &self.data.state.blocks[kernel].interface else {
            unreachable!("dispatch target {kernel:?} has no kernel interface");
        };
        let name = &interface.name;
        let Some(dispatch) = self.resources.stages.remove(&(op, name.clone())) else {
            unreachable!("kernel {name} for {op:?} has no planned dispatch");
        };
        self.emit(host, Instruction::Dispatch(dispatch));
        dispatch
    }

    fn captures(&self, op: OperationId) -> Vec<ExprId> {
        let mut captures = BTreeSet::new();
        self.data.operations[op].kind.for_each_operand(&mut |operand| {
            if let Operand::Value(_, e) = operand {
                captures.insert(e);
            }
        });
        captures.into_iter().collect()
    }
}

fn chunks(n: Value) -> Value {
    Value::op("ceil_div", [n, Value::Int(WIDTH)])
}
fn singleton(buffer: Value) -> Value {
    if matches!(buffer, Value::Discarded) {
        return buffer;
    }
    Value::op("index", [buffer, Value::Int(0)])
}
