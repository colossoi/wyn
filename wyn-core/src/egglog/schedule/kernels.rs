//! Instantiate the selected GPU recipe with ordinary calls, loads, stores and
//! control edges. No source SOAC survives as an executable instruction.
use super::super::data::Array;
use super::super::visit::Operand;
use super::{
    array_value, error, length, Control, FunctionKind, Instruction, OptimizeError, Planner, Recipe,
    Storage, Value, WIDTH,
};
use crate::egglog::data::{BlockId, BufferId, DispatchId, ExprId, OperationId, OperationKind};
use std::collections::BTreeSet;

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
        let outputs: Vec<_> = self.data.state.outputs.values().filter(|o| o.copy).cloned().collect();
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
        for (root, entry) in roots {
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
                for output in outputs.iter().filter(|o| o.entry == entry) {
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
                        self.store(loop_.body, buffer, loop_.index.clone(), value);
                        self.finish_loop(&loop_, loop_.body, vec![]);
                        end = loop_.done;
                    } else {
                        self.store(end, buffer, Value::Int(0), source);
                    }
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
        let end = self.operation(op, kernel, true)?;
        let Some(result) = self.operation_values.get(&op) else {
            return Err(error("missing scalar result expression"));
        };
        let result = *result;
        let buffer = self.resources.slots[&(op, "scalar".into(), 0)];
        if self.data.state.buffers[buffer].storage != Storage::Discarded {
            self.emit(host, Instruction::Allocate(buffer));
        }
        self.store(end, buffer, Value::Int(0), Value::Source(result));
        self.returns(end, vec![]);
        self.dispatch(op, host, kernel);
        self.emit(
            host,
            Instruction::BindResult(op, Value::op("index", [Value::Buffer(buffer), Value::Int(0)])),
        );
        Ok(host)
    }

    pub(super) fn parallel(&mut self, op: OperationId, host: BlockId) -> Result<BlockId, OptimizeError> {
        let previous = self.current_operation.replace(op);
        let counts = std::mem::take(&mut self.allocation_counts);
        let Some(recipe) = self.resources.recipes.get(&op).copied() else {
            return Err(error("missing execution recipe"));
        };
        let (value, _) = match recipe {
            Recipe::Elements | Recipe::Totals | Recipe::Prefixes => self.parallel_screma(op, host)?,
            Recipe::Compact => self.parallel_filter(op, host)?,
            Recipe::Serial => {
                let kernel = self.kernel(op, "ordered");
                let (end, value, outputs) = self.serial_body(op, kernel, host, Storage::Device)?;
                self.returns(end, vec![]);
                self.dispatch(op, host, kernel);
                (value, outputs)
            }
        };
        self.current_operation = previous;
        self.allocation_counts = counts;
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
        let previous = self.current_operation.replace(op);
        let counts = std::mem::take(&mut self.allocation_counts);
        let (end, value, _) = self.serial_body(op, function, function, Storage::Function)?;
        self.current_operation = previous;
        self.allocation_counts = counts;
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
        allocate: BlockId,
        storage: Storage,
    ) -> Result<(BlockId, Value, Vec<BufferId>), OptimizeError> {
        match self.data.operations[op].kind.clone() {
            OperationKind::Screma { form, inputs, .. } => {
                self.serial_screma(block, allocate, storage, &form, &inputs)
            }
            OperationKind::Filter {
                map, body, inputs, ..
            } => self.serial_filter(block, allocate, storage, &map, &body, &inputs),
            OperationKind::Scatter { .. }
            | OperationKind::BucketScatter { .. }
            | OperationKind::ReduceByIndex { .. } => self.serial_indexed(op, block, allocate, storage),
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

    fn allocate(&mut self, host: BlockId, name: &str, storage: Storage) -> BufferId {
        let Some(op) = self.current_operation else {
            unreachable!("allocation {name} in {host:?} has no source operation");
        };
        let index = self.allocation_counts.entry(name.into()).or_default();
        let slots =
            if storage == Storage::Function { &self.resources.local_slots } else { &self.resources.slots };
        let buffer = slots[&(op, name.into(), *index)];
        *index += 1;
        if self.data.state.buffers[buffer].storage != Storage::Discarded {
            self.emit(host, Instruction::Allocate(buffer));
        }
        buffer
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

    fn store(&mut self, block: BlockId, buffer: BufferId, index: Value, value: Value) {
        if self.data.state.buffers[buffer].storage == Storage::Discarded {
            return;
        }
        self.emit(
            block,
            Instruction::Store {
                buffer: Value::Buffer(buffer),
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
fn singleton(buffer: BufferId) -> Value {
    Value::op("index", [Value::Buffer(buffer), Value::Int(0)])
}
