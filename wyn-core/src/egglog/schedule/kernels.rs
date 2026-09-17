//! Instantiate the selected GPU recipe with ordinary calls, loads, stores and
//! control edges. No source SOAC survives as an executable instruction.
use super::super::data::{Array, ExprKind};
use super::super::visit::Operand;
use super::{
    array_value, error, length, DispatchData, FunctionKind, GridData, Instruction, OptimizeError, Planner,
    Recipe, Storage, Value, WIDTH,
};
use crate::egglog::data::{BlockId, BufferId, DispatchId, ExprId, OperationId, OperationKind, SoacBody};
use crate::types::Type;
use std::collections::BTreeSet;

mod filter;
mod indexed;
mod screma;

struct Loop {
    header: BlockId,
    body: BlockId,
    done: BlockId,
    index: Value,
    state: Vec<Value>,
    step: Value,
}

impl Planner<'_> {
    pub(super) fn scalar_dispatch(
        &mut self,
        op: OperationId,
        host: BlockId,
    ) -> Result<BlockId, OptimizeError> {
        let captures = self.scalar_captures(op);
        let kernel = self.kernel("scalar", &captures, 1);
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
        self.dispatch(op, host, kernel, captures);
        self.emit(
            host,
            Instruction::BindResult(op, Value::op("index", [Value::Buffer(buffer), Value::Int(0)])),
        );
        Ok(host)
    }

    pub(super) fn parallel(&mut self, op: OperationId, host: BlockId) -> Result<BlockId, OptimizeError> {
        let previous = self.current_operation.replace(op);
        let Some(recipe) = self.recipes.get(&op).copied() else {
            return Err(error("missing execution recipe"));
        };
        let (value, _) = match recipe {
            Recipe::Elements | Recipe::Totals | Recipe::Prefixes => self.parallel_screma(op, host)?,
            Recipe::Compact => self.parallel_filter(op, host)?,
            Recipe::Serial => {
                let captures = self.captures(op);
                let kernel = self.kernel("ordered", &captures, 1);
                let (end, value, outputs) = self.serial_body(op, kernel, host, Storage::Device)?;
                self.returns(end, vec![]);
                self.dispatch(op, host, kernel, captures);
                (value, outputs)
            }
        };
        self.current_operation = previous;
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
        let (end, value, _) = self.serial_body(op, function, function, Storage::Function)?;
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

    fn kernel(&mut self, name: &str, captures: &[ExprId], width: u32) -> BlockId {
        let names = (0..captures.len()).map(|i| format!("c{i}")).collect();
        let entry = self.function(name.into(), FunctionKind::Kernel([width, 1, 1]), names, 0);
        self.bind_captures(entry, captures);
        entry
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
        let header = self.block(owner, names);
        let body = self.block(owner, vec![]);
        let done = self.block(owner, vec![]);
        self.data.state.blocks[header].loop_exit = Some(done);
        self.jump(from, header, std::iter::once(start).chain(state).collect());
        self.branch(header, Value::op("lt", [index.clone(), bound]), body, done);
        Loop {
            header,
            body,
            done,
            index,
            state: values,
            step: Value::Int(1),
        }
    }

    fn finish_loop(&mut self, loop_: &Loop, end: BlockId, state: Vec<Value>) {
        let next = Value::op("add", [loop_.index.clone(), loop_.step.clone()]);
        self.jump(end, loop_.header, std::iter::once(next).chain(state).collect());
    }

    fn allocate(
        &mut self,
        host: BlockId,
        name: &str,
        n: Value,
        element: Type,
        storage: Storage,
    ) -> BufferId {
        let buffer = if storage == Storage::Device {
            let Some(op) = self.current_operation else {
                unreachable!("device allocation {name} in block {host:?} has no source operation");
            };
            let index = self.allocation_counts.entry((op, name.into())).or_default();
            let buffer = self.resources.slots[&(op, name.into(), *index)];
            *index += 1;
            buffer
        } else {
            self.buffer(name, n, element, storage)
        };
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

    fn dispatch(
        &mut self,
        op: OperationId,
        host: BlockId,
        kernel: BlockId,
        captures: Vec<ExprId>,
    ) -> DispatchId {
        let Some(interface) = &self.data.state.blocks[kernel].interface else {
            unreachable!("dispatch target {kernel:?} has no kernel interface");
        };
        let name = interface.name.clone();
        let stage = &self.resources.stages[&(op, name.clone())];
        let Some(groups) = stage.groups.clone() else {
            unreachable!("planned stage {name} for {op:?} has no dispatch domain");
        };
        let Some(owner) = stage.owner else {
            unreachable!("planned stage {name} for {op:?} has no entry owner");
        };
        let grid = self.data.state.grids.alloc(GridData {
            groups: [groups, Value::Int(1), Value::Int(1)],
        });
        let reads = stage.reads.clone();
        let writes = stage.writes.clone();
        let dispatch = self.data.state.dispatches.alloc(DispatchData {
            owner,
            kernel,
            grid,
            dependencies: BTreeSet::new(),
            reads,
            writes,
            captures,
        });
        self.launches.push_str(&format!(
            "(check (Phase (Stage {} \"{name}\") r))\n(Emitted (Stage {} \"{name}\") {})\n",
            op.egglog(),
            op.egglog(),
            dispatch.as_u32()
        ));
        self.emit(host, Instruction::Dispatch(dispatch));
        dispatch
    }

    fn captures(&self, op: OperationId) -> Vec<ExprId> {
        let mut result = BTreeSet::new();
        match &self.data.operations[op].kind {
            OperationKind::Screma { form, inputs, .. } => {
                for array in inputs {
                    array_captures(array, &mut result);
                }
                body_captures(&form.pre, &mut result);
                body_captures(&form.post, &mut result);
                for scan in &form.scans {
                    result.extend(&scan.neutral);
                    body_captures(&scan.operator, &mut result);
                }
                for reduction in &form.reductions {
                    result.extend(&reduction.neutral);
                    body_captures(&reduction.operator, &mut result);
                }
            }
            OperationKind::Filter {
                inputs, map, body, ..
            } => {
                for input in inputs {
                    array_captures(input, &mut result);
                }
                body_captures(map, &mut result);
                body_captures(body, &mut result);
            }
            OperationKind::Scatter {
                destination,
                body,
                inputs,
            }
            | OperationKind::BucketScatter {
                destination,
                body,
                inputs,
                ..
            } => {
                result.insert(destination.value);
                body_captures(body, &mut result);
                for array in inputs {
                    array_captures(array, &mut result);
                }
            }
            OperationKind::ReduceByIndex {
                destination,
                map,
                body,
                neutral,
                inputs,
            } => {
                result.extend([destination.value, *neutral]);
                body_captures(map, &mut result);
                body_captures(body, &mut result);
                for array in inputs {
                    array_captures(array, &mut result);
                }
            }
            _ => {}
        }
        result.into_iter().collect()
    }

    // Capture external leaves, not whole expressions: branches and partial
    // expressions must still execute inside the scalar invocation that owns them.
    fn scalar_captures(&self, op: OperationId) -> Vec<ExprId> {
        use ExprKind;
        use Operand;
        let mut operations = vec![op];
        let mut regions = BTreeSet::new();
        let mut pending_regions = self.data.operations[op].kind.structured_regions();
        let mut pending = vec![];
        while let Some(r) = pending_regions.pop() {
            if !regions.insert(r) {
                continue;
            }
            let region = &self.data.regions[r];
            pending.extend(&region.results);
            for &child in &region.members {
                operations.push(child);
                pending_regions.extend(self.data.operations[child].kind.structured_regions());
            }
        }
        for op in operations {
            self.data.operations[op].kind.for_each_operand(&mut |operand| {
                if let Operand::Value(_, e) = operand {
                    pending.push(e);
                }
            });
        }
        let mut seen = BTreeSet::new();
        let mut captures = BTreeSet::new();
        while let Some(e) = pending.pop() {
            if !seen.insert(e) {
                continue;
            }
            match &self.data.expressions[e].kind {
                ExprKind::Parameter(p) if !regions.contains(&self.data.parameters[*p].region) => {
                    captures.insert(e);
                }
                ExprKind::OperationResult(op) if !regions.contains(&self.data.operations[*op].region) => {
                    captures.insert(e);
                }
                kind => pending.extend(kind.children()),
            }
        }
        captures.into_iter().collect()
    }
}

fn array_captures(array: &Array, result: &mut BTreeSet<ExprId>) {
    match array {
        Array::Value(expr) => {
            result.insert(*expr);
        }
        Array::Zip(arrays) => {
            for array in arrays {
                array_captures(array, result);
            }
        }
        Array::Literal(values) => result.extend(values),
        Array::Range { start, len, step } => {
            result.extend([*start, *len]);
            result.extend(step);
        }
    }
}

fn body_captures(body: &SoacBody, result: &mut BTreeSet<ExprId>) {
    match body {
        SoacBody::Apply { captures, .. } => result.extend(captures),
        SoacBody::Compose { first, then } => {
            body_captures(first, result);
            body_captures(then, result);
        }
        SoacBody::Parallel { left, right } => {
            body_captures(left, result);
            body_captures(right, result);
        }
        SoacBody::Identity(_) | SoacBody::Route { .. } => {}
    }
}

fn chunks(n: Value) -> Value {
    Value::op(
        "max",
        [Value::Int(1), Value::op("ceil_div", [n, Value::Int(WIDTH)])],
    )
}
fn singleton(buffer: BufferId) -> Value {
    Value::op("index", [Value::Buffer(buffer), Value::Int(0)])
}
