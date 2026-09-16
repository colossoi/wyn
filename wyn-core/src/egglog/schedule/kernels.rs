//! Instantiate the selected GPU recipe with ordinary calls, loads, stores and
//! control edges. No source SOAC survives as an executable instruction.

use super::super::data::{Array, ExprKind};
use super::{
    array_value, error, length, BlockId, BufferId, DispatchData, DispatchId, ExprId, FunctionKind,
    GridData, Instruction, OperationId, OperationKind, OptimizeError, Planner, Recipe, SoacBody, Storage,
    Value, WIDTH,
};
use crate::ast::TypeName;
use crate::types::{self, TypeExt};
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
    pub(super) fn parallel(&mut self, op: OperationId, host: BlockId) -> Result<BlockId, OptimizeError> {
        let Some(recipe) = self.recipes.get(&op).copied() else {
            return Err(error("missing execution recipe"));
        };
        let (value, outputs) = match recipe {
            Recipe::Elements | Recipe::Totals | Recipe::Prefixes => self.parallel_screma(op, host)?,
            Recipe::Compact => self.parallel_filter(op, host)?,
            Recipe::Serial => {
                let captures = self.captures(op);
                let kernel = self.kernel("ordered", &captures, 1);
                let (end, value, outputs) = self.serial_body(op, kernel, host, Storage::Device)?;
                self.returns(end, vec![]);
                let writes = outputs.iter().copied().chain(self.destination_buffer(op)).collect();
                self.dispatch(op, host, kernel, Value::Int(1), captures, BTreeSet::new(), writes);
                (value, outputs)
            }
        };
        self.outputs.insert(op, outputs);
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
        let owner = self.data.blocks[from].function;
        let prefix = format!("loop{}", self.data.blocks.len());
        let index = Value::Local(format!("{prefix}_i"));
        let mut names = vec![format!("{prefix}_i")];
        names.extend((0..state.len()).map(|i| format!("{prefix}_s{i}")));
        let values: Vec<_> = names.iter().skip(1).cloned().map(Value::Local).collect();
        let header = self.block(owner, names);
        let body = self.block(owner, vec![]);
        let done = self.block(owner, vec![]);
        self.data.blocks[header].loop_exit = Some(done);
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
        element: types::Type,
        storage: Storage,
    ) -> BufferId {
        let buffer = self.buffer(name, n, element, storage);
        self.emit(host, Instruction::Allocate(buffer));
        buffer
    }

    fn load(&mut self, block: BlockId, buffer: Value, index: Value, prefix: &str) -> Value {
        let name = format!(
            "{prefix}_{}_{}",
            block.as_u32(),
            self.data.bodies[self.data.blocks[block].body].instructions.len()
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
        groups: Value,
        captures: Vec<ExprId>,
        mut reads: BTreeSet<BufferId>,
        writes: BTreeSet<BufferId>,
    ) -> DispatchId {
        // These dependencies are static sites within the same lexical scope.
        // Host CFG edges handle conditional paths, calls, and repeated launches.
        let mut dependencies = BTreeSet::new();
        self.summary.walk_dependencies(&self.data, op, |current| {
            if let Some(previous) = self.dispatches.get(&current).and_then(|ds| ds.last()) {
                dependencies.insert(*previous);
            }
            if let Some(buffers) = self.outputs.get(&current) {
                reads.extend(buffers);
            }
        });
        for &capture in &captures {
            if let Some(buffer) = self.external_buffer(capture) {
                reads.insert(buffer);
            }
        }
        let grid = self.data.grids.alloc(GridData {
            groups: [groups, Value::Int(1), Value::Int(1)],
        });
        let dispatch = self.data.dispatches.alloc(DispatchData {
            kernel,
            grid,
            dependencies,
            reads,
            writes,
            captures,
        });
        self.dispatches.entry(op).or_default().push(dispatch);
        self.emit(host, Instruction::Dispatch(dispatch));
        dispatch
    }

    fn external_buffer(&mut self, expr: ExprId) -> Option<BufferId> {
        // Reuse a materialized producer's resource identity where the source
        // value gives us an exact alias. Branch/call-selected views stay opaque.
        match &self.data.expressions[expr].kind {
            ExprKind::Coerce(inner) | ExprKind::Array(Array::Value(inner)) => {
                return self.external_buffer(*inner)
            }
            ExprKind::Project { tuple, index } => {
                let mut tuple = *tuple;
                while let ExprKind::Coerce(inner) = self.data.expressions[tuple].kind {
                    tuple = inner;
                }
                match &self.data.expressions[tuple].kind {
                    ExprKind::OperationResult(op) => {
                        let component = match &self.data.operations[*op].kind {
                            OperationKind::Screma { .. } => Some(*index),
                            OperationKind::Filter { .. } => Some(0),
                            OperationKind::BucketScatter { destination, .. } if *index == 0 => {
                                return self.external_buffer(destination.value)
                            }
                            OperationKind::BucketScatter { .. } => index.checked_sub(1),
                            _ => None,
                        };
                        if let Some(&buffer) =
                            component.and_then(|index| self.outputs.get(op).and_then(|ids| ids.get(index)))
                        {
                            return Some(buffer);
                        }
                    }
                    ExprKind::Tuple(fields) => {
                        if let Some(&field) = fields.get(*index) {
                            return self.external_buffer(field);
                        }
                    }
                    _ => {}
                }
            }
            ExprKind::OperationResult(op) => {
                if matches!(self.data.operations[*op].kind, OperationKind::Filter { .. }) {
                    if let Some(&buffer) = self.outputs.get(op).and_then(|ids| ids.first()) {
                        return Some(buffer);
                    }
                }
            }
            _ => {}
        }
        let ty = &self.data.types[self.data.expressions[expr].ty].ty;
        let element = logical_element(ty)?;
        if let Some(&buffer) = self.external_buffers.get(&expr) {
            return Some(buffer);
        }
        let buffer = self.buffer(
            "capture",
            Value::op("length", [Value::Source(expr)]),
            element,
            Storage::External(expr),
        );
        self.external_buffers.insert(expr, buffer);
        Some(buffer)
    }

    fn destination_buffer(&mut self, op: OperationId) -> Option<BufferId> {
        let expr = match &self.data.operations[op].kind {
            OperationKind::Scatter { destination, .. }
            | OperationKind::BucketScatter { destination, .. }
            | OperationKind::ReduceByIndex { destination, .. } => destination.value,
            _ => return None,
        };
        self.external_buffer(expr)
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
}

fn logical_element(ty: &types::Type) -> Option<types::Type> {
    if let Some(element) = ty.elem_type() {
        return Some(element.clone());
    }
    // TLC can retain an array of tuples as a tuple of component arrays.
    if let types::Type::Constructed(TypeName::Tuple(_), fields) = ty {
        if !fields.is_empty() {
            return Some(types::tuple(
                fields.iter().map(logical_element).collect::<Option<_>>()?,
            ));
        }
    }
    None
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
fn groups(n: Value) -> Value {
    Value::op("min", [Value::Int(65_535), chunks(n)])
}
fn singleton(buffer: BufferId) -> Value {
    Value::op("index", [Value::Buffer(buffer), Value::Int(0)])
}
