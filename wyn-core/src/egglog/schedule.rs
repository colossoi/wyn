//! Parallel dispatch planning and lowering to blocks with opaque scalar bodies.
use super::blocks::{
    BlockData, BodyData, Control, Edge, Exit, Function, FunctionKind, Instruction, Storage, Value,
};
use super::data::intern_type;
use super::planning::{abi, facts, outputs, read, Readout, Recipe, KEYS, RULES, RUN};
use super::scalar::placement_index;
use super::{OptimizeError, Placed, PlacementSite, Program, Scheduled};
use crate::egglog::data::{
    Array, BlockId, BodyId, DefinitionId, ExprId, ExprKind, LoopKind, OperationId, OperationKind, RegionId,
    SoacBody,
};
use crate::egglog::dependencies::analyze;
use crate::egglog::timing::{span, time};
use crate::interface::EntryKind;
use crate::types::{Type, TypeName};
use crate::PipelineTopologyPolicy;
use egglog_engine::EGraph;
use std::collections::BTreeMap;

mod kernels;
mod publication;
mod validation;

const WIDTH: u32 = 64;

/// Derive a relational GPU plan, then generate blocks with opaque scalar bodies.
/// Recipes, resources, domains and dispatch order come from that plan.
/// No executable block contains a SOAC operation.
/// Requires completed scalar placement.
pub fn schedule(
    program: Program<Placed>,
    topology: PipelineTopologyPolicy,
) -> Result<Program<Scheduled>, OptimizeError> {
    let _timing = span("scheduling");
    let mut converted = Program {
        ir: program.ir,
        state: Scheduled {
            placements: program.state.placements,
            ..Scheduled::default()
        },
    };
    let summary = time("analyze dependencies", || analyze(&converted));
    let schedules = time("validate dependency order", || summary.schedules(&converted))?;
    let entries: Vec<_> = converted.entries.iter().map(|(&id, e)| (id, e.definition)).collect();
    let count_type = intern_type(&mut converted.ir, Type::Constructed(TypeName::UInt(32), vec![]));
    converted.state.abi.inputs = super::abi::inputs(
        &converted.entries,
        &converted.entry_params,
        &converted.input_bounds,
        &converted.symbols,
        &converted.regions,
        &converted.definitions,
        &converted.types,
        &converted.parameters,
    )?;
    let mut host_inputs = vec![];
    let mut graph = EGraph::default();
    time("load planning schema", || {
        graph.parse_and_run_program(Some("ids.egg".into()), include_str!("ids.egg"))?;
        graph.parse_and_run_program(None, KEYS)?;
        graph.parse_and_run_program(Some("planning-rules.egg".into()), RULES)
    })?;
    time("read planning facts", || {
        graph.update(|mut sink| {
            outputs(&mut converted, &mut sink)?;
            facts(&converted, &summary, count_type, topology, &mut sink)?;
            abi::facts(
                &converted.state.abi.inputs,
                &converted.state.outputs,
                &converted.entries,
                &converted.types,
                &mut sink,
                &mut host_inputs,
            )
        })
    })?;
    time("derive stages, storage and dispatch order", || {
        graph.parse_and_run_program(None, RUN)
    })?;
    let resources = read(&graph, &mut converted)?;
    let plan = span("build blocks and dispatches");
    let operation_values = converted
        .ir
        .expressions
        .iter()
        .filter_map(
            |(&id, e)| {
                if let ExprKind::OperationResult(op) = e.kind {
                    Some((op, id))
                } else {
                    None
                }
            },
        )
        .collect();
    let mut planner = Planner {
        placements: placement_index(&converted.ir, &converted.state.placements),
        data: &mut converted,
        schedules,
        functions: BTreeMap::new(),
        resources,
        operation_values,
    };
    let mut entry_roots = BTreeMap::new();
    for (entry, definition) in entries {
        let device = topology == PipelineTopologyPolicy::AuthoredOnly
            || planner.data.entries[entry].declaration.entry_kind != EntryKind::Compute;
        let root = planner.definition(definition, device)?;
        entry_roots.insert(entry, root);
        if let Some(interface) = &mut planner.data.state.blocks[root].interface {
            interface.kind = FunctionKind::Entry(entry);
        }
    }
    drop(plan);
    if !planner.resources.stages.is_empty() {
        return Err(error("planned dispatches were not lowered"));
    }
    let dispatch_order = time("validate blocks", || validation::validate(planner.data, topology))?;
    let roots = time("read shader interfaces", || {
        abi::read(
            &graph,
            planner.data,
            &planner.resources.buffers,
            &planner.resources.launches,
            &entry_roots,
            &host_inputs,
        )
    })?;
    planner.output_copies()?;
    planner.data.state.physical_kernels = time("finalize physical kernel graph", || {
        publication::build_physical_kernel_graph(
            &graph,
            &roots,
            &dispatch_order,
            &mut planner.data.state.abi,
            &planner.data.state.dispatches,
            &planner.data.ir.entries,
            &planner.data.state.blocks,
        )
    })?;
    Ok(converted)
}

fn error(message: &str) -> OptimizeError {
    OptimizeError::Output(format!("GPU scaffold: {message}"))
}

struct Planner<'a> {
    placements: BTreeMap<PlacementSite, Vec<ExprId>>,
    data: &'a mut Program<Scheduled>,
    schedules: BTreeMap<RegionId, Vec<OperationId>>,
    functions: BTreeMap<(RegionId, bool), BlockId>,
    resources: Readout,
    operation_values: BTreeMap<OperationId, ExprId>,
}

impl Planner<'_> {
    fn body(&mut self, instructions: Vec<Instruction>, results: Vec<Value>) -> BodyId {
        self.data.state.bodies.alloc(BodyData {
            instructions,
            results,
        })
    }
    fn values(&mut self, values: Vec<Value>) -> BodyId {
        self.body(vec![], values)
    }
    fn block(&mut self, function: BlockId, parameters: Vec<String>) -> BlockId {
        let body = self.body(vec![], vec![]);
        let returns = self.values(vec![]);
        let id = self.data.state.blocks.alloc(BlockData {
            function,
            control: None,
            interface: None,
            parameters,
            body,
            exit: Exit::Return(returns),
        });
        if let Some(interface) = &mut self.data.state.blocks[function].interface {
            interface.blocks.push(id);
        }
        id
    }
    fn function(
        &mut self,
        name: String,
        kind: FunctionKind,
        parameters: Vec<String>,
        results: usize,
    ) -> BlockId {
        Function {
            name,
            kind,
            results,
            blocks: vec![],
        }
        .insert(
            parameters,
            &mut self.data.state.blocks,
            &mut self.data.state.bodies,
        )
    }
    fn emit(&mut self, block: BlockId, instruction: Instruction) {
        if matches!(
            self.data.state.blocks[self.data.state.blocks[block].function]
                .interface
                .as_ref()
                .map(|f| &f.kind),
            Some(FunctionKind::Host | FunctionKind::Entry(_))
        ) {
            if let Instruction::BindResult(op, value) = &instruction {
                if materialized(value) {
                    self.data.state.materialized.insert(*op, value.clone());
                }
            }
        }
        let body = self.data.state.blocks[block].body;
        self.data.state.bodies[body].instructions.push(instruction);
    }
    fn edge(&mut self, target: BlockId, arguments: Vec<Value>) -> Edge {
        Edge {
            target,
            arguments: self.values(arguments),
        }
    }
    fn jump(&mut self, from: BlockId, target: BlockId, arguments: Vec<Value>) {
        let edge = self.edge(target, arguments);
        self.data.state.blocks[from].exit = Exit::Jump(edge);
    }
    fn branch(
        &mut self,
        from: BlockId,
        condition: Value,
        yes: BlockId,
        no: BlockId,
        merge: Option<BlockId>,
    ) {
        self.data.state.blocks[from].control = merge.map(|merge| Control::Selection { merge });
        let condition = self.values(vec![condition]);
        let yes = self.edge(yes, vec![]);
        let no = self.edge(no, vec![]);
        self.data.state.blocks[from].exit = Exit::Branch { condition, yes, no };
    }
    fn returns(&mut self, block: BlockId, values: Vec<Value>) {
        let values = self.values(values);
        self.data.state.blocks[block].exit = Exit::Return(values);
    }
    fn definition(&mut self, id: DefinitionId, device: bool) -> Result<BlockId, OptimizeError> {
        let definition = self.data.definitions[id].clone();
        let entry = self.region(definition.body, device)?;
        let name = self.data.symbols[definition.symbol].name.clone();
        if let Some(interface) = &mut self.data.state.blocks[entry].interface {
            interface.name = name;
        }
        Ok(entry)
    }
    fn region(&mut self, region: RegionId, device: bool) -> Result<BlockId, OptimizeError> {
        if let Some(&entry) = self.functions.get(&(region, device)) {
            return Ok(entry);
        }
        let source = self.data.regions[region].clone();
        let names: Vec<_> = (0..source.parameters.len()).map(|i| format!("p{i}")).collect();
        let entry = self.function(
            format!("r{}", region.as_u32()),
            if device { FunctionKind::Device } else { FunctionKind::Host },
            names.clone(),
            source.results.len(),
        );
        self.functions.insert((region, device), entry);
        for (parameter, name) in source.parameters.into_iter().zip(names) {
            self.emit(entry, Instruction::BindParameter(parameter, Value::Local(name)));
        }
        let end = self.region_into(region, entry, device)?;
        self.returns(
            end,
            self.data.regions[region].results.iter().copied().map(Value::Source).collect(),
        );
        Ok(entry)
    }
    fn region_into(
        &mut self,
        region: RegionId,
        mut block: BlockId,
        device: bool,
    ) -> Result<BlockId, OptimizeError> {
        for op in self.schedules.get(&region).cloned().unwrap_or_default() {
            for value in self.placements.get(&PlacementSite::Operation(op)).cloned().unwrap_or_default() {
                self.emit(block, Instruction::BindExpression(value, Value::Source(value)));
            }
            block = self.operation(op, block, device)?;
        }
        Ok(block)
    }
    fn result(&self, region: RegionId) -> Value {
        let values: Vec<_> = self.data.regions[region].results.iter().copied().map(Value::Source).collect();
        if values.len() == 1 {
            values[0].clone()
        } else {
            Value::Tuple(values)
        }
    }
    fn operation(
        &mut self,
        op: OperationId,
        block: BlockId,
        device: bool,
    ) -> Result<BlockId, OptimizeError> {
        if !device && self.resources.stages.contains_key(&(op, "scalar".into())) {
            return self.scalar_dispatch(op, block);
        }
        let kind = self.data.operations[op].kind.clone();
        let owner = self.data.state.blocks[block].function;
        match kind {
            OperationKind::If {
                condition,
                then_region,
                else_region,
            } => {
                let yes = self.block(owner, vec![]);
                let no = self.block(owner, vec![]);
                let selected = format!("selected{}", op.as_u32());
                let merge = self.block(owner, vec![selected.clone()]);
                self.branch(block, Value::Source(condition), yes, no, Some(merge));
                let yes_end = self.region_into(then_region, yes, device)?;
                let no_end = self.region_into(else_region, no, device)?;
                self.jump(yes_end, merge, vec![self.result(then_region)]);
                self.jump(no_end, merge, vec![self.result(else_region)]);
                self.emit(merge, Instruction::BindResult(op, Value::Local(selected)));
                Ok(merge)
            }
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            } => self.source_loop(op, block, init, header, &kind, body, device),
            OperationKind::Call { function, mut args } => {
                if let ExprKind::Lambda(region) = self.data.expressions[function].kind {
                    let target = self.region(region, device)?;
                    let names: Vec<_> = (0..self.data.regions[region].results.len())
                        .map(|i| format!("call{}_r{i}", op.as_u32()))
                        .collect();
                    self.emit(
                        block,
                        Instruction::Call {
                            function: target,
                            arguments: args.into_iter().map(Value::Source).collect(),
                            results: names.clone(),
                        },
                    );
                    let values: Vec<_> = names.into_iter().map(Value::Local).collect();
                    let value = if values.len() == 1 { values[0].clone() } else { Value::Tuple(values) };
                    self.emit(block, Instruction::BindResult(op, value));
                    return Ok(block);
                }
                let code = match &self.data.expressions[function].kind {
                    ExprKind::Global(symbol) => Some(*symbol),
                    ExprKind::Closure { code, captures, .. } => {
                        args.extend(captures);
                        Some(*code)
                    }
                    _ => None,
                };
                let definition = code.and_then(|symbol| {
                    self.data.definitions.iter().find_map(|(&id, def)| (def.symbol == symbol).then_some(id))
                });
                if let Some(definition) = definition {
                    let target = self.definition(definition, device)?;
                    self.emit(
                        block,
                        Instruction::Call {
                            function: target,
                            arguments: args.into_iter().map(Value::Source).collect(),
                            results: vec!["called".into()],
                        },
                    );
                    self.emit(block, Instruction::BindResult(op, Value::local("called")));
                } else {
                    self.emit(block, Instruction::Evaluate(op));
                }
                Ok(block)
            }
            OperationKind::EvalGlobal(symbol) => {
                let definition = self
                    .data
                    .definitions
                    .iter()
                    .find_map(|(&id, def)| (def.symbol == symbol).then_some(id));
                if let Some(definition) = definition {
                    let target = self.definition(definition, device)?;
                    self.emit(
                        block,
                        Instruction::Call {
                            function: target,
                            arguments: vec![],
                            results: vec!["called".into()],
                        },
                    );
                    self.emit(block, Instruction::BindResult(op, Value::local("called")));
                } else {
                    self.emit(block, Instruction::Evaluate(op));
                }
                Ok(block)
            }
            OperationKind::Index { .. } => {
                self.emit(block, Instruction::Evaluate(op));
                Ok(block)
            }
            OperationKind::Screma { .. }
            | OperationKind::Filter { .. }
            | OperationKind::Scatter { .. }
            | OperationKind::BucketScatter { .. }
            | OperationKind::ReduceByIndex { .. } => {
                if device {
                    self.serial(op, block)
                } else {
                    self.parallel(op, block)
                }
            }
        }
    }
    fn source_loop(
        &mut self,
        op: OperationId,
        from: BlockId,
        init: ExprId,
        header: RegionId,
        kind: &LoopKind,
        body: RegionId,
        device: bool,
    ) -> Result<BlockId, OptimizeError> {
        let owner = self.data.state.blocks[from].function;
        let acc_name = format!("acc{}", op.as_u32());
        let index_name = format!("index{}", op.as_u32());
        let acc = Value::Local(acc_name.clone());
        let index = Value::Local(index_name.clone());
        let loop_header = self.block(owner, vec![acc_name, index_name]);
        let test = self.block(owner, vec![]);
        let continuing = self.block(
            owner,
            vec![
                format!("next_acc{}", op.as_u32()),
                format!("next_index{}", op.as_u32()),
            ],
        );
        self.jump(
            continuing,
            loop_header,
            vec![
                Value::local(&format!("next_acc{}", op.as_u32())),
                Value::local(&format!("next_index{}", op.as_u32())),
            ],
        );
        self.jump(loop_header, test, vec![]);
        let iterate = self.block(owner, vec![]);
        let after = self.block(owner, vec![]);
        self.data.state.blocks[loop_header].control = Some(Control::Loop {
            merge: after,
            continuing,
        });
        self.jump(from, loop_header, vec![Value::Source(init), Value::Int(0)]);
        let parameters = self.data.regions[header].parameters.clone();
        if let Some(&parameter) = parameters.first() {
            self.emit(test, Instruction::BindParameter(parameter, acc.clone()));
        }
        let (test_end, condition) = match kind {
            LoopKind::While => {
                let end = self.region_into(header, test, device)?;
                (end, self.result(header))
            }
            LoopKind::For(array) => (
                test,
                Value::op(
                    "lt",
                    [index.clone(), Value::op("length", [Value::Source(*array)])],
                ),
            ),
            LoopKind::ForRange(bound) => (test, Value::op("lt", [index.clone(), Value::Source(*bound)])),
        };
        self.branch(test_end, condition, iterate, after, None);
        if let Some(&parameter) = parameters.get(1) {
            let value = match kind {
                LoopKind::For(array) => Value::op("index", [Value::Source(*array), index.clone()]),
                _ => index.clone(),
            };
            self.emit(iterate, Instruction::BindParameter(parameter, value));
        }
        let iterate = if matches!(kind, LoopKind::While) {
            iterate
        } else {
            self.region_into(header, iterate, device)?
        };
        let end = self.region_into(body, iterate, device)?;
        self.jump(
            end,
            continuing,
            vec![self.result(body), Value::op("add", [index, Value::Int(1)])],
        );
        self.emit(after, Instruction::BindResult(op, acc));
        Ok(after)
    }
    fn invoke_body(
        &mut self,
        block: BlockId,
        body: &SoacBody,
        arguments: Vec<Value>,
        prefix: &str,
    ) -> Result<Vec<Value>, OptimizeError> {
        if arguments.len() != signature(body).0 {
            return Err(error("opaque body argument arity mismatch"));
        }
        match body {
            SoacBody::Identity(_) => Ok(arguments),
            SoacBody::Route { indices, .. } => indices
                .iter()
                .map(|&i| {
                    let Some(value) = arguments.get(i) else {
                        return Err(error("body route is out of bounds"));
                    };
                    Ok(value.clone())
                })
                .collect(),
            SoacBody::Compose { first, then } => {
                let values = self.invoke_body(block, first, arguments, &format!("{prefix}_first"))?;
                self.invoke_body(block, then, values, prefix)
            }
            SoacBody::Parallel { left, right } => {
                let mut values =
                    self.invoke_body(block, left, arguments.clone(), &format!("{prefix}_left"))?;
                values.extend(self.invoke_body(block, right, arguments, &format!("{prefix}_right"))?);
                Ok(values)
            }
            SoacBody::Apply {
                region,
                captures,
                results,
                ..
            } => {
                let target = self.region(*region, true)?;
                let mut args = arguments;
                args.extend(captures.iter().copied().map(Value::Source));
                let names: Vec<_> = (0..results.len()).map(|i| format!("{prefix}{i}")).collect();
                self.emit(
                    block,
                    Instruction::Call {
                        function: target,
                        arguments: args,
                        results: names.clone(),
                    },
                );
                Ok(names.into_iter().map(Value::Local).collect())
            }
        }
    }
}

fn signature(body: &SoacBody) -> (usize, usize) {
    match body {
        SoacBody::Apply {
            parameters, results, ..
        } => (parameters.len(), results.len()),
        SoacBody::Identity(types) => (types.len(), types.len()),
        SoacBody::Route { parameters, indices } => (parameters.len(), indices.len()),
        SoacBody::Compose { first, then } => (signature(first).0, signature(then).1),
        SoacBody::Parallel { left, right } => (signature(left).0, signature(left).1 + signature(right).1),
    }
}

fn array_value(array: &Array) -> Value {
    Value::Array(array.clone())
}
fn length(inputs: &[Array]) -> Value {
    inputs.first().map(|a| Value::op("length", [array_value(a)])).unwrap_or(Value::Int(0))
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod schedule_tests;

fn materialized(value: &Value) -> bool {
    match value {
        Value::Buffer(_) | Value::Source(_) | Value::Discarded => true,
        Value::Field(value, _) => materialized(value),
        Value::Tuple(values) | Value::Primitive(_, values) => values.iter().any(materialized),
        _ => false,
    }
}
