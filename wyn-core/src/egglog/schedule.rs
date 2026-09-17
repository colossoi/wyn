//! Parallel dispatch planning and lowering to blocks with opaque scalar bodies.
use super::blocks::{
    BlockData, BodyData, BufferData, DispatchData, Edge, Exit, Function, FunctionKind, GridData,
    Instruction, Storage, Value,
};
use super::data::intern_type;
use super::planning::{facts, outputs, read, read_dispatch_order, Readout, KEYS, RULES, RUN};
use super::scalar::placement_index;
use super::{OptimizeError, Placed, PlacementSite, Program, Scheduled};
use crate::egglog::data::{
    Array, BlockId, BodyId, BufferId, DefinitionId, ExprId, ExprKind, LoopKind, OperationId, OperationKind,
    RegionId, SoacBody,
};
use crate::egglog::dependencies::analyze;
use crate::egglog::parse_program;
use crate::egglog::timing::{span, time};
use crate::interface::EntryKind;
use crate::types::{Type, TypeName};
use egglog_engine::ast::{Literal, Parser};
use egglog_engine::{EGraph, Term};
use std::collections::BTreeMap;

mod kernels;
mod output;
mod validation;

const WIDTH: u32 = 64;

/// Derive a relational GPU plan, then generate blocks with opaque scalar bodies.
/// Recipes, resources, domains and dispatch order come from that plan.
/// No executable block contains a SOAC operation.
/// Requires completed scalar placement; the expression layer is retained alongside
/// block facts, linked by source-region provenance rather than scalar placement.
pub fn schedule(program: Program<Placed>) -> Result<Program<Scheduled>, OptimizeError> {
    let _timing = span("scheduling");
    let expressions = program.state.facts;
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
    let input_fields = outputs(&mut converted);
    let count_type = intern_type(&mut converted.ir, Type::Constructed(TypeName::UInt(32), vec![]));
    let facts = time("derive planning facts", || {
        facts(&converted, &summary, count_type, &input_fields)
    });
    let mut planning = parse_program("planning-rules.egg", RULES)?;
    planning.extend(parse_program("wyn-planning.egg", &facts)?);
    let mut graph = EGraph::default();
    time("load planning graph", || {
        graph.parse_and_run_program(Some("ids.egg".into()), include_str!("ids.egg"))?;
        graph.parse_and_run_program(None, KEYS)?;
        graph.run_program(planning.clone())
    })?;
    let run = parse_program("planning-run.egg", RUN)?;
    time("derive stages and storage", || graph.run_program(run.clone()))?;
    planning.extend(run);
    let recipes = time("read kernel recipes", || recipes(&graph))?;
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
        launches: resources.identities.clone(),
        resources,
        allocation_counts: BTreeMap::new(),
        current_operation: None,
        recipes,
        current_region: None,
        operation_values,
    };
    for (entry, definition) in entries {
        let device = planner.data.entries[entry].declaration.entry_kind != EntryKind::Compute;
        let root = planner.definition(definition, device)?;
        if let Some(interface) = &mut planner.data.state.blocks[root].interface {
            interface.kind = FunctionKind::Entry(entry);
        }
    }
    drop(plan);
    let mut launches = parse_program("wyn-launches.egg", &planner.launches)?;
    launches.extend(parse_program(
        "dispatch-run.egg",
        "(run-schedule (saturate (run dispatch)))",
    )?);
    time("dispatch rules", || graph.run_program(launches.clone()))?;
    planning.extend(launches);
    time("read dispatch order", || {
        read_dispatch_order(&graph, planner.data)
    })?;
    time("validate blocks", || validation::validate(planner.data))?;
    let _output = span("export block facts");
    converted.state.facts = expressions;
    converted.state.facts.extend(planning);
    converted.state.facts.extend(
        Parser::default()
            .get_program_from_string(Some("wyn-blocks.egg".into()), &output::program(&converted))
            .map_err(|e| error(&e.to_string()))?,
    );
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
    launches: String,
    resources: Readout,
    allocation_counts: BTreeMap<(OperationId, String), u32>,
    current_operation: Option<OperationId>,
    recipes: BTreeMap<OperationId, Recipe>,
    current_region: Option<RegionId>,
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
        self.data.state.blocks.alloc(BlockData {
            function,
            source_regions: self.current_region.into_iter().collect(),
            loop_exit: None,
            interface: None,
            parameters,
            body,
            exit: Exit::Return(returns),
        })
    }
    fn function(
        &mut self,
        name: String,
        kind: FunctionKind,
        parameters: Vec<String>,
        results: usize,
    ) -> BlockId {
        let id = self.data.state.blocks.alloc_id();
        let body = self.body(vec![], vec![]);
        let returns = self.values(vec![]);
        self.data.state.blocks.insert(
            id,
            BlockData {
                function: id,
                source_regions: self.current_region.into_iter().collect(),
                loop_exit: None,
                interface: Some(Function { name, kind, results }),
                parameters,
                body,
                exit: Exit::Return(returns),
            },
        );
        id
    }
    fn emit(&mut self, block: BlockId, instruction: Instruction) {
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
    fn branch(&mut self, from: BlockId, condition: Value, yes: BlockId, no: BlockId) {
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
        let previous_region = self.current_region.replace(region);
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
        self.current_region = previous_region;
        Ok(entry)
    }
    fn region_into(
        &mut self,
        region: RegionId,
        mut block: BlockId,
        device: bool,
    ) -> Result<BlockId, OptimizeError> {
        let previous_region = self.current_region.replace(region);
        self.data.state.blocks[block].source_regions.insert(region);
        for op in self.schedules.get(&region).cloned().unwrap_or_default() {
            for value in self.placements.get(&PlacementSite::Operation(op)).cloned().unwrap_or_default() {
                self.emit(block, Instruction::BindExpression(value, Value::Source(value)));
            }
            block = self.operation(op, block, device)?;
            self.data.state.blocks[block].source_regions.insert(region);
        }
        self.current_region = previous_region;
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
                self.branch(block, Value::Source(condition), yes, no);
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
        let test = self.block(owner, vec![acc_name, index_name]);
        let iterate = self.block(owner, vec![]);
        let after = self.block(owner, vec![]);
        self.data.state.blocks[test].loop_exit = Some(after);
        self.jump(from, test, vec![Value::Source(init), Value::Int(0)]);
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
        self.branch(test_end, condition, iterate, after);
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
            test,
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
    fn buffer(&mut self, name: &str, length: Value, element: Type, storage: Storage) -> BufferId {
        self.data.state.buffers.alloc(BufferData {
            name: name.into(),
            length,
            element,
            storage,
        })
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

#[derive(Clone, Copy, Debug)]
enum Recipe {
    Elements,
    Totals,
    Prefixes,
    Compact,
    Serial,
}

fn recipes(graph: &EGraph) -> Result<BTreeMap<OperationId, Recipe>, OptimizeError> {
    let (rows, _, dag) = graph.function_to_dag("Plan", usize::MAX, false)?;
    let mut result = BTreeMap::new();
    for row in rows {
        let Term::App(_, args) = dag.get(row) else {
            return Err(error("invalid scheduling decision"));
        };
        let [operation, recipe] = args.as_slice() else {
            return Err(error("invalid scheduling decision arity"));
        };
        let Term::App(_, key) = dag.get(*operation) else {
            return Err(error("invalid operation key"));
        };
        let [key] = key.as_slice() else {
            return Err(error("invalid operation key arity"));
        };
        let Term::Lit(Literal::Int(id)) = dag.get(*key) else {
            return Err(error("invalid operation identity"));
        };
        let op =
            OperationId::from(u32::try_from(*id).map_err(|_| error("operation identity is out of range"))?);
        let Term::App(recipe, _) = dag.get(*recipe) else {
            return Err(error("invalid kernel recipe"));
        };
        let recipe = match recipe.as_str() {
            "Elements" => Recipe::Elements,
            "Totals" => Recipe::Totals,
            "Prefixes" => Recipe::Prefixes,
            "Compact" => Recipe::Compact,
            "Serial" => Recipe::Serial,
            _ => return Err(error("unknown kernel recipe")),
        };
        result.insert(op, recipe);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod schedule_tests;
