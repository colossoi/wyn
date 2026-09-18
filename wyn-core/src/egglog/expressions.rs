//! Insert a separate expression DAG after fusion has selected its operations.
//! Region uses and execution dependencies are relations over that DAG; neither
//! expression ownership nor evaluation placement is implied by interning.
use super::{Expressions, Fused, OptimizeError, Program};
use crate::egglog::data::{
    Array, ExprId, ExprKind, Ir, LoopKind, OperationId, OperationKind, RegionId, SoacBody, SymbolId,
};
use crate::egglog::dependencies::{analyze, Dependencies};
use crate::egglog::timing::span;
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, EGraph, FullState, RawValues, Value, Write};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[cfg(test)]
pub(super) const RUN: &str = "(run-schedule (saturate (run expressions)))";

/// Read typed expressions, region interfaces, structured control, and data-flow
/// facts directly into egglog. The graph and its source-root table pass to
/// scalar simplification without an intervening textual or command representation.
pub fn insert_expressions(program: Program<Fused>) -> Result<Program<Expressions>, OptimizeError> {
    let _timing = span("egglog insert expressions");
    let dependencies = analyze(&program.ir);
    dependencies.schedules(&program.ir)?;
    let mut graph = EGraph::default();
    graph.parse_and_run_program(Some("ids.egg".into()), include_str!("ids.egg"))?;
    graph.parse_and_run_program(Some("expressions.egg".into()), include_str!("expressions.egg"))?;
    graph.update(|sink| Ok(read(&program.ir, &dependencies, sink)))??;
    Ok(Program {
        ir: program.ir,
        state: Expressions { graph },
    })
}

fn read(data: &Ir, summary: &Dependencies, sink: FullState<'_, '_>) -> Result<(), OptimizeError> {
    let mut reader = Reader {
        data,
        live: &summary.live,
        symbols: data.definitions.values().map(|d| (d.symbol, d.body)).collect(),
        sink,
        expressions: BTreeMap::new(),
        operations: BTreeSet::new(),
        declared: BTreeSet::new(),
        queued: BTreeSet::new(),
        pending: VecDeque::new(),
    };
    for (&id, entry) in &data.entries {
        let region = data.definitions[entry.definition].body;
        reader.queue(region);
        let region = reader.key("RegionId", region.as_u32())?;
        reader.sink.add(
            "EntryRegion",
            (reader.sink.base_to_value(i64::from(id.as_u32())), region),
        )?;
    }
    while let Some(region) = reader.pending.pop_front() {
        reader.region(region)?;
    }
    for (before, after) in summary.effects.pairs(&reader.operations) {
        let before = reader.sink.add("OperationId", (i64::from(before.as_u32()),))?;
        let after = reader.sink.add("OperationId", (i64::from(after.as_u32()),))?;
        reader.sink.add("ExecutionOrder", (before, after))?;
    }
    Ok(())
}

fn error(message: &str) -> OptimizeError {
    OptimizeError::Output(format!("expression insertion: {message}"))
}

struct Reader<'ir, 'a, 'db> {
    data: &'ir Ir,
    live: &'ir BTreeSet<OperationId>,
    symbols: BTreeMap<SymbolId, RegionId>,
    sink: FullState<'a, 'db>,
    expressions: BTreeMap<ExprId, Value>,
    operations: BTreeSet<OperationId>,
    declared: BTreeSet<RegionId>,
    queued: BTreeSet<RegionId>,
    pending: VecDeque<RegionId>,
}

impl Reader<'_, '_, '_> {
    fn key(&mut self, name: &str, id: u32) -> Result<Value, OptimizeError> {
        Ok(self.sink.add(name, (i64::from(id),))?)
    }
    fn vector(&mut self, data: Vec<Value>, do_rebuild: bool) -> Value {
        self.sink.register_container(VecContainer { data, do_rebuild })
    }
    fn queue(&mut self, region: RegionId) {
        if self.queued.insert(region) {
            self.pending.push_back(region);
        }
    }
    fn named(&mut self, symbol: SymbolId) {
        if let Some(&region) = self.symbols.get(&symbol) {
            self.queue(region);
        }
    }
    fn scope(&mut self, region: RegionId) -> Result<(), OptimizeError> {
        if !self.declared.insert(region) {
            return Ok(());
        }
        let Some(data) = self.data.regions.get(region) else {
            return Err(error("missing region"));
        };
        let r = self.key("RegionId", region.as_u32())?;
        self.sink.add("Region", (r,))?;
        if let Some(parent) = data.parent {
            self.scope(parent)?;
            let parent = self.key("RegionId", parent.as_u32())?;
            self.sink.add("RegionParent", (r, parent))?;
        }
        for (i, &p) in data.parameters.iter().enumerate() {
            let parameter = self.key("ParameterId", p.as_u32())?;
            let ty = self.key("TypeId", self.data.parameters[p].ty.as_u32())?;
            self.sink.add(
                "RegionParameter",
                (r, self.sink.base_to_value(i as i64), parameter, ty),
            )?;
        }
        for (&symbol, &body) in &self.symbols {
            if body == region {
                let symbol = self.sink.add("SymbolId", (i64::from(symbol.as_u32()),))?;
                self.sink.add("NamedRegion", (symbol, r))?;
            }
        }
        Ok(())
    }
    fn region(&mut self, region: RegionId) -> Result<(), OptimizeError> {
        self.scope(region)?;
        let source = &self.data.regions[region];
        let r = self.key("RegionId", region.as_u32())?;
        for (i, &value) in source.results.iter().enumerate() {
            let value = self.expression(value)?;
            self.sink.add("RegionResult", (r, self.sink.base_to_value(i as i64), value))?;
        }
        for &op in source.members.intersection(self.live) {
            self.operation(op)?;
        }
        Ok(())
    }
    fn values(&mut self, values: &[ExprId]) -> Result<Value, OptimizeError> {
        let values = values.iter().map(|&id| self.expression(id)).collect::<Result<_, _>>()?;
        Ok(self.vector(values, true))
    }
    fn expression(&mut self, id: ExprId) -> Result<Value, OptimizeError> {
        if let Some(value) = self.expressions.get(&id) {
            return Ok(*value);
        }
        let Some(data) = self.data.expressions.get(id) else {
            return Err(error("missing interned expression"));
        };
        let node = match &data.kind {
            ExprKind::Global(symbol) => {
                self.named(*symbol);
                let symbol = self.key("SymbolId", symbol.as_u32())?;
                self.sink.add("Global", (symbol,))?
            }
            ExprKind::Parameter(p) => {
                self.scope(self.data.parameters[*p].region)?;
                let parameter = self.key("ParameterId", p.as_u32())?;
                self.sink.add("Parameter", (parameter,))?
            }
            ExprKind::Builtin(b) => {
                let builtin = self.key("BuiltinId", b.as_u32())?;
                self.sink.add("Builtin", (builtin,))?
            }
            ExprKind::Extern(e) => {
                let external = self.key("ExternId", e.as_u32())?;
                self.sink.add("Extern", (external,))?
            }
            ExprKind::BinOp(op) => self.sink.add("BinOp", (op.as_str(),))?,
            ExprKind::UnOp(op) => self.sink.add("UnOp", (op.as_str(),))?,
            ExprKind::Int(text) => self.sink.add("Int", (text.as_str(),))?,
            ExprKind::FloatBits(bits) => {
                self.sink.add("FloatBits", (self.sink.base_to_value(i64::from(*bits)),))?
            }
            ExprKind::Bool(value) => self.sink.add("Bool", (self.sink.base_to_value(*value),))?,
            ExprKind::Unit => self.sink.add("UnitValue", RawValues(vec![]))?,
            ExprKind::PureApp { function, args } => {
                let function = self.expression(*function)?;
                let args = self.values(args)?;
                self.sink.add("PureApp", (function, args))?
            }
            ExprKind::Lambda(region) => {
                self.queue(*region);
                let region = self.key("RegionId", region.as_u32())?;
                self.sink.add("Lambda", (region,))?
            }
            ExprKind::Closure {
                code,
                param_count,
                captures,
            } => {
                self.named(*code);
                let code = self.key("SymbolId", code.as_u32())?;
                let captures = self.values(captures)?;
                self.sink.add(
                    "Closure",
                    (code, self.sink.base_to_value(*param_count as i64), captures),
                )?
            }
            ExprKind::Coerce(inner) => {
                let inner = self.expression(*inner)?;
                self.sink.add("Coerce", (inner,))?
            }
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => {
                let condition = self.expression(*condition)?;
                let then_value = self.expression(*then_value)?;
                let else_value = self.expression(*else_value)?;
                self.sink.add("Select", (condition, then_value, else_value))?
            }
            ExprKind::Array(array) => {
                let array = self.array(array)?;
                self.sink.add("ArrayValue", (array,))?
            }
            ExprKind::Tuple(values) => {
                let values = self.values(values)?;
                self.sink.add("Tuple", (values,))?
            }
            ExprKind::Project { tuple, index } => {
                let tuple = self.expression(*tuple)?;
                self.sink.add("Project", (tuple, self.sink.base_to_value(*index as i64)))?
            }
            ExprKind::Vector(values) => {
                let values = self.values(values)?;
                self.sink.add("Vector", (values,))?
            }
            // Result identity keeps effectful executions distinct even when
            // their arguments match.
            ExprKind::OperationResult(op) => {
                self.operation(*op)?;
                let op = self.key("OperationId", op.as_u32())?;
                self.sink.add("OperationResult", (op,))?
            }
        };
        let ty = self.key("TypeId", data.ty.as_u32())?;
        let value = self.sink.add("Typed", (ty, node))?;
        self.sink.set("SourceExpression", (i64::from(id.as_u32()),), value)?;
        self.expressions.insert(id, value);
        Ok(value)
    }
    fn array(&mut self, array: &Array) -> Result<Value, OptimizeError> {
        Ok(match array {
            Array::Value(value) => {
                let value = self.expression(*value)?;
                self.sink.add("ArrayInput", (value,))?
            }
            Array::Literal(values) => {
                let values = self.values(values)?;
                self.sink.add("ArrayLiteral", (values,))?
            }
            Array::Zip(arrays) => {
                let arrays = arrays.iter().map(|a| self.array(a)).collect::<Result<_, _>>()?;
                let arrays = self.vector(arrays, true);
                self.sink.add("Zip", (arrays,))?
            }
            Array::Range { start, len, step } => {
                let start = self.expression(*start)?;
                let len = self.expression(*len)?;
                if let Some(step) = step {
                    let step = self.expression(*step)?;
                    self.sink.add("StridedRange", (start, len, step))?
                } else {
                    self.sink.add("Range", (start, len))?
                }
            }
        })
    }
    fn body(&mut self, body: &SoacBody) -> Result<Value, OptimizeError> {
        Ok(match body {
            SoacBody::Apply { region, captures, .. } => {
                self.queue(*region);
                let region = self.key("RegionId", region.as_u32())?;
                let captures = self.values(captures)?;
                self.sink.add("ApplyRegion", (region, captures))?
            }
            SoacBody::Identity(types) => {
                let types =
                    types.iter().map(|t| self.key("TypeId", t.as_u32())).collect::<Result<_, _>>()?;
                let types = self.vector(types, true);
                self.sink.add("Identity", (types,))?
            }
            SoacBody::Route { parameters, indices } => {
                let types =
                    parameters.iter().map(|t| self.key("TypeId", t.as_u32())).collect::<Result<_, _>>()?;
                let types = self.vector(types, true);
                let indices = indices.iter().map(|&i| self.sink.base_to_value(i as i64)).collect();
                let indices = self.vector(indices, false);
                self.sink.add("Route", (types, indices))?
            }
            SoacBody::Compose { first, then } => {
                let first = self.body(first)?;
                let then = self.body(then)?;
                self.sink.add("Compose", (first, then))?
            }
            SoacBody::Parallel { left, right } => {
                let left = self.body(left)?;
                let right = self.body(right)?;
                self.sink.add("Parallel", (left, right))?
            }
        })
    }
    fn operation_body(&mut self, op: Value, role: Value, body: &SoacBody) -> Result<(), OptimizeError> {
        let body = self.body(body)?;
        self.sink.add("OperationBody", (op, role, body))?;
        Ok(())
    }
    fn inputs(&mut self, op: Value, inputs: &[Array]) -> Result<(), OptimizeError> {
        for (i, input) in inputs.iter().enumerate() {
            let input = self.array(input)?;
            self.sink.add("OperationInput", (op, self.sink.base_to_value(i as i64), input))?;
        }
        Ok(())
    }
    fn operand(&mut self, relation: &str, op: Value, value: ExprId) -> Result<(), OptimizeError> {
        let value = self.expression(value)?;
        self.sink.add(relation, (op, value))?;
        Ok(())
    }
    fn neutrals(&mut self, op: Value, role: Value, values: &[ExprId]) -> Result<(), OptimizeError> {
        for (i, &value) in values.iter().enumerate() {
            let value = self.expression(value)?;
            self.sink.add(
                "OperationNeutral",
                (op, role, self.sink.base_to_value(i as i64), value),
            )?;
        }
        Ok(())
    }
    fn operation(&mut self, id: OperationId) -> Result<(), OptimizeError> {
        if !self.operations.insert(id) {
            return Ok(());
        }
        if !self.live.contains(&id) {
            return Err(error("reachable expression refers to an inactive operation"));
        }
        let Some(source) = self.data.operations.get(id) else {
            return Err(error("missing source operation"));
        };
        self.scope(source.region)?;
        let op = self.key("OperationId", id.as_u32())?;
        let region = self.key("RegionId", source.region.as_u32())?;
        let ty = self.key("TypeId", source.ty.as_u32())?;
        self.sink.add("Execution", (region, op, ty))?;
        match &source.kind {
            OperationKind::Call { function, args } => {
                self.operand("Callee", op, *function)?;
                let args = self.values(args)?;
                self.sink.add("Arguments", (op, args))?;
            }
            OperationKind::EvalGlobal(symbol) => {
                self.named(*symbol);
                let symbol = self.key("SymbolId", symbol.as_u32())?;
                self.sink.add("GlobalEvaluation", (op, symbol))?;
            }
            OperationKind::If {
                condition,
                then_region,
                else_region,
            } => {
                let condition = self.expression(*condition)?;
                self.queue(*then_region);
                self.queue(*else_region);
                let then_region = self.key("RegionId", then_region.as_u32())?;
                let else_region = self.key("RegionId", else_region.as_u32())?;
                self.sink.add("Conditional", (op, condition, then_region, else_region))?;
            }
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            } => {
                self.queue(*header);
                self.queue(*body);
                let header = self.key("RegionId", header.as_u32())?;
                let body = self.key("RegionId", body.as_u32())?;
                self.sink.add("LoopRegions", (op, header, body))?;
                self.operand("LoopInitial", op, *init)?;
                match kind {
                    LoopKind::For(array) => self.operand("LoopArray", op, *array)?,
                    LoopKind::ForRange(bound) => self.operand("LoopBound", op, *bound)?,
                    LoopKind::While => {
                        self.sink.add("WhileLoop", (op,))?;
                    }
                }
            }
            OperationKind::Index { array, index } => {
                let array = self.expression(*array)?;
                let index = self.expression(*index)?;
                self.sink.add("IndexOperands", (op, array, index))?;
            }
            OperationKind::Screma { form, inputs, .. } => {
                self.inputs(op, inputs)?;
                let pre = self.sink.add("Pre", RawValues(vec![]))?;
                let post = self.sink.add("Post", RawValues(vec![]))?;
                self.operation_body(op, pre, &form.pre)?;
                self.operation_body(op, post, &form.post)?;
                for (i, scan) in form.scans.iter().enumerate() {
                    let role = self.sink.add("ScanOp", (self.sink.base_to_value(i as i64),))?;
                    self.operation_body(op, role, &scan.operator)?;
                    self.neutrals(op, role, &scan.neutral)?;
                }
                for (i, reduction) in form.reductions.iter().enumerate() {
                    let role = self.sink.add("ReduceOp", (self.sink.base_to_value(i as i64),))?;
                    self.operation_body(op, role, &reduction.operator)?;
                    self.neutrals(op, role, &reduction.neutral)?;
                }
            }
            OperationKind::Filter {
                map, body, inputs, ..
            } => {
                self.inputs(op, inputs)?;
                let pre = self.sink.add("Pre", RawValues(vec![]))?;
                let callback = self.sink.add("Callback", RawValues(vec![]))?;
                self.operation_body(op, pre, map)?;
                self.operation_body(op, callback, body)?;
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
                self.operand("Destination", op, destination.value)?;
                self.inputs(op, inputs)?;
                let callback = self.sink.add("Callback", RawValues(vec![]))?;
                self.operation_body(op, callback, body)?;
            }
            OperationKind::ReduceByIndex {
                destination,
                map,
                body,
                neutral,
                inputs,
            } => {
                self.operand("Destination", op, destination.value)?;
                self.inputs(op, inputs)?;
                let pre = self.sink.add("Pre", RawValues(vec![]))?;
                let callback = self.sink.add("Callback", RawValues(vec![]))?;
                self.operation_body(op, pre, map)?;
                self.operation_body(op, callback, body)?;
                self.neutrals(op, callback, &[*neutral])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "expressions_tests.rs"]
mod expressions_tests;
