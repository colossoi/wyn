//! Insert a separate expression DAG after fusion has selected its operations.
//! Region uses and execution dependencies are relations over that DAG; neither
//! expression ownership nor evaluation placement is implied by interning.

use super::data::{
    Array, AssociatedData, ExprId, ExprKind, LoopKind, OperationId, OperationKind, RegionId, SoacBody,
    SymbolId,
};
use super::{from_tlc::Converted, optimize::OptimizeError, snapshot};
use egglog_engine::{
    ast::{Command, Parser},
    EGraph,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Add typed expressions, region interfaces, structured control, and data-flow
/// facts to the selected fusion result. Scalar values retain their globally
/// interned sidecar identities. This pass does not rewrite or place expressions.
/// Run after `optimize` and before `schedule`; repeated insertion is an error.
pub fn insert_expressions(mut converted: Converted) -> Result<Converted, OptimizeError> {
    if converted.expression_program.is_some() || !converted.data.blocks.is_empty() {
        return Err(error("expression insertion must run once, before scheduling"));
    }
    let summary = snapshot::analyze(&converted.data);
    summary.schedules(&converted.data)?;
    let mut emitter = Emitter {
        data: &converted.data,
        live: &summary.live,
        symbols: converted.data.definitions.values().map(|d| (d.symbol, d.body)).collect(),
        output: include_str!("expressions.egg").into(),
        expressions: BTreeSet::new(),
        operations: BTreeSet::new(),
        declared: BTreeSet::new(),
        queued: BTreeSet::new(),
        pending: VecDeque::new(),
    };
    for (&id, entry) in &converted.data.entries {
        let region = converted.data.definitions[entry.definition].body;
        emitter.queue(region);
        emitter.fact(format!("(EntryRegion {} {})", id.as_u32(), region.egglog()));
    }
    while let Some(region) = emitter.pending.pop_front() {
        emitter.region(region)?;
    }
    for &(before, after) in &summary.effects {
        if emitter.operations.contains(&before) && emitter.operations.contains(&after) {
            emitter.fact(format!("(ExecutionOrder {} {})", before.egglog(), after.egglog()));
        }
    }
    emitter.fact("(run-schedule (saturate (run expressions)))".into());
    let commands = parse(&emitter.output)?;
    // Validate the independent layer with only the shared identity declarations.
    // No fusion rule or relation is required to interpret it.
    let mut graph = EGraph::default();
    graph.parse_and_run_program(Some("ids.egg".into()), include_str!("ids.egg"))?;
    graph.run_program(commands.clone())?;
    converted.program.extend(commands.iter().cloned());
    converted.expression_program = Some(commands);
    Ok(converted)
}

pub(super) fn parse(source: &str) -> Result<Vec<Command>, OptimizeError> {
    Parser::default()
        .get_program_from_string(Some("wyn-expressions.egg".into()), source)
        .map_err(|e| error(&e.to_string()))
}

fn error(message: &str) -> OptimizeError {
    OptimizeError::Output(format!("expression insertion: {message}"))
}

struct Emitter<'a> {
    data: &'a AssociatedData,
    live: &'a BTreeSet<OperationId>,
    symbols: BTreeMap<SymbolId, RegionId>,
    output: String,
    expressions: BTreeSet<ExprId>,
    operations: BTreeSet<OperationId>,
    declared: BTreeSet<RegionId>,
    queued: BTreeSet<RegionId>,
    pending: VecDeque<RegionId>,
}

impl Emitter<'_> {
    fn fact(&mut self, fact: String) {
        self.output.push_str(&fact);
        self.output.push('\n');
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
        self.fact(format!("(Region {})", region.egglog()));
        if let Some(parent) = data.parent {
            self.scope(parent)?;
            self.fact(format!("(RegionParent {} {})", region.egglog(), parent.egglog()));
        }
        for (i, &p) in data.parameters.iter().enumerate() {
            self.fact(format!(
                "(RegionParameter {} {i} (ParameterId {}) (TypeId {}))",
                region.egglog(),
                p.as_u32(),
                self.data.parameters[p].ty.as_u32()
            ));
        }
        for (&symbol, &body) in &self.symbols {
            if body == region {
                self.output.push_str(&format!(
                    "(NamedRegion (SymbolId {}) {})\n",
                    symbol.as_u32(),
                    region.egglog()
                ));
            }
        }
        Ok(())
    }
    fn region(&mut self, region: RegionId) -> Result<(), OptimizeError> {
        self.scope(region)?;
        let source = &self.data.regions[region];
        for (i, &value) in source.results.iter().enumerate() {
            let value = self.expression(value)?;
            self.fact(format!("(RegionResult {} {i} {value})", region.egglog()));
        }
        for &op in source.members.intersection(self.live) {
            self.operation(op)?;
        }
        Ok(())
    }
    fn values(&mut self, values: &[ExprId]) -> Result<String, OptimizeError> {
        let values = values.iter().map(|&id| self.expression(id)).collect::<Result<Vec<_>, _>>()?;
        Ok(vector(values))
    }
    fn expression(&mut self, id: ExprId) -> Result<String, OptimizeError> {
        let name = format!("$expr-{}", id.as_u32());
        if !self.expressions.insert(id) {
            return Ok(name);
        }
        let Some(data) = self.data.expressions.get(id) else {
            return Err(error("missing interned expression"));
        };
        let node = match &data.kind {
            ExprKind::Global(symbol) => {
                self.named(*symbol);
                format!("(Global (SymbolId {}))", symbol.as_u32())
            }
            ExprKind::Parameter(p) => {
                self.scope(self.data.parameters[*p].region)?;
                format!("(Parameter (ParameterId {}))", p.as_u32())
            }
            ExprKind::Builtin(b) => format!("(Builtin (BuiltinId {}))", b.as_u32()),
            ExprKind::Extern(e) => format!("(Extern (ExternId {}))", e.as_u32()),
            ExprKind::BinOp(op) => format!("(BinOp {})", quote(op)?),
            ExprKind::UnOp(op) => format!("(UnOp {})", quote(op)?),
            ExprKind::Int(text) => format!("(Int {})", quote(text)?),
            ExprKind::FloatBits(bits) => format!("(FloatBits {bits})"),
            ExprKind::Bool(value) => format!("(Bool {value})"),
            ExprKind::Unit => "(UnitValue)".into(),
            ExprKind::PureApp { function, args } => {
                format!("(PureApp {} {})", self.expression(*function)?, self.values(args)?)
            }
            ExprKind::Lambda(region) => {
                self.queue(*region);
                format!("(Lambda {})", region.egglog())
            }
            ExprKind::Closure {
                code,
                param_count,
                captures,
            } => {
                self.named(*code);
                format!(
                    "(Closure (SymbolId {}) {param_count} {})",
                    code.as_u32(),
                    self.values(captures)?
                )
            }
            ExprKind::Coerce(inner) => format!("(Coerce {})", self.expression(*inner)?),
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => format!(
                "(Select {} {} {})",
                self.expression(*condition)?,
                self.expression(*then_value)?,
                self.expression(*else_value)?
            ),
            ExprKind::Array(array) => format!("(ArrayValue {})", self.array(array)?),
            ExprKind::Tuple(values) => format!("(Tuple {})", self.values(values)?),
            ExprKind::Project { tuple, index } => format!("(Project {} {index})", self.expression(*tuple)?),
            ExprKind::Vector(values) => format!("(Vector {})", self.values(values)?),
            // Naming the result never duplicates its execution. In particular,
            // effectful calls remain distinct even when their arguments match.
            ExprKind::OperationResult(op) => {
                self.operation(*op)?;
                format!("(OperationResult {})", op.egglog())
            }
        };
        self.fact(format!(
            "(let {name} (Typed (TypeId {}) {node}))",
            data.ty.as_u32()
        ));
        self.fact(format!("(SourceExpression (ExprId {}) {name})", id.as_u32()));
        Ok(name)
    }
    fn array(&mut self, array: &Array) -> Result<String, OptimizeError> {
        Ok(match array {
            Array::Value(value) => format!("(ArrayInput {})", self.expression(*value)?),
            Array::Literal(values) => format!("(ArrayLiteral {})", self.values(values)?),
            Array::Zip(arrays) => format!(
                "(Zip {})",
                vector(arrays.iter().map(|a| self.array(a)).collect::<Result<Vec<_>, _>>()?)
            ),
            Array::Range { start, len, step } => {
                let start = self.expression(*start)?;
                let len = self.expression(*len)?;
                if let Some(step) = step {
                    format!("(StridedRange {start} {len} {})", self.expression(*step)?)
                } else {
                    format!("(Range {start} {len})")
                }
            }
        })
    }
    fn body(&mut self, body: &SoacBody) -> Result<String, OptimizeError> {
        Ok(match body {
            SoacBody::Apply { region, captures, .. } => {
                self.queue(*region);
                format!("(ApplyRegion {} {})", region.egglog(), self.values(captures)?)
            }
            SoacBody::Identity(types) => format!(
                "(Identity {})",
                vector(types.iter().map(|t| format!("(TypeId {})", t.as_u32())))
            ),
            SoacBody::Route { parameters, indices } => format!(
                "(Route {} {})",
                vector(parameters.iter().map(|t| format!("(TypeId {})", t.as_u32()))),
                vector(indices.iter().map(usize::to_string))
            ),
            SoacBody::Compose { first, then } => {
                format!("(Compose {} {})", self.body(first)?, self.body(then)?)
            }
            SoacBody::Parallel { left, right } => {
                format!("(Parallel {} {})", self.body(left)?, self.body(right)?)
            }
        })
    }
    fn operation_body(
        &mut self,
        op: OperationId,
        role: &str,
        body: &SoacBody,
    ) -> Result<(), OptimizeError> {
        let body = self.body(body)?;
        self.fact(format!("(OperationBody {} {role} {body})", op.egglog()));
        Ok(())
    }
    fn inputs(&mut self, op: OperationId, inputs: &[Array]) -> Result<(), OptimizeError> {
        for (i, input) in inputs.iter().enumerate() {
            let input = self.array(input)?;
            self.fact(format!("(OperationInput {} {i} {input})", op.egglog()));
        }
        Ok(())
    }
    fn operand(&mut self, relation: &str, op: OperationId, value: ExprId) -> Result<(), OptimizeError> {
        let value = self.expression(value)?;
        self.fact(format!("({relation} {} {value})", op.egglog()));
        Ok(())
    }
    fn neutrals(&mut self, op: OperationId, role: &str, values: &[ExprId]) -> Result<(), OptimizeError> {
        for (i, &value) in values.iter().enumerate() {
            let value = self.expression(value)?;
            self.fact(format!("(OperationNeutral {} {role} {i} {value})", op.egglog()));
        }
        Ok(())
    }
    fn operation(&mut self, op: OperationId) -> Result<(), OptimizeError> {
        if !self.operations.insert(op) {
            return Ok(());
        }
        if !self.live.contains(&op) {
            return Err(error("reachable expression refers to an inactive operation"));
        }
        let Some(source) = self.data.operations.get(op) else {
            return Err(error("missing source operation"));
        };
        self.scope(source.region)?;
        self.fact(format!(
            "(Execution {} {} (TypeId {}))",
            source.region.egglog(),
            op.egglog(),
            source.ty.as_u32()
        ));
        match &source.kind {
            OperationKind::Call { function, args } => {
                self.operand("Callee", op, *function)?;
                let args = self.values(args)?;
                self.fact(format!("(Arguments {} {args})", op.egglog()));
            }
            OperationKind::EvalGlobal(symbol) => {
                self.named(*symbol);
                self.fact(format!(
                    "(GlobalEvaluation {} (SymbolId {}))",
                    op.egglog(),
                    symbol.as_u32()
                ));
            }
            OperationKind::If {
                condition,
                then_region,
                else_region,
            } => {
                let condition = self.expression(*condition)?;
                self.queue(*then_region);
                self.queue(*else_region);
                self.fact(format!(
                    "(Conditional {} {condition} {} {})",
                    op.egglog(),
                    then_region.egglog(),
                    else_region.egglog()
                ));
            }
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            } => {
                self.queue(*header);
                self.queue(*body);
                self.fact(format!(
                    "(LoopRegions {} {} {})",
                    op.egglog(),
                    header.egglog(),
                    body.egglog()
                ));
                self.operand("LoopInitial", op, *init)?;
                match kind {
                    LoopKind::For(array) => self.operand("LoopArray", op, *array)?,
                    LoopKind::ForRange(bound) => self.operand("LoopBound", op, *bound)?,
                    LoopKind::While => self.fact(format!("(WhileLoop {})", op.egglog())),
                }
            }
            OperationKind::Index { array, index } => {
                let array = self.expression(*array)?;
                let index = self.expression(*index)?;
                self.fact(format!("(IndexOperands {} {array} {index})", op.egglog()));
            }
            OperationKind::Screma { form, inputs, .. } => {
                self.inputs(op, inputs)?;
                self.operation_body(op, "(Pre)", &form.pre)?;
                self.operation_body(op, "(Post)", &form.post)?;
                for (i, scan) in form.scans.iter().enumerate() {
                    let role = format!("(ScanOp {i})");
                    self.operation_body(op, &role, &scan.operator)?;
                    self.neutrals(op, &role, &scan.neutral)?;
                }
                for (i, reduction) in form.reductions.iter().enumerate() {
                    let role = format!("(ReduceOp {i})");
                    self.operation_body(op, &role, &reduction.operator)?;
                    self.neutrals(op, &role, &reduction.neutral)?;
                }
            }
            OperationKind::Filter { body, input, .. } => {
                self.inputs(op, std::slice::from_ref(input))?;
                self.operation_body(op, "(Callback)", body)?;
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
                self.operation_body(op, "(Callback)", body)?;
            }
            OperationKind::ReduceByIndex {
                destination,
                body,
                neutral,
                indices,
                values,
            } => {
                self.operand("Destination", op, destination.value)?;
                self.inputs(op, &[indices.clone(), values.clone()])?;
                self.operation_body(op, "(Callback)", body)?;
                self.neutrals(op, "(Callback)", &[*neutral])?;
            }
        }
        Ok(())
    }
}

fn vector(values: impl IntoIterator<Item = String>) -> String {
    format!("(vec-of {})", values.into_iter().collect::<Vec<_>>().join(" "))
}
fn quote(value: &str) -> Result<String, OptimizeError> {
    serde_json::to_string(value).map_err(|e| error(&e.to_string()))
}

#[cfg(test)]
#[path = "expressions_tests.rs"]
mod expressions_tests;
