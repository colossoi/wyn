//! An intentionally small diagnostic program printer. Entry roots drive the
//! traversal; metadata and the full expression arena are never reachability roots.

use super::data::{
    Array, AssociatedData, DefinitionId, DefinitionKind, ExprId, ExprKind, LoopKind, OperationId,
    OperationKind, ParameterId, RegionId, ScremaForm, SoacBody, SymbolId, TypeId,
};
use super::{
    extract,
    optimize::{analyze, OptimizeError},
};
use crate::{builtins, types, LookupMap, LookupSet};
use std::collections::{BTreeMap, VecDeque};

/// Reconstruct a readable program from the current graph, starting
/// at all entries. Within a region, walk backward from returned values and
/// required effects, then topologically order the reachable execution graph.
/// Membership and diagnostic metadata are never liveness roots.
/// This is a diagnostic syntax, not Wyn.
pub fn readout(data: &AssociatedData) -> Result<String, OptimizeError> {
    let graph = analyze(data)?;
    let schedules = extract::schedules(&graph, data)?;
    let mut printer = Printer {
        data,
        schedules,
        output: String::new(),
        pending: VecDeque::new(),
        queued: LookupSet::new(),
        definitions: data.definitions.iter().map(|(&id, def)| (def.symbol, id)).collect(),
    };
    for entry in data.entries.values() {
        printer.enqueue(entry.definition);
    }
    while let Some(id) = printer.pending.pop_front() {
        let def = &data.definitions[id];
        let kind = if matches!(def.kind, DefinitionKind::Entry(_)) { "entry" } else { "fn" };
        let heading = format!("{kind} {}", printer.name(def.symbol));
        printer.region(def.body, &heading, 0);
        printer.output.push('\n');
    }
    if printer.output.is_empty() {
        printer.output.push_str("// No reachable entry points.\n");
    }
    Ok(printer.output)
}

struct Printer<'a> {
    data: &'a AssociatedData,
    schedules: BTreeMap<RegionId, Vec<OperationId>>,
    output: String,
    pending: VecDeque<DefinitionId>,
    queued: LookupSet<DefinitionId>,
    definitions: LookupMap<SymbolId, DefinitionId>,
}

impl Printer<'_> {
    fn enqueue(&mut self, id: DefinitionId) {
        if self.queued.insert(id) {
            self.pending.push_back(id);
        }
    }
    fn name(&self, id: SymbolId) -> String {
        let name = &self.data.symbols[id].name;
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            name.clone()
        } else {
            format!("{name:?}")
        }
    }
    fn symbol(&mut self, id: SymbolId) -> String {
        if let Some(&def) = self.definitions.get(&id) {
            self.enqueue(def);
        }
        self.name(id)
    }
    fn parameter(&self, id: ParameterId) -> String {
        format!("{}#{}", self.name(self.data.parameters[id].symbol), id.as_u32())
    }
    fn ty(&self, id: TypeId) -> String {
        types::format_type(&self.data.types[id].ty)
    }
    fn line(&mut self, indent: usize, line: String) {
        self.output.push_str(&"  ".repeat(indent));
        self.output.push_str(&line);
        self.output.push('\n');
    }
    fn region(&mut self, id: RegionId, heading: &str, indent: usize) {
        let region = &self.data.regions[id];
        let parameters = region
            .parameters
            .iter()
            .map(|&id| format!("{}: {}", self.parameter(id), self.ty(self.data.parameters[id].ty)))
            .collect::<Vec<_>>()
            .join(", ");
        self.line(indent, format!("{heading}({parameters}) {{"));
        let mut printed = LookupSet::new();
        for operation in self.schedules.get(&id).cloned().unwrap_or_default() {
            self.operation(operation, indent + 1, &mut printed);
        }
        let results = self.values(&region.results, indent + 1, &mut printed);
        self.line(indent + 1, format!("return {results}"));
        self.line(indent, "}".into());
    }
    fn nested(&mut self, id: RegionId, indent: usize) -> String {
        let name = format!("r{}", id.as_u32());
        self.region(id, &format!("region {name}"), indent);
        name
    }
    fn values(&mut self, values: &[ExprId], indent: usize, printed: &mut LookupSet<ExprId>) -> String {
        values.iter().map(|&id| self.value(id, indent, printed)).collect::<Vec<_>>().join(", ")
    }
    fn value(&mut self, id: ExprId, indent: usize, printed: &mut LookupSet<ExprId>) -> String {
        let data = &self.data.expressions[id];
        match &data.kind {
            ExprKind::Global(symbol) => return self.symbol(*symbol),
            ExprKind::Parameter(parameter) => return self.parameter(*parameter),
            ExprKind::Builtin(id) => {
                return builtins::by_id(self.data.builtins[*id].builtin).raw.surface_name.into()
            }
            ExprKind::BinOp(op) | ExprKind::UnOp(op) => return op.clone(),
            ExprKind::Int(value) => return value.clone(),
            ExprKind::FloatBits(bits) => return format!("{:?}", f32::from_bits(*bits)),
            ExprKind::Bool(value) => return value.to_string(),
            ExprKind::Unit => return "()".into(),
            ExprKind::Extern(id) => return format!("extern {:?}", self.data.externs[*id].linkage_name),
            ExprKind::OperationResult(id) => return format!("op{}", id.as_u32()),
            _ => {}
        }
        let name = format!("v{}", id.as_u32());
        if !printed.insert(id) {
            return name;
        }
        let body = match &data.kind {
            ExprKind::PureApp { function, args } => {
                let callee = self.value(*function, indent, printed);
                let values = args.iter().map(|&id| self.value(id, indent, printed)).collect::<Vec<_>>();
                match (&self.data.expressions[*function].kind, values.as_slice()) {
                    (ExprKind::BinOp(_), [left, right]) => format!("({left} {callee} {right})"),
                    (ExprKind::UnOp(_), [value]) => format!("({callee}{value})"),
                    _ => format!("{callee}({})", values.join(", ")),
                }
            }
            ExprKind::Lambda(region) => self.nested(*region, indent),
            ExprKind::Closure { code, captures, .. } => format!(
                "closure {} [{}]",
                self.symbol(*code),
                self.values(captures, indent, printed)
            ),
            ExprKind::Coerce(inner) => format!(
                "cast({}, {})",
                self.value(*inner, indent, printed),
                self.ty(data.ty)
            ),
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => format!(
                "if {} then {} else {}",
                self.value(*condition, indent, printed),
                self.value(*then_value, indent, printed),
                self.value(*else_value, indent, printed)
            ),
            ExprKind::Array(array) => self.array(array, indent, printed),
            ExprKind::Tuple(values) => format!("({})", self.values(values, indent, printed)),
            ExprKind::Project { tuple, index } => {
                format!("{}.{index}", self.value(*tuple, indent, printed))
            }
            ExprKind::Vector(values) => format!("@[{}]", self.values(values, indent, printed)),
            _ => unreachable!("atomic values return before expression emission"),
        };
        self.line(indent, format!("let {name}: {} = {body}", self.ty(data.ty)));
        name
    }
    fn array(&mut self, array: &Array, indent: usize, printed: &mut LookupSet<ExprId>) -> String {
        match array {
            Array::Value(value) => self.value(*value, indent, printed),
            Array::Literal(values) => format!("[{}]", self.values(values, indent, printed)),
            Array::Zip(arrays) => format!("zip({})", self.arrays(arrays, indent, printed)),
            Array::Range { start, len, step } => {
                let start = self.value(*start, indent, printed);
                let len = self.value(*len, indent, printed);
                let step = step.map(|id| self.value(id, indent, printed)).unwrap_or_else(|| "1".into());
                format!("range(start={start}, len={len}, step={step})")
            }
        }
    }
    fn arrays(&mut self, arrays: &[Array], indent: usize, printed: &mut LookupSet<ExprId>) -> String {
        arrays.iter().map(|array| self.array(array, indent, printed)).collect::<Vec<_>>().join(", ")
    }
    fn body(&mut self, body: &SoacBody, indent: usize, printed: &mut LookupSet<ExprId>) -> String {
        match body {
            SoacBody::Compose { first, then } => format!(
                "compose({}, {})",
                self.body(first, indent, printed),
                self.body(then, indent, printed)
            ),
            SoacBody::Identity(types) => format!("identity/{}", types.len()),
            SoacBody::Function {
                function, captures, ..
            } => format!(
                "{} captures [{}]",
                self.symbol(*function),
                self.values(captures, indent, printed)
            ),
            SoacBody::Inline { region, captures, .. } => {
                let captures = self.values(captures, indent, printed);
                let body = self.nested(*region, indent);
                format!("{body} captures [{captures}]")
            }
        }
    }
    fn form(&mut self, form: &ScremaForm, indent: usize, printed: &mut LookupSet<ExprId>) -> String {
        let pre = self.body(&form.pre, indent, printed);
        let scans = form
            .scans
            .iter()
            .map(|scan| {
                format!(
                    "{} neutral [{}]",
                    self.body(&scan.operator, indent, printed),
                    self.values(&scan.neutral, indent, printed)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let reductions = form
            .reductions
            .iter()
            .map(|reduction| {
                format!(
                    "{} neutral [{}] commutative={}",
                    self.body(&reduction.operator, indent, printed),
                    self.values(&reduction.neutral, indent, printed),
                    reduction.commutative
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let post = self.body(&form.post, indent, printed);
        format!("pre={pre}, scans=[{scans}], reductions=[{reductions}], post={post}")
    }
    fn operation(&mut self, id: OperationId, indent: usize, printed: &mut LookupSet<ExprId>) {
        let operation = &self.data.operations[id];
        let body = match &operation.kind {
            OperationKind::Call { function, args } => format!(
                "{}({})",
                self.value(*function, indent, printed),
                self.values(args, indent, printed)
            ),
            OperationKind::EvalGlobal(symbol) => format!("evaluate {}", self.symbol(*symbol)),
            OperationKind::If {
                condition,
                then_region,
                else_region,
            } => {
                let condition = self.value(*condition, indent, printed);
                let then_region = self.nested(*then_region, indent);
                let else_region = self.nested(*else_region, indent);
                format!("if {condition} then {then_region} else {else_region}")
            }
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            } => {
                let init = self.value(*init, indent, printed);
                let kind = match kind {
                    LoopKind::For(value) => {
                        format!("for values in {}", self.value(*value, indent, printed))
                    }
                    LoopKind::ForRange(value) => {
                        format!("for index < {}", self.value(*value, indent, printed))
                    }
                    LoopKind::While => "while header condition".into(),
                };
                let header = self.nested(*header, indent);
                let body = self.nested(*body, indent);
                format!("loop init={init}, {kind}, header={header}, body={body}")
            }
            OperationKind::Index { array, index } => format!(
                "{}[{}]",
                self.value(*array, indent, printed),
                self.value(*index, indent, printed)
            ),
            OperationKind::Screma {
                form,
                inputs,
                ownership,
            } => format!(
                "screma(inputs=[{}], {}, ownership={ownership:?})",
                self.arrays(inputs, indent, printed),
                self.form(form, indent, printed)
            ),
            OperationKind::Filter {
                body,
                input,
                ownership,
            } => format!(
                "filter({}, {}, ownership={ownership:?})",
                self.body(body, indent, printed),
                self.array(input, indent, printed)
            ),
            OperationKind::Scatter {
                destination,
                body,
                inputs,
            } => format!(
                "scatter({}, {}, [{}])",
                self.value(destination.value, indent, printed),
                self.body(body, indent, printed),
                self.arrays(inputs, indent, printed)
            ),
            OperationKind::BucketScatter {
                destination,
                body,
                inputs,
                shape,
            } => {
                let shape = &self.data.bucket_shapes[*shape];
                format!(
                    "bucket_scatter({}, {}, [{}], dimensions={:?}, rank={})",
                    self.value(destination.value, indent, printed),
                    self.body(body, indent, printed),
                    self.arrays(inputs, indent, printed),
                    shape.input_dimensions,
                    shape.domain_rank
                )
            }
            OperationKind::ReduceByIndex {
                destination,
                body,
                neutral,
                indices,
                values,
            } => format!(
                "reduce_by_index({}, {}, {}, {}, {})",
                self.value(destination.value, indent, printed),
                self.body(body, indent, printed),
                self.value(*neutral, indent, printed),
                self.array(indices, indent, printed),
                self.array(values, indent, printed)
            ),
        };
        self.line(
            indent,
            format!("let op{}: {} = {body}", id.as_u32(), self.ty(operation.ty)),
        );
    }
}
