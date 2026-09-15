//! Internal serialization of the constructed graph. No public rendering API.

use super::data::{
    Array, AssociatedData, ExprId, ExprKind, LoopKind, OperationId, OperationKind, Place, RegionId,
    ScremaForm, SoacBody, TypeId,
};
use super::SCHEMA;
use crate::{
    types::{self, SoacOwnership},
    LookupSet,
};

pub(super) fn program(data: &AssociatedData) -> String {
    let mut emitter = Emitter {
        data,
        output: SCHEMA.into(),
        next_aux: 0,
        expressions: LookupSet::new(),
        operations: LookupSet::new(),
        regions: LookupSet::new(),
    };
    for id in data.programs.ids() {
        emitter.emit(format!("(Program {})", id.egglog()));
    }
    for (id, ty) in &data.types {
        if types::is_copy(&ty.ty) {
            emitter.emit(format!("(CopyType {})", id.egglog()));
        }
    }
    for (id, def) in &data.definitions {
        let body = emitter.region(def.body);
        emitter.emit(format!(
            "(Definition {} {} {} {body})",
            id.egglog(),
            def.symbol.egglog(),
            def.ty.egglog()
        ));
    }
    for (id, entry) in &data.entries {
        emitter.emit(format!("(Entry {} {})", id.egglog(), entry.definition.egglog()));
    }
    for (id, param) in &data.entry_params {
        emitter.emit(format!(
            "(EntryParam {} {} {})",
            id.egglog(),
            param.entry.egglog(),
            param.position
        ));
    }
    for (id, bound) in &data.input_bounds {
        emitter.emit(format!(
            "(InputBound {} {} {})",
            id.egglog(),
            bound.entry.egglog(),
            bound.symbol.egglog()
        ));
    }
    // Preserve all constructed values and diagnostic origins. Selection of
    // live values after optimization belongs to the eventual final readout.
    for id in data.expressions.ids() {
        emitter.expr(id);
    }
    for (id, origin) in &data.origins {
        emitter.emit(format!(
            "(Origin {} {} {})",
            id.egglog(),
            origin.expression.binding_name(),
            origin.definition.egglog()
        ));
    }
    emitter.output
}

struct Emitter<'a> {
    data: &'a AssociatedData,
    output: String,
    next_aux: usize,
    expressions: LookupSet<ExprId>,
    operations: LookupSet<OperationId>,
    regions: LookupSet<RegionId>,
}

impl Emitter<'_> {
    fn emit(&mut self, text: String) {
        self.output.push_str(&text);
        self.output.push('\n');
    }
    fn bind(&mut self, expression: String) -> String {
        let name = format!("$aux-{}", self.next_aux);
        self.next_aux += 1;
        self.emit(format!("(let {name} {expression})"));
        name
    }
    // Flat bindings keep lists from creating deeply nested parser input.
    fn list(&mut self, nil: &str, cons: &str, values: Vec<String>) -> String {
        let mut tail = format!("({nil})");
        for value in values.into_iter().rev() {
            tail = self.bind(format!("({cons} {value} {tail})"));
        }
        tail
    }
    fn types(&mut self, values: &[TypeId]) -> String {
        self.list(
            "NoTypes",
            "TypesCons",
            values.iter().map(|id| id.egglog()).collect(),
        )
    }
    fn exprs(&mut self, values: &[ExprId]) -> String {
        let values = values.iter().map(|&id| self.expr(id)).collect();
        self.list("NoExprs", "ExprsCons", values)
    }
    fn expr(&mut self, id: ExprId) -> String {
        let name = id.binding_name();
        if !self.expressions.insert(id) {
            return name;
        }
        let value = &self.data.expressions[id];
        let body = match &value.kind {
            ExprKind::Global(id) => format!("(Global {})", id.egglog()),
            ExprKind::Parameter(id) => format!("(Parameter {})", id.egglog()),
            ExprKind::Builtin(id) => format!("(Builtin {})", id.egglog()),
            ExprKind::BinOp(op) => format!("(BinOp {})", quote(op)),
            ExprKind::UnOp(op) => format!("(UnOp {})", quote(op)),
            ExprKind::PureApp { function, args } => {
                format!("(PureApp {} {})", self.expr(*function), self.exprs(args))
            }
            ExprKind::Lambda(region) => format!("(LambdaValue {})", self.region(*region)),
            ExprKind::Closure {
                code,
                param_count,
                captures,
            } => format!(
                "(Closure {} {param_count} {})",
                code.egglog(),
                self.exprs(captures)
            ),
            ExprKind::Int(value) => format!("(Int {})", quote(value)),
            ExprKind::FloatBits(value) => format!("(FloatBits {value})"),
            ExprKind::Bool(value) => format!("(Bool {value})"),
            ExprKind::Unit => "(UnitLit)".into(),
            ExprKind::Coerce(value) => format!("(Coerce {})", self.expr(*value)),
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => format!(
                "(If {} {} {})",
                self.expr(*condition),
                self.expr(*then_value),
                self.expr(*else_value)
            ),
            ExprKind::Array(array) => format!("(ArrayValue {})", self.array(array)),
            ExprKind::Tuple(values) => format!("(Tuple {})", self.exprs(values)),
            ExprKind::Project { tuple, index } => format!("(Project {} {index})", self.expr(*tuple)),
            ExprKind::Vector(values) => format!("(Vector {})", self.exprs(values)),
            ExprKind::Extern(id) => format!("(Extern {})", id.egglog()),
            ExprKind::OperationResult(id) => format!("(OperationResult {})", id.egglog()),
        };
        self.emit(format!("(let {name} (Typed {} {body}))", value.ty.egglog()));
        self.emit(format!("(Expression {} {name})", id.egglog()));
        name
    }
    fn array(&mut self, array: &Array) -> String {
        let body = match array {
            Array::Value(value) => format!("(ArrayInput {})", self.expr(*value)),
            Array::Zip(arrays) => format!("(Zip {})", self.arrays(arrays)),
            Array::Literal(values) => format!("(ArrayLiteral {})", self.exprs(values)),
            Array::Range { start, len, step } => {
                let start = self.expr(*start);
                let len = self.expr(*len);
                let step = step
                    .map(|id| format!("(SomeExpr {})", self.expr(id)))
                    .unwrap_or_else(|| "(NoExpr)".into());
                format!("(Range {start} {len} {step})")
            }
        };
        self.bind(body)
    }
    fn arrays(&mut self, arrays: &[Array]) -> String {
        let values = arrays.iter().map(|array| self.array(array)).collect();
        self.list("NoArrays", "ArraysCons", values)
    }
    fn place(&mut self, place: &Place) -> String {
        format!("(Place {} {})", self.expr(place.value), place.elem_ty.egglog())
    }
    fn soac_body(&mut self, body: &SoacBody) -> String {
        let body = match body {
            SoacBody::Compose { first, then } => {
                format!("(ComposeBody {} {})", self.soac_body(first), self.soac_body(then))
            }
            SoacBody::Identity(types) => format!("(IdentityBody {})", self.types(types)),
            SoacBody::Function {
                function,
                parameters,
                results,
                captures,
            } => format!(
                "(FunctionBody {} {} {} {})",
                function.egglog(),
                self.types(parameters),
                self.types(results),
                self.exprs(captures)
            ),
            SoacBody::Inline {
                region,
                results,
                captures,
            } => format!(
                "(InlineBody {} {} {})",
                self.region(*region),
                self.types(results),
                self.exprs(captures)
            ),
        };
        self.bind(body)
    }
    fn form(&mut self, form: &ScremaForm) -> String {
        let pre = self.soac_body(&form.pre);
        let scans = form
            .scans
            .iter()
            .map(|scan| {
                let operator = self.soac_body(&scan.operator);
                let neutral = self.exprs(&scan.neutral);
                self.bind(format!("(Scan {operator} {neutral})"))
            })
            .collect();
        let scans = self.list("NoCollectives", "CollectivesCons", scans);
        let reductions = form
            .reductions
            .iter()
            .map(|reduction| {
                let operator = self.soac_body(&reduction.operator);
                let neutral = self.exprs(&reduction.neutral);
                self.bind(format!(
                    "(Reduction {operator} {neutral} {})",
                    reduction.commutative
                ))
            })
            .collect();
        let reductions = self.list("NoCollectives", "CollectivesCons", reductions);
        let post = self.soac_body(&form.post);
        self.bind(format!("(Screma {pre} {scans} {reductions} {post})"))
    }
    fn operation(&mut self, id: OperationId) -> String {
        let name = id.binding_name();
        if !self.operations.insert(id) {
            return name;
        }
        let operation = &self.data.operations[id];
        let body = match &operation.kind {
            OperationKind::Call { function, args } => {
                format!("(Call {} {})", self.expr(*function), self.exprs(args))
            }
            OperationKind::EvalGlobal(symbol) => format!("(EvalGlobal {})", symbol.egglog()),
            OperationKind::If {
                condition,
                then_region,
                else_region,
            } => format!(
                "(Choose {} {} {})",
                self.expr(*condition),
                self.region(*then_region),
                self.region(*else_region)
            ),
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            } => {
                let init = self.expr(*init);
                let header = self.region(*header);
                let kind = match kind {
                    LoopKind::For(value) => format!("(For {})", self.expr(*value)),
                    LoopKind::ForRange(value) => format!("(ForRange {})", self.expr(*value)),
                    LoopKind::While => "(While)".into(),
                };
                format!("(Loop {init} {header} {kind} {})", self.region(*body))
            }
            OperationKind::Index { array, index } => {
                format!("(Index {} {})", self.expr(*array), self.expr(*index))
            }
            OperationKind::Screma {
                form,
                inputs,
                ownership: destinations,
            } => {
                let form = self.form(form);
                let inputs = self.arrays(inputs);
                let destinations = self.list(
                    "NoOwnerships",
                    "OwnershipsCons",
                    destinations.iter().map(|&dest| ownership(dest).into()).collect(),
                );
                format!("(ScremaApp {form} {inputs} {destinations})")
            }
            OperationKind::Filter {
                body,
                input,
                ownership: destination,
            } => format!(
                "(Filter {} {} {})",
                self.soac_body(body),
                self.array(input),
                ownership(*destination)
            ),
            OperationKind::Scatter {
                destination,
                body,
                inputs,
            } => format!(
                "(Scatter {} {} {})",
                self.place(destination),
                self.soac_body(body),
                self.arrays(inputs)
            ),
            OperationKind::BucketScatter {
                destination,
                body,
                inputs,
                shape,
            } => format!(
                "(BucketScatter {} {} {} {})",
                self.place(destination),
                self.soac_body(body),
                self.arrays(inputs),
                shape.egglog()
            ),
            OperationKind::ReduceByIndex {
                destination,
                body,
                neutral,
                indices,
                values,
            } => format!(
                "(ReduceByIndex {} {} {} {} {})",
                self.place(destination),
                self.soac_body(body),
                self.expr(*neutral),
                self.array(indices),
                self.array(values)
            ),
        };
        self.emit(format!("(let {name} {body})"));
        self.emit(format!(
            "(Operation {} {} {} {name})",
            operation.region.egglog(),
            id.egglog(),
            operation.ty.egglog()
        ));
        self.emit(format!(
            "(SourcePosition {} {})",
            id.egglog(),
            operation.source_position
        ));
        name
    }
    fn region(&mut self, id: RegionId) -> String {
        let name = id.binding_name();
        if !self.regions.insert(id) {
            return name;
        }
        let region = &self.data.regions[id];
        self.emit(format!("(let {name} {})", id.egglog()));
        self.emit(format!("(Region {name})"));
        if let Some(parent) = region.parent {
            self.emit(format!("(ParentRegion {name} {})", parent.egglog()));
        }
        for (position, id) in region.parameters.iter().enumerate() {
            self.emit(format!(
                "(RegionParameter {name} {position} {} {})",
                id.egglog(),
                self.data.parameters[*id].ty.egglog()
            ));
        }
        for &id in &region.members {
            self.operation(id);
        }
        for (position, &id) in region.results.iter().enumerate() {
            let value = self.expr(id);
            self.emit(format!("(RegionResult {name} {position} {value})"));
        }
        name
    }
}

fn ownership(value: SoacOwnership) -> &'static str {
    match value {
        SoacOwnership::Fresh => "(Fresh)",
        SoacOwnership::UniqueInput => "(UniqueInput)",
    }
}
fn quote(value: &str) -> String {
    serde_json::Value::String(value.into()).to_string()
}
