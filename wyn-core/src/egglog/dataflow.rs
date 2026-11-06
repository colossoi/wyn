//! Shared structural flow analysis; arithmetic remains in TLC.
use super::query::Query;
use crate::tlc::{self, ArrayExpr, LoopKind, Payload, SoacOp, Term, TermId, TermKind, VarRef};
use crate::{LookupMap, LookupSet, StableMap, SymbolId};
pub(crate) use egglog_engine::Value;
use egglog_engine::{EGraph, FullState, IntoValues, RawValues, Write};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Origin {
    pub producer: SymbolId,
    pub path: Vec<usize>,
}

#[derive(Clone, Copy)]
struct Expression {
    value: Value,
    evaluation: Value,
}

pub(crate) struct Dependencies(LookupSet<Value>);

#[derive(Default)]
pub(crate) struct Dataflow {
    graph: EGraph,
    next: i64,
    bindings: LookupMap<SymbolId, Expression>,
    terms: LookupMap<TermId, Expression>,
    producers: LookupMap<Value, SymbolId>,
    references: StableMap<SymbolId, (TermId, crate::types::Type, crate::ast::Span)>,
    locals: LookupSet<SymbolId>,
}

fn fact(sink: &mut FullState<'_, '_>, name: &str, args: impl IntoValues) -> Value {
    sink.add(name, args).expect("valid dataflow fact")
}

impl Dataflow {
    pub fn build<C: Payload, S: Payload>(term: &Term<C, S>) -> (Self, Value) {
        let mut graph = EGraph::default();
        graph
            .parse_and_run_program(Some("dataflow".into()), include_str!("flow.egg"))
            .expect("valid flow rules");
        let mut flow = Self::default();
        let result =
            graph.update(|mut sink| Ok(flow.import(term, &mut sink).value)).expect("valid flow import");
        graph
            .parse_and_run_program(
                None,
                "(run-schedule (saturate flow-forward) flow-projections (saturate source-summary))",
            )
            .expect("valid flow analysis");
        flow.graph = graph;
        (flow, result)
    }

    pub fn free_variables<C: Payload, S: Payload>(
        &self,
        mut excluded: impl FnMut(SymbolId) -> bool,
    ) -> Vec<Term<C, S>> {
        self.references
            .iter()
            .filter(|(symbol, _)| !self.locals.contains(symbol) && !excluded(**symbol))
            .map(|(&symbol, (id, ty, span))| Term {
                id: *id,
                ty: ty.clone(),
                span: *span,
                kind: TermKind::Var(VarRef::Symbol(symbol)),
            })
            .collect()
    }

    pub fn free_symbols(&self) -> impl Iterator<Item = SymbolId> + '_ {
        self.references.keys().copied().filter(|symbol| !self.locals.contains(symbol))
    }

    pub fn dependencies(&self, mut selected: impl FnMut(SymbolId) -> bool) -> Dependencies {
        let selected: LookupSet<_> =
            self.bindings.keys().copied().filter(|symbol| selected(*symbol)).collect();
        let mut depends = LookupSet::default();
        Query(&self.graph)
            .for_each("FlowRead", |row| {
                let symbol = Query(&self.graph).required("FlowInput", (row[1],))?;
                if selected.contains(&SymbolId(self.graph.value_to_base::<i64>(symbol) as u32)) {
                    depends.insert(row[0]);
                }
                Ok(())
            })
            .expect("valid flow dependencies");
        Dependencies(depends)
    }

    pub fn dependent_bindings(&self, selected: impl FnMut(SymbolId) -> bool) -> LookupSet<SymbolId> {
        let depends = self.dependencies(selected);
        self.bindings
            .iter()
            .filter_map(|(&symbol, expr)| depends.0.contains(&expr.value).then_some(symbol))
            .collect()
    }

    pub fn evaluation_depends(&self, term: TermId, depends: &Dependencies) -> bool {
        self.terms.get(&term).is_none_or(|expr| depends.0.contains(&expr.evaluation))
    }

    pub fn evaluation_uses<C: Payload, S: Payload>(
        term: &Term<C, S>,
        selected: impl FnMut(SymbolId) -> bool,
    ) -> bool {
        let (flow, _) = Self::build(term);
        flow.evaluation_depends(term.id, &flow.dependencies(selected))
    }

    pub fn array_depends<C: Payload, S: Payload>(
        &self,
        array: &ArrayExpr<C, S>,
        depends: &Dependencies,
    ) -> bool {
        match array {
            ArrayExpr::Var(VarRef::Symbol(symbol), _) => {
                self.bindings.get(symbol).is_none_or(|expr| depends.0.contains(&expr.evaluation))
            }
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => false,
            ArrayExpr::Zip(parts) => parts.iter().any(|part| self.array_depends(part, depends)),
            ArrayExpr::Literal(terms) => terms.iter().any(|term| self.evaluation_depends(term.id, depends)),
            ArrayExpr::Range { start, len, step } => {
                self.evaluation_depends(start.id, depends)
                    || self.evaluation_depends(len.id, depends)
                    || step.as_deref().is_some_and(|step| self.evaluation_depends(step.id, depends))
            }
        }
    }

    pub fn binding_input(&self, symbol: SymbolId) -> Option<SymbolId> {
        Query(&self.graph)
            .lookup("FlowStructuralInput", (self.bindings.get(&symbol)?.value,))
            .expect("valid input")
            .map(|symbol| SymbolId(self.graph.value_to_base::<i64>(symbol) as u32))
    }

    pub fn input_origin(&self, value: Value, fields: &[usize]) -> Option<SymbolId> {
        let query = Query(&self.graph);
        let mut path = query.constructor("FlowEnd", RawValues(vec![])).expect("valid path")?;
        for &field in fields.iter().rev() {
            path = query.constructor("FlowAt", (field as i64, path)).expect("valid path")?;
        }
        query
            .lookup("FlowInputAt", (value, path))
            .expect("valid input origin")
            .map(|symbol| SymbolId(self.graph.value_to_base::<i64>(symbol) as u32))
    }

    pub fn name_producer(&mut self, symbol: SymbolId) {
        let Some(&expr) = self.bindings.get(&symbol) else {
            return;
        };
        if self.forwarding_origin(expr).is_none() {
            self.producers.insert(expr.value, symbol);
        }
    }

    pub fn binding_origin(&self, symbol: SymbolId, fields: &[usize]) -> Option<Origin> {
        self.origin(self.bindings.get(&symbol)?.value, fields)
    }

    pub fn term_origin(&self, term: TermId) -> Option<Origin> {
        self.forwarding_origin(*self.terms.get(&term)?)
    }

    fn forwarding_origin(&self, expr: Expression) -> Option<Origin> {
        let origin = self.origin(expr.value, &[])?;
        (expr.evaluation == self.bindings.get(&origin.producer)?.evaluation).then_some(origin)
    }

    fn origin(&self, value: Value, fields: &[usize]) -> Option<Origin> {
        if let Some(&producer) = self.producers.get(&value) {
            return Some(Origin {
                producer,
                path: fields.to_vec(),
            });
        }
        let query = Query(&self.graph);
        if let Some(row) = query.row("SourceSelect", |row| row[0] == value).expect("valid projection") {
            let mut path = vec![self.graph.value_to_base::<i64>(row[2]) as usize];
            path.extend_from_slice(fields);
            return self.origin(row[1], &path);
        }
        self.origin(
            query.lookup("SourceAlias", (value,)).expect("valid alias")?,
            fields,
        )
    }

    fn fresh(&mut self, sink: &mut FullState<'_, '_>) -> Value {
        let region = fact(sink, "RegionId", 0i64);
        let value = fact(sink, "SourceTerm", (region, self.next));
        self.next += 1;
        value
    }

    fn symbol(&mut self, symbol: SymbolId, sink: &mut FullState<'_, '_>) -> Expression {
        if let Some(&expr) = self.bindings.get(&symbol) {
            return expr;
        }
        let value = fact(sink, "SourceGlobal", i64::from(symbol.0));
        sink.set("FlowInput", value, i64::from(symbol.0)).expect("valid input");
        let expr = Expression {
            value,
            evaluation: value,
        };
        self.bindings.insert(symbol, expr);
        expr
    }

    fn computed(&mut self, operands: &[Value], sink: &mut FullState<'_, '_>) -> Value {
        let value = self.fresh(sink);
        for &operand in operands {
            fact(sink, "FlowOperand", (value, operand));
        }
        value
    }

    fn import<C: Payload, S: Payload>(
        &mut self,
        term: &Term<C, S>,
        sink: &mut FullState<'_, '_>,
    ) -> Expression {
        match &term.kind {
            TermKind::Let { name, .. } => {
                self.locals.insert(*name);
            }
            TermKind::Lambda(lambda) => self.locals.extend(lambda.params.iter().map(|(name, _)| *name)),
            TermKind::Loop {
                loop_var,
                init_bindings,
                kind,
                ..
            } => {
                self.locals.insert(*loop_var);
                self.locals.extend(init_bindings.iter().map(|(name, _, _)| *name));
                if let LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } = kind {
                    self.locals.insert(*var);
                }
            }
            TermKind::Soac(soac) => {
                let body = match soac {
                    SoacOp::Map { lam, .. }
                    | SoacOp::Scatter { lam, .. }
                    | SoacOp::BucketScatter { lam, .. } => lam,
                    SoacOp::Reduce { op, .. }
                    | SoacOp::Scan { op, .. }
                    | SoacOp::ReduceByIndex { op, .. } => op,
                    SoacOp::Filter { pred, .. } => pred,
                };
                self.locals.extend(body.lam.params.iter().map(|(name, _)| *name));
            }
            _ => {}
        }
        let update = tlc::record_update_parts(term);
        let expr = match &term.kind {
            TermKind::Var(VarRef::Symbol(symbol))
            | TermKind::ArrayExpr(ArrayExpr::Var(VarRef::Symbol(symbol), _)) => {
                self.references.entry(*symbol).or_insert_with(|| (term.id, term.ty.clone(), term.span));
                self.symbol(*symbol, sink)
            }
            TermKind::Coerce { inner, .. } => self.import(inner, sink),
            TermKind::App { .. } if update.is_some() => {
                let (record, index, replacement) = update.unwrap();
                let record = self.import(record, sink);
                let replacement = self.import(replacement, sink);
                let value = self.computed(&[record.value, replacement.value], sink);
                fact(
                    sink,
                    "SourceFieldUpdate",
                    (value, record.value, index as i64, replacement.value),
                );
                self.evaluated(value, &[record, replacement], sink)
            }
            TermKind::Let { name, rhs, body, .. } => {
                let rhs = self.import(rhs, sink);
                let value = self.fresh(sink);
                sink.set("SourceAlias", value, rhs.value).expect("valid binding");
                self.bindings.insert(
                    *name,
                    Expression {
                        value,
                        evaluation: rhs.evaluation,
                    },
                );
                let body = self.import(body, sink);
                let evaluation = self.computed(&[rhs.evaluation, body.evaluation], sink);
                Expression {
                    value: body.value,
                    evaluation,
                }
            }
            TermKind::Tuple(fields)
            | TermKind::VecLit(fields)
            | TermKind::ArrayExpr(ArrayExpr::Literal(fields)) => {
                let fields: Vec<_> = fields.iter().map(|field| self.import(field, sink)).collect();
                let value =
                    self.computed(&fields.iter().map(|field| field.value).collect::<Vec<_>>(), sink);
                for (index, field) in fields.iter().enumerate() {
                    fact(sink, "SourceField", (value, index as i64, field.value));
                }
                self.evaluated(value, &fields, sink)
            }
            TermKind::TupleProj { tuple, idx } => {
                let input = self.import(tuple, sink);
                let value = self.fresh(sink);
                fact(sink, "SourceSelect", (value, input.value, *idx as i64));
                Expression {
                    value,
                    evaluation: input.evaluation,
                }
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let condition = self.import(cond, sink);
                let yes = self.import(then_branch, sink);
                let no = self.import(else_branch, sink);
                let value = self.computed(&[condition.value, yes.value, no.value], sink);
                fact(sink, "FlowChoice", (value, yes.value, no.value));
                self.evaluated(value, &[condition, yes, no], sink)
            }
            _ => {
                let mut inputs = Vec::new();
                term.for_each_child(&mut |child| inputs.push(self.import(child, sink).evaluation));
                if let TermKind::Soac(
                    SoacOp::Scatter { dest, .. }
                    | SoacOp::BucketScatter { dest, .. }
                    | SoacOp::ReduceByIndex { dest, .. },
                ) = &term.kind
                {
                    inputs.push(self.symbol(dest.id, sink).evaluation);
                }
                let value = self.computed(&inputs, sink);
                Expression {
                    value,
                    evaluation: value,
                }
            }
        };
        if term.id != TermId::SYNTHETIC {
            self.terms.insert(term.id, expr);
        }
        expr
    }

    fn evaluated(
        &mut self,
        value: Value,
        operands: &[Expression],
        sink: &mut FullState<'_, '_>,
    ) -> Expression {
        let evaluation = if operands.iter().all(|expr| expr.value == expr.evaluation) {
            value
        } else {
            self.computed(
                &operands.iter().map(|expr| expr.evaluation).collect::<Vec<_>>(),
                sink,
            )
        };
        Expression { value, evaluation }
    }
}

#[cfg(test)]
#[path = "dataflow_tests.rs"]
mod tests;
