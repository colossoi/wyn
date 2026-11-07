//! Shared structural flow analysis; arithmetic remains in TLC.
use super::query::Query;
use super::OptimizeError;
use crate::tlc::{
    self, ArrayExpr, Family, LoopKind, Payload, Program, SoacOp, Term, TermId, TermKind, VarRef,
    WalkDecision,
};
use crate::{Interner, LookupMap, LookupSet, StableMap, SymbolId};
pub(crate) use egglog_engine::Value;
use egglog_engine::{EGraph, Error, FullState, RawValues, Write};
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Origin {
    pub producer: SymbolId,
    pub path: Vec<usize>,
}

#[derive(Clone, Copy)]
struct Expression {
    value: Value,
    evaluation: Value,
    // Forwarding may borrow a producer, but must not discard other evaluations.
    forwarding: Value,
    control: (Value, Value),
    references: (usize, usize),
    reference: Option<usize>,
}

impl Expression {
    fn new(value: Value, evaluation: Value) -> Self {
        Self {
            value,
            evaluation,
            forwarding: evaluation,
            control: (evaluation, evaluation),
            references: (0, 0),
            reference: None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Action {
    Evaluate(TermId),
    Define(TermId),
    Read(SymbolId),
}

// Authored reference paths share their prefixes; semantic aliases remain in egglog.
enum Reference {
    Name(SymbolId),
    Field(usize, usize),
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
    actions: LookupMap<Value, Action>,
    // Postorder occurrences let capture queries select subtrees without walking TLC again.
    reference_occurrences: Vec<Reference>,
}

impl Dataflow {
    pub fn live_out<Tag, F: Family, G, R: Copy + Eq + Hash>(
        program: &Program<Tag, F, G>,
        mut transfer: impl FnMut(Action) -> (LookupSet<R>, LookupSet<R>),
    ) -> Result<LookupMap<TermId, LookupSet<R>>, OptimizeError> {
        let mut graph = EGraph::default();
        graph.parse_and_run_program(None, include_str!("flow.egg"))?;
        let mut flow = Self::default();
        let mut resources = Interner::<i64, R>::new();
        graph.update(|mut sink| {
            for def in &program.defs {
                let body = match &def.body.kind {
                    TermKind::Lambda(lam) => &lam.body,
                    _ => &def.body,
                };
                flow.import(body, &mut sink)?;
            }
            for (&point, &action) in &flow.actions {
                let (uses, kills) = transfer(action);
                for resource in uses {
                    sink.add("FlowLive", (point, resources.intern(&resource)))?;
                }
                for resource in kills {
                    sink.set("FlowKilled", (point, resources.intern(&resource)), true)?;
                }
            }
            Ok(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate flow-live))")?;
        let mut live = LookupMap::<_, LookupSet<_>>::default();
        Query(&graph).for_each("FlowLiveAfter", |row| {
            live.entry(row[0]).or_default().insert(*resources.resolve(graph.value_to_base::<i64>(row[1])));
            Ok(())
        })?;
        Ok(flow
            .terms
            .iter()
            .map(|(&term, expr)| (term, live.get(&expr.evaluation).cloned().unwrap_or_default()))
            .collect())
    }
    pub fn build<C: Payload, S: Payload>(term: &Term<C, S>) -> (Self, Value) {
        Self::try_build(term).expect("valid flow analysis")
    }

    pub fn try_build<C: Payload, S: Payload>(term: &Term<C, S>) -> Result<(Self, Value), OptimizeError> {
        let mut graph = EGraph::default();
        graph.parse_and_run_program(Some("dataflow".into()), include_str!("flow.egg"))?;
        let mut flow = Self::default();
        let result = graph.update(|mut sink| Ok(flow.import(term, &mut sink)?.value))?;
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate flow-forward) flow-projections (saturate source-summary) (saturate flow-evaluation))",
        )?;
        flow.graph = graph;
        Ok((flow, result))
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
            .for_each("FlowEvaluationRead", |row| {
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

    pub fn resolve_projection(&self, symbol: SymbolId, path: &[usize]) -> SymbolPath {
        self.binding_origin(symbol, path).map_or_else(
            || (symbol, path.to_vec()),
            |origin| (origin.producer, origin.path),
        )
    }

    /// Read authored names from the callback's import and resolve producers in its enclosing scope.
    pub fn capture_sources(
        &self,
        occurrences: &Self,
        term: TermId,
    ) -> Option<LookupMap<SymbolPath, LookupSet<SymbolPath>>> {
        let expr = occurrences.terms.get(&term)?;
        let (start, mut end) = expr.references;
        let mut sources = LookupMap::<_, LookupSet<_>>::default();
        while end > start {
            let mut index = end - 1;
            let mut path = Vec::new();
            while let Reference::Field(parent, field) = occurrences.reference_occurrences[index] {
                path.push(field);
                index = parent;
            }
            if let Reference::Name(symbol) = occurrences.reference_occurrences[index] {
                path.reverse();
                sources.entry(self.resolve_projection(symbol, &path)).or_default().insert((symbol, path));
            }
            end = index;
        }
        Some(sources)
    }

    pub fn binding_origin(&self, symbol: SymbolId, fields: &[usize]) -> Option<Origin> {
        self.origin(self.bindings.get(&symbol)?.value, fields)
    }

    pub fn term_origin(&self, term: TermId) -> Option<Origin> {
        self.forwarding_origin(*self.terms.get(&term)?)
    }

    fn forwarding_origin(&self, expr: Expression) -> Option<Origin> {
        let origin = self.origin(expr.value, &[])?;
        (expr.forwarding == self.bindings.get(&origin.producer)?.forwarding).then_some(origin)
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

    fn fresh(&mut self, sink: &mut FullState<'_, '_>) -> Result<Value, Error> {
        let region = sink.add("RegionId", 0i64)?;
        let value = sink.add("SourceTerm", (region, self.next))?;
        self.next += 1;
        Ok(value)
    }

    fn symbol(&mut self, symbol: SymbolId, sink: &mut FullState<'_, '_>) -> Result<Expression, Error> {
        if let Some(&expr) = self.bindings.get(&symbol) {
            return Ok(expr);
        }
        let value = sink.add("SourceGlobal", i64::from(symbol.0))?;
        sink.set("FlowInput", value, i64::from(symbol.0))?;
        let expr = Expression::new(value, value);
        self.bindings.insert(symbol, expr);
        Ok(expr)
    }

    fn computed(
        &mut self,
        table: &str,
        operands: &[Value],
        sink: &mut FullState<'_, '_>,
    ) -> Result<Value, Error> {
        let value = self.fresh(sink)?;
        for &operand in operands {
            sink.add(table, (value, operand))?;
        }
        Ok(value)
    }

    fn import<C: Payload, S: Payload>(
        &mut self,
        term: &Term<C, S>,
        sink: &mut FullState<'_, '_>,
    ) -> Result<Expression, Error> {
        let first_reference = self.reference_occurrences.len();
        let mut children = Vec::new();
        let mut body = None;
        let mut captures = 0;
        match &term.kind {
            TermKind::Let { name, .. } => {
                self.locals.insert(*name);
            }
            TermKind::Lambda(lambda) => {
                self.locals.extend(lambda.params.iter().map(|(name, _)| *name));
                body = Some(lambda.body.as_ref());
            }
            TermKind::Loop {
                loop_var,
                init_bindings,
                kind,
                body: repeated,
                ..
            } => {
                body = Some(repeated.as_ref());
                self.locals.insert(*loop_var);
                self.locals.extend(init_bindings.iter().map(|(name, _, _)| *name));
                if let LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } = kind {
                    self.locals.insert(*var);
                }
            }
            TermKind::Soac(soac) => {
                let callback = match soac {
                    SoacOp::Map { lam, .. }
                    | SoacOp::Scatter { lam, .. }
                    | SoacOp::BucketScatter { lam, .. } => lam,
                    SoacOp::Reduce { op, .. }
                    | SoacOp::Scan { op, .. }
                    | SoacOp::ReduceByIndex { op, .. } => op,
                    SoacOp::Filter { pred, .. } => pred,
                };
                self.locals.extend(callback.lam.params.iter().map(|(name, _)| *name));
                body = Some(callback.lam.body.as_ref());
                S::for_each(&callback.data, &mut |_| captures += 1);
            }
            _ => {}
        }
        let update = tlc::record_update_parts(term);
        let mut expr = match &term.kind {
            TermKind::Var(VarRef::Symbol(symbol))
            | TermKind::ArrayExpr(ArrayExpr::Var(VarRef::Symbol(symbol), _)) => {
                self.references.entry(*symbol).or_insert_with(|| (term.id, term.ty.clone(), term.span));
                let input = self.symbol(*symbol, sink)?;
                let evaluation = self.fresh(sink)?;
                sink.add("FlowReference", (evaluation, input.evaluation))?;

                let reference = self.reference_occurrences.len();
                self.reference_occurrences.push(Reference::Name(*symbol));
                Expression {
                    evaluation,
                    reference: Some(reference),
                    ..input
                }
            }
            TermKind::Coerce { inner, .. } => {
                let input = self.child(inner, sink, &mut children)?;
                let evaluation = self.computed("FlowEvaluation", &[input.evaluation], sink)?;
                Expression {
                    evaluation,
                    reference: None,
                    ..input
                }
            }
            TermKind::App { .. } if update.is_some() => {
                let (record, index, replacement) = update.unwrap();
                let record = self.child(record, sink, &mut children)?;
                let replacement = self.child(replacement, sink, &mut children)?;
                let value = self.computed("FlowOperand", &[record.value, replacement.value], sink)?;
                sink.add(
                    "SourceFieldUpdate",
                    (value, record.value, index as i64, replacement.value),
                )?;
                self.evaluated(value, &[record, replacement], sink)?
            }
            TermKind::Let { name, rhs, body, .. } => {
                let rhs = self.child(rhs, sink, &mut children)?;
                let value = self.fresh(sink)?;
                sink.set("SourceAlias", value, rhs.value)?;
                self.bindings.insert(*name, Expression { value, ..rhs });
                let body = self.child(body, sink, &mut children)?;
                let evaluation =
                    self.computed("FlowEvaluation", &[rhs.evaluation, body.evaluation], sink)?;
                Expression::new(body.value, evaluation)
            }
            TermKind::Tuple(fields)
            | TermKind::VecLit(fields)
            | TermKind::ArrayExpr(ArrayExpr::Literal(fields)) => {
                let fields: Vec<_> = fields
                    .iter()
                    .map(|field| self.child(field, sink, &mut children))
                    .collect::<Result<_, _>>()?;
                let value = self.computed(
                    "FlowOperand",
                    &fields.iter().map(|field| field.value).collect::<Vec<_>>(),
                    sink,
                )?;
                for (index, field) in fields.iter().enumerate() {
                    sink.add("SourceField", (value, index as i64, field.value))?;
                }
                self.evaluated(value, &fields, sink)?
            }
            TermKind::TupleProj { tuple, idx } => {
                let input = self.child(tuple, sink, &mut children)?;
                let value = self.fresh(sink)?;
                sink.add("SourceSelect", (value, input.value, *idx as i64))?;
                let reference = input.reference.map(|parent| {
                    let reference = self.reference_occurrences.len();
                    self.reference_occurrences.push(Reference::Field(parent, *idx));
                    reference
                });
                let evaluation = self.computed("FlowEvaluation", &[input.evaluation], sink)?;
                Expression {
                    value,
                    evaluation,
                    reference,
                    ..input
                }
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let condition = self.child(cond, sink, &mut children)?;
                let yes = self.child(then_branch, sink, &mut children)?;
                let no = self.child(else_branch, sink, &mut children)?;
                let value = self.computed("FlowOperand", &[condition.value, yes.value, no.value], sink)?;
                sink.add("FlowChoice", (value, yes.value, no.value))?;
                let expr = self.evaluated(value, &[condition], sink)?;
                sink.add("FlowCall", (expr.evaluation, yes.evaluation))?;
                sink.add("FlowCall", (expr.evaluation, no.evaluation))?;
                expr
            }
            _ => {
                let value = self.fresh(sink)?;
                let mut result: Result<(), Error> = Ok(());
                term.for_each_child(&mut |child| {
                    if result.is_err() {
                        return;
                    }
                    result = (|| {
                        let edge = if body.is_some_and(|body| std::ptr::eq(child, body)) {
                            if matches!(term.kind, TermKind::Lambda(_)) {
                                "FlowReference"
                            } else {
                                "FlowCall"
                            }
                        } else {
                            "FlowEvaluation"
                        };
                        let child = self.child(child, sink, &mut children)?.evaluation;
                        sink.add(edge, (value, child))?;
                        Ok(())
                    })();
                });
                result?;
                if let TermKind::Soac(
                    SoacOp::Scatter { dest, .. }
                    | SoacOp::BucketScatter { dest, .. }
                    | SoacOp::ReduceByIndex { dest, .. },
                ) = &term.kind
                {
                    let destination = self.symbol(dest.id, sink)?.evaluation;
                    sink.add("FlowReference", (value, destination))?;
                    self.reference_occurrences.push(Reference::Name(dest.id));
                }
                Expression::new(value, value)
            }
        };
        expr.references = (first_reference, self.reference_occurrences.len());
        expr.control = self.control(term, &children, captures, sink)?;
        sink.add("FlowExecution", (expr.evaluation, expr.control.0, expr.control.1))?;
        if term.id != TermId::SYNTHETIC {
            self.terms.insert(term.id, expr);
        }
        Ok(expr)
    }

    fn evaluated(
        &mut self,
        value: Value,
        operands: &[Expression],
        sink: &mut FullState<'_, '_>,
    ) -> Result<Expression, Error> {
        let evaluation = self.computed(
            "FlowEvaluation",
            &operands.iter().map(|expr| expr.evaluation).collect::<Vec<_>>(),
            sink,
        )?;
        Ok(Expression::new(value, evaluation))
    }

    fn child<C: Payload, S: Payload>(
        &mut self,
        term: &Term<C, S>,
        sink: &mut FullState<'_, '_>,
        children: &mut Vec<Expression>,
    ) -> Result<Expression, Error> {
        let expr = self.import(term, sink)?;
        children.push(expr);
        Ok(expr)
    }

    fn sequence(
        &mut self,
        children: &[Expression],
        mut after: Value,
        sink: &mut FullState<'_, '_>,
    ) -> Result<Value, Error> {
        for child in children.iter().rev() {
            sink.add("FlowNext", (child.control.1, after))?;
            after = child.control.0;
        }
        Ok(after)
    }

    fn control<C: Payload, S: Payload>(
        &mut self,
        term: &Term<C, S>,
        children: &[Expression],
        captures: usize,
        sink: &mut FullState<'_, '_>,
    ) -> Result<(Value, Value), Error> {
        let exit = self.fresh(sink)?;
        Ok(match &term.kind {
            TermKind::Coerce { .. } => children[0].control,
            TermKind::Let { .. } => {
                self.actions.insert(exit, Action::Define(term.id));
                sink.add("FlowNext", (children[0].control.1, exit))?;
                sink.add("FlowNext", (exit, children[1].control.0))?;
                (children[0].control.0, children[1].control.1)
            }
            TermKind::If { .. } => {
                for arm in &children[1..] {
                    sink.add("FlowNext", (children[0].control.1, arm.control.0))?;
                    sink.add("FlowNext", (arm.control.1, exit))?;
                }
                (children[0].control.0, exit)
            }
            TermKind::Lambda(_) | TermKind::Soac(_) => {
                let repeat = self.fresh(sink)?;
                if matches!(term.kind, TermKind::Soac(_)) {
                    self.actions.insert(repeat, Action::Define(term.id));
                }
                sink.add("FlowNext", (repeat, children[0].control.0))?;
                sink.add("FlowNext", (children[0].control.1, repeat))?;
                let enter = self.fresh(sink)?;
                sink.add("FlowNext", (enter, repeat))?;
                sink.add("FlowNext", (enter, exit))?;
                let enter = self.sequence(&children[1..1 + captures], enter, sink)?;
                (self.sequence(&children[1 + captures..], enter, sink)?, exit)
            }
            TermKind::Loop { kind, .. } => {
                self.actions.insert(exit, Action::Evaluate(term.id));
                let define = self.fresh(sink)?;
                self.actions.insert(define, Action::Define(term.id));
                let body = children[children.len() - 1];
                let enter = if matches!(kind, LoopKind::While { .. }) {
                    let condition = children[children.len() - 2];
                    sink.add("FlowNext", (define, condition.control.0))?;
                    sink.add("FlowNext", (condition.control.1, body.control.0))?;
                    sink.add("FlowNext", (condition.control.1, exit))?;
                    sink.add("FlowNext", (body.control.1, define))?;
                    self.sequence(&children[..children.len() - 2], define, sink)?
                } else {
                    let repeat = self.fresh(sink)?;
                    sink.add("FlowNext", (repeat, define))?;
                    sink.add("FlowNext", (repeat, exit))?;
                    sink.add("FlowNext", (define, body.control.0))?;
                    sink.add("FlowNext", (body.control.1, repeat))?;
                    self.sequence(&children[..children.len() - 1], repeat, sink)?
                };
                (enter, exit)
            }
            _ => {
                let action = match term.kind {
                    TermKind::Var(VarRef::Symbol(symbol)) if term.id == TermId::SYNTHETIC => {
                        Action::Read(symbol)
                    }
                    _ => Action::Evaluate(term.id),
                };
                self.actions.insert(exit, action);
                let children = if matches!(&term.kind, TermKind::TupleProj { tuple, .. } if matches!(tuple.kind, TermKind::Var(VarRef::Symbol(_))))
                {
                    &[]
                } else {
                    children
                };
                (self.sequence(children, exit, sink)?, exit)
            }
        })
    }
}

/// Exact symbol/field occurrences, before semantic forwarding is resolved.
pub(crate) type SymbolPath = (SymbolId, Vec<usize>);

pub(crate) fn projected_symbol_path<C: Payload, S: Payload>(term: &Term<C, S>) -> Option<SymbolPath> {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(symbol)) => Some((*symbol, vec![])),
        TermKind::TupleProj { tuple, idx } => {
            let (symbol, mut path) = projected_symbol_path(tuple)?;
            path.push(*idx);
            Some((symbol, path))
        }
        _ => None,
    }
}

pub(crate) fn reference_paths<C: Payload, S: Payload>(term: &Term<C, S>) -> Vec<SymbolPath> {
    let mut references = Vec::new();
    term.walk(&mut |term: &Term<C, S>| {
        if let Some(path) = projected_symbol_path(term) {
            references.push(path);
            return WalkDecision::Prune;
        }
        if let TermKind::Soac(
            SoacOp::Scatter { dest, .. }
            | SoacOp::BucketScatter { dest, .. }
            | SoacOp::ReduceByIndex { dest, .. },
        ) = &term.kind
        {
            references.push((dest.id, vec![]));
        }
        WalkDecision::Recurse
    });
    references
}
#[cfg(test)]
#[path = "dataflow_tests.rs"]
mod tests;
