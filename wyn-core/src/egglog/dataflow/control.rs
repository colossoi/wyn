//! Evaluation topology shared by analyses of effects and resource lifetimes.
use crate::egglog::{query::Query, OptimizeError};
use crate::tlc::{
    ArrayExpr, Family, Lambda, LoopKind, Payload, Program, SoacBody, SoacOp, Term, TermId, TermKind, VarRef,
};
use crate::{Interner, LookupMap, LookupSet, SymbolId};
use egglog_engine::{EGraph, Write};
use std::hash::Hash;

#[derive(Clone, Copy)]
pub(crate) enum Action {
    Evaluate(TermId),
    Define(TermId),
    Read(SymbolId),
}

#[derive(Default)]
pub(crate) struct ControlFlow {
    points: Vec<(Option<Action>, Vec<usize>)>,
    exits: LookupMap<TermId, usize>,
}

impl ControlFlow {
    pub fn build<Tag, F: Family, G>(program: &Program<Tag, F, G>) -> Self {
        let mut flow = Self::default();
        let end = flow.point(None, vec![]);
        for def in &program.defs {
            // Definition parameters belong to each call, not a repeated closure.
            let body = match &def.body.kind {
                TermKind::Lambda(lam) => &lam.body,
                _ => &def.body,
            };
            flow.term(body, end);
        }
        flow
    }

    pub fn live_out<R: Copy + Eq + Hash>(
        &self,
        mut transfer: impl FnMut(Action) -> (LookupSet<R>, LookupSet<R>),
    ) -> Result<LookupMap<TermId, LookupSet<R>>, OptimizeError> {
        let mut graph = EGraph::default();
        let mut resources = Interner::<i64, R>::new();
        graph.parse_and_run_program(Some("dataflow".into()), include_str!("../flow.egg"))?;
        graph.update(|mut sink| {
            for (point, (action, next)) in self.points.iter().enumerate() {
                for &next in next {
                    sink.add("FlowNext", (point as i64, next as i64))?;
                }
                if let Some(action) = action {
                    let (uses, kills) = transfer(*action);
                    for resource in uses {
                        sink.add("FlowLive", (point as i64, resources.intern(&resource)))?;
                    }
                    for resource in kills {
                        sink.set("FlowKilled", (point as i64, resources.intern(&resource)), true)?;
                    }
                }
            }
            Ok(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate flow-live))")?;
        let mut live = vec![LookupSet::default(); self.points.len()];
        Query(&graph).for_each("FlowLive", |row| {
            let point = graph.value_to_base::<i64>(row[0]) as usize;
            live[point].insert(*resources.resolve(graph.value_to_base::<i64>(row[1])));
            Ok(())
        })?;
        Ok(self.exits.iter().map(|(&term, &point)| (term, live[point].clone())).collect())
    }

    fn point(&mut self, action: Option<Action>, next: Vec<usize>) -> usize {
        let point = self.points.len();
        self.points.push((action, next));
        point
    }

    fn term<C: Payload, S: Payload>(&mut self, term: &Term<C, S>, after: usize) -> usize {
        self.exits.insert(term.id, after);
        match &term.kind {
            TermKind::Coerce { inner, .. } => self.term(inner, after),
            TermKind::Let { rhs, body, .. } => {
                let body = self.term(body, after);
                let bind = self.point(Some(Action::Define(term.id)), vec![body]);
                self.term(rhs, bind)
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let yes = self.term(then_branch, after);
                let no = self.term(else_branch, after);
                let join = self.point(None, vec![yes, no]);
                self.term(cond, join)
            }
            TermKind::Loop {
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                let exit = self.point(Some(Action::Evaluate(term.id)), vec![after]);
                let repeat = self.point(None, vec![exit]);
                let body = self.term(body, repeat);
                let bind = self.point(Some(Action::Define(term.id)), vec![body]);
                self.points[repeat].1.push(bind);
                let bound = match kind {
                    LoopKind::For { iter, .. } => iter,
                    LoopKind::ForRange { bound, .. } => bound,
                    LoopKind::While { cond } => cond,
                };
                let mut before = self.term(bound, bind);
                for (_, _, extract) in init_bindings.iter().rev() {
                    before = self.term(extract, before);
                }
                self.term(init, before)
            }
            TermKind::Lambda(lam) => {
                let body = self.repeated(lam, None);
                self.point(None, vec![after, body])
            }
            TermKind::Soac(op) => self.soac(op, term.id, after),
            TermKind::ArrayExpr(array) => self.array(array, after),
            TermKind::TupleProj { tuple, .. } if matches!(tuple.kind, TermKind::Var(VarRef::Symbol(_))) => {
                // The projection's resource use is component-sensitive.
                self.point(Some(Action::Evaluate(term.id)), vec![after])
            }
            TermKind::Var(_)
            | TermKind::App { .. }
            | TermKind::Closure(_)
            | TermKind::Tuple(_)
            | TermKind::VecLit(_)
            | TermKind::TupleProj { .. }
            | TermKind::Index { .. }
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {
                let finish = self.point(Some(Action::Evaluate(term.id)), vec![after]);
                let first = self.point(None, vec![]);
                let mut last = first;
                term.for_each_child(&mut |child| {
                    let next = self.point(None, vec![]);
                    let child = self.term(child, next);
                    self.points[last].1.push(child);
                    last = next;
                });
                self.points[last].1.push(finish);
                first
            }
        }
    }

    fn repeated<C: Payload, S: Payload>(
        &mut self,
        lam: &Lambda<C, S>,
        definition: Option<TermId>,
    ) -> usize {
        let repeat = self.point(definition.map(Action::Define), vec![]);
        let body = self.term(&lam.body, repeat);
        self.points[repeat].1.push(body);
        body
    }

    fn callback<C: Payload, S: Payload>(
        &mut self,
        body: &SoacBody<C, S>,
        definition: TermId,
        after: usize,
    ) -> usize {
        let repeated = self.repeated(&body.lam, Some(definition));
        let mut before = self.point(None, vec![after, repeated]);
        S::for_each_rev(&body.data, &mut |(_, _, capture)| {
            before = self.term(capture, before);
        });
        before
    }

    fn soac<C: Payload, S: Payload>(
        &mut self,
        op: &SoacOp<C, S>,
        definition: TermId,
        after: usize,
    ) -> usize {
        match op {
            SoacOp::Map { lam, inputs }
            | SoacOp::Scatter { lam, inputs, .. }
            | SoacOp::BucketScatter { lam, inputs, .. } => {
                let mut before = self.callback(lam, definition, after);
                for input in inputs.iter().rev() {
                    before = self.array(input, before);
                }
                before
            }
            SoacOp::Reduce { op, ne, input } | SoacOp::Scan { op, ne, input } => {
                let before = self.callback(op, definition, after);
                let before = self.array(input, before);
                self.term(ne, before)
            }
            SoacOp::Filter { pred, input } => {
                let before = self.callback(pred, definition, after);
                self.array(input, before)
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                let before = self.callback(op, definition, after);
                let before = self.array(values, before);
                let before = self.array(indices, before);
                self.term(ne, before)
            }
        }
    }

    fn array<C: Payload, S: Payload>(&mut self, array: &ArrayExpr<C, S>, mut after: usize) -> usize {
        match array {
            ArrayExpr::Var(VarRef::Symbol(symbol), _) => {
                self.point(Some(Action::Read(*symbol)), vec![after])
            }
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => after,
            ArrayExpr::Zip(arrays) => {
                for array in arrays.iter().rev() {
                    after = self.array(array, after);
                }
                after
            }
            ArrayExpr::Literal(terms) => {
                for term in terms.iter().rev() {
                    after = self.term(term, after);
                }
                after
            }
            ArrayExpr::Range { start, len, step } => {
                if let Some(step) = step {
                    after = self.term(step, after);
                }
                after = self.term(len, after);
                self.term(start, after)
            }
        }
    }
}
