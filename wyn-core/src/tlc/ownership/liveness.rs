//! Backward ownership liveness over the structured TLC tree.

use super::analysis::{AnalysisState, OwnerId};
use crate::tlc::{
    ArrayExpr, Family, Lambda, LoopKind, Payload, Program, SoacBody, SoacOp, Stage, Term, TermId, TermKind,
    VarRef,
};
use crate::LookupSet;

type LiveSet = LookupSet<OwnerId>;

/// Fill `state.live_out` for every stored term in `program`.
pub(super) fn solve<S: Stage>(program: &Program<S>, state: &mut AnalysisState) {
    let mut solver = Liveness::<
        <<S as Stage>::Family as Family>::ClosureData,
        <<S as Stage>::Family as Family>::SoacBodyData,
    > {
        state,
        payloads: std::marker::PhantomData,
    };
    for def in &program.defs {
        solver.analyze_def(&def.body);
    }
}

/// Walks the TLC tree in reverse evaluation order, computing the set
/// of owners reachable from some future use at every program point.
///
/// Iteration is structured-recursive: the tree's shape encodes the
/// successor relation. Loops and iterating SOACs run a fixed point
/// over the body so back-edges propagate.
struct Liveness<'a, C: Payload, S: Payload> {
    state: &'a mut AnalysisState,
    payloads: std::marker::PhantomData<fn() -> (C, S)>,
}

impl<C: Payload, S: Payload> Liveness<'_, C, S> {
    fn analyze_def(&mut self, def_body: &Term<C, S>) {
        // A top-level Def is not a stored lambda value: each call gets fresh
        // arguments, so bypass the repeated-invocation lambda fixed point.
        let body = match &def_body.kind {
            TermKind::Lambda(lam) => &*lam.body,
            _ => def_body,
        };
        self.analyze(body, LiveSet::new());
    }

    /// Given the live-out set, return the live-in set and record
    /// `live_out[term.id]`.
    fn analyze(&mut self, term: &Term<C, S>, live_after: LiveSet) -> LiveSet {
        self.state.live_out.insert(term.id, live_after.clone());

        match &term.kind {
            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => self.transfer(term.id, live_after),

            TermKind::Coerce { inner, .. } => self.analyze(inner, live_after),

            TermKind::Let { rhs, body, .. } => {
                let live_in_body = self.analyze(body, live_after);
                let defs_here = self.defs(term.id);
                let live_after_rhs = sub(&live_in_body, &defs_here);
                self.analyze(rhs, live_after_rhs)
            }

            TermKind::App { func, args } => {
                let mut live = self.transfer(term.id, live_after);
                for arg in args.iter().rev() {
                    live = self.analyze(arg, live);
                }
                self.analyze(func, live)
            }

            TermKind::Closure(data) => {
                let mut live = self.transfer(term.id, live_after);
                C::for_each_rev(data, &mut |capture| {
                    live = self.analyze(capture, std::mem::take(&mut live));
                });
                live
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let live_before_then = self.analyze(then_branch, live_after.clone());
                let live_before_else = self.analyze(else_branch, live_after);
                self.analyze(cond, union(&live_before_then, &live_before_else))
            }

            TermKind::Loop {
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                let live_after = self.transfer(term.id, live_after);
                let defs_here = self.defs(term.id);
                let mut live_after_body = live_after.clone();
                loop {
                    let live_in_body = self.analyze(body, live_after_body.clone());
                    let next = union(&live_after, &sub(&live_in_body, &defs_here));
                    if next == live_after_body {
                        break;
                    }
                    live_after_body = next;
                }
                let live_in_body = self.analyze(body, live_after_body);
                let live_after_kind = sub(&live_in_body, &defs_here);
                let mut live = self.analyze_loop_kind(kind, live_after_kind);
                for (_, _, extract) in init_bindings.iter().rev() {
                    live = self.analyze(extract, live);
                }
                self.analyze(init, live)
            }

            TermKind::Lambda(lam) => self.analyze_lambda(lam, live_after),
            TermKind::Soac(op) => self.analyze_soac(op, live_after, term.id),
            TermKind::ArrayExpr(array) => self.analyze_array_expr(array, live_after),

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                let mut live = self.transfer(term.id, live_after);
                for part in parts.iter().rev() {
                    live = self.analyze(part, live);
                }
                live
            }
            TermKind::TupleProj { tuple, .. } => {
                let live = self.transfer(term.id, live_after);
                if matches!(&tuple.kind, TermKind::Var(VarRef::Symbol(_))) {
                    live
                } else {
                    self.analyze(tuple, live)
                }
            }
            TermKind::Index { array, index } => {
                let live = self.transfer(term.id, live_after);
                let live = self.analyze(index, live);
                self.analyze(array, live)
            }
        }
    }

    fn analyze_loop_kind(&mut self, kind: &LoopKind<C, S>, live_after: LiveSet) -> LiveSet {
        match kind {
            LoopKind::For { iter, .. } => self.analyze(iter, live_after),
            LoopKind::ForRange { bound, .. } => self.analyze(bound, live_after),
            LoopKind::While { cond } => self.analyze(cond, live_after),
        }
    }

    /// A lambda value can be invoked repeatedly, so its body is a fixed point
    /// with no per-call owners to subtract.
    fn analyze_lambda(&mut self, lambda: &Lambda<C, S>, live_after: LiveSet) -> LiveSet {
        let live_in_body = self.lambda_body_fixed_point(lambda, &LiveSet::new());
        union(&live_after, &live_in_body)
    }

    fn analyze_soac(&mut self, op: &SoacOp<C, S>, live_after: LiveSet, soac_id: TermId) -> LiveSet {
        let per_call_defs = self.state.defs.get(&soac_id).cloned().unwrap_or_default();
        match op {
            SoacOp::Map { lam, inputs, .. } => {
                let mut live = self.soac_envelope_fixed_point(lam, &per_call_defs, live_after);
                for input in inputs.iter().rev() {
                    live = self.analyze_array_expr(input, live);
                }
                live
            }
            SoacOp::Reduce { op, ne, input, .. } | SoacOp::Scan { op, ne, input, .. } => {
                let after_op = self.soac_envelope_fixed_point(op, &per_call_defs, live_after);
                let after_input = self.analyze_array_expr(input, after_op);
                self.analyze(ne, after_input)
            }
            SoacOp::Filter { pred, input, .. } => {
                let after_pred = self.soac_envelope_fixed_point(pred, &per_call_defs, live_after);
                self.analyze_array_expr(input, after_pred)
            }
            SoacOp::Scatter { lam, inputs, .. } => {
                let mut live = self.soac_envelope_fixed_point(lam, &per_call_defs, live_after);
                for input in inputs.iter().rev() {
                    live = self.analyze_array_expr(input, live);
                }
                live
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                let after_op = self.soac_envelope_fixed_point(op, &per_call_defs, live_after);
                let after_values = self.analyze_array_expr(values, after_op);
                let after_indices = self.analyze_array_expr(indices, after_values);
                self.analyze(ne, after_indices)
            }
        }
    }

    /// Solve the body fixed point, then thread liveness backward through
    /// explicit captures.
    fn soac_envelope_fixed_point(
        &mut self,
        body: &SoacBody<C, S>,
        per_call_defs: &LiveSet,
        live_after: LiveSet,
    ) -> LiveSet {
        let live_in_body = self.lambda_body_fixed_point(&body.lam, per_call_defs);
        let mut live = union(&live_after, &live_in_body);
        S::for_each_rev(&body.data, &mut |(_, _, capture)| {
            live = self.analyze(capture, std::mem::take(&mut live));
        });
        live
    }

    fn lambda_body_fixed_point(&mut self, lambda: &Lambda<C, S>, per_call_defs: &LiveSet) -> LiveSet {
        let mut live_after_body = LiveSet::new();
        loop {
            let live_in_body = self.analyze(&lambda.body, live_after_body.clone());
            let next = sub(&live_in_body, per_call_defs);
            if next == live_after_body {
                break;
            }
            live_after_body = next;
        }
        self.analyze(&lambda.body, live_after_body)
    }

    fn analyze_array_expr(&mut self, array: &ArrayExpr<C, S>, live_after: LiveSet) -> LiveSet {
        match array {
            // Named SOAC-input atoms have no TermId, so add their owner uses
            // directly rather than fabricating an ID.
            ArrayExpr::Var(VarRef::Symbol(symbol), _) => {
                let mut live = live_after;
                live.extend(self.state.aliases_of(*symbol));
                live
            }
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => live_after,
            ArrayExpr::Zip(arrays) => {
                let mut live = live_after;
                for array in arrays.iter().rev() {
                    live = self.analyze_array_expr(array, live);
                }
                live
            }
            ArrayExpr::Literal(terms) => {
                let mut live = live_after;
                for term in terms.iter().rev() {
                    live = self.analyze(term, live);
                }
                live
            }
            ArrayExpr::Range { start, len, step } => {
                let after_step =
                    if let Some(step) = step { self.analyze(step, live_after) } else { live_after };
                let after_len = self.analyze(len, after_step);
                self.analyze(start, after_len)
            }
        }
    }

    /// Standard transfer: `live_in = (live_out − kills) ∪ uses`.
    fn transfer(&self, id: TermId, live_after: LiveSet) -> LiveSet {
        let mut live = live_after;
        if let Some(kills) = self.state.kills.get(&id) {
            for owner in kills {
                live.remove(owner);
            }
        }
        if let Some(uses) = self.state.uses.get(&id) {
            live.extend(uses.iter().copied());
        }
        live
    }

    fn defs(&self, id: TermId) -> LiveSet {
        self.state.defs.get(&id).cloned().unwrap_or_default()
    }
}

fn union(left: &LiveSet, right: &LiveSet) -> LiveSet {
    left.union(right).copied().collect()
}

fn sub(left: &LiveSet, right: &LiveSet) -> LiveSet {
    left.difference(right).copied().collect()
}
