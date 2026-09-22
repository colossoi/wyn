//! Structural traversal of the sidecar IR. Passes decide what each use means.
use crate::egglog::data::{Array, ExprId, ExprKind, LoopKind, OperationKind, RegionId, SoacBody};

#[derive(Clone, Copy)]
pub(super) enum OperandRole {
    Argument,
    Input,
    Capture,
    Neutral,
}

pub(super) enum Operand {
    Value(OperandRole, ExprId),
    Region(RegionId),
}

impl ExprKind {
    /// Direct value children; executions and lambda bodies are separate graphs.
    pub(super) fn children(&self) -> Vec<ExprId> {
        match self {
            Self::PureApp { function, args } => {
                std::iter::once(*function).chain(args.iter().copied()).collect()
            }
            Self::Tuple(xs) | Self::Vector(xs) | Self::Closure { captures: xs, .. } => xs.clone(),
            Self::Coerce(x) | Self::Project { tuple: x, .. } => vec![*x],
            Self::If {
                condition,
                then_value,
                else_value,
            } => vec![*condition, *then_value, *else_value],
            Self::Array(a) => {
                let mut out = Vec::new();
                a.for_each_value(&mut |e| out.push(e));
                out
            }
            Self::Global(_)
            | Self::Parameter(_)
            | Self::Builtin(_)
            | Self::BinOp(_)
            | Self::UnOp(_)
            | Self::Lambda(_)
            | Self::Int(_)
            | Self::FloatBits(_)
            | Self::Bool(_)
            | Self::Unit
            | Self::Extern(_)
            | Self::OperationResult(_) => vec![],
        }
    }

    pub(super) fn for_each_child_mut(&mut self, f: &mut impl FnMut(&mut ExprId)) {
        match self {
            Self::PureApp { function, args } => {
                f(function);
                args.iter_mut().for_each(f);
            }
            Self::Tuple(xs) | Self::Vector(xs) | Self::Closure { captures: xs, .. } => {
                xs.iter_mut().for_each(f)
            }
            Self::Coerce(x) | Self::Project { tuple: x, .. } => f(x),
            Self::If {
                condition,
                then_value,
                else_value,
            } => {
                f(condition);
                f(then_value);
                f(else_value);
            }
            Self::Array(a) => a.for_each_value_mut(f),
            Self::Global(_)
            | Self::Parameter(_)
            | Self::Builtin(_)
            | Self::BinOp(_)
            | Self::UnOp(_)
            | Self::Lambda(_)
            | Self::Int(_)
            | Self::FloatBits(_)
            | Self::Bool(_)
            | Self::Unit
            | Self::Extern(_)
            | Self::OperationResult(_) => {}
        }
    }
}

impl Array {
    pub(super) fn for_each_value(&self, f: &mut impl FnMut(ExprId)) {
        match self {
            Self::Value(e) => f(*e),
            Self::Literal(xs) => xs.iter().copied().for_each(f),
            Self::Zip(xs) => xs.iter().for_each(|a| a.for_each_value(f)),
            Self::Range { start, len, step } => {
                f(*start);
                f(*len);
                if let Some(e) = step {
                    f(*e);
                }
            }
        }
    }

    pub(super) fn for_each_value_mut(&mut self, f: &mut impl FnMut(&mut ExprId)) {
        match self {
            Self::Value(e) => f(e),
            Self::Literal(xs) => xs.iter_mut().for_each(f),
            Self::Zip(xs) => xs.iter_mut().for_each(|a| a.for_each_value_mut(f)),
            Self::Range { start, len, step } => {
                f(start);
                f(len);
                if let Some(e) = step {
                    f(e);
                }
            }
        }
    }
}

impl SoacBody {
    pub(super) fn for_each_apply<'a>(&'a self, f: &mut impl FnMut(&'a SoacBody)) {
        match self {
            Self::Apply { .. } => f(self),
            Self::Compose { first, then } => {
                first.for_each_apply(f);
                then.for_each_apply(f);
            }
            Self::Parallel { left, right } => {
                left.for_each_apply(f);
                right.for_each_apply(f);
            }
            Self::Identity(_) | Self::Route { .. } => {}
        }
    }

    pub(super) fn for_each_apply_mut(&mut self, f: &mut impl FnMut(&mut SoacBody)) {
        match self {
            Self::Apply { .. } => f(self),
            Self::Compose { first, then } => {
                first.for_each_apply_mut(f);
                then.for_each_apply_mut(f);
            }
            Self::Parallel { left, right } => {
                left.for_each_apply_mut(f);
                right.for_each_apply_mut(f);
            }
            Self::Identity(_) | Self::Route { .. } => {}
        }
    }
}

impl OperationKind {
    /// Invoked scalar regions, including callbacks in composed and parallel bodies.
    pub(super) fn callbacks<'a>(&'a self) -> Vec<&'a SoacBody> {
        let mut out = Vec::new();
        let mut body = |b: &'a SoacBody| b.for_each_apply(&mut |b| out.push(b));
        match self {
            Self::Screma { form, .. } => {
                body(&form.pre);
                body(&form.post);
                for s in &form.scans {
                    body(&s.operator);
                }
                for r in &form.reductions {
                    body(&r.operator);
                }
            }
            Self::Filter {
                map, body: b, post, ..
            } => {
                body(map);
                body(b);
                body(post);
            }
            Self::ReduceByIndex { map, body: b, .. } => {
                body(map);
                body(b);
            }
            Self::Scatter { body: b, .. } | Self::BucketScatter { body: b, .. } => body(b),
            Self::Call { .. }
            | Self::EvalGlobal(_)
            | Self::If { .. }
            | Self::Loop { .. }
            | Self::Index { .. } => {}
        }
        out
    }

    pub(super) fn for_each_callback_mut(&mut self, f: &mut impl FnMut(&mut SoacBody)) {
        match self {
            Self::Screma { form, .. } => {
                form.pre.for_each_apply_mut(f);
                form.post.for_each_apply_mut(f);
                for s in &mut form.scans {
                    s.operator.for_each_apply_mut(f);
                }
                for r in &mut form.reductions {
                    r.operator.for_each_apply_mut(f);
                }
            }
            Self::Filter { map, body, post, .. } => {
                map.for_each_apply_mut(f);
                body.for_each_apply_mut(f);
                post.for_each_apply_mut(f);
            }
            Self::ReduceByIndex { map, body, .. } => {
                map.for_each_apply_mut(f);
                body.for_each_apply_mut(f);
            }
            Self::Scatter { body, .. } | Self::BucketScatter { body, .. } => body.for_each_apply_mut(f),
            Self::Call { .. }
            | Self::EvalGlobal(_)
            | Self::If { .. }
            | Self::Loop { .. }
            | Self::Index { .. } => {}
        }
    }

    /// Structured control children, excluding separately invoked scalar callbacks.
    pub(super) fn structured_regions(&self) -> Vec<RegionId> {
        match self {
            Self::If {
                then_region,
                else_region,
                ..
            } => vec![*then_region, *else_region],
            Self::Loop { header, body, .. } => vec![*header, *body],
            _ => vec![],
        }
    }

    pub(super) fn for_each_operand(&self, f: &mut impl FnMut(Operand)) {
        for body in self.callbacks() {
            if let SoacBody::Apply { region, captures, .. } = body {
                f(Operand::Region(*region));
                captures.iter().for_each(|&e| f(Operand::Value(OperandRole::Capture, e)));
            }
        }
        for r in self.structured_regions() {
            f(Operand::Region(r));
        }
        let mut value = |role, e| f(Operand::Value(role, e));
        match self {
            Self::Call { function, args } => {
                value(OperandRole::Argument, *function);
                args.iter().for_each(|&e| value(OperandRole::Argument, e));
            }
            Self::EvalGlobal(_) => {}
            Self::If { condition, .. } => value(OperandRole::Argument, *condition),
            Self::Loop { init, kind, .. } => {
                value(OperandRole::Argument, *init);
                if let LoopKind::For(e) | LoopKind::ForRange(e) = kind {
                    value(OperandRole::Argument, *e);
                }
            }
            Self::Index { array, index } => {
                value(OperandRole::Argument, *array);
                value(OperandRole::Argument, *index);
            }
            Self::Screma { form, inputs, .. } => {
                for a in inputs {
                    a.for_each_value(&mut |e| value(OperandRole::Input, e));
                }
                for s in &form.scans {
                    s.neutral.iter().for_each(|&e| value(OperandRole::Neutral, e));
                }
                for r in &form.reductions {
                    r.neutral.iter().for_each(|&e| value(OperandRole::Neutral, e));
                }
            }
            Self::Filter { inputs, .. } => {
                for a in inputs {
                    a.for_each_value(&mut |e| value(OperandRole::Input, e));
                }
            }
            Self::Scatter {
                destination, inputs, ..
            }
            | Self::BucketScatter {
                destination, inputs, ..
            } => {
                value(OperandRole::Argument, destination.value);
                for a in inputs {
                    a.for_each_value(&mut |e| value(OperandRole::Input, e));
                }
            }
            Self::ReduceByIndex {
                destination,
                neutral,
                inputs,
                ..
            } => {
                value(OperandRole::Argument, destination.value);
                value(OperandRole::Neutral, *neutral);
                for a in inputs {
                    a.for_each_value(&mut |e| value(OperandRole::Input, e));
                }
            }
        }
    }

    pub(super) fn operands(&self, values: &mut Vec<ExprId>, regions: &mut Vec<RegionId>) {
        self.for_each_operand(&mut |operand| match operand {
            Operand::Value(_, e) => values.push(e),
            Operand::Region(r) => regions.push(r),
        });
    }

    /// Rewrite direct value operands, leaving invocation and binding identities intact.
    pub(super) fn for_each_operand_mut(&mut self, f: &mut impl FnMut(&mut ExprId)) {
        self.for_each_callback_mut(&mut |body| {
            if let SoacBody::Apply { captures, .. } = body {
                captures.iter_mut().for_each(&mut *f);
            }
        });
        match self {
            Self::Call { function, args } => {
                f(function);
                args.iter_mut().for_each(f);
            }
            Self::EvalGlobal(_) => {}
            Self::If { condition, .. } => f(condition),
            Self::Loop { init, kind, .. } => {
                f(init);
                if let LoopKind::For(e) | LoopKind::ForRange(e) = kind {
                    f(e);
                }
            }
            Self::Index { array, index } => {
                f(array);
                f(index);
            }
            Self::Screma { form, inputs, .. } => {
                for a in inputs {
                    a.for_each_value_mut(f);
                }
                for s in &mut form.scans {
                    s.neutral.iter_mut().for_each(&mut *f);
                }
                for r in &mut form.reductions {
                    r.neutral.iter_mut().for_each(&mut *f);
                }
            }
            Self::Filter { inputs, .. } => {
                for a in inputs {
                    a.for_each_value_mut(f);
                }
            }
            Self::Scatter {
                destination, inputs, ..
            }
            | Self::BucketScatter {
                destination, inputs, ..
            } => {
                f(&mut destination.value);
                for a in inputs {
                    a.for_each_value_mut(f);
                }
            }
            Self::ReduceByIndex {
                destination,
                neutral,
                inputs,
                ..
            } => {
                f(&mut destination.value);
                f(neutral);
                for a in inputs {
                    a.for_each_value_mut(f);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "visit_tests.rs"]
mod tests;
