//! Structural traversal of the sidecar IR. Passes decide what each use means.
use super::data::*;

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
            Self::Filter { map, body: b, .. } | Self::ReduceByIndex { map, body: b, .. } => {
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
            Self::Filter { map, body, .. } | Self::ReduceByIndex { map, body, .. } => {
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
        use OperandRole::*;
        for body in self.callbacks() {
            if let SoacBody::Apply { region, captures, .. } = body {
                f(Operand::Region(*region));
                captures.iter().for_each(|&e| f(Operand::Value(Capture, e)));
            }
        }
        for r in self.structured_regions() {
            f(Operand::Region(r));
        }
        let mut value = |role, e| f(Operand::Value(role, e));
        match self {
            Self::Call { function, args } => {
                value(Argument, *function);
                args.iter().for_each(|&e| value(Argument, e));
            }
            Self::EvalGlobal(_) => {}
            Self::If { condition, .. } => value(Argument, *condition),
            Self::Loop { init, kind, .. } => {
                value(Argument, *init);
                if let LoopKind::For(e) | LoopKind::ForRange(e) = kind {
                    value(Argument, *e);
                }
            }
            Self::Index { array, index } => {
                value(Argument, *array);
                value(Argument, *index);
            }
            Self::Screma { form, inputs, .. } => {
                for a in inputs {
                    a.for_each_value(&mut |e| value(Input, e));
                }
                for s in &form.scans {
                    s.neutral.iter().for_each(|&e| value(Neutral, e));
                }
                for r in &form.reductions {
                    r.neutral.iter().for_each(|&e| value(Neutral, e));
                }
            }
            Self::Filter { inputs, .. } => {
                for a in inputs {
                    a.for_each_value(&mut |e| value(Input, e));
                }
            }
            Self::Scatter {
                destination, inputs, ..
            }
            | Self::BucketScatter {
                destination, inputs, ..
            } => {
                value(Argument, destination.value);
                for a in inputs {
                    a.for_each_value(&mut |e| value(Input, e));
                }
            }
            Self::ReduceByIndex {
                destination,
                neutral,
                inputs,
                ..
            } => {
                value(Argument, destination.value);
                value(Neutral, *neutral);
                for a in inputs {
                    a.for_each_value(&mut |e| value(Input, e));
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
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn screma_traversal_covers_composed_captures_collectives_and_array_descriptors() {
        let apply = |id| SoacBody::Apply {
            region: RegionId::from(id),
            parameters: vec![],
            results: vec![],
            captures: vec![ExprId::from(id)],
        };
        let mut operation = OperationKind::Screma {
            form: ScremaForm {
                pre: SoacBody::Compose {
                    first: Box::new(apply(1)),
                    then: Box::new(SoacBody::Parallel {
                        left: Box::new(apply(2)),
                        right: Box::new(apply(3)),
                    }),
                },
                post: apply(4),
                scans: vec![Scan {
                    operator: apply(5),
                    neutral: vec![ExprId::from(6)],
                }],
                reductions: vec![Reduction {
                    operator: apply(7),
                    neutral: vec![ExprId::from(8)],
                    commutative: true,
                }],
            },
            inputs: vec![Array::Zip(vec![
                Array::Value(ExprId::from(9)),
                Array::Literal(vec![ExprId::from(10)]),
                Array::Range {
                    start: ExprId::from(11),
                    len: ExprId::from(12),
                    step: Some(ExprId::from(13)),
                },
            ])],
            ownership: vec![],
        };
        let (mut values, mut regions) = (Vec::new(), Vec::new());
        operation.operands(&mut values, &mut regions);
        assert_eq!(
            values.iter().map(|e| e.as_u32()).collect::<BTreeSet<_>>(),
            (1..=13).collect()
        );
        assert_eq!(
            regions.iter().map(|r| r.as_u32()).collect::<Vec<_>>(),
            [1, 2, 3, 4, 5, 7]
        );
        assert!(operation.structured_regions().is_empty());

        operation.for_each_operand_mut(&mut |e| *e = ExprId::from(e.as_u32() + 100));
        let (mut rewritten, mut same_regions) = (Vec::new(), Vec::new());
        operation.operands(&mut rewritten, &mut same_regions);
        assert_eq!(
            rewritten.iter().map(|e| e.as_u32()).collect::<Vec<_>>(),
            values.iter().map(|e| e.as_u32() + 100).collect::<Vec<_>>()
        );
        assert_eq!(
            same_regions, regions,
            "rewriting values cannot change invocation identity"
        );
        let mut neutrals = Vec::new();
        operation.for_each_operand(&mut |op| {
            if let Operand::Value(OperandRole::Neutral, e) = op {
                neutrals.push(e.as_u32());
            }
        });
        assert_eq!(neutrals, [106, 108]);
    }

    #[test]
    fn structured_regions_and_expression_children_do_not_cross_invocations() {
        let mut operation = OperationKind::Loop {
            init: ExprId::from(1),
            header: RegionId::from(2),
            body: RegionId::from(3),
            kind: LoopKind::ForRange(ExprId::from(4)),
        };
        operation.for_each_operand_mut(&mut |e| *e = ExprId::from(e.as_u32() + 10));
        let (mut values, mut regions) = (Vec::new(), Vec::new());
        operation.operands(&mut values, &mut regions);
        assert_eq!(values, [ExprId::from(11), ExprId::from(14)]);
        assert_eq!(regions, operation.structured_regions());
        assert!(operation.callbacks().is_empty());
        assert!(ExprKind::Lambda(RegionId::from(2)).children().is_empty());
        assert!(ExprKind::OperationResult(OperationId::from(0)).children().is_empty());
        let mut expr = ExprKind::Closure {
            code: SymbolId::from(0),
            param_count: 1,
            captures: values,
        };
        expr.for_each_child_mut(&mut |e| *e = ExprId::from(e.as_u32() + 10));
        assert_eq!(expr.children(), [ExprId::from(21), ExprId::from(24)]);
    }
}
