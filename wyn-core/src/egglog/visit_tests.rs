use super::{Operand, OperandRole};
use crate::egglog::data::{
    Array, ExprId, ExprKind, LoopKind, OperationId, OperationKind, Reduction, RegionId, Scan, ScremaForm,
    SoacBody, SymbolId,
};
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
