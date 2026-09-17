//! Rebuild interned expressions when changing their operands or bindings.
use crate::egglog::data::{ExprData, ExprId, ExprKind, Ir, OperationKind, TypeId};
use std::collections::BTreeMap;
use wyn_base::InternIndex;

pub(super) fn all(data: &mut Ir, replacements: &BTreeMap<ExprId, ExprId>) {
    let mut memo = replacements.clone();
    let mut rewrite = Rewriter::new(data);
    let operations: Vec<_> = data.operations.iter().map(|(&id, o)| (id, o.kind.clone())).collect();
    for (id, mut kind) in operations {
        rewrite.operation(data, &mut kind, &mut memo);
        data.operations[id].kind = kind;
    }
    let regions: Vec<_> = data.regions.iter().map(|(&id, r)| (id, r.results.clone())).collect();
    for (id, values) in regions {
        data.regions[id].results = values.into_iter().map(|v| rewrite.value(data, v, &mut memo)).collect();
    }
}
/// A rewrite session indexes interned values once and incrementally adds results.
/// Keep it across related substitutions to avoid arena scans for every node.
pub(super) struct Rewriter {
    expressions: InternIndex<ExprId, ExprData>,
}
impl Rewriter {
    pub fn new(data: &Ir) -> Self {
        Self {
            expressions: InternIndex::from_arena(&data.expressions),
        }
    }
    pub fn intern(&mut self, data: &mut Ir, ty: TypeId, kind: ExprKind) -> ExprId {
        self.expressions.intern(&mut data.expressions, &ExprData { ty, kind })
    }
    pub(super) fn value(
        &mut self,
        data: &mut Ir,
        id: ExprId,
        memo: &mut BTreeMap<ExprId, ExprId>,
    ) -> ExprId {
        if let Some(&v) = memo.get(&id) {
            return v;
        }
        let ExprData { ty, mut kind } = data.expressions[id].clone();
        kind.for_each_child_mut(&mut |v| *v = self.value(data, *v, memo));
        if let ExprKind::Project { tuple, index } = &kind {
            if let ExprKind::Tuple(fields) = &data.expressions[*tuple].kind {
                if let Some(&v) = fields.get(*index) {
                    memo.insert(id, v);
                    return v;
                }
            }
        }
        let v = self.intern(data, ty, kind);
        memo.insert(id, v);
        v
    }
    pub(super) fn operation(
        &mut self,
        data: &mut Ir,
        kind: &mut OperationKind,
        memo: &mut BTreeMap<ExprId, ExprId>,
    ) {
        kind.for_each_operand_mut(&mut |v| *v = self.value(data, *v, memo));
    }
}
