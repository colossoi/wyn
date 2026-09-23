//! Summarize scalar calculations by their structural dependencies. Preserve
//! tuple fields, views, execution results, and arithmetic used by size proofs.
use crate::egglog::data::{is_slice, Array, ExprId, ExprKind, Ir, OperationKind};
use crate::types::{strip_existentials, Type, TypeName};
use crate::{LookupMap, LookupSet};
use wyn_base::persistent_sets::{Set, Sets, EMPTY};

// Keep a shared calculation as a boundary when expanding it would duplicate
// a wide input set at every use. Imported edges stay linear in source edges.
const MAX_FRONTIER: usize = 32;

pub(super) struct Expressions<'a> {
    data: &'a Ir,
    frontiers: LookupMap<ExprId, Set>,
    sets: Sets,
    sizes: LookupSet<ExprId>,
}

#[cfg(test)]
#[path = "expressions_tests.rs"]
mod tests;

impl<'a> Expressions<'a> {
    pub(super) fn new(data: &'a Ir) -> Self {
        Self {
            data,
            frontiers: LookupMap::new(),
            sets: Sets::default(),
            sizes: size_values(data),
        }
    }

    pub(super) fn needs_size(&self, e: ExprId) -> bool {
        self.sizes.contains(&e)
    }

    /// Vector lanes share their inputs' resources. Only size calculations need
    /// their individual field identities in the scheduling graph.
    pub(super) fn compact_vector(&self, e: ExprId) -> bool {
        if self.needs_size(e) {
            return false;
        }
        match self.data.expressions[e].kind {
            ExprKind::Vector(_) => true,
            ExprKind::Project { tuple, .. } => matches!(
                strip_existentials(&self.data.types[self.data.expressions[tuple].ty].ty),
                Type::Constructed(TypeName::Vec, _)
            ),
            _ => false,
        }
    }

    fn dependency_free(&self, e: ExprId) -> bool {
        match self.data.expressions[e].kind {
            ExprKind::Int(_) => !self.needs_size(e),
            ExprKind::FloatBits(_)
            | ExprKind::Bool(_)
            | ExprKind::Unit
            | ExprKind::Builtin(_)
            | ExprKind::BinOp(_)
            | ExprKind::UnOp(_) => true,
            _ => false,
        }
    }

    fn transparent(&self, e: ExprId) -> bool {
        if self.compact_vector(e) {
            return true;
        }
        match &self.data.expressions[e].kind {
            ExprKind::PureApp { function, args } => {
                if is_slice(self.data, *function) {
                    return false;
                }
                if self.needs_size(e)
                    && matches!(
                        strip_existentials(&self.data.types[self.data.expressions[e].ty].ty),
                        Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _)
                    )
                    && args.len() == 2
                    && matches!(&self.data.expressions[*function].kind, ExprKind::BinOp(op) if matches!(op.as_str(), "+" | "-" | "*"))
                {
                    return false;
                }
                true
            }
            ExprKind::If { .. } | ExprKind::Closure { .. } => true,
            _ => false,
        }
    }

    fn direct_children(&self, e: ExprId) -> Vec<ExprId> {
        if self.compact_vector(e) {
            if let ExprKind::Project { tuple, index } = self.data.expressions[e].kind {
                if let ExprKind::Vector(fields) = &self.data.expressions[tuple].kind {
                    return vec![fields[index]];
                }
            }
        }
        self.data.expressions[e].kind.children()
    }

    pub(super) fn children(&mut self, e: ExprId) -> Vec<ExprId> {
        let children = self.direct_children(e);
        // Field and view facts refer to their immediate children by identity.
        // Their values must remain imported even when they are calculations.
        if !self.transparent(e) {
            return children;
        }
        let order = wyn_graph::dag_postorder(
            children.iter().copied(),
            |e| self.frontiers.contains_key(&e),
            |e, out| {
                if self.transparent(e) {
                    out.extend(self.direct_children(e));
                }
            },
        );
        for e in order {
            let mut frontier = EMPTY;
            if self.transparent(e) {
                for child in self.direct_children(e) {
                    frontier = self.sets.union(frontier, self.frontiers[&child]);
                }
                if self.sets.iter(frontier).nth(MAX_FRONTIER).is_some() {
                    frontier = self.sets.singleton(e.as_u32());
                }
            } else if !self.dependency_free(e) {
                frontier = self.sets.singleton(e.as_u32());
            }
            self.frontiers.insert(e, frontier);
        }
        let mut frontier = EMPTY;
        for child in children {
            frontier = self.sets.union(frontier, self.frontiers[&child]);
        }
        self.sets.iter(frontier).map(ExprId::from).collect()
    }
}

/// Scalar extents originate in range counts and slice bounds. Follow their
/// expression dependencies and the branch-result aliases imported as AbiChoice.
/// Array metadata has its own facts, so an array length need not retain the
/// arithmetic that computes its elements.
fn size_values(data: &Ir) -> LookupSet<ExprId> {
    let mut pending = vec![];
    for expression in data.expressions.values() {
        match &expression.kind {
            ExprKind::Array(array) => range_counts(array, &mut pending),
            ExprKind::PureApp { function, args } if is_slice(data, *function) && args.len() == 3 => {
                pending.extend_from_slice(&args[1..]);
            }
            _ => {}
        }
    }
    for operation in data.operations.values() {
        match &operation.kind {
            OperationKind::Screma { inputs, .. }
            | OperationKind::Filter { inputs, .. }
            | OperationKind::Scatter { inputs, .. }
            | OperationKind::BucketScatter { inputs, .. }
            | OperationKind::ReduceByIndex { inputs, .. } => {
                for array in inputs {
                    range_counts(array, &mut pending);
                }
            }
            _ => {}
        }
    }
    let mut needed = LookupSet::new();
    while let Some(e) = pending.pop() {
        if !needed.insert(e) {
            continue;
        }
        let kind = &data.expressions[e].kind;
        pending.extend(kind.children());
        if let ExprKind::OperationResult(op) = kind {
            if let OperationKind::If {
                then_region,
                else_region,
                ..
            } = data.operations[*op].kind
            {
                pending.extend_from_slice(&data.regions[then_region].results);
                pending.extend_from_slice(&data.regions[else_region].results);
            }
        }
    }
    needed
}

fn range_counts(array: &Array, out: &mut Vec<ExprId>) {
    match array {
        Array::Range { len, .. } => out.push(*len),
        Array::Zip(arrays) => arrays.iter().for_each(|array| range_counts(array, out)),
        Array::Value(_) | Array::Literal(_) => {}
    }
}
