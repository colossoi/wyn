//! Bounded weight queries over settled native region edges. No graph is copied.
use crate::egglog::OptimizeError;
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{I64Sort, SetContainer};
use egglog_engine::{ArcSort, Core, EGraph, Primitive, Read, ReadPrim, ReadState, Value};
use std::collections::HashSet;

pub(super) fn register(graph: &mut EGraph) -> Result<(), OptimizeError> {
    for (sort, shared) in [("RegionKey", false), ("OperationKey", true)] {
        let Some(input) = graph.get_sort_by_name(sort).cloned() else {
            return Err(OptimizeError::Output(format!("work query sort {sort} missing")));
        };
        graph.add_read_primitive(Weight { input, shared }, None);
    }
    Ok(())
}

#[derive(Clone)]
struct Weight {
    input: ArcSort,
    shared: bool,
}
impl Primitive for Weight {
    fn name(&self) -> &str {
        if self.shared {
            "wyn-operation-weight"
        } else {
            "wyn-inline-weight"
        }
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.input.clone(), I64Sort.to_arcsort(), I64Sort.to_arcsort()],
            span.clone(),
        )
        .into_box()
    }
}
impl ReadPrim for Weight {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let &[root, limit] = args else {
            unreachable!("invalid work query arity")
        };
        let limit = state.value_to_base::<i64>(limit);
        assert!(limit > 0, "work query bound must be positive");
        let lookup = |table, node| match state.lookup(table, (node,)) {
            Ok(value) => value,
            Err(error) => panic!("invalid native work table {table}: {error}"),
        };
        let children = |table, node| {
            let value = lookup(table, node)?;
            state
                .value_to_container::<SetContainer>(value)
                .map(|set| set.data.iter().copied().collect::<Vec<_>>())
        };
        let (mut weight, roots) = if self.shared {
            let value = lookup("SourceOperationValue", root)?;
            let weight = state.value_to_base::<i64>(lookup("SourceWork", value)?);
            (
                weight,
                children("OperationWorkRegions", root)?.into_iter().collect(),
            )
        } else {
            (0, vec![root])
        };
        let mut pending: Vec<_> = roots.into_iter().map(|node| (node, false)).collect();
        let mut active = HashSet::new();
        let mut seen = HashSet::new();
        while let Some((node, exit)) = pending.pop() {
            if exit {
                active.remove(&node);
                continue;
            }
            // Recursion cannot establish bounded work, even with zero local cost.
            if active.contains(&node) {
                return Some(state.base_to_value(limit));
            }
            if self.shared && !seen.insert(node) {
                continue;
            }
            let cost = state.value_to_base::<i64>(lookup("SourceRegionWork", node)?);
            weight = weight.saturating_add(cost);
            if weight >= limit {
                return Some(state.base_to_value(limit));
            }
            active.insert(node);
            pending.push((node, true));
            pending.extend(children("WorkChildren", node)?.into_iter().map(|child| (child, false)));
        }
        Some(state.base_to_value(weight.min(limit)))
    }
}
