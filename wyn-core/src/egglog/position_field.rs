//! Read executable positions from the Rust graph in planning rules.
use super::graph::Positions;
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::{ArcSort, Core, Primitive, PurePrim, PureState, Value};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) enum Field {
    Region,
    Order,
}

#[derive(Clone)]
pub(super) struct PositionField {
    pub positions: Arc<Positions>,
    pub field: Field,
    pub operation: ArcSort,
    pub output: ArcSort,
}

impl Primitive for PositionField {
    fn name(&self) -> &str {
        match self.field {
            Field::Region => "wyn-region-of",
            Field::Order => "wyn-operation-position",
        }
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![self.operation.clone(), self.output.clone()],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for PositionField {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [operation] = *args else {
            unreachable!("a position query takes one operation");
        };
        let position = self.positions.read().operations.get(&operation).copied()?;
        match self.field {
            Field::Region => position.region,
            Field::Order => position.order.map(|order| state.base_to_value(order)),
        }
    }
}
