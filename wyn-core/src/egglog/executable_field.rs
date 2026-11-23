//! Read executable fields from the Rust graph in planning rules.
use super::graph::SharedStructure;
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::{ArcSort, Core, Primitive, PurePrim, PureState, Value};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) enum Field {
    Region,
    Order,
    FunctionRoot,
    CallTarget,
    ControlRegion,
    EdgeArgument,
}

#[derive(Clone)]
pub(super) struct ExecutableField {
    pub structure: Arc<SharedStructure>,
    pub field: Field,
    pub inputs: Vec<ArcSort>,
    pub output: ArcSort,
}

impl Primitive for ExecutableField {
    fn name(&self) -> &str {
        match self.field {
            Field::Region => "wyn-region-of",
            Field::Order => "wyn-operation-position",
            Field::FunctionRoot => "wyn-function-root",
            Field::CallTarget => "wyn-call-target",
            Field::ControlRegion => "wyn-control-region",
            Field::EdgeArgument => "wyn-edge-argument",
        }
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            self.inputs.iter().cloned().chain([self.output.clone()]).collect(),
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for ExecutableField {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let key = args[0];
        let graph = self.structure.read();
        match self.field {
            Field::Region => graph.operations.get(&key)?.region,
            Field::Order => graph.operations.get(&key)?.order.map(|order| state.base_to_value(order)),
            Field::FunctionRoot => graph.roots.get(&key).copied(),
            Field::CallTarget => graph.targets.get(&key).copied(),
            Field::ControlRegion => {
                let slot: i64 = state.value_to_base(args[1]);
                graph.controls.get(&(key, slot)).map(|edge| edge.region)
            }
            Field::EdgeArgument => {
                let edge: i64 = state.value_to_base(args[1]);
                let slot = usize::try_from(state.value_to_base::<i64>(args[2])).ok()?;
                graph.controls.get(&(key, edge))?.arguments.get(slot).copied()
            }
        }
    }
}
