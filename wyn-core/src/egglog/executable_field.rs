//! Read executable fields from the Rust graph in planning rules.
use super::graph::{callback_at, CollectiveInput, Formula, SharedStructure};
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::{ArcSort, Core, Primitive, PurePrim, PureState, Value};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) enum Field {
    Region,
    Order,
    FunctionRoot,
    External,
    Entry,
    CallTarget,
    ControlRegion,
    Reducer,
    Combiner,
    EdgeArgument,
    Input,
    InputCount,
    Output,
    OutputCount,
    Formula(Formula),
    IterationDomain,
    DomainRank,
    DomainAxis,
    DomainOrdinal,
    DomainExtent,
    ResultKind,
    ResultCount,
    OutputClause,
    OutputField,
    OutputRole,
    AccumulatorPolicy,
    AccumulatorGroupCount,
    Collective(CollectiveInput),
    CollectiveCount(CollectiveInput),
    Yield,
    YieldCount,
    Parameter,
    ParameterCount,
    ControlYield,
    Yielded,
    RegionYields,
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
            Field::External => "wyn-external",
            Field::Entry => "wyn-entry",
            Field::CallTarget => "wyn-call-target",
            Field::ControlRegion => "wyn-control-region",
            Field::Reducer => "wyn-reducer",
            Field::Combiner => "wyn-combiner",
            Field::EdgeArgument => "wyn-edge-argument",
            Field::Input => "wyn-input",
            Field::InputCount => "wyn-input-count",
            Field::Output => "wyn-output",
            Field::OutputCount => "wyn-output-count",
            Field::Formula(kind) => kind.access(),
            Field::IterationDomain => "wyn-iteration-domain",
            Field::DomainRank => "wyn-domain-rank",
            Field::DomainAxis => "wyn-domain-axis",
            Field::DomainOrdinal => "wyn-domain-ordinal",
            Field::DomainExtent => "wyn-domain-extent",
            Field::ResultKind => "wyn-result-kind",
            Field::ResultCount => "wyn-result-count",
            Field::OutputClause => "wyn-output-clause",
            Field::OutputField => "wyn-output-field",
            Field::OutputRole => "wyn-output-role",
            Field::AccumulatorPolicy => "wyn-accumulator-policy",
            Field::AccumulatorGroupCount => "wyn-accumulator-group-count",
            Field::Collective(CollectiveInput::Contribution) => "wyn-contribution",
            Field::CollectiveCount(CollectiveInput::Contribution) => "wyn-contribution-count",
            Field::Collective(CollectiveInput::Neutral) => "wyn-fold-seed",
            Field::CollectiveCount(CollectiveInput::Neutral) => "wyn-fold-seed-count",
            Field::Collective(CollectiveInput::Predicate) => "wyn-pack-predicate",
            Field::CollectiveCount(CollectiveInput::Predicate) => "wyn-pack-predicate-count",
            Field::Collective(CollectiveInput::Backing) => "wyn-accumulator-backing",
            Field::CollectiveCount(CollectiveInput::Backing) => "wyn-accumulator-backing-count",
            Field::Yield => "wyn-yield",
            Field::YieldCount => "wyn-yield-count",
            Field::Parameter => "wyn-parameter",
            Field::ParameterCount => "wyn-parameter-count",
            Field::ControlYield => "wyn-control-yield",
            Field::Yielded => "wyn-is-yielded",
            Field::RegionYields => "wyn-region-yields",
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
            Field::External => Some(state.base_to_value(graph.externals.contains_key(&key))),
            Field::Entry => Some(state.base_to_value(graph.entries.contains_key(&key))),
            Field::CallTarget => graph.targets.get(&key).copied(),
            Field::Reducer => callback_at(&graph.reducers, key, state.value_to_base(args[1])),
            Field::Combiner => callback_at(&graph.combiners, key, state.value_to_base(args[1])),
            Field::ControlRegion => {
                let slot: i64 = state.value_to_base(args[1]);
                graph.controls.get(&(key, slot)).map(|edge| edge.region)
            }
            Field::Input => {
                let slot = usize::try_from(state.value_to_base::<i64>(args[1])).ok()?;
                graph.inputs.values.get(&key)?.get(slot).copied()
            }
            Field::InputCount => {
                let count = graph.inputs.values.get(&key).map_or(0, |values| values.len() as i64);
                Some(state.base_to_value(count))
            }
            Field::Output => {
                let slot = usize::try_from(state.value_to_base::<i64>(args[1])).ok()?;
                graph.outputs.get(&key)?.get(slot).copied()
            }
            Field::Collective(kind) => {
                let group = state.value_to_base(args[1]);
                let field = if kind == CollectiveInput::Predicate {
                    0
                } else {
                    usize::try_from(state.value_to_base::<i64>(args[2])).ok()?
                };
                graph.collective.get(key, group, kind).get(field).copied()
            }
            Field::CollectiveCount(kind) => {
                let group = state.value_to_base(args[1]);
                let count = graph.collective.get(key, group, kind).len() as i64;
                Some(state.base_to_value(count))
            }
            Field::Formula(kind) => graph.formulas.get(key, kind),
            Field::IterationDomain => graph.domains.operations.get(&key).copied(),
            Field::DomainRank => {
                Some(state.base_to_value(graph.domains.values.get(&key)?.dimensions.len() as i64))
            }
            Field::DomainOrdinal => graph.domains.values.get(&key)?.ordinal,
            Field::DomainAxis | Field::DomainExtent => {
                let slot = usize::try_from(state.value_to_base::<i64>(args[1])).ok()?;
                let dimension = graph.domains.values.get(&key)?.dimensions.get(slot)?;
                if matches!(self.field, Field::DomainAxis) {
                    dimension.coordinate
                } else {
                    dimension.extent
                }
            }
            Field::ResultCount => Some(state.base_to_value(graph.parallels.get(&key)?.kinds.len() as i64)),
            Field::AccumulatorGroupCount => {
                Some(state.base_to_value(graph.accumulator_policies.get(&key)?.len() as i64))
            }
            Field::ResultKind
            | Field::AccumulatorPolicy
            | Field::OutputClause
            | Field::OutputField
            | Field::OutputRole => {
                let slot = usize::try_from(state.value_to_base::<i64>(args[1])).ok()?;
                match self.field {
                    Field::ResultKind => graph.parallels.get(&key)?.kinds.get(slot).copied().flatten(),
                    Field::AccumulatorPolicy => {
                        graph.accumulator_policies.get(&key)?.get(slot).copied().flatten()
                    }
                    field => {
                        let binding = graph.parallels.get(&key)?.bindings.get(slot)?;
                        match field {
                            Field::OutputClause => Some(state.base_to_value(binding.clause?)),
                            Field::OutputField => Some(state.base_to_value(binding.field?)),
                            _ => binding.role,
                        }
                    }
                }
            }
            Field::OutputCount => {
                let count = graph.outputs.get(&key).map_or(0, |names| names.len() as i64);
                Some(state.base_to_value(count))
            }
            Field::Parameter => {
                let slot = usize::try_from(state.value_to_base::<i64>(args[1])).ok()?;
                graph.parameters.get(&key)?.get(slot).copied()
            }
            Field::ParameterCount => {
                let count = graph.parameters.get(&key).map_or(0, |names| names.len() as i64);
                Some(state.base_to_value(count))
            }
            Field::Yield => {
                let slot = usize::try_from(state.value_to_base::<i64>(args[1])).ok()?;
                graph.yields.values.get(&key)?.get(slot).copied()
            }
            Field::YieldCount => {
                let count = graph.yields.values.get(&key).map_or(0, |values| values.len() as i64);
                Some(state.base_to_value(count))
            }
            Field::Yielded => Some(state.base_to_value(graph.yields.uses.contains_key(&key))),
            Field::RegionYields => {
                let yields = graph.yields.values.get(&key);
                Some(state.base_to_value(yields.is_some_and(|values| values.contains(&args[1]))))
            }
            Field::ControlYield => {
                let edge = state.value_to_base::<i64>(args[1]);
                let slot = usize::try_from(state.value_to_base::<i64>(args[2])).ok()?;
                let region = graph.controls.get(&(key, edge))?.region;
                graph.yields.values.get(&region)?.get(slot).copied()
            }
            Field::EdgeArgument => {
                let edge: i64 = state.value_to_base(args[1]);
                let slot = usize::try_from(state.value_to_base::<i64>(args[2])).ok()?;
                graph.controls.get(&(key, edge))?.arguments.get(slot).copied()
            }
        }
    }
}
