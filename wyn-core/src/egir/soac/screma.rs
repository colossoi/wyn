use polytype::Type;

use crate::ast::TypeName;

use super::super::program::OutputSlotId;
use super::super::types::{
    EgirPhase, GraphResource, NodeId, SegBody, SegResourceAccess, SegSpace, Semantic, SoacDestination,
    SoacInputType,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Kernel,
    LaneLocal,
}

#[derive(Clone, Debug)]
pub struct NonEmpty<T> {
    pub first: T,
    pub rest: Vec<T>,
}

impl<T> NonEmpty<T> {
    pub fn from_vec(values: Vec<T>) -> Option<Self> {
        let mut values = values.into_iter();
        Some(Self {
            first: values.next()?,
            rest: values.collect(),
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        std::iter::once(&mut self.first).chain(&mut self.rest)
    }
}

/// An index into `Body::inputs` used by a map or accumulator lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputId(pub usize);

impl InputId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct Map {
    pub body: SegBody,
    pub input_indices: Vec<InputId>,
    pub output_element_type: Type<TypeName>,
    pub destination: SoacDestination,
    pub result_type: Type<TypeName>,
}

#[derive(Clone, Debug)]
pub struct Operator {
    pub step: SegBody,
    pub combine: SegBody,
    pub input_indices: Vec<InputId>,
    pub neutral: NodeId,
    pub shape: Vec<NodeId>,
    pub commutative: bool,
    pub destination: SoacDestination,
    pub result_type: Type<TypeName>,
}

#[derive(Clone, Debug)]
pub enum CompositeOperator {
    Reduce(Operator),
    Scan(Operator),
}

#[derive(Clone, Debug)]
pub struct Lanes {
    pub inputs: Vec<SoacInputType>,
    pub maps: Vec<Map>,
}

impl Lanes {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.array);
        }
        for map in &mut self.maps {
            visit(&mut map.output_element_type);
            visit(&mut map.result_type);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawState;

#[derive(Clone, Debug)]
pub enum SemanticState<R> {
    Serial,
    Segmented {
        space: SegSpace<R>,
        placement: Placement,
        output_slots: Vec<OutputSlotId>,
        resources: Vec<SegResourceAccess<R>>,
    },
}

#[derive(Clone, Debug)]
pub struct Segmented<R> {
    pub space: SegSpace<R>,
    pub output_slots: Vec<OutputSlotId>,
    pub resources: Vec<SegResourceAccess<R>>,
}

#[derive(Clone, Debug)]
pub enum ScheduledState<R> {
    Serial,
    Segmented(Segmented<R>),
}

#[derive(Clone, Debug, Default)]
pub struct PhysicalSerialState;

#[derive(Clone, Debug)]
pub enum Op<P: EgirPhase> {
    Map {
        lanes: Lanes,
        state: P::MapState,
    },
    Reduce {
        lanes: Lanes,
        operators: NonEmpty<Operator>,
        state: P::ReduceState,
    },
    Scan {
        lanes: Lanes,
        operators: NonEmpty<Operator>,
        state: P::ScanState,
    },
    Composite {
        lanes: Lanes,
        operators: NonEmpty<CompositeOperator>,
        state: P::CompositeState,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    Map,
    Reduce,
    Scan,
    Composite,
}

impl<P: EgirPhase> Op<P> {
    pub fn flavor(&self) -> Flavor {
        match self {
            Self::Map { .. } => Flavor::Map,
            Self::Reduce { .. } => Flavor::Reduce,
            Self::Scan { .. } => Flavor::Scan,
            Self::Composite { .. } => Flavor::Composite,
        }
    }

    pub fn lanes(&self) -> &Lanes {
        match self {
            Self::Map { lanes, .. }
            | Self::Reduce { lanes, .. }
            | Self::Scan { lanes, .. }
            | Self::Composite { lanes, .. } => lanes,
        }
    }

    pub fn lanes_mut(&mut self) -> &mut Lanes {
        match self {
            Self::Map { lanes, .. }
            | Self::Reduce { lanes, .. }
            | Self::Scan { lanes, .. }
            | Self::Composite { lanes, .. } => lanes,
        }
    }

    pub fn operators(&self) -> Vec<&Operator> {
        match self {
            Self::Map { .. } => Vec::new(),
            Self::Reduce { operators, .. } | Self::Scan { operators, .. } => operators.iter().collect(),
            Self::Composite { operators, .. } => operators
                .iter()
                .map(|operator| match operator {
                    CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => operator,
                })
                .collect(),
        }
    }

    pub fn operators_mut(&mut self) -> Vec<&mut Operator> {
        match self {
            Self::Map { .. } => Vec::new(),
            Self::Reduce { operators, .. } | Self::Scan { operators, .. } => operators.iter_mut().collect(),
            Self::Composite { operators, .. } => operators
                .iter_mut()
                .map(|operator| match operator {
                    CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => operator,
                })
                .collect(),
        }
    }

    pub fn is_scan(&self, index: usize) -> bool {
        match self {
            Self::Scan { operators, .. } => index < 1 + operators.rest.len(),
            Self::Composite { operators, .. } => matches!(
                if index == 0 { Some(&operators.first) } else { operators.rest.get(index - 1) },
                Some(CompositeOperator::Scan(_))
            ),
            Self::Map { .. } | Self::Reduce { .. } => false,
        }
    }

    pub fn result_count(&self) -> usize {
        self.lanes().maps.len() + self.operators().len()
    }

    pub fn destination(&self, field: usize) -> Option<SoacDestination> {
        if let Some(map) = self.lanes().maps.get(field) {
            return Some(map.destination);
        }
        self.operators()
            .get(field.checked_sub(self.lanes().maps.len())?)
            .map(|operator| operator.destination)
    }

    pub fn place_destination(
        &mut self,
        field: usize,
        placement: super::super::types::SoacPlacement,
    ) -> bool {
        let map_count = self.lanes().maps.len();
        if field < map_count {
            self.lanes_mut().maps[field].destination.place(placement);
            return true;
        }
        let operator = field - map_count;
        let mut operators = self.operators_mut();
        let Some(operator) = operators.get_mut(operator) else {
            return false;
        };
        operator.destination.place(placement);
        true
    }

    pub fn result_types(&self) -> Vec<Type<TypeName>> {
        self.lanes()
            .maps
            .iter()
            .map(|map| map.result_type.clone())
            .chain(self.operators().into_iter().map(|operator| operator.result_type.clone()))
            .collect()
    }

    pub fn set_result_types(&mut self, result_types: &[Type<TypeName>]) {
        assert_eq!(self.result_count(), result_types.len());
        let map_count = self.lanes().maps.len();
        for (map, result_type) in self.lanes_mut().maps.iter_mut().zip(result_types) {
            map.result_type = result_type.clone();
        }
        for (operator, result_type) in self.operators_mut().into_iter().zip(&result_types[map_count..]) {
            operator.result_type = result_type.clone();
        }
    }

    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        match self {
            Self::Map { lanes, .. } => lanes.for_each_type_mut(visit),
            Self::Reduce { lanes, operators, .. } | Self::Scan { lanes, operators, .. } => {
                lanes.for_each_type_mut(visit);
                for operator in operators.iter_mut() {
                    visit(&mut operator.result_type);
                }
            }
            Self::Composite { lanes, operators, .. } => {
                lanes.for_each_type_mut(visit);
                for operator in operators.iter_mut() {
                    match operator {
                        CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => {
                            visit(&mut operator.result_type)
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        let mut nodes =
            self.lanes().maps.iter().flat_map(|map| map.body.captures.iter().copied()).collect::<Vec<_>>();
        for operator in self.operators() {
            nodes.extend(operator.step.captures.iter().copied());
            nodes.extend(operator.combine.captures.iter().copied());
        }
        nodes
    }

    fn base_referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.capture_nodes();
        for operator in self.operators() {
            nodes.push(operator.neutral);
            nodes.extend(operator.shape.iter().copied());
        }
        nodes
    }
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub fn semantic_state(&self) -> &SemanticState<R> {
        match self {
            Self::Map { state, .. }
            | Self::Reduce { state, .. }
            | Self::Scan { state, .. }
            | Self::Composite { state, .. } => state,
        }
    }

    pub fn semantic_state_mut(&mut self) -> &mut SemanticState<R> {
        match self {
            Self::Map { state, .. }
            | Self::Reduce { state, .. }
            | Self::Scan { state, .. }
            | Self::Composite { state, .. } => state,
        }
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.base_referenced_nodes();
        match self {
            Self::Map {
                state: SemanticState::Segmented { space, .. },
                ..
            }
            | Self::Reduce {
                state: SemanticState::Segmented { space, .. },
                ..
            }
            | Self::Scan {
                state: SemanticState::Segmented { space, .. },
                ..
            }
            | Self::Composite {
                state: SemanticState::Segmented { space, .. },
                ..
            } => nodes.extend(space.referenced_nodes()),
            _ => {}
        }
        nodes
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        match self {
            Self::Map { lanes, state } => {
                let mut nodes =
                    lanes.maps.iter_mut().flat_map(|map| map.body.captures.iter_mut()).collect::<Vec<_>>();
                if let SemanticState::Segmented { space, .. } = state {
                    nodes.extend(space.referenced_node_slots());
                }
                nodes
            }
            Self::Reduce {
                lanes,
                operators,
                state,
            }
            | Self::Scan {
                lanes,
                operators,
                state,
            } => {
                let mut nodes =
                    lanes.maps.iter_mut().flat_map(|map| map.body.captures.iter_mut()).collect::<Vec<_>>();
                for operator in operators.iter_mut() {
                    nodes.extend(operator.step.captures.iter_mut());
                    nodes.extend(operator.combine.captures.iter_mut());
                    nodes.push(&mut operator.neutral);
                    nodes.extend(operator.shape.iter_mut());
                }
                if let SemanticState::Segmented { space, .. } = state {
                    nodes.extend(space.referenced_node_slots());
                }
                nodes
            }
            Self::Composite {
                lanes,
                operators,
                state,
            } => {
                let mut nodes =
                    lanes.maps.iter_mut().flat_map(|map| map.body.captures.iter_mut()).collect::<Vec<_>>();
                for composite in operators.iter_mut() {
                    let operator = match composite {
                        CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => operator,
                    };
                    nodes.extend(operator.step.captures.iter_mut());
                    nodes.extend(operator.combine.captures.iter_mut());
                    nodes.push(&mut operator.neutral);
                    nodes.extend(operator.shape.iter_mut());
                }
                if let SemanticState::Segmented { space, .. } = state {
                    nodes.extend(space.referenced_node_slots());
                }
                nodes
            }
        }
    }
}

impl Op<super::super::types::Physical> {
    pub fn is_serial(&self) -> bool {
        matches!(
            self,
            Self::Map {
                state: ScheduledState::Serial,
                ..
            } | Self::Reduce { .. }
                | Self::Scan { .. }
                | Self::Composite { .. }
        )
    }
}
