use polytype::Type;

use crate::ast::TypeName;

use super::super::program::{OutputSlotId, PhysicalResourceRef};
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

    fn iter(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
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
pub enum Kind {
    Map,
    Reduce(NonEmpty<Operator>),
    Scan(NonEmpty<Operator>),
    Composite(NonEmpty<CompositeOperator>),
}

impl Kind {
    pub fn len(&self) -> usize {
        match self {
            Self::Map => 0,
            Self::Reduce(operators) | Self::Scan(operators) => 1 + operators.rest.len(),
            Self::Composite(operators) => 1 + operators.rest.len(),
        }
    }

    pub fn operator(&self, index: usize) -> Option<&Operator> {
        match self {
            Self::Map => None,
            Self::Reduce(operators) | Self::Scan(operators) => {
                (index == 0).then_some(&operators.first).or_else(|| operators.rest.get(index - 1))
            }
            Self::Composite(operators) => {
                let operator =
                    if index == 0 { Some(&operators.first) } else { operators.rest.get(index - 1) }?;
                Some(match operator {
                    CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => operator,
                })
            }
        }
    }

    pub fn operator_mut(&mut self, index: usize) -> Option<&mut Operator> {
        match self {
            Self::Map => None,
            Self::Reduce(operators) | Self::Scan(operators) => {
                if index == 0 {
                    Some(&mut operators.first)
                } else {
                    operators.rest.get_mut(index - 1)
                }
            }
            Self::Composite(operators) => {
                let operator = if index == 0 {
                    Some(&mut operators.first)
                } else {
                    operators.rest.get_mut(index - 1)
                }?;
                Some(match operator {
                    CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => operator,
                })
            }
        }
    }

    pub fn is_scan(&self, index: usize) -> bool {
        match self {
            Self::Scan(_) => index < self.len(),
            Self::Composite(operators) => {
                let operator =
                    if index == 0 { Some(&operators.first) } else { operators.rest.get(index - 1) };
                matches!(operator, Some(CompositeOperator::Scan(_)))
            }
            Self::Map | Self::Reduce(_) => false,
        }
    }

    pub fn operators(&self) -> Vec<&Operator> {
        match self {
            Self::Map => Vec::new(),
            Self::Reduce(operators) | Self::Scan(operators) => operators.iter().collect(),
            Self::Composite(operators) => operators
                .iter()
                .map(|operator| match operator {
                    CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => operator,
                })
                .collect(),
        }
    }

    pub fn operators_mut(&mut self) -> Vec<&mut Operator> {
        match self {
            Self::Map => Vec::new(),
            Self::Reduce(operators) | Self::Scan(operators) => operators.iter_mut().collect(),
            Self::Composite(operators) => operators
                .iter_mut()
                .map(|operator| match operator {
                    CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => operator,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Body {
    pub inputs: Vec<SoacInputType>,
    pub maps: Vec<Map>,
    pub kind: Kind,
}

impl Body {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.array);
            visit(&mut input.element);
        }
        for map in &mut self.maps {
            visit(&mut map.output_element_type);
            visit(&mut map.result_type);
        }
        for operator in self.kind.operators_mut() {
            visit(&mut operator.result_type);
        }
    }

    pub fn result_count(&self) -> usize {
        self.maps.len() + self.kind.len()
    }

    pub fn destination(&self, field: usize) -> Option<SoacDestination> {
        if let Some(map) = self.maps.get(field) {
            return Some(map.destination);
        }
        self.kind.operator(field - self.maps.len()).map(|operator| operator.destination)
    }

    pub fn set_destination(&mut self, field: usize, destination: SoacDestination) -> bool {
        if let Some(map) = self.maps.get_mut(field) {
            map.destination = destination;
            return true;
        }
        let Some(operator) = self.kind.operator_mut(field - self.maps.len()) else {
            return false;
        };
        operator.destination = destination;
        true
    }

    pub fn result_types(&self) -> Vec<Type<TypeName>> {
        self.maps
            .iter()
            .map(|map| map.result_type.clone())
            .chain(self.kind.operators().into_iter().map(|operator| operator.result_type.clone()))
            .collect()
    }

    pub fn set_result_types(&mut self, result_types: &[Type<TypeName>]) {
        assert_eq!(self.result_count(), result_types.len());
        let map_count = self.maps.len();
        for (map, result_type) in self.maps.iter_mut().zip(result_types) {
            map.result_type = result_type.clone();
        }
        for (operator, result_type) in self.kind.operators_mut().into_iter().zip(&result_types[map_count..])
        {
            operator.result_type = result_type.clone();
        }
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        let mut nodes =
            self.maps.iter().flat_map(|map| map.body.captures.iter().copied()).collect::<Vec<_>>();
        for index in 0..self.kind.len() {
            let operator = self.kind.operator(index).expect("operator index from kind length");
            nodes.extend(operator.step.captures.iter().copied());
            nodes.extend(operator.combine.captures.iter().copied());
        }
        nodes
    }

    fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.capture_nodes();
        for index in 0..self.kind.len() {
            let operator = self.kind.operator(index).expect("operator index from kind length");
            nodes.push(operator.neutral);
            nodes.extend(operator.shape.iter().copied());
        }
        nodes
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        fn append_operator<'a>(operator: &'a mut Operator, nodes: &mut Vec<&'a mut NodeId>) {
            nodes.extend(operator.step.captures.iter_mut());
            nodes.extend(operator.combine.captures.iter_mut());
            nodes.push(&mut operator.neutral);
            nodes.extend(operator.shape.iter_mut());
        }

        fn append_composite<'a>(operator: &'a mut CompositeOperator, nodes: &mut Vec<&'a mut NodeId>) {
            match operator {
                CompositeOperator::Reduce(operator) | CompositeOperator::Scan(operator) => {
                    append_operator(operator, nodes)
                }
            }
        }

        let mut nodes =
            self.maps.iter_mut().flat_map(|map| map.body.captures.iter_mut()).collect::<Vec<_>>();
        match &mut self.kind {
            Kind::Map => {}
            Kind::Reduce(operators) | Kind::Scan(operators) => {
                append_operator(&mut operators.first, &mut nodes);
                for operator in &mut operators.rest {
                    append_operator(operator, &mut nodes);
                }
            }
            Kind::Composite(operators) => {
                append_composite(&mut operators.first, &mut nodes);
                for operator in &mut operators.rest {
                    append_composite(operator, &mut nodes);
                }
            }
        }
        nodes
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
pub enum ScheduledState<R> {
    Serial,
    SegMap {
        space: SegSpace<R>,
        output_slots: Vec<OutputSlotId>,
        resources: Vec<SegResourceAccess<R>>,
    },
    SegRed {
        space: SegSpace<R>,
        output_slots: Vec<OutputSlotId>,
        resources: Vec<SegResourceAccess<R>>,
    },
    SegScan {
        space: SegSpace<R>,
        output_slots: Vec<OutputSlotId>,
        resources: Vec<SegResourceAccess<R>>,
    },
    SegComposite {
        space: SegSpace<R>,
        output_slots: Vec<OutputSlotId>,
        resources: Vec<SegResourceAccess<R>>,
    },
}

/// Executable Screma states. Parallel reductions, scans, and composites must
/// be split into physical kernels before this boundary; only an unsplit map
/// reaches the physical graph.
#[derive(Clone, Debug)]
pub enum PhysicalState {
    Serial,
    SegMap {
        space: SegSpace<PhysicalResourceRef>,
        output_slots: Vec<OutputSlotId>,
        resources: Vec<SegResourceAccess<PhysicalResourceRef>>,
    },
}

#[derive(Clone, Debug)]
pub struct Op<P: EgirPhase> {
    pub body: Body,
    pub state: P::ScremaState,
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        self.body.capture_nodes()
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.body.referenced_nodes();
        if let SemanticState::Segmented { space, .. } = &self.state {
            nodes.extend(space.referenced_nodes());
        }
        nodes
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { body, state } = self;
        let mut nodes = body.referenced_node_slots();
        if let SemanticState::Segmented { space, .. } = state {
            nodes.extend(space.referenced_node_slots());
        }
        nodes
    }
}
