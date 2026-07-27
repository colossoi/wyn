use polytype::Type;

use crate::ast::TypeName;

use super::super::program::OutputSlotId;
use super::super::types::{
    GraphResource, NodeId, SegBody, SegResourceAccess, SegSpace, Semantic, SoacDestination, SoacInputType,
    WynSoacPhase,
};

/// One position in a Screma side effect's compact operand list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    pub node: NodeId,
    pub slot: usize,
}

/// A validated view of a Screma side effect's operands. The view borrows the
/// compact IR representation, so decoding adds no allocation or copied state.
#[derive(Clone, Copy, Debug)]
pub struct ScremaOperands<'a, P: WynSoacPhase> {
    op: &'a Op<P>,
    nodes: &'a [NodeId],
    result: NodeId,
}

impl<'a, P: WynSoacPhase> ScremaOperands<'a, P> {
    pub fn decode(op: &'a Op<P>, nodes: &'a [NodeId], result: Option<NodeId>) -> Result<Self, String> {
        let input_count = op.lanes().inputs.len();
        let output_count = (0..op.result_count())
            .filter(|&field| op.destination(field).is_some_and(SoacDestination::is_output_view))
            .count();
        let expected = input_count + output_count;
        if nodes.len() != expected {
            return Err(format!(
                "Screma requires {expected} typed input and output-view operands, found {}",
                nodes.len()
            ));
        }
        let result = result.ok_or_else(|| "Screma has no result node".to_owned())?;
        Ok(Self { op, nodes, result })
    }

    pub fn inputs(&self) -> impl Iterator<Item = Operand> + '_ {
        self.nodes[..self.input_count()]
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, node)| Operand { node, slot })
    }

    pub fn input(&self, slot: usize) -> Operand {
        Operand {
            node: self.nodes[slot],
            slot,
        }
    }

    pub fn input_count(&self) -> usize {
        self.op.lanes().inputs.len()
    }

    pub fn output(&self, field: usize) -> Option<Operand> {
        self.op.destination(field).filter(|destination| destination.is_output_view())?;
        let slot = self.input_count()
            + (0..field)
                .filter(|&field| self.op.destination(field).is_some_and(SoacDestination::is_output_view))
                .count();
        Some(Operand {
            node: self.nodes[slot],
            slot,
        })
    }

    pub fn outputs(&self) -> impl Iterator<Item = Option<Operand>> + '_ {
        (0..self.op.result_count()).map(|field| self.output(field))
    }

    pub fn result(&self) -> NodeId {
        self.result
    }
}

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

    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> NonEmpty<U> {
        NonEmpty {
            first: map(self.first),
            rest: self.rest.into_iter().map(map).collect(),
        }
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
    pub kind: OperatorKind,
    pub step: SegBody,
    pub combine: SegBody,
    pub input_indices: Vec<InputId>,
    pub neutral: NodeId,
    pub shape: Vec<NodeId>,
    pub commutative: bool,
    pub destination: SoacDestination,
    pub result_type: Type<TypeName>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorKind {
    Reduce,
    Scan,
}

impl Operator {
    pub fn is_scan(&self) -> bool {
        self.kind == OperatorKind::Scan
    }
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

#[derive(Clone, Debug)]
pub enum PhysicalState {
    Serial,
    Segmented(Segmented<super::super::program::PhysicalResourceRef>),
}

/// The canonical semantic form for map/scan/reduce compositions.
///
/// Operator kind is data rather than an outer representation variant. This
/// gives construction, traversal, fusion, and lowering one stable shape even
/// when scans and reductions are mixed.
#[derive(Clone, Debug)]
pub struct Op<P: WynSoacPhase> {
    pub lanes: Lanes,
    pub operators: Vec<Operator>,
    /// Per-element work evaluated after accumulator updates.
    pub post_maps: Vec<Map>,
    /// Scan operators consumed only by post-map work and omitted from top-level results.
    pub hidden_scan_outputs: Vec<usize>,
    pub state: P::ScremaState,
}

impl<P: WynSoacPhase> Op<P> {
    pub fn is_map(&self) -> bool {
        self.operators.is_empty()
    }

    pub fn is_reduce(&self) -> bool {
        !self.operators.is_empty() && self.operators.iter().all(|operator| !operator.is_scan())
    }

    pub fn is_scan_only(&self) -> bool {
        !self.operators.is_empty() && self.operators.iter().all(Operator::is_scan)
    }

    pub fn is_mixed(&self) -> bool {
        !self.is_map() && !self.is_reduce() && !self.is_scan_only()
    }

    pub fn lanes(&self) -> &Lanes {
        &self.lanes
    }

    pub fn lanes_mut(&mut self) -> &mut Lanes {
        &mut self.lanes
    }

    pub fn operators(&self) -> &[Operator] {
        &self.operators
    }

    pub fn operators_mut(&mut self) -> &mut [Operator] {
        &mut self.operators
    }

    pub fn is_scan(&self, index: usize) -> bool {
        self.operators.get(index).is_some_and(Operator::is_scan)
    }

    pub fn map_count(&self) -> usize {
        self.lanes.maps.len() + self.post_maps.len()
    }

    pub fn result_count(&self) -> usize {
        self.map_count() + self.visible_operator_count()
    }

    pub fn has_post_map(&self) -> bool {
        !self.post_maps.is_empty()
    }

    pub fn operator_is_output(&self, index: usize) -> bool {
        !self.hidden_scan_outputs.contains(&index)
    }

    fn visible_operator_count(&self) -> usize {
        (0..self.operators.len()).filter(|&index| self.operator_is_output(index)).count()
    }

    pub fn operator_index_for_field(&self, field: usize) -> Option<usize> {
        let visible = field.checked_sub(self.map_count())?;
        (0..self.operators.len()).filter(|&index| self.operator_is_output(index)).nth(visible)
    }

    pub fn result_field_for_operator(&self, operator_index: usize) -> Option<usize> {
        self.operator_is_output(operator_index).then(|| {
            self.map_count() + (0..operator_index).filter(|&index| self.operator_is_output(index)).count()
        })
    }

    fn map(&self, index: usize) -> Option<&Map> {
        self.lanes.maps.get(index).or_else(|| self.post_maps.get(index.checked_sub(self.lanes.maps.len())?))
    }

    fn map_mut(&mut self, index: usize) -> Option<&mut Map> {
        let pre_count = self.lanes.maps.len();
        if index < pre_count {
            self.lanes.maps.get_mut(index)
        } else {
            self.post_maps.get_mut(index - pre_count)
        }
    }

    pub fn destination(&self, field: usize) -> Option<SoacDestination> {
        let map_count = self.map_count();
        if field < map_count {
            return self.map(field).map(|map| map.destination);
        }
        self.operators
            .iter()
            .enumerate()
            .filter(|(index, _)| self.operator_is_output(*index))
            .nth(field - map_count)
            .map(|(_, operator)| operator.destination)
    }

    pub fn place_destination(
        &mut self,
        field: usize,
        placement: super::super::types::SoacPlacement,
    ) -> bool {
        let map_count = self.map_count();
        if field < map_count {
            self.map_mut(field).expect("map result field exists").destination.place(placement);
            return true;
        }
        let visible = field - map_count;
        let Some(index) =
            (0..self.operators.len()).filter(|&index| self.operator_is_output(index)).nth(visible)
        else {
            return false;
        };
        self.operators[index].destination.place(placement);
        true
    }

    pub fn result_types(&self) -> Vec<Type<TypeName>> {
        self.lanes
            .maps
            .iter()
            .chain(&self.post_maps)
            .map(|map| map.result_type.clone())
            .chain(
                self.operators
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| self.operator_is_output(*index))
                    .map(|(_, operator)| operator.result_type.clone()),
            )
            .collect()
    }

    pub fn set_result_types(&mut self, result_types: &[Type<TypeName>]) {
        assert_eq!(self.result_count(), result_types.len());
        let map_count = self.map_count();
        for (map, result_type) in self.lanes.maps.iter_mut().chain(&mut self.post_maps).zip(result_types) {
            map.result_type = result_type.clone();
        }
        let visible_indices = (0..self.operators.len())
            .filter(|&index| !self.hidden_scan_outputs.contains(&index))
            .collect::<Vec<_>>();
        for (index, result_type) in visible_indices.into_iter().zip(&result_types[map_count..]) {
            self.operators[index].result_type = result_type.clone();
        }
    }
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        self.lanes.for_each_type_mut(visit);
        for map in &mut self.post_maps {
            visit(&mut map.output_element_type);
            visit(&mut map.result_type);
        }
        for operator in &mut self.operators {
            visit(&mut operator.result_type);
        }
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self
            .lanes
            .maps
            .iter()
            .chain(&self.post_maps)
            .flat_map(|map| map.body.captures.iter().copied())
            .collect::<Vec<_>>();
        for operator in &self.operators {
            nodes.extend(operator.step.captures.iter().copied());
            nodes.extend(operator.combine.captures.iter().copied());
        }
        nodes
    }

    fn base_referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.capture_nodes();
        for operator in &self.operators {
            nodes.push(operator.neutral);
            nodes.extend(operator.shape.iter().copied());
        }
        nodes
    }
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub fn semantic_state(&self) -> &SemanticState<R> {
        &self.state
    }

    pub fn semantic_state_mut(&mut self) -> &mut SemanticState<R> {
        &mut self.state
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.base_referenced_nodes();
        if let SemanticState::Segmented { space, .. } = &self.state {
            nodes.extend(space.referenced_nodes());
        }
        nodes
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let mut nodes = self
            .lanes
            .maps
            .iter_mut()
            .chain(&mut self.post_maps)
            .flat_map(|map| map.body.captures.iter_mut())
            .collect::<Vec<_>>();
        for operator in &mut self.operators {
            nodes.extend(operator.step.captures.iter_mut());
            nodes.extend(operator.combine.captures.iter_mut());
            nodes.push(&mut operator.neutral);
            nodes.extend(operator.shape.iter_mut());
        }
        if let SemanticState::Segmented { space, .. } = &mut self.state {
            nodes.extend(space.referenced_node_slots());
        }
        nodes
    }
}

impl Op<super::super::types::Physical> {
    pub fn is_serial(&self) -> bool {
        matches!(self.state, PhysicalState::Serial)
    }
}
