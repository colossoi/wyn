use polytype::Type;

use crate::ast::TypeName;

use super::super::program::PhysicalResourceRef;
use super::super::types::{
    GraphResource, NodeId, SegBody, SegSpace, Semantic, SoacInputType, WynSoacPhase,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdatePolicy {
    OrderedOverwrite,
}

#[derive(Clone, Debug)]
pub struct Body {
    pub body: SegBody,
    pub inputs: Vec<SoacInputType>,
    pub index_type: Type<TypeName>,
    pub value_type: Type<TypeName>,
    pub dest_elem_type: Type<TypeName>,
    pub update_policy: UpdatePolicy,
}

impl Body {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.array);
        }
        visit(&mut self.index_type);
        visit(&mut self.value_type);
        visit(&mut self.dest_elem_type);
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        self.body.captures.clone()
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        self.body.captures.iter_mut().collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawState;

#[derive(Clone, Debug)]
pub enum State<R> {
    Serial,
    Segmented(SegSpace<R>),
}

pub type PhysicalState = State<PhysicalResourceRef>;

#[derive(Clone, Debug)]
pub struct Op<P: WynSoacPhase> {
    pub body: Body,
    pub state: P::HistState,
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        self.body.capture_nodes()
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.body.capture_nodes();
        if let State::Segmented(space) = &self.state {
            nodes.extend(space.referenced_nodes());
        }
        nodes
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { body, state } = self;
        let mut nodes = body.referenced_node_slots();
        if let State::Segmented(space) = state {
            nodes.extend(space.referenced_node_slots());
        }
        nodes
    }
}
