use polytype::Type;

use crate::ast::TypeName;

use super::super::program::{PhysicalResourceRef, SemanticResourceRef};
use super::super::types::{
    GraphResource, NodeId, SegBody, SegSpace, Semantic, SoacInputType, WynSoacPhase,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdatePolicy<R = SemanticResourceRef> {
    OrderedOverwrite,
    /// Allocate a distinct slot with `atomicAdd(counts[key], 1)`, then write
    /// the value when the returned slot is below `capacity`. `overflow` is a
    /// one-element u32 resource set to one for invalid keys or full buckets.
    BucketInsert {
        counts: R,
        overflow: R,
        bucket_count: NodeId,
        capacity: NodeId,
        /// Number of nested input-array dimensions to flatten.
        input_rank: u8,
    },
}

#[derive(Clone, Debug)]
pub struct Body<R = SemanticResourceRef> {
    pub body: SegBody,
    pub inputs: Vec<SoacInputType>,
    pub index_type: Type<TypeName>,
    pub value_type: Type<TypeName>,
    pub dest_elem_type: Type<TypeName>,
    pub update_policy: UpdatePolicy<R>,
}

impl<R> Body<R> {
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

    fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.capture_nodes();
        if let UpdatePolicy::BucketInsert {
            bucket_count,
            capacity,
            ..
        } = self.update_policy
        {
            nodes.extend([bucket_count, capacity]);
        }
        nodes
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let mut nodes = self.body.captures.iter_mut().collect::<Vec<_>>();
        if let UpdatePolicy::BucketInsert {
            bucket_count,
            capacity,
            ..
        } = &mut self.update_policy
        {
            nodes.extend([bucket_count, capacity]);
        }
        nodes
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelStage {
    Init,
    Insert,
    Finish,
}

#[derive(Clone, Debug)]
pub enum State<R> {
    Serial,
    Segmented(SegSpace<R>),
    Pipeline {
        space: SegSpace<R>,
        stage: ParallelStage,
    },
}

pub type PhysicalState = State<PhysicalResourceRef>;

#[derive(Clone, Debug)]
pub struct Op<P: WynSoacPhase> {
    pub body: Body<P::Resource>,
    pub state: P::HistState,
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        self.body.capture_nodes()
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.body.referenced_nodes();
        match &self.state {
            State::Segmented(space) | State::Pipeline { space, .. } => {
                nodes.extend(space.referenced_nodes());
            }
            State::Serial => {}
        }
        nodes
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { body, state } = self;
        let mut nodes = body.referenced_node_slots();
        match state {
            State::Segmented(space) | State::Pipeline { space, .. } => {
                nodes.extend(space.referenced_node_slots());
            }
            State::Serial => {}
        }
        nodes
    }
}
