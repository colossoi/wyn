use polytype::Type;

use crate::ast::TypeName;

use super::super::program::{PhysicalResourceRef, SemanticResourceRef};
use super::super::types::{
    GraphResource, NodeId, SegBody, SegSpace, Semantic, SoacDestination, SoacInputType, WynSoacPhase,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkBuffers<R = SemanticResourceRef> {
    pub flags: R,
    pub offsets: R,
    pub block_sums: R,
    pub block_offsets: R,
}

#[derive(Clone, Debug)]
pub enum Output<R = SemanticResourceRef> {
    Local {
        capacity: Type<TypeName>,
        destination: SoacDestination,
    },
    Runtime {
        scratch: R,
        length: RuntimeLength<R>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeLength<R = SemanticResourceRef> {
    ViewOnly,
    /// Logical length stored in a scalar resource. Public filter outputs and
    /// compiler-internal runtime-array handoffs use the same representation;
    /// publication decides whether the resource belongs to the host ABI.
    Stored(R),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan<R = SemanticResourceRef> {
    Loop,
    Flags(ParallelConfig<R>),
    Scan(ParallelConfig<R>),
    Scatter(ParallelConfig<R>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelConfig<R = SemanticResourceRef> {
    pub buffers: WorkBuffers<R>,
    /// Width selected when the recipe was planned. Expansion consumes this
    /// value instead of consulting parallelization policy again.
    pub scan_workgroup_width: u32,
}

#[derive(Clone, Debug)]
pub enum Input {
    Plain(SoacInputType),
    Mapped {
        input: SoacInputType,
        body: SegBody,
        output_element_type: Type<TypeName>,
    },
}

#[derive(Clone, Debug)]
pub struct Body {
    pub input: Input,
    pub predicate: SegBody,
}

impl Body {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        let input = match &mut self.input {
            Input::Plain(input) => input,
            Input::Mapped {
                input,
                output_element_type,
                ..
            } => {
                visit(output_element_type);
                input
            }
        };
        visit(&mut input.array);
    }

    pub fn output_element_type(&self) -> Type<TypeName> {
        match &self.input {
            Input::Plain(input) => input.element(),
            Input::Mapped {
                output_element_type, ..
            } => output_element_type.clone(),
        }
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        let mut nodes = match &self.input {
            Input::Plain(_) => Vec::new(),
            Input::Mapped { body, .. } => body.captures.clone(),
        };
        nodes.extend(self.predicate.captures.iter().copied());
        nodes
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { input, predicate } = self;
        let mut nodes = match input {
            Input::Plain(_) => Vec::new(),
            Input::Mapped { body, .. } => body.captures.iter_mut().collect(),
        };
        nodes.extend(predicate.captures.iter_mut());
        nodes
    }
}

#[derive(Clone, Debug)]
pub struct RawState<R> {
    pub storage: Output<R>,
}

impl<R> RawState<R> {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        if let Output::Local { capacity, .. } = &mut self.storage {
            visit(capacity);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SemanticState<R> {
    pub space: SegSpace<R>,
    pub storage: Output<R>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeStorage<R> {
    pub scratch: R,
    pub length: RuntimeLength<R>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelStage {
    Flags,
    Scan,
    Scatter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelPlan<R> {
    pub stage: ParallelStage,
    pub buffers: WorkBuffers<R>,
    pub scan_workgroup_width: u32,
}

#[derive(Clone, Debug)]
pub enum ScheduledState<R> {
    Loop {
        space: SegSpace<R>,
        storage: Output<R>,
    },
    Pipeline {
        space: SegSpace<R>,
        storage: RuntimeStorage<R>,
        plan: ParallelPlan<R>,
    },
}

impl<R> ScheduledState<R> {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        if let Self::Loop {
            storage: Output::Local { capacity, .. },
            ..
        } = self
        {
            visit(capacity);
        }
    }
}

pub type PhysicalState = ScheduledState<PhysicalResourceRef>;

#[derive(Clone, Debug)]
pub struct Op<P: WynSoacPhase> {
    pub body: Body,
    pub state: P::FilterState,
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        self.body.capture_nodes()
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.body.capture_nodes();
        nodes.extend(self.state.space.referenced_nodes());
        nodes
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { body, state } = self;
        let mut nodes = body.referenced_node_slots();
        nodes.extend(state.space.referenced_node_slots());
        nodes
    }
}
