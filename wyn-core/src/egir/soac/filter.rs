use polytype::Type;

use crate::ast::TypeName;

use super::super::program::{PhysicalResourceRef, SemanticResourceRef};
use super::super::types::{
    GraphResource, SegSpace, Semantic, SoacDestination, SoacInputType, ValueId, WynSoacPhase,
};
use super::screma;

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
pub struct Body {
    /// Co-iterated array inputs consumed by `map`.
    pub inputs: Vec<SoacInputType>,
    /// Computes the candidate output element from one element of each input.
    pub map: screma::Lambda,
    /// Decides whether the mapped candidate is retained.
    pub predicate: screma::Lambda,
}

impl Body {
    pub fn validate(&self) -> Result<(), String> {
        let input_types = self.inputs.iter().map(SoacInputType::element).collect::<Vec<_>>();
        if self.map.parameter_types != input_types {
            return Err(format!(
                "Filter map parameters {:?} do not match input element types {input_types:?}",
                self.map.parameter_types
            ));
        }
        if self.map.result_types.len() != 1 {
            return Err(format!(
                "Filter map must produce one candidate element, found {}",
                self.map.result_types.len()
            ));
        }
        if self.predicate.parameter_types != self.map.result_types {
            return Err(format!(
                "Filter predicate parameters {:?} do not match mapped element type {:?}",
                self.predicate.parameter_types, self.map.result_types
            ));
        }
        if self.predicate.result_types != [Type::Constructed(TypeName::Bool, Vec::new())] {
            return Err(format!(
                "Filter predicate must return bool, found {:?}",
                self.predicate.result_types
            ));
        }
        Ok(())
    }

    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.array);
        }
        self.map.for_each_type_mut(visit);
        self.predicate.for_each_type_mut(visit);
    }

    pub fn output_element_type(&self) -> Type<TypeName> {
        self.map.result_types[0].clone()
    }

    pub(crate) fn capture_nodes(&self) -> Vec<ValueId> {
        lambda_capture_values(&self.map).chain(lambda_capture_values(&self.predicate)).collect()
    }

    fn remap_capture_values(&mut self, map: &mut impl FnMut(ValueId) -> ValueId) {
        self.map.remap_capture_values(map);
        self.predicate.remap_capture_values(map);
    }
}

fn lambda_capture_values(lambda: &screma::Lambda) -> impl Iterator<Item = ValueId> + '_ {
    lambda.seg_body().into_iter().flat_map(|body| body.capture_values())
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
    pub(crate) fn capture_nodes(&self) -> Vec<ValueId> {
        self.body.capture_nodes()
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<ValueId> {
        let mut nodes = self.body.capture_nodes();
        nodes.extend(self.state.space.referenced_nodes());
        nodes
    }

    pub(crate) fn remap_referenced_values(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        let Self { body, state } = self;
        body.remap_capture_values(&mut map);
        for slot in state.space.referenced_node_slots() {
            *slot = map(*slot);
        }
    }
}
