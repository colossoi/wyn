use polytype::Type;

use crate::ast::TypeName;

use super::super::program::PhysicalResourceRef;
use super::super::types::{GraphResource, NodeId, SegSpace, Semantic, SoacInputType, WynSoacPhase};
use super::screma;

/// Per-bin update semantics. `OrderedOverwrite` is source-language scatter;
/// `Reduce` is Futhark-style histogram accumulation with an explicit neutral
/// value and associative operator.
#[derive(Clone, Debug)]
pub enum Update {
    OrderedOverwrite,
    Reduce {
        operator: screma::Lambda,
        neutral: NodeId,
    },
}

/// All `inputs` are co-iterated at one logical width. Semantic state carries
/// that width explicitly; bucket parameters correspond one-for-one to input
/// elements.
#[derive(Clone, Debug)]
pub struct Body {
    pub bucket: screma::Lambda,
    pub inputs: Vec<SoacInputType>,
    pub index_type: Type<TypeName>,
    pub value_type: Type<TypeName>,
    pub dest_elem_type: Type<TypeName>,
    pub update: Update,
}

impl Body {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.array);
        }
        self.bucket.for_each_type_mut(visit);
        visit(&mut self.index_type);
        visit(&mut self.value_type);
        visit(&mut self.dest_elem_type);
        if let Update::Reduce { operator, .. } = &mut self.update {
            operator.for_each_type_mut(visit);
        }
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self
            .bucket
            .seg_body()
            .into_iter()
            .flat_map(|body| body.captures.iter().copied())
            .collect::<Vec<_>>();
        if let Update::Reduce { operator, .. } = &self.update {
            nodes.extend(operator.seg_body().into_iter().flat_map(|body| body.captures.iter().copied()));
        }
        nodes
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { bucket, update, .. } = self;
        let mut nodes =
            bucket.seg_body_mut().into_iter().flat_map(|body| body.captures.iter_mut()).collect::<Vec<_>>();
        if let Update::Reduce { operator, neutral } = update {
            if let Some(body) = operator.seg_body_mut() {
                nodes.extend(body.captures.iter_mut());
            }
            nodes.push(neutral);
        }
        nodes
    }

    pub(crate) fn validate(&self, neutral_type: Option<&Type<TypeName>>) -> Result<(), String> {
        if self.inputs.is_empty() {
            return Err("histogram requires at least one input array".into());
        }
        let expected_bucket_parameters = self.inputs.iter().map(SoacInputType::element).collect::<Vec<_>>();
        let expected_bucket_results = vec![self.index_type.clone(), self.value_type.clone()];
        if self.bucket.parameter_types != expected_bucket_parameters
            || self.bucket.result_types != expected_bucket_results
        {
            return Err(format!(
                "histogram bucket lambda must have type {:?} -> {:?}, found {:?} -> {:?}",
                expected_bucket_parameters,
                expected_bucket_results,
                self.bucket.parameter_types,
                self.bucket.result_types,
            ));
        }
        let Update::Reduce { operator, .. } = &self.update else {
            return Ok(());
        };
        let expected_parameters = vec![self.dest_elem_type.clone(), self.value_type.clone()];
        let expected_results = vec![self.dest_elem_type.clone()];
        if operator.is_identity()
            || operator.parameter_types != expected_parameters
            || operator.result_types != expected_results
        {
            return Err(format!(
                "histogram reducer must have type ({:?}, {:?}) -> {:?}, found {:?} -> {:?}",
                self.dest_elem_type,
                self.value_type,
                self.dest_elem_type,
                operator.parameter_types,
                operator.result_types,
            ));
        }
        if neutral_type != Some(&self.dest_elem_type) {
            return Err(format!(
                "histogram neutral type {:?} does not match destination element type {:?}",
                neutral_type, self.dest_elem_type,
            ));
        }
        Ok(())
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
        if let Update::Reduce { neutral, .. } = &self.body.update {
            nodes.push(*neutral);
        }
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
