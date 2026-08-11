use polytype::Type;

use crate::ast::TypeName;

use super::super::program::PhysicalResourceRef;
use super::super::types::{GraphResource, NodeId, SegSpace, Semantic, SoacInputType, WynSoacPhase};
use super::screma;

/// How one histogram operation combines bucket values with its destinations.
///
/// `Reduce` is the canonical Futhark form. `OrderedOverwrite` is Wyn's
/// source-level scatter extension; keeping it per operation permits mixed,
/// multi-destination histograms without changing the surrounding form.
#[derive(Clone, Debug)]
pub enum Update {
    OrderedOverwrite {
        value_types: Vec<Type<TypeName>>,
    },
    Reduce {
        operator: screma::Lambda,
        neutral: Vec<NodeId>,
    },
    /// Capacity-bounded insertion. `counts` and `overflow` are storage-view
    /// nodes over compiler resources, which keeps resource identity in the
    /// graph even when every item input is produced by fused computation.
    BucketInsert {
        value_types: Vec<Type<TypeName>>,
        counts: NodeId,
        overflow: NodeId,
        capacity: NodeId,
    },
}

impl Update {
    pub(crate) fn value_types(&self) -> &[Type<TypeName>] {
        match self {
            Self::OrderedOverwrite { value_types } => value_types,
            Self::Reduce { operator, .. } => &operator.result_types,
            Self::BucketInsert { value_types, .. } => value_types,
        }
    }
}

/// One Futhark-style histogram operation.
///
/// Every value component shares the same multidimensional bucket index. The
/// destination list and value list are componentwise. `race_factor` is a
/// lowering hint and does not change serial semantics.
#[derive(Clone, Debug)]
pub struct HistOp {
    pub shape: Vec<NodeId>,
    pub race_factor: NodeId,
    pub destinations: Vec<NodeId>,
    pub update: Update,
}

impl HistOp {
    pub(crate) fn index_count(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn value_count(&self) -> usize {
        self.update.value_types().len()
    }
}

/// The phase-independent meaning of a histogram.
///
/// Inputs are co-iterated at one logical width. The bucket lambda receives one
/// element from each input and returns all operation indices first, followed by
/// all operation values. Operation order, and component order within each
/// operation, define both portions of that result ABI.
#[derive(Clone, Debug)]
pub struct HistForm {
    pub bucket: screma::Lambda,
    pub operations: Vec<HistOp>,
}

impl HistForm {
    pub(crate) fn index_count(&self) -> usize {
        self.operations.iter().map(HistOp::index_count).sum()
    }

    pub(crate) fn value_count(&self) -> usize {
        self.operations.iter().map(HistOp::value_count).sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawState;

#[derive(Clone, Debug)]
pub enum SemanticState<R> {
    Serial,
    Segmented(SegSpace<R>),
}

/// Parallel update selected for one histogram operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicUpdate {
    Direct(crate::ssa::types::AtomicOp),
    CompareExchange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelStage {
    Init,
    Insert,
    InsertTiled,
    Finish,
}

/// Target execution selected after reducer legality has been proven.
#[derive(Clone, Debug)]
pub enum ScheduledState<R> {
    Serial,
    Atomic {
        space: SegSpace<R>,
        operations: Vec<AtomicUpdate>,
    },
    Bucket {
        space: SegSpace<R>,
        stage: ParallelStage,
    },
}

pub type PhysicalState = ScheduledState<PhysicalResourceRef>;

#[derive(Clone, Debug)]
pub struct Op<P: WynSoacPhase> {
    pub inputs: Vec<SoacInputType>,
    pub form: HistForm,
    pub state: P::HistState,
}

impl<P: WynSoacPhase> Op<P> {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.array);
        }
        self.form.bucket.for_each_type_mut(visit);
        for operation in &mut self.form.operations {
            match &mut operation.update {
                Update::OrderedOverwrite { value_types } => {
                    for ty in value_types {
                        visit(ty);
                    }
                }
                Update::Reduce { operator, .. } => operator.for_each_type_mut(visit),
                Update::BucketInsert { value_types, .. } => {
                    for ty in value_types {
                        visit(ty);
                    }
                }
            }
        }
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.form.bucket.captures().to_vec();
        for operation in &self.form.operations {
            if let Update::Reduce { operator, .. } = &operation.update {
                nodes.extend(operator.captures());
            }
        }
        nodes
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.capture_nodes();
        for operation in &self.form.operations {
            nodes.extend(operation.shape.iter().copied());
            nodes.push(operation.race_factor);
            nodes.extend(operation.destinations.iter().copied());
            if let Update::Reduce { neutral, .. } = &operation.update {
                nodes.extend(neutral.iter().copied());
            }
            if let Update::BucketInsert {
                counts,
                overflow,
                capacity,
                ..
            } = operation.update
            {
                nodes.extend([counts, overflow, capacity]);
            }
        }
        nodes
    }

    pub(crate) fn validate(
        &self,
        mut node_type: impl FnMut(NodeId) -> Option<Type<TypeName>>,
    ) -> Result<(), String> {
        if self.inputs.is_empty() {
            return Err("histogram requires at least one input array".into());
        }
        if self.form.operations.is_empty() {
            return Err("histogram requires at least one operation".into());
        }

        let expected_parameters = self.inputs.iter().map(SoacInputType::element).collect::<Vec<_>>();
        let index_type = Type::Constructed(TypeName::Int(32), vec![]);
        let expected_results = std::iter::repeat_n(index_type, self.form.index_count())
            .chain(
                self.form
                    .operations
                    .iter()
                    .flat_map(|operation| operation.update.value_types().iter().cloned()),
            )
            .collect::<Vec<_>>();
        if self.form.bucket.parameter_types != expected_parameters
            || self.form.bucket.result_types != expected_results
        {
            return Err(format!(
                "histogram bucket lambda must have type {:?} -> {:?}, found {:?} -> {:?}",
                expected_parameters,
                expected_results,
                self.form.bucket.parameter_types,
                self.form.bucket.result_types,
            ));
        }

        for (index, operation) in self.form.operations.iter().enumerate() {
            if operation.destinations.len() != operation.value_count() {
                return Err(format!(
                    "histogram operation {index} has {} destinations for {} values",
                    operation.destinations.len(),
                    operation.value_count(),
                ));
            }
            for &dimension in &operation.shape {
                let ty = node_type(dimension);
                if ty.as_ref() != Some(&Type::Constructed(TypeName::Int(32), vec![])) {
                    return Err(format!(
                        "histogram operation {index} shape dimension must be i32, found {ty:?}"
                    ));
                }
            }
            let race_type = node_type(operation.race_factor);
            if race_type.as_ref() != Some(&Type::Constructed(TypeName::Int(32), vec![])) {
                return Err(format!(
                    "histogram operation {index} race factor must be i32, found {race_type:?}"
                ));
            }

            let value_types = operation.update.value_types();
            for (component, (&destination, value_type)) in
                operation.destinations.iter().zip(value_types).enumerate()
            {
                let destination_type = node_type(destination);
                let destination_element = destination_type.as_ref().and_then(|ty| {
                    let element = crate::types::array_elem(ty)?;
                    if matches!(operation.update, Update::BucketInsert { .. }) {
                        crate::types::array_elem(element)
                    } else {
                        Some(element)
                    }
                });
                if destination_element != Some(value_type) {
                    return Err(format!(
                        "histogram operation {index} destination {component} element type {:?} does not match value type {:?}",
                        destination_element, value_type,
                    ));
                }
            }

            let Update::Reduce { operator, neutral } = &operation.update else {
                continue;
            };
            if neutral.len() != value_types.len() {
                return Err(format!(
                    "histogram operation {index} has {} neutral values for {} components",
                    neutral.len(),
                    value_types.len(),
                ));
            }
            let expected_operator_parameters =
                value_types.iter().cloned().chain(value_types.iter().cloned()).collect::<Vec<_>>();
            if operator.is_identity()
                || operator.parameter_types != expected_operator_parameters
                || operator.result_types != value_types
            {
                return Err(format!(
                    "histogram operation {index} reducer must have type {:?} -> {:?}, found {:?} -> {:?}",
                    expected_operator_parameters,
                    value_types,
                    operator.parameter_types,
                    operator.result_types,
                ));
            }
            for (component, (&neutral, value_type)) in neutral.iter().zip(value_types).enumerate() {
                if node_type(neutral).as_ref() != Some(value_type) {
                    return Err(format!(
                        "histogram operation {index} neutral {component} does not have value type {value_type:?}"
                    ));
                }
            }
        }
        Ok(())
    }
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub(crate) fn referenced_nodes_with_state(&self) -> Vec<NodeId> {
        let mut nodes = self.referenced_nodes();
        if let SemanticState::Segmented(space) = &self.state {
            nodes.extend(space.referenced_nodes());
        }
        nodes
    }

    pub(crate) fn referenced_node_slots_with_state(&mut self) -> Vec<&mut NodeId> {
        let Self {
            inputs: _,
            form,
            state,
        } = self;
        let mut nodes = form_node_slots(form);
        if let SemanticState::Segmented(space) = state {
            nodes.extend(space.referenced_node_slots());
        }
        nodes
    }
}

fn form_node_slots(form: &mut HistForm) -> Vec<&mut NodeId> {
    let mut nodes = form
        .bucket
        .seg_body_mut()
        .into_iter()
        .flat_map(|body| body.captures.iter_mut())
        .collect::<Vec<_>>();
    for operation in &mut form.operations {
        nodes.extend(operation.shape.iter_mut());
        nodes.push(&mut operation.race_factor);
        nodes.extend(operation.destinations.iter_mut());
        match &mut operation.update {
            Update::Reduce { operator, neutral } => {
                if let Some(body) = operator.seg_body_mut() {
                    nodes.extend(body.captures.iter_mut());
                }
                nodes.extend(neutral.iter_mut());
            }
            Update::BucketInsert {
                counts,
                overflow,
                capacity,
                ..
            } => nodes.extend([counts, overflow, capacity]),
            Update::OrderedOverwrite { .. } => {}
        }
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egir::types::{Raw, RegionId, SegBody};
    use std::collections::HashMap;

    fn node(index: u64) -> NodeId {
        NodeId::from(slotmap::KeyData::from_ffi(index))
    }

    fn scalar(name: TypeName) -> Type<TypeName> {
        Type::Constructed(name, vec![])
    }

    fn array(element: Type<TypeName>) -> Type<TypeName> {
        Type::Constructed(
            TypeName::Array,
            vec![
                element,
                Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
                Type::Constructed(TypeName::Size(8), vec![]),
                crate::types::no_buffer(),
            ],
        )
    }

    fn general_histogram() -> (Op<Raw>, HashMap<NodeId, Type<TypeName>>) {
        let i32_type = scalar(TypeName::Int(32));
        let u32_type = scalar(TypeName::UInt(32));
        let f32_type = scalar(TypeName::Float(32));
        let bool_type = scalar(TypeName::Bool);
        let nodes = HashMap::from([
            (node(1), i32_type.clone()),
            (node(2), i32_type.clone()),
            (node(3), i32_type.clone()),
            (node(4), array(f32_type.clone())),
            (node(5), array(u32_type.clone())),
            (node(6), f32_type.clone()),
            (node(7), u32_type.clone()),
            (node(8), i32_type.clone()),
            (node(9), i32_type.clone()),
            (node(10), array(bool_type.clone())),
        ]);
        let op = Op::<Raw> {
            inputs: vec![SoacInputType::array(array(i32_type.clone()))],
            form: HistForm {
                bucket: screma::Lambda::region(
                    SegBody {
                        region: RegionId::from_index(0),
                        captures: vec![],
                    },
                    vec![i32_type.clone()],
                    vec![
                        i32_type.clone(),
                        i32_type.clone(),
                        i32_type,
                        f32_type.clone(),
                        u32_type.clone(),
                        bool_type.clone(),
                    ],
                ),
                operations: vec![
                    HistOp {
                        shape: vec![node(1), node(2)],
                        race_factor: node(3),
                        destinations: vec![node(4), node(5)],
                        update: Update::Reduce {
                            operator: screma::Lambda::region(
                                SegBody {
                                    region: RegionId::from_index(1),
                                    captures: vec![],
                                },
                                vec![
                                    f32_type.clone(),
                                    u32_type.clone(),
                                    f32_type.clone(),
                                    u32_type.clone(),
                                ],
                                vec![f32_type, u32_type],
                            ),
                            neutral: vec![node(6), node(7)],
                        },
                    },
                    HistOp {
                        shape: vec![node(8)],
                        race_factor: node(9),
                        destinations: vec![node(10)],
                        update: Update::OrderedOverwrite {
                            value_types: vec![bool_type],
                        },
                    },
                ],
            },
            state: RawState,
        };
        (op, nodes)
    }

    #[test]
    fn accepts_multiple_multidimensional_component_operations() {
        let (op, nodes) = general_histogram();
        op.validate(|node| nodes.get(&node).cloned())
            .expect("general Futhark-shaped histogram should validate");
        assert_eq!(op.form.index_count(), 3);
        assert_eq!(op.form.value_count(), 3);
    }

    #[test]
    fn bucket_results_put_all_indices_before_all_values() {
        let (mut op, nodes) = general_histogram();
        op.form.bucket.result_types.swap(2, 3);
        let error = op
            .validate(|node| nodes.get(&node).cloned())
            .expect_err("interleaving an operation value with indices must be rejected");
        assert!(error.contains("bucket lambda"), "unexpected error: {error}");
    }
}
