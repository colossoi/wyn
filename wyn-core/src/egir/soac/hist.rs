use polytype::Type;

use crate::ast::TypeName;

use super::super::program::PhysicalResourceRef;
use super::super::types::{
    GraphResource, SegSpace, Semantic, SoacInputType, ValueId, ViewId, WynSoacPhase,
};
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
        neutral: Vec<ValueId>,
    },
    /// Capacity-bounded insertion. `counts` and `overflow` are storage-view
    /// nodes over compiler resources, which keeps resource identity in the
    /// graph even when every item input is produced by fused computation.
    BucketInsert {
        value_types: Vec<Type<TypeName>>,
        counts: ViewId,
        overflow: ViewId,
        capacity: ValueId,
    },
}

/// Whether one logical histogram emission participates in the update.
///
/// Guard values are produced by the bucket lambda before all indices and
/// values. Keeping this in canonical histogram IR means discard is not tied
/// to a sentinel key in the backend, and future filtered/expanded producers
/// can use the same mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emission {
    Always,
    Guarded,
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
    pub emission: Emission,
    pub shape: Vec<ValueId>,
    pub race_factor: ValueId,
    pub destinations: Vec<ViewId>,
    pub update: Update,
}

impl HistOp {
    pub(crate) fn guard_count(&self) -> usize {
        usize::from(matches!(self.emission, Emission::Guarded))
    }

    pub(crate) fn index_count(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn value_count(&self) -> usize {
        self.update.value_types().len()
    }

    pub(crate) fn written_views(&self) -> Vec<ViewId> {
        let mut views = self.destinations.clone();
        if let Update::BucketInsert { counts, overflow, .. } = self.update {
            views.extend([counts, overflow]);
        }
        views
    }
}

/// The phase-independent meaning of a histogram.
///
/// Inputs are co-iterated at one logical width. The bucket lambda receives one
/// element from each input and returns guards for guarded operations first,
/// followed by all operation indices and then all operation values. Operation
/// order, and component order within each operation, define every portion of
/// that result ABI.
#[derive(Clone, Debug)]
pub struct HistForm {
    pub bucket: screma::Lambda,
    pub operations: Vec<HistOp>,
}

impl HistForm {
    pub(crate) fn guard_count(&self) -> usize {
        self.operations.iter().map(HistOp::guard_count).sum()
    }

    pub(crate) fn index_count(&self) -> usize {
        self.operations.iter().map(HistOp::index_count).sum()
    }

    pub(crate) fn value_count(&self) -> usize {
        self.operations.iter().map(HistOp::value_count).sum()
    }

    pub(crate) fn written_views(&self) -> impl Iterator<Item = ViewId> + '_ {
        self.operations.iter().flat_map(HistOp::written_views)
    }

    fn remap_referenced_values(&mut self, map: &mut impl FnMut(ValueId) -> ValueId) {
        self.bucket.remap_capture_values(map);
        for operation in &mut self.operations {
            for dimension in &mut operation.shape {
                *dimension = map(*dimension);
            }
            operation.race_factor = map(operation.race_factor);
            for destination in &mut operation.destinations {
                destination.remap_value(&mut *map);
            }
            match &mut operation.update {
                Update::OrderedOverwrite { .. } => {}
                Update::Reduce { operator, neutral } => {
                    operator.remap_capture_values(map);
                    for value in neutral {
                        *value = map(*value);
                    }
                }
                Update::BucketInsert {
                    counts,
                    overflow,
                    capacity,
                    ..
                } => {
                    counts.remap_value(&mut *map);
                    overflow.remap_value(&mut *map);
                    *capacity = map(*capacity);
                }
            }
        }
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
    Finish,
}

/// A contiguous range of logical dimensions assigned to one physical
/// dispatch axis. Empty ranges represent an unused axis whose logical extent
/// is one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchAxis {
    pub start: usize,
    pub end: usize,
}

/// Target-checked mapping from a row-major logical domain to WebGPU's x/y/z
/// invocation grid. Axes are stored in physical x, y, z order; ranges are
/// contiguous, disjoint, and together cover every logical dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchTopology {
    pub axes: [DispatchAxis; 3],
    /// One physical axis may be strip-mined when its logical extent exceeds
    /// WebGPU's direct workgroup limit. Each invocation advances by this many
    /// logical axis items until the full axis has been visited.
    pub grid_stride: Option<GridStride>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridStride {
    pub axis: usize,
    pub items: u32,
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
        topology: Option<DispatchTopology>,
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

    pub(crate) fn capture_nodes(&self) -> Vec<ValueId> {
        let mut nodes =
            self.form.bucket.captures().iter().filter_map(|capture| capture.value()).collect::<Vec<_>>();
        for operation in &self.form.operations {
            if let Update::Reduce { operator, .. } = &operation.update {
                nodes.extend(operator.captures().iter().filter_map(|capture| capture.value()));
            }
        }
        nodes
    }

    pub(crate) fn remap_base_referenced_values(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        self.form.remap_referenced_values(&mut map);
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<ValueId> {
        let mut nodes = self.capture_nodes();
        for operation in &self.form.operations {
            nodes.extend(operation.shape.iter().copied());
            nodes.push(operation.race_factor);
            nodes.extend(operation.destinations.iter().map(|destination| destination.value()));
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
                nodes.extend([counts.value(), overflow.value(), capacity]);
            }
        }
        nodes
    }

    pub(crate) fn validate(
        &self,
        mut node_type: impl FnMut(ValueId) -> Option<Type<TypeName>>,
    ) -> Result<(), String> {
        if self.inputs.is_empty() {
            return Err("histogram requires at least one input array".into());
        }
        if self.form.operations.is_empty() {
            return Err("histogram requires at least one operation".into());
        }

        let expected_parameters = self.inputs.iter().map(SoacInputType::element).collect::<Vec<_>>();
        let index_type = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_type = Type::Constructed(TypeName::Bool, vec![]);
        let expected_results = std::iter::repeat_n(bool_type, self.form.guard_count())
            .chain(std::iter::repeat_n(index_type, self.form.index_count()))
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
            for (component, (destination, value_type)) in
                operation.destinations.iter().zip(value_types).enumerate()
            {
                let destination_type = node_type(destination.value());
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
    pub(crate) fn referenced_nodes_with_state(&self) -> Vec<ValueId> {
        let mut nodes = self.referenced_nodes();
        if let SemanticState::Segmented(space) = &self.state {
            nodes.extend(space.referenced_nodes());
        }
        nodes
    }

    pub(crate) fn remap_referenced_values(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        self.form.remap_referenced_values(&mut map);
        if let SemanticState::Segmented(space) = &mut self.state {
            for slot in space.referenced_node_slots() {
                *slot = map(*slot);
            }
        }
    }
}

#[cfg(test)]
#[path = "hist_tests.rs"]
mod tests;
