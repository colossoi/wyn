use crate::BindingRef;
use polytype::Type;

use super::metadata::{impl_metadata, Metadata};
use super::Lambda;
use crate::ast::TypeName;

use super::super::program::OutputSlotId;
use super::super::types::{
    GraphResource, OperandRef, ResultBinding, SegResourceAccess, SegSpace, Semantic, SoacInputType,
    SoacOwnership, ValueId, WynSoacPhase,
};

/// One position in a Screma side effect's compact operand list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    pub operand: OperandRef,
    pub slot: usize,
}

/// A validated view of a Screma side effect's compact operands.
#[derive(Clone, Debug)]
pub struct ScremaOperands<'a, P: WynSoacPhase> {
    op: &'a Op<P>,
    operands: &'a [OperandRef],
    result: &'a ResultBinding<Type<TypeName>>,
}

impl<'a, P: WynSoacPhase> ScremaOperands<'a, P> {
    pub fn decode(
        op: &'a Op<P>,
        operands: &'a [OperandRef],
        result: Option<&'a ResultBinding<Type<TypeName>>>,
    ) -> Result<Self, String> {
        op.validate()?;
        let expected = op.inputs.len();
        if operands.len() != expected {
            return Err(format!(
                "Screma requires {expected} typed input operands, found {}",
                operands.len()
            ));
        }
        let result = result.ok_or_else(|| "Screma has no result binding".to_owned())?;
        if result.field_count() != op.result_count() {
            return Err(format!(
                "Screma produces {} logical results, but its result binding has {} fields",
                op.result_count(),
                result.field_count()
            ));
        }
        Ok(Self { op, operands, result })
    }

    pub fn input_count(&self) -> usize {
        self.op.inputs.len()
    }

    pub fn inputs(&self) -> impl Iterator<Item = Operand> + '_ {
        self.operands[..self.input_count()]
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, operand)| Operand { operand, slot })
    }

    pub fn input(&self, slot: usize) -> Operand {
        Operand {
            operand: self.operands[slot],
            slot,
        }
    }

    pub fn result(&self) -> &ResultBinding<Type<TypeName>> {
        self.result
    }

    pub fn result_fields(&self) -> Vec<ResultBinding<Type<TypeName>>> {
        self.result.top_level_fields()
    }
}
/// One associative scan operator at the Screma's collective barrier.
#[derive(Clone, Debug)]
pub struct Scan {
    pub operator: Lambda,
    pub neutral: Vec<ValueId>,
}

/// One associative reduction operator at the Screma's collective barrier.
#[derive(Clone, Debug)]
pub struct Reduce {
    pub operator: Lambda,
    pub neutral: Vec<ValueId>,
    pub commutative: bool,
}

/// The phase-independent meaning of a Screma.
///
/// This follows Futhark's `ScremaForm`: one pre-lambda, sibling scans and
/// reductions, and one post-lambda. Results from `pre` are ordered as scan
/// inputs, reduction inputs, then ordinary mapped values. The post-lambda
/// receives scan results followed by ordinary mapped values. Reduction results
/// bypass the post-lambda.
#[derive(Clone, Debug)]
pub struct ScremaForm {
    pub pre: Lambda,
    pub scans: Vec<Scan>,
    pub reductions: Vec<Reduce>,
    pub post: Lambda,
}

impl_metadata!(ScremaForm, |form, visit| {
    let Self {
        pre,
        scans,
        reductions,
        post,
    } = form;
    visit.lambda(pre);
    for Scan { operator, neutral } in scans {
        visit.lambda(operator);
        for value in neutral {
            visit.value(value);
        }
    }
    for Reduce {
        operator, neutral, ..
    } in reductions
    {
        visit.lambda(operator);
        for value in neutral {
            visit.value(value);
        }
    }
    visit.lambda(post);
});

/// A result position derived from Futhark's fixed Screma result convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResultId {
    Reduction {
        reduction: usize,
        component: usize,
    },
    Post(usize),
}
impl ScremaForm {
    pub fn scan_count(&self) -> usize {
        self.scans.len()
    }

    pub fn reduction_count(&self) -> usize {
        self.reductions.len()
    }

    pub fn scan_input_count(&self) -> usize {
        self.scans.iter().map(|scan| scan.neutral.len()).sum()
    }

    pub fn reduction_input_count(&self) -> usize {
        self.reductions.iter().map(|reduction| reduction.neutral.len()).sum()
    }

    pub fn operator_input_count(&self) -> usize {
        self.scan_input_count() + self.reduction_input_count()
    }

    pub fn reduction_result_count(&self) -> usize {
        self.reduction_input_count()
    }

    /// The ordinary mapped values returned after operator inputs by `pre`.
    pub fn mapped_types(&self) -> Option<&[Type<TypeName>]> {
        self.pre.result_types.get(self.operator_input_count()..)
    }

    /// Post-lambda parameters: scan results first, then mapped values.
    pub fn post_input_types(&self) -> Option<Vec<Type<TypeName>>> {
        let mapped = self.mapped_types()?;
        Some(
            self.scans
                .iter()
                .flat_map(|scan| scan.operator.result_types.iter().cloned())
                .chain(mapped.iter().cloned())
                .collect(),
        )
    }

    /// Futhark result order: reduction scalars, then post-lambda arrays.
    pub fn result_count(&self) -> usize {
        self.reduction_result_count() + self.post.result_types.len()
    }

    pub fn result_id(&self, field: usize) -> Option<ResultId> {
        let mut offset = 0;
        for (reduction, operator) in self.reductions.iter().enumerate() {
            let end = offset + operator.neutral.len();
            if field < end {
                return Some(ResultId::Reduction {
                    reduction,
                    component: field - offset,
                });
            }
            offset = end;
        }
        let post = field.checked_sub(offset)?;
        (post < self.post.result_types.len()).then_some(ResultId::Post(post))
    }

    pub fn result_element_type(&self, field: usize) -> Option<&Type<TypeName>> {
        match self.result_id(field)? {
            ResultId::Reduction { reduction, component } => {
                self.reductions[reduction].operator.result_types.get(component)
            }
            ResultId::Post(index) => self.post.result_types.get(index),
        }
    }

    pub fn validate(&self, input_types: &[SoacInputType]) -> Result<(), String> {
        self.pre.validate("Screma pre")?;
        self.post.validate("Screma post")?;

        let pre_parameter_types = input_types.iter().map(SoacInputType::element).collect::<Vec<_>>();
        if self.pre.parameter_types != pre_parameter_types {
            return Err(format!(
                "Screma pre-lambda parameters {:?} do not match input element types {:?}",
                self.pre.parameter_types, pre_parameter_types
            ));
        }

        let operator_inputs = self.operator_input_count();
        if self.pre.result_types.len() < operator_inputs {
            return Err(format!(
                "Screma pre-lambda returns {} values, but {} scan/reduction inputs are required",
                self.pre.result_types.len(),
                operator_inputs
            ));
        }

        let mut pre_offset = 0;
        for (index, scan) in self.scans.iter().enumerate() {
            scan.operator.validate_operator(&format!("Screma scan {index}"), scan.neutral.len())?;
            let end = pre_offset + scan.neutral.len();
            let expected = &self.pre.result_types[pre_offset..end];
            if scan.operator.result_types != expected {
                return Err(format!(
                    "Screma scan {index} operates on {:?}, but pre-lambda results {pre_offset}..{end} are {expected:?}",
                    scan.operator.result_types
                ));
            }
            pre_offset = end;
        }

        for (index, reduction) in self.reductions.iter().enumerate() {
            reduction
                .operator
                .validate_operator(&format!("Screma reduction {index}"), reduction.neutral.len())?;
            let end = pre_offset + reduction.neutral.len();
            let expected = &self.pre.result_types[pre_offset..end];
            if reduction.operator.result_types != expected {
                return Err(format!(
                    "Screma reduction {index} operates on {:?}, but pre-lambda results {pre_offset}..{end} are {expected:?}",
                    reduction.operator.result_types
                ));
            }
            pre_offset = end;
        }

        if self.scans.is_empty() && !self.post.is_identity() {
            return Err("Screma has a non-identity post-lambda but no scans".to_owned());
        }
        let expected_post_parameters = self.post_input_types().expect("pre-lambda arity checked");
        if self.post.parameter_types != expected_post_parameters {
            return Err(format!(
                "Screma post-lambda parameters {:?} do not match scan/map element types {:?}",
                self.post.parameter_types, expected_post_parameters
            ));
        }

        Ok(())
    }

    fn validate_neutral_types(
        &self,
        node_type: &mut impl FnMut(ValueId) -> Option<Type<TypeName>>,
    ) -> Result<(), String> {
        for (index, scan) in self.scans.iter().enumerate() {
            validate_neutral_values(
                "scan",
                index,
                &scan.neutral,
                &scan.operator.result_types,
                node_type,
            )?;
        }
        for (index, reduction) in self.reductions.iter().enumerate() {
            validate_neutral_values(
                "reduction",
                index,
                &reduction.neutral,
                &reduction.operator.result_types,
                node_type,
            )?;
        }
        Ok(())
    }
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        self.pre.for_each_type_mut(visit);
        for scan in &mut self.scans {
            scan.operator.for_each_type_mut(visit);
        }
        for reduction in &mut self.reductions {
            reduction.operator.for_each_type_mut(visit);
        }
        self.post.for_each_type_mut(visit);
    }

    fn base_referenced_nodes(&self) -> Vec<ValueId> {
        self.metadata_values()
    }

    fn remap_referenced_values(&mut self, map: &mut impl FnMut(ValueId) -> ValueId) {
        self.remap_metadata_values(map);
    }
}

fn validate_neutral_values(
    kind: &str,
    index: usize,
    neutral: &[ValueId],
    types: &[Type<TypeName>],
    node_type: &mut impl FnMut(ValueId) -> Option<Type<TypeName>>,
) -> Result<(), String> {
    for (component, (&node, expected)) in neutral.iter().zip(types).enumerate() {
        let actual = node_type(node);
        if actual.as_ref() != Some(expected) {
            return Err(format!(
                "Screma {kind} {index} neutral {component} has type {actual:?}, expected {expected:?}"
            ));
        }
    }
    Ok(())
}
/// Storage metadata for one semantic Screma result field.
///
/// This is phase data, not part of `ScremaForm`. Entries follow the form's
/// derived result order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultState {
    pub ownership: SoacOwnership,
}

/// Common access required by generic EGIR plumbing. Result metadata is a
/// separate phase-associated tree from execution state.
pub trait PhaseResults: Clone + std::fmt::Debug {
    fn results(&self) -> &[ResultState];
    fn results_mut(&mut self) -> &mut [ResultState];
}

impl PhaseResults for Vec<ResultState> {
    fn results(&self) -> &[ResultState] {
        self
    }

    fn results_mut(&mut self) -> &mut [ResultState] {
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawState;

#[derive(Clone, Debug)]
pub enum SemanticState<R> {
    Serial,
    Segmented {
        space: SegSpace<R>,
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
    Segmented(Segmented<BindingRef>),
}

/// A Screma plus information owned by one EGIR phase.
#[derive(Clone, Debug)]
pub struct Op<P: WynSoacPhase> {
    pub inputs: Vec<SoacInputType>,
    pub form: ScremaForm,
    pub result_state: P::ScremaResults,
    pub state: P::ScremaState,
}

impl<P: WynSoacPhase> Op<P> {
    pub fn is_map(&self) -> bool {
        self.form.scans.is_empty() && self.form.reductions.is_empty()
    }

    pub fn is_reduce(&self) -> bool {
        self.form.scans.is_empty() && !self.form.reductions.is_empty()
    }

    pub fn result_count(&self) -> usize {
        self.form.result_count()
    }

    pub fn ownership(&self, field: usize) -> Option<SoacOwnership> {
        self.result_state.results().get(field).map(|result| result.ownership)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.form.validate(&self.inputs)?;
        let result_count = self.form.result_count();
        if self.result_state.results().len() != result_count {
            return Err(format!(
                "Screma form produces {result_count} results, but phase state describes {}",
                self.result_state.results().len()
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_with_nodes(
        &self,
        mut node_type: impl FnMut(ValueId) -> Option<Type<TypeName>>,
    ) -> Result<(), String> {
        self.validate()?;
        self.form.validate_neutral_types(&mut node_type)
    }
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.array);
        }
        self.form.for_each_type_mut(visit);
    }

    fn base_referenced_nodes(&self) -> Vec<ValueId> {
        self.form.base_referenced_nodes()
    }

    pub(crate) fn remap_base_referenced_values(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        self.form.remap_referenced_values(&mut map);
    }
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub fn semantic_state(&self) -> &SemanticState<R> {
        &self.state
    }

    pub fn semantic_state_mut(&mut self) -> &mut SemanticState<R> {
        &mut self.state
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<ValueId> {
        let mut nodes = self.base_referenced_nodes();
        if let SemanticState::Segmented { space, .. } = &self.state {
            nodes.extend(space.referenced_nodes());
        }
        nodes
    }

    pub(crate) fn remap_referenced_values(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        self.form.remap_referenced_values(&mut map);
        if let SemanticState::Segmented { space, .. } = &mut self.state {
            for slot in space.referenced_node_slots() {
                *slot = map(*slot);
            }
        }
    }
}

impl Op<super::super::types::Physical> {
    pub fn is_serial(&self) -> bool {
        matches!(self.state, PhysicalState::Serial)
    }
}
#[cfg(test)]
#[path = "screma_tests.rs"]
mod tests;
