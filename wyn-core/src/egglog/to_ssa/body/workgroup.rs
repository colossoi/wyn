//! Lower group-wide control and transfers with explicit shared storage.
use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::egglog::graph::CollectiveInput;
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::array_copy::matching_view_element;
use crate::ssa::types::InstKind;
use crate::types::{Type, TypeExt, TypeName};
use crate::ResourceAccess;
use egglog_engine::Value;

enum GroupCarry {
    Scalar(usize),
    Array(usize),
    Product(TypeName, Vec<GroupCarry>),
}

impl Body<'_, '_, '_> {
    fn workgroup_width(&self) -> Result<u32, OptimizeError> {
        let Some(dispatch) = self.active_dispatch else {
            return Err(error("group-wide operation is outside its dispatch"));
        };
        Ok(self.compiler.execution.dispatch(dispatch)?.width.get())
    }

    fn workgroup_leader(&mut self) -> Result<Typed, OptimizeError> {
        let lane = self.local_id()?;
        self.binary(BinaryOperator::Equal, lane, Self::number(0))
    }

    fn shared_array_for(&mut self, value: &Typed) -> Result<Typed, OptimizeError> {
        let Some(Type::Constructed(TypeName::Size(count), _)) = value.ty.array_size() else {
            return Err(error("group-wide array must have a fixed length"));
        };
        let Some(element) = value.ty.elem_type() else {
            return Err(error("group-wide array is missing an element type"));
        };
        let count = u32::try_from(*count).map_err(|_| error("workgroup array size exceeds u32"))?;
        self.workgroup_array(count, element)
    }

    fn workgroup_copy(&mut self, source: Typed, destination: Typed) -> Result<(), OptimizeError> {
        let lane = self.local_id()?;
        let step = Self::number(self.workgroup_width()?);
        let count = self.array_length(destination.clone())?;
        let count = self.cast(count, &lane.ty)?;
        if matching_view_element(&source.ty, &destination.ty).is_some() {
            self.storage_access(&source.ty, ResourceAccess::Read);
            self.storage_access(&destination.ty, ResourceAccess::Write);
            self.builder
                .push_void_inst(InstKind::Copy {
                    destination: destination.value,
                    source: source.value,
                    length: count.value,
                    start: lane.value,
                    step: step.value,
                    collective: true,
                })
                .map_err(builder_error)?;
        } else {
            self.counted(lane, count, step, vec![], |body, index, _| {
                let item = body.index(source, index.clone())?;
                body.store(destination, index, item)?;
                Ok(vec![])
            })?;
        }
        Ok(())
    }

    fn workgroup_condition(&mut self, slot: Typed, condition: Typed) -> Result<Typed, OptimizeError> {
        let leader = self.workgroup_leader()?;
        let bit = self.select(condition, Self::number(1), Self::number(0))?;
        self.when(leader, |body| body.store(slot.clone(), Self::number(0), bit))?;
        let (place, ty) = self.index_place(slot, Self::number(0))?;
        let value = self
            .builder
            .push_inst(InstKind::WorkgroupUniformLoad { place }, ty.clone())
            .map_err(builder_error)?;
        self.binary(
            BinaryOperator::NotEqual,
            Typed {
                value: value.into(),
                ty,
            },
            Self::number(0),
        )
    }

    pub(super) fn workgroup_while(
        &mut self,
        region: Value,
        condition: Value,
        params: &[Value],
        initial: Vec<Typed>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let flag = self.workgroup_array(1, &Self::number(0).ty)?;
        let mut shared = Vec::new();
        let mut scalars = Vec::new();
        let carries = initial
            .into_iter()
            .map(|value| self.init_group_carry(value, &mut shared, &mut scalars))
            .collect::<Result<Vec<_>, _>>()?;
        self.workgroup_barrier()?;
        let install = |body: &mut Self, scalars: &[Typed]| -> Result<(), OptimizeError> {
            for (&name, carry) in params.iter().zip(&carries) {
                let value = body.group_carry_value(carry, &shared, scalars)?;
                body.values.insert(name, value);
            }
            Ok(())
        };
        let scalars = self.while_loop(
            scalars,
            |body, scalars| {
                install(body, scalars)?;
                body.workgroup_condition(flag, body.named(condition)?)
            },
            |body, scalars| {
                install(body, &scalars)?;
                let next_values = body.block(region)?;
                if next_values.len() != carries.len() {
                    return Err(error("group-wide loop carry arity mismatch"));
                }
                let mut next_scalars = Vec::new();
                // Stage every new array before overwriting any old carry.
                for (value, carry) in next_values.into_iter().zip(&carries) {
                    body.stage_group_carry(carry, value, &shared, &mut next_scalars)?;
                }
                body.workgroup_barrier()?;
                for (current, next) in &shared {
                    body.workgroup_copy(next.clone(), current.clone())?;
                }
                body.workgroup_barrier()?;
                Ok(next_scalars)
            },
        )?;
        carries.iter().map(|carry| self.group_carry_value(carry, &shared, &scalars)).collect()
    }

    fn init_group_carry(
        &mut self,
        value: Typed,
        shared: &mut Vec<(Typed, Typed)>,
        scalars: &mut Vec<Typed>,
    ) -> Result<GroupCarry, OptimizeError> {
        if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = &value.ty {
            let fields = (0..fields.len())
                .map(|i| {
                    let field = self.field(value.clone(), i)?;
                    self.init_group_carry(field, shared, scalars)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(GroupCarry::Product(name.clone(), fields));
        }
        if value.ty.is_array() {
            let current = self.shared_array_for(&value)?;
            let next = self.shared_array_for(&value)?;
            self.workgroup_copy(value, current.clone())?;
            let slot = shared.len();
            shared.push((current, next));
            return Ok(GroupCarry::Array(slot));
        }
        let slot = scalars.len();
        scalars.push(value);
        Ok(GroupCarry::Scalar(slot))
    }

    fn group_carry_value(
        &mut self,
        carry: &GroupCarry,
        shared: &[(Typed, Typed)],
        scalars: &[Typed],
    ) -> Result<Typed, OptimizeError> {
        match carry {
            GroupCarry::Scalar(slot) => Ok(scalars[*slot].clone()),
            GroupCarry::Array(slot) => Ok(shared[*slot].0.clone()),
            GroupCarry::Product(name, fields) => {
                let fields = fields
                    .iter()
                    .map(|field| self.group_carry_value(field, shared, scalars))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = Type::Constructed(
                    name.clone(),
                    fields.iter().map(|field| field.ty.clone()).collect(),
                );
                self.op(OpTag::Tuple, fields, ty)
            }
        }
    }

    fn stage_group_carry(
        &mut self,
        carry: &GroupCarry,
        value: Typed,
        shared: &[(Typed, Typed)],
        scalars: &mut Vec<Typed>,
    ) -> Result<(), OptimizeError> {
        match carry {
            GroupCarry::Scalar(_) => scalars.push(value),
            GroupCarry::Array(slot) => self.workgroup_copy(value, shared[*slot].1.clone())?,
            GroupCarry::Product(_, fields) => {
                for (i, field) in fields.iter().enumerate() {
                    let value = self.field(value.clone(), i)?;
                    self.stage_group_carry(field, value, shared, scalars)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn workgroup_accumulator(
        &mut self,
        statement: Value,
        region: Value,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let checkpoint = self.values.checkpoint();
        let mut outputs = Vec::new();
        for group in 0..self.query().0.accumulator_group_count(statement) {
            for slot in 0..self.query().0.collective_input_count(statement, group, CollectiveInput::Backing)
            {
                let name =
                    self.query().0.collective_input(statement, group, CollectiveInput::Backing, slot)?;
                let source = self.named(name)?;
                let destination = self.shared_array_for(&source)?;
                self.workgroup_copy(source, destination.clone())?;
                self.values.insert(name, destination.clone());
                outputs.push(destination);
            }
        }
        self.workgroup_barrier()?;
        let leader = self.workgroup_leader()?;
        self.when(leader, |body| body.statements(&body.query().operations(region)?))?;
        self.workgroup_barrier()?;
        self.values.restore(checkpoint);
        Ok(outputs)
    }
}
