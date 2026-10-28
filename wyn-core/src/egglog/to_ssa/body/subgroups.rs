//! Native subgroup arithmetic joined through workgroup memory. Logical lane order
//! follows active subgroup ranks, never an assumed mapping from local invocation IDs.
use super::{error, Body, OptimizeError, Typed};
use crate::builtins::{catalog, BuiltinId};
use crate::op::{BinaryOperator, OpTag};

#[derive(Clone)]
pub(in crate::egglog::to_ssa) struct Subgroup {
    pub lane: Typed,
    group: Typed,
    rank: Typed,
    groups: Typed,
}

impl Body<'_, '_, '_> {
    fn subgroup_intrinsic(&mut self, id: BuiltinId, args: Vec<Typed>) -> Result<Typed, OptimizeError> {
        let ty = args.first().map_or_else(|| Self::number(0).ty, |value| value.ty.clone());
        self.op(OpTag::Intrinsic { id, overload_idx: 0 }, args, ty)
    }

    pub fn prepare_subgroups(&mut self, width: u32) -> Result<(), OptimizeError> {
        let known = catalog().known();
        let group = self.subgroup_intrinsic(known.subgroup_id, vec![])?;
        let groups = self.subgroup_intrinsic(known.num_subgroups, vec![])?;
        let rank = self.subgroup_intrinsic(known.subgroup_exclusive_add, vec![Self::number(1)])?;
        let count = self.subgroup_intrinsic(known.subgroup_add, vec![Self::number(1)])?;
        let counts = self.workgroup_array(width, &Self::number(0).ty)?;
        let leader = self.binary(BinaryOperator::Equal, rank.clone(), Self::number(0))?;
        self.when(leader, |body| body.store(counts.clone(), group.clone(), count))?;
        self.workgroup_barrier()?;
        let base = self.counted(
            Self::number(0),
            group.clone(),
            Self::number(1),
            vec![Self::number(0)],
            |body, i, state| {
                let count = body.index(counts.clone(), i)?;
                Ok(vec![body.binary(BinaryOperator::Add, state[0].clone(), count)?])
            },
        )?;
        let lane = self.binary(BinaryOperator::Add, base[0].clone(), rank.clone())?;
        self.subgroup = Some(Subgroup {
            lane,
            group,
            rank,
            groups,
        });
        Ok(())
    }

    pub fn subgroup_scan(
        &mut self,
        values: Vec<Typed>,
        width: u32,
    ) -> Result<(Vec<Typed>, Vec<Typed>, Vec<Typed>), OptimizeError> {
        let Some(subgroup) = self.subgroup.clone() else {
            return Err(error("subgroup scan has no lane assignment"));
        };
        let known = catalog().known();
        let mut inclusive = Vec::new();
        let mut exclusive = Vec::new();
        let mut buffers = Vec::new();
        for value in values {
            let total = self.subgroup_intrinsic(known.subgroup_add, vec![value.clone()])?;
            inclusive.push(self.subgroup_intrinsic(known.subgroup_inclusive_add, vec![value.clone()])?);
            exclusive.push(self.subgroup_intrinsic(known.subgroup_exclusive_add, vec![value.clone()])?);
            let buffer = self.workgroup_array(width, &value.ty)?;
            let leader = self.binary(BinaryOperator::Equal, subgroup.rank.clone(), Self::number(0))?;
            self.when(leader, |body| {
                body.store(buffer.clone(), subgroup.group.clone(), total)
            })?;
            buffers.push(buffer);
        }
        self.workgroup_barrier()?;
        let mut totals = Vec::new();
        for (i, buffer) in buffers.iter().enumerate() {
            let zero = self.literal("0", &inclusive[i].ty)?;
            let sums = self.counted(
                Self::number(0),
                subgroup.groups.clone(),
                Self::number(1),
                vec![zero.clone(), zero],
                |body, group, state| {
                    let value = body.index(buffer.clone(), group.clone())?;
                    let total = body.binary(BinaryOperator::Add, state[0].clone(), value.clone())?;
                    let before = body.binary(BinaryOperator::Less, group, subgroup.group.clone())?;
                    let offset = body.binary(BinaryOperator::Add, state[1].clone(), value)?;
                    let offset = body.select(before, offset, state[1].clone())?;
                    Ok(vec![total, offset])
                },
            )?;
            inclusive[i] = self.binary(BinaryOperator::Add, sums[1].clone(), inclusive[i].clone())?;
            exclusive[i] = self.binary(BinaryOperator::Add, sums[1].clone(), exclusive[i].clone())?;
            totals.push(sums[0].clone());
        }
        self.workgroup_barrier()?;
        Ok((inclusive, exclusive, totals))
    }
}
