//! Lower aggregate phis to fields without moving operand evaluation.
use super::{dr, spirv, SpirvBuilder};

impl SpirvBuilder {
    /// Insert phis after all predecessor blocks are complete. Keep structs and
    /// arrays out of phis: Naga lowers them to locals with invalid Vulkan layout.
    /// Reconstruct the original value after the block's scalar/vector phis.
    pub fn insert_value_phi(
        &mut self,
        result_type: spirv::Word,
        result_id: spirv::Word,
        incoming: &[(spirv::Word, spirv::Word)],
    ) -> Result<(), dr::Error> {
        let function = self.inner.selected_function().ok_or(dr::Error::DetachedBlock)?;
        let target = self.inner.selected_block().ok_or(dr::Error::DetachedBlock)?;
        let mut pending = vec![(result_type, result_id, Vec::new())];
        let mut constructs = Vec::new();
        while let Some((ty, id, path)) = pending.pop() {
            if let Some(types) = self.aggregate_fields(ty) {
                let fields: Vec<_> = types.iter().map(|_| self.inner.id()).collect();
                for (index, (field_ty, &field)) in types.into_iter().zip(&fields).enumerate() {
                    let mut field_path = path.clone();
                    field_path.push(index as u32);
                    pending.push((field_ty, field, field_path));
                }
                constructs.push(dr::Instruction::new(
                    spirv::Op::CompositeConstruct,
                    Some(ty),
                    Some(id),
                    fields.into_iter().map(dr::Operand::IdRef).collect(),
                ));
                continue;
            }
            let mut inputs = Vec::with_capacity(incoming.len());
            for &(value, source) in incoming {
                let value = if path.is_empty() {
                    value
                } else {
                    let blocks = &self.inner.module_ref().functions[function].blocks;
                    let index = blocks
                        .iter()
                        .position(|b| b.label_id() == Some(source))
                        .ok_or(dr::Error::BlockNotFound)?;
                    let instructions = &blocks[index].instructions;
                    // A merge instruction must remain adjacent to its branch.
                    let merge =
                        instructions.len().checked_sub(2).and_then(|i| instructions.get(i)).is_some_and(
                            |i| matches!(i.class.opcode, spirv::Op::LoopMerge | spirv::Op::SelectionMerge),
                        );
                    self.inner.select_block(Some(index))?;
                    self.inner.insert_composite_extract(
                        dr::InsertPoint::FromEnd(1 + usize::from(merge)),
                        ty,
                        None,
                        value,
                        path.clone(),
                    )?
                };
                inputs.push((value, source));
            }
            self.inner.select_block(Some(target))?;
            self.inner.insert_phi(dr::InsertPoint::Begin, ty, Some(id), inputs)?;
        }
        let instructions = &mut self.inner.module_mut().functions[function].blocks[target].instructions;
        let position = instructions.iter().take_while(|i| i.class.opcode == spirv::Op::Phi).count();
        instructions.splice(position..position, constructs.into_iter().rev());
        Ok(())
    }

    fn aggregate_fields(&self, ty: spirv::Word) -> Option<Vec<spirv::Word>> {
        let types = &self.inner.module_ref().types_global_values;
        let definition = types.iter().find(|i| i.result_id == Some(ty))?;
        match (definition.class.opcode, definition.operands.as_slice()) {
            (spirv::Op::TypeStruct, fields) => fields
                .iter()
                .map(
                    |field| {
                        if let dr::Operand::IdRef(id) = field {
                            Some(*id)
                        } else {
                            None
                        }
                    },
                )
                .collect(),
            (spirv::Op::TypeArray, [dr::Operand::IdRef(element), dr::Operand::IdRef(length)]) => {
                let length = types.iter().find(|i| i.result_id == Some(*length))?;
                match (length.class.opcode, length.operands.as_slice()) {
                    (spirv::Op::Constant, [dr::Operand::LiteralBit32(n)]) => {
                        Some(vec![*element; *n as usize])
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
