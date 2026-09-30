//! Forward existing aggregate operands. These are definition facts, not a value
//! numbering cache: following a valid SSA operand never crosses its dominance
//! boundary or moves evaluation. Raw builder emission remains a conservative
//! fallback (including vector constructions with vector constituents).
use super::{dr, spirv, SpirvBuilder};
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct Definitions {
    reserved_types: HashMap<spirv::Word, spirv::Word>,
    constructs: HashMap<spirv::Word, (spirv::Word, Vec<spirv::Word>)>,
    extracts: HashMap<spirv::Word, (spirv::Word, Vec<u32>)>,
    shapes: HashMap<spirv::Word, Shape>,
}

#[derive(Clone)]
enum Shape {
    Fields(Vec<spirv::Word>),
    Repeated(spirv::Word, usize),
}

impl Shape {
    fn len(&self) -> usize {
        match self {
            Self::Fields(fields) => fields.len(),
            Self::Repeated(_, n) => *n,
        }
    }
    fn field(&self, index: u32) -> Option<spirv::Word> {
        match self {
            Self::Fields(fields) => fields.get(index as usize).copied(),
            Self::Repeated(ty, n) => ((index as usize) < *n).then_some(*ty),
        }
    }
}

impl SpirvBuilder {
    /// Reserve a typed value whose definition will be emitted later (e.g. a
    /// loop phi). Its type is available to exact reconstruction immediately.
    pub fn reserve_value(&mut self, ty: super::TypeId) -> super::ValueId {
        let value = self.inner.id();
        self.aggregates.reserved_types.insert(value, *ty);
        super::ValueId::new(value)
    }

    fn aggregate_shape(&mut self, ty: spirv::Word) -> Option<Shape> {
        if let Some(shape) = self.aggregates.shapes.get(&ty) {
            return Some(shape.clone());
        }
        let types = &self.inner.module_ref().types_global_values;
        let definition = types.iter().find(|i| i.result_id == Some(ty))?;
        let shape = match (definition.class.opcode, definition.operands.as_slice()) {
            (spirv::Op::TypeStruct, fields) => Shape::Fields(
                fields
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
                    .collect::<Option<_>>()?,
            ),
            (
                spirv::Op::TypeVector | spirv::Op::TypeMatrix,
                [dr::Operand::IdRef(element), dr::Operand::LiteralBit32(n)],
            ) => Shape::Repeated(*element, *n as usize),
            (spirv::Op::TypeArray, [dr::Operand::IdRef(element), dr::Operand::IdRef(length)]) => {
                let length = types.iter().find(|i| i.result_id == Some(*length))?;
                if length.class.opcode != spirv::Op::Constant {
                    return None;
                }
                let [dr::Operand::LiteralBit32(n)] = length.operands.as_slice() else {
                    return None;
                };
                Shape::Repeated(*element, *n as usize)
            }
            _ => return None,
        };
        self.aggregates.shapes.insert(ty, shape.clone());
        Some(shape)
    }

    // Only needed after the operands prove a potential exact reconstruction.
    // Parameters, loads and phis may have been emitted through raw rspirv APIs.
    fn aggregate_value_type(&self, value: spirv::Word) -> Option<spirv::Word> {
        if let Some(&ty) = self.aggregates.reserved_types.get(&value) {
            return Some(ty);
        }
        let module = self.inner.module_ref();
        self.inner
            .selected_function()
            .and_then(|index| module.functions.get(index))
            .into_iter()
            .flat_map(|function| {
                function
                    .parameters
                    .iter()
                    .chain(function.blocks.iter().flat_map(|block| block.instructions.iter()))
            })
            .chain(module.types_global_values.iter())
            .find(|i| i.result_id == Some(value))
            .and_then(|i| i.result_type)
    }

    /// Construct an aggregate, or forward a complete ordered reconstruction of
    /// the same SPIR-V type. Explicit result IDs must still be defined.
    pub fn composite_construct(
        &mut self,
        result_type: spirv::Word,
        result_id: Option<spirv::Word>,
        constituents: impl IntoIterator<Item = spirv::Word>,
    ) -> Result<spirv::Word, dr::Error> {
        let constituents: Vec<_> = constituents.into_iter().collect();
        let shape = self.aggregate_shape(result_type);
        let direct = shape.as_ref().is_some_and(|shape| shape.len() == constituents.len());
        if result_id.is_none() && direct && !constituents.is_empty() {
            let source = constituents
                .iter()
                .enumerate()
                .try_fold(None, |source, (i, value)| {
                    let (original, indices) = self.aggregates.extracts.get(value)?;
                    (indices.as_slice() == [i as u32] && source.is_none_or(|source| source == *original))
                        .then_some(Some(*original))
                })
                .flatten();
            if let Some(source) = source {
                if self.aggregate_value_type(source) == Some(result_type) {
                    return Ok(source);
                }
            }
        }
        let value = self.inner.composite_construct(result_type, result_id, constituents.clone())?;
        if direct {
            self.aggregates.constructs.insert(value, (result_type, constituents));
        }
        Ok(value)
    }

    /// Extract through known constructors, including nested literal index paths.
    /// Unknown definitions and vector-packed constructors retain normal emission.
    pub fn composite_extract(
        &mut self,
        result_type: spirv::Word,
        result_id: Option<spirv::Word>,
        composite: spirv::Word,
        indexes: impl IntoIterator<Item = u32>,
    ) -> Result<spirv::Word, dr::Error> {
        let indexes: Vec<_> = indexes.into_iter().collect();
        let mut source = composite;
        let mut consumed = 0;
        if result_id.is_none() {
            while let Some(&index) = indexes.get(consumed) {
                let Some((ty, fields)) = self.aggregates.constructs.get(&source) else {
                    break;
                };
                let (ty, field) = (*ty, fields.get(index as usize).copied());
                let Some(field) = field else { break };
                let mut field_ty = self.aggregate_shape(ty).and_then(|shape| shape.field(index));
                for &index in &indexes[consumed + 1..] {
                    field_ty = field_ty.and_then(|ty| self.aggregate_shape(ty)?.field(index));
                }
                if field_ty != Some(result_type) {
                    break;
                }
                source = field;
                consumed += 1;
            }
            if consumed > 0 && consumed == indexes.len() {
                return Ok(source);
            }
        }
        let remaining = indexes[consumed..].to_vec();
        let value = self.inner.composite_extract(result_type, result_id, source, remaining.clone())?;
        self.aggregates.extracts.insert(value, (source, remaining));
        Ok(value)
    }
}
