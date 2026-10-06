//! Structural TLC metadata consumed by the retained placement rules.
use super::{Import, OperatorBody};
use crate::egglog::OptimizeError;
use crate::interface::{EntryParamBinding, EntryParamBindingKind};
use crate::op::BinaryOperator;
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::{SoacOp, TermKind, VarRef};
use crate::types::{Type, TypeExt, TypeName};
use egglog_engine::{RawValues, Value, Write};

impl Import<'_, '_, '_, '_> {
    pub(super) fn entry_binding(&mut self, binding: &EntryParamBinding) -> Result<(), OptimizeError> {
        let value = self.resolve(binding.param_sym)?;
        self.sink.set("SourceStorage", value, true)?;
        self.summaries.values.entry(value).or_default().device = true;
        match &binding.kind {
            EntryParamBindingKind::Single {
                binding, elem_bytes, ..
            } => {
                self.storage_binding(value, binding.set, binding.binding, *elem_bytes)?;
            }
            EntryParamBindingKind::TupleOfViews(fields) => {
                for (index, field) in fields.iter().enumerate() {
                    let component = self.sink.add("SourceProjected", (value, index as i64))?;
                    self.register_source(component)?;
                    self.sink.add("SourceProjection", (component, value, index as i64))?;
                    self.sink.add("SourceField", (value, index as i64, component))?;
                    self.sink.add("SourceStorageField", component)?;
                    self.storage_binding(
                        component,
                        field.binding.set,
                        field.binding.binding,
                        field.elem_bytes,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn storage_binding(
        &mut self,
        value: Value,
        set: u32,
        binding: u32,
        stride: u32,
    ) -> Result<(), OptimizeError> {
        self.sink.add("SourceBinding", (value, i64::from(set), i64::from(binding)))?;
        self.register_source(value)?;
        let binding = self.sink.add("InputBinding", (i64::from(set), i64::from(binding)))?;
        let storage = self.sink.add("StorageInput", (binding, i64::from(stride)))?;
        self.sink.set("AbiStorage", value, storage)?;
        Ok(())
    }

    pub(super) fn collective_metadata(
        &mut self,
        operation: Value,
        soac: &SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> Result<(), OptimizeError> {
        match soac {
            SoacOp::ReduceByIndex { dest, op, .. } => {
                let safe = matches!(
                    &dest.elem_ty,
                    Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _)
                );
                let update = self.sink.add(reducer_operator(op, &self.definitions), RawValues(vec![]))?;
                self.sink.add("SourceIndexedReducer", (operation, safe, update))?;
            }
            SoacOp::BucketScatter {
                inputs,
                input_dimensions,
                domain_rank,
                ..
            } => {
                let mut axes = vec![None; usize::from(*domain_rank)];
                for (input, mapping) in input_dimensions.iter().enumerate() {
                    self.sink.set(
                        "SourceBucketInputRank",
                        (operation, input as i64),
                        mapping.len() as i64,
                    )?;
                    for (dimension, &axis) in mapping.iter().enumerate() {
                        self.sink.set(
                            "SourceBucketInputDimension",
                            (operation, input as i64, dimension as i64),
                            i64::from(axis),
                        )?;
                    }
                }
                for (input, mapping) in inputs.iter().zip(input_dimensions).enumerate() {
                    let (array, mapping) = mapping;
                    let ty = array.array_type();
                    for (axis, &domain_axis) in mapping.iter().enumerate() {
                        let Some(slot) = axes.get_mut(usize::from(domain_axis)) else {
                            return Err(OptimizeError::Output("invalid bucket-scatter domain axis".into()));
                        };
                        *slot = if axis == 0 {
                            Some((input, None))
                        } else {
                            inner_dimension(&ty, axis).map(|n| (input, Some(n)))
                        };
                    }
                }
                self.sink.add(
                    "SourceKnownBucketDomain",
                    (operation, axes.iter().all(Option::is_some)),
                )?;
                self.sink.set("SourceBucketRank", operation, i64::from(*domain_rank))?;
                for (axis, dimension) in axes.into_iter().enumerate() {
                    if let Some((input, fixed)) = dimension {
                        if let Some(n) = fixed {
                            let extent = self.sink.add("Fixed", n as i64)?;
                            self.sink.set("SourceBucketAxis", (operation, axis as i64), extent)?;
                        } else {
                            self.sink
                                .add("SourceBucketInputAxis", (operation, axis as i64, input as i64))?;
                        }
                    }
                }
            }
            SoacOp::Map { .. }
            | SoacOp::Reduce { .. }
            | SoacOp::Scan { .. }
            | SoacOp::Filter { .. }
            | SoacOp::Scatter { .. } => {}
        }
        Ok(())
    }
}

fn reducer_operator(
    body: &OperatorBody,
    definitions: &crate::LookupMap<crate::SymbolId, &super::Term>,
) -> &'static str {
    let (term, parameters) = if let TermKind::Var(VarRef::Symbol(symbol)) = body.lam.body.kind {
        let Some(definition) = definitions.get(&symbol) else {
            return "AtomicCas";
        };
        crate::tlc::extract_lambda_params_ref(definition)
    } else {
        (body.lam.body.as_ref(), body.lam.params.clone())
    };
    let [a, b] = parameters.as_slice() else {
        return "AtomicCas";
    };
    let TermKind::App { func, args } = &term.kind else {
        return "AtomicCas";
    };
    let [left, right] = args.as_slice() else {
        return "AtomicCas";
    };
    let (TermKind::Var(VarRef::Symbol(left)), TermKind::Var(VarRef::Symbol(right))) =
        (&left.kind, &right.kind)
    else {
        return "AtomicCas";
    };
    if !body.data.captures.is_empty()
        || left == right
        || ![a.0, b.0].contains(left)
        || ![a.0, b.0].contains(right)
    {
        return "AtomicCas";
    }
    let TermKind::BinOp(operator) = &func.kind else {
        return "AtomicCas";
    };
    match operator.op {
        BinaryOperator::Add => "AtomicAdd",
        BinaryOperator::BitwiseAnd => "AtomicAnd",
        BinaryOperator::BitwiseOr => "AtomicOr",
        BinaryOperator::BitwiseXor => "AtomicXor",
        _ => "AtomicCas",
    }
}

fn inner_dimension(mut ty: &Type, mut axis: usize) -> Option<usize> {
    loop {
        if let Type::Constructed(TypeName::Tuple(_), fields) = ty {
            ty = fields.first()?;
            continue;
        }
        let dimensions = ty.array_dims()?;
        if axis < dimensions.len() {
            let Type::Constructed(TypeName::Size(n), _) = dimensions[axis] else {
                return None;
            };
            return Some(n);
        }
        axis -= dimensions.len();
        ty = ty.elem_type()?;
    }
}
