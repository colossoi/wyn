//! Algebraic folds used while constructing EGIR from TLC.
//!
//! Rules:
//! - `Project{i}(Tuple/Vector/ArrayLit(a,b,…)) → i-th operand`
//! - `Index(Tuple/Vector/ArrayLit(a,b,…), const k) → k-th operand`
//! - identity elim: `x+0`, `0+x`, `x-0`, `x*1`, `1*x`, `x/1`
//! - absorbing:    `x*0 → 0`, `0*x → 0`
//! - reciprocal:   `x/c → x*(1/c)` for a finite, non-zero f32 constant `c`
//! - double negation: `-(-x) → x`, `!(!x) → x`
//! - negate literal:   `-(c) → const` for an i32/f32 literal `c`
//! - constant-constant: `BinOp(c1, c2) → const` for `+ - * /` over i32/u32/f32
//! - redundant bitcasts: identity bitcasts and inverse bitcast pairs are removed
//!
//! The low-level `EGraph::intern_pure` operation only hash-conses. The TLC
//! converter decides whether to apply these rules before interning a node.
//! Rules never keep borrows across mutations: constant folding extracts
//! operand values first, then interns the result.

use crate::ast::TypeName;
use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::builtins::{by_id, Purity};
use crate::op::{BinaryOperator, UnaryOperator};
use crate::scalar_eval::{self, Scalar};
use crate::ssa::types::ConstantValue;
use crate::types::TypeExt;
use polytype::Type;
use smallvec::smallvec;

use super::types::{EGraph, Family, PureOp, ValueId, ValueKind};

impl<P: Family> EGraph<P> {
    /// Try every rewrite rule and return a substitute node if any fires.
    pub(super) fn try_algebraic_fold(
        &mut self,
        op: &PureOp<P::Resource>,
        operands: &[ValueId],
        result_ty: &Type<TypeName>,
    ) -> Option<ValueId> {
        match op {
            PureOp::Project { index } if operands.len() == 1 => self.fold_project(*index, operands[0]),
            PureOp::Index if operands.len() == 2 => self.fold_index(operands[0], operands[1]),
            PureOp::BinOp(name) if operands.len() == 2 => {
                let (a, b) = (operands[0], operands[1]);
                self.fold_binop_identity(*name, a, b)
                    .or_else(|| self.fold_binop_const(*name, a, b, result_ty))
                    .or_else(|| self.fold_fdiv_const_to_mul(*name, a, b, result_ty))
            }
            PureOp::UnaryOp(name) if operands.len() == 1 => self.fold_unary(*name, operands[0], result_ty),
            PureOp::Intrinsic { id, overload_idx } => {
                self.fold_intrinsic(*id, *overload_idx, operands, result_ty)
            }
            _ => None,
        }
    }

    /// `Index(Tuple/Vector/ArrayLit(e0,…,en), const k) → e_k`
    fn fold_index(&self, base: ValueId, index: ValueId) -> Option<ValueId> {
        let k =
            self.as_i32(index).map(|v| v as usize).or_else(|| self.as_u32(index).map(|v| v as usize))?;
        let ValueKind::Pure {
            op: base_op,
            operands: base_operands,
        } = &self.nodes[base].kind
        else {
            return None;
        };
        let len = match base_op {
            PureOp::Tuple(n) | PureOp::Vector(n) | PureOp::ArrayLit(n) => *n,
            _ => return None,
        };
        (k < len).then(|| base_operands[k])
    }

    /// `Project{i}(Tuple/Vector/ArrayLit(e0,…,en)) → e_i`
    fn fold_project(&mut self, index: u32, base: ValueId) -> Option<ValueId> {
        let origin_field = self.nodes[base]
            .result_origins()
            .iter()
            .find_map(|origin| origin.field(index as usize))
            .and_then(|field| {
                let (ty, destination) = field.single_destination()?;
                Some((ty.clone(), destination.clone()))
            });
        if let Some((ty, destination)) = origin_field {
            return Some(match destination {
                super::types::ResultDestination::ReturnValue(value) => self.canonical_value(value),
                super::types::ResultDestination::Place(
                    super::types::PlaceDestination::Fixed(place)
                    | super::types::PlaceDestination::Bounded { storage: place, .. },
                ) => {
                    let view_ty = crate::types::view_array_of(&ty, crate::types::no_buffer());
                    self.add_place_view(place, view_ty, None).value()
                }
            });
        }

        let ValueKind::Pure {
            op: base_op,
            operands: base_operands,
        } = &self.nodes[base].kind
        else {
            return None;
        };
        let len = match base_op {
            PureOp::Tuple(n) | PureOp::Vector(n) | PureOp::ArrayLit(n) => *n,
            _ => return None,
        };
        let i = index as usize;
        (i < len).then(|| base_operands[i])
    }

    /// Identity and absorbing rules for `+`, `-`, `*`, `/`.
    fn fold_binop_identity(&self, name: BinaryOperator, a: ValueId, b: ValueId) -> Option<ValueId> {
        match name {
            BinaryOperator::Add if self.is_zero_literal(a) => Some(b),
            BinaryOperator::Add if self.is_zero_literal(b) => Some(a),
            BinaryOperator::Subtract if self.is_zero_literal(b) => Some(a),
            BinaryOperator::Multiply if self.is_one_literal(a) => Some(b),
            BinaryOperator::Multiply if self.is_one_literal(b) => Some(a),
            BinaryOperator::Multiply if self.is_zero_literal(a) => Some(a),
            BinaryOperator::Multiply if self.is_zero_literal(b) => Some(b),
            BinaryOperator::Divide if self.is_one_literal(b) => Some(a),
            _ => None,
        }
    }

    /// Constant-constant fold for `+ - * /` when both operands are literals.
    /// Dispatches on the result type; skips division by zero.
    fn fold_binop_const(
        &mut self,
        name: BinaryOperator,
        a: ValueId,
        b: ValueId,
        result_ty: &Type<TypeName>,
    ) -> Option<ValueId> {
        let operand_ty = self.nodes.get(a)?.ty.clone();
        let value = scalar_eval::binary(name, self.as_scalar(a)?, self.as_scalar(b)?, &operand_ty)?;
        self.intern_scalar(value, result_ty)
    }

    /// `x / c → x * (1 / c)` for a runtime f32 scalar/vector and a
    /// finite, non-zero f32 scalar constant. Constant/constant division has
    /// already folded before this rule runs. Avoid materializing infinities
    /// for zero or a reciprocal-overflowing subnormal divisor.
    fn fold_fdiv_const_to_mul(
        &mut self,
        name: BinaryOperator,
        value: ValueId,
        divisor: ValueId,
        result_ty: &Type<TypeName>,
    ) -> Option<ValueId> {
        let f32_result = matches!(result_ty, Type::Constructed(TypeName::Float(32), _));
        let f32_vector_result = result_ty.is_vec()
            && matches!(
                result_ty.elem_type(),
                Some(Type::Constructed(TypeName::Float(32), _))
            );
        if name != BinaryOperator::Divide || (!f32_result && !f32_vector_result) {
            return None;
        }

        let divisor_ty = self.nodes.get(divisor)?.ty.clone();
        if !matches!(divisor_ty, Type::Constructed(TypeName::Float(32), _)) {
            return None;
        }
        let divisor = self.as_f32(divisor)?;
        let reciprocal = 1.0f32 / divisor;
        if divisor == 0.0 || !divisor.is_finite() || !reciprocal.is_finite() {
            return None;
        }

        let reciprocal = self.intern_constant(ConstantValue::from_f32(reciprocal), divisor_ty);
        Some(self.intern_pure(
            PureOp::BinOp(BinaryOperator::Multiply),
            smallvec![value, reciprocal],
            result_ty.clone(),
            None,
        ))
    }

    /// `-(const) → negated const`, plus `-(-x) → x` and `!(!x) → x`.
    ///
    /// Folding a negated numeric literal matters for constant arrays: a
    /// negative element is written `-0.5`, which parses as a negation of the
    /// (unsigned) literal `0.5`. Left unfolded, the array has a runtime
    /// operand and lowers to `OpCompositeConstruct` rebuilt per invocation —
    /// it can't become an `OpConstantComposite`, so it never hoists to a
    /// shared `Private` global. Folding the negation keeps the array constant.
    fn fold_unary(
        &mut self,
        name: UnaryOperator,
        inner: ValueId,
        result_ty: &Type<TypeName>,
    ) -> Option<ValueId> {
        if !matches!(name, UnaryOperator::Negate | UnaryOperator::LogicalNot) {
            return None;
        }
        let operand_ty = self.nodes.get(inner)?.ty.clone();
        if let Some(value) =
            self.as_scalar(inner).and_then(|value| scalar_eval::unary(name, value, &operand_ty))
        {
            return self.intern_scalar(value, result_ty);
        }
        let ValueKind::Pure {
            op: PureOp::UnaryOp(inner_name),
            operands: inner_ops,
        } = &self.nodes[inner].kind
        else {
            return None;
        };
        (*inner_name == name && inner_ops.len() == 1).then(|| inner_ops[0])
    }

    fn fold_intrinsic(
        &mut self,
        id: crate::builtins::BuiltinId,
        overload_idx: usize,
        operands: &[ValueId],
        result_ty: &Type<TypeName>,
    ) -> Option<ValueId> {
        let def = by_id(id);
        if def.raw.purity != Purity::Pure {
            return None;
        }
        let lowering = &def.overloads().get(overload_idx)?.lowering;
        if matches!(lowering, BuiltinLowering::PrimOp(PrimOp::Bitcast)) && operands.len() == 1 {
            let operand = operands[0];
            if self.nodes.get(operand).map(|node| &node.ty) == Some(result_ty) {
                return Some(operand);
            }
            if let ValueKind::Pure {
                op:
                    PureOp::Intrinsic {
                        id: inner_id,
                        overload_idx: inner_overload_idx,
                    },
                operands: inner_operands,
            } = &self.nodes[operand].kind
            {
                let inner_def = by_id(*inner_id);
                let inner_lowering = &inner_def.overloads().get(*inner_overload_idx)?.lowering;
                if inner_def.raw.purity == Purity::Pure
                    && matches!(inner_lowering, BuiltinLowering::PrimOp(PrimOp::Bitcast))
                    && inner_operands.len() == 1
                    && self.nodes.get(inner_operands[0]).map(|node| &node.ty) == Some(result_ty)
                {
                    return Some(inner_operands[0]);
                }
            }
        }
        let value = match lowering {
            BuiltinLowering::PrimOp(PrimOp::GlslExt(8)) if operands.len() == 1 => {
                ConstantValue::from_f32(self.as_f32(operands[0])?.floor())
            }
            BuiltinLowering::PrimOp(PrimOp::GlslExt(9)) if operands.len() == 1 => {
                ConstantValue::from_f32(self.as_f32(operands[0])?.ceil())
            }
            BuiltinLowering::PrimOp(prim) if operands.len() == 1 => {
                self.fold_conversion_value(prim, operands[0], result_ty)?
            }
            _ => return None,
        };
        Some(self.intern_constant(value, result_ty.clone()))
    }

    fn fold_conversion_value(
        &self,
        prim: &PrimOp,
        operand: ValueId,
        result_ty: &Type<TypeName>,
    ) -> Option<ConstantValue> {
        match (prim, result_ty) {
            (PrimOp::FPToSI, Type::Constructed(TypeName::Int(32), _)) => {
                let v = self.as_f32(operand)?;
                (v.is_finite() && v.trunc() >= i32::MIN as f32 && v.trunc() < 2147483648.0)
                    .then(|| ConstantValue::I32(v.trunc() as i32))
            }
            (PrimOp::FPToUI, Type::Constructed(TypeName::UInt(32), _)) => {
                let v = self.as_f32(operand)?;
                (v.is_finite() && v.trunc() >= 0.0 && v.trunc() < 4294967296.0)
                    .then(|| ConstantValue::U32(v.trunc() as u32))
            }
            (PrimOp::SIToFP, Type::Constructed(TypeName::Float(32), _)) => {
                Some(ConstantValue::from_f32(self.as_i32(operand)? as f32))
            }
            (PrimOp::UIToFP, Type::Constructed(TypeName::Float(32), _)) => {
                Some(ConstantValue::from_f32(self.as_u32(operand)? as f32))
            }
            (PrimOp::SConvert, Type::Constructed(TypeName::Int(32), _)) => {
                self.as_i32(operand).map(ConstantValue::I32)
            }
            (PrimOp::UConvert, Type::Constructed(TypeName::UInt(32), _)) => {
                self.as_u32(operand).map(ConstantValue::U32)
            }
            (PrimOp::FPConvert, Type::Constructed(TypeName::Float(32), _)) => {
                self.as_f32(operand).map(ConstantValue::from_f32)
            }
            (PrimOp::Bitcast, _) => self.bitcast_constant(operand, result_ty),
            _ => None,
        }
    }

    fn bitcast_constant(&self, operand: ValueId, result_ty: &Type<TypeName>) -> Option<ConstantValue> {
        match result_ty {
            Type::Constructed(TypeName::Int(32), _) => self
                .as_u32(operand)
                .map(|v| ConstantValue::I32(v as i32))
                .or_else(|| self.as_i32(operand).map(ConstantValue::I32))
                .or_else(|| self.as_f32(operand).map(|v| ConstantValue::I32(v.to_bits() as i32))),
            Type::Constructed(TypeName::UInt(32), _) => self
                .as_i32(operand)
                .map(|v| ConstantValue::U32(v as u32))
                .or_else(|| self.as_u32(operand).map(ConstantValue::U32))
                .or_else(|| self.as_f32(operand).map(|v| ConstantValue::U32(v.to_bits()))),
            Type::Constructed(TypeName::Float(32), _) => self
                .as_u32(operand)
                .map(|v| ConstantValue::from_f32(f32::from_bits(v)))
                .or_else(|| self.as_i32(operand).map(|v| ConstantValue::from_f32(f32::from_bits(v as u32))))
                .or_else(|| self.as_f32(operand).map(ConstantValue::from_f32)),
            _ => None,
        }
    }

    // ---- literal predicates ------------------------------------------------

    fn is_zero_literal(&self, nid: ValueId) -> bool {
        match &self.nodes[nid].kind {
            ValueKind::Constant(ConstantValue::I32(0)) | ValueKind::Constant(ConstantValue::U32(0)) => true,
            ValueKind::Constant(ConstantValue::F32(bits)) => f32::from_bits(*bits) == 0.0,
            ValueKind::Pure { op, .. } => match op {
                PureOp::Int(s) | PureOp::Uint(s) => s == "0",
                PureOp::Float(s) => s.parse::<f32>().map(|f| f == 0.0).unwrap_or(false),
                _ => false,
            },
            _ => false,
        }
    }

    fn is_one_literal(&self, nid: ValueId) -> bool {
        match &self.nodes[nid].kind {
            ValueKind::Constant(ConstantValue::I32(1)) | ValueKind::Constant(ConstantValue::U32(1)) => true,
            ValueKind::Constant(ConstantValue::F32(bits)) => f32::from_bits(*bits) == 1.0,
            ValueKind::Pure { op, .. } => match op {
                PureOp::Int(s) | PureOp::Uint(s) => s == "1",
                PureOp::Float(s) => s.parse::<f32>().map(|f| f == 1.0).unwrap_or(false),
                _ => false,
            },
            _ => false,
        }
    }

    // ---- value extractors --------------------------------------------------

    fn as_scalar(&self, node: ValueId) -> Option<Scalar> {
        match &self.nodes.get(node)?.ty {
            Type::Constructed(TypeName::Int(32), _) => self.as_i32(node).map(|v| Scalar::Int(v as i64)),
            Type::Constructed(TypeName::UInt(32), _) => self.as_u32(node).map(|v| Scalar::Int(v as i64)),
            Type::Constructed(TypeName::Float(32), _) => self.as_f32(node).map(|v| Scalar::Float(v as f64)),
            Type::Constructed(TypeName::Bool, _) => self.as_bool(node).map(Scalar::Bool),
            _ => None,
        }
    }

    fn intern_scalar(&mut self, value: Scalar, ty: &Type<TypeName>) -> Option<ValueId> {
        let constant = match (value, ty) {
            (Scalar::Int(value), Type::Constructed(TypeName::Int(32), _)) => {
                ConstantValue::I32(value as i32)
            }
            (Scalar::Int(value), Type::Constructed(TypeName::UInt(32), _)) => {
                ConstantValue::U32(value as u32)
            }
            (Scalar::Float(value), Type::Constructed(TypeName::Float(32), _)) => {
                ConstantValue::from_f32(value as f32)
            }
            (Scalar::Bool(value), Type::Constructed(TypeName::Bool, _)) => ConstantValue::Bool(value),
            _ => return None,
        };
        Some(self.intern_constant(constant, ty.clone()))
    }

    pub(super) fn as_i32(&self, nid: ValueId) -> Option<i32> {
        match &self.nodes[nid].kind {
            ValueKind::Constant(ConstantValue::I32(v)) => Some(*v),
            ValueKind::Pure {
                op: PureOp::Int(s),
                operands,
            } if operands.is_empty() => s.parse().ok(),
            _ => None,
        }
    }

    pub(super) fn as_u32(&self, nid: ValueId) -> Option<u32> {
        match &self.nodes[nid].kind {
            ValueKind::Constant(ConstantValue::U32(v)) => Some(*v),
            ValueKind::Pure {
                op: PureOp::Uint(s),
                operands,
            } if operands.is_empty() => s.parse().ok(),
            _ => None,
        }
    }

    pub(super) fn as_f32(&self, nid: ValueId) -> Option<f32> {
        match &self.nodes[nid].kind {
            ValueKind::Constant(ConstantValue::F32(bits)) => Some(f32::from_bits(*bits)),
            ValueKind::Pure {
                op: PureOp::Float(s),
                operands,
            } if operands.is_empty() => s.parse().ok(),
            _ => None,
        }
    }

    fn as_bool(&self, nid: ValueId) -> Option<bool> {
        match &self.nodes[nid].kind {
            ValueKind::Constant(ConstantValue::Bool(v)) => Some(*v),
            ValueKind::Pure {
                op: PureOp::Bool(v),
                operands,
            } if operands.is_empty() => Some(*v),
            _ => None,
        }
    }
}
