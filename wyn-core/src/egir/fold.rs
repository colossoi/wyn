//! Algebraic folds applied eagerly inside `EGraph::intern_pure`.
//!
//! Rules:
//! - `Project{i}(Tuple/Vector/ArrayLit(a,b,…)) → i-th operand`
//! - identity elim: `x+0`, `0+x`, `x-0`, `x*1`, `1*x`, `x/1`
//! - absorbing:    `x*0 → 0`, `0*x → 0`
//! - double negation: `-(-x) → x`, `!(!x) → x`
//! - constant-constant: `BinOp(c1, c2) → const` for `+ - * /` over i32/u32/f32
//!
//! Rules never keep borrows across mutations: constant folding extracts
//! operand values first, then interns the result.

use crate::ast::TypeName;
use crate::ssa::types::ConstantValue;
use polytype::Type;

use super::types::{EGraph, ENode, NodeId, PureOp};

impl EGraph {
    /// Try every rewrite rule and return a substitute node if any fires.
    pub(super) fn try_algebraic_fold(
        &mut self,
        op: &PureOp,
        operands: &[NodeId],
        result_ty: &Type<TypeName>,
    ) -> Option<NodeId> {
        match op {
            PureOp::Project { index } if operands.len() == 1 => self.fold_project(*index, operands[0]),
            PureOp::BinOp(name) if operands.len() == 2 => {
                let (a, b) = (operands[0], operands[1]);
                self.fold_binop_identity(name, a, b)
                    .or_else(|| self.fold_binop_const(name, a, b, result_ty))
            }
            PureOp::UnaryOp(name) if operands.len() == 1 => self.fold_unary(name, operands[0]),
            _ => None,
        }
    }

    /// `Project{i}(Tuple/Vector/ArrayLit(e0,…,en)) → e_i`
    fn fold_project(&self, index: u32, base: NodeId) -> Option<NodeId> {
        let ENode::Pure {
            op: base_op,
            operands: base_operands,
        } = &self.nodes[base]
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
    fn fold_binop_identity(&self, name: &str, a: NodeId, b: NodeId) -> Option<NodeId> {
        match name {
            "+" if self.is_zero_literal(a) => Some(b),
            "+" if self.is_zero_literal(b) => Some(a),
            "-" if self.is_zero_literal(b) => Some(a),
            "*" if self.is_one_literal(a) => Some(b),
            "*" if self.is_one_literal(b) => Some(a),
            "*" if self.is_zero_literal(a) => Some(a),
            "*" if self.is_zero_literal(b) => Some(b),
            "/" if self.is_one_literal(b) => Some(a),
            _ => None,
        }
    }

    /// Constant-constant fold for `+ - * /` when both operands are literals.
    /// Dispatches on the result type; skips division by zero.
    fn fold_binop_const(
        &mut self,
        name: &str,
        a: NodeId,
        b: NodeId,
        result_ty: &Type<TypeName>,
    ) -> Option<NodeId> {
        match result_ty {
            Type::Constructed(TypeName::Int(32), _) => {
                let (ai, bi) = (self.as_i32(a)?, self.as_i32(b)?);
                let v = eval_i32(name, ai, bi)?;
                Some(self.intern_constant(ConstantValue::I32(v), result_ty.clone()))
            }
            Type::Constructed(TypeName::UInt(32), _) => {
                let (au, bu) = (self.as_u32(a)?, self.as_u32(b)?);
                let v = eval_u32(name, au, bu)?;
                Some(self.intern_constant(ConstantValue::U32(v), result_ty.clone()))
            }
            Type::Constructed(TypeName::Float(32), _) => {
                let (af, bf) = (self.as_f32(a)?, self.as_f32(b)?);
                let v = eval_f32(name, af, bf)?;
                Some(self.intern_constant(ConstantValue::F32(v.to_bits()), result_ty.clone()))
            }
            _ => None,
        }
    }

    /// `-(-x) → x` and `!(!x) → x` (no-op for any other unary op).
    fn fold_unary(&self, name: &str, inner: NodeId) -> Option<NodeId> {
        if name != "-" && name != "!" {
            return None;
        }
        let ENode::Pure {
            op: PureOp::UnaryOp(inner_name),
            operands: inner_ops,
        } = &self.nodes[inner]
        else {
            return None;
        };
        (inner_name == name && inner_ops.len() == 1).then(|| inner_ops[0])
    }

    // ---- literal predicates ------------------------------------------------

    fn is_zero_literal(&self, nid: NodeId) -> bool {
        match &self.nodes[nid] {
            ENode::Constant(ConstantValue::I32(0)) | ENode::Constant(ConstantValue::U32(0)) => true,
            ENode::Constant(ConstantValue::F32(bits)) => f32::from_bits(*bits) == 0.0,
            ENode::Pure { op, .. } => match op {
                PureOp::Int(s) | PureOp::Uint(s) => s == "0",
                PureOp::Float(s) => s.parse::<f32>().map(|f| f == 0.0).unwrap_or(false),
                _ => false,
            },
            _ => false,
        }
    }

    fn is_one_literal(&self, nid: NodeId) -> bool {
        match &self.nodes[nid] {
            ENode::Constant(ConstantValue::I32(1)) | ENode::Constant(ConstantValue::U32(1)) => true,
            ENode::Constant(ConstantValue::F32(bits)) => f32::from_bits(*bits) == 1.0,
            ENode::Pure { op, .. } => match op {
                PureOp::Int(s) | PureOp::Uint(s) => s == "1",
                PureOp::Float(s) => s.parse::<f32>().map(|f| f == 1.0).unwrap_or(false),
                _ => false,
            },
            _ => false,
        }
    }

    // ---- value extractors --------------------------------------------------

    fn as_i32(&self, nid: NodeId) -> Option<i32> {
        match &self.nodes[nid] {
            ENode::Constant(ConstantValue::I32(v)) => Some(*v),
            ENode::Pure {
                op: PureOp::Int(s),
                operands,
            } if operands.is_empty() => s.parse().ok(),
            _ => None,
        }
    }

    fn as_u32(&self, nid: NodeId) -> Option<u32> {
        match &self.nodes[nid] {
            ENode::Constant(ConstantValue::U32(v)) => Some(*v),
            ENode::Pure {
                op: PureOp::Uint(s),
                operands,
            } if operands.is_empty() => s.parse().ok(),
            _ => None,
        }
    }

    fn as_f32(&self, nid: NodeId) -> Option<f32> {
        match &self.nodes[nid] {
            ENode::Constant(ConstantValue::F32(bits)) => Some(f32::from_bits(*bits)),
            ENode::Pure {
                op: PureOp::Float(s),
                operands,
            } if operands.is_empty() => s.parse().ok(),
            _ => None,
        }
    }
}

fn eval_i32(op: &str, a: i32, b: i32) -> Option<i32> {
    match op {
        "+" => a.checked_add(b),
        "-" => a.checked_sub(b),
        "*" => a.checked_mul(b),
        "/" if b != 0 => a.checked_div(b),
        _ => None,
    }
}

fn eval_u32(op: &str, a: u32, b: u32) -> Option<u32> {
    match op {
        "+" => a.checked_add(b),
        "-" => a.checked_sub(b),
        "*" => a.checked_mul(b),
        "/" if b != 0 => Some(a / b),
        _ => None,
    }
}

fn eval_f32(op: &str, a: f32, b: f32) -> Option<f32> {
    match op {
        "+" => Some(a + b),
        "-" => Some(a - b),
        "*" => Some(a * b),
        "/" if b != 0.0 => Some(a / b),
        _ => None,
    }
}
