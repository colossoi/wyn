//! Array dimensions use ordinary type terms so size variables follow substitution.
use super::{Type, TypeName};
use crate::op::BinaryOperator;
use polytype::{Context, UnificationError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizeOperator {
    Add,
    Subtract,
    Multiply,
}

impl SizeOperator {
    pub(crate) fn from_binary(operator: BinaryOperator) -> Option<Self> {
        match operator {
            BinaryOperator::Add => Some(Self::Add),
            BinaryOperator::Subtract => Some(Self::Subtract),
            BinaryOperator::Multiply => Some(Self::Multiply),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
        }
    }

    fn evaluate(self, left: usize, right: usize) -> Option<usize> {
        match self {
            Self::Add => left.checked_add(right),
            Self::Subtract => left.checked_sub(right),
            Self::Multiply => left.checked_mul(right),
        }
    }
}

/// Reduce a dimension once its operands are known. Returns false for a
/// closed expression whose value is outside the supported array length range.
pub(crate) fn fold(ty: &mut Type) -> bool {
    if let Type::Constructed(TypeName::SizeArithmetic(operator), operands) = ty {
        if let [Type::Constructed(TypeName::Size(left), _), Type::Constructed(TypeName::Size(right), _)] =
            operands.as_slice()
        {
            let Some(value) =
                operator.evaluate(*left, *right).filter(|&value| i64::try_from(value).is_ok())
            else {
                return false;
            };
            *ty = Type::Constructed(TypeName::Size(value), vec![]);
        }
    }
    true
}

pub(crate) fn fold_type(ty: &mut Type) -> bool {
    if let Type::Constructed(name, args) = ty {
        if let TypeName::Sum(variants) = name {
            if !variants.iter_mut().flat_map(|(_, types)| types).all(fold_type) {
                return false;
            }
        }
        if !args.iter_mut().all(fold_type) {
            return false;
        }
    }
    fold(ty)
}

pub(crate) fn contains_arithmetic(ty: &Type) -> bool {
    let Type::Constructed(name, args) = ty else {
        return false;
    };
    matches!(name, TypeName::SizeArithmetic(_))
        || matches!(name, TypeName::Sum(variants)
            if variants.iter().flat_map(|(_, payload)| payload).any(contains_arithmetic))
        || args.iter().any(contains_arithmetic)
}

/// Let ordinary unification handle symbolic expressions structurally. If a
/// shape comparison fails, retry with any now-concrete expressions reduced.
pub(crate) fn unify(
    context: &mut Context<TypeName>,
    left: &Type,
    right: &Type,
) -> Result<(), UnificationError<TypeName>> {
    let Err(error) = context.unify(left, right) else {
        return Ok(());
    };
    let UnificationError::Failure(a, b) = &error else {
        return Err(error);
    };
    if ![a, b].into_iter().any(|ty| matches!(ty, Type::Constructed(TypeName::SizeArithmetic(_), _))) {
        return Err(error);
    }
    let checkpoint = context.len();
    let result = unify_reduced(context, left, right);
    if result.is_err() {
        context.rollback(checkpoint);
    }
    result
}

fn unify_reduced(
    context: &mut Context<TypeName>,
    left: &Type,
    right: &Type,
) -> Result<(), UnificationError<TypeName>> {
    let mut left = left.apply(context);
    let mut right = right.apply(context);
    if !fold_type(&mut left) || !fold_type(&mut right) {
        return Err(UnificationError::Failure(left, right));
    }
    if let (Type::Constructed(a, xs), Type::Constructed(b, ys)) = (&left, &right) {
        if a == b && xs.len() == ys.len() {
            for (x, y) in xs.iter().zip(ys) {
                unify_reduced(context, x, y)?;
            }
            return Ok(());
        }
    }
    context.unify(&left, &right)
}
