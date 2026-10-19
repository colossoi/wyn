use super::*;

#[test]
fn integer_arithmetic_wraps_at_each_declared_width() {
    for bits in [8, 16, 32, 64] {
        let signed = Type::Constructed(TypeName::Int(bits), vec![]);
        let unsigned = Type::Constructed(TypeName::UInt(bits), vec![]);
        let min = (-(1_i128 << (bits - 1))) as i64;
        let max = ((1_i128 << (bits - 1)) - 1) as i64;
        let mask = ((1_u128 << bits) - 1) as i64;
        for (ty, op, lhs, rhs, expected) in [
            (&signed, BinaryOperator::Add, max, 1, min),
            (&signed, BinaryOperator::Subtract, min, 1, max),
            (&signed, BinaryOperator::Multiply, min, -1, min),
            (&unsigned, BinaryOperator::Add, mask, 1, 0),
            (&unsigned, BinaryOperator::Subtract, 0, 1, mask),
            (&unsigned, BinaryOperator::Multiply, mask, mask, 1),
        ] {
            assert_eq!(
                binary(op, Scalar::Int(lhs), Scalar::Int(rhs), ty),
                Some(Scalar::Int(expected)),
                "{ty:?}: {lhs} {op:?} {rhs}"
            );
        }
    }
}

#[test]
fn integer_shifts_respect_signedness_and_width() {
    for bits in [8, 16, 32, 64] {
        let signed = Type::Constructed(TypeName::Int(bits), vec![]);
        let unsigned = Type::Constructed(TypeName::UInt(bits), vec![]);
        let high_bit = (1_u64 << (bits - 1)) as i64;
        let mask = ((1_u128 << bits) - 1) as i64;
        for (ty, op, lhs, rhs, expected) in [
            (&unsigned, BinaryOperator::ShiftLeft, 1, 0, 1),
            (&unsigned, BinaryOperator::ShiftLeft, 1, bits as i64 - 1, high_bit),
            (&unsigned, BinaryOperator::ShiftLeft, high_bit, 1, 0),
            (&unsigned, BinaryOperator::ShiftRight, mask, bits as i64 - 1, 1),
            (&signed, BinaryOperator::ShiftRight, -2, 1, -1),
            (&signed, BinaryOperator::ShiftRightLogical, -1, bits as i64 - 1, 1),
        ] {
            assert_eq!(
                binary(op, Scalar::Int(lhs), Scalar::Int(rhs), ty),
                Some(Scalar::Int(expected)),
                "{ty:?}: {lhs} {op:?} {rhs}"
            );
        }
        for ty in [&signed, &unsigned] {
            for op in [
                BinaryOperator::ShiftLeft,
                BinaryOperator::ShiftRight,
                BinaryOperator::ShiftRightLogical,
            ] {
                for count in [-1, bits as i64, bits as i64 + 1] {
                    assert_eq!(binary(op, Scalar::Int(1), Scalar::Int(count), ty), None);
                }
            }
        }
    }
}
