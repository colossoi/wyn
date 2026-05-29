use super::*;
use crate::builtins::names::{INTRINSIC_DOT, INTRINSIC_LENGTH};

#[test]
fn intrinsic_arity_for_length_is_one() {
    // length: [n]A -> i32
    assert_eq!(intrinsic_arity(INTRINSIC_LENGTH), Some(1));
}

#[test]
fn intrinsic_arity_for_dot_is_two() {
    // dot: vecN A -> vecN A -> A
    assert_eq!(intrinsic_arity(INTRINSIC_DOT), Some(2));
}

#[test]
fn intrinsic_arity_for_unknown_is_none() {
    assert_eq!(intrinsic_arity("definitely_not_a_real_intrinsic"), None);
}
