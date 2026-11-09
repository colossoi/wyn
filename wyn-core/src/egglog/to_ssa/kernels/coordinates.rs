//! Row-major domains and input indexing; the last axis varies fastest.

use super::super::{Body, OptimizeError, Typed};
use crate::op::BinaryOperator::{Divide, Remainder};
use crate::types;

/// Call only inside the domain's bounds, so an empty dimension is never divided by.
pub fn decode(
    body: &mut Body<'_, '_, '_>,
    dimensions: &[Typed],
    index: Typed,
) -> Result<Vec<Typed>, OptimizeError> {
    let zero = body.literal(0, &types::i32())?;
    let mut coordinates = vec![zero; dimensions.len()];
    let mut remainder = index;
    for i in (0..dimensions.len()).rev() {
        coordinates[i] = body.binary(Remainder, remainder.clone(), dimensions[i].clone())?;
        remainder = body.binary(Divide, remainder, dimensions[i].clone())?;
    }
    Ok(coordinates)
}
