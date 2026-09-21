//! Fixed-width arithmetic for host capacities and launch dimensions.
use super::HostError;

pub fn size(value: i64) -> Result<u64, HostError> {
    u64::try_from(value).map_err(|_| HostError::Invalid(format!("invalid byte size: {value}")))
}

pub fn dimension(value: i64) -> Result<u32, HostError> {
    u32::try_from(value).map_err(|_| HostError::Invalid(format!("invalid GPU dimension: {value}")))
}

pub fn signed_size(value: u64) -> Result<i64, HostError> {
    i64::try_from(value).map_err(|_| HostError::Invalid(format!("byte size exceeds i64: {value}")))
}

pub fn i32_value(value: i64) -> Result<i32, HostError> {
    i32::try_from(value).map_err(|_| HostError::Invalid(format!("value exceeds i32: {value}")))
}

pub fn u32_value(value: i64) -> Result<u32, HostError> {
    u32::try_from(value).map_err(|_| HostError::Invalid(format!("value exceeds u32: {value}")))
}

pub fn add(a: i64, b: i64) -> Result<i64, HostError> {
    let Some(value) = a.checked_add(b) else {
        return Err(HostError::Invalid("i64 addition overflow".into()));
    };
    Ok(value)
}

pub fn subtract(a: i64, b: i64) -> Result<i64, HostError> {
    let Some(value) = a.checked_sub(b) else {
        return Err(HostError::Invalid("i64 subtraction overflow".into()));
    };
    Ok(value)
}

pub fn multiply(a: i64, b: i64) -> Result<i64, HostError> {
    let Some(value) = a.checked_mul(b) else {
        return Err(HostError::Invalid("i64 multiplication overflow".into()));
    };
    Ok(value)
}

fn quotient(a: i64, b: i64) -> Result<(i64, i64), HostError> {
    let Some(quotient) = a.checked_div(b) else {
        return Err(HostError::Invalid(
            "division by zero or i64 quotient overflow".into(),
        ));
    };
    Ok((quotient, a % b))
}

pub fn floor(a: i64, b: i64) -> Result<i64, HostError> {
    let (q, r) = quotient(a, b)?;
    if r != 0 && (r < 0) != (b < 0) {
        subtract(q, 1)
    } else {
        Ok(q)
    }
}

pub fn ceiling(a: i64, b: i64) -> Result<i64, HostError> {
    let (q, r) = quotient(a, b)?;
    if r != 0 && (r < 0) == (b < 0) {
        add(q, 1)
    } else {
        Ok(q)
    }
}

pub fn modulo(a: i64, b: i64) -> Result<i64, HostError> {
    let (_, r) = quotient(a, b)?;
    if r != 0 && (r < 0) != (b < 0) {
        add(r, b)
    } else {
        Ok(r)
    }
}
