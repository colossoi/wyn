//! Checked byte decoding shared by the compiler tests and emitted host code.
use super::HostError;
use std::ops::Range;

/// A single buffer copy packed into a shared staging allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyRange {
    pub source: Range<u64>,
    pub staging_offset: u64,
    pub mapped: Range<u64>,
}

/// Plan aligned copies while preserving the exact requested byte spans.
/// The returned size is zero when every requested span is empty.
pub fn copy_ranges(requests: &[(Range<u64>, u64)]) -> Result<(Vec<CopyRange>, u64), HostError> {
    let mut copies = Vec::new();
    let mut total = 0u64;
    for (range, capacity) in requests {
        if range.start > range.end || range.end > *capacity {
            return Err(HostError::Invalid("result read lies outside its buffer".into()));
        }
        if range.is_empty() {
            copies.push(CopyRange {
                source: 0..0,
                staging_offset: total,
                mapped: total..total,
            });
            continue;
        }
        let start = range.start / 4 * 4;
        let Some(end) = range.end.checked_add(3).map(|end| end / 4 * 4) else {
            return Err(HostError::Invalid("result copy alignment overflow".into()));
        };
        if end > *capacity {
            return Err(HostError::Invalid(
                "aligned result copy exceeds its backing buffer".into(),
            ));
        }
        let Some(next) = total.checked_add(end - start) else {
            return Err(HostError::Invalid("result staging size overflow".into()));
        };
        copies.push(CopyRange {
            source: start..end,
            staging_offset: total,
            mapped: (total + (range.start - start))..(total + (range.end - start)),
        });
        total = next;
    }
    Ok((copies, total))
}

pub fn bytes<const N: usize>(data: &[u8], offset: u64) -> Result<[u8; N], HostError> {
    let start = usize::try_from(offset)
        .map_err(|_| HostError::Invalid("result byte offset exceeds address space".into()))?;
    let Some(end) = start.checked_add(N) else {
        return Err(HostError::Invalid("result byte offset overflow".into()));
    };
    let Some(span) = data.get(start..end) else {
        return Err(HostError::Invalid(
            "result value lies outside its byte span".into(),
        ));
    };
    span.try_into().map_err(|_| HostError::Invalid("invalid result byte span".into()))
}

pub fn at(offset: u64, field_offset: u32) -> Result<u64, HostError> {
    let Some(offset) = offset.checked_add(u64::from(field_offset)) else {
        return Err(HostError::Invalid("result field offset overflow".into()));
    };
    Ok(offset)
}

pub fn element_range(elements: Range<u32>, stride: u32, capacity: u64) -> Result<Range<u64>, HostError> {
    if stride == 0 || elements.start > elements.end {
        return Err(HostError::Invalid(
            "invalid result element range or stride".into(),
        ));
    }
    let start = u64::from(elements.start) * u64::from(stride);
    let end = u64::from(elements.end) * u64::from(stride);
    if end > capacity {
        return Err(HostError::Invalid(
            "result element range exceeds its backing buffer".into(),
        ));
    }
    Ok(start..end)
}

pub fn array<T>(
    data: &[u8],
    count: u32,
    stride: u32,
    decode: impl Fn(&[u8], u64) -> Result<T, HostError>,
) -> Result<Vec<T>, HostError> {
    element_range(0..count, stride, data.len() as u64)?;
    let mut values = Vec::new();
    let count_usize = usize::try_from(count)
        .map_err(|_| HostError::Invalid("result count exceeds address space".into()))?;
    values
        .try_reserve_exact(count_usize)
        .map_err(|e| HostError::Invalid(format!("result allocation: {e}")))?;
    for index in 0..count {
        values.push(decode(data, u64::from(index) * u64::from(stride))?);
    }
    Ok(values)
}
