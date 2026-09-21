//! Support code emitted with each Rust/WGPU host module.
pub use arithmetic::{ceiling, dimension, floor, size};
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::Range;
use std::sync::mpsc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, MapMode, PollType, Queue};

#[derive(Debug)]
pub enum HostError {
    Invalid(String),
}

impl Display for HostError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Invalid(message) => message.fmt(f),
        }
    }
}
impl Error for HostError {}

pub fn read_buffers(
    device: &Device,
    queue: &Queue,
    requests: &[(&Buffer, Range<u64>)],
) -> Result<Vec<Vec<u8>>, HostError> {
    let ranges: Vec<_> = requests.iter().map(|(buffer, range)| (range.clone(), buffer.size())).collect();
    let (copies, size) = readback::copy_ranges(&ranges)?;
    if size == 0 {
        return Ok(vec![Vec::new(); requests.len()]);
    }
    for ((buffer, _), copy) in requests.iter().zip(&copies) {
        if !copy.source.is_empty() && !buffer.usage().contains(BufferUsages::COPY_SRC) {
            return Err(HostError::Invalid("result buffer requires COPY_SRC usage".into()));
        }
    }
    if size > device.limits().max_buffer_size {
        return Err(HostError::Invalid(
            "result staging allocation exceeds device limits".into(),
        ));
    }
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("Wyn result readback"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for ((buffer, _), copy) in requests.iter().zip(&copies) {
        if !copy.source.is_empty() {
            encoder.copy_buffer_to_buffer(
                buffer,
                copy.source.start,
                &staging,
                copy.staging_offset,
                copy.source.end - copy.source.start,
            );
        }
    }
    queue.submit(Some(encoder.finish()));
    let (sender, receiver) = mpsc::channel();
    staging.slice(..).map_async(MapMode::Read, move |result| {
        // Dropping the receiver cancels observation of this completed mapping.
        if sender.send(result).is_err() {
            return;
        }
    });
    device.poll(PollType::wait_indefinitely()).map_err(|e| HostError::Invalid(e.to_string()))?;
    receiver
        .recv()
        .map_err(|e| HostError::Invalid(e.to_string()))?
        .map_err(|e| HostError::Invalid(e.to_string()))?;
    let mapped = staging.slice(..).get_mapped_range();
    let mut values = Vec::new();
    for copy in copies {
        let first = usize::try_from(copy.mapped.start)
            .map_err(|_| HostError::Invalid("result offset exceeds address space".into()))?;
        let last = usize::try_from(copy.mapped.end)
            .map_err(|_| HostError::Invalid("result end exceeds address space".into()))?;
        let Some(span) = mapped.get(first..last) else {
            return Err(HostError::Invalid("invalid result mapping".into()));
        };
        values.push(span.to_vec());
    }
    drop(mapped);
    staging.unmap();
    Ok(values)
}

fn read_bytes(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    range: Range<u64>,
) -> Result<Vec<u8>, HostError> {
    let Some(bytes) = read_buffers(device, queue, &[(buffer, range)])?.pop() else {
        return Err(HostError::Invalid("missing scalar readback".into()));
    };
    Ok(bytes)
}

pub fn read_i32(device: &Device, queue: &Queue, buffer: &Buffer, offset: u32) -> Result<i32, HostError> {
    let offset = u64::from(offset);
    Ok(i32::from_le_bytes(readback::bytes::<4>(
        &read_bytes(device, queue, buffer, offset..offset + 4)?,
        0,
    )?))
}

pub fn read_u32(device: &Device, queue: &Queue, buffer: &Buffer, offset: u32) -> Result<u32, HostError> {
    let offset = u64::from(offset);
    Ok(u32::from_le_bytes(readback::bytes::<4>(
        &read_bytes(device, queue, buffer, offset..offset + 4)?,
        0,
    )?))
}

pub fn draw_end(first: u32, count: u32) -> Result<u32, HostError> {
    let Some(end) = first.checked_add(count) else {
        return Err(HostError::Invalid("draw index range overflow".into()));
    };
    Ok(end)
}
