//! Support code emitted with each Rust/WGPU host module.
pub use arithmetic::{ceiling, dimension, floor, size};
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::mpsc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, MapMode, PollType, Queue, Sampler, Texture};

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

pub enum Resource {
    Buffer(Buffer),
    Texture(Texture),
    Sampler(Sampler),
    Bytes(Vec<u8>),
}

pub fn read_scalar(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    offset: u32,
    signed: bool,
) -> Result<i64, HostError> {
    let offset = u64::from(offset);
    if offset % 4 != 0 || offset + 4 > buffer.size() {
        return Err(HostError::Invalid(
            "scalar read is outside its buffer or unaligned".into(),
        ));
    }
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("Wyn scalar readback"),
        size: 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, 4);
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
    let bytes: [u8; 4] =
        mapped.as_ref().try_into().map_err(|_| HostError::Invalid("invalid scalar mapping".into()))?;
    let value =
        if signed { i64::from(i32::from_le_bytes(bytes)) } else { i64::from(u32::from_le_bytes(bytes)) };
    drop(mapped);
    staging.unmap();
    Ok(value)
}

pub fn read_host_scalar(bytes: &[u8], offset: u32, signed: bool) -> Result<i64, HostError> {
    let start = offset as usize;
    let Some(end) = start.checked_add(4) else {
        return Err(HostError::Invalid("scalar offset overflow".into()));
    };
    let Some(bytes) = bytes.get(start..end) else {
        return Err(HostError::Invalid("scalar outside host span".into()));
    };
    let bytes: [u8; 4] = bytes.try_into().map_err(|_| HostError::Invalid("invalid scalar span".into()))?;
    Ok(if signed { i64::from(i32::from_le_bytes(bytes)) } else { i64::from(u32::from_le_bytes(bytes)) })
}

pub fn draw_end(first: u32, count: u32) -> Result<u32, HostError> {
    let Some(end) = first.checked_add(count) else {
        return Err(HostError::Invalid("draw index range overflow".into()));
    };
    Ok(end)
}
