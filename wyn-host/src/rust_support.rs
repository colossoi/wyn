//! Private support for generated host arithmetic and draw commands.
pub use arithmetic::{ceiling, dimension, floor, size};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::mpsc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode, PollType, Queue};

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

pub fn scratch_buffer(
    device: &Device,
    buffers: &mut BTreeMap<usize, Buffer>,
    slot: usize,
    descriptor: &BufferDescriptor<'_>,
) -> Buffer {
    if let Some(buffer) = buffers.get(&slot) {
        if buffer.size() == descriptor.size {
            return buffer.clone();
        }
    }
    let buffer = device.create_buffer(descriptor);
    buffers.insert(slot, buffer.clone());
    buffer
}

pub fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, HostError> {
    if bytes.len() < 20 || bytes.len() % 4 != 0 {
        return Err(HostError::Invalid("invalid SPIR-V module size".into()));
    }
    let words: Vec<_> =
        bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if words[0] != 0x0723_0203 {
        return Err(HostError::Invalid("invalid SPIR-V module header".into()));
    }
    Ok(words)
}

pub fn push_constant_bytes(bytes: &[u8], size: u32) -> Result<&[u8], HostError> {
    let Some(bytes) = bytes.get(..size as usize) else {
        return Err(HostError::Invalid("push constant input is too short".into()));
    };
    Ok(bytes)
}

fn read_gpu_word(
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    buffer: &Buffer,
    offset: u32,
) -> Result<[u8; 4], HostError> {
    let offset = u64::from(offset);
    if offset % 4 != 0 || offset + 4 > buffer.size() {
        return Err(HostError::Invalid(
            "host arithmetic scalar lies outside its buffer or is unaligned".into(),
        ));
    }
    if !buffer.usage().contains(BufferUsages::COPY_SRC) {
        return Err(HostError::Invalid(
            "host arithmetic scalar buffer requires COPY_SRC usage".into(),
        ));
    }
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("Wyn host arithmetic scalar"),
        size: 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, 4);
    // Include all producers and the readback copy in the same submission.
    let pending = std::mem::replace(encoder, device.create_command_encoder(&Default::default()));
    queue.submit(Some(pending.finish()));
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
    let bytes = mapped
        .as_ref()
        .try_into()
        .map_err(|_| HostError::Invalid("invalid host arithmetic scalar mapping".into()))?;
    drop(mapped);
    staging.unmap();
    Ok(bytes)
}

pub fn scalar_bytes(bytes: &[u8], offset: u32) -> Result<[u8; 4], HostError> {
    let start = usize::try_from(offset)
        .map_err(|_| HostError::Invalid("host scalar offset exceeds address space".into()))?;
    let Some(end) = start.checked_add(4) else {
        return Err(HostError::Invalid("host scalar offset overflow".into()));
    };
    let Some(bytes) = bytes.get(start..end) else {
        return Err(HostError::Invalid(
            "scalar lies outside its host byte span".into(),
        ));
    };
    bytes.try_into().map_err(|_| HostError::Invalid("invalid host scalar byte span".into()))
}

pub fn read_i32(
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    buffer: &Buffer,
    offset: u32,
) -> Result<i32, HostError> {
    Ok(i32::from_le_bytes(read_gpu_word(
        device, queue, encoder, buffer, offset,
    )?))
}

pub fn read_u32(
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    buffer: &Buffer,
    offset: u32,
) -> Result<u32, HostError> {
    Ok(u32::from_le_bytes(read_gpu_word(
        device, queue, encoder, buffer, offset,
    )?))
}

pub fn read_f32(
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    buffer: &Buffer,
    offset: u32,
) -> Result<f32, HostError> {
    Ok(f32::from_le_bytes(read_gpu_word(
        device, queue, encoder, buffer, offset,
    )?))
}

pub fn write_buffer(
    device: &Device,
    encoder: &mut CommandEncoder,
    buffer: &Buffer,
    offset: u64,
    bytes: &[u8],
) {
    // Queue::write_buffer executes before the entire next submission. Record
    // a copy instead so uploads retain their order relative to clears and uses,
    // including multiple generated calls recorded into one caller-owned encoder.
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("Wyn scalar upload"),
        size: bytes.len() as u64,
        usage: BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    staging.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
    staging.unmap();
    encoder.copy_buffer_to_buffer(&staging, 0, buffer, offset, bytes.len() as u64);
}

pub fn draw_end(first: u32, count: u32) -> Result<u32, HostError> {
    let Some(end) = first.checked_add(count) else {
        return Err(HostError::Invalid("draw index range overflow".into()));
    };
    Ok(end)
}
