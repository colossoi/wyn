//! Support code emitted with each Rust/WGPU host module.
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::mpsc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, MapMode, PollType, Queue, Sampler, Texture};

#[derive(Debug)]
pub struct HostError(pub String);

impl Display for HostError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.0.fmt(f)
    }
}
impl Error for HostError {}

pub enum Resource {
    Buffer(Buffer),
    Texture(Texture),
    Sampler(Sampler),
    Bytes(Vec<u8>),
}

pub fn size(value: &BigInt) -> Result<u64, HostError> {
    let Some(value) = value.to_u64() else {
        return Err(HostError(format!("invalid byte size: {value}")));
    };
    Ok(value)
}

pub fn dimension(value: &BigInt) -> Result<u32, HostError> {
    let Some(value) = value.to_u32() else {
        return Err(HostError(format!("invalid GPU dimension: {value}")));
    };
    Ok(value)
}

pub fn floor(a: BigInt, b: BigInt) -> Result<BigInt, HostError> {
    if b == BigInt::from(0) {
        return Err(HostError("division by zero".into()));
    }
    Ok(a.div_floor(&b))
}

pub fn ceiling(a: BigInt, b: BigInt) -> Result<BigInt, HostError> {
    if b == BigInt::from(0) {
        return Err(HostError("division by zero".into()));
    }
    Ok(a.div_ceil(&b))
}

pub fn read_scalar(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    offset: u32,
    signed: bool,
) -> Result<BigInt, HostError> {
    let offset = u64::from(offset);
    if offset % 4 != 0 || offset + 4 > buffer.size() {
        return Err(HostError("scalar read is outside its buffer or unaligned".into()));
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
    device.poll(PollType::wait_indefinitely()).map_err(|e| HostError(e.to_string()))?;
    receiver.recv().map_err(|e| HostError(e.to_string()))?.map_err(|e| HostError(e.to_string()))?;
    let mapped = staging.slice(..).get_mapped_range();
    let bytes: [u8; 4] =
        mapped.as_ref().try_into().map_err(|_| HostError("invalid scalar mapping".into()))?;
    let value = if signed {
        BigInt::from(i32::from_le_bytes(bytes))
    } else {
        BigInt::from(u32::from_le_bytes(bytes))
    };
    drop(mapped);
    staging.unmap();
    Ok(value)
}

pub fn read_host_scalar(bytes: &[u8], offset: u32, signed: bool) -> Result<BigInt, HostError> {
    let start = offset as usize;
    let Some(end) = start.checked_add(4) else {
        return Err(HostError("scalar offset overflow".into()));
    };
    let Some(bytes) = bytes.get(start..end) else {
        return Err(HostError("scalar outside host span".into()));
    };
    let bytes: [u8; 4] = bytes.try_into().map_err(|_| HostError("invalid scalar span".into()))?;
    Ok(if signed {
        BigInt::from(i32::from_le_bytes(bytes))
    } else {
        BigInt::from(u32::from_le_bytes(bytes))
    })
}

pub fn draw_end(first: u32, count: u32) -> Result<u32, HostError> {
    let Some(end) = first.checked_add(count) else {
        return Err(HostError("draw index range overflow".into()));
    };
    Ok(end)
}

pub fn modulo(a: BigInt, b: BigInt) -> Result<BigInt, HostError> {
    if b == BigInt::from(0) {
        return Err(HostError("division by zero".into()));
    }
    Ok(a.mod_floor(&b))
}
