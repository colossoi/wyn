//! Opt-in diagnostics distinguish driver compilation, GPU work and host waiting.
use super::{gpu_error, Result};
use std::time::Instant;
use wgpu::{Buffer, BufferUsages, CommandEncoder, Device, QuerySet};

const QUERIES: u32 = 2048;

pub(super) struct Span {
    start: Option<Instant>,
    kind: &'static str,
    name: String,
}
impl Span {
    pub fn new(kind: &'static str, name: &str) -> Self {
        let start = std::env::var_os("WYN_GPU_PROFILE").map(|_| {
            eprintln!("WYN_PROFILE_BEGIN\t{kind}\t{name}");
            Instant::now()
        });
        Self {
            start,
            kind,
            name: name.into(),
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            eprintln!(
                "WYN_PROFILE_CPU\t{}\t{}\t{}",
                self.kind,
                self.name,
                start.elapsed().as_nanos()
            );
        }
    }
}

pub(super) struct Pass {
    pub queries: QuerySet,
    pub start: u32,
}
pub(super) struct Profile {
    enabled: bool,
    query_set: Option<QuerySet>,
    labels: Vec<String>,
    pending: Vec<(Buffer, Vec<String>)>,
    submissions: usize,
}
impl Profile {
    pub fn new() -> Self {
        Self {
            enabled: std::env::var_os("WYN_GPU_PROFILE").is_some(),
            query_set: None,
            labels: Vec::new(),
            pending: Vec::new(),
            submissions: 0,
        }
    }
    pub fn full(&self) -> bool {
        self.labels.len() * 2 >= QUERIES as usize
    }
    pub fn discard_unsubmitted(&mut self) {
        self.query_set = None;
        self.labels.clear();
    }
    pub fn pass(&mut self, device: &Device, label: &str) -> Option<Pass> {
        if !self.enabled || !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let queries = self
            .query_set
            .get_or_insert_with(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("WHL phase profile"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERIES,
                })
            })
            .clone();
        let start = self.labels.len() as u32 * 2;
        self.labels.push(label.into());
        Some(Pass { queries, start })
    }
    pub fn resolve(&mut self, device: &Device, encoder: &mut CommandEncoder) {
        if !self.enabled {
            return;
        }
        self.submissions += 1;
        let Some(queries) = self.query_set.take() else {
            return;
        };
        let count = self.labels.len() as u32 * 2;
        let bytes = u64::from(count) * 8;
        let resolved = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WHL timestamps"),
            size: bytes,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WHL timestamp readback"),
            size: bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(&queries, 0..count, &resolved, 0);
        encoder.copy_buffer_to_buffer(&resolved, 0, &readback, 0, bytes);
        self.pending.push((readback, std::mem::take(&mut self.labels)));
    }
    pub fn collect(&mut self, device: &Device, period: f32, frame: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        for (buffer, labels) in self.pending.drain(..) {
            let (send, receive) = std::sync::mpsc::sync_channel(1);
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
            device.poll(wgpu::PollType::wait_indefinitely()).map_err(gpu_error)?;
            receive.recv().map_err(gpu_error)?.map_err(gpu_error)?;
            let data = buffer.slice(..).get_mapped_range();
            for (label, pair) in labels.iter().zip(data.chunks_exact(16)) {
                let begin = u64::from_le_bytes(pair[..8].try_into().map_err(gpu_error)?);
                let end = u64::from_le_bytes(pair[8..].try_into().map_err(gpu_error)?);
                eprintln!(
                    "WYN_PROFILE_GPU\t{frame}\t{label}\t{}",
                    end.wrapping_sub(begin) as f64 * f64::from(period)
                );
            }
            drop(data);
            buffer.unmap();
        }
        eprintln!("WYN_PROFILE_SUBMISSIONS\t{frame}\t{}", self.submissions);
        self.submissions = 0;
        Ok(())
    }
}
