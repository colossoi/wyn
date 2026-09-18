//! Scoped wall-clock timings.
use std::time::Instant;

/// Print a named duration to stderr when the span leaves scope, including on errors.
#[must_use = "keep the span alive until the timed work completes"]
pub struct Span<'a> {
    name: &'a str,
    start: Instant,
}

impl<'a> Span<'a> {
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        eprintln!(
            "{}: {:.3} ms",
            self.name,
            self.start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
