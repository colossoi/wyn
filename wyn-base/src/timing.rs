//! Scoped wall-clock timings.
use std::cell::Cell;
use std::time::Instant;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

struct Enabled(bool);

impl Drop for Enabled {
    fn drop(&mut self) {
        ENABLED.set(self.0);
    }
}

/// Enable or disable pass timings on this thread for the duration of `f`.
/// Restore the previous setting on return or unwind.
pub fn with_enabled<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    let _restore = Enabled(ENABLED.replace(enabled));
    f()
}

pub fn enabled() -> bool {
    ENABLED.get()
}

/// Parent spans include their children; repeated names represent separate calls.
pub fn span(name: &str) -> Option<Span<'_>> {
    enabled().then(|| {
        eprintln!("{name}: starting");
        Span::new(name)
    })
}

pub fn time<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let _span = span(name);
    f()
}

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

#[cfg(test)]
#[path = "timing_tests.rs"]
mod tests;
