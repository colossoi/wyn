//! Scoped wall-clock timings for egglog passes and their sub-passes.
use wyn_base::timing::Span;
pub(super) use wyn_base::timing::{enabled, span, time};

/// Print total egglog time to stderr, with pass and sub-pass timings when `verbose`.
/// Completed passes print immediately, including on errors. Timing is local to
/// this thread. Parent timings include their sub-passes.
pub fn with_timings<T>(verbose: bool, f: impl FnOnce() -> T) -> T {
    wyn_base::timing::with_enabled(verbose, || {
        let _span = Span::new("egglog");
        f()
    })
}
