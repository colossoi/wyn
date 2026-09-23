//! Scoped wall-clock timings for egglog passes and their sub-passes.
use std::cell::Cell;
use wyn_base::timing::Span;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

struct Report {
    _span: Span<'static>,
    previous: bool,
}

/// Print total egglog time to stderr, with pass and sub-pass timings when `verbose`.
/// Completed passes print immediately, including on errors. Timing is local to
/// this thread. Parent timings include their sub-passes.
pub fn with_timings<T>(verbose: bool, f: impl FnOnce() -> T) -> T {
    let previous = ENABLED.replace(verbose);
    let _report = Report {
        _span: Span::new("egglog"),
        previous,
    };
    f()
}

impl Drop for Report {
    fn drop(&mut self) {
        ENABLED.set(self.previous);
    }
}

pub(super) fn span(name: &'static str) -> Option<Span<'static>> {
    ENABLED.get().then(|| Span::new(name))
}

pub(super) fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let _timing = span(name);
    f()
}
