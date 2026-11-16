//! Scoped wall-clock timings for egglog passes and their sub-passes.
use egglog_engine::{CommandOutput, EGraph};
use std::time::Instant;
pub(super) use wyn_base::timing::span;
use wyn_base::timing::Span;

/// Snapshot every registered table, including empty ones, without running rules.
pub(super) fn tables(graph: &EGraph, phase: &str) {
    if std::env::var_os("WYN_EGGLOG_TABLES").is_none() {
        return;
    }
    let mut names = graph.get_function_names();
    names.sort_unstable();
    for name in names {
        eprintln!("egglog snapshot\t{phase}\t{name}\t{}", graph.get_size(&name));
    }
}

/// Time the original parser/executor call without changing its schedule.
#[track_caller]
pub(super) fn evaluate(
    graph: &mut EGraph,
    filename: Option<String>,
    source: &str,
) -> Result<Vec<CommandOutput>, egglog_engine::Error> {
    if std::env::var_os("WYN_EGGLOG_CALLS").is_none() {
        return graph.parse_and_run_program(filename, source);
    }
    let started = Instant::now();
    let result = graph.parse_and_run_program(filename.clone(), source);
    let elapsed = started.elapsed();
    let label = filename.as_deref().unwrap_or(source).split_whitespace().collect::<Vec<_>>().join(" ");
    eprintln!(
        "egglog call\t{}\t{:.6}\t{label}",
        std::panic::Location::caller(),
        elapsed.as_secs_f64() * 1000.0
    );
    result
}

/// Print total egglog time to stderr, with pass and sub-pass timings when `verbose`.
/// Completed passes print immediately, including on errors. Timing is local to
/// this thread. Parent timings include their sub-passes.
pub fn with_timings<T>(verbose: bool, f: impl FnOnce() -> T) -> T {
    wyn_base::timing::with_enabled(verbose, || {
        let _span = Span::new("egglog");
        f()
    })
}
