//! Scoped wall-clock timings for egglog passes and their sub-passes.
use super::OptimizeError;
use egglog_engine::ast::{Command, Schedule};
use egglog_engine::{CommandOutput, EGraph};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use wyn_base::timing::Span;
pub(super) use wyn_base::timing::{span, time};

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

/// Opt-in schedule tracing; program values remain entirely in egglog.
#[track_caller]
pub(super) fn run(graph: &mut EGraph, source: &str) -> Result<(), OptimizeError> {
    // Bracket each schedule so native deltas exclude work between passes.
    report(graph);
    if std::env::var_os("WYN_EGGLOG_TRACE").is_none() {
        evaluate(graph, None, source)?;
        report(graph);
        return Ok(());
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let invocation = NEXT.fetch_add(1, Ordering::Relaxed);
    for command in super::parse_program("schedule trace", source)? {
        if let Command::RunSchedule(schedule) = command {
            trace(graph, &schedule, &format!("{invocation}"))?;
        } else {
            graph.run_program(vec![command])?;
        }
    }
    report(graph);
    Ok(())
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

fn report(graph: &EGraph) {
    if std::env::var_os("WYN_EGGLOG_REPORT").is_none() {
        return;
    }
    let report = graph.get_overall_run_report();
    eprintln!("egglog native report");
    for (rule, time) in &report.search_and_apply_time_per_rule {
        eprintln!(
            "egglog rule\t{:.9}\t{}\t{rule:?}",
            time.as_secs_f64(),
            report.num_matches_per_rule.get(rule).copied().unwrap_or(0)
        );
    }
    for (ruleset, time) in &report.search_and_apply_time_per_ruleset {
        eprintln!(
            "egglog native set\t{ruleset}\t{:.9}\t{:.9}\t{:.9}",
            time.as_secs_f64(),
            report.merge_time_per_ruleset.get(ruleset).copied().unwrap_or_default().as_secs_f64(),
            report.rebuild_time_per_ruleset.get(ruleset).copied().unwrap_or_default().as_secs_f64()
        );
    }
}

fn trace(graph: &mut EGraph, schedule: &Schedule, path: &str) -> Result<bool, OptimizeError> {
    let mut updated = false;
    match schedule {
        Schedule::Saturate(_, body) => {
            for iteration in 0.. {
                let changed = trace(graph, body, &format!("{path}/i{iteration}"))?;
                updated |= changed;
                if !path.split('/').any(|part| part.starts_with('i')) {
                    let query = super::query::Query(graph);
                    let current = query.row("CurrentProgram", |_| true)?.unwrap()[0];
                    let mut active = 0;
                    query.for_each("ProgramFunction", |row| {
                        active += usize::from(row[0] == current);
                        Ok(())
                    })?;
                    eprintln!("egglog active functions\t{path}/i{iteration}\t{active}");
                    for name in [
                        "MemoryProgram",
                        "ContainsStm",
                        "CheckStatement",
                        "LogicalOuterCount",
                        "ProductComponent",
                        "ProjectionPath",
                        "LengthSource",
                        "CountLength",
                    ] {
                        eprintln!(
                            "egglog table\t{path}/i{iteration}\t{name}\t{}",
                            graph.print_size(Some(name))?
                        );
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        Schedule::Sequence(_, stages) => {
            for (index, stage) in stages.iter().enumerate() {
                updated |= trace(graph, stage, &format!("{path}/s{index}"))?;
            }
        }
        Schedule::Run(..) | Schedule::Repeat(..) => {
            let started = Instant::now();
            let outputs = graph.run_program(vec![Command::RunSchedule(schedule.clone())])?;
            eprintln!(
                "egglog run\t{path}\t{:.6}\t{schedule}",
                started.elapsed().as_secs_f64()
            );
            for output in outputs {
                if let CommandOutput::RunSchedule(report) = output {
                    updated |= report.updated;
                }
            }
        }
    }
    Ok(updated)
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
