//! Scoped wall-clock timings, aggregated by pass and nested activity.
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

thread_local! {
    static CURRENT: RefCell<Option<Rc<RefCell<Profile>>>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct Profile {
    rows: Vec<Row>,
    stack: Vec<usize>,
}

struct Row {
    parent: Option<usize>,
    name: &'static str,
    elapsed: Duration,
    calls: usize,
}

struct Report {
    start: Instant,
    previous: Option<Rc<RefCell<Profile>>>,
}

/// Print total egglog time to stderr, with nested pass timings when `detailed`.
/// Times are inclusive wall times; repeated activities are summed with a call
/// count. Completed passes print immediately, including on errors. Collection is
/// local to this thread.
pub fn with_timings<T>(detailed: bool, f: impl FnOnce() -> T) -> T {
    let profile = Rc::new(RefCell::new(Profile::default()));
    let previous = CURRENT.with(|current| current.replace(detailed.then(|| profile.clone())));
    let _report = Report {
        start: Instant::now(),
        previous,
    };
    f()
}

impl Drop for Report {
    fn drop(&mut self) {
        CURRENT.with(|current| current.replace(self.previous.take()));
        eprintln!("egglog: {:.3} ms", self.start.elapsed().as_secs_f64() * 1000.0);
    }
}

fn print(rows: &[Row], index: usize, depth: usize) {
    let row = &rows[index];
    let count = if row.calls > 1 { format!(" ({} calls)", row.calls) } else { String::new() };
    let prefix = if depth == 0 { "egglog ".into() } else { "  ".repeat(depth) };
    eprintln!(
        "{prefix}{}: {:.3} ms{count}",
        row.name,
        row.elapsed.as_secs_f64() * 1000.0
    );
    for (child, _) in rows.iter().enumerate().filter(|(_, row)| row.parent == Some(index)) {
        print(rows, child, depth + 1);
    }
}

pub(super) struct Span(Option<(Rc<RefCell<Profile>>, usize, Instant)>);

pub(super) fn span(name: &'static str) -> Span {
    CURRENT.with(|current| {
        let Some(profile) = current.borrow().clone() else {
            return Span(None);
        };
        let mut p = profile.borrow_mut();
        let parent = p.stack.last().copied();
        let index = match p.rows.iter().position(|row| row.parent == parent && row.name == name) {
            Some(index) => index,
            None => {
                let index = p.rows.len();
                p.rows.push(Row {
                    parent,
                    name,
                    elapsed: Duration::ZERO,
                    calls: 0,
                });
                index
            }
        };
        p.stack.push(index);
        drop(p);
        Span(Some((profile, index, Instant::now())))
    })
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((profile, index, start)) = &self.0 {
            let elapsed = start.elapsed();
            let mut p = profile.borrow_mut();
            p.stack.pop();
            p.rows[*index].elapsed += elapsed;
            p.rows[*index].calls += 1;
            if p.stack.is_empty() {
                print(&p.rows, *index, 0);
            } else if elapsed >= Duration::from_secs(1) {
                let path = p
                    .stack
                    .iter()
                    .chain(std::iter::once(index))
                    .map(|&index| p.rows[index].name)
                    .collect::<Vec<_>>()
                    .join(" / ");
                eprintln!("egglog activity {path}: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
            }
        }
    }
}

pub(super) fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let _span = span(name);
    f()
}
