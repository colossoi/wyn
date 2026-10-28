//! Run emitted WHL against a tracing backend; no compiler implementation APIs or GPU execution.
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use wyn_host_interp::{Backend, Error, Form, Number, Options, Program, Result, Value};

#[path = "../../../wyn-host-interp/src/test_backend.rs"]
mod test_backend;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expectation {
    entry: String,
    // Buffer contents as 32-bit words, including scalar/record host parameters.
    #[serde(default)]
    inputs: BTreeMap<String, Vec<i32>>,
    #[serde(default)]
    scalar_arguments: Vec<i64>,
    allocations: Option<Vec<u64>>,
    parameters: Option<usize>,
    #[serde(default)]
    calls: BTreeMap<String, usize>,
    #[serde(default)]
    minimum_calls: BTreeMap<String, usize>,
    host_inputs: Option<bool>,
    result_ownership: Option<Vec<String>>,
    result_names: Option<Vec<String>>,
    captured_results: Option<Vec<bool>>,
    #[serde(default)]
    texture_draw_chain: bool,
    #[serde(default)]
    first_result_owned: bool,
    #[serde(default)]
    no_buffer_inputs: bool,
    #[serde(default)]
    texture_types: BTreeMap<String, String>,
    result_min_bytes: Option<Vec<u64>>,
    result_layouts: Option<Vec<String>>,
    result_bytes: Option<usize>,
    result_at_least_bytes: Option<usize>,
    result_words: Option<Vec<i32>>,
    dispatches: Option<usize>,
    #[serde(default)]
    no_device_reads: bool,
    #[serde(default)]
    draws: Vec<String>,
}

struct Event {
    draw: bool,
    reads: BTreeSet<u64>,
    writes: BTreeSet<u64>,
}

#[derive(Default)]
struct Trace {
    storage: test_backend::Trace,
    names: BTreeMap<u64, String>,
    generated: BTreeSet<u64>,
    written: BTreeSet<u64>,
    allocations: Vec<u64>,
    draws: Vec<Value>,
    events: Vec<Event>,
    origins: BTreeMap<u64, u64>,
}

impl Trace {
    fn origin(&self, id: u64) -> u64 {
        self.origins.get(&id).copied().unwrap_or(id)
    }

    fn normalize(&self, value: &Value) -> Value {
        match value {
            Value::Resource(id) => Value::Symbol(self.names[id].clone()),
            Value::Number(n) => Value::Number(Number::Literal(n.integer().expect("integer draw operand"))),
            Value::List(values) => Value::List(values.iter().map(|v| self.normalize(v)).collect()),
            other => other.clone(),
        }
    }
}

impl Backend for Trace {
    fn call(&mut self, program: &Program, operation: &str, args: &[Value]) -> Result<Value> {
        match operation {
            "gpu-texture-view" => Ok(args[0].clone()),
            "gpu-texture-mip-levels" => Ok(Value::Number(Number::Literal(1))),
            "gpu-buffer-view" => {
                let value = self.storage.call(program, operation, args)?;
                self.origins.insert(value.handle()?, self.origin(args[0].handle()?));
                Ok(value)
            }
            "gpu-alloc" => {
                let value = self.storage.call(program, operation, args)?;
                let id = value.handle()?;
                let bytes = args[0].u64()?;
                let ordinal = self.allocations.iter().filter(|&&size| size == bytes).count();
                self.names.insert(id, format!("allocation-{bytes}-{ordinal}"));
                self.generated.insert(id);
                self.allocations.push(args[0].u64()?);
                Ok(value)
            }
            "gpu-dispatch" => {
                let kernel = program
                    .kernels
                    .get(args[0].text()?)
                    .ok_or_else(|| Error::Invalid("unknown dispatched kernel".into()))?;
                let options = Options::parse(&args[1..], &[":groups", ":args"])?;
                let arguments = options.get(":args")?.list()?;
                if arguments.len() != kernel.parameters.len() {
                    return Err(Error::Invalid("wrong kernel argument count".into()));
                }
                let mut event = Event {
                    draw: false,
                    reads: BTreeSet::new(),
                    writes: BTreeSet::new(),
                };
                for (parameter, value) in kernel.parameters.iter().zip(arguments) {
                    if matches!(parameter.access.as_deref(), Some(":read" | ":read-write")) {
                        event.reads.insert(self.origin(value.handle()?));
                    }
                    if matches!(parameter.access.as_deref(), Some(":write" | ":read-write")) {
                        event.writes.insert(self.origin(value.handle()?));
                    }

                    if matches!(parameter.access.as_deref(), Some(":write" | ":read-write")) {
                        self.written.insert(value.handle()?);
                    }
                }
                self.events.push(event);
                self.storage.call(program, operation, args)
            }
            "gpu-draw" => {
                if !program.graphics.contains_key(args[0].text()?) {
                    return Err(Error::Invalid("unknown graphics pipeline".into()));
                }
                let options = Options::parse(
                    &args[1..],
                    &[
                        ":args",
                        ":vertices",
                        ":colors",
                        ":depth",
                        ":viewport",
                        ":scissor",
                        ":draw",
                    ],
                )?;
                let mut event = Event {
                    draw: true,
                    reads: BTreeSet::new(),
                    writes: BTreeSet::new(),
                };
                for value in options.get(":args")?.list()? {
                    if let Value::Resource(id) = value {
                        event.reads.insert(self.origin(*id));
                    }
                }
                for color in options.get(":colors")?.list()? {
                    event.writes.insert(self.origin(color.list()?[1].handle()?));
                }
                self.events.push(event);
                let draw = options.get(":draw")?;
                for value in draw.list()? {
                    if let Value::Resource(id) = value {
                        if !self.storage.buffers.contains_key(id) {
                            return Err(Error::Invalid("draw uses a missing/freed buffer".into()));
                        }
                        if self.generated.contains(id) && !self.written.contains(id) {
                            return Err(Error::Invalid("draw precedes its buffer writer".into()));
                        }
                    }
                }
                self.draws.push(self.normalize(draw));
                Ok(Value::Nil)
            }
            _ => self.storage.call(program, operation, args),
        }
    }
}

fn count_calls(form: &Form, counts: &mut BTreeMap<String, usize>) {
    if let Form::List(values) = form {
        if let Some(Form::Symbol(name)) = values.first() {
            if name == "quote" {
                return;
            }
            *counts.entry(name.clone()).or_default() += 1;
        }
        for value in values {
            count_calls(value, counts);
        }
    }
}

pub fn check(source: &str, expectations: &[Expectation]) {
    let program = Program::parse(source).expect("valid WHL program");
    for expected in expectations {
        let entry = program.entry(&expected.entry).expect("expected host entry");
        let forms = wyn_host_interp::read(source).unwrap();
        let function = forms
            .iter()
            .find(|form| {
                matches!(form, Form::List(values)
            if matches!(values.as_slice(), [Form::Symbol(kind), Form::Symbol(name), ..]
                if kind == "defun" && name == &entry.function))
            })
            .expect("entry function body");
        let mut calls = BTreeMap::new();
        count_calls(function, &mut calls);
        for (name, count) in &expected.calls {
            assert_eq!(
                calls.get(name).copied().unwrap_or(0),
                *count,
                "{} {name} calls",
                expected.entry
            );
        }
        for (name, count) in &expected.minimum_calls {
            assert!(
                calls.get(name).copied().unwrap_or(0) >= *count,
                "{} {name} calls",
                expected.entry
            );
        }
        if let Some(present) = expected.host_inputs {
            let spans: Vec<_> = entry.parameters.iter().filter_map(|p| p.host_bytes().unwrap()).collect();
            assert_eq!(!spans.is_empty(), present, "{} CPU dependencies", expected.entry);
            assert!(spans.iter().all(|&bytes| bytes >= 4 && bytes % 4 == 0));
        }
        if let Some(ownership) = &expected.result_ownership {
            let actual: Vec<_> = entry
                .results
                .iter()
                .map(|p| p.options.get(":ownership").unwrap().text().unwrap())
                .collect();
            assert_eq!(&actual, ownership, "{} result ownership", expected.entry);
        }
        if let Some(names) = &expected.result_names {
            assert_eq!(
                entry.results.iter().map(|p| p.source_name()).collect::<Vec<_>>(),
                *names
            );
        }
        if expected.first_result_owned {
            assert_eq!(
                entry.results[0].options.get(":ownership").unwrap().text().unwrap(),
                ":owned"
            );
            assert!(entry.results[0].options.optional(":alias").is_none());
        }
        if expected.no_buffer_inputs {
            assert!(entry.parameters.iter().all(|p| p.kind != ":buffer"));
        }
        if !expected.texture_types.is_empty() {
            let actual: BTreeMap<_, _> = entry
                .parameters
                .iter()
                .filter_map(|p| {
                    p.options
                        .optional(":sample-type")
                        .map(|ty| (p.source_name().to_owned(), ty.text().unwrap().to_owned()))
                })
                .collect();
            assert_eq!(
                actual, expected.texture_types,
                "{} texture sampling",
                expected.entry
            );
        }

        if let Some(count) = expected.parameters {
            assert_eq!(
                entry.parameters.len(),
                count,
                "{} parameter count",
                expected.entry
            );
        }
        if let Some(lengths) = &expected.result_min_bytes {
            let actual: Vec<_> = entry
                .results
                .iter()
                .map(|p| p.minimum_bytes().unwrap().expect("fixed result bytes"))
                .collect();
            assert_eq!(&actual, lengths, "{} published result capacity", expected.entry);
        }
        if let Some(layouts) = &expected.result_layouts {
            assert_eq!(entry.results.len(), layouts.len());
            for (result, layout) in entry.results.iter().zip(layouts) {
                let forms = wyn_host_interp::read(layout).expect("result layout syntax");
                assert_eq!(forms.len(), 1);
                assert_eq!(
                    result.options.get(":value-layout").unwrap(),
                    &Value::quoted(&forms[0])
                );
            }
        }
        for input in expected.inputs.keys() {
            assert!(
                entry.parameters.iter().any(|p| p.source_name() == input
                    || p.fields().unwrap().iter().any(|(name, _, _)| name == input)),
                "unknown input {input}"
            );
        }
        // Capacity-only expectations inspect declarations without simulating GPU-dependent control flow.
        if expected.captured_results.is_none()
            && !expected.texture_draw_chain
            && expected.allocations.is_none()
            && expected.draws.is_empty()
            && expected.result_bytes.is_none()
            && expected.result_at_least_bytes.is_none()
            && expected.result_words.is_none()
            && expected.dispatches.is_none()
            && !expected.no_device_reads
        {
            assert!(expected.inputs.is_empty() && expected.scalar_arguments.is_empty());
            continue;
        }
        let mut trace = Trace::default();
        let mut scalars = expected.scalar_arguments.iter();
        let arguments: Vec<_> = entry
            .parameters
            .iter()
            .map(|p| {
                if let Some(kind) = wyn_host_interp::NumberType::parse(p.kind.trim_start_matches(':')) {
                    let value = *scalars.next().expect("scalar host argument");
                    return Value::Number(Number::Literal(value.into()).convert(kind).unwrap());
                }
                let name = p.source_name();
                let bytes = if let Some(words) = expected.inputs.get(name) {
                    words.iter().flat_map(|n| n.to_le_bytes()).collect()
                } else {
                    let mut bytes = vec![
                        0;
                        p.host_bytes().unwrap().or(p.minimum_bytes().unwrap()).unwrap_or(0)
                            as usize
                    ];
                    for (field, offset, size) in p.fields().unwrap() {
                        if let Some(words) = expected.inputs.get(&field) {
                            let data: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                            assert_eq!(data.len(), size as usize, "{field} host field size");
                            bytes[offset as usize..(offset + size) as usize].copy_from_slice(&data);
                        }
                    }
                    bytes
                };
                let value = trace.storage.input(bytes);
                trace.names.insert(
                    value.handle().unwrap(),
                    format!("input-{}", name.replace('_', "-")),
                );
                value
            })
            .collect();
        assert!(scalars.next().is_none(), "unused scalar host arguments");
        let result = program
            .run(&expected.entry, &arguments, &mut trace)
            .unwrap_or_else(|e| panic!("{}: {e}", expected.entry));
        if let Some(captured) = &expected.captured_results {
            let results = result.list().expect("tuple results");
            assert_eq!(results.len(), captured.len());
            for (value, &wanted) in results.iter().zip(captured) {
                let id = trace.origin(value.handle().unwrap());
                assert_eq!(
                    trace.events.iter().any(|e| e.draw && e.reads.contains(&id)),
                    wanted,
                    "{} captured result identity",
                    expected.entry
                );
            }
        }
        if expected.texture_draw_chain {
            let results = result.list().expect("sample, scene and surface results");
            let sampled = trace.origin(results[0].handle().unwrap());
            let scene = trace.origin(results[1].handle().unwrap());
            let writers: Vec<_> = trace
                .events
                .iter()
                .enumerate()
                .filter(|(_, e)| e.draw && e.writes.contains(&scene))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(writers.len(), 2, "ground and props write scene");
            assert!(
                trace.events[writers[0]].reads.is_disjoint(&trace.generated),
                "ground draw uses original inputs"
            );
            assert!(
                !trace.events[writers[1]].reads.is_disjoint(&trace.generated),
                "props draw consumes the generated vertex buffer"
            );
            let sample = trace
                .events
                .iter()
                .position(|e| !e.draw && e.reads.contains(&scene) && e.writes.contains(&sampled))
                .expect("scene sampling");
            let resolve = trace
                .events
                .iter()
                .position(|e| e.draw && e.reads.contains(&sampled))
                .expect("resolve consumes sampled buffer");
            assert!(writers[0] < writers[1] && writers[1] < sample && sample < resolve);
        }
        if expected.result_bytes.is_some()
            || expected.result_at_least_bytes.is_some()
            || expected.result_words.is_some()
        {
            let bytes = &trace.storage.buffers[&result.handle().expect("single buffer result")];
            if let Some(count) = expected.result_bytes {
                assert_eq!(bytes.len(), count, "{} result size", expected.entry);
            }
            if let Some(count) = expected.result_at_least_bytes {
                assert!(bytes.len() >= count, "{} result capacity", expected.entry);
            }
            if let Some(words) = &expected.result_words {
                let wanted: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                assert_eq!(bytes, &wanted, "{} result contents", expected.entry);
            }
        }
        if let Some(count) = expected.dispatches {
            assert_eq!(
                trace.storage.dispatches.len(),
                count,
                "{} dispatch count",
                expected.entry
            );
        }
        if expected.no_device_reads {
            assert!(
                trace.storage.device_reads.is_empty(),
                "{} unexpected device readback",
                expected.entry
            );
        }
        if let Some(allocations) = &expected.allocations {
            let mut actual = trace.allocations.clone();
            let mut expected_sizes = allocations.clone();
            actual.sort_unstable();
            expected_sizes.sort_unstable();
            assert_eq!(actual, expected_sizes, "{} allocations", expected.entry);
        }
        let draws: Vec<_> = expected
            .draws
            .iter()
            .map(|text| {
                let forms = wyn_host_interp::read(text).expect("draw expectation syntax");
                assert_eq!(forms.len(), 1);
                assert!(matches!(forms[0], Form::List(_)));
                Value::quoted(&forms[0])
            })
            .collect();
        if expected.captured_results.is_none() && !expected.texture_draw_chain {
            assert_eq!(
                trace.draws, draws,
                "{} draw arguments/resource identity",
                expected.entry
            );
        }
    }
}

#[test]
fn trace_checks_reject_missing_writers_wrong_resources_and_wrong_sizes() {
    let source = include_str!("fixtures/draw_order.wynhost");
    let expected: Expectation = toml::from_str(
        r#"
entry = "reproduce"
allocations = [12, 20]
draws = ["(:indexed-indirect allocation-12-0 :u32 allocation-20-0 0 1 20)"]
"#,
    )
    .unwrap();
    check(source, std::slice::from_ref(&expected));
    // Redirect the first producer: the index buffer is no longer written before its draw.
    let missing_writer = source.replace(":args (list resource-0))", ":args (list resource-1))");
    assert_ne!(missing_writer, source);
    assert!(std::panic::catch_unwind(|| check(&missing_writer, std::slice::from_ref(&expected))).is_err());
    let wrong_resource = source.replace(
        ":indexed-indirect resource-0 :u32",
        ":indexed-indirect resource-1 :u32",
    );
    assert_ne!(wrong_resource, source);
    assert!(std::panic::catch_unwind(|| check(&wrong_resource, std::slice::from_ref(&expected))).is_err());
    let wrong_size = source.replace("(gpu-alloc (i64 12))", "(gpu-alloc (i64 16))");
    assert_ne!(wrong_size, source);
    assert!(std::panic::catch_unwind(|| check(&wrong_size, std::slice::from_ref(&expected))).is_err());
}

pub fn check_rust(source: &str, rust: &str, expectations: &[Expectation]) {
    let program = Program::parse(source).unwrap();
    let forms = wyn_host_interp::read(source).unwrap();
    fn operations(form: &Form, output: &mut Vec<String>) {
        let Form::List(values) = form else {
            return;
        };
        if let [Form::Symbol(kind), Form::List(quoted), ..] = values.as_slice() {
            if let [Form::Symbol(quote), Form::Symbol(name)] = quoted.as_slice() {
                if quote == "quote" {
                    if kind == "gpu-draw" {
                        output.push(format!(
                            "context.render_{}(",
                            name.strip_prefix("graphics-").unwrap()
                        ));
                    } else if kind == "gpu-dispatch" {
                        output.push(format!(
                            "&context.compute_{}",
                            name.strip_prefix("kernel-").unwrap().replace('-', "_")
                        ));
                    }
                }
            }
        }
        if matches!(values.first(), Some(Form::Symbol(name)) if name == "quote") {
            return;
        }
        for value in values {
            operations(value, output);
        }
    }
    for expected in expectations {
        if expected.captured_results.is_none() && !expected.texture_draw_chain {
            continue;
        }
        let entry = program.entry(&expected.entry).unwrap();
        let marker = format!("pub fn host_{}(", expected.entry);
        assert_eq!(
            rust.matches(&marker).count(),
            1,
            "one host function for source entry"
        );
        let call = rust.split_once(&marker).unwrap().1.split("pub fn host_").next().unwrap();
        let compact: String = call.split_whitespace().collect();
        let function = forms
            .iter()
            .find(|form| {
                matches!(form, Form::List(values)
            if matches!(values.as_slice(), [Form::Symbol(kind), Form::Symbol(name), ..]
                if kind == "defun" && name == &entry.function))
            })
            .unwrap();
        let mut expected_calls = Vec::new();
        operations(function, &mut expected_calls);
        let mut remaining = compact.as_str();
        for operation in expected_calls {
            let position = remaining
                .find(&operation)
                .unwrap_or_else(|| panic!("missing ordered Rust operation {operation}"));
            remaining = &remaining[position + operation.len()..];
        }
    }
}
