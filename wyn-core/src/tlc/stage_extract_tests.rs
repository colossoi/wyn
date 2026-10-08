//! Source-level alias regressions stop at the stage-extraction boundary.
use super::*;
use crate::test_pipeline::compile_thru_stage_extraction;

// Each spelling must capture the same buffer in both vertex and fragment stages.
const ALIASES: &[(&str, &str, &str)] = &[
    ("direct", "", "INPUT[0]"),
    ("alias", "let alias = INPUT", "alias[0]"),
    ("chain", "let a = INPUT let b = a let alias = b", "alias[0]"),
    ("unused", "let ignored = INPUT", "INPUT[0]"),
    (
        "tuple_pattern",
        "let packed = (INPUT, INPUT) let (alias, _) = packed",
        "alias[0]",
    ),
    (
        "tuple_projection",
        "let packed = (INPUT, INPUT) let alias = packed.1",
        "alias[0]",
    ),
    ("tuple_capture", "let packed = (INPUT, INPUT)", "packed.0[0]"),
    (
        "mixed_tuple",
        "let packed = (values, INPUT) let alias = packed.1",
        "alias[0]",
    ),
    (
        "nested_pattern",
        "let packed = ((INPUT, 0i32), (1i32, INPUT)) let (_, (_, alias)) = packed",
        "alias[0]",
    ),
    (
        "repack",
        "let first = (INPUT, 0i32) let (unpacked, _) = first let copy = unpacked
         let second = { data = (copy, INPUT), tag = 2i32 }
         let pair = second.data let alias = pair.0",
        "alias[0]",
    ),
    (
        "record",
        "let packed = { left = INPUT, right = INPUT } let alias = packed.right",
        "alias[0]",
    ),
    (
        "mixed_record",
        "let packed = { original = values, selected = INPUT } let alias = packed.selected",
        "alias[0]",
    ),
    (
        "callback_unpack",
        "let packed = (INPUT, 0i32)",
        "let (alias, _) = packed in alias[0]",
    ),
    (
        "shadowing",
        "let alias = INPUT let packed = (alias, 0i32) let alias = packed.0",
        "alias[0]",
    ),
    ("helper_alias", "let alias = forward(INPUT)", "alias[0]"),
    (
        "helper_tuple",
        "let packed = pack(INPUT) let alias = packed.0",
        "alias[0]",
    ),
    (
        "nested_let",
        "let alias = (let first = INPUT in first)",
        "alias[0]",
    ),
    (
        "target_sibling",
        "let packed = (INPUT, target) let (alias, _) = packed",
        "alias[0]",
    ),
];

fn source(prefix: &str, input: &str, bindings: &str, read: &str) -> String {
    format!(
        "def forward(xs: []vec4f32) []vec4f32 = xs
         def pack(xs: []vec4f32) ([]vec4f32, i32) = (xs, 0i32)
         entry reproduce(values: []vec4f32, target: render_target<vec4f32>) render_target<vec4f32> =
           {prefix}
           {bindings}
           let fragments = rasterize_triangles(direct_draw(3u32, 1u32),
             |_, _, _| vertex_output(({read}), ())) in
           shade(target, fragments, |_, _, _, _, _| ({read}))"
    )
    .replace("INPUT", input)
}

/// Compare descriptor identity and actual reads, ignoring fresh symbols, spans,
/// and generated names. Equal stage counts alone would miss a wrong capture.
#[derive(Debug, PartialEq)]
struct StageBindings {
    kind: EntryKind,
    inputs: Vec<BindingRef>,
    reads: Vec<BindingRef>,
    outputs: Vec<BindingRef>,
}

fn storage_binding(attribute: &interface::ResolvedAttribute) -> Option<BindingRef> {
    match attribute {
        Attribute::Storage { set, binding, .. } => Some(BindingRef::new(*set, *binding)),
        _ => None,
    }
}

fn stage_bindings(source: &str) -> Vec<StageBindings> {
    let program = compile_thru_stage_extraction(source);
    program
        .defs
        .iter()
        .filter_map(|def| {
            let DefMeta::EntryPoint(entry) = &def.meta else {
                return None;
            };
            let TermKind::Lambda(lambda) = &def.body.kind else {
                panic!("stage lambda")
            };
            let used = captured_symbols(&lambda.body, &LookupSet::new(), &program.symbols);
            let bound = lambda.params.iter().map(|(symbol, _)| *symbol).collect();
            assert!(
                captured_symbols(&lambda.body, &bound, &program.symbols).is_empty(),
                "{} retains a capture that is not a stage parameter",
                entry.declaration.name
            );
            assert_eq!(lambda.params.len(), entry.declaration.params.len());
            let mut inputs = Vec::new();
            let mut reads = Vec::new();
            for ((symbol, _), param) in lambda.params.iter().zip(&entry.declaration.params) {
                for binding in param.attributes.iter().filter_map(storage_binding) {
                    inputs.push(binding);
                    if used.contains(symbol) {
                        reads.push(binding);
                    }
                }
            }
            let outputs = entry
                .declaration
                .outputs
                .iter()
                .filter_map(|output| output.attribute.as_ref().and_then(storage_binding))
                .collect();
            Some(StageBindings {
                kind: entry.declaration.entry_kind,
                inputs,
                reads,
                outputs,
            })
        })
        .collect()
}

/// Establish the control's meaning independently of alias/control equality:
/// each producer reads its predecessor's first output, and both shader stages
/// read that same final buffer. The other projection leaf must not be captured.
fn assert_producer_chain(stages: &[StageBindings], output_counts: &[usize]) {
    let mut kinds = vec![EntryKind::Compute; output_counts.len()];
    kinds.extend([EntryKind::Vertex, EntryKind::Fragment]);
    assert_eq!(stages.iter().map(|stage| stage.kind).collect::<Vec<_>>(), kinds);
    let input = stages[0].inputs[0];
    let mut capture = input;
    for (stage, &count) in stages.iter().zip(output_counts) {
        assert_eq!(stage.outputs.len(), count, "{stage:?}");
        assert_eq!(stage.reads, [capture], "producer must read the selected buffer");
        for output in &stage.outputs {
            assert!(!stage.inputs.contains(output), "producer output must be fresh");
        }
        capture = stage.outputs[0];
    }
    for stage in &stages[output_counts.len()..] {
        assert_eq!(
            stage.reads,
            [capture],
            "both shaders must read the selected buffer"
        );
        assert!(
            stage.outputs.is_empty(),
            "aliases must not create storage outputs"
        );
    }
}

fn check_aliases(prefix: &str, input: &str, output_counts: &[usize]) {
    let control = stage_bindings(&source(prefix, input, "", "INPUT[0]"));
    assert_producer_chain(&control, output_counts);
    // The direct spelling is already compiled as the control above.
    for &(name, bindings, read) in ALIASES.iter().filter(|(name, _, _)| *name != "direct") {
        let actual = stage_bindings(&source(prefix, input, bindings, read));
        assert_eq!(actual, control, "alias form: {name}");
    }
}

#[test]
fn graphics_aliases_preserve_input_capture_identity() {
    check_aliases("", "values", &[]);
}

#[test]
fn graphics_aliases_preserve_computed_capture_identity() {
    check_aliases(
        "let produced = map(|v| v + @[1.0, 0.0, 0.0, 0.0], values)",
        "produced",
        &[1],
    );
}

#[test]
fn graphics_aliases_preserve_computed_projection_identity() {
    check_aliases(
        "let produced = (map(|v| v + @[1.0, 0.0, 0.0, 0.0], values),
                         map(|v| v * 2.0, values))
         let projected = produced.0",
        "projected",
        &[2],
    );
}

#[test]
fn graphics_aliases_preserve_projected_map_input_identity() {
    let producer = "let produced = (map(|v| v + @[1.0, 0.0, 0.0, 0.0], values),
                                    map(|v| v * 2.0, values))
                    let selected = produced.0";
    let direct = source(
        producer,
        "mapped",
        "let mapped = map(|v| v * 3.0, selected)",
        "INPUT[0]",
    );
    let aliased = source(
        producer,
        "mapped",
        "let first = (selected, values) let (a, _) = first
         let second = { item = (a, a) } let alias = second.item.1
         let mapped = map(|v| v * 3.0, alias)",
        "INPUT[0]",
    );
    let control = stage_bindings(&direct);
    assert_producer_chain(&control, &[2, 1]);
    assert_eq!(stage_bindings(&aliased), control, "projected map input");
}
