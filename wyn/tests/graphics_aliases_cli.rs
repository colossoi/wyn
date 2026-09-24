use std::fs;
use std::path::Path;
use std::process::Command;

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

fn compile(directory: &Path, source: &str, target: &str, optimize: bool) -> Result<String, String> {
    let input = directory.join("aliases.wyn");
    let output = directory.join(if target == "spirv" { "aliases.spv" } else { "aliases.wgsl" });
    fs::write(&input, source).expect("write source");
    let mut command = Command::new(env!("CARGO_BIN_EXE_wyn"));
    command.args(["build", "--graphics", "--target", target, "--max-warnings", "0"]);
    command.arg(&input).arg("--output").arg(&output);
    if optimize {
        command.arg("-O");
    }
    let result = command.output().expect("run compiler");
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).into_owned());
    }
    fs::read_to_string(output.with_extension("wynhost")).map_err(|error| error.to_string())
}

#[test]
fn graphics_array_aliases_preserve_input_and_computed_buffer_identity() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("wyn_graphics_aliases_{}_{timestamp}", std::process::id()));
    fs::create_dir(&directory).expect("create test directory");
    let mut failures = Vec::new();
    for target in ["spirv", "wgsl"] {
        for optimize in [false, true] {
            // Keep genuine computed producers/projections while removing only
            // administrative aliases and packing. Their host program must also
            // remain identical to its direct-capture control.
            for (family, prefix, input) in [
                ("input", "", "values"),
                (
                    "computed",
                    "let produced = map(|v| v + @[1.0, 0.0, 0.0, 0.0], values)",
                    "produced",
                ),
                (
                    "computed_projection",
                    "let produced = (map(|v| v + @[1.0, 0.0, 0.0, 0.0], values),
                                     map(|v| v * 2.0, values))
                     let projected = produced.0",
                    "projected",
                ),
            ] {
                let control = compile(
                    &directory,
                    &source(prefix, input, "", "INPUT[0]"),
                    target,
                    optimize,
                )
                .unwrap_or_else(|error| panic!("{family} control, {target}, -O={optimize}: {error}"));
                if family == "input" {
                    assert!(!control.contains("(gpu-dispatch "));
                    assert!(!control.contains("(gpu-alloc "));
                } else {
                    assert!(control.contains("(gpu-dispatch "), "real producers must remain");
                }
                for &(name, bindings, read) in ALIASES {
                    let program = source(prefix, input, bindings, read);
                    match compile(&directory, &program, target, optimize) {
                        Ok(host) if host == control => {}
                        Ok(_) => failures.push(format!(
                            "{family}/{name}, {target}, -O={optimize}: host resources or operations changed"
                        )),
                        Err(error) => {
                            failures.push(format!("{family}/{name}, {target}, -O={optimize}: {error}"))
                        }
                    }
                }
            }

            // A projected computed array must remain valid as a SOAC input
            // after being unpacked, aliased, and packed again.
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
            let control = compile(&directory, &direct, target, optimize).expect("map control");
            match compile(&directory, &aliased, target, optimize) {
                Ok(host) if host == control => {}
                result => failures.push(format!("map input, {target}, -O={optimize}: {result:?}")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    fs::remove_dir_all(directory).expect("remove test directory");
}
