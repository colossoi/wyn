//! Graphics host semantics and serialization, tested without processes or files.
//! Keep CLI coverage in wyn/tests/graphics_aliases_cli.rs limited to artifact smoke tests.
use crate::egglog::ScalarOptimization;
use crate::host::ShaderFormat;
use crate::{
    compile_thru_ssa_with_policy, lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget,
};

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

fn compile(source: &str, target: &str, optimize: bool) -> Result<String, String> {
    let (target, format, shader) = match target {
        "spirv" => (CodegenTarget::Spirv, ShaderFormat::Spirv, "aliases.spv"),
        "wgsl" => (CodegenTarget::Wgsl, ShaderFormat::Wgsl, "aliases.wgsl"),
        _ => panic!("unsupported test target: {target}"),
    };
    // Match the CLI's -O policy; the default SSA helper always uses Full.
    let policy = if optimize { ScalarOptimization::Full } else { ScalarOptimization::Basic };
    let ssa = compile_thru_ssa_with_policy(source, target, policy).map_err(|error| error.to_string())?;
    let program = match target {
        CodegenTarget::Spirv => lower_ssa_to_spirv(ssa).map(|lowered| lowered.program),
        CodegenTarget::Wgsl => lower_ssa_to_wgsl_with_program(ssa).map(|lowered| lowered.program),
        CodegenTarget::Portable => unreachable!("test uses concrete targets"),
    }
    .map_err(|error| error.to_string())?;
    program.to_whl(shader, format).map_err(|error| error.to_string())
}

#[test]
fn render_helper_preserves_tuple_results() {
    let source = include_str!("../../testfiles/regressions/render_helper_tuple.wyn");
    for target in ["spirv", "wgsl"] {
        for optimize in [false, true] {
            let host = compile(source, target, optimize).unwrap();
            assert_eq!(host.matches("(gpu-draw ").count(), 1, "{host}");
            assert_eq!(host.matches("(gpu-alloc ").count(), 1, "{host}");
            assert!(
                host.contains(":source-name \"result_0\" :ownership :owned"),
                "{host}"
            );
            assert!(
                host.contains(":source-name \"screen\" :ownership :borrowed"),
                "{host}"
            );
        }
    }
}

#[test]
fn inline_index_arrays_are_materialized_before_the_indexed_draw() {
    for target in ["spirv", "wgsl"] {
        for optimize in [false, true] {
            for draw in [
                "indexed_draw(INDICES, 2u32)",
                "indexed_draw_from(INDICES, 3u32, 2u32, 0u32, 0i32, 0u32)",
            ] {
                for (parameters, prefix, indices, generated) in [
                    ("", "", "[0u32, 1u32, 2u32]", true),
                    ("", "let indices = [0u32, 1u32, 2u32] in", "indices", true),
                    ("indices: []u32,", "", "indices", false),
                ] {
                    let draw = draw.replace("INDICES", indices);
                    let source = format!(
                        "entry reproduce({parameters} screen: render_target<vec4f32>) render_target<vec4f32> =
                          {prefix}
                          let covered = rasterize_triangles({draw},
                            |vi,ii,_| vertex_output(@[f32.u32(vi)*0.5,f32.u32(ii)*0.5,0.5,1.0],0.0)) in
                          shade_with({{depth_test = #less_equal, depth_write = true,
                                      blend = #replace, color_write = true}},
                            screen, covered, |_,_,_,_,_| @[1.0,0.0,0.0,1.0])"
                    );
                    let host = compile(&source, target, optimize)
                        .unwrap_or_else(|error| panic!("{source}, {target}, -O={optimize}: {error}"));
                    let count = if !generated && draw.starts_with("indexed_draw(") {
                        "count-resource-0"
                    } else {
                        "3"
                    };
                    assert!(host.contains(&format!(" :u32 {count} 2 0 0 0)")), "{host}");
                    if generated {
                        assert_eq!(host.matches("(gpu-alloc ").count(), 1, "{host}");
                        let allocated =
                            host.split(" (gpu-alloc ").next().unwrap().rsplit('(').next().unwrap();
                        assert!(
                            host.contains(&format!(":draw (list :indexed {allocated} :u32")),
                            "{host}"
                        );
                        let dispatch = host.find("(gpu-dispatch ").expect("index producer");
                        let draw = host.find("(gpu-draw ").expect("indexed draw");
                        assert!(dispatch < draw, "{host}");
                        let parameters = host
                            .split(":source-name \"reproduce\"")
                            .nth(1)
                            .unwrap()
                            .split(":results")
                            .next()
                            .unwrap();
                        assert_eq!(parameters.matches(":source-name ").count(), 2, "{host}");
                    } else {
                        assert!(!host.contains("(gpu-alloc "), "{host}");
                        assert!(!host.contains("(gpu-dispatch "), "{host}");
                    }
                }
            }
        }
    }
}

#[test]
fn indexed_indirect_draw_materializes_indices_and_command() {
    let source = include_str!("../../testfiles/regressions/indexed_indirect_draw.wyn");
    let mut failures = Vec::new();
    for target in ["spirv", "wgsl"] {
        for optimize in [false, true] {
            match compile(source, target, optimize) {
                Ok(host) => {
                    assert!(host.contains(":draw (list :indexed-indirect "), "{host}");
                    assert!(host.contains(" 0 1 20))"), "one 20-byte indirect command: {host}");
                    let parameters = host
                        .split(":source-name \"reproduce\"")
                        .nth(1)
                        .expect("source entry")
                        .split(":results")
                        .next()
                        .unwrap();
                    assert!(
                        !parameters.contains(":buffer"),
                        "draw buffers must be internal: {host}"
                    );
                }
                Err(error) => failures.push(format!("{target}, -O={optimize}: {error}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn filtered_indirect_draw_keeps_generated_outputs_internal() {
    let program = "type command = {vertex_count:u32, instance_count:u32, first_vertex:u32, first_instance:u32}
        def cull(values: []vec4f32) ([]vec4f32, command) =
          let live = filter(|v|v.x>0.0, values) in
          (live, {vertex_count=3u32, instance_count=u32(length(live)), first_vertex=0u32, first_instance=0u32})
        entry reproduce(values: []vec4f32, target: render_target<vec4f32>) render_target<vec4f32> =
          let (compacted, draw) = cull(values)
          let fragments = rasterize_triangles(indirect_draw(draw),
            |_,_,_| vertex_output(compacted[0], ())) in
          shade(target, fragments, |_,_,_,_,_| compacted[0])";
    for target in ["spirv", "wgsl"] {
        for optimize in [false, true] {
            let host = compile(program, target, optimize).expect("compile indirect draw");
            let entry = host.split("(define-host-entry").nth(1).expect("source host entry");
            let parameters = entry.split(":parameters '(").nth(1).expect("entry parameters");
            let parameters = parameters.split(":results '(").next().unwrap();
            assert_eq!(parameters.matches(":source-name ").count(), 2, "{host}");
            assert_eq!(host.matches("(gpu-alloc ").count(), 3, "{host}");
        }
    }
}

// The exhaustive spelling/producer matrix stops at stage extraction.
// These representative controls cover interactions with later passes.
#[test]
fn graphics_aliases_preserve_host_programs_across_backends_and_optimization() {
    let producer = "let produced = (map(|v| v + @[1.0, 0.0, 0.0, 0.0], values),
                                    map(|v| v * 2.0, values))
                    let selected = produced.0";
    let cases = [
        (
            "repacked projection",
            source(producer, "selected", "", "INPUT[0]"),
            source(
                producer,
                "selected",
                "let first = (INPUT, 0i32) let (unpacked, _) = first let copy = unpacked
                 let second = { data = (copy, INPUT), tag = 2i32 }
                 let pair = second.data let alias = pair.0",
                "alias[0]",
            ),
        ),
        (
            "projected map input",
            source(
                producer,
                "mapped",
                "let mapped = map(|v| v * 3.0, selected)",
                "INPUT[0]",
            ),
            source(
                producer,
                "mapped",
                "let first = (selected, values) let (a, _) = first
                 let second = { item = (a, a) } let alias = second.item.1
                 let mapped = map(|v| v * 3.0, alias)",
                "INPUT[0]",
            ),
        ),
    ];
    for target in ["spirv", "wgsl"] {
        for optimize in [false, true] {
            for (name, direct, aliased) in &cases {
                let context = format!("{name}, {target}, -O={optimize}");
                let control = compile(direct, target, optimize)
                    .unwrap_or_else(|error| panic!("{context} control: {error}"));
                assert!(
                    control.contains("(gpu-dispatch "),
                    "{context}: real producers must remain"
                );
                let actual =
                    compile(aliased, target, optimize).unwrap_or_else(|error| panic!("{context}: {error}"));
                assert_eq!(actual, control, "{context}: host resources or operations changed");
            }
        }
    }
}

#[test]
fn graphics_callbacks_preserve_scalar_helpers_for_later_optimization() {
    let source = format!(
        "{}\n{}",
        include_str!("../../scripts/playground_image_header.wyn"),
        include_str!("../../testfiles/playground/truncated_octahedra.wyn"),
    );
    let unified = crate::test_pipeline::compile_thru_unified_helpers(&source);
    assert!(unified.defs.iter().any(|def| {
        matches!(def.meta, crate::tlc::DefMeta::Function)
            && unified.symbols.get(def.name).is_some_and(|name| name.contains("clip_slab"))
    }));
    crate::tlc::extract_stages(unified).expect("extract graphics with scalar helpers");
}

#[test]
fn polymorphic_graphics_helpers_keep_ordered_target_versions() {
    let source = r#"
      def forward<T>(value: T) T = value
      def vertex<T>(value: T) vertex<T> =
        vertex_output(@[0.0, 0.0, 0.0, 1.0], value)
      def draw<T>(target: *render_target<vec4f32>, value: T, color: T -> vec4f32)
          *render_target<vec4f32> =
        let fragments = rasterize_triangles(direct_draw(3u32, 1u32),
          |_, _, _| forward(vertex(value))) in
        shade(target, forward(fragments), |v, _, _, _, _| color(v))
      entry reproduce(target: render_target<vec4f32>) render_target<vec4f32> =
        let first = draw(target, 0.5f32, |v| @[v, v, v, 1.0]) in
        draw(first, @[0.0, 1.0, 0.0, 1.0], |v| v)
    "#;
    let unified = crate::test_pipeline::compile_thru_unified_helpers(source);
    let roots: Vec<_> = unified
        .defs
        .iter()
        .filter_map(|def| match &def.meta {
            crate::tlc::DefMeta::EntryPoint(entry) => Some(entry.declaration.entry_kind),
            _ => None,
        })
        .collect();
    assert_eq!(roots, [crate::interface::EntryKind::Root]);
    let stages = crate::tlc::extract_stages(unified).unwrap();
    let operations: Vec<_> = stages
        .defs
        .iter()
        .filter_map(|def| match &def.meta {
            crate::tlc::DefMeta::EntryPoint(entry) => {
                entry.declaration.graphics_group.as_ref().map(|group| group.operation)
            }
            _ => None,
        })
        .collect();
    assert_eq!(operations, [0, 0, 1, 1]);
    for target in ["spirv", "wgsl"] {
        let host = compile(source, target, false).unwrap();
        assert_eq!(host.matches("(gpu-draw ").count(), 2, "{host}");
        assert_eq!(host.matches("(gpu-alloc ").count(), 0, "{host}");
    }
}
