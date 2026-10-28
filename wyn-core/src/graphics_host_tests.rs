//! Focused graphics tests; broader host contracts use CLI/WHL fixtures.
use crate::egglog::ScalarOptimization;
use crate::host::ShaderFormat;
use crate::{
    compile_thru_ssa_with_policy, lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget,
};

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
