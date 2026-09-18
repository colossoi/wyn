//! Host-side unit tests for the playground interface builder.

use super::*;

#[test]
fn shadertoy_example_compiles_with_standard_image_inputs() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            assert!(init_compiler(), "compiler initialization failed");
            let result = compile_to_wgsl_impl(&get_example_program(), true, true);
            assert!(
                result.success,
                "default example failed: {:?}",
                result.error.map(|e| e.message)
            );
            assert!(result.wgsl.is_some(), "default example emitted no WGSL");

            let used_inputs = get_example_program().replace(
                "let phase = iTime in",
                "let phase = iTime + iTimeDelta + iFrameRate + f32.i32(iFrame) +\n\
                 iChannelTime[0] + iChannelResolution[0].x + iMouse.x +\n\
                 iDate.x + iSampleRate in",
            );
            let result = compile_to_wgsl_impl(&used_inputs, true, true);
            assert!(
                result.success,
                "referenced Shadertoy inputs failed: {:?}",
                result.error.map(|e| e.message)
            );
            let interface = result.interface.expect("referenced inputs emitted no interface");
            for name in [
                "iResolution",
                "iTime",
                "iTimeDelta",
                "iFrameRate",
                "iFrame",
                "iMouse",
                "iDate",
                "iSampleRate",
            ] {
                assert!(
                    interface.uniforms.iter().any(|uniform| uniform.name == name),
                    "referenced input {name} was not published as a uniform"
                );
            }
            for name in ["iChannelTime", "iChannelResolution"] {
                assert!(
                    interface
                        .entries
                        .iter()
                        .flat_map(|entry| &entry.inputs)
                        .any(|input| { input.name == name && input.decoration.starts_with("storage(") }),
                    "referenced array input {name} was not published as an entry storage input; got {:?}",
                    interface
                        .entries
                        .iter()
                        .flat_map(|entry| &entry.inputs)
                        .map(|input| (&input.name, &input.decoration))
                        .collect::<Vec<_>>()
                );
            }
        })
        .expect("spawn test thread")
        .join()
        .expect("default Shadertoy example failed to compile");
}

/// Fragment-local collective work keeps the authored graphics interface.
#[test]
fn fragment_local_reduce_keeps_graphics_interface() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(fragment_local_reduce_keeps_graphics_interface_impl)
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

fn fragment_local_reduce_keeps_graphics_interface_impl() {
    let src = r#"
def vertex_main(vertex_index: u32, instance_index: u32, draw_index: u32) vertex<vec2f32> =
  vertex_output(
    if vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
    else if vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
    else @[-1.0, 3.0, 0.0, 1.0],
    @[0.0, 0.0])

def fragment_main(iTime: f32,
                  fragment_value: vec2f32, fragment_position: vec4f32, fragment_front_facing: bool, fragment_primitive_index: u32, fragment_sample_index: u32) vec4f32 =
  let samples = map(|i: i32| f32.cos(iTime + f32.i32(i)), 0..<64) in
  let breath = reduce(|a: f32, b: f32| a + b, 0.0, samples) in
  @[breath, 0.0, 0.0, 1.0]

entry image(iTime: f32,
            screen: render_target<vec4f32>) render_target<vec4f32> =
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, raster, |fragment_value, fragment_position, fragment_front_facing, fragment_primitive_index, fragment_sample_index| fragment_main(iTime, fragment_value, fragment_position, fragment_front_facing, fragment_primitive_index, fragment_sample_index))
"#;
    let result = compile_to_wgsl_impl(src, true, false);
    assert!(result.success, "{:?}", result.error.map(|error| error.message));
    let iface = result.interface.expect("compiled program interface");

    // The WGSL backend emits entry-point names verbatim. In particular,
    // compiler-generated stage names contain underscores and must not be
    // passed through the ordinary identifier mangler a second time.
    assert!(
        iface.entries.iter().all(|entry| entry.wgsl_name == entry.name),
        "interface entry names diverged from emitted WGSL names: {:?}",
        iface.entries.iter().map(|entry| (&entry.name, &entry.wgsl_name)).collect::<Vec<_>>()
    );

    // The captured scalar must remain available to the fragment-local reduce.
    assert!(
        iface.uniforms.iter().any(|u| u.name == "iTime"),
        "uniform iTime missing from interface.uniforms; got {:?}",
        iface.uniforms.iter().map(|u| (u.set, u.binding, u.name.clone())).collect::<Vec<_>>()
    );

    assert!(iface.entries.iter().any(|entry| entry.kind == "vertex"));
    assert!(iface.entries.iter().any(|entry| entry.kind == "fragment"));
    assert!(iface.entries.iter().all(|entry| entry.kind != "compute"));
    assert!(iface.storage.is_empty(), "local reduction needs no host scratch");
    let wgsl = result.wgsl.expect("emitted WGSL");
    assert!(wgsl.contains("loop {"), "local reduction loop missing: {wgsl}");
    assert!(wgsl.contains("cos("), "captured reduction body missing: {wgsl}");
}

#[test]
fn compute_reduce_surfaces_scheduled_storage_bindings() {
    let result = compile_to_wgsl_impl(
        "entry sum(xs: []f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)",
        false,
        false,
    );
    assert!(result.success, "{:?}", result.error.map(|error| error.message));
    let iface = result.interface.expect("compiled interface");
    let wgsl = result.wgsl.expect("emitted WGSL");
    assert!(iface.entries.len() >= 2, "chunk and combine kernels missing");
    assert!(
        iface.storage.len() >= 3,
        "input, partials, and result buffers missing"
    );
    for entry in &iface.entries {
        assert_eq!(entry.kind, "compute");
        assert_eq!(entry.wgsl_name, entry.name);
        assert!(wgsl.contains(&format!("fn {}(", entry.wgsl_name)));
        assert!(entry.inputs.iter().any(|binding| binding.decoration.starts_with("storage(")));
    }
}
