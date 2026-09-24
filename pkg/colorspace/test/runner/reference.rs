use std::{path::Path, process::Command};
use wyn_host_interp::{gpu::WgpuBackend, Program};

// Independent f64 reference, compared with runtime f32 shader results.
fn decode(x: f64) -> f64 {
    if x.abs() <= 0.04045 {
        x / 12.92
    } else {
        x.signum() * ((x.abs() + 0.055) / 1.055).powf(2.4)
    }
}
fn encode(x: f64) -> f64 {
    if x.abs() <= 0.0031308 {
        x * 12.92
    } else {
        x.signum() * (1.055 * x.abs().powf(1.0 / 2.4) - 0.055)
    }
}
fn close(actual: f32, expected: f64, label: &str) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() <= 2e-5 * expected.abs().max(1.0),
        "{label}: got {actual}, expected {expected}"
    );
}
fn run(
    program: &Program,
    gpu: &mut WgpuBackend,
    entry: &str,
    input: &[f32],
    parameter: Option<f32>,
) -> Vec<f32> {
    let mut input = input.to_vec();
    if let Some(parameter) = parameter {
        for sample in input.chunks_exact_mut(4) {
            sample[3] = parameter;
        }
    }
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = gpu.allocate_buffer(bytes.len() as u64).unwrap();
    gpu.write_buffer(&buffer, 0, &bytes).unwrap();
    let result = program.run(entry, &[buffer], gpu).unwrap();
    let size = gpu.buffer_size(&result).unwrap();
    let bytes = gpu.read_buffer(&result, 0, size).unwrap();
    gpu.retain_resources(&[]);
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}
fn channels(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    // vec3 storage elements have a 16-byte stride; padding is not colour data.
    for (a, e) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        for i in 0..3 {
            close(a[i], f64::from(e[i]), label);
        }
    }
}

#[test]
fn reference_vectors_and_hdr_roundtrips_on_both_shader_targets() {
    let package_test = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = std::env::temp_dir().join(format!("wyn-colorspace-{}", std::process::id()));
    std::fs::create_dir_all(&output).unwrap();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    for (target, ext) in [("wgsl", "wgsl"), ("spirv", "spv")] {
        let result = Command::new(std::env::var_os("WYN").unwrap_or_else(|| "wyn".into()))
            .args(["build", "-O", "--target", target])
            .arg(package_test.join("colorspace.wyn"))
            .arg("-o")
            .arg(output.join(format!("colorspace.{ext}")))
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let program =
            Program::parse(&std::fs::read_to_string(output.join("colorspace.wynhost")).unwrap()).unwrap();
        let mut gpu = WgpuBackend::new(device.clone(), queue.clone(), &program, &output).unwrap();

        let mut samples: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
        samples.extend([
            -4., -1., -0.040451, -0.04045, -0.040449, -0.003131, -0.0031308, -0.0031306, -0.00001,
            0.0031306, 0.0031308, 0.003131, 0.040449, 0.04045, 0.040451, 2., 4., 16.,
        ]);
        let result = run(&program, &mut gpu, "transfer", &samples, None);
        assert_eq!(result.len(), samples.len() * 4);
        for (&x, y) in samples.iter().zip(result.chunks_exact(4)) {
            let x = f64::from(x);
            close(y[0], decode(x), "sRGB decode");
            close(y[1], encode(x), "sRGB encode");
            close(y[2], x.signum() * x.abs().powf(f64::from(2.2f32)), "gamma decode");
            close(
                y[3],
                x.signum() * x.abs().powf(1.0 / f64::from(2.2f32)),
                "gamma encode",
            );
        }
        // Known anchors catch reversed transfer directions, beyond round trips.
        close(
            run(&program, &mut gpu, "transfer", &[0.5], None)[0],
            0.21404114048223255,
            "sRGB 0.5",
        );
        let rgb: Vec<f32> = [
            [0., 0., 0., 0.],
            [1., 1., 1., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [-0.25, 2., 4., 0.],
            [0.18, 0.5, 0.9, 0.],
            [0.0031308, 0.04045, 16., 0.],
        ]
        .into_iter()
        .flatten()
        .collect();
        let xyz = run(&program, &mut gpu, "xyz", &rgb, None);
        let matrix = [
            [506752. / 1228815., 87881. / 245763., 12673. / 70218.],
            [87098. / 409605., 175762. / 245763., 12673. / 175545.],
            [7918. / 409605., 87881. / 737289., 1001167. / 1053270.],
        ];
        for (rgb, xyz) in rgb.chunks_exact(4).zip(xyz.chunks_exact(4)) {
            for i in 0..3 {
                close(
                    xyz[i],
                    (0..3).map(|j| matrix[i][j] * f64::from(rgb[j])).sum(),
                    "XYZ reference",
                );
            }
        }
        close(xyz[4], 0.3127 / 0.3290, "D65 X");
        close(xyz[5], 1., "D65 Y");
        close(xyz[6], (1. - 0.3127 - 0.3290) / 0.3290, "D65 Z");
        let recovered = run(&program, &mut gpu, "rgb", &xyz, None);
        channels(&recovered, &rgb, "XYZ roundtrip including negative/HDR");
        let luma = run(&program, &mut gpu, "luminance", &rgb, None);
        assert_eq!(luma.len(), rgb.len() / 4);
        for (&y, xyz) in luma.iter().zip(xyz.chunks_exact(4)) {
            close(y, f64::from(xyz[1]), "luminance = Y");
        }

        for (enc, dec, parameter) in [
            ("encode", "decode", None),
            ("gamma_encode", "gamma_decode", Some(2.2)),
        ] {
            let encoded = run(&program, &mut gpu, enc, &rgb, parameter);
            for (input, actual) in rgb.chunks_exact(4).zip(encoded.chunks_exact(4)) {
                for i in 0..3 {
                    let x = f64::from(input[i]);
                    let expected = match parameter {
                        None => encode(x),
                        Some(g) => x.signum() * x.abs().powf(1.0 / f64::from(g)),
                    };
                    close(actual[i], expected, "vector encode");
                }
            }
            let decoded = run(&program, &mut gpu, dec, &encoded, parameter);
            channels(&decoded, &rgb, "transfer roundtrip including negative/HDR");
        }
        for stops in [-2., 0., 1., 2.5] {
            let exposed = run(&program, &mut gpu, "expose", &rgb, Some(stops));
            for (input, actual) in rgb.chunks_exact(4).zip(exposed.chunks_exact(4)) {
                for i in 0..3 {
                    close(
                        actual[i],
                        f64::from(input[i]) * f64::from(stops).exp2(),
                        "exposure stops",
                    );
                }
            }
        }
        for white in [80., 203., 1000.] {
            let nits = run(&program, &mut gpu, "to_nits", &rgb, Some(white));
            for (input, actual) in rgb.chunks_exact(4).zip(nits.chunks_exact(4)) {
                for i in 0..3 {
                    close(
                        actual[i],
                        f64::from(input[i]) * f64::from(white),
                        "reference white scaling",
                    );
                }
            }
            let relative = run(&program, &mut gpu, "from_nits", &nits, Some(white));
            channels(&relative, &rgb, "nit roundtrip");
        }
        println!("{target}: transfer thresholds, 8-bit ramp, D65/primaries, negative/HDR values, exposure and nit scaling passed");
    }
}
