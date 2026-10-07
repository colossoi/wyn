use super::*;
use crate::Number;

fn backend(source: &str) -> (Program, WgpuBackend) {
    let program = Program::parse(source).unwrap();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let backend = WgpuBackend::with_sources(device, queue, &program, &BTreeMap::new()).unwrap();
    (program, backend)
}

#[test]
fn cpu_parameter_sizes_never_fall_back_to_device_reads() {
    let (program, mut backend) = backend(
        "(define-host-program :version 1)
         (define-host-entry 'sizes :source-name \"sizes\" :function 'host-sizes
           :parameters '((frame :buffer :read :min-bytes 48 :host-bytes 24))
           :results '((pixels :buffer :read)))
         (defun host-sizes (frame)
           (gpu-alloc (* 4 (max 1 (* (wyn-f32-to-i32 (host-read-scalar frame 16 'f32))
                                     (wyn-f32-to-i32 (host-read-scalar frame 20 'f32)))))))",
    );
    // A fallback to GPU readback would fail because COPY_SRC is absent.
    let buffer = backend.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 48,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let input = backend.insert(
        Resource::ParameterBuffer {
            buffer: buffer.clone(),
            bytes: vec![0; 48],
        },
        true,
    );
    for (width, height) in [(17, 9), (65, 33), (0, 0)] {
        let bytes: Vec<_> = [width as f32, height as f32].into_iter().flat_map(f32::to_le_bytes).collect();
        backend.write_buffer(&input, 16, &bytes).unwrap();
        let output = program.run("sizes", std::slice::from_ref(&input), &mut backend).unwrap();
        assert_eq!(backend.buffer_size(&output).unwrap(), (width * height).max(1) * 4);
    }
    let gpu_only = backend.import_buffer(buffer, 48).unwrap();
    let error = program.run("sizes", &[gpu_only], &mut backend).unwrap_err().to_string();
    assert!(error.contains("CPU parameter bytes are unavailable"), "{error}");
    let error = backend
        .call(
            &program,
            "host-read-scalar",
            &[input, Value::Number(Number::U32(48)), Value::Symbol("f32".into())],
        )
        .unwrap_err();
    assert!(error.to_string().contains("outside its CPU byte span"));
}

#[test]
fn parameter_uploads_and_updates_preserve_one_value_and_device_reads_stay_explicit() {
    let (program, mut backend) = backend("(define-host-program :version 1)");
    let input = backend.upload_parameter_buffer(vec![1, 2, 3, 4, 5, 6, 7]).unwrap();
    backend.write_buffer(&input, 1, &[11, 12, 13]).unwrap();
    backend.write_buffer(&input, 6, &[17]).unwrap();
    backend.write_buffer(&input, 0, &[21, 22, 23, 24]).unwrap();
    let expected = [21, 22, 23, 24, 5, 6, 17];
    assert_eq!(backend.parameter_bytes(&input).unwrap(), expected);
    assert_eq!(backend.read_buffer(&input, 0, 7).unwrap(), expected);
    let device = backend.allocate_buffer(4).unwrap();
    backend.write_buffer(&device, 0, &37i32.to_le_bytes()).unwrap();
    let copy = [
        input.clone(),
        Value::Number(Number::U32(0)),
        device.clone(),
        Value::Number(Number::U32(0)),
        Value::Number(Number::U32(4)),
    ];
    let error = backend.call(&program, "gpu-copy", &copy).unwrap_err();
    assert!(error.to_string().contains("must be updated with CPU bytes"));
    assert_eq!(backend.parameter_bytes(&input).unwrap(), expected);
    let args = [device, Value::Number(Number::U32(0)), Value::Symbol("i32".into())];
    assert_eq!(
        backend.call(&program, "gpu-read-scalar", &args).unwrap(),
        Value::Number(Number::I32(37))
    );
    assert!(backend.call(&program, "host-read-scalar", &args).is_err());
}

#[test]
fn shader_writes_cannot_invalidate_cpu_parameters() {
    let (_, mut backend) = backend("(define-host-program :version 1)");
    let program = Program::parse(
        "(define-host-program :version 1)
         (define-gpu-module 'shaders :format :wgsl :path \"unused.wgsl\")
         (define-gpu-kernel 'write :module 'shaders :entry \"write\"
           :workgroup-size '(1 1 1)
           :parameters '((input :buffer :read-write :element :u32 :stride 4))
           :abi '((input :storage 0 0)))",
    )
    .unwrap();
    let input = backend.upload_parameter_buffer(vec![0; 4]).unwrap();
    let declaration = &program.kernels["write"];
    let error = backend.bind_groups(declaration, &[input], &[]).unwrap_err();
    assert!(error.to_string().contains("cannot be written by shaders"));
}
