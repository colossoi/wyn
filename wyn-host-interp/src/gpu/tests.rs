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
fn buffer_views_keep_storage_and_logical_size_separate() {
    let (program, mut backend) = backend("(define-host-program :version 1)");
    let storage = backend.allocate_buffer(16).unwrap();
    let view = backend
        .call(
            &program,
            "gpu-buffer-view",
            &[storage.clone(), Value::Number(Number::U64(4))],
        )
        .unwrap();
    assert_eq!(backend.buffer_size(&view).unwrap(), 4);
    assert_eq!(backend.buffer_size(&storage).unwrap(), 16);
    backend.write_buffer(&view, 0, &7i32.to_le_bytes()).unwrap();
    assert_eq!(backend.read_buffer(&storage, 0, 4).unwrap(), 7i32.to_le_bytes());
    let empty = backend
        .call(
            &program,
            "gpu-buffer-view",
            &[storage.clone(), Value::Number(Number::U64(0))],
        )
        .unwrap();
    assert!(backend.read_buffer(&empty, 0, 0).unwrap().is_empty());
    assert!(backend
        .call(
            &program,
            "gpu-buffer-view",
            &[storage, Value::Number(Number::U64(17))]
        )
        .is_err());
}

#[test]
fn buffer_views_reject_cpu_parameter_storage() {
    let (program, mut backend) = backend("(define-host-program :version 1)");
    let parameter = backend.upload_parameter_buffer(1i32.to_le_bytes().to_vec()).unwrap();
    let error = backend
        .call(
            &program,
            "gpu-buffer-view",
            &[parameter.clone(), Value::Number(Number::U32(4))],
        )
        .unwrap_err();
    assert!(error.to_string().contains("CPU parameter"), "{error}");
    assert_eq!(backend.parameter_bytes(&parameter).unwrap(), 1i32.to_le_bytes());
    assert_eq!(backend.read_buffer(&parameter, 0, 4).unwrap(), 1i32.to_le_bytes());
}

#[test]
fn copies_between_buffer_views_use_backing_identity() {
    let (program, mut backend) = backend("(define-host-program :version 1)");
    let storage = backend.allocate_buffer(16).unwrap();
    backend.write_buffer(&storage, 0, &7i32.to_le_bytes()).unwrap();
    let view = backend
        .call(
            &program,
            "gpu-buffer-view",
            &[storage.clone(), Value::Number(Number::U32(16))],
        )
        .unwrap();
    let nested = backend
        .call(
            &program,
            "gpu-buffer-view",
            &[view.clone(), Value::Number(Number::U32(12))],
        )
        .unwrap();
    for destination in [&storage, &view, &nested] {
        backend
            .call(
                &program,
                "gpu-copy",
                &[
                    destination.clone(),
                    Value::Number(Number::U32(8)),
                    storage.clone(),
                    Value::Number(Number::U32(0)),
                    Value::Number(Number::U32(4)),
                ],
            )
            .unwrap();
        assert_eq!(
            backend.read_buffer(destination, 8, 4).unwrap(),
            7i32.to_le_bytes()
        );
        let error = backend
            .call(
                &program,
                "gpu-copy",
                &[
                    destination.clone(),
                    Value::Number(Number::U32(0)),
                    storage.clone(),
                    Value::Number(Number::U32(0)),
                    Value::Number(Number::U32(4)),
                ],
            )
            .unwrap_err();
        assert!(error.to_string().contains("overlapping buffer copy"), "{error}");
    }
    assert!(backend
        .call(
            &program,
            "gpu-copy",
            &[
                nested,
                Value::Number(Number::U32(12)),
                storage,
                Value::Number(Number::U32(0)),
                Value::Number(Number::U32(4)),
            ]
        )
        .unwrap_err()
        .to_string()
        .contains("logical capacity"));
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

#[test]
fn storage_array_length_excludes_vertex_buffer_padding() {
    let source = r#"
        (define-host-program :version 1)
        (define-gpu-module 'shaders :format :wgsl :path "length.wgsl")
        (define-gpu-kernel 'length :module 'shaders :entry "length"
          :workgroup-size '(1 1 1)
          :parameters '((input :buffer :read) (output :buffer :write :min-bytes 4))
          :abi '((input :storage 0 0) (output :storage 0 1)))
        (define-host-entry 'length :source-name "length" :function 'host-length
          :parameters '((input :buffer :read))
          :results '((output :buffer :write)))
        (defun host-length (input)
          (let ((output (gpu-alloc (i64 4))))
            (gpu-dispatch 'length :groups (list (i64 1) (i64 1) (i64 1))
              :args (list input output))
            output))
    "#;
    let program = Program::parse(source).unwrap();
    let (_, backend) = backend("(define-host-program :version 1)");
    let shader = br#"
        @group(0) @binding(0) var<storage, read> input: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32>;
        @compute @workgroup_size(1) fn length() { output[0] = arrayLength(&input); }
    "#;
    let sources = BTreeMap::from([("shaders".into(), shader.to_vec())]);
    let mut backend = WgpuBackend::with_sources(backend.device, backend.queue, &program, &sources).unwrap();
    for length in [1, 2, 3, 257] {
        let input = backend.allocate_buffer(length * 4).unwrap();
        assert!(backend.buffer(&input).unwrap().usage().contains(BufferUsages::VERTEX));
        let output = program.run("length", &[input], &mut backend).unwrap();
        let bytes = backend.read_buffer(&output, 0, 4).unwrap();
        assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), length as u32);
    }
}
