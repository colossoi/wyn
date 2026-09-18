use crate::egglog::{
    Array, BlockId, BufferData, BufferId, ExprData, ExprId, ExprKind, Function, FunctionKind, Instruction,
    OperationId, Program, Scheduled, Storage, TypeData, Value,
};
use crate::ssa::types::InstKind;
use crate::types::{Type, TypeName};

struct Capture {
    data: Program<Scheduled>,
    root: BlockId,
    buffer: BufferId,
    operation: OperationId,
    result: ExprId,
    fields: [ExprId; 2],
}

impl Capture {
    fn new() -> Self {
        use crate::interface::{StorageBindingDecl, StorageRole};
        let mut data = Program {
            ir: Default::default(),
            state: Scheduled::default(),
        };
        let scalar = data.ir.types.alloc(TypeData {
            ty: crate::types::i32(),
        });
        let pair_ty = crate::types::tuple(vec![crate::types::i32(); 2]);
        let pair = data.ir.types.alloc(TypeData { ty: pair_ty.clone() });
        let buffer = data.state.buffers.alloc(BufferData {
            name: "capture".into(),
            length: Value::Int(1),
            element: pair_ty.clone(),
            storage: Storage::Device,
        });
        data.state.abi.bindings.insert(
            buffer,
            StorageBindingDecl {
                binding: crate::BindingRef::new(0, 0),
                elem_ty: pair_ty,
                role: StorageRole::Input,
                logical_resource: None,
                length: None,
            },
        );
        let operation = OperationId::from(0);
        data.state.materialized.insert(
            operation,
            Value::op("index", [Value::Buffer(buffer), Value::Int(0)]),
        );
        let result = data.ir.expressions.alloc(ExprData {
            ty: pair,
            kind: ExprKind::OperationResult(operation),
        });
        let fields = [0, 1].map(|index| {
            data.ir.expressions.alloc(ExprData {
                ty: scalar,
                kind: ExprKind::Project { tuple: result, index },
            })
        });
        let root = Function {
            name: "capture".into(),
            kind: FunctionKind::Device,
            results: 1,
            blocks: vec![],
        }
        .insert(vec![], &mut data.state.blocks, &mut data.state.bodies);
        Self {
            data,
            root,
            buffer,
            operation,
            result,
            fields,
        }
    }

    fn compiler(&self) -> super::Compiler<'_> {
        super::Compiler {
            data: &self.data,
            origins: Default::default(),
            placements: Default::default(),
            functions: vec![],
            externs: Default::default(),
            specializations: Default::default(),
            active: Default::default(),
            used: Default::default(),
        }
    }
}

#[test]
fn capture_reads_are_shared_through_array_and_argument_emission() {
    let capture = Capture::new();
    let mut compiler = capture.compiler();
    let mut body = super::Body::new(&mut compiler, capture.root, &[], 1).unwrap();
    body.values(&[
        Value::Array(Array::Literal(capture.fields.to_vec())),
        Value::Source(capture.result),
        Value::Source(capture.fields[0]),
    ])
    .unwrap();
    assert_eq!(
        body.builder
            .func()
            .insts
            .values()
            .filter(|node| matches!(node.data, InstKind::Load { .. }))
            .count(),
        1
    );
}

#[test]
fn capture_reads_do_not_cross_writes_or_result_rebindings() {
    let capture = Capture::new();
    let mut compiler = capture.compiler();
    let mut body = super::Body::new(&mut compiler, capture.root, &[], 1).unwrap();
    let sources = capture.fields.map(Value::Source);
    let before = body.values(&sources).unwrap();
    body.instruction(&Instruction::Store {
        buffer: Value::Buffer(capture.buffer),
        index: Value::Int(0),
        value: Value::Tuple(vec![Value::Int(7), Value::Int(9)]),
    })
    .unwrap();
    let after = body.values(&sources).unwrap();
    assert_ne!(before[0].value, after[0].value);
    let memory: Vec<_> = body.builder.func().blocks[body.builder.entry()]
        .insts
        .iter()
        .filter_map(|id| match body.builder.func().insts[*id].data {
            InstKind::Load { .. } => Some("load"),
            InstKind::Store { .. } => Some("store"),
            _ => None,
        })
        .collect();
    assert_eq!(memory, ["load", "store", "load"]);

    body.instruction(&Instruction::BindResult(
        capture.operation,
        Value::Tuple(vec![Value::Int(11), Value::Int(13)]),
    ))
    .unwrap();
    let rebound = body.values(&sources).unwrap();
    assert_ne!(after[0].value, rebound[0].value);
    assert_eq!(
        body.builder
            .func()
            .insts
            .values()
            .filter(|node| matches!(node.data, InstKind::Load { .. }))
            .count(),
        2
    );
}

#[test]
fn capture_reads_stay_in_their_conditional_arm_or_merge() {
    let mut capture = Capture::new();
    let boolean = capture.data.ir.types.alloc(TypeData {
        ty: crate::types::bool_type(),
    });
    let condition = capture.data.ir.expressions.alloc(ExprData {
        ty: boolean,
        kind: ExprKind::Bool(true),
    });
    let conditional = capture.data.ir.expressions.alloc(ExprData {
        ty: capture.data.expressions[capture.fields[0]].ty,
        kind: ExprKind::If {
            condition,
            then_value: capture.fields[0],
            else_value: capture.fields[1],
        },
    });
    let mut compiler = capture.compiler();
    let mut body = super::Body::new(&mut compiler, capture.root, &[], 1).unwrap();
    body.values(&[Value::Source(conditional), Value::Source(capture.fields[0])]).unwrap();
    let function = body.builder.func();
    let load_blocks: std::collections::HashSet<_> = function
        .insts
        .values()
        .filter(|node| matches!(node.data, InstKind::Load { .. }))
        .map(|node| node.placement.block().unwrap())
        .collect();
    assert_eq!(
        load_blocks.len(),
        3,
        "each arm and the merge must load independently"
    );
    assert!(
        !load_blocks.contains(&function.entry),
        "do not speculate a capture read"
    );
    let dominators = wyn_graph::DominatorTree::build(function.entry, |block, successors| {
        successors.extend(function.blocks[block].term.successors());
    });
    for node in function.insts.values() {
        for value in node.data.ssa_uses() {
            if let Some(producer) = function.block_of_value(value) {
                assert!(dominators.dominates(producer, node.placement.block().unwrap()));
            }
        }
    }
}

#[test]
fn split_capture_projections_do_not_materialize_discarded_slots() {
    let mut capture = Capture::new();
    capture.data.state.buffers[capture.buffer].element = crate::types::i32();
    capture.data.state.abi.bindings.get_mut(&capture.buffer).unwrap().elem_ty = crate::types::i32();
    capture.data.state.materialized.insert(
        capture.operation,
        Value::Tuple(vec![
            Value::op("index", [Value::Buffer(capture.buffer), Value::Int(0)]),
            Value::Discarded,
        ]),
    );
    let mut compiler = capture.compiler();
    let mut body = super::Body::new(&mut compiler, capture.root, &[], 1).unwrap();
    let fields =
        body.values(&[Value::Source(capture.fields[0]), Value::Source(capture.fields[0])]).unwrap();
    assert_eq!(fields[0].value, fields[1].value);
    assert_eq!(
        body.builder
            .func()
            .insts
            .values()
            .filter(|node| matches!(node.data, InstKind::Load { .. }))
            .count(),
        1
    );
}

#[test]
fn captured_record_projections_share_one_load_in_both_backends() {
    for source in [
        r#"entry repro(events: []i32, xs: []i32) []i32 =
          let (a, b) = loop (a, b) = (0, 1) for k < length(events) do
            (a + events[k], b * events[k])
          let captured = { a = a, b = b } in
          map(|x| x + captured.a + captured.b, xs)"#,
        r#"entry repro(events: []i32, xs: []i32) []i32 =
          let (a, b) = loop (a, b) = (0, 1) for k < length(events) do
            (a + events[k], b * events[k]) in
          map(|x| x + a + b, xs)"#,
        r#"entry repro(events: []i32, xs: []i32) []i32 =
          let captured = loop (a, b) = (0, 1) for k < length(events) do
            (a + events[k], b * events[k]) in
          map(|x| x + captured.0 + captured.1, xs)"#,
        r#"entry repro(events: []i32, xs: []i32) []i32 =
          let captured = loop c = { a = 0, b = 1 } for k < length(events) do
            { a = c.a + events[k], b = c.b * events[k] } in
          map(|x| x + captured.a + captured.b, xs)"#,
    ] {
        let ssa = crate::compile_thru_ssa(source).unwrap();
        super::tests::assert_ssa_dominance("capture lowering", &ssa);
        let aggregate_loads = ssa.entry_points.iter().flat_map(|entry| {
            entry.body.inner.insts.values().filter(|node| {
                matches!(node.data, InstKind::Load { place }
                    if matches!(entry.body.place_elem_ty(place),
                        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)))
            })
        });
        assert_eq!(aggregate_loads.count(), 1, "{source}");

        let wgsl = crate::lower_ssa_to_wgsl(ssa.clone()).unwrap();
        let module = naga::front::wgsl::parse_str(&wgsl)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&wgsl)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let spirv = crate::lower_ssa_to_spirv(ssa).unwrap().spirv;
        let module = wspirv::dr::load_words(&spirv).unwrap();
        let structs: std::collections::HashSet<_> = module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == wspirv::spirv::Op::TypeStruct)
            .filter_map(|inst| inst.result_id)
            .collect();
        assert_eq!(
            module
                .functions
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.instructions)
                .filter(|inst| inst.class.opcode == wspirv::spirv::Op::Load
                    && inst.result_type.is_some_and(|ty| structs.contains(&ty)))
                .count(),
            1,
            "{source}"
        );
        let bytes: Vec<_> = spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
        let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
