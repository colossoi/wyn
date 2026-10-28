use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use wspirv::{dr::Operand, spirv::Op};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrayLoads {
    entry: String,
    minimum: usize,
}

pub fn check(bytes: &[u8], expectations: &[ArrayLoads]) {
    let module = wspirv::dr::load_bytes(bytes).expect("SPIR-V module");
    let arrays: BTreeSet<_> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypeArray)
        .filter_map(|i| i.result_id)
        .collect();
    let functions: BTreeMap<_, _> =
        module.functions.iter().map(|f| (f.def.as_ref().unwrap().result_id.unwrap(), f)).collect();
    for expected in expectations {
        assert!(expected.minimum > 0, "array-load check must be affirmative");
        let mut pending = Vec::new();
        for entry in &module.entry_points {
            if let [_, Operand::IdRef(function), Operand::LiteralString(name), ..] =
                entry.operands.as_slice()
            {
                if name == &expected.entry
                    || name.starts_with(&format!("{}__", expected.entry))
                    || name.starts_with(&format!("{}_", expected.entry))
                {
                    pending.push(*function);
                }
            }
        }
        assert!(!pending.is_empty(), "missing SPIR-V entry {}", expected.entry);
        let mut visited = BTreeSet::new();
        let mut loads = 0;
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            for instruction in functions[&id].blocks.iter().flat_map(|b| &b.instructions) {
                if instruction.class.opcode == Op::Load
                    && instruction.result_type.is_some_and(|ty| arrays.contains(&ty))
                {
                    loads += 1;
                }
                if instruction.class.opcode == Op::FunctionCall {
                    if let Some(Operand::IdRef(callee)) = instruction.operands.first() {
                        pending.push(*callee);
                    }
                }
            }
        }
        assert!(
            loads >= expected.minimum,
            "{} requires at least {} whole-array loads, got {loads}",
            expected.entry,
            expected.minimum
        );
    }
}

#[test]
fn array_load_checks_follow_calls_without_counting_other_entries() {
    use wspirv::{binary::Assemble, dr::Builder, spirv};
    let mut builder = Builder::new();
    builder.capability(spirv::Capability::Shader);
    builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = builder.type_void();
    let integer = builder.type_int(32, 0);
    let count = builder.constant_bit32(integer, 4);
    let array = builder.type_array(integer, count);
    let pointer = builder.type_pointer(None, spirv::StorageClass::Function, array);
    let signature = builder.type_function(void, []);
    let helper = builder.begin_function(void, None, spirv::FunctionControl::NONE, signature).unwrap();
    builder.begin_block(None).unwrap();
    let local = builder.variable(pointer, None, spirv::StorageClass::Function, None);
    builder.load(array, None, local, None, []).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    for (name, call_helper) in [("wanted_finish", true), ("unrelated_finish", false)] {
        let function = builder.begin_function(void, None, spirv::FunctionControl::NONE, signature).unwrap();
        builder.begin_block(None).unwrap();
        if call_helper {
            builder.function_call(void, None, helper, []).unwrap();
        }
        builder.ret().unwrap();
        builder.end_function().unwrap();
        builder.entry_point(spirv::ExecutionModel::GLCompute, function, name, []);
        builder.execution_mode(function, spirv::ExecutionMode::LocalSize, [1, 1, 1]);
    }
    let bytes: Vec<_> = builder.module().assemble().iter().flat_map(|word| word.to_le_bytes()).collect();
    check(
        &bytes,
        &[ArrayLoads {
            entry: "wanted".into(),
            minimum: 1,
        }],
    );
    let expected: Instructions =
        toml::from_str("entry = 'wanted'\ncounts = { Load = 1 }\narray_traffic = [1,1,0]\narray_phis = 0")
            .unwrap();
    instructions(&bytes, &[expected], None);
    for config in [
        "entry = 'unrelated'\ncounts = { Load = 1 }",
        "entry = 'wanted'\nmaximum = { Load = 0 }",
        "entry = 'missing'\ncounts = { Load = 0 }",
    ] {
        let expected: Instructions = toml::from_str(config).unwrap();
        assert!(std::panic::catch_unwind(|| instructions(&bytes, &[expected], None)).is_err());
    }
    for (entry, minimum) in [("wanted", 2), ("unrelated", 1), ("missing", 1)] {
        assert!(std::panic::catch_unwind(|| check(
            &bytes,
            &[ArrayLoads {
                entry: entry.into(),
                minimum
            }]
        ))
        .is_err());
    }
}

/// Ensure determinism coverage actually exercises multiple phis in one block.
pub fn check_block_phis(bytes: &[u8], minimum: usize) {
    assert!(minimum > 0);
    let module = wspirv::dr::load_bytes(bytes).expect("SPIR-V module");
    assert!(
        module.functions.iter().flat_map(|f| &f.blocks).any(|block| block
            .instructions
            .iter()
            .filter(|i| i.class.opcode == Op::Phi)
            .count()
            >= minimum),
        "expected a block with at least {minimum} phis"
    );
}

/// Instruction expectations are scoped to an entry and every reachable helper.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instructions {
    entry: String,
    #[serde(default)]
    counts: BTreeMap<String, usize>,
    #[serde(default)]
    maximum: BTreeMap<String, usize>,
    #[serde(default)]
    present: Vec<String>,
    #[serde(default)]
    absent: Vec<String>,
    #[serde(default)]
    ext_inst: BTreeMap<String, usize>,
    array_traffic: Option<[usize; 3]>,
    array_phis: Option<usize>,
    array_loads: Option<usize>,
    minimum_array_allocations: Option<usize>,
}

pub fn instructions(bytes: &[u8], expectations: &[Instructions], helpers: Option<usize>) {
    let module = wspirv::dr::load_bytes(bytes).unwrap();
    if let Some(helpers) = helpers {
        assert_eq!(
            module.functions.len(),
            module.entry_points.len() + helpers,
            "SPIR-V helper count"
        );
    }
    let arrays: BTreeSet<_> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypeArray)
        .filter_map(|i| i.result_id)
        .collect();
    let pointers: BTreeSet<_> = module.types_global_values.iter().filter_map(|i| {
        matches!(i.operands.as_slice(), [Operand::StorageClass(wspirv::spirv::StorageClass::Function), Operand::IdRef(ty)]
            if i.class.opcode == Op::TypePointer && arrays.contains(ty)).then_some(i.result_id).flatten()
    }).collect();
    let types: BTreeMap<_, _> =
        module.all_inst_iter().filter_map(|i| Some((i.result_id?, i.result_type?))).collect();
    let functions: BTreeMap<_, _> =
        module.functions.iter().map(|f| (f.def.as_ref().unwrap().result_id.unwrap(), f)).collect();
    for expected in expectations {
        let mut pending = Vec::new();
        for entry in &module.entry_points {
            if let [_, Operand::IdRef(id), Operand::LiteralString(name), ..] = entry.operands.as_slice() {
                if name == &expected.entry || name.starts_with(&format!("{}_", expected.entry)) {
                    pending.push(*id);
                }
            }
        }
        assert!(!pending.is_empty(), "missing SPIR-V entry {}", expected.entry);
        let mut visited = BTreeSet::new();
        let mut instructions = Vec::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            for i in functions[&id].blocks.iter().flat_map(|b| &b.instructions) {
                if i.class.opcode == Op::FunctionCall {
                    if let Some(Operand::IdRef(callee)) = i.operands.first() {
                        pending.push(*callee);
                    }
                }
                instructions.push(i);
            }
        }
        let count =
            |name: &str| instructions.iter().filter(|i| format!("{:?}", i.class.opcode) == name).count();
        for (opcode, wanted) in &expected.counts {
            assert_eq!(count(opcode), *wanted, "{} {opcode}", expected.entry);
        }
        for (opcode, wanted) in &expected.maximum {
            assert!(
                count(opcode) <= *wanted,
                "{} {opcode}: {} > {wanted}",
                expected.entry,
                count(opcode)
            );
        }
        for opcode in &expected.present {
            assert!(count(opcode) > 0, "{} missing {opcode}", expected.entry);
        }
        for opcode in &expected.absent {
            assert_eq!(count(opcode), 0, "{} unexpected {opcode}", expected.entry);
        }
        for (name, wanted) in &expected.ext_inst {
            let opcode = match name.as_str() {
                "Sin" => wspirv::spirv::GLOp::Sin as u32,
                _ => panic!("unknown extended instruction {name}"),
            };
            let actual = instructions
                .iter()
                .filter(|i| {
                    i.class.opcode == Op::ExtInst
                        && i.operands.get(1) == Some(&Operand::LiteralExtInstInteger(opcode))
                })
                .count();
            assert_eq!(actual, *wanted, "{} {name}", expected.entry);
        }
        let allocations = instructions
            .iter()
            .filter(|i| {
                i.class.opcode == Op::Variable && i.result_type.is_some_and(|ty| pointers.contains(&ty))
            })
            .count();
        let loads = instructions
            .iter()
            .filter(|i| i.class.opcode == Op::Load && i.result_type.is_some_and(|ty| arrays.contains(&ty)))
            .count();
        let stores = instructions.iter().filter(|i| i.class.opcode == Op::Store && matches!(i.operands.first(), Some(Operand::IdRef(ptr)) if types.get(ptr).is_some_and(|ty| pointers.contains(ty)))).count();
        if let Some(wanted) = expected.array_traffic {
            assert_eq!(
                [allocations, loads, stores],
                wanted,
                "{} array traffic",
                expected.entry
            );
        }
        if let Some(wanted) = expected.array_phis {
            assert_eq!(
                instructions
                    .iter()
                    .filter(|i| i.class.opcode == Op::Phi
                        && i.result_type.is_some_and(|ty| arrays.contains(&ty)))
                    .count(),
                wanted,
                "{} array phis",
                expected.entry
            );
        }
        if let Some(wanted) = expected.array_loads {
            assert_eq!(loads, wanted, "{} array loads", expected.entry);
        }
        if let Some(minimum) = expected.minimum_array_allocations {
            assert!(
                allocations >= minimum,
                "{} missing array allocation",
                expected.entry
            );
        }
    }
}

pub fn round_trip(bytes: &[u8]) {
    use wspirv::spirv::{Decoration, StorageClass};
    let module = naga::front::spv::parse_u8_slice(bytes, &Default::default()).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    for entry in &module.entry_points {
        let options = naga::back::spv::Options {
            lang_version: (1, 6),
            ..Default::default()
        };
        let pipeline = naga::back::spv::PipelineOptions {
            shader_stage: entry.stage,
            entry_point: entry.name.clone(),
        };
        let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).unwrap();
        let output = wspirv::dr::load_words(words).unwrap();
        // Explicit array/struct layouts are forbidden on function-local variables.
        let layouts: BTreeSet<_> = output
            .annotations
            .iter()
            .filter_map(|i| match i.operands.as_slice() {
                [Operand::IdRef(ty), Operand::Decoration(Decoration::ArrayStride), ..]
                | [Operand::IdRef(ty), _, Operand::Decoration(Decoration::Offset), ..] => Some(*ty),
                _ => None,
            })
            .collect();
        let invalid: BTreeSet<_> = output
            .types_global_values
            .iter()
            .filter_map(|i| match (i.class.opcode, i.operands.as_slice()) {
                (Op::TypePointer, [Operand::StorageClass(StorageClass::Function), Operand::IdRef(ty)])
                    if layouts.contains(ty) =>
                {
                    i.result_id
                }
                _ => None,
            })
            .collect();
        assert!(
            !output
                .all_inst_iter()
                .any(|i| i.class.opcode == Op::Variable
                    && i.result_type.is_some_and(|ty| invalid.contains(&ty))),
            "{} has explicitly laid-out function storage",
            entry.name
        );
    }
}
