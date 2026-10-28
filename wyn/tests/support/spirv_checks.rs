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
