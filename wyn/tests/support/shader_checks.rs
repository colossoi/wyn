//! Inspect emitted shader structure, scoped to an entry and its reachable helpers.
use naga::{AddressSpace, Block, Expression, Function, Handle, Module, ShaderStage, Statement};
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expectation {
    entry: String,
    stage: Option<String>,
    workgroup_size: Option<[u32; 3]>,
    #[serde(default)]
    workgroup_storage: bool,
    #[serde(default)]
    barrier: bool,
    loop_depths: Option<Vec<usize>>,
    guarded_loops: Option<[usize; 2]>,
    atomic: Option<String>,
}

fn statements(
    block: &Block,
    loops: usize,
    guards: usize,
    visit: &mut impl FnMut(&Statement, usize, usize),
) {
    for statement in block.iter() {
        visit(statement, loops, guards);
        match statement {
            Statement::Block(block) => statements(block, loops, guards, visit),
            Statement::If { accept, reject, .. } => {
                statements(accept, loops, guards + 1, visit);
                statements(reject, loops, guards + 1, visit);
            }
            Statement::Loop { body, continuing, .. } => {
                statements(body, loops + 1, guards, visit);
                statements(continuing, loops + 1, guards, visit);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    statements(&case.body, loops, guards, visit);
                }
            }
            _ => {}
        }
    }
}

pub fn check(bytes: &[u8], target: &str, expectations: &[Expectation]) {
    let module = match target {
        "spirv" => naga::front::spv::parse_u8_slice(bytes, &Default::default()).expect("SPIR-V module"),
        "wgsl" => naga::front::wgsl::parse_str(std::str::from_utf8(bytes).unwrap()).expect("WGSL module"),
        _ => unreachable!(),
    };
    for expected in expectations {
        check_module(&module, expected);
    }
}

fn check_module(module: &Module, expected: &Expectation) {
    let entries: Vec<_> = module
        .entry_points
        .iter()
        .filter(|entry| {
            (entry.name == expected.entry || entry.name.starts_with(&format!("{}_", expected.entry)))
                && expected.stage.as_deref().is_none_or(|stage| match stage {
                    "compute" => entry.stage == ShaderStage::Compute,
                    "fragment" => entry.stage == ShaderStage::Fragment,
                    "vertex" => entry.stage == ShaderStage::Vertex,
                    _ => panic!("unknown shader stage {stage}"),
                })
        })
        .collect();
    assert!(!entries.is_empty(), "missing shader entry {}", expected.entry);
    if let Some(size) = expected.workgroup_size {
        assert!(
            entries.iter().any(|entry| entry.workgroup_size == size),
            "{} workgroup size",
            expected.entry
        );
    }
    let mut functions: Vec<&Function> = entries.iter().map(|entry| &entry.function).collect();
    let mut visited: BTreeSet<Handle<Function>> = BTreeSet::new();
    let mut index = 0;
    while index < functions.len() {
        let mut callees = Vec::new();
        statements(&functions[index].body, 0, 0, &mut |statement, _, _| {
            if let Statement::Call { function, .. } = statement {
                callees.push(*function);
            }
        });
        for callee in callees {
            if visited.insert(callee) {
                functions.push(&module.functions[callee]);
            }
        }
        index += 1;
    }
    let mut depths = Vec::new();
    let mut guards = Vec::new();
    let mut barrier = false;
    let mut atomics = Vec::new();
    for function in &functions {
        statements(
            &function.body,
            0,
            0,
            &mut |statement, depth, guard| match statement {
                Statement::Loop { .. } => {
                    depths.push(depth);
                    guards.push(guard);
                }
                Statement::ControlBarrier(_) => barrier = true,
                Statement::Atomic { fun, .. } => atomics.push(format!("{fun:?}")),
                _ => {}
            },
        );
    }
    if let Some(wanted) = &expected.loop_depths {
        depths.sort_unstable();
        let mut wanted = wanted.clone();
        wanted.sort_unstable();
        assert_eq!(depths, wanted, "{} loop nesting", expected.entry);
    }
    if let Some([count, unguarded]) = expected.guarded_loops {
        assert_eq!(guards.len(), count, "{} loop count", expected.entry);
        assert_eq!(
            guards.iter().filter(|&&depth| depth == 0).count(),
            unguarded,
            "{} unguarded loops",
            expected.entry
        );
    }
    if expected.barrier {
        assert!(barrier, "{} missing workgroup barrier", expected.entry);
    }
    if let Some(atomic) = &expected.atomic {
        assert!(
            atomics.iter().any(|actual| actual.starts_with(atomic)),
            "{} missing atomic {atomic}: {atomics:?}",
            expected.entry
        );
    }
    if expected.workgroup_storage {
        assert!(
            functions.iter().any(|function| function.expressions.iter().any(|(_, expression)| {
                matches!(expression, Expression::GlobalVariable(variable)
                if module.global_variables[*variable].space == AddressSpace::WorkGroup)
            })),
            "{} missing reachable workgroup storage",
            expected.entry
        );
    }
}

#[test]
fn checks_do_not_count_unrelated_entries() {
    let module = naga::front::wgsl::parse_str("@compute @workgroup_size(1) fn plain() {}\n@compute @workgroup_size(2) fn looping() { loop { break; } }").unwrap();
    let expected: Expectation =
        toml::from_str("entry = 'plain'\nloop_depths = []\nworkgroup_size = [1,1,1]").unwrap();
    check_module(&module, &expected);
    let wrong: Expectation = toml::from_str("entry = 'plain'\nloop_depths = [0]").unwrap();
    assert!(std::panic::catch_unwind(|| check_module(&module, &wrong)).is_err());
}
