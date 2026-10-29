//! Inspect emitted shader structure, scoped to an entry and its reachable helpers.
use naga::{AddressSpace, Block, Expression, Function, Handle, Module, ShaderStage, Statement};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expectation {
    entry: String,
    #[serde(default)]
    targets: Vec<String>,
    entry_count: Option<usize>,
    #[serde(default)]
    binary: BTreeMap<String, usize>,
    #[serde(default)]
    math: BTreeMap<String, usize>,
    #[serde(default)]
    parameter_loads: BTreeMap<String, usize>,
    local_array_traffic: Option<[usize; 3]>,
    #[serde(default)]
    forbidden_local_array_lengths: Vec<u32>,
    workgroup_array_scalar: Option<String>,
    selects_present: Option<bool>,
    ifs_present: Option<bool>,
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

pub fn check(bytes: &[u8], target: &str, expectations: &[Expectation], host: &str) {
    let host = wyn_host_interp::Program::parse(host).expect("valid shader host ABI");
    let module = match target {
        "spirv" => naga::front::spv::parse_u8_slice(bytes, &Default::default()).expect("SPIR-V module"),
        "wgsl" => naga::front::wgsl::parse_str(std::str::from_utf8(bytes).unwrap()).expect("WGSL module"),
        _ => unreachable!(),
    };
    for expected in expectations {
        assert!(
            expected.targets.iter().all(|target| matches!(target.as_str(), "spirv" | "wgsl")),
            "unknown shader-check target"
        );
        if expected.targets.is_empty() || expected.targets.iter().any(|t| t == target) {
            check_module(&module, expected, Some(&host));
        }
    }
}

fn check_module(module: &Module, expected: &Expectation, host: Option<&wyn_host_interp::Program>) {
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
    if let Some(count) = expected.entry_count {
        assert_eq!(entries.len(), count, "{} shader entries", expected.entry);
    }
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
    let mut binary = BTreeMap::<String, usize>::new();
    let mut math = BTreeMap::<String, usize>::new();
    let mut loads = BTreeMap::<u32, usize>::new();
    let mut traffic = [0usize; 3];
    let mut selects = 0;
    let mut ifs = 0;
    let mut workgroup_arrays = BTreeSet::new();
    for function in &functions {
        for (_, variable) in function.local_variables.iter() {
            if let naga::TypeInner::Array {
                size: naga::ArraySize::Constant(n),
                ..
            } = module.types[variable.ty].inner
            {
                assert!(
                    !expected.forbidden_local_array_lengths.contains(&n.get()),
                    "{} unexpectedly materializes a local array of {} elements",
                    expected.entry,
                    n
                );
            }
        }
        let arrays: BTreeSet<_> = function
            .local_variables
            .iter()
            .filter(|(_, v)| matches!(module.types[v.ty].inner, naga::TypeInner::Array { .. }))
            .map(|(id, _)| id)
            .collect();
        traffic[0] += arrays.len();
        let whole_array = |pointer| matches!(function.expressions[pointer], Expression::LocalVariable(id) if arrays.contains(&id));
        for (_, expression) in function.expressions.iter() {
            match expression {
                Expression::Binary { op, .. } => *binary.entry(format!("{op:?}")).or_default() += 1,
                Expression::Math { fun, .. } => *math.entry(format!("{fun:?}")).or_default() += 1,
                Expression::Select { .. } => selects += 1,
                Expression::GlobalVariable(id)
                    if module.global_variables[*id].space == AddressSpace::WorkGroup =>
                {
                    workgroup_arrays.insert(*id);
                }
                Expression::Load { pointer } => {
                    if whole_array(*pointer) {
                        traffic[1] += 1;
                    }
                    let mut base = *pointer;
                    loop {
                        match function.expressions[base] {
                            Expression::Access { base: parent, .. }
                            | Expression::AccessIndex { base: parent, .. } => base = parent,
                            Expression::GlobalVariable(id) => {
                                let variable = &module.global_variables[id];
                                if matches!(variable.space, AddressSpace::Storage { .. }) {
                                    if let Some(binding) = &variable.binding {
                                        if binding.group == 0 {
                                            *loads.entry(binding.binding).or_default() += 1;
                                        }
                                    }
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                }
                _ => {}
            }
        }
        statements(&function.body, 0, 0, &mut |statement, _, _| match statement {
            Statement::If { .. } => ifs += 1,
            Statement::Store { pointer, .. } if whole_array(*pointer) => traffic[2] += 1,
            _ => {}
        });
    }
    for (op, count) in &expected.binary {
        assert_eq!(
            binary.get(op).copied().unwrap_or(0),
            *count,
            "{} {op}",
            expected.entry
        );
    }
    for (op, count) in &expected.math {
        assert_eq!(
            math.get(op).copied().unwrap_or(0),
            *count,
            "{} {op}",
            expected.entry
        );
    }
    for (parameter, count) in &expected.parameter_loads {
        assert_eq!(entries.len(), 1, "parameter-load checks need one shader entry");
        let host = host.expect("host ABI for parameter loads");
        let kernel = host
            .kernels
            .values()
            .find(|kernel| kernel.options.get(":entry").unwrap().text().unwrap() == entries[0].name)
            .expect("shader kernel ABI");
        let index: usize = parameter.parse().expect("numeric parameter index");
        let name = &kernel.parameters[index].name;
        let abi = kernel.options.get(":abi").unwrap().list().unwrap();
        let binding = abi
            .iter()
            .map(|binding| binding.list().unwrap())
            .find(|binding| binding[0].text().unwrap() == name)
            .unwrap();
        assert_eq!(binding[1].text().unwrap(), ":storage");
        assert_eq!(binding[2].u64().unwrap(), 0);
        let slot = binding[3].u64().unwrap() as u32;
        assert_eq!(
            loads.get(&slot).copied().unwrap_or(0),
            *count,
            "{} parameter {parameter} loads",
            expected.entry
        );
    }
    if let Some(wanted) = expected.local_array_traffic {
        assert_eq!(traffic, wanted, "{} local array traffic", expected.entry);
    }
    if let Some(wanted) = expected.selects_present {
        assert_eq!(selects > 0, wanted, "{} selects", expected.entry);
    }
    if let Some(wanted) = expected.ifs_present {
        assert_eq!(ifs > 0, wanted, "{} conditionals", expected.entry);
    }
    if let Some(scalar) = &expected.workgroup_array_scalar {
        assert!(
            !workgroup_arrays.is_empty(),
            "{} missing workgroup arrays",
            expected.entry
        );
        for id in workgroup_arrays {
            let naga::TypeInner::Array { base, .. } = module.types[module.global_variables[id].ty].inner
            else {
                panic!("workgroup array required");
            };
            let wanted = match scalar.as_str() {
                "u32" => naga::Scalar::U32,
                _ => panic!("unknown workgroup scalar {scalar}"),
            };
            assert_eq!(
                module.types[base].inner,
                naga::TypeInner::Scalar(wanted),
                "{} workgroup element",
                expected.entry
            );
        }
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
    check_module(&module, &expected, None);
    let wrong: Expectation = toml::from_str("entry = 'plain'\nloop_depths = [0]").unwrap();
    assert!(std::panic::catch_unwind(|| check_module(&module, &wrong, None)).is_err());
}

#[test]
fn parameter_load_checks_follow_host_abi_instead_of_assuming_binding_zero() {
    let module = naga::front::wgsl::parse_str("@group(0) @binding(7) var<storage, read> input: array<u32>; @compute @workgroup_size(1) fn wanted() { let value = input[0]; }").unwrap();
    let host = wyn_host_interp::Program::parse(
        r#"
(define-host-program :version 1)
(define-gpu-module 'shaders :format :wgsl :path "unused.wgsl")
(define-gpu-kernel 'kernel :module 'shaders :entry "wanted"
  :workgroup-size '(1 1 1)
  :parameters '((argument-0 :buffer :read :element :u32 :stride 4))
  :abi '((argument-0 :storage 0 7)))
"#,
    )
    .unwrap();
    let expected: Expectation = toml::from_str("entry = 'wanted'\nparameter_loads = { '0' = 1 }").unwrap();
    check_module(&module, &expected, Some(&host));
    let wrong: Expectation = toml::from_str("entry = 'wanted'\nparameter_loads = { '0' = 2 }").unwrap();
    assert!(std::panic::catch_unwind(|| check_module(&module, &wrong, Some(&host))).is_err());
}
