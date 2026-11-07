//! Public shader-artifact checks for branch-dependent array update scheduling.
//! CPU-only: parse and validate WGSL; never create a GPU device or execute it.
use wyn_core::CodegenTarget;
#[path = "support/pipeline.rs"]
mod pipeline;

fn validate_wgsl(text: &str) {
    let module = naga::front::wgsl::parse_str(text).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

fn assert_branch_dependent_updates_snapshot_old_elements(same_index: bool) {
    use naga::{BinaryOperator, Expression, Statement, TypeInner};

    let source = include_str!("../../tests/compiler/cases/paired_updates_scalar_branch.wyn");
    let source = if same_index { source.replace("let b = pairs[n].y", "let b = a") } else { source.into() };
    let prepared = pipeline::source(&source);
    let ssa = if same_index {
        pipeline::lower_with_options(
            &prepared,
            CodegenTarget::Wgsl,
            wyn_core::egglog::ScalarOptimization::Full,
            wyn_core::PipelineTopologyPolicy::AllowGenerated,
        )
    } else {
        pipeline::lower(&prepared, CodegenTarget::Wgsl)
    };
    let text = wyn_core::lower_ssa_to_wgsl(ssa).unwrap();
    validate_wgsl(&text);
    let module = naga::front::wgsl::parse_str(&text).unwrap();
    fn blocks<'a>(block: &'a naga::Block, out: &mut Vec<&'a naga::Block>) {
        out.push(block);
        for statement in block.iter() {
            match statement {
                Statement::Block(block) => blocks(block, out),
                Statement::If { accept, reject, .. } => {
                    blocks(accept, out);
                    blocks(reject, out);
                }
                Statement::Loop { body, continuing, .. } => {
                    blocks(body, out);
                    blocks(continuing, out);
                }
                _ => {}
            }
        }
    }
    let mut checked = 0;
    for entry in &module.entry_points {
        let function = &entry.function;
        let array_pointer = |pointer| {
            matches!(function.expressions[pointer], Expression::LocalVariable(v)
                if matches!(module.types[function.local_variables[v].ty].inner, TypeInner::Array { .. }))
        };
        let element = |pointer| match function.expressions[pointer] {
            Expression::Access { base, index } if array_pointer(base) => Some((base, index)),
            _ => None,
        };
        for (_, expression) in function.expressions.iter() {
            assert!(
                !matches!(expression, Expression::Load { pointer } if array_pointer(*pointer)),
                "branch-dependent update retained a whole-array snapshot: {text}"
            );
        }
        let mut nested = Vec::new();
        blocks(&function.body, &mut nested);
        for block in nested {
            let writes: Vec<_> = block
                .iter()
                .enumerate()
                .filter_map(|(at, statement)| {
                    let Statement::Store { pointer, value } = *statement else {
                        return None;
                    };
                    let address = element(pointer)?;
                    matches!(
                        function.expressions[value],
                        Expression::Binary {
                            op: BinaryOperator::Add | BinaryOperator::Subtract,
                            ..
                        }
                    )
                    .then_some((at, address, value))
                })
                .collect();
            if !writes.iter().any(|&(_, _, value)| {
                matches!(
                    function.expressions[value],
                    Expression::Binary {
                        op: BinaryOperator::Subtract,
                        ..
                    }
                )
            }) {
                continue;
            }
            checked += 1;
            assert!(writes.len() == 2 || (same_index && writes.len() == 1));
            for &(_, address, value) in &writes {
                let Expression::Binary { left, .. } = function.expressions[value] else {
                    unreachable!()
                };
                let Expression::Load { pointer } = function.expressions[left] else {
                    panic!("old element read")
                };
                assert_eq!(element(pointer), Some(address));
                assert!(
                    block.iter().take(writes[0].0).any(|statement| matches!(statement,
                    Statement::Emit(range) if range.clone().any(|e| e == left))),
                    "old element must be read before either update: {text}"
                );
            }
            if same_index && writes.len() == 2 {
                assert_eq!(writes[0].1, writes[1].1);
            }
        }
    }
    assert_eq!(checked, 1, "must inspect the nested update loop: {text}");
}

#[test]
fn branch_dependent_updates_snapshot_both_old_elements() {
    assert_branch_dependent_updates_snapshot_old_elements(false);
}

#[test]
fn branch_dependent_updates_preserve_equal_index_semantics() {
    assert_branch_dependent_updates_snapshot_old_elements(true);
}
