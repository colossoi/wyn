//! Source-to-artifact contracts migrated out of the unit-test harness.
#[path = "support/pipeline.rs"]
mod pipeline;

use std::sync::OnceLock;
use wyn_core::egglog::ScalarOptimization;
use wyn_core::host::{ComputePipeline, DispatchSize, Expr, Pipeline, ResultLayout};
use wyn_core::ssa::stage::Elaborated;
use wyn_core::tlc::{stage::OwnershipApplied, DefMeta};
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl, CodegenTarget, PipelineTopologyPolicy};

fn compute(program: &Elaborated) -> &ComputePipeline {
    let [Pipeline::Compute(compute)] = program.global_context.pipeline.pipelines.as_slice() else {
        panic!("one compute pipeline per source entry");
    };
    compute
}

// Share frontend/TLC work, without coupling unrelated entries during optimization.
fn entry(source: &OwnershipApplied, name: &str) -> OwnershipApplied {
    let mut source = source.clone();
    let mut found = 0;
    source.defs.retain(|definition| match &definition.meta {
        DefMeta::EntryPoint(entry) => {
            let keep = entry.declaration.name == name;
            found += usize::from(keep);
            keep
        }
        _ => true,
    });
    assert_eq!(found, 1, "source entry {name}");
    source
}

fn validate_wgsl(text: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    validate(&module);
    module
}

fn validate(module: &naga::Module) {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .unwrap();
}

fn shaders(program: Elaborated) {
    let spirv = lower_ssa_to_spirv(program.clone()).unwrap();
    assert!(!spirv.spirv.is_empty());
    let module =
        naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &Default::default()).parse().unwrap();
    validate(&module);
    validate_wgsl(&lower_ssa_to_wgsl(program).unwrap());
}

fn fixed(expression: &Expr) -> i64 {
    match expression {
        Expr::Integer(n) => *n,
        Expr::Add(a, b) => fixed(a) + fixed(b),
        Expr::Subtract(a, b) => fixed(a) - fixed(b),
        Expr::Multiply(a, b) => fixed(a) * fixed(b),
        Expr::Floor(a, b) => fixed(a).div_euclid(fixed(b)),
        Expr::Ceiling(a, b) => -(-fixed(a)).div_euclid(fixed(b)),
        Expr::Min(a, b) => fixed(a).min(fixed(b)),
        Expr::Max(a, b) => fixed(a).max(fixed(b)),
        other => panic!("fixed-size dispatch needs a constant extent: {other:?}"),
    }
}

fn groups(stage: &wyn_core::host::ComputeStage) -> u32 {
    match &stage.dispatch_size {
        DispatchSize::Fixed { x, y: 1, z: 1, .. } => *x,
        DispatchSize::Computed { groups, .. } => {
            assert_eq!((fixed(&groups[1]), fixed(&groups[2])), (1, 1));
            fixed(&groups[0]).try_into().unwrap()
        }
        other => panic!("one-dimensional dispatch: {other:?}"),
    }
}

#[test]
fn radix_step_has_three_phases() {
    let source = pipeline::source(&format!(
        "{}\nentry main(xs:[]i32) []i32=radix_sort_step(xs,|i:i32,x:i32|(x>>i)&1,0)",
        include_str!("../../pkg/sort/src/radix_sort.wyn")
    ));
    let program = pipeline::lower(&source, CodegenTarget::Portable);
    assert_eq!(compute(&program).stages.len(), 3);
}

fn four_bit_source() -> &'static OwnershipApplied {
    static SOURCE: OnceLock<OwnershipApplied> = OnceLock::new();
    SOURCE.get_or_init(|| {
        pipeline::source(&format!(
            "{}\nentry dynamic(xs:[]vec2u32) []vec2u32=radix_sort_step(xs,|i,p|u32.get_bit(i,p.x),0)
         entry fluid(xs:[4320]vec2u32) [4320]vec2u32=radix_sort_step(xs,|i,p|u32.get_bit(i,p.x),0)",
            include_str!("../../pkg/sort/src/four_bit.wyn")
        ))
    })
}

#[test]
fn four_bit_radix_step_has_three_phases() {
    let source = entry(four_bit_source(), "dynamic");
    let program = pipeline::lower(&source, CodegenTarget::Portable);
    let stages = &compute(&program).stages;
    assert_eq!(stages.len(), 3);
    assert_eq!(stages[0].workgroup_size, (64, 1, 1));
}

#[test]
fn fluid_radix_scan_avoids_empty_groups_and_oversized_carries() {
    let source = entry(four_bit_source(), "fluid");
    let program = pipeline::lower(&source, CodegenTarget::Portable);
    let stages = &compute(&program).stages;
    assert_eq!(stages.len(), 3);
    assert_eq!(stages[0].workgroup_size, (64, 1, 1));
    let count = groups(&stages[0]);
    assert!((1..=4320_u32.div_ceil(64)).contains(&count));
    assert_eq!(groups(&stages[1]), 1);
    assert_eq!(
        stages[1].workgroup_size,
        (count.next_power_of_two().max(32), 1, 1)
    );
}

#[test]
fn selected_traversal_uses_shared_region_declarations() {
    let source = r#"
def row(x:i32) [3]i32=map(|y|x+y,iota(3))
          entry case_0(xs:[]i32) [][3]i32=map(row,xs)
entry case_1(xs:[][3][17]i32) [][3]i32=
          map(|rows|map(|row|reduce((+),0,map(|x|x+row[0],row)),rows),xs)
entry case_2(xs:[][3][17]i32) [][3][17]i32=
          map(|rows|map(|row|map(|x|x*2,scan((+),0,map(|x|x+1,row))),rows),xs)
def sum_row(row:[17]i32) i32=reduce((+),0,row)
entry case_3(xs:[][17]i32) []i32=map(sum_row,xs)
def repeat_sum(n:i32,x:i32) i32=reduce((+),0,map(|y|x+y,iota(n)))
entry case_4(xs:[]i32,n:i32) []i32=map(|x|repeat_sum(n,x),xs)
entry case_5(xs:[][17]i32) []i32=map(|row|if row[0]>0 then reduce((+),0,row) else reduce((*),1,row),xs)
"#;
    let source = pipeline::source(source);
    for (index, rank) in [2, 2, 3, 1, 1, 1].into_iter().enumerate() {
        let source = entry(&source, &format!("case_{index}"));
        let program = pipeline::lower(&source, CodegenTarget::Portable);
        let results = &program.global_context.pipeline.source_results;
        assert_eq!(results.len(), 1);
        let mut dimensions = results[0].shape.expressions().len();
        let ResultLayout::Array { element, .. } = &results[0].layout else {
            panic!("source array result");
        };
        let mut layout = element.as_ref();
        while let ResultLayout::Sequence { element, .. } | ResultLayout::Array { element, .. } = layout {
            dimensions += 1;
            layout = element;
        }
        assert_eq!(dimensions, rank, "case_{index}: source array rank");
        shaders(program);
    }
}
#[test]
fn output_coordinate_maps_cover_every_axis_of_ranked_producers() {
    let sources = [
        "entry main() [12]i32=let rows=map(|i|[[10*i,10*i+1],[10*i+2,10*i+3]],iota(3)) in map(|a|rows[a%3][(a/3)%2][a/6],iota(12))",
        "entry main() [24]i32=let rows=map(|i|map(|j|map(|k|100*i+10*j+k,iota(4)),iota(3)),iota(2)) in map(|a|rows[a%2][(a/2)%3][a/6],iota(24))",
    ];
    let source = sources
        .iter()
        .enumerate()
        .map(|(i, source)| source.replacen("entry main(", &format!("entry case_{i}("), 1))
        .collect::<Vec<_>>()
        .join("\n");
    let source = pipeline::source(&source);
    for index in 0..sources.len() {
        let source = entry(&source, &format!("case_{index}"));
        let program = pipeline::lower(&source, CodegenTarget::Portable);
        assert_eq!(compute(&program).stages.len(), 1, "{}", sources[index]);
        shaders(program);
    }
}
#[test]
fn memory_rewrites_lower_through_nested_scopes() {
    let sources = [
        "entry main(xs:[]i32) [4]i32=map(|p|reduce((+),0i32,xs),iota(4))",
        "entry main(xs:[]i32,j:i32) [4]i32=map(|p|scan((+),0i32,xs)[j],iota(4))",
        "entry main(xs:[]i32) [2]i32=map(|x|x+1,xs[1..7][2..4])",
        "entry main(xs:[]i32,b:bool) [4]i32=map(|p|if b then reduce((+),0i32,xs) else p,iota(4))",
        "entry main(xs:[]i32,n:i32) i32=loop x=0i32 for i<n do x+reduce((+),0i32,xs)",
        "entry main() [4]i32=map(|p|let a=map(|i|i+2,iota(8)) in let b=map(|i|i+1,iota(8)) in a[0]+b[0],iota(4))",
    ];
    let source = sources
        .iter()
        .enumerate()
        .map(|(i, source)| source.replacen("entry main(", &format!("entry case_{i}("), 1))
        .collect::<Vec<_>>()
        .join("\n");
    let source = pipeline::source(&source);
    for index in 0..sources.len() {
        let source = entry(&source, &format!("case_{index}"));
        let program = pipeline::lower(&source, CodegenTarget::Portable);
        assert!(!compute(&program).stages.is_empty());
    }
}
#[test]
fn paired_updates_snapshot_elements_before_writing_even_when_indices_match() {
    use naga::{BinaryOperator, Expression, Literal, Statement, TypeInner};

    let source = include_str!("../../tests/compiler/cases/paired_array_updates.wyn");
    for same_index in [false, true] {
        let source =
            if same_index { source.replace("let b = pairs[k].y", "let b = a") } else { source.into() };
        let source = pipeline::source(&source);
        for policy in [ScalarOptimization::Basic, ScalarOptimization::Full] {
            let ssa = pipeline::lower_with_options(
                &source,
                CodegenTarget::Wgsl,
                policy,
                PipelineTopologyPolicy::AllowGenerated,
            );
            let text = lower_ssa_to_wgsl(ssa).unwrap();
            let module = validate_wgsl(&text);
            let mut checked = 0;
            for entry in &module.entry_points {
                let function = &entry.function;
                let array_pointer = |pointer| {
                    matches!(function.expressions[pointer], Expression::LocalVariable(v)
                        if matches!(module.types[function.local_variables[v].ty].inner, TypeInner::Array { .. }))
                };
                for (_, expression) in function.expressions.iter() {
                    assert!(
                        !matches!(expression, Expression::Load { pointer } if array_pointer(*pointer)),
                        "whole-array snapshot: {text}"
                    );
                }
                let element = |pointer| match function.expressions[pointer] {
                    Expression::Access { base, index } if array_pointer(base) => Some((base, index)),
                    _ => None,
                };
                for statement in entry.function.body.iter() {
                    let Statement::Loop { body, .. } = statement else {
                        continue;
                    };
                    let stores: Vec<_> = body
                        .iter()
                        .enumerate()
                        .filter_map(|(at, statement)| {
                            let Statement::Store { pointer, value } = *statement else {
                                return None;
                            };
                            element(pointer).map(|address| (at, address, value))
                        })
                        .collect();
                    let Some(&(_, _, last)) = stores.last() else {
                        continue;
                    };
                    if !matches!(
                        function.expressions[last],
                        Expression::Binary {
                            op: BinaryOperator::Subtract,
                            ..
                        }
                    ) {
                        continue;
                    }
                    checked += 1;
                    assert!(stores.len() == 2 || (same_index && stores.len() == 1));
                    let first_write = stores[0].0;
                    for (i, &(_, address, value)) in stores.iter().enumerate() {
                        let Expression::Binary { op, left, right } = function.expressions[value] else {
                            panic!("update arithmetic")
                        };
                        assert_eq!(
                            op,
                            if i + 1 == stores.len() {
                                BinaryOperator::Subtract
                            } else {
                                BinaryOperator::Add
                            }
                        );
                        let Expression::Splat { value: one, .. } = function.expressions[right] else {
                            panic!("vector update")
                        };
                        assert_eq!(function.expressions[one], Expression::Literal(Literal::F32(1.0)));
                        let Expression::Load { pointer } = function.expressions[left] else {
                            panic!("old element read")
                        };
                        assert_eq!(
                            element(pointer),
                            Some(address),
                            "replacement reads its original element"
                        );
                        assert!(
                            body.iter().take(first_write).any(|s| matches!(s,
                            Statement::Emit(range) if range.clone().any(|e| e == left))),
                            "both old elements must be read before either write: {text}"
                        );
                    }
                    if same_index && stores.len() == 2 {
                        assert_eq!(stores[0].1, stores[1].1, "exercise aliasing indices");
                    }
                }
            }
            assert_eq!(checked, 1, "must inspect the update loop: {text}");
        }
    }
}
#[test]
fn record_updates_survive_to_ssa() {
    let source = include_str!("../../testfiles/regressions/record_update.wyn");
    let source = pipeline::source(source);
    let program = pipeline::lower_with_options(
        &source,
        CodegenTarget::Portable,
        ScalarOptimization::Full,
        PipelineTopologyPolicy::AllowGenerated,
    );
    let count = program
        .functions
        .iter()
        .map(|f| &f.body)
        .chain(program.entry_points.iter().map(|e| &e.body))
        .flat_map(|body| body.inner.insts.values())
        .filter(|inst| {
            matches!(&inst.data,
            wyn_core::ssa::types::InstKind::Op {
                tag: wyn_core::op::OpTag::Intrinsic { id, .. }, ..
            } if *id == wyn_core::builtins::catalog().known().record_with)
        })
        .count();
    assert!(count >= 7, "dependent field updates must survive to SSA");
}
