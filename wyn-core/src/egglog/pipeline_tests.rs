//! Regression inputs exercise the public pipeline and backend-visible output.
use crate::{compile_thru_ssa, lower_ssa_to_spirv, lower_ssa_to_wgsl};
use naga::{BinaryOperator, Expression, Statement};

fn storage_loads(function: &naga::Function, global: naga::Handle<naga::GlobalVariable>) -> usize {
    function
        .expressions
        .iter()
        .filter(|(_, expression)| {
            let Expression::Load { pointer } = expression else {
                return false;
            };
            let mut pointer = *pointer;
            loop {
                match function.expressions[pointer] {
                    Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                        pointer = base
                    }
                    Expression::GlobalVariable(value) => return value == global,
                    _ => return false,
                }
            }
        })
        .count()
}

#[test]
fn fused_collectives_share_producer_elements() {
    let module = shaders(
        "entry main(xs:[]i32) (i32,i32) =
        let a=map(|x|x*x+17,xs) in
        (reduce((+),0,a),reduce(|x,y|if x>y then x else y,0,a))",
    );
    let input = module
        .global_variables
        .iter()
        .find(|(_, v)| v.binding.as_ref().is_some_and(|b| b.group == 0 && b.binding == 0))
        .unwrap()
        .0;
    let partials = module.entry_points.iter().find(|e| e.name.ends_with("_partials")).unwrap();
    assert_eq!(
        storage_loads(&partials.function, input),
        1,
        "one input load feeds both reductions"
    );
}

#[test]
fn dead_fused_outputs_do_not_evaluate_their_elements() {
    let module = shaders(
        "entry main(xs:[8]i32) [8]i32 =
        let dead=map(|x|x*x,xs) in let live=map(|x|x+17,xs) in
        scatter((#[scratch] dead),iota(8),live)",
    );
    for entry in &module.entry_points {
        assert!(
            !entry.function.expressions.iter().any(|(_, expression)| matches!(expression,
                Expression::Binary { op: BinaryOperator::Multiply, left, right } if left == right
            )),
            "unused producer arithmetic in {}",
            entry.name
        );
    }
}

#[test]
fn shared_producer_matrix_reaches_both_backends() {
    shaders(include_str!("../../../testfiles/rust_host_sharing.wyn"));
}

#[test]
fn radix_scratch_initializer_has_no_array_allocation() {
    let source = format!(
        "{}\n entry main(xs: []i32) []i32 = radix_sort_step(xs, |i:i32,x:i32| (x >> i) & 1, 0)",
        include_str!("../../../pkg/sort/src/radix_sort.wyn")
    );
    let compiled = crate::lower_ssa_to_wgsl_with_program(compile_thru_ssa(&source).unwrap()).unwrap();
    let entry = &compiled.program.entries[0];
    let arrays = entry.allocations.iter().filter(|allocation| matches!(allocation,
        crate::host::Allocation::Buffer { bytes, .. } if !matches!(bytes, crate::host::Expr::Integer(_))
    )).count();
    assert_eq!(
        arrays, 2,
        "only scan prefixes and sorted output scale with input length"
    );
}

#[test]
fn basic_scalar_policy_retains_calls_and_validates_both_backends() {
    use super::{from_tlc, fuse, optimize_with_policy, place, schedule, to_ssa, ScalarOptimization};
    use crate::{CodegenTarget, PipelineTopologyPolicy};
    let helper = "def helper(x:i32) i32 =
        (x-x)+(x-x)+(x-x)+(x-x)+(x-x)+(x-x)+(x-x)+(x-x)
        entry main(x:i32) i32 = helper(x)+1";
    for source in [
        helper,
        include_str!("../../../testfiles/regressions/aggregate_forwarding.wyn"),
        include_str!("../../../testfiles/select_lowering.wyn"),
        include_str!("../../../testfiles/filter_then_map.wyn"),
    ] {
        let tlc = crate::tlc::infer_input_slice_bounds(crate::compile_thru_tlc(source).unwrap());
        for policy in [ScalarOptimization::Basic, ScalarOptimization::Full] {
            for target in [CodegenTarget::Spirv, CodegenTarget::Wgsl] {
                let program = optimize_with_policy(
                    schedule(
                        place(
                            fuse(from_tlc(&tlc).unwrap()).unwrap(),
                            PipelineTopologyPolicy::AllowGenerated,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                    policy,
                )
                .unwrap();
                let mut templates = 0;
                program.stage.scalars.constructor_enodes("ScalarInlineBody", |_| templates += 1).unwrap();
                if policy == ScalarOptimization::Basic {
                    assert_eq!(templates, 0, "basic mode must not import optional templates");
                } else if source == helper {
                    assert!(templates > 0, "fixture must exercise optional helper expansion");
                }
                let ssa = to_ssa(program, target).unwrap();
                let module = match target {
                    CodegenTarget::Portable => unreachable!("test uses concrete shader targets"),
                    CodegenTarget::Spirv => {
                        let binary = lower_ssa_to_spirv(ssa).unwrap();
                        if source == helper {
                            let module = wspirv::dr::load_words(&binary.spirv).unwrap();
                            assert!(
                                !module.all_inst_iter().any(|i| i.class.opcode == wspirv::spirv::Op::ISub),
                                "cheap x-x cleanup must remain enabled"
                            );
                        }
                        let bytes: Vec<_> = binary.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                        naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap()
                    }
                    CodegenTarget::Wgsl => {
                        naga::front::wgsl::parse_str(&lower_ssa_to_wgsl(ssa).unwrap()).unwrap()
                    }
                };
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
    }
}

fn shaders(source: &str) -> naga::Module {
    let program = compile_thru_ssa(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
    let wgsl = lower_ssa_to_wgsl(program.clone()).unwrap_or_else(|error| panic!("{error}\n{source}"));
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("{}\n{wgsl}", error.emit_to_string(&wgsl)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{error:?}\n{wgsl}"));
    let binary = lower_ssa_to_spirv(program).unwrap_or_else(|error| panic!("{error}\n{source}"));
    wspirv::dr::load_words(&binary.spirv).expect("valid SPIR-V encoding");
    module
}

#[test]
fn scalar_entry_reaches_both_backends() {
    let module = shaders("entry scalar(x:i32) i32=x+1");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].stage, naga::ShaderStage::Compute);
}

#[test]
fn selections_reach_both_backends_without_rewriting_an_internal_ir() {
    shaders("entry choose(c:bool,x:i32,y:i32) i32=if c then x+1 else y+2");
}

#[test]
fn lexical_aliases_and_shadowing_preserve_values() {
    shaders("entry aliases(a:i32) i32=let x=a in let y=x+7 in let x=y*2 in x+y");
}

#[test]
fn helper_calls_in_different_functions_keep_their_parameters() {
    shaders("def a(x:i32) i32=x+1 def b(x:i32) i32=x-1 entry result(x:i32,y:i32) i32=a(x)+b(y)");
}

#[test]
fn batched_scalar_extraction_keeps_entry_contexts_distinct() {
    let module = shaders(
        "entry first(x:u32) u32=x+11u32
         entry second(x:u32) u32=x+29u32",
    );
    for (name, expected, other) in [("first", 11, 29), ("second", 29, 11)] {
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.name.contains(name))
            .expect("each authored entry must reach the backend");
        let literals: Vec<_> = entry
            .function
            .expressions
            .iter()
            .filter_map(|(_, expression)| match expression {
                Expression::Literal(naga::Literal::U32(value)) => Some(*value),
                _ => None,
            })
            .collect();
        assert!(
            literals.contains(&expected),
            "{name} lost its own arithmetic operand"
        );
        assert!(
            !literals.contains(&other),
            "{name} acquired another context's operand"
        );
    }
}

#[test]
fn selected_record_expansion_is_used_as_a_helper_argument() {
    // The redundant differences survive early TLC simplification and keep
    // repack above the early inlining threshold; egglog reduces them to zero.
    let module = shaders(
        "type camera = { target:vec3i32, az:i32, elev:i32, dist:i32, jitter:vec2i32 }
         type orbit = { target:vec3i32, azimuth:i32, elevation:i32, distance:i32 }
         def repack(o:camera) orbit = {
             target=o.target, azimuth=o.az, elevation=o.elev,
             distance=(o.dist-o.dist)+(o.dist-o.dist)+(o.dist-o.dist)+(o.dist-o.dist) +
                      (o.dist-o.dist)+(o.dist-o.dist)+(o.dist-o.dist)+(o.dist-o.dist) }
         def project(o:orbit) i32 =
             (o.target.x+o.azimuth+o.elevation)*o.distance +
             (o.target.y+o.azimuth+o.elevation)*o.distance +
             (o.target.z+o.azimuth+o.elevation)*o.distance
         entry main(o:camera) i32 = project(repack(o))",
    );
    assert!(
        module
            .functions
            .iter()
            .all(|(_, function)| { !function.name.as_deref().is_some_and(|name| name.contains("repack")) }),
        "the expanded record must not leave a callable repacking helper"
    );
}

#[test]
fn inlining_cost_counts_shared_arithmetic_once() {
    let module = shaders(
        "def shared(x:u32,y:u32) u32 =
             let a=x*y+x in
             let b=a*a+a in
             let c=b*b+b in
             let d=c*c+c in
             let e=d*d+d in
             if y==0u32 then x else e
         entry main(x:u32,y:u32) u32 = shared(x,y)",
    );
    assert!(
        module.functions.is_empty(),
        "a shared arithmetic DAG should inline"
    );
    let multiplies = module.entry_points[0]
        .function
        .expressions
        .iter()
        .filter(|(_, expression)| {
            matches!(
                expression,
                Expression::Binary {
                    op: BinaryOperator::Multiply,
                    ..
                }
            )
        })
        .count();
    assert_eq!(multiplies, 5, "inlining must preserve shared intermediate values");
}

#[test]
fn repeated_helper_substitution_keeps_independent_arguments_and_shared_work() {
    let module = shaders(
        "def shared(x:u32,y:u32) u32 =
             let a=x*y+x in
             let b=a*a+a in
             let c=b*b+b in
             let d=c*c+c in
             let e=d*d+d in
             if y==0u32 then x else e
         entry main(x:u32,y:u32,u:u32,v:u32) (u32,u32) = (shared(x,y),shared(u,v))",
    );
    assert!(
        module.functions.is_empty(),
        "both helper invocations should inline"
    );
    let multiplies = module.entry_points[0]
        .function
        .expressions
        .iter()
        .filter(|(_, expression)| {
            matches!(
                expression,
                Expression::Binary {
                    op: BinaryOperator::Multiply,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        multiplies, 10,
        "each call must retain its own five shared products"
    );
}

#[test]
fn guarded_partial_helpers_inline_at_their_call_sites() {
    for (expression, operator) in [
        ("x/y", BinaryOperator::Divide),
        ("x<<y", BinaryOperator::ShiftLeft),
    ] {
        let module = shaders(&format!(
            "def guarded(x:u32,y:u32) u32=if y==0u32 then x else {expression}
             entry main(x:u32,y:u32) u32=guarded(x,y)"
        ));
        assert!(
            module.functions.is_empty(),
            "guarded arithmetic should inline: {expression}"
        );
        let function = &module.entry_points[0].function;
        let emits_partial = |statement: &Statement| {
            let Statement::Emit(range) = statement else {
                return false;
            };
            range.clone().any(|value| {
                matches!(function.expressions[value], Expression::Binary { op, .. } if op == operator)
            })
        };
        assert!(
            !function.body.iter().any(&emits_partial),
            "partial arithmetic escaped its guard"
        );
        assert!(
            function.body.iter().any(|statement| {
                matches!(statement, Statement::If { accept, reject, .. }
                if !accept.iter().any(&emits_partial) && reject.iter().any(&emits_partial))
            }),
            "partial arithmetic must remain in the nonzero branch"
        );
    }
}

#[test]
fn partial_arithmetic_stays_conditional() {
    let module = shaders("entry choose(c:bool,x:i32,y:i32) i32=if c then x/y else 0");
    assert_eq!(module.entry_points.len(), 1);
}

#[test]
fn loops_with_accumulators_reach_both_backends() {
    shaders("entry sum(n:i32) i32=loop acc=0 for i<n do acc+i");
}

#[test]
fn tuples_and_vectors_reach_both_backends() {
    shaders("entry pair(x:i32) (i32,i32)=(x+1,x*2)");
}

#[test]
fn storage_input_reads_preserve_the_public_interface() {
    let module = shaders("entry first(xs:[]i32) i32=xs[0]");
    assert!(module.global_variables.iter().any(|(_, value)| value.binding.is_some()));
}

#[test]
fn pointwise_map_emits_a_parallel_dispatch() {
    let module = shaders("entry doubled(xs:[]i32) []i32=map(|x:i32|x*2,xs)");
    assert!(module.entry_points.iter().any(|entry| entry.workgroup_size == [64, 1, 1]));
}

#[test]
fn fused_maps_preserve_captured_inputs() {
    shaders("entry result(xs:[]i32,n:i32) []i32=map(|x:i32|x*2,map(|x:i32|x+n,xs))");
}

#[test]
fn reductions_have_chunk_and_combine_dispatches() {
    let module = shaders("entry sum(xs:[]i32) i32=reduce(|a:i32,b:i32|a+b,0,xs)");
    assert!(module.entry_points.iter().any(|entry| entry.workgroup_size == [256, 1, 1]));
}
#[test]
fn scans_have_prefix_and_offset_dispatches() {
    shaders("entry sums(xs:[]i32) []i32=scan(|a:i32,b:i32|a+b,0,xs)");
}

#[test]
fn filters_emit_stable_workgroup_compaction() {
    shaders("entry positive(xs:[]i32) []i32=filter(|x:i32|x>0,xs)");
}
#[test]
fn nested_reduction_stays_inside_the_map_invocation() {
    shaders("entry sums(xs:[]i32) []i32=map(|x:i32|reduce(|a:i32,b:i32|a+b,0,map(|y:i32|y+x,iota(5))),xs)");
}

#[test]
fn ranked_buckets_preserve_nested_storage_and_atomic_counters() {
    shaders("entry bins(dest:*[2][2]i32) ([2][2]i32,[2]u32,u32)=bucket_scatter_2d(dest,[[(-1,9),(0,10),(0,11)],[(0,12),(1,20),(2,9)]])");
}

#[test]
fn integer_histogram_uses_the_selected_atomic_update() {
    let module=shaders("entry bins(dest:*[3]i32,xs:[5]i32) [3]i32=reduce_by_index(dest,|a:i32,b:i32|a+b,0,map(|x:i32|x-1,xs),map(|x:i32|x*3,xs))");
    assert_eq!(module.entry_points.len(), 1);
}

#[test]
fn float_histogram_retains_ordered_execution() {
    let module = shaders(
        "entry bins(dest:*[3]f32) [3]f32=reduce_by_index(dest,|a:f32,b:f32|a+b,0.0,[0,1,0],[1.0,2.0,3.0])",
    );
    assert!(module.entry_points.iter().all(|entry| entry.workgroup_size == [1, 1, 1]));
}

#[test]
fn local_scatter_keeps_updates_inside_each_invocation() {
    shaders(
        "entry result(xs:[]i32) []i32=map(|x:i32|let a=scatter([1,2,3],[0,1],[x,x+1]) in a[0]+a[1],xs)",
    );
}

#[test]
fn local_filter_preserves_its_live_length() {
    shaders("entry result(xs:[]i32) []i32=map(|x:i32|length(filter(|y:i32|y>x,[1,2,3])),xs)");
}

#[test]
fn independent_scalar_results_do_not_share_a_host_resource() {
    let source = include_str!("../../../testfiles/scalar_epilogues.wyn");
    let output = crate::lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap();
    let mut resources = std::collections::BTreeMap::new();
    for pipeline in &output.program.interface.pipelines {
        let crate::host::Pipeline::Compute(pipeline) = pipeline else {
            continue;
        };
        for binding in &pipeline.bindings {
            if let crate::host::Binding::StorageBuffer {
                set,
                binding,
                resource: Some(name),
                ..
            } = binding
            {
                assert!(!name.is_empty());
                if let Some(previous) = resources.insert(name, (*set, *binding)) {
                    assert_eq!(
                        previous,
                        (*set, *binding),
                        "distinct buffers must not acquire one host identity"
                    );
                }
            }
        }
    }
}

#[test]
fn nested_loop_invariants_keep_outer_iteration_dependencies() {
    shaders("entry nested(xs:[]i32) []i32 = map(|x:i32| loop a=0 for i<4 do a+(loop b=0 for j<3 do b+x*i+j), xs)");
}

#[test]
fn guarded_partial_arithmetic_keeps_its_control_dependency_inside_loops() {
    shaders("entry guarded(xs:[]i32) []i32 = map(|x:i32| loop a=0 for i<7 do a+(if x==0 then i else 120/x+i), xs)");
}

#[test]
fn guarded_division_is_shared_across_control_indexing_and_composed_callbacks() {
    for source in [
        "entry main(xs:[]i32) []i32 = map(|n|
         if n==0 then 7 else if 100/n>2 then 100/n+1 else 100/n+2,xs)",
        "entry main(xs:[]i32) []i32 = map(|n|
         if n==0 then 7 else loop a=100/n for i<2 do a+100/n,xs)",
        "entry main(xs:[]i32) []i32 = map(|x|
         let values=[x,x+1]
         let i=if x==0 then 0 else 10/x in values[i%2]+i,xs)",
        "entry main(xs:[]i32) []i32 =
         map(|p|p.0+p.1,map(|x|let q=if x==0 then 0 else 100/x in (q+1,q+2),xs))",
    ] {
        let module = shaders(source);
        let divisions = module
            .functions
            .iter()
            .map(|(_, f)| f)
            .chain(module.entry_points.iter().map(|e| &e.function))
            .flat_map(|f| f.expressions.iter())
            .filter(|(_, expression)| {
                matches!(
                    expression,
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(divisions, 1, "{source}");
    }
}

#[test]
fn camera_field_math_is_shared_across_indexing() {
    for (ty, angle) in [("{angle:f32}", "camera.angle"), ("vec2f32", "camera.x")] {
        // Keep the calculation per-element so host capture cannot remove it
        // from the shader whose sharing this regression checks.
        let source = format!(
            "entry main(xs:[]i32,camera:{ty}) []f32 = map(|x|
             let q=f32.sin({angle}+f32(x))
             let values=[x,x+1] in f32(values[i32(f32.abs(q))%2])+q,xs)"
        );
        let module = shaders(&source);
        let sines = module
            .functions
            .iter()
            .map(|(_, f)| f)
            .chain(module.entry_points.iter().map(|e| &e.function))
            .flat_map(|f| f.expressions.iter())
            .filter(|(_, expression)| {
                matches!(
                    expression,
                    Expression::Math {
                        fun: naga::MathFunction::Sin,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(sines, 1, "{source}");
    }
}
