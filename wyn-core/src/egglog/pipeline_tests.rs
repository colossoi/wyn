//! Regression inputs exercise the public pipeline and backend-visible output.
use crate::egglog::prepare_interfaces;
use crate::{compile_thru_ssa, lower_ssa_to_spirv, lower_ssa_to_wgsl};
use naga::{BinaryOperator, Expression, Statement};

#[test]
fn size_hint_survives_egglog_optimization() {
    use super::{finish_for_test, from_tlc, fuse, optimize_for_test, place, schedule};
    use crate::interface::Attribute;
    use crate::tlc::DefMeta;
    use crate::{CodegenTarget, PipelineTopologyPolicy};

    for source in [
        include_str!("../../../testfiles/reduce_compute.wyn"),
        include_str!("../../../testfiles/scan_compute.wyn"),
        "entry frame(#[size_hint(1024)] data: []vec4f32,
                     target: render_target<vec4f32>) render_target<vec4f32> =
           let fragments = rasterize_triangles(direct_draw(3u32, 1u32),
             |_, _, _| vertex_output(data[0], ())) in
           shade(target, fragments, |_, _, _, _, _| data[0])",
    ] {
        let tlc = crate::compile_thru_tlc(source).unwrap();
        let program = optimize_for_test(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AllowGenerated,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let mut entries = 0;
        for definition in &program.source.defs {
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                entries += 1;
                let data = entry.declaration.params.iter().find(|param| param.name == "data").unwrap();
                assert!(data
                    .attributes
                    .contains(&Attribute::SizeHint(std::num::NonZeroU32::new(1024).unwrap())));
            }
        }
        assert!(entries > 0);
        let ssa = finish_for_test(program, CodegenTarget::Portable).unwrap();
        lower_ssa_to_wgsl(ssa.clone()).unwrap();
        lower_ssa_to_spirv(ssa).unwrap();
    }
}

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
        "only scan prefixes and sorted output scale with input length\n{}\n{:?}",
        compiled.wgsl, entry.allocations
    );
}

#[test]
fn radix_host_loop_dispatches_three_phases_per_pass() {
    let source = format!(
        "{}\n entry main(xs: []i32) []i32 = radix_sort(32, i32.get_bit, xs)",
        include_str!("../../../pkg/sort/src/radix_sort.wyn")
    );
    let compiled = crate::lower_ssa_to_wgsl_with_program(compile_thru_ssa(&source).unwrap()).unwrap();
    let [region] = compiled.program.interface.dispatch_loops.as_slice() else {
        panic!("expected one host loop");
    };
    assert_eq!(region.body.len(), 3);
}

#[test]
fn expression_projection_retains_host_capture_inputs_without_admission_tables() {
    use super::{from_tlc, fuse, optimize_with_policy, place, schedule, ScalarOptimization};
    use crate::PipelineTopologyPolicy;

    let tlc =
        crate::compile_thru_tlc("entry main(xs:[]i32,bias:i32) []i32=map(|x:i32|x+bias*bias,xs)").unwrap();
    for policy in [ScalarOptimization::Basic, ScalarOptimization::Full] {
        let program = optimize_with_policy(
            prepare_interfaces(
                schedule(
                    place(
                        fuse(from_tlc(&tlc).unwrap()).unwrap(),
                        PipelineTopologyPolicy::AllowGenerated,
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
            .unwrap(),
            policy,
        )
        .unwrap();
        assert!(
            program.stage.captures.is_empty(),
            "scalar optimization must not lower host captures"
        );
        let program = super::prepare_host(program).unwrap().program;
        if policy == ScalarOptimization::Full {
            assert!(
                !program.stage.captures.is_empty(),
                "expanded callback arithmetic must reach host evaluation"
            );
        }
        let mut phase_roots = 0;
        program
            .stage
            .scalars
            .constructor_enodes("HostEvalPhaseRoot", |row| {
                phase_roots += 1;
                assert!(program.stage.selected.roots.contains_key(&(row.children[1], row.children[2])));
            })
            .unwrap();
        assert!(phase_roots > 0);
        let mut requests = 0;
        program.graph.constructor_enodes("ExprImportRoot", |_| requests += 1).unwrap();
        assert!(requests > 0);
        program
            .stage
            .scalars
            .constructor_enodes("ExprImportRoot", |_| {
                panic!("admission requests must remain in the structural graph")
            })
            .unwrap();
    }
}

// Broad policy/backend coverage lives in tests/compiler/cases/scalar_policy_*.wyn.
#[test]
fn scalar_policy_controls_optional_templates() {
    let tlc =
        crate::compile_thru_tlc("def helper(x:i32) i32=x+1 entry main(x:i32) i32=helper(x)*2").unwrap();
    let program = interfaces_for_test(&tlc);
    for policy in [super::ScalarOptimization::Basic, super::ScalarOptimization::Full] {
        let (_, scalars) = super::scalar::run(&program.graph, &program.identities, policy).unwrap();
        let mut templates = 0;
        scalars.constructor_enodes("ExprTemplateBody", |_| templates += 1).unwrap();
        assert_eq!(templates > 0, policy == super::ScalarOptimization::Full);
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
fn compiler_repro_local_array_capacity() {
    shaders(include_str!(
        "../../../testfiles/regressions/local_array_capacity.wyn"
    ));
    shaders(
        "entry check(xs: []f32) [3]f32 =
         reduce(|a: [3]f32, b: [3]f32| map(|(x,y)| x+y, zip(a,b)),
                [0.0,0.0,0.0], map(|x| [x,x+1.0,x+2.0], xs))",
    );
}

#[test]
fn compiler_repro_array_loop() {
    shaders(include_str!("../../../testfiles/regressions/array_loop.wyn"));
}

#[test]
fn compiler_repro_array_reduce() {
    shaders(include_str!("../../../testfiles/regressions/array_reduce.wyn"));
}

#[test]
fn compiler_repro_matrix_map() {
    placement_shaders(
        include_str!("../../../testfiles/regressions/matrix_map.wyn"),
        |_| {},
    );
}

#[test]
fn compiler_repro_matrix_reduce_loop() {
    placement_shaders(
        include_str!("../../../testfiles/regressions/matrix_reduce_loop.wyn"),
        |_| {},
    );
}

#[test]
fn compiler_repro_boolean_record() {
    shaders(include_str!("../../../testfiles/regressions/boolean_record.wyn"));
    shaders(
        "type config = { flags: { enabled: bool }, scale: f32 }
         entry check(config: config, xs: []f32) []f32 =
           map(|x| if config.flags.enabled then x*config.scale else x, xs)",
    );
}

#[test]
fn compiler_repro_boolean_reduce() {
    let module = shaders(include_str!("../../../testfiles/regressions/boolean_reduce.wyn"));
    let mut workgroup_arrays = 0;
    for (_, global) in module.global_variables.iter() {
        if global.space == naga::AddressSpace::WorkGroup {
            let naga::TypeInner::Array { base, .. } = module.types[global.ty].inner else {
                panic!("expected workgroup array");
            };
            assert_eq!(
                module.types[base].inner,
                naga::TypeInner::Scalar(naga::Scalar::U32)
            );
            workgroup_arrays += 1;
        }
    }
    assert!(workgroup_arrays > 0);
    shaders("entry check(xs: []bool) []bool = scan(|a: bool,b: bool| a && b, true, xs)");
    shaders(
        "entry check(xs: []i32) (bool,i32) =
         reduce(|(a,n),(b,m)| (a && b,n+m), (true,0), map(|x| (x>0,1),xs))",
    );
}

#[test]
fn compiler_repro_record_update() {
    shaders(include_str!("../../../testfiles/regressions/record_update.wyn"));
}

#[test]
fn record_update_survives_to_ssa_and_lowers_in_both_backends() {
    let source = include_str!("../../../testfiles/regressions/record_update.wyn");
    let program = compile_thru_ssa(source).unwrap();
    let count = program
        .functions
        .iter()
        .map(|f| &f.body)
        .chain(program.entry_points.iter().map(|e| &e.body))
        .flat_map(|body| body.inner.insts.values())
        .filter(|inst| {
            matches!(&inst.data,
            crate::ssa::types::InstKind::Op {
                tag: crate::op::OpTag::Intrinsic { id, .. }, ..
            } if *id == crate::builtins::catalog().known().record_with)
        })
        .count();
    assert!(count >= 7, "dependent field updates must survive to SSA");
    let binary = lower_ssa_to_spirv(program).unwrap();
    let module = wspirv::dr::load_words(&binary.spirv).unwrap();
    assert!(module.all_inst_iter().any(|inst| inst.class.opcode == wspirv::spirv::Op::CompositeInsert));
    shaders(include_str!(
        "../../../testfiles/regressions/record_update_values.wyn"
    ));
}

#[test]
fn record_update_handles_loop_state_and_record_fields() {
    shaders(
        "type state = { x: i32, pair: { y: i32, z: i32 } }
        entry main(x: i32) state =
        loop s = { x = x, pair = { y = 1, z = 2 } } for i < 4 do
          (s with pair.y = s.pair.y + s.x) with x = s.x + 1",
    );
    shaders(
        "entry main(xs: [3]i32, ys: [3]i32, n: i32) []i32 =
        let original = { count = n, data = xs } in
        let next = original with count = n + 1 in
        let last = next with data = ys in
        map(|x: i32| x + next.count + last.count, last.data)",
    );
}

#[test]
fn record_update_fields_can_size_host_allocations() {
    for update in ["offset = c.offset + 1", "size = c.size + 1"] {
        placement_shaders(
            &format!(
                "type config = {{ size: i32, offset: i32 }}
                 entry check(c: config) []i32 =
                   let next = c with {update} in iota(next.size)"
            ),
            |_| {},
        );
    }
}

#[test]
fn record_update_resource_fields_resolve_before_shader_emission() {
    placement_shaders(
        "type textures = { image: texture2d, filtering: sampler }
         def sample(maps: textures, uv: vec2f32) vec4f32 =
           texture_sample(maps.image, maps.filtering, uv, 0.0)
         entry main(image: texture2d, old_filtering: sampler, filtering: sampler, pixels: []vec2f32) []vec4f32 =
           let maps = { image = image, filtering = old_filtering } in
           let maps = maps with filtering = filtering in
           map(|uv| sample(maps, uv), pixels)",
        |_| {},
    );
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

fn guarded_loops(block: &naga::Block, depth: usize) -> Vec<usize> {
    block
        .iter()
        .flat_map(|statement| match statement {
            Statement::If { accept, reject, .. } => {
                let mut loops = guarded_loops(accept, depth + 1);
                loops.extend(guarded_loops(reject, depth + 1));
                loops
            }
            Statement::Loop { body, .. } => {
                let mut loops = vec![depth];
                loops.extend(guarded_loops(body, depth));
                loops
            }
            Statement::Block(block) => guarded_loops(block, depth),
            _ => Vec::new(),
        })
        .collect()
}

#[test]
fn grouped_materializations_keep_guarded_availability() {
    let source = include_str!("../../../testfiles/regressions/sibling_branch_producer.wyn");
    for (last, loops, unguarded) in [
        ("if skip then 0.0 else s*3.0", 1, 0),
        ("if skip then 0.0 else direct+specular", 1, 0),
        ("s", 1, 1),
    ] {
        let source = source.replace(
            "@[direct,specular,0.0,1.0]",
            &format!("@[direct,specular,{last},1.0]"),
        );
        let module = shaders(&source);
        let fragment =
            &module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Fragment).unwrap().function;
        let depths = guarded_loops(&fragment.body, 0);
        assert_eq!(depths.len(), loops, "{source}");
        assert_eq!(
            depths.iter().filter(|&&depth| depth == 0).count(),
            unguarded,
            "{source}"
        );
    }
}

#[test]
fn shared_branch_producers_keep_one_guarded_loop() {
    let source = include_str!("../../../testfiles/regressions/shared_branch_producer.wyn");
    for (consumer, loops) in [
        ("if y > 1.0 then color else color * 0.8", 1),
        (
            "if y > 1.0 then (if y > 2.0 then color else @[0.0,0.0,0.0,0.0])
             else (if y < 0.0 then color * 0.8 else @[0.0,0.0,0.0,0.0])",
            1,
        ),
    ] {
        let source = source.replace("if y > 1.0 then color else color * 0.8", consumer);
        let module = shaders(&source);
        let fragment =
            &module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Fragment).unwrap().function;
        let depths = guarded_loops(&fragment.body, 0);
        assert_eq!(depths.len(), loops, "{source}");
        assert!(
            depths.iter().all(|&depth| depth > 0),
            "producer escaped its x guard: {source}"
        );
    }
}

#[test]
fn enclosing_producer_is_shared_across_nested_consumer_guards() {
    let source = include_str!("../../../testfiles/regressions/shared_branch_producer.wyn").replace(
        "if y > 1.0 then color else color * 0.8",
        "if y > 1.0 then color else if y < 0.0 then @[0.0,0.0,0.0,0.0] else color * 0.8",
    );
    let module = shaders(&source);
    let fragment =
        &module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Fragment).unwrap().function;
    let depths = guarded_loops(&fragment.body, 0);
    assert_eq!(depths.len(), 1);
    assert!(depths[0] > 0, "producer must retain its own guard");
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
fn filters_of_tuples_with_boolean_fields() {
    shaders("entry kept(xs:[]i32) [](i32,bool)=filter(|(x,b)|b,map(|x|(x,x>0),xs))");
}

#[test]
fn resource_records_passed_to_helpers_reach_both_backends() {
    placement_shaders(
        "type textures = { image: texture2d, filtering: sampler }
         def sample(maps: textures, uv: vec2f32) vec4f32 =
           texture_sample(maps.image, maps.filtering, uv, 0.0)
         entry main(image: texture2d, filtering: sampler, pixels: []vec2f32) []vec4f32 =
           let maps = { image = image, filtering = filtering } in
           map(|uv| sample(maps, uv), pixels)",
        |_| {},
    );
}

#[test]
fn conditional_tuple_and_zipped_maps_reach_both_backends() {
    for source in [
        "entry main(xs:[4]i32,ys:[4]i32,flag:bool) [4](i32,i32) =
         if flag then map(|x|(x,x+1),xs) else map(|y|(y,y+2),ys)",
        "entry main(xs:[4]i32,ys:[4]i32,flag:bool) [4]i32 =
         if flag then map(|(x,y)|x+y,zip(xs,ys)) else map(|(y,x)|y-x,zip(ys,xs))",
    ] {
        shaders(source);
    }
}

#[test]
fn compaction_retains_producer_elements_across_the_scan() {
    for source in [
        "entry kept(xs:[]i32) []i32=filter(|x|x>0,map(|x|x*x+17,xs))",
        "entry kept(xs:[]i32) []{x:i32,y:i32}=filter(|p|p.x>0,map(|x|{x=x,y=x*3+1},xs))",
        "entry kept(xs:[]i32) []{x:i32,keep:bool}=filter(|p|p.keep,map(|x|{x=x,keep=x>0},xs))",
    ] {
        let module = shaders(source);
        let input = module
            .global_variables
            .iter()
            .find(|(_, v)| v.binding.as_ref().is_some_and(|b| b.group == 0 && b.binding == 0))
            .unwrap()
            .0;
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(
            storage_loads(&module.entry_points[0].function, input),
            1,
            "the predicate and output must share one input load: {source}"
        );
    }
}

#[test]
fn scans_and_reductions_use_one_shared_bank_per_phase() {
    let reduce = shaders("entry total(xs:[]i32) i32=reduce((+),0,xs)");
    let scan = shaders("entry prefixes(xs:[]i32) []i32=scan((+),0,xs)");
    let shared = |module: &naga::Module| {
        module.global_variables.iter().filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup).count()
    };
    assert!(shared(&reduce) > 0);
    assert_eq!(
        shared(&reduce),
        shared(&scan),
        "scans reuse their bank after a read barrier"
    );
}
#[test]
fn nested_reduction_stays_inside_the_map_invocation() {
    shaders("entry sums(xs:[]i32) []i32=map(|x:i32|reduce(|a:i32,b:i32|a+b,0,map(|y:i32|y+x,iota(5))),xs)");
}

#[test]
fn local_map_captures_are_computed_before_the_element_loop() {
    // Keep these traversals above the small-loop expansion limit.
    let module = shaders(
        "entry main(xs:[][32]i32) [][32]i32 =
          map(|row|let prefixes=scan((+),0i32,row) in
            let offset=reduce((+),0i32,row)+1i32 in
            map(|j|prefixes[j%32]+offset,iota(32)),xs)",
    );
    let mut loops = 0;
    for (_, function) in module.functions.iter() {
        visit_loop_depths(&function.body, 0, &mut |statement, depth| {
            if matches!(statement, Statement::Loop { .. }) {
                loops += 1;
                assert_eq!(depth, 0, "captured scan must not repeat for every output element");
            }
        });
    }
    for entry in &module.entry_points {
        visit_loop_depths(&entry.function.body, 0, &mut |statement, depth| {
            if matches!(statement, Statement::Loop { .. }) && depth > 0 {
                loops += 1;
                assert_eq!(
                    depth, 1,
                    "inlined captured scan must not repeat for every output element"
                );
            }
        });
    }
    assert!(loops >= 2, "fixture must retain the scan and output traversal");
}

#[test]
fn fused_local_slice_reductions_do_not_allocate_unused_producer_arrays() {
    shaders(
        "entry bounds(xs:[][8]f32) []f32 = map(|row|
      let values=map(|x|x+1.0,row[1..i32(row[0])]) in
      reduce(|a,b|min(a,b),f32.highest,values)+reduce(|a,b|max(a,b),f32.lowest,values),xs)",
    );
}

#[test]
fn owned_scatter_inside_map_keeps_queries_parallel() {
    for source in [
        "entry main(xs:[]i32) [][4]i32 = map(|x|scatter(replicate(4,0i32),[x%4],[x]),xs)",
        "entry main(xs:[]i32) [][4]i32 = map(|x|loop a=replicate(4,0i32) for i<3 do scatter(a,[i],[x+i]),xs)",
    ] {
        let module = shaders(source);
        assert!(
            module.entry_points.iter().any(|entry| entry.workgroup_size[0] > 1),
            "independent owned scatter must not serialize the outer map: {source}"
        );
    }
}

#[test]
fn batched_traversal_has_shared_state_and_no_partial_lane_barriers() {
    fn inspect(block: &naga::Block, conditional: bool) -> usize {
        let mut barriers = 0;
        for statement in block.iter() {
            match statement {
                Statement::ControlBarrier(_) => {
                    assert!(!conditional, "traversal barrier inside a partial-lane branch");
                    barriers += 1;
                }
                Statement::If { accept, reject, .. } => {
                    barriers += inspect(accept, true) + inspect(reject, true);
                }
                Statement::Loop { body, continuing, .. } => {
                    barriers += inspect(body, conditional) + inspect(continuing, conditional);
                }
                Statement::Block(block) => barriers += inspect(block, conditional),
                _ => {}
            }
        }
        barriers
    }
    let module = shaders(include_str!(
        "../../../fluid-simulation/test/batched_traversal.wyn"
    ));
    assert!(module.global_variables.iter().any(|(_, v)| v.space == naga::AddressSpace::WorkGroup));
    let mut barriers = 0;
    for function in
        module.functions.iter().map(|(_, f)| f).chain(module.entry_points.iter().map(|e| &e.function))
    {
        barriers += inspect(&function.body, false);
    }
    assert!(barriers > 2, "traversal must execute cooperative collectives");
    assert_eq!(
        module.entry_points.len(),
        1,
        "traversal should stay in one kernel"
    );
    assert!(
        module.entry_points[0].workgroup_size[0] > 1,
        "queries need a parallel workgroup"
    );
}

#[test]
fn ranked_buckets_preserve_nested_storage_and_atomic_counters() {
    shaders("entry bins(dest:*[2][2]i32) ([2][2]i32,[2]u32,u32)=bucket_scatter_2d(dest,[[(-1,9),(0,10),(0,11)],[(0,12),(1,20),(2,9)]])");
}

#[test]
#[ignore = "nested scheduling follow-up: ranked buckets inside map fail with an unbound TLC symbol during source import"]
fn ranked_bucket_scatter_inside_map_reaches_both_backends() {
    shaders(
        "entry ranked_local(bias: []i32) [2][24][1]i32 =
          map(|b|
            let items: [2][3][4](i32,i32) =
              map(|i|map(|j|map(|k|
                ((i*3+j)*4+k,100*i+10*j+k+b),iota(4)),iota(3)),iota(2))
            in let (values,_,_)=bucket_scatter_3d(replicate(24,[0]),items) in values,
            bias[0..2])",
    );
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

// Target-independent preparation is shared; each backend still lowers and validates separately.
fn interfaces_for_test(
    tlc: &crate::tlc::stage::OwnershipApplied,
) -> super::Program<'_, super::InterfacesPrepared> {
    let program = super::from_tlc(tlc).unwrap();
    let program = super::fuse(program).unwrap();
    let program = super::place(program, crate::PipelineTopologyPolicy::AllowGenerated).unwrap();
    let program = super::schedule(program).unwrap();
    prepare_interfaces(program).unwrap()
}

// Inspect both emitted shader formats under the CLI's default and -O policies.
fn placement_shaders(source: &str, mut check: impl FnMut(&naga::Module)) {
    let tlc = crate::compile_thru_tlc(source).unwrap();
    for policy in [super::ScalarOptimization::Basic, super::ScalarOptimization::Full] {
        let program = super::optimize_with_policy(interfaces_for_test(&tlc), policy).unwrap();
        let prepared = super::prepare_host(program).unwrap();
        for target in [crate::CodegenTarget::Spirv, crate::CodegenTarget::Wgsl] {
            let program = super::to_ssa::lower(&prepared.program, target).unwrap();
            let module = match target {
                crate::CodegenTarget::Spirv => {
                    let binary = lower_ssa_to_spirv(program).unwrap();
                    naga::front::spv::Frontend::new(
                        binary.spirv.iter().copied(),
                        &naga::front::spv::Options::default(),
                    )
                    .parse()
                    .unwrap()
                }
                crate::CodegenTarget::Wgsl => {
                    naga::front::wgsl::parse_str(&lower_ssa_to_wgsl(program).unwrap()).unwrap()
                }
                _ => unreachable!(),
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            check(&module);
        }
    }
}

fn visit_loop_depths(block: &naga::Block, depth: usize, visit: &mut impl FnMut(&Statement, usize)) {
    for statement in block.iter() {
        visit(statement, depth);
        match statement {
            Statement::Loop { body, continuing, .. } => {
                visit_loop_depths(body, depth + 1, visit);
                visit_loop_depths(continuing, depth + 1, visit);
            }
            Statement::If { accept, reject, .. } => {
                visit_loop_depths(accept, depth, visit);
                visit_loop_depths(reject, depth, visit);
            }
            Statement::Block(block) => visit_loop_depths(block, depth, visit),
            Statement::Switch { cases, .. } => {
                for case in cases {
                    visit_loop_depths(&case.body, depth, visit);
                }
            }
            _ => {}
        }
    }
}

#[test]
fn fluid_leaf_bounds_are_computed_outside_the_bfs() {
    stacker::grow(32 * 1024 * 1024, || {
        // Use the simulator's implementation, including its queue sizes and guards.
        let neighbors = include_str!("../../../fluid-simulation/packages/spatial/src/neighbors.wyn")
            .replace(
                "module Tree = import \"octree\"",
                &format!(
                    "module Tree = {{ {} }}",
                    include_str!("../../../fluid-simulation/packages/spatial/src/octree.wyn")
                ),
            )
            .replace(
                "module Hilbert = import \"hilbert\"",
                &format!(
                    "module Hilbert = {{ {} }}",
                    include_str!("../../../fluid-simulation/packages/spatial/src/hilbert.wyn")
                ),
            );
        let source = format!(
            "{neighbors}
        entry neighbors(positions:[]vec4f32, corners:[]u32, offsets:[]i32) [258]i32 =
            leaf_neighbors(0,positions,corners,offsets,1.7,16.0)"
        );
        placement_shaders(&source, |module| {
            let mut bounds_depths = Vec::new();
            let mut queues = 0;
            for function in module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|e| &e.function))
            {
                let queue_variables: Vec<_> = function
                    .local_variables
                    .iter()
                    .filter_map(|(id, v)| {
                        matches!(module.types[v.ty].inner,
                        naga::TypeInner::Array { base, .. } if matches!(module.types[base].inner,
                            naga::TypeInner::Vector { size: naga::VectorSize::Bi, scalar }
                                if scalar.kind == naga::ScalarKind::Uint))
                        .then_some(id)
                    })
                    .collect();
                queues += queue_variables.len();
                let whole_queue = |pointer| {
                    matches!(function.expressions[pointer],
                    Expression::LocalVariable(v) if queue_variables.contains(&v))
                };
                assert!(!function
                    .expressions
                    .iter()
                    .any(|(_, e)| matches!(e, Expression::Load { pointer } if whole_queue(*pointer))));
                visit_loop_depths(&function.body, 0, &mut |statement, depth| {
                    if let Statement::Store { pointer, .. } = statement {
                        assert!(!whole_queue(*pointer), "BFS must update queue elements in place");
                    }
                    if let Statement::Emit(range) = statement {
                        for value in range.clone() {
                            if matches!(
                                function.expressions[value],
                                Expression::Math {
                                    fun: naga::MathFunction::Max,
                                    ..
                                }
                            ) {
                                bounds_depths.push(depth);
                            }
                        }
                    }
                });
            }
            // Max updates the upper bound inside the bounds loop, never in a
            // second enclosing loop (the BFS). Also reject duplicated bounds scans.
            assert_eq!(bounds_depths, vec![1]);
            assert_eq!(queues, 1);
        });
    });
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
fn runtime_array_metadata_shares_storage_and_fits_binding_limits() {
    for (source, limit, inputs) in [
        (
            include_str!("../../../testfiles/regressions/runtime_array_three_outputs.wyn"),
            8,
            1,
        ),
        (
            include_str!("../../../testfiles/regressions/runtime_array_four_inputs.wyn"),
            6,
            4,
        ),
    ] {
        let output = crate::lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap();
        let [crate::host::Pipeline::Compute(pipeline)] = output.program.interface.pipelines.as_slice()
        else {
            panic!("compute pipeline");
        };
        for stage in &pipeline.stages {
            let storage_bindings = pipeline
                .bindings
                .iter()
                .enumerate()
                .filter(|(index, binding)| {
                    matches!(binding, crate::host::Binding::StorageBuffer { .. })
                        && stage.uses.access(*index).is_some()
                })
                .count();
            assert!(
                storage_bindings <= limit,
                "binding limit exceeded: {storage_bindings} > {limit}"
            );
        }
        let lengths: Vec<_> = output
            .program
            .interface
            .scalar_tasks
            .iter()
            .filter(|task| matches!(task.value, crate::host::ScalarExpr::BufferLength { .. }))
            .collect();
        assert!(
            lengths.len() >= inputs,
            "each dynamic input needs its logical length"
        );
        for task in &lengths {
            assert_eq!(task.destination, lengths[0].destination);
            assert_eq!(task.offset % 4, 0);
        }
    }
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

#[test]
fn branch_loop_state_keeps_inferred_array_layout() {
    shaders(include_str!(
        "../../../fluid-simulation/test/repro_branch_array_layout.wyn"
    ));
}

#[test]
fn cooperative_loop_parallel_assignment_preserves_array_sources() {
    shaders(include_str!(
        "../../../fluid-simulation/test/repro_shared_loop_swap.wyn"
    ));
}

#[test]
fn module_counted_loop_counters_have_distinct_types() {
    shaders(
        "module S = { def run(x:i32) i32 = loop x=x for i<8 do x+i }
        entry main() (i32,i32) = (S.run(1),S.run(2))",
    );
}

#[test]
fn narrowing_loop_bounds_device_sized_allocations() {
    shaders(include_str!(
        "../../../testfiles/regressions/narrowing_loop_capacity.wyn"
    ));
}

#[test]
fn narrowing_loop_capacity_requires_bounded_backedge() {
    let source = include_str!("../../../testfiles/regressions/narrowing_loop_capacity.wyn")
        .replace("(mid+1,hi)", "(hi+1,hi)");
    assert!(compile_thru_ssa(&source).is_err());
}

#[test]
fn inferred_loop_result_arrays_retain_logical_extents() {
    let fixture = include_str!("../../../fluid-simulation/test/parity_tree.wyn")
        .replace("module Tree = import \"../packages/spatial/src/octree\"", "")
        .replace("Tree.", "");
    shaders(&format!(
        "{}\n{fixture}",
        include_str!("../../../fluid-simulation/packages/spatial/src/octree.wyn")
    ));
}

#[test]
fn public_scratch_allocators_reach_both_backends() {
    for source in [
        "entry main(xs:[]i32) []i32 = scatter(duplicate_scratch(xs),iota(length(xs)),xs)",
        "entry main(xs:[]i32) []i32 = scatter(scratch(length(xs)),iota(length(xs)),xs)",
        "entry main(xs:[4]vec2u32) [4]vec2u32 = scatter(duplicate_scratch(xs),[3,2,1,0],xs)",
    ] {
        shaders(source);
    }
}
