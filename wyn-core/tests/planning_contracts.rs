//! Planning contracts at the published shader/host boundary, without planner tables.
#[path = "support/pipeline.rs"]
mod pipeline;

use std::collections::BTreeSet;
use wyn_core::host::{self, Binding, BufferLen, BufferUsage, DispatchSize, Expr, Pipeline};
use wyn_core::{egglog, lower_ssa_to_wgsl_with_program, CodegenTarget, PipelineTopologyPolicy};

#[test]
fn scheduling_costs_control_collective_launches_after_placement() {
    for source in [
        "entry main(xs:[8192]i32) i32=reduce((+),0,xs)",
        "entry main(xs:[8192]i32) [8192]i32=scan((+),0,xs)",
    ] {
        let source = pipeline::source(source);
        let mut launches = Vec::new();
        for resident_workgroups in [1, 256] {
            let program = egglog::fuse(egglog::from_tlc(&source).unwrap()).unwrap();
            // Placement is identical; only the scheduling model changes.
            let program = egglog::place(program, PipelineTopologyPolicy::AllowGenerated).unwrap();
            let program = egglog::schedule_with_cost_model(
                program,
                false,
                egglog::cost::CostModel {
                    resident_workgroups,
                    ..Default::default()
                },
            )
            .unwrap();
            let program = egglog::prepare_interfaces(program).unwrap();
            let program = egglog::optimize_with_policy(program, egglog::ScalarOptimization::Basic).unwrap();
            let module =
                egglog::to_ssa(egglog::prepare_host(program).unwrap(), CodegenTarget::Portable).unwrap();
            wyn_core::lower_ssa_to_spirv(module.clone()).unwrap();
            let output = lower_ssa_to_wgsl_with_program(module).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&naga::front::wgsl::parse_str(&output.wgsl).unwrap())
            .unwrap();
            let [Pipeline::Compute(compute)] = output.program.interface.pipelines.as_slice() else {
                panic!("one compute pipeline");
            };
            let DispatchSize::Computed { groups, .. } = &compute.stages[0].dispatch_size else {
                panic!("collective chunk launch");
            };
            launches.push(constant(&groups[0]).unwrap());
            assert!(depends_on(&output.program.interface.frame_graph, 1, 0));
        }
        assert_eq!(launches[0], 1);
        assert!(
            launches[1] > launches[0],
            "schedule calibration must affect launches: {launches:?}"
        );
    }
}

#[test]
fn parameter_blocks_preserve_aggregate_padding_and_member_alignment() {
    for (source, offsets) in [
        ("type params={direction:vec3f32}
          entry main(values:[]f32,frame:params,bias:f32) []f32=map(|v|v*frame.direction.x+bias,values)", vec![0,16]),
        ("type params={direction:vec4f32}
          entry main(values:[]f32,frame:params,bias:f32) []f32=map(|v|v*frame.direction.x+bias,values)", vec![0,16]),
        ("entry main(values:[]f32,head:f32,direction:vec4f32,bias:f32) []f32=map(|v|head+v*direction.x+bias,values)", vec![0,16,32]),
        ("entry main(values:[]f32,direction:vec3f32,bias:f32) []f32=map(|v|v*direction.x+bias,values)", vec![0,12]),
    ] {
        let source = pipeline::source(source);
        let ssa = pipeline::lower(&source, CodegenTarget::Portable);
        let spirv = wyn_core::lower_ssa_to_spirv(ssa.clone()).unwrap();
        let module = naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &Default::default()).parse().unwrap();
        let (_, block) = module.global_variables.iter().find(|(_, v)| v.space == naga::AddressSpace::PushConstant).unwrap();
        let naga::TypeInner::Struct { members, .. } = &module.types[block.ty].inner else { panic!("parameter block"); };
        assert_eq!(members.iter().map(|m| m.offset).collect::<Vec<_>>(), offsets);
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        let wgsl = lower_ssa_to_wgsl_with_program(ssa).unwrap();
        let module = naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        let [Pipeline::Compute(compute)] = wgsl.program.interface.pipelines.as_slice() else { panic!("compute pipeline"); };
        let blocks = compute.bindings.iter().filter_map(|binding| match binding {
            Binding::StorageBuffer { members, .. } if !members.is_empty() => Some(members),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].iter().map(|m| m.offset).collect::<Vec<_>>(), offsets);
    }
}

#[test]
fn published_collective_dependencies_survive_both_backends() {
    let source = pipeline::source("entry main(xs:[]i32) i32=reduce((+),0,xs)");
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    let programs = [
        lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap().program,
        wyn_core::lower_ssa_to_spirv(ssa).unwrap().program,
    ];
    for program in programs {
        let graph = &program.interface.frame_graph;
        assert_eq!(graph.passes.len(), 2, "chunk and combine dispatches");
        assert!(depends_on(graph, 1, 0), "combine must wait for chunk results");
        assert_eq!(graph.topological_order().unwrap(), [0, 1]);
    }
}

#[test]
fn texture_reads_are_ordered_between_draws_that_overwrite_the_target() {
    let source = pipeline::source(include_str!(
        "../../tests/compiler/library/graphics_aliases_preserve_host_buffer_identity.wyn"
    ));
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap();
    let program = output.program;
    let [sampled, target] = program.entries[0].results.as_slice() else {
        panic!("sampled array and render target");
    };
    let graph = &program.interface.frame_graph;
    let draw = |operation| {
        graph.passes.iter().position(|pass|
        matches!(&program.interface.pipelines[pass.pipeline_index], Pipeline::Graphics(g) if g.source_operation == Some(operation))).unwrap()
    };
    let reader =
        graph.passes.iter().position(|pass| pass.writes.iter().any(|a| a.resource == sampled.0)).unwrap();
    assert!(graph.passes[reader].reads.iter().any(|a| a.resource == target.0));
    assert!(
        depends_on(graph, reader, draw(0)),
        "read follows the first draw: {:?}",
        graph.passes
    );
    assert!(
        depends_on(graph, draw(1), reader),
        "overwrite follows the read: {:?}",
        graph.passes
    );
}

#[test]
fn empty_array_results_publish_their_actual_storage_stride() {
    let artifact = compile(
        "entry empty() [0]i32=map(|x|x+1,iota(0))
        entry rows() [3][0]i32=map(|x|if x>0 then map(|y|1/x+y,iota(0)) else map(|y|x-y,iota(0)),[0,1,-1])
        entry fields() [2]([0]i32,i32)=map(|x|(iota(0),x),[1,2])",
    );
    assert_eq!(artifact.interface.source_results.len(), 3);
    for result in &artifact.interface.source_results {
        let host::ResultLayout::Array { stride, length, .. } = result.layout else {
            panic!("array result");
        };
        assert_eq!(
            stride,
            if result.entry == "fields" { 8 } else { 4 },
            "{}: host stride",
            result.entry
        );
        assert_eq!(
            length,
            Some(match result.entry.as_str() {
                "empty" => 0,
                "rows" => 3,
                "fields" => 2,
                _ => unreachable!(),
            })
        );
        let (_, global) = artifact
            .shader
            .global_variables
            .iter()
            .find(|(_, v)| {
                v.binding.as_ref().is_some_and(|b| b.group == result.set && b.binding == result.binding)
            })
            .unwrap();
        let array = match &artifact.shader.types[global.ty].inner {
            naga::TypeInner::Struct { members, .. } => members[0].ty,
            _ => global.ty,
        };
        let naga::TypeInner::Array {
            stride: shader_stride,
            ..
        } = artifact.shader.types[array].inner
        else {
            panic!("storage array");
        };
        assert_eq!(stride, shader_stride, "{}: host and shader agree", result.entry);
    }
}

struct Artifact {
    interface: host::ModuleInterface,
    shader: naga::Module,
}

fn compile(source: &str) -> Artifact {
    let tlc = pipeline::source(source);
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&tlc, CodegenTarget::Wgsl)).unwrap();
    let shader = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&shader)
    .unwrap();
    Artifact {
        interface: output.program.interface,
        shader,
    }
}

impl Artifact {
    fn compute(&self) -> &host::ComputePipeline {
        let [Pipeline::Compute(compute)] = self.interface.pipelines.as_slice() else {
            panic!("one authored compute pipeline");
        };
        compute
    }

    fn result(&self, slot: usize) -> usize {
        let result = self
            .interface
            .source_results
            .iter()
            .find(|r| r.entry == "main" && r.result == slot)
            .expect("published source result");
        self.compute()
            .bindings
            .iter()
            .position(|b| b.slot() == Some((result.set, result.binding)))
            .expect("result storage binding")
    }

    fn stages(&self, expected: usize) {
        let accesses = self
            .compute()
            .stages
            .iter()
            .map(|s| (&s.reads, &s.writes, &s.dependencies))
            .collect::<Vec<_>>();
        assert_eq!(
            accesses.len(),
            expected,
            "published dispatch reads/writes/dependencies: {accesses:?}"
        );
    }

    fn allocated(&self, bytes: u64, except: Option<usize>) -> usize {
        self.compute()
            .bindings
            .iter()
            .enumerate()
            .filter(|(index, binding)| {
                Some(*index) != except
                    && matches!(binding,
                        Binding::StorageBuffer { usage, length: Some(length), .. }
                            if *usage != BufferUsage::Input && capacity(length) == Some(bytes)
                    )
            })
            .count()
    }
}

fn constant(expression: &Expr) -> Option<i64> {
    Some(match expression {
        Expr::Integer(n) => *n,
        Expr::Scalar(host::ScalarExpr::I32(n)) => i64::from(*n),
        Expr::Scalar(host::ScalarExpr::U32(n)) => i64::from(*n),
        Expr::Add(a, b) => constant(a)? + constant(b)?,
        Expr::Subtract(a, b) => constant(a)? - constant(b)?,
        Expr::Multiply(a, b) => constant(a)? * constant(b)?,
        Expr::Floor(a, b) => constant(a)?.div_euclid(constant(b)?),
        Expr::Ceiling(a, b) => -(-constant(a)?).div_euclid(constant(b)?),
        Expr::Min(a, b) => constant(a)?.min(constant(b)?),
        Expr::Max(a, b) => constant(a)?.max(constant(b)?),
        _ => return None,
    })
}

fn capacity(length: &BufferLen) -> Option<u64> {
    match length {
        BufferLen::Computed { bytes } => constant(bytes)?.try_into().ok(),
        _ => length.resolve_bytes(|_, _| None),
    }
}

#[test]
fn fused_filter_postmap_retains_the_traversed_input_domain() {
    let artifact = compile("entry main(xs:[]i32) []i32=map(|x|x+3,filter(|x|x>0,xs))");
    artifact.stages(1);
    let compute = artifact.compute();
    let input = compute
        .bindings
        .iter()
        .find_map(|binding| match binding {
            Binding::StorageBuffer {
                name, set, binding, ..
            } if name == "xs" => Some(host::ScalarSource::Binding {
                set: *set,
                binding: *binding,
            }),
            _ => None,
        })
        .unwrap();
    let DispatchSize::Computed { elements, .. } = &compute.stages[0].dispatch_size else {
        panic!("runtime input domain");
    };
    let mut uses_input_length = false;
    elements.clone().reads_mut(&mut |source, _| uses_input_length |= *source == input);
    assert!(
        uses_input_length,
        "filter traverses the original input, including rejected lanes"
    );
    assert!(compute.stages[0].writes.contains(&artifact.result(0)));
    assert_ne!(
        artifact.interface.source_results[0].shape.expressions(),
        &[elements.clone()],
        "the output's packed length is distinct from the traversal bound"
    );
}

#[test]
fn fused_scan_and_filter_count_have_distinct_accumulator_storage() {
    let artifact = compile(
        "entry main(xs:[]i32) ([]i32,i32,i32)=let ys=filter(|x|x>0,xs) in
        (scan((+),0,xs),length(ys),reduce((+),0,ys))",
    );
    let results = (0..3).map(|i| artifact.result(i)).collect::<BTreeSet<_>>();
    assert_eq!(results.len(), 3, "prefixes, count, and total must not alias");
    let stages = &artifact.compute().stages;
    let partials: Vec<_> = stages[0].writes.iter().filter(|b| stages[1].reads.contains(b)).collect();
    assert_eq!(
        partials.len(),
        3,
        "all three accumulators need independent partial state; stages: {stages:?}"
    );
    assert!(stages[1].writes.contains(&artifact.result(1)));
    assert!(stages[1].writes.contains(&artifact.result(2)));
    assert!(stages[2].writes.contains(&artifact.result(0)));
}

#[test]
fn observing_an_intermediate_filter_length_preserves_its_cardinality() {
    let artifact = compile(
        "entry main(xs:[]i32) (i32,i32)=let a=filter(|x|x>0,xs)
        let b=filter(|x|x<20,map(|x|x+1,a)) in (length(a),length(b))",
    );
    assert_eq!(artifact.interface.source_results.len(), 2);
    assert_ne!(
        artifact.result(0),
        artifact.result(1),
        "each observed cardinality needs its own count"
    );
    for result in &artifact.interface.source_results {
        assert_eq!(result.layout, host::ResultLayout::Scalar(host::ResultScalar::I32));
        assert!(artifact
            .compute()
            .stages
            .iter()
            .any(|stage| stage.writes.contains(&artifact.result(result.result))));
    }
}

#[test]
fn guarded_streams_feed_updates_and_prefixes_without_a_compaction_buffer() {
    // Both fixed and runtime domains preserve the callback guard without packing an intermediate.
    for (source, stages) in [
        (
            "entry main(xs:[17]i32) [3]i32=let ys=filter(|x|x!=0,xs) let dest=[-1,-1,-1]
          in scatter(dest,map(|x|x-1,ys),map(|x|100/x,ys))",
            1,
        ),
        (
            "entry main(xs:[17]i32) []i32=map(|x|x*2,scan((+),0,map(|x|100/x,filter(|x|x!=0,xs))))",
            1,
        ),
    ]
    .into_iter()
    .flat_map(|(source, stages)| {
        [
            (source.to_owned(), stages),
            (source.replace("[17]", "[]"), stages),
        ]
    }) {
        let artifact = compile(&source);
        artifact.stages(stages);
        let output = artifact.result(0);
        let payload_buffers: Vec<_> = artifact
            .compute()
            .bindings
            .iter()
            .enumerate()
            .filter(|(i, b)| {
                *i != output
                    && matches!(b, Binding::StorageBuffer { usage, length: Some(length), .. }
                if *usage != BufferUsage::Input && capacity(length) == Some(17*4))
            })
            .collect();
        assert!(
            payload_buffers.is_empty(),
            "filtered values must stream directly into the consumer"
        );
        let mut divisions = 0;
        for function in artifact
            .shader
            .entry_points
            .iter()
            .map(|entry| &entry.function)
            .chain(artifact.shader.functions.iter().map(|(_, function)| function))
        {
            visit(&function.body, 0, &mut |statement, guards| {
                if let naga::Statement::Emit(range) = statement {
                    for handle in range.clone() {
                        if let naga::Expression::Binary {
                            op: naga::BinaryOperator::Divide,
                            left,
                            ..
                        } = function.expressions[handle]
                        {
                            if matches!(
                                function.expressions[left],
                                naga::Expression::Literal(naga::Literal::I32(100))
                            ) {
                                divisions += 1;
                                assert!(guards > 0, "the callback division must remain conditional");
                            }
                        }
                    }
                }
            });
        }
        assert!(
            divisions > 0,
            "inspect the guarded callback, not just dispatch metadata"
        );
    }
}

fn visit(block: &naga::Block, guards: usize, f: &mut impl FnMut(&naga::Statement, usize)) {
    for statement in block.iter() {
        f(statement, guards);
        match statement {
            naga::Statement::If { accept, reject, .. } => {
                visit(accept, guards + 1, f);
                visit(reject, guards + 1, f);
            }
            naga::Statement::Block(block) => visit(block, guards, f),
            naga::Statement::Loop { body, continuing, .. } => {
                visit(body, guards, f);
                visit(continuing, guards, f);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    visit(&case.body, guards + 1, f);
                }
            }
            _ => {}
        }
    }
}

#[test]
fn outputs_that_capture_a_total_wait_for_the_collective_barrier() {
    for (source, expected) in [
        ("entry main(xs:[257]i32) [257]i32=let prefix=scan((+),0,xs) let total=reduce((+),0,xs) in map(|x|x+total,prefix)", 3),
        ("entry main(xs:[257]i32) [257]i32=let total=reduce((+),0,xs) in map(|x|x+total,xs)", 3),
        ("entry main(xs:[257]i32) [3]i32=let total=reduce((+),0,xs) let dest=[0,0,0] in reduce_by_index(dest,(+),0,map(|x|x%3,xs),map(|x|x+total,xs))", 4),
    ] {
        let artifact = compile(source);
        let graph = &artifact.interface.frame_graph;
        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), artifact.compute().stages.len());
        let stages = &artifact.compute().stages;
        let total_buffers: Vec<_> = stages[1].writes.iter().filter(|b| stages[2].reads.contains(b)).collect();
        assert!(!total_buffers.is_empty(), "consumer reads the completed combine output");
        for buffer in total_buffers {
            assert!(!stages[0].reads.contains(buffer), "chunk evaluation cannot read an unfinished total");
        }
        let pass = |stage| graph.passes.iter().position(|p| p.pipeline_index == 0 && p.stage_index == stage).unwrap();
        assert!(depends_on(graph, pass(2), pass(1)), "consumer waits for the combine dispatch: {:?}", graph.passes);
        artifact.stages(expected);
    }
}

fn depends_on(graph: &host::FrameGraph, consumer: usize, producer: usize) -> bool {
    consumer == producer
        || graph.passes[consumer].depends_on.iter().any(|&parent| depends_on(graph, parent, producer))
}

#[test]
fn scalar_epilogues_share_the_completed_collective_dispatch() {
    for (source, stages) in [
        ("entry main(xs:[]i32) i32=reduce((+),0,xs)+1", 2),
        (
            "entry main(xs:[]i32) ([]i32,i32)=let ys=filter(|x|x>0,xs) in (ys,length(ys)*2)",
            1,
        ),
    ] {
        compile(source).stages(stages);
    }
}

#[test]
fn output_coordinate_maps_accept_permutations_and_reject_overlapping_digits() {
    for (row, column, bound, bijective) in [
        ("a/2", "a%2", 8, true),
        ("a%8", "a/8", 8, true),
        ("4*(a/8)+a%4", "(a/4)%2", 8, true),
        ("4*(a%2)+2*((a/4)%2)+a/8", "(a/2)%2", 8, true),
        ("a%8", "a%2", 8, false),
        ("a%8", "a/8", 4, false),
        ("(a/2)%4+4*(a/3)", "a%2", 8, false),
    ] {
        let source = format!(
            "entry main() [16]i32=let rows=map(|i|[10*i,10*i+1],iota(8)) in
            map(|a|let r={row} in if r<{bound} then rows[r][{column}] else -1,iota(16))"
        );
        let artifact = compile(&source);
        assert_eq!(
            artifact.allocated(8 * 2 * 4, Some(artifact.result(0))) > 0,
            !bijective,
            "only a complete permutation can eliminate the intermediate rows: {source}"
        );
        artifact.stages(if bijective { 1 } else { 2 });
    }
}

#[test]
fn tiled_outputs_require_complete_tiles_and_preserve_other_consumers() {
    for (count, complete) in [(8, true), (7, false)] {
        for reduced in [false, true] {
            for retained in [false, true] {
                let metadata =
                    if reduced { "reduce((+),0,map(|r|r[2],rows))" } else { "map(|r|r[2],rows)" };
                let metadata_type = if reduced { "i32".into() } else { format!("[{count}]i32") };
                let row_type = if retained { format!(",[{count}][3]i32") } else { String::new() };
                let row_result = if retained { ",rows" } else { "" };
                let source = format!("def row(i:i32) [3]i32 = [i,i+1,i+2]
                    entry main() ([16]i32,{metadata_type}{row_type}) = let rows=map(row,iota({count})) in
                    (map(|a|let i=(a/8)*4+a%4 in if i<length(rows) then rows[i][(a%8)/4] else -1,iota(16)),{metadata}{row_result})");
                let artifact = compile(&source);
                artifact.stages(if complete { 1 } else { 2 });
                assert_eq!(
                    artifact.allocated(count * 3 * 4, None) > 0,
                    retained || !complete,
                    "retained rows and incomplete tiles need backing storage: {source}"
                );
                assert_eq!(
                    artifact.interface.source_results.len(),
                    if retained { 3 } else { 2 }
                );
            }
        }
    }
}

#[test]
fn independent_update_scopes_share_a_traversal_and_keep_distinct_exports() {
    for (source, results) in [
        ("entry main(xs:[4]i32) ([3]i32,[5]i32)=let da=[0,0,0] let db=[1,1,1,1,1] let a=reduce_by_index(da,(+),0,map(|x|x%3,xs),xs) let b=reduce_by_index(db,|a,b|a+b+1,0,map(|x|x%5,xs),xs) in (a,b)", 2),
        ("entry main(xs:[4]i32) ([4]i32,[3]i32,[3]i32)=let p=scan((+),0,xs) let da=[1,1,1] let db=[2,2,2] let a=reduce_by_index(da,(+),0,map(|x|x%3,p),p) let b=reduce_by_index(db,|a,b|a+b+1,0,map(|x|x%3,p),p) in (p,map(|x|2*x,a),b)", 3),
        ("entry main() ([2]u32,u32,[2]u32,u32)=let da=[[-1,-1],[-1,-1]] let db=[[-2],[-2]] let (a,ac,ao)=bucket_scatter_2d(da,[[(0,10),(0,11),(1,20)]]) let (b,bc,bo)=bucket_scatter_2d(db,[[(0,30),(0,31),(1,40)]]) in (ac,ao,bc,bo)", 4),
    ] {
        let artifact = compile(source);
        assert_eq!(artifact.interface.source_results.len(), results);
        let storage = (0..results).map(|i| artifact.result(i)).collect::<BTreeSet<_>>();
        assert_eq!(storage.len(), results, "independent exports cannot alias");
        for buffer in storage { assert!(artifact.compute().stages.iter().any(|s| s.writes.contains(&buffer))); }
        let shared = if results < 4 {
            artifact.shader.entry_points.iter().map(|entry| {
                let mut destinations = BTreeSet::new();
                visit(&entry.function.body, 0, &mut |statement, _| {
                    if let naga::Statement::Atomic { pointer, .. } = statement {
                        if let Some(global) = storage_root(&entry.function, *pointer) {
                            destinations.insert(global);
                        }
                    }
                });
                destinations.len()
            }).max().unwrap()
        } else {
            let counters = [artifact.result(0), artifact.result(2)];
            artifact.compute().stages.iter().map(|s|
                counters.iter().filter(|b| s.reads.contains(b) && s.writes.contains(b)).count()
            ).max().unwrap()
        };
        assert_eq!(shared, 2, "one traversal updates both independent destinations: {source}");
    }
}

fn storage_root(
    function: &naga::Function,
    pointer: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::GlobalVariable>> {
    match function.expressions[pointer] {
        naga::Expression::GlobalVariable(global) => Some(global),
        naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
            storage_root(function, base)
        }
        _ => None,
    }
}

#[test]
fn fixed_length_row_selection_keeps_both_producer_backings() {
    let artifact = compile("entry main(xs:[137]i32,i:i32) i32=[map(|x|x+1,xs),map(|x|x+2,xs)][i][0]");
    let stages = &artifact.compute().stages;
    assert_eq!(stages.len(), 2, "produce rows, then select one");
    assert_eq!(stages[0].writes.len(), 2, "two distinct row backings");
    let selector = &artifact.shader.entry_points.last().unwrap().function;
    let reads = selector
        .expressions
        .iter()
        .filter_map(|(_, expression)| {
            let naga::Expression::Load { pointer } = expression else {
                return None;
            };
            let global = storage_root(selector, *pointer)?;
            let binding = artifact.shader.global_variables[global].binding.as_ref()?;
            Some((binding.group, binding.binding))
        })
        .collect::<BTreeSet<_>>();
    for &written in &stages[0].writes {
        let slot = artifact.compute().bindings[written].slot().unwrap();
        assert!(
            reads.contains(&slot),
            "runtime row selection must retain producer {slot:?}: {reads:?}"
        );
    }
}

#[test]
fn fixed_length_rows_publish_their_nested_shape() {
    let artifact = compile("entry main(xs:[137]i32) [2][137]i32=[map(|x|x+1,xs),map(|x|x+2,xs)]");
    let [result] = artifact.interface.source_results.as_slice() else {
        panic!("one nested result");
    };
    let rank = match &result.shape {
        host::ResultShape::Rectangular(dimensions) => dimensions.len(),
        host::ResultShape::Segmented { offsets, .. } => offsets.len() + 1,
    };
    assert_eq!(rank, 2, "the source result keeps both array dimensions");
    assert_eq!(result.kind, host::ResultKind::Value);
    if let host::ResultShape::Segmented { .. } = &result.shape {
        let host::ResultLayout::Array { element, stride, .. } = &result.layout else {
            panic!("segmented payload");
        };
        assert_eq!(*stride, 4);
        assert_eq!(**element, host::ResultLayout::Scalar(host::ResultScalar::I32));
    }
}

#[test]
fn closed_array_regions_distribute_before_physical_placement() {
    for (source, distributed) in [
        ("entry main(xs:[4]i32,ys:[4]i32,flag:bool) [4]i32=if flag then map(|x|x+1,xs) else map(|y|y+2,ys)", true),
        ("entry main(xs:[4]i32,ys:[4]i32,flag:bool) [4](i32,i32)=if flag then map(|x|(x,x+1),xs) else map(|y|(y,y+2),ys)", true),
        ("entry main(xs:[]i32,ys:[]i32,flag:bool) []i32=if flag then map(|x|x+1,xs) else map(|y|y+2,ys)", false),
        ("entry main(xs:[0]i32,k:i32) [0]i32=if 1/k>0 then map(|x|x+1,xs) else map(|x|x+2,xs)", false),
    ] {
        let artifact = compile(source);
        artifact.stages(if distributed { 1 } else { 2 });
        if distributed {
            assert_eq!(artifact.shader.entry_points.len(), 1);
            let mut choices = 0;
            visit(&artifact.shader.entry_points[0].function.body, 0, &mut |statement, _| {
                choices += usize::from(matches!(statement, naga::Statement::If { .. }));
            });
            assert!(choices > 0, "the selected branch stays inside the distributed shader");
        }
    }
}

#[test]
fn ranked_fusion_preserves_prefix_evaluation_and_retained_rows() {
    for (source, fused, retained) in [
        ("entry main(dest:*[4][8]i32) ([4][8]i32,[4]u32,u32)=let items:[4][6](i32,i32)=map(|bucket|let offset=bucket*2 let shifted=offset+1 in map(|pair|((shifted+pair)%4,shifted+pair),iota(6)),iota(4)) in bucket_scatter_2d(dest,items)", true, false),
        ("entry main(dest:*[4][8]i32,k:i32) ([4][8]i32,[4]u32,u32)=let items:[4][0](i32,i32)=map(|bucket|let offset=1/k in map(|pair|(bucket,offset+pair),iota(0)),iota(4)) in bucket_scatter_2d(dest,items)", false, false),
        ("entry main(dest:*[4][8]i32) ([4][6](i32,i32),([4][8]i32,[4]u32,u32))=let items:[4][6](i32,i32)=map(|bucket|map(|pair|((bucket+pair)%4,bucket+pair),iota(6)),iota(4)) in (items,bucket_scatter_2d(dest,items))", false, true),
    ] {
        let artifact = compile(source);
        artifact.stages(if fused { 1 } else { 2 });
        if retained { assert_eq!(artifact.allocated(4*6*2*4, None), 1, "exported rows retain their backing allocation"); }
    }
}

#[test]
fn workgroup_candidates_cover_uniform_control_flow_and_varying_inner_domains() {
    for source in [
        "entry main(xs:[][17]i32) []i32=map(|row|if row[0]>0 then reduce((+),0,row) else 0,xs)",
        "entry main(xs:[][17]i32) []i32=map(|row|loop a=0 for i<2 do a+reduce((+),0,row),xs)",
        "entry main(xs:[]i32) []i32=map(|n|reduce((+),0,iota(n)),xs)",
        "entry main(xs:[][2][17]i32) [][2]i32=map(|rows|map(|row|if row[0]>0 then reduce((+),0,row) else 0,rows),xs)",
        "entry main(xs:[][2][17]i32) [][2]i32=map(|rows|map(|row|loop a=0 for i<2 do a+reduce((+),0,row),rows),xs)",
    ] {
        let artifact = compile(source);
        artifact.stages(1);
        assert!(artifact.compute().stages[0].workgroup_size.0 > 1, "parallel work within each segment");
        let mut barriers = 0;
        for entry in &artifact.shader.entry_points {
            visit(&entry.function.body, 0, &mut |statement, _| {
                barriers += usize::from(matches!(statement, naga::Statement::ControlBarrier(_)));
            });
        }
        assert!(barriers > 0, "use cooperative workgroup reduction, not a serial loop per lane: {source}");
    }
}

#[test]
fn segment_lengths_are_bound_inside_each_workgroup() {
    for source in [
        "entry main(xs:[4]i32,n:i32) []i32=map(|x|reduce((+),0,map(|y|x+y,iota(n))),xs)",
        "entry main(xs:[4]i32,n:i32) [][]i32=map(|x|scan((+),0,map(|y|x+y,iota(n))),xs)",
        "entry main(xs:[4]i32) []i32=map(|x|reduce((+),0,iota(x)),xs)",
    ] {
        let artifact = compile(source);
        artifact.stages(1);
        let stage = &artifact.compute().stages[0];
        assert!(stage.workgroup_size.0 > 1);
        match &stage.dispatch_size {
            DispatchSize::Fixed { x, .. } => assert_eq!(*x, 4, "one workgroup per source row"),
            DispatchSize::Computed { groups, .. } => assert_eq!(
                constant(&groups[0]),
                Some(4),
                "row lengths do not determine the number of segments"
            ),
        }
    }
}

#[test]
fn pointwise_scan_consumers_share_the_final_prefix_dispatch() {
    for (extent, stages) in [("0", 1), ("17", 1), ("257", 3), ("", 3)] {
        let source = pipeline::source(&format!(
            "entry main(xs:[{extent}]i32,bias:i32) ([{extent}]i32,[{extent}]i32)=
             let prefix=scan((+),0,xs) in (prefix,map(|x|x*x+bias,prefix))"
        ));
        let ssa = pipeline::lower(&source, CodegenTarget::Portable);
        let wgsl = lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap();
        let module = naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let spirv = wyn_core::lower_ssa_to_spirv(ssa).unwrap();
        let module =
            naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &Default::default()).parse().unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        for program in [wgsl.program, spirv.program] {
            let [Pipeline::Compute(compute)] = program.interface.pipelines.as_slice() else {
                panic!("compute pipeline");
            };
            assert_eq!(compute.stages.len(), stages, "extent {extent}");
            assert_eq!(program.interface.source_results.len(), 2);
            for result in &program.interface.source_results {
                let buffer = compute
                    .bindings
                    .iter()
                    .position(|b| b.slot() == Some((result.set, result.binding)))
                    .unwrap();
                assert!(
                    compute.stages.last().unwrap().writes.contains(&buffer),
                    "both prefixes and mapped values complete in the last dispatch"
                );
            }
        }
    }
}

#[test]
fn packed_prefix_epilogues_preserve_retained_inputs_and_changed_elements() {
    for source in [
        "entry main(xs:[17]i32) ([]i32,[]i32)=let p=scan((+),0,filter(|x|x!=0,xs)) in (p,map(|x|x*2,p))",
        "entry main(xs:[]i32) ([]i32,[]i32)=let p=scan((+),0,filter(|x|x!=0,xs)) in (p,map(|x|x*2,p))",
        "entry main(xs:[]i32) []bool=map(|x|x>0,scan((+),0,filter(|x|x!=0,xs)))",
    ] {
        let artifact = compile(source);
        artifact.stages(1);
        if artifact.interface.source_results.len() == 2 {
            assert_ne!(
                artifact.result(0),
                artifact.result(1),
                "retained prefixes cannot be overwritten"
            );
        }
        for result in &artifact.interface.source_results {
            assert!(
                result.shape.expressions().iter().all(|extent| constant(extent).is_none()),
                "packed result lengths are runtime values, not capacities"
            );
        }
    }
}
