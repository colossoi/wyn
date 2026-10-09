use crate::egglog::ScalarOptimization;
use crate::{
    compile_thru_ssa_with_policy, lower_ssa_to_spirv, lower_ssa_to_wgsl, CodegenTarget, LookupMap,
    LookupSet,
};
use wspirv::dr::Operand;
use wspirv::spirv::Op;

fn shaders(source: &str, inspect: impl Fn(&wspirv::dr::Module)) {
    for policy in [ScalarOptimization::Basic, ScalarOptimization::Full] {
        for target in [CodegenTarget::Spirv, CodegenTarget::Wgsl] {
            let program = compile_thru_ssa_with_policy(source, target, policy).unwrap();
            let module = match target {
                CodegenTarget::Spirv => {
                    let binary = lower_ssa_to_spirv(program).unwrap();
                    inspect(&wspirv::dr::load_words(&binary.spirv).unwrap());
                    naga::front::spv::Frontend::new(binary.spirv.into_iter(), &Default::default())
                        .parse()
                        .unwrap()
                }
                CodegenTarget::Wgsl => {
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
        }
    }
}

/// Whole local-array traffic, excluding storage-buffer output. This checks the
/// emitted instructions, so driver copy elimination cannot hide a regression.
fn array_traffic(module: &wspirv::dr::Module) -> (usize, usize, usize) {
    let arrays: LookupSet<_> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypeArray)
        .map(|i| i.result_id.unwrap())
        .collect();
    let pointers: LookupSet<_> = module.types_global_values.iter().filter_map(|i| {
        if i.class.opcode == Op::TypePointer && matches!(i.operands.as_slice(),
            [Operand::StorageClass(wspirv::spirv::StorageClass::Function), Operand::IdRef(ty)] if arrays.contains(ty)) {
            i.result_id
        } else { None }
    }).collect();
    let types: LookupMap<_, _> =
        module.all_inst_iter().filter_map(|i| Some((i.result_id?, i.result_type?))).collect();
    let mut counts = (0, 0, 0);
    for i in module.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions) {
        match i.class.opcode {
            Op::Variable if i.result_type.is_some_and(|ty| pointers.contains(&ty)) => counts.0 += 1,
            Op::Load if i.result_type.is_some_and(|ty| arrays.contains(&ty)) => counts.1 += 1,
            Op::Store if matches!(i.operands.first(), Some(Operand::IdRef(ptr)) if types.get(ptr).is_some_and(|t| pointers.contains(t))) => {
                counts.2 += 1
            }
            _ => {}
        }
    }
    counts
}

#[test]
fn fluid_queue_reproducer_has_no_whole_array_traffic() {
    let source = include_str!("../../../fluid-simulation/test/repro_local_queue_copy.wyn");
    shaders(source, |module| assert_eq!(array_traffic(module), (1, 0, 0)));
    for policy in [ScalarOptimization::Basic, ScalarOptimization::Full] {
        let program = compile_thru_ssa_with_policy(source, CodegenTarget::Wgsl, policy).unwrap();
        let module = naga::front::wgsl::parse_str(&lower_ssa_to_wgsl(program).unwrap()).unwrap();
        let mut allocations = 0;
        for (_, function) in module.functions.iter() {
            let arrays: LookupSet<_> = function
                .local_variables
                .iter()
                .filter(|(_, v)| matches!(module.types[v.ty].inner, naga::TypeInner::Array { .. }))
                .map(|(id, _)| id)
                .collect();
            allocations += arrays.len();
            let whole_array = |pointer| {
                matches!(function.expressions[pointer],
                naga::Expression::LocalVariable(v) if arrays.contains(&v))
            };
            assert!(!function
                .expressions
                .iter()
                .any(|(_, e)| matches!(e, naga::Expression::Load { pointer } if whole_array(*pointer))));
            let mut blocks = vec![&function.body];
            while let Some(block) = blocks.pop() {
                for statement in block.iter() {
                    match statement {
                        naga::Statement::Store { pointer, .. } => assert!(!whole_array(*pointer)),
                        naga::Statement::Block(block) => blocks.push(block),
                        naga::Statement::If { accept, reject, .. } => blocks.extend([accept, reject]),
                        naga::Statement::Loop { body, continuing, .. } => blocks.extend([body, continuing]),
                        naga::Statement::Switch { cases, .. } => {
                            blocks.extend(cases.iter().map(|c| &c.body))
                        }
                        _ => {}
                    }
                }
            }
        }
        assert_eq!(allocations, 1);
    }
}

#[test]
fn queue_versions_share_one_allocation_without_array_round_trips() {
    for (size, ty, zero, one) in [
        (4, "i32", "0i32", "1i32"),
        (1024, "vec2u32", "@[0u32,0u32]", "@[1u32,1u32]"),
    ] {
        let source = format!(
            "def walk(seed:i32, steps:i32) {ty} =
            let initial=replicate({size},{zero}) in
            let (queue,k)=loop (queue,k)=(initial,0i32) while k<steps do
                (queue with [(k+1)%{size}]=queue[k%{size}]+{one},k+1) in queue[k%{size}]
            entry reproduce(steps:i32) []{ty}=map(|i|walk(i,steps),0i32..<4)"
        );
        shaders(&source, |m| assert_eq!(array_traffic(m), (1, 0, 0), "{source}"));
    }
}

#[test]
fn branch_updates_and_record_loop_state_keep_array_storage() {
    let source = "def walk(steps:i32) i32 =
        let initial=replicate(8,0i32) in
        let state=loop state={queue=initial,k=0i32} while state.k<steps do
            {queue=if state.k%2==0 then state.queue with [state.k%8]=state.k else state.queue,
             k=state.k+1} in state.queue[(state.k-1)%8]
        entry main(steps:i32) []i32=map(|i|walk(steps+i),0i32..<4)";
    shaders(source, |m| assert_eq!(array_traffic(m), (1, 0, 0)));
}

#[test]
fn retained_versions_and_parallel_array_swaps_keep_value_semantics() {
    for body in [
        "let a=replicate(8,seed) in let (b,k)=loop (b,k)=(a,0i32) while k<steps do
            (b with [k%8]=k,k+1) in a[b[seed%8]%8]+b[seed%8]",
        "let a=replicate(8,seed) in let b=replicate(8,seed+1) in
            let (a,b,k)=loop (a,b,k)=(a,b,0i32) while k<steps do
            (b,a with [k%8]=k,k+1) in a[seed%8]+b[seed%8]",
        "let a=replicate(8,seed) in let (b,k)=loop (b,k)=(a,0i32) while k<steps do
            let next=b with [k%8]=k in (next,k+1+b[next[k%8]%8]) in b[seed%8]",
    ] {
        let source = format!(
            "def walk(seed:i32,steps:i32) i32={body}
            entry main(steps:i32) []i32=map(|i|walk(i,steps),0i32..<4)"
        );
        shaders(&source, |m| {
            assert!(
                array_traffic(m).1 > 0,
                "unproven reuse must keep array copies: {body}"
            )
        });
    }
}

#[test]
fn nested_counted_loops_reinitialize_private_arrays_each_iteration() {
    let source = "def walk(seed:i32,steps:i32) i32 =
        loop total=0 for outer<3 do
            let initial=replicate(8,seed+outer) in
            let queue=loop queue=initial for k<steps do
                queue with [(k+1)%8]=queue[k%8]+1 in total+queue[steps%8]
        entry main(steps:i32) []i32=map(|i|walk(i,steps),0i32..<4)";
    // Initialization may copy from a separate producer allocation in the outer
    // iteration. The inner walk must still carry only scalar SSA state.
    shaders(source, |module| {
        let arrays: LookupSet<_> = module
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::TypeArray)
            .map(|i| i.result_id.unwrap())
            .collect();
        assert!(!module
            .all_inst_iter()
            .any(|i| i.class.opcode == Op::Phi && i.result_type.is_some_and(|ty| arrays.contains(&ty))));
    });
}

#[test]
fn arrays_passed_to_opaque_helpers_keep_independent_contents() {
    let source = "def observe(xs:[8]i32,i:i32) i32=xs[i%8]
        def walk(seed:i32,steps:i32) i32 =
            let a=replicate(8,seed) in
            let (b,k)=loop (b,k)=(a,0i32) while k<steps do
                let next=b with [k%8]=k in (next,k+1+observe(b,next[k%8])) in b[seed%8]
        entry main(steps:i32) []i32=map(|i|walk(i,steps),0i32..<4)";
    shaders(source, |_| {});
}
