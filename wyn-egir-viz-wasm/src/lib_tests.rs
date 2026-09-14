use super::*;

#[test]
fn semantic_operation_fixpoint_exposes_dead_elimination_and_fusion() {
    let result = inspect_impl(
        r#"
entry main(xs: [4]i32) [4]i32 =
  let dead = map(|x: i32| x + 99, xs) in
  let a = map(|x: i32| x + 1, xs) in
  let b = map(|x: i32| x * 2, a) in
  map(|x: i32| x - 3, b)
"#,
    );
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.pass, InspectPass::OPTIMIZE_SEMANTIC_OPERATIONS);
    let before = result.before.expect("before snapshot");
    let after = result.after.expect("after snapshot");
    assert_eq!(
        before.nodes.iter().filter(|node| node.variant == "segmap").count(),
        4
    );
    assert_eq!(
        after.nodes.iter().filter(|node| node.variant == "segmap").count(),
        1
    );
    let before_map = before
        .nodes
        .iter()
        .find(|node| node.variant == "segmap")
        .expect("before snapshot has a map-shaped Screma");
    let before_operation = before_map.operation.as_ref().expect("Screma display is structured");
    assert_eq!(before_map.label, "soac.screma");
    assert!(
        !before_operation.results.is_empty(),
        "Screma result routes are structured"
    );
    assert_eq!(
        before_operation
            .operand_groups
            .iter()
            .find(|group| group.role == "inputs")
            .expect("Screma inputs")
            .values
            .len(),
        1
    );
    let pre =
        before_operation.regions.iter().find(|region| region.role == "pre").expect("Screma pre lambda");
    assert!(!pre.identity);
    assert!(pre.symbol.as_deref().is_some_and(|symbol| symbol.starts_with("_w_lambda")));
    assert!(before_operation.regions.iter().any(|region| region.role == "post" && region.identity));

    let after_map = after
        .nodes
        .iter()
        .find(|node| node.variant == "segmap")
        .expect("after snapshot has a map-shaped Screma");
    assert_eq!(after_map.label, "soac.screma");
    assert!(after_map.operation.is_some());
    assert!(
        result.relations.iter().any(|relation| relation.before.len() > relation.after.len()),
        "expected compiler-authored many-to-one fusion provenance"
    );
    assert!(
        result.relations.iter().any(|relation| relation.before.len() == 1
            && relation.before[0].ends_with("/op:0")
            && relation.after.is_empty()),
        "expected compiler-authored dead-operation provenance"
    );
}

#[test]
fn reification_records_route_writers() {
    let result = inspect_pass_impl(
        r#"
entry main(xs: [4]i32) [4]i32 =
  map(|x: i32| x + 1, xs)
"#,
        InspectPass::ReifySoacs,
    );
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.pass, InspectPass::REIFY_SOACS);
    let before = result.before.expect("before snapshot");
    let after = result.after.expect("after snapshot");
    let before_output = &before
        .groups
        .iter()
        .find(|group| group.kind == "entry")
        .expect("before snapshot has an entry")
        .outputs[0];
    let after_output = &after
        .groups
        .iter()
        .find(|group| group.kind == "entry")
        .expect("after snapshot has an entry")
        .outputs[0];
    assert!(
        before_output.routes.iter().all(|route| route.writers.is_empty()),
        "conversion records output sources without claiming concrete writers"
    );
    assert!(
        after_output.routes.iter().any(|route| !route.writers.is_empty()),
        "reification records the semantic values that publish the slot"
    );

    let before_map = before
        .nodes
        .iter()
        .find(|node| node.variant == "segmap")
        .and_then(|node| node.operation.as_ref())
        .expect("raw map operation");
    let after_map = after
        .nodes
        .iter()
        .find(|node| node.variant == "segmap")
        .and_then(|node| node.operation.as_ref())
        .expect("semantic map operation");
    assert!(before_map.semantic_id.is_none());
    assert!(after_map.semantic_id.as_deref().is_some_and(|id| id.starts_with("op:")));
    let before_state = before_map.soac_state.as_ref().expect("raw Screma state");
    let after_state = after_map.soac_state.as_ref().expect("semantic Screma state");
    assert_eq!(before_state.phase, "raw");
    assert_eq!(before_state.variant, "raw");
    assert_eq!(after_state.phase, "semantic");
    assert_eq!(after_state.variant, "segmented");
    assert_eq!(after_state.output_slots, [0]);
    assert_eq!(after_state.space.len(), 1);
    assert_eq!(after_state.space[0].variant, "fixed");
    assert_eq!(after_state.space[0].fixed, Some(4));
    assert!(after_state.resources.iter().any(|access| access.access == "write"));
}

#[test]
fn reification_leaves_runtime_filter_allocation_deferred() {
    let result = inspect_pass_impl(
        r#"
entry evens(xs: []i32) []i32 =
  filter(|x: i32| x % 2 == 0, xs)
"#,
        InspectPass::ReifySoacs,
    );
    assert!(result.success, "{:?}", result.error);
    let before = result.before.expect("before snapshot");
    let after = result.after.expect("after snapshot");
    let before_filter = before
        .nodes
        .iter()
        .find(|node| node.variant == "filter")
        .and_then(|node| node.operation.as_ref())
        .and_then(|operation| operation.soac_state.as_ref())
        .expect("converted filter state");
    let after_filter = after
        .nodes
        .iter()
        .find(|node| node.variant == "filter")
        .and_then(|node| node.operation.as_ref())
        .and_then(|operation| operation.soac_state.as_ref())
        .expect("semantic filter state");
    assert_eq!(before_filter.phase, "raw");
    assert_eq!(before_filter.variant, "raw");
    assert_eq!(after_filter.phase, "semantic");
    assert_eq!(after_filter.variant, "segmented");
    assert_eq!(after_filter.output_slots, [0]);
    let before_output_state = before_filter.filter_output.as_ref().expect("raw Filter output");
    let after_output_state = after_filter.filter_output.as_ref().expect("semantic Filter output");
    assert_eq!(before_output_state.capacity.variant, "like_input");
    assert_eq!(before_output_state.capacity.input, Some(0));
    assert!(before_output_state.backing.is_none());
    assert!(before_output_state.length.is_none());
    assert_eq!(
        after_output_state.backing.as_ref().map(|backing| backing.variant.as_str()),
        Some("deferred")
    );
    assert_eq!(
        after_output_state.length.as_ref().map(|length| length.variant.as_str()),
        Some("implicit")
    );

    let after_entry = after.groups.iter().find(|group| group.kind == "entry").unwrap();
    let output = &after_entry.outputs[0];
    assert_eq!(
        output.kind.length.as_ref().map(|length| length.variant.as_str()),
        Some("like_input")
    );
}

#[test]
fn residency_exposes_filter_resources_and_handoff() {
    let result = inspect_pass_impl(
        r#"
entry main(xs: []i32) []i32 =
  let selected = filter(|x: i32| x % 2 == 0, xs) in
  map(|x: i32| x * 2, selected)
"#,
        InspectPass::ResolveResidency,
    );
    assert!(result.success, "{:?}", result.error);
    let allocation = inspect_pass_impl(
        r#"entry main(xs: []i32) []i32 =
  let selected = filter(|x: i32| x % 2 == 0, xs) in
  map(|x: i32| x * 2, selected)"#,
        InspectPass::AllocateSemanticResources,
    );
    assert!(allocation.success, "{:?}", allocation.error);
    let before = allocation.before.expect("optimized snapshot");
    let after = result.after.expect("allocated snapshot");
    assert!(before.resources.is_empty());
    assert!(before.stages.is_empty());
    assert!(before.flows.is_empty());

    let data = after
        .resources
        .iter()
        .find(|resource| resource.origin.compiler_kind.as_deref() == Some("filter_data"))
        .expect("filter data resource");
    let length = after
        .resources
        .iter()
        .find(|resource| resource.origin.compiler_kind.as_deref() == Some("filter_len_cell"))
        .expect("filter length resource");
    assert_eq!(data.origin.owner.as_deref(), Some("op:0"));
    assert_eq!(data.size.variant, "like_resource");
    assert_eq!(length.origin.owner.as_deref(), Some("op:0"));
    assert_eq!(length.size.variant, "fixed_bytes");
    assert_eq!(length.size.bytes, Some(4));

    let producer_stage = after
        .stages
        .iter()
        .find(|stage| {
            after.nodes.iter().any(|node| stage.kernels.contains(&node.group) && node.variant == "filter")
        })
        .expect("generated Filter producer stage");
    let main_stage =
        after.stages.iter().find(|stage| stage.id != producer_stage.id).expect("authored consumer stage");
    assert!(after
        .flows
        .iter()
        .any(|flow| { flow.producer == producer_stage.id && flow.consumers.contains(&main_stage.id) }));

    let before_filter = before
        .nodes
        .iter()
        .find(|node| node.operation.as_ref().and_then(|op| op.semantic_id.as_deref()) == Some("op:0"))
        .expect("optimized Filter");
    let after_filter = after
        .nodes
        .iter()
        .find(|node| node.operation.as_ref().and_then(|op| op.semantic_id.as_deref()) == Some("op:0"))
        .expect("allocated Filter");
    assert_eq!(before_filter.group, "entry:0");
    assert_eq!(after_filter.group, producer_stage.kernels[0]);
    let before_output = before_filter
        .operation
        .as_ref()
        .and_then(|operation| operation.soac_state.as_ref())
        .and_then(|state| state.filter_output.as_ref())
        .expect("optimized Filter output state");
    assert_eq!(
        before_output.backing.as_ref().map(|value| value.variant.as_str()),
        Some("deferred")
    );
    assert_eq!(
        before_output.length.as_ref().map(|value| value.variant.as_str()),
        Some("implicit")
    );
    let after_output = after_filter
        .operation
        .as_ref()
        .and_then(|operation| operation.soac_state.as_ref())
        .and_then(|state| state.filter_output.as_ref())
        .expect("allocated Filter output state");
    assert_eq!(
        after_output.backing.as_ref().and_then(|value| value.resource.as_deref()),
        Some(data.id.as_str())
    );
    assert_eq!(
        after_output.length.as_ref().and_then(|value| value.resource.as_deref()),
        Some(length.id.as_str())
    );

    let main =
        after.groups.iter().find(|group| group.id == main_stage.kernels[0]).expect("main stage body");
    assert!(main
        .resource_declarations
        .iter()
        .any(|decl| { decl.resource == data.id && decl.role == "input" }));
    let producer = after
        .groups
        .iter()
        .find(|group| group.id == producer_stage.kernels[0])
        .expect("producer stage body");
    assert!(producer
        .resource_declarations
        .iter()
        .any(|decl| { decl.resource == data.id && decl.role == "output" }));
}

#[test]
fn allocation_checkpoints_retain_stage_bodies_and_resident_flows() {
    let source = r#"entry main(xs: []i32) []i32 =
  let selected = filter(|x: i32| x % 2 == 0, xs) in
  map(|x: i32| x * 2, selected)"#;
    let allocation = inspect_pass_impl(source, InspectPass::AllocateSemanticResources);
    assert!(allocation.success, "{:?}", allocation.error);
    let allocated = allocation.after.expect("allocation snapshot");
    assert_eq!(allocated.stages.len(), 1, "allocation retains the authored stage");
    let main = &allocated.stages[0];
    assert_eq!(main.kernels.len(), 1);
    assert!(allocated.nodes.iter().any(|node| node.group == main.kernels[0] && node.variant == "filter"));
    assert!(
        allocated.flows.is_empty(),
        "the handoff is introduced by residency"
    );

    let residency = inspect_pass_impl(source, InspectPass::ResolveResidency);
    assert!(residency.success, "{:?}", residency.error);
    let before = residency.before.expect("before residency");
    assert_eq!(before.stages[0].kernels[0], main.kernels[0]);
    assert_eq!(before.nodes.len(), allocated.nodes.len());
    let resident = residency.after.expect("after residency");
    let producer = resident
        .stages
        .iter()
        .find(|stage| {
            resident
                .nodes
                .iter()
                .any(|node| stage.kernels.contains(&node.group) && node.variant == "filter")
        })
        .expect("generated Filter producer is visible before finalization");
    let consumer = resident.stages.iter().find(|stage| stage.id != producer.id).unwrap();
    let flow = resident
        .flows
        .iter()
        .find(|flow| flow.producer == producer.id && flow.consumers.contains(&consumer.id))
        .expect("resident handoff is visible before finalization");
    assert!(producer.outgoing_flows.contains(&flow.id));
    assert!(consumer.incoming_flows.contains(&flow.id));
    assert!(
        flow.length_resource.is_some(),
        "dynamic handoff retains its logical length"
    );
    assert!(resident
        .nodes
        .iter()
        .any(|node| node.group == producer.kernels[0] && node.variant == "filter"));
    assert!(
        resident.external_inputs.is_empty(),
        "external inputs are linked at finalization"
    );

    let finalization = inspect_pass_impl(source, InspectPass::FinalizeStagedIr);
    assert!(finalization.success, "{:?}", finalization.error);
    let before = finalization.before.expect("before finalization");
    let after = finalization.after.expect("after finalization");
    assert_eq!(before.stages.len(), resident.stages.len());
    assert_eq!(before.flows.len(), resident.flows.len());
    assert_eq!(after.stages.len(), before.stages.len());
    for stage in &before.stages {
        let finalized = after.stages.iter().find(|candidate| candidate.id == stage.id).unwrap();
        assert_eq!(finalized.origin, stage.origin);
        assert!(!finalized.kernels.is_empty());
        for kernel in &finalized.kernels {
            assert!(after.groups.iter().any(|group| &group.id == kernel));
            assert!(after
                .recipes
                .iter()
                .any(|recipe| &recipe.entry_group == kernel && recipe.stage == stage.id));
        }
    }
    let finalized_flow = after.flows.iter().find(|candidate| candidate.id == flow.id).unwrap();
    assert_eq!(finalized_flow.producer, flow.producer);
    assert_eq!(finalized_flow.consumers, flow.consumers);
    assert_eq!(finalized_flow.data_resource, flow.data_resource);
    assert_eq!(finalized_flow.length_resource, flow.length_resource);
    assert!(!after.external_inputs.is_empty());
    assert!(after.flows.iter().any(|flow| flow.published));
}

#[test]
fn physical_planning_exposes_kernel_dag_and_owned_bodies() {
    let result = inspect_pass_impl(
        r#"
entry sum(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0, xs)
"#,
        InspectPass::PlanPhysicalKernels,
    );
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.pass, InspectPass::PLAN_PHYSICAL_KERNELS);
    let before = result.before.expect("staged snapshot");
    let after = result.after.expect("physical snapshot");
    assert!(before.stages.is_empty());
    assert!(before.groups.iter().any(|group| group.kind == "entry"));
    assert!(before.kernels.is_empty());
    assert!(after.stages.is_empty());
    assert!(
        after.kernels.len() >= 2,
        "parallel reduction should produce a kernel chain"
    );
    assert_eq!(
        after.kernels.len(),
        after.groups.iter().filter(|group| group.kind == "kernel").count()
    );
    assert!(after.kernels.iter().skip(1).any(|kernel| !kernel.dependencies.is_empty()));
    for kernel in &after.kernels {
        assert!(after.groups.iter().any(|group| group.id == kernel.entry_group));
    }
}

#[test]
fn physical_planning_exposes_entry_parameter_channels() {
    let result = inspect_pass_impl(
        r#"
entry shifted(xs: [4]i32, offsets: []i32, index: i32) [4]i32 =
  map(|x: i32| x + offsets[index], xs)
"#,
        InspectPass::PlanPhysicalKernels,
    );
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.pass, InspectPass::PLAN_PHYSICAL_KERNELS);
    let before = result.before.expect("staged snapshot");
    let after = result.after.expect("physical snapshot");

    let before_parameters =
        before.nodes.iter().filter(|node| node.variant == "parameter").collect::<Vec<_>>();
    let after_parameters =
        after.nodes.iter().filter(|node| node.variant == "parameter").collect::<Vec<_>>();
    assert!(
        before_parameters.iter().any(|parameter| {
            parameter.representation.as_deref() == Some("value")
                && parameter.ty.as_deref() == Some("[4]i32")
        }),
        "the fixed array starts on the value channel: {before_parameters:#?}"
    );
    assert!(
        after_parameters.iter().any(|parameter| {
            parameter.representation.as_deref() == Some("place")
                && parameter.ty.as_deref() == Some("[4]i32")
        }),
        "the fixed array moves to the read-only-place channel: {after_parameters:#?}"
    );
    assert!(
        after_parameters.iter().any(|parameter| {
            parameter.representation.as_deref() == Some("view")
                && parameter.ty.as_deref().is_some_and(|ty| ty.starts_with("[?"))
        }),
        "runtime view parameter: {after_parameters:#?}"
    );
    assert!(after_parameters.iter().any(|parameter| {
        parameter.representation.as_deref() == Some("value") && parameter.ty.as_deref() == Some("i32")
    }));
    let fixed_group = after_parameters
        .iter()
        .find(|parameter| parameter.representation.as_deref() == Some("place"))
        .unwrap()
        .group
        .clone();
    assert_eq!(
        after.nodes.iter().filter(|node| node.group == fixed_group && node.variant == "place").count(),
        1,
        "the fixed input should have one place.view"
    );
}

#[test]
fn physical_planning_exposes_final_callable_boundaries_and_calls() {
    let result = inspect_pass_impl(
        r#"
open f32
def use_world(points: [4]vec2f32, items: [4]vec4f32, dom: [4]u32) f32 =
  use_world(points, items, dom)

entry step(dom: [4]u32, points_in: [4]vec2f32, items_in: [4]vec4f32)
  f32 =
  use_world(points_in, items_in, dom)
"#,
        InspectPass::PlanPhysicalKernels,
    );
    assert!(result.success, "{:?}", result.error);
    let after = result.after.expect("physical snapshot");
    let function =
        after.groups.iter().find(|group| group.label.starts_with("fn use_world")).unwrap_or_else(|| {
            panic!(
                "use_world function group among {:?}",
                after.groups.iter().map(|group| &group.label).collect::<Vec<_>>()
            )
        });
    let parameters = after
        .nodes
        .iter()
        .filter(|node| {
            node.group == function.id
                && node.variant == "parameter"
                && node.representation.as_deref() == Some("place")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        parameters.iter().filter(|parameter| parameter.detail.contains("ReadOnly")).count(),
        3,
        "the record fields and domain are final read-only place inputs: {parameters:#?}"
    );
    assert!(
        after.nodes.iter().any(|node| {
            node.variant == "call"
                && node.operation.as_ref().is_some_and(|operation| {
                    operation
                        .operand_groups
                        .iter()
                        .find(|group| group.role == "arguments")
                        .is_some_and(|arguments| arguments.values.len() == 3)
                })
        }),
        "the caller should expose three correspondingly adapted arguments"
    );
}

#[test]
fn inline_debug_preserves_long_constructs() {
    let value = "ResourceLen(SemanticResourceRef(ResourceIdentifierThatMustRemainVisible))";
    let rendered = inline_debug(&value);
    assert!(rendered.contains("ResourceIdentifierThatMustRemainVisible"));
    assert!(!rendered.contains('…'));
}

#[test]
fn physical_subpasses_are_individually_inspectable() {
    let cases = [
        (
            InspectPass::LowerSoacs,
            r#"entry scan_offsets(xs: []i32) []i32 =
  scan(|a: i32, b: i32| a + b, 0, xs)"#,
        ),
        (
            InspectPass::EliminateInternalPlaceCalls,
            r#"def choose_sum(values: [4]i32, flag: u32) i32 =
  let left = values[0] + values[1] in
  let right = values[2] + values[3] in
  if flag == 0u32 then left else right

entry call_place(values: [4]i32, flag: u32) i32 =
  choose_sum(values, flag)"#,
        ),
        (
            InspectPass::PartiallyInlineCalls,
            r#"def choose_and_scale(varying: u32, invariant: u32) u32 =
  let scale = invariant * invariant in
  if varying == 0u32 then scale else varying + scale

entry mixed_loop(seed: u32, scale: u32) u32 =
  loop value = seed for i < 4 do
    let stable = choose_and_scale(0u32, scale) in
    choose_and_scale(value + u32.i32(i), stable)"#,
        ),
        (
            InspectPass::MaterializeDynamicExtracts,
            r#"entry dynamic_local(index: i32) i32 =
  let values = [10, 20, 30, 40] in
  values[index]"#,
        ),
        (
            InspectPass::Rewrite,
            r#"entry power_chain(x: f32) f32 =
  x ** 5.0f32"#,
        ),
        (
            InspectPass::OptimizeSkeleton,
            r#"def choose_and_scale(varying: u32, invariant: u32) u32 =
  let scale = invariant * invariant in
  if varying == 0u32 then scale else varying + scale

entry mixed_loop(seed: u32, scale: u32) u32 =
  loop value = seed for i < 4 do
    let stable = choose_and_scale(0u32, scale) in
    choose_and_scale(value + u32.i32(i), stable)"#,
        ),
        (
            InspectPass::EraseResources,
            r#"entry unchanged_scalar(value: i32) i32 = value + 1"#,
        ),
    ];

    for (pass, source) in cases {
        let result = inspect_pass_impl(source, pass);
        assert!(result.success, "{} failed: {:?}", pass.id(), result.error);
        assert!(result.before.is_some(), "{} has no before snapshot", pass.id());
        assert!(result.after.is_some(), "{} has no after snapshot", pass.id());
    }
}

#[test]
fn semantic_and_allocation_subpasses_are_individually_inspectable() {
    let cases = [
        (
            InspectPass::EliminateDeadSemanticOperations,
            r#"entry discard_map(xs: [4]i32) [4]i32 =
  let dead = map(|x: i32| x + 99, xs) in
  xs"#,
        ),
        (
            InspectPass::FuseSemanticOperations,
            r#"entry fuse_maps(xs: [4]i32) [4]i32 =
  let shifted = map(|x: i32| x + 1, xs) in
  map(|x: i32| x * 2, shifted)"#,
        ),
        (
            InspectPass::LiftStageUniformValues,
            r#"entry uniform_capture(xs: [4]i32, bias: i32) [4]i32 =
  map(|x: i32| x + bias * bias, xs)"#,
        ),
        (
            InspectPass::AllocateSemanticResources,
            r#"entry allocate_filter(xs: []i32) []i32 =
  filter(|x: i32| x % 2 == 0, xs)"#,
        ),
        (
            InspectPass::ResolveResidency,
            r#"entry resident_filter(xs: []i32) []i32 =
  let selected = filter(|x: i32| x > 0, xs) in
  map(|x: i32| x + 1, selected)"#,
        ),
        (
            InspectPass::FinalizeStagedIr,
            r#"entry staged_filter(xs: []i32) []i32 =
  let selected = filter(|x: i32| x != 0, xs) in
  map(|x: i32| x * x, selected)"#,
        ),
    ];

    for (pass, source) in cases {
        let result = inspect_pass_impl(source, pass);
        assert!(result.success, "{} failed: {:?}", pass.id(), result.error);
        assert!(result.before.is_some(), "{} has no before snapshot", pass.id());
        assert!(result.after.is_some(), "{} has no after snapshot", pass.id());
    }
}

#[test]
fn physical_planning_subpasses_are_individually_inspectable() {
    let source = r#"entry planned_sum(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0, xs)"#;
    let passes = [
        InspectPass::FinalizeStagedIr,
        InspectPass::AllocateRecipeScratch,
        InspectPass::BuildKernelSchedule,
        InspectPass::PhysicalizeKernelSchedule,
    ];

    for pass in passes {
        let result = inspect_pass_impl(source, pass);
        assert!(result.success, "{} failed: {:?}", pass.id(), result.error);
        assert!(result.before.is_some(), "{} has no before snapshot", pass.id());
        assert!(result.after.is_some(), "{} has no after snapshot", pass.id());
    }
}

#[test]
fn fusion_step_performs_one_action_while_aggregate_reaches_fixpoint() {
    let source = r#"entry chain(xs: [4]i32) [4]i32 =
  let a = map(|x: i32| x + 1, xs) in
  let b = map(|x: i32| x * 2, a) in
  map(|x: i32| x - 3, b)"#;
    let step = inspect_pass_impl(source, InspectPass::FuseSemanticOperations);
    assert!(step.success, "{:?}", step.error);
    assert_eq!(
        step.before.unwrap().nodes.iter().filter(|node| node.variant == "segmap").count(),
        3
    );
    assert_eq!(
        step.after.unwrap().nodes.iter().filter(|node| node.variant == "segmap").count(),
        2
    );
    assert_eq!(step.relations.len(), 1);
    let aggregate = inspect_impl(source);
    assert!(aggregate.success, "{:?}", aggregate.error);
    assert_eq!(
        aggregate.after.unwrap().nodes.iter().filter(|node| node.variant == "segmap").count(),
        1
    );
    assert_eq!(aggregate.relations.len(), 2);
}

#[test]
fn recipes_own_body_references_and_scratch_across_public_transitions() {
    let source = "entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)";
    let finalized = inspect_pass_impl(source, InspectPass::FinalizeStagedIr);
    assert!(finalized.success, "{:?}", finalized.error);
    assert!(finalized.before.unwrap().recipes.is_empty());
    let planned = finalized.after.unwrap();
    let recipe = planned.recipes.iter().find(|recipe| recipe.kind == "reduce").unwrap();
    assert!(planned
        .nodes
        .iter()
        .any(|node| Some(&node.id) == recipe.operation.as_ref() && node.group == recipe.entry_group));
    assert!(!recipe.details["routing"].as_array().unwrap().is_empty());
    assert!(!recipe.scratch.is_empty());
    assert!(recipe.scratch.iter().all(|slot| slot["state"] == "required"));

    let allocation = inspect_pass_impl(source, InspectPass::AllocateRecipeScratch);
    assert!(allocation.success, "{:?}", allocation.error);
    let bound = allocation.after.unwrap();
    let allocated = bound.recipes.iter().find(|candidate| candidate.id == recipe.id).unwrap();
    assert_eq!(allocated.operation, recipe.operation);
    assert_eq!(allocated.details, recipe.details);
    assert_eq!(allocated.output_projection, recipe.output_projection);
    for (requirement, resource) in recipe.scratch.iter().zip(&allocated.scratch) {
        assert_eq!(requirement["role"], resource["role"]);
        assert_eq!(resource["state"], "bound");
        assert!(bound.resources.iter().any(|decl| resource["resource"] == decl.id));
    }
    let repeated = inspect_pass_impl(source, InspectPass::AllocateRecipeScratch).after.unwrap();
    assert_eq!(
        serde_json::to_value(&bound.resources).unwrap(),
        serde_json::to_value(&repeated.resources).unwrap()
    );
    assert_eq!(
        bound.recipes.iter().map(|recipe| (&recipe.id, &recipe.scratch)).collect::<Vec<_>>(),
        repeated.recipes.iter().map(|recipe| (&recipe.id, &recipe.scratch)).collect::<Vec<_>>(),
    );

    let scheduled = inspect_pass_impl(source, InspectPass::BuildKernelSchedule);
    assert!(scheduled.success, "{:?}", scheduled.error);
    let schedule = scheduled.after.unwrap();
    assert!(schedule.stages.is_empty());
    assert!(schedule.recipes.is_empty());
    assert!(schedule.kernels.len() >= 2);
    assert_eq!(schedule.publications, planned.publications);
    for kernel in &schedule.kernels {
        assert_eq!(kernel.planned_component.as_ref(), Some(&recipe.id));
        assert!(schedule.nodes.iter().any(|node| node.group == kernel.entry_group));
        assert!(kernel.resources.iter().all(|access| access.resource.is_some()));
    }
    let physicalized = inspect_pass_impl(source, InspectPass::PhysicalizeKernelSchedule);
    assert!(physicalized.success, "{:?}", physicalized.error);
    let physical = physicalized.after.unwrap();
    assert_eq!(
        schedule.kernels.iter().map(|kernel| &kernel.id).collect::<Vec<_>>(),
        physical.kernels.iter().map(|kernel| &kernel.id).collect::<Vec<_>>()
    );
}

#[test]
fn projected_bodies_have_scoped_nodes_and_unique_output_owners() {
    let source = "entry mixed(a: []i32, b: []i32) ([]i32, []i32) =
        (map(|x: i32| x + 1, a), map(|x: i32| x * 2, b))";
    let result = inspect_pass_impl(source, InspectPass::FinalizeStagedIr);
    assert!(result.success, "{:?}", result.error);
    let snapshot = result.after.unwrap();
    assert_eq!(snapshot.recipes.len(), 2);
    let ids = snapshot.nodes.iter().map(|node| &node.id).collect::<std::collections::HashSet<_>>();
    assert_eq!(ids.len(), snapshot.nodes.len());
    let mut outputs = snapshot
        .recipes
        .iter()
        .flat_map(|recipe| recipe.output_projection.as_ref().unwrap())
        .copied()
        .collect::<Vec<_>>();
    outputs.sort_unstable();
    assert_eq!(outputs, [0, 1]);
    for recipe in &snapshot.recipes {
        assert!(ids.contains(recipe.operation.as_ref().unwrap()));
    }
    for relation in &result.relations {
        assert!(relation.after.iter().all(|id| ids.contains(id)));
    }
}
