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
        result.relations.iter().any(|relation| relation.before == ["op:0"] && relation.after.is_empty()),
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
fn logical_resource_planning_exposes_filter_stage_and_flow() {
    let result = inspect_pass_impl(
        r#"
entry main(xs: []i32) []i32 =
  let selected = filter(|x: i32| x % 2 == 0, xs) in
  map(|x: i32| x * 2, selected)
"#,
        InspectPass::PlanLogicalResources,
    );
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.pass, InspectPass::PLAN_LOGICAL_RESOURCES);
    let before = result.before.expect("optimized snapshot");
    let after = result.after.expect("allocated snapshot");
    assert!(before.resources.is_empty());
    assert!(before.stages.is_empty());
    assert!(before.flows.is_empty());

    let scratch = after
        .resources
        .iter()
        .find(|resource| resource.origin.compiler_kind.as_deref() == Some("filter_scratch"))
        .expect("filter scratch resource");
    let length = after
        .resources
        .iter()
        .find(|resource| resource.origin.compiler_kind.as_deref() == Some("filter_len_cell"))
        .expect("filter length resource");
    assert_eq!(scratch.origin.owner.as_deref(), Some("op:0"));
    assert_eq!(scratch.size.variant, "like_resource");
    assert_eq!(length.origin.owner.as_deref(), Some("op:0"));
    assert_eq!(length.size.variant, "fixed_bytes");
    assert_eq!(length.size.bytes, Some(4));

    let producer_stage = after
        .stages
        .iter()
        .find(|stage| stage.entry_name.starts_with("main_materialize_filter_"))
        .expect("generated Filter producer stage");
    let main_stage =
        after.stages.iter().find(|stage| stage.entry_name == "main").expect("authored consumer stage");
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
    assert_eq!(after_filter.group, producer_stage.entry_group);
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
        Some(scratch.id.as_str())
    );
    assert_eq!(
        after_output.length.as_ref().and_then(|value| value.resource.as_deref()),
        Some(length.id.as_str())
    );

    let main =
        after.groups.iter().find(|group| group.id == main_stage.entry_group).expect("main stage body");
    assert!(main
        .resource_declarations
        .iter()
        .any(|decl| { decl.resource == scratch.id && decl.role == "input" }));
    let producer = after
        .groups
        .iter()
        .find(|group| group.id == producer_stage.entry_group)
        .expect("producer stage body");
    assert!(producer
        .resource_declarations
        .iter()
        .any(|decl| { decl.resource == scratch.id && decl.role == "output" }));
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
    assert!(!before.stages.is_empty());
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

    let before_parameters = before
        .nodes
        .iter()
        .filter(|node| node.variant == "parameter")
        .collect::<Vec<_>>();
    let after_parameters = after
        .nodes
        .iter()
        .filter(|node| node.variant == "parameter")
        .collect::<Vec<_>>();
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
        parameter.representation.as_deref() == Some("value")
            && parameter.ty.as_deref() == Some("i32")
    }));
    let fixed_group = after_parameters
        .iter()
        .find(|parameter| parameter.representation.as_deref() == Some("place"))
        .unwrap()
        .group
        .clone();
    assert_eq!(
        after
            .nodes
            .iter()
            .filter(|node| node.group == fixed_group && node.variant == "place")
            .count(),
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
    let function = after
        .groups
        .iter()
        .find(|group| group.label.starts_with("fn use_world"))
        .unwrap_or_else(|| {
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
