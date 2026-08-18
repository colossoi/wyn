use super::*;

#[test]
fn map_chain_produces_structured_before_and_after_snapshots() {
    let result = inspect_impl(
        r#"
entry main(xs: [4]i32) [4]i32 =
  let a = map(|x: i32| x + 1, xs) in
  let b = map(|x: i32| x * 2, a) in
  map(|x: i32| x - 3, b)
"#,
    );
    assert!(result.success, "{:?}", result.error);
    let before = result.before.expect("before snapshot");
    let after = result.after.expect("after snapshot");
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
fn inline_debug_preserves_long_constructs() {
    let value = "ResourceLen(SemanticResourceRef(ResourceIdentifierThatMustRemainVisible))";
    let rendered = inline_debug(&value);
    assert!(rendered.contains("ResourceIdentifierThatMustRemainVisible"));
    assert!(!rendered.contains('…'));
}
