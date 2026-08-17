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
fn output_realization_records_route_writers() {
    let result = inspect_pass_impl(
        r#"
entry main(xs: [4]i32) [4]i32 =
  map(|x: i32| x + 1, xs)
"#,
        InspectPass::RealizeOutputs,
    );
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.pass, InspectPass::REALIZE_OUTPUTS);
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
        "output realization records the semantic values that publish the slot"
    );
}

#[test]
fn output_realization_exposes_runtime_filter_resource_changes() {
    let result = inspect_pass_impl(
        r#"
entry evens(xs: []i32) []i32 =
  filter(|x: i32| x % 2 == 0, xs)
"#,
        InspectPass::RealizeOutputs,
    );
    assert!(result.success, "{:?}", result.error);
    let before = result.before.expect("before snapshot");
    let after = result.after.expect("after snapshot");
    let before_filter = before
        .nodes
        .iter()
        .find(|node| node.variant == "filter")
        .and_then(|node| node.operation.as_ref())
        .and_then(|operation| operation.filter_state.as_ref())
        .expect("converted filter state");
    let after_filter = after
        .nodes
        .iter()
        .find(|node| node.variant == "filter")
        .and_then(|node| node.operation.as_ref())
        .and_then(|operation| operation.filter_state.as_ref())
        .expect("realized filter state");
    let before_scratch = before_filter.storage.scratch.as_ref().expect("converted scratch");
    assert_eq!(
        before_filter.storage.length.as_ref().map(|length| length.variant.as_str()),
        Some("view_only")
    );
    assert_eq!(
        after_filter.storage.length.as_ref().map(|length| length.variant.as_str()),
        Some("stored")
    );
    assert_eq!(
        after_filter.storage.length.as_ref().and_then(|length| length.resource.as_ref()),
        Some(before_scratch),
        "the old compaction buffer becomes the stored length cell"
    );

    let after_entry = after.groups.iter().find(|group| group.kind == "entry").unwrap();
    let output = &after_entry.outputs[0];
    assert_eq!(after_filter.storage.scratch.as_ref(), output.resource.as_ref());
    assert_eq!(
        output.kind.length.as_ref().map(|length| length.variant.as_str()),
        Some("like_input")
    );

    let length_resource = after
        .resources
        .iter()
        .find(|resource| &resource.id == before_scratch)
        .expect("stored length resource remains in the arena");
    assert_eq!(length_resource.elem_ty, "u32");
    assert_eq!(length_resource.size.variant, "fixed_bytes");
    assert_eq!(length_resource.size.bytes, Some(4));
    assert!(after_entry.resource_declarations.iter().any(|declaration| {
        &declaration.resource == before_scratch
            && declaration.elem_ty == "u32"
            && declaration.size.variant == "fixed_bytes"
            && declaration.size.bytes == Some(4)
    }));
}

#[test]
fn inline_debug_preserves_long_constructs() {
    let value = "ResourceLen(SemanticResourceRef(ResourceIdentifierThatMustRemainVisible))";
    let rendered = inline_debug(&value);
    assert!(rendered.contains("ResourceIdentifierThatMustRemainVisible"));
    assert!(!rendered.contains('…'));
}
