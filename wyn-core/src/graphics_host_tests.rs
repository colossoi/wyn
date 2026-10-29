//! Focused graphics tests; broader host contracts use CLI/WHL fixtures.
#[test]
fn graphics_callbacks_preserve_scalar_helpers_for_later_optimization() {
    let source = format!(
        "{}\n{}",
        include_str!("../../scripts/playground_image_header.wyn"),
        include_str!("../../testfiles/playground/truncated_octahedra.wyn"),
    );
    let unified = crate::test_pipeline::compile_thru_unified_helpers(&source);
    assert!(unified.defs.iter().any(|def| {
        matches!(def.meta, crate::tlc::DefMeta::Function)
            && unified.symbols.get(def.name).is_some_and(|name| name.contains("clip_slab"))
    }));
    crate::tlc::extract_stages(unified).expect("extract graphics with scalar helpers");
}

#[test]
fn polymorphic_graphics_helpers_keep_ordered_target_versions() {
    let source = include_str!("../../tests/compiler/cases/polymorphic_graphics_versions.wyn");
    let unified = crate::test_pipeline::compile_thru_unified_helpers(source);
    let roots: Vec<_> = unified
        .defs
        .iter()
        .filter_map(|def| match &def.meta {
            crate::tlc::DefMeta::EntryPoint(entry) => Some(entry.declaration.entry_kind),
            _ => None,
        })
        .collect();
    assert_eq!(roots, [crate::interface::EntryKind::Root]);
    let stages = crate::tlc::extract_stages(unified).unwrap();
    let operations: Vec<_> = stages
        .defs
        .iter()
        .filter_map(|def| match &def.meta {
            crate::tlc::DefMeta::EntryPoint(entry) => {
                entry.declaration.graphics_group.as_ref().map(|group| group.operation)
            }
            _ => None,
        })
        .collect();
    assert_eq!(operations, [0, 0, 1, 1]);
}
