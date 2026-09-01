use super::*;

fn reported_imports(source: &str) -> Vec<(ImportSiteId, ImportTarget, TextRange)> {
    let mut node_ids = NodeCounter::new();
    let mut frontend = WynFrontend::new(&mut node_ids, CompilerOptions::default());
    let mut imports = Vec::new();
    frontend
        .parse(ModuleId::from(7), source, &mut |site, target, range| {
            imports.push((site, target, range));
        })
        .expect("source should parse");
    imports
}

#[test]
fn reports_every_import_form_in_source_order() {
    let source = concat!(
        "import \"first\"\n",
        "module Nested = { import \"second\" }\n",
        "module Bound = import \"third\"\n",
    );
    let imports = reported_imports(source);

    assert_eq!(imports.len(), 3);
    for (index, (site, _, range)) in imports.iter().enumerate() {
        assert_eq!(*site, ImportSiteId::from(index as u32));
        assert!(range.start() < range.end());
        assert!(source[range.start() as usize..range.end() as usize].starts_with("import"));
    }
}

#[test]
fn decodes_local_and_package_imports() {
    let imports = reported_imports(concat!(
        "import \"local/module\"\n",
        "module Root = import \"pkg:rng\"\n",
        "module Child = import \"pkg:rng/algorithm/xoshiro\"\n",
    ));

    assert!(matches!(
        &imports[0].1,
        ImportTarget::Local(path) if path.as_str() == "local/module.wyn"
    ));
    assert!(matches!(
        &imports[1].1,
        ImportTarget::Dependency { alias, module: None } if alias.as_str() == "rng"
    ));
    assert!(matches!(
        &imports[2].1,
        ImportTarget::Dependency { alias, module: Some(path) }
            if alias.as_str() == "rng" && path.as_str() == "algorithm/xoshiro.wyn"
    ));
}
