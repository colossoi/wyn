use super::*;
use crate::{ModulePath, PackageId};

#[test]
fn exact_memory_override_requires_no_package_root() {
    let module = ModuleKey::new(
        PackageId::from(0),
        ModulePath::new("main.wyn").expect("valid path"),
    );
    let mut sources = LocalSources::new();
    sources
        .add_override(module.clone(), "def value: i32 = 42")
        .expect("first override should be accepted");

    assert_eq!(
        sources.load(&module).expect("override should load").as_ref(),
        "def value: i32 = 42"
    );
}

#[test]
fn duplicate_memory_override_is_rejected() {
    let module = ModuleKey::new(
        PackageId::from(0),
        ModulePath::new("main.wyn").expect("valid path"),
    );
    let mut sources = LocalSources::new();
    sources.add_override(module.clone(), "first").expect("first override should be accepted");

    assert!(matches!(
        sources.add_override(module.clone(), "second"),
        Err(LocalSourceError::DuplicateOverride { module: duplicate }) if duplicate == module
    ));
}
