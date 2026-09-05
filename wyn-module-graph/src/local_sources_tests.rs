use super::*;
use crate::{ModulePath, PackageId};
use std::io;

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

#[test]
fn local_source_errors_hide_host_paths_by_default() {
    let module = ModuleKey::new(
        PackageId::from(0),
        ModulePath::new("src/lib.wyn").expect("valid path"),
    );
    let secret = PathBuf::from("C:/private/cache/package/src/lib.wyn");
    let error = LocalSourceError::ReadModule {
        module,
        path: secret.clone(),
        source: io::Error::new(io::ErrorKind::PermissionDenied, "access denied"),
    };
    let message = error.to_string();

    assert_eq!(message, "failed to read source module: access denied");
    assert!(!message.contains(&secret.to_string_lossy().to_string()));
}
