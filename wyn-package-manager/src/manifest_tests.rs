use super::{Manifest, ManifestError, PackageName, PackageNameError};

const MANIFEST: &str = r#"
manifest-version = 1

[package]
name = "example/noise"
version = "v1.0.0"
wyn = "v0.1.0"
library = "src/lib.wyn"

[dependencies]
rng = { package = "wyn/rng", version = "v1.4.2", path = "../rng" }
"#;

#[test]
fn parses_the_minimal_local_manifest() {
    let manifest = Manifest::parse(MANIFEST).expect("valid manifest");

    assert_eq!(manifest.package().as_str(), "example/noise");
    assert_eq!(manifest.version().to_string(), "v1.0.0");
    assert_eq!(manifest.minimum_wyn().to_string(), "v0.1.0");
    assert_eq!(manifest.library().as_str(), "src/lib.wyn");
    let dependencies: Vec<_> = manifest.dependencies().collect();
    assert_eq!(dependencies.len(), 1);
    assert_eq!(dependencies[0].0.as_str(), "rng");
    assert_eq!(dependencies[0].1.package().as_str(), "wyn/rng");
    assert_eq!(dependencies[0].1.minimum().to_string(), "v1.4.2");
    assert_eq!(dependencies[0].1.path().to_string_lossy(), "../rng");
}

#[test]
fn rejects_unknown_fields() {
    let source = MANIFEST.replace(
        "library = \"src/lib.wyn\"",
        "library = \"src/lib.wyn\"\nextra = true",
    );
    assert!(matches!(Manifest::parse(&source), Err(ManifestError::Toml(_))));
}

#[test]
fn rejects_non_path_dependencies_in_the_local_stage() {
    let source = MANIFEST.replace("path = \"../rng\"", "git = \"https://example.invalid/rng\"");
    assert!(matches!(Manifest::parse(&source), Err(ManifestError::Toml(_))));
}

#[test]
fn package_names_are_lowercase_component_paths() {
    assert_eq!(
        PackageName::new("Example/noise"),
        Err(PackageNameError::InvalidCharacter { character: 'E' })
    );
    assert_eq!(
        PackageName::new("example//noise"),
        Err(PackageNameError::EmptyComponent)
    );
    assert_eq!(
        PackageName::new("example/noise").expect("valid name").as_str(),
        "example/noise"
    );
}
