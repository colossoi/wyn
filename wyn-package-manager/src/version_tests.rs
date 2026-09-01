use super::{PackageVersion, VersionError};

#[test]
fn versions_require_the_canonical_v_prefix() {
    assert!(matches!(
        PackageVersion::parse("1.2.3"),
        Err(VersionError::MissingPrefix)
    ));
    assert!(matches!(
        PackageVersion::parse("v1.2"),
        Err(VersionError::InvalidSemver(_))
    ));
}

#[test]
fn versions_order_by_semver_rules() {
    let prerelease = PackageVersion::parse("v1.4.0-rc.1").expect("valid prerelease");
    let release = PackageVersion::parse("v1.4.0").expect("valid release");
    let later = PackageVersion::parse("v1.5.0").expect("valid release");

    assert!(prerelease < release);
    assert!(release < later);
    assert!(prerelease.is_prerelease());
    assert_eq!(later.to_string(), "v1.5.0");
}

#[test]
fn minimums_must_share_a_major() {
    let minimum = PackageVersion::parse("v1.4.2").expect("valid minimum");
    assert!(PackageVersion::parse("v1.8.0").expect("valid release").satisfies(&minimum));
    assert!(!PackageVersion::parse("v1.4.1").expect("valid release").satisfies(&minimum));
    assert!(!PackageVersion::parse("v2.0.0").expect("valid release").satisfies(&minimum));
}

#[test]
fn build_metadata_is_rejected() {
    assert!(matches!(
        PackageVersion::parse("v1.2.3+local"),
        Err(VersionError::BuildMetadata)
    ));
}
