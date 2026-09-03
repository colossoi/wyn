mod local;
mod manifest;
mod materialize;
mod version;

pub use local::{prepare_package, prepare_standalone, PreparationError};
pub use manifest::{Dependency, DependencySource, Manifest, ManifestError, PackageName, PackageNameError};
pub use version::{PackageVersion, VersionError};
