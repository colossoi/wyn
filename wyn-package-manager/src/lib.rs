mod local;
mod manifest;
mod materialize;
mod version;

pub use local::{find_build_input, prepare_package, prepare_standalone, BuildInput, PreparationError};
pub use manifest::{Dependency, DependencySource, Manifest, ManifestError, PackageName, PackageNameError};
pub use version::{PackageVersion, VersionError};
