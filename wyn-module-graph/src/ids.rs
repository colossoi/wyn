/// Identifies one package within a compilation plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageId(u32);

impl From<u32> for PackageId {
    fn from(raw: u32) -> Self {
        Self(raw)
    }
}

/// Identifies one physical source module within a module graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleId(u32);

impl From<u32> for ModuleId {
    fn from(raw: u32) -> Self {
        Self(raw)
    }
}

/// Identifies one import expression within a source module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImportSiteId(u32);

impl ImportSiteId {
    /// Construct a file-local import-site ID assigned by a frontend.
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }
}
