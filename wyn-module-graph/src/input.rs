use std::sync::Arc;

use crate::{LocalSources, ModulePath, PackageGraph, PackageIdentity};

/// A closed package graph paired with access to its materialized sources.
///
/// Package management produces this value. A language frontend consumes it to
/// construct the syntax-bearing module graph.
#[derive(Debug)]
pub struct PackagePlan<S = LocalSources> {
    pub(crate) package_graph: PackageGraph,
    pub(crate) sources: S,
}

impl<S> PackagePlan<S> {
    pub const fn new(package_graph: PackageGraph, sources: S) -> Self {
        Self {
            package_graph,
            sources,
        }
    }
}

impl PackagePlan {
    /// Construct a complete in-memory plan containing one source module.
    pub fn single_source(
        identity: PackageIdentity,
        module: ModulePath,
        source: impl Into<Arc<str>>,
    ) -> Self {
        let (package_graph, root) = PackageGraph::single_package(identity, module);
        Self {
            package_graph,
            sources: LocalSources::from_override(root, source),
        }
    }

    /// Read the root module from memory instead of its package source tree.
    pub fn with_root_source(
        mut self,
        source: impl Into<Arc<str>>,
    ) -> Result<Self, crate::LocalSourceError> {
        self.sources.add_override(self.package_graph.root().clone(), source)?;
        Ok(self)
    }
}
