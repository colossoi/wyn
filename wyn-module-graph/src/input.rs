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
    /// Physical roots retained by tools that need to navigate loaded sources.
    pub fn source_roots(&self) -> &std::collections::HashMap<crate::PackageId, std::path::PathBuf> {
        self.sources.package_roots()
    }

    /// Apply editor buffers to all matching materialized packages.
    pub fn with_file_sources(
        mut self,
        buffers: &std::collections::HashMap<std::path::PathBuf, String>,
    ) -> Result<Self, crate::LocalSourceError> {
        let roots = self.sources.package_roots().clone();
        for (package, root) in roots {
            for (path, text) in buffers {
                if let Ok(relative) = path.strip_prefix(&root) {
                    if let Some(relative) = relative.to_str() {
                        if let Ok(path) = ModulePath::new(relative) {
                            self.sources
                                .add_override(crate::ModuleKey::new(package, path), text.as_str())?;
                        }
                    }
                }
            }
        }
        Ok(self)
    }

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
