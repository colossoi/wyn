use crate::{LocalSources, PackageGraph};

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

    pub const fn package_graph(&self) -> &PackageGraph {
        &self.package_graph
    }

    /// Separate retained package metadata from the source reader.
    pub fn into_parts(self) -> (PackageGraph, S) {
        (self.package_graph, self.sources)
    }
}
