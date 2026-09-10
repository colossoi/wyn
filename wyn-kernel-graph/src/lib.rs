//! Owned kernel expansion topology. Bodies, dispatch and host ABI belong to callers.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::hash::Hash;
use wyn_base::LookupMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(u32);

impl From<u32> for KernelId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl KernelId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("kernel {0:?} belongs to more than one fragment")]
    DuplicateKernel(KernelId),
    #[error("unknown kernel {0:?}")]
    UnknownKernel(KernelId),
    #[error("unknown stage")]
    UnknownStage,
    #[error("stage is already registered")]
    DuplicateStage,
    #[error("stage is already bound")]
    AlreadyBound,
    #[error("stage has no completed fragment")]
    UnboundStage,
    #[error("unknown publication group")]
    UnknownGroup,
    #[error("publication group is already registered")]
    DuplicateGroup,
    #[error("fragment composition must contain a kernel")]
    EmptyFragment,
    #[error("unknown fragment boundary")]
    UnknownBoundary,
    #[error("self dependency")]
    SelfDependency,
    #[error("kernel expansion graph contains a cycle")]
    Cycle,
    #[error("publication group graph contains a cycle")]
    PublicationCycle,
}

/// A path indexes children of sequence/parallel fragments. The empty path
/// names the enclosing fragment. Paths are scoped to their owned fragment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragmentEndpoint {
    Kernel(KernelId),
    Entry(Vec<usize>),
    Completion(Vec<usize>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Shape {
    Kernel(KernelId, u64),
    Sequence(Vec<Fragment>),
    Parallel(Vec<Fragment>),
}

/// A complete, checked expansion with structurally derived boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fragment {
    shape: Shape,
    edges: Vec<(FragmentEndpoint, FragmentEndpoint)>,
}

impl Fragment {
    pub fn kernel(id: KernelId, source_rank: u64) -> Self {
        Self {
            shape: Shape::Kernel(id, source_rank),
            edges: Vec::new(),
        }
    }
    pub fn sequence(children: Vec<Self>) -> Result<Self, Error> {
        Self::compose(Shape::Sequence(children))
    }
    pub fn parallel(children: Vec<Self>) -> Result<Self, Error> {
        Self::compose(Shape::Parallel(children))
    }
    fn compose(shape: Shape) -> Result<Self, Error> {
        let fragment = Self {
            shape,
            edges: Vec::new(),
        };
        fragment.compile()?;
        Ok(fragment)
    }
    /// Add an edge transactionally, including edges to nested boundaries.
    pub fn add_dependency(
        &mut self,
        before: FragmentEndpoint,
        after: FragmentEndpoint,
    ) -> Result<(), Error> {
        let mut candidate = self.clone();
        candidate.edges.push((before, after));
        candidate.compile()?;
        *self = candidate;
        Ok(())
    }
    fn compile(&self) -> Result<Topology, Error> {
        let mut topology = Topology::default();
        self.append(&mut topology)?;
        topology.validate()?;
        Ok(topology)
    }
    fn append(&self, topology: &mut Topology) -> Result<(usize, usize), Error> {
        fn build(
            fragment: &Fragment,
            topology: &mut Topology,
            path: Vec<usize>,
            boundaries: &mut BTreeMap<Vec<usize>, (usize, usize)>,
        ) -> Result<(usize, usize), Error> {
            let entry = topology.boundary();
            let completion = topology.boundary();
            boundaries.insert(path.clone(), (entry, completion));
            match &fragment.shape {
                Shape::Kernel(id, rank) => {
                    let kernel = topology.kernel(*id, *rank)?;
                    topology.edge(entry, kernel)?;
                    topology.edge(kernel, completion)?;
                }
                Shape::Sequence(children) | Shape::Parallel(children) => {
                    if children.is_empty() {
                        return Err(Error::EmptyFragment);
                    }
                    let mut previous = entry;
                    for (index, child) in children.iter().enumerate() {
                        let mut child_path = path.clone();
                        child_path.push(index);
                        let (start, finish) = build(child, topology, child_path, boundaries)?;
                        topology.edge(previous, start)?;
                        if matches!(fragment.shape, Shape::Sequence(_)) {
                            previous = finish;
                        } else {
                            topology.edge(finish, completion)?;
                        }
                    }
                    if matches!(fragment.shape, Shape::Sequence(_)) {
                        topology.edge(previous, completion)?;
                    }
                }
            }
            for (before, after) in &fragment.edges {
                let endpoint = |endpoint: &FragmentEndpoint| -> Result<usize, Error> {
                    match endpoint {
                        FragmentEndpoint::Kernel(id) => {
                            topology.kernels.get(id).copied().ok_or(Error::UnknownKernel(*id))
                        }
                        FragmentEndpoint::Entry(relative) | FragmentEndpoint::Completion(relative) => {
                            let mut absolute = path.clone();
                            absolute.extend(relative);
                            let &(start, finish) =
                                boundaries.get(&absolute).ok_or(Error::UnknownBoundary)?;
                            Ok(if matches!(endpoint, FragmentEndpoint::Entry(_)) { start } else { finish })
                        }
                    }
                };
                let (before, after) = (endpoint(before)?, endpoint(after)?);
                topology.edge(before, after)?;
            }
            Ok((entry, completion))
        }
        build(self, topology, Vec::new(), &mut BTreeMap::new())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Topology {
    nodes: Vec<Option<(KernelId, u64)>>,
    dependencies: Vec<BTreeSet<usize>>,
    kernels: BTreeMap<KernelId, usize>,
}

impl Topology {
    fn boundary(&mut self) -> usize {
        let id = self.nodes.len();
        self.nodes.push(None);
        self.dependencies.push(BTreeSet::new());
        id
    }
    fn kernel(&mut self, id: KernelId, rank: u64) -> Result<usize, Error> {
        if self.kernels.contains_key(&id) {
            return Err(Error::DuplicateKernel(id));
        }
        let node = self.boundary();
        self.nodes[node] = Some((id, rank));
        self.kernels.insert(id, node);
        Ok(node)
    }
    fn edge(&mut self, before: usize, after: usize) -> Result<(), Error> {
        if before == after {
            return Err(Error::SelfDependency);
        }
        self.dependencies[after].insert(before);
        Ok(())
    }
    fn validate(&self) -> Result<(), Error> {
        wyn_graph::topo_sort_by_dependencies(0..self.nodes.len(), |node, out| {
            out.extend(&self.dependencies[node])
        })
        .map(|_| ())
        .map_err(|_| Error::Cycle)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint<S> {
    Kernel(KernelId),
    StageEntry(S),
    StageCompletion(S),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage<G> {
    entry: usize,
    completion: usize,
    bound: bool,
    group: Option<G>,
    rank: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Group {
    rank: u64,
    coalesce: bool,
}

/// Register identities first; install each stage once, in any order.
/// Every fallible mutation validates a private candidate before committing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Builder<S, R, G> {
    topology: Topology,
    stages: BTreeMap<S, Stage<G>>,
    groups: BTreeMap<G, Group>,
    membership: BTreeMap<KernelId, S>,
    resource_edges: Vec<(R, KernelId, KernelId)>,
}

impl<S, R, G> Default for Builder<S, R, G> {
    fn default() -> Self {
        Self {
            topology: Topology::default(),
            stages: BTreeMap::new(),
            groups: BTreeMap::new(),
            membership: BTreeMap::new(),
            resource_edges: Vec::new(),
        }
    }
}

impl<S, R, G> Builder<S, R, G>
where
    S: Copy + Ord + Hash + Debug,
    R: Copy + Ord + Debug,
    G: Copy + Ord + Hash + Debug,
{
    pub fn register_group(
        &mut self,
        group: G,
        source_rank: u64,
        coalesce_connected: bool,
    ) -> Result<(), Error> {
        if self.groups.contains_key(&group) {
            return Err(Error::DuplicateGroup);
        }
        self.groups.insert(
            group,
            Group {
                rank: source_rank,
                coalesce: coalesce_connected,
            },
        );
        Ok(())
    }
    pub fn register_stage(&mut self, stage: S, group: Option<G>, source_rank: u64) -> Result<(), Error> {
        if self.stages.contains_key(&stage) {
            return Err(Error::DuplicateStage);
        }
        if group.is_some_and(|group| !self.groups.contains_key(&group)) {
            return Err(Error::UnknownGroup);
        }
        let entry = self.topology.boundary();
        let completion = self.topology.boundary();
        // This provisional path permits cycle checking before binding. It is
        // replaced by the complete expansion when the stage is installed.
        self.topology.edge(entry, completion)?;
        self.stages.insert(
            stage,
            Stage {
                entry,
                completion,
                bound: false,
                group,
                rank: source_rank,
            },
        );
        Ok(())
    }
    pub fn bind_stage(&mut self, stage: S, fragment: Fragment) -> Result<(), Error> {
        let info = self.stages.get(&stage).ok_or(Error::UnknownStage)?;
        if info.bound {
            return Err(Error::AlreadyBound);
        }
        let mut candidate = self.clone();
        let (entry, completion) = fragment.append(&mut candidate.topology)?;
        candidate.topology.dependencies[info.completion].remove(&info.entry);
        candidate.topology.edge(info.entry, entry)?;
        candidate.topology.edge(completion, info.completion)?;
        candidate.topology.validate()?;
        for &kernel in candidate.topology.kernels.keys() {
            candidate.membership.entry(kernel).or_insert(stage);
        }
        candidate.stages.get_mut(&stage).ok_or(Error::UnknownStage)?.bound = true;
        *self = candidate;
        Ok(())
    }
    pub fn stage_of(&self, kernel: KernelId) -> Option<S> {
        self.membership.get(&kernel).copied()
    }
    fn endpoint(&self, endpoint: Endpoint<S>) -> Result<usize, Error> {
        match endpoint {
            Endpoint::Kernel(id) => self.topology.kernels.get(&id).copied().ok_or(Error::UnknownKernel(id)),
            Endpoint::StageEntry(id) => {
                self.stages.get(&id).map(|stage| stage.entry).ok_or(Error::UnknownStage)
            }
            Endpoint::StageCompletion(id) => {
                self.stages.get(&id).map(|stage| stage.completion).ok_or(Error::UnknownStage)
            }
        }
    }
    pub fn add_dependency(&mut self, before: Endpoint<S>, after: Endpoint<S>) -> Result<(), Error> {
        let (before, after) = (self.endpoint(before)?, self.endpoint(after)?);
        let mut candidate = self.topology.clone();
        candidate.edge(before, after)?;
        candidate.validate()?;
        self.topology = candidate;
        Ok(())
    }
    pub fn sequence_stages(&mut self, before: S, after: S) -> Result<(), Error> {
        if before == after {
            return Err(Error::SelfDependency);
        }
        self.add_dependency(Endpoint::StageCompletion(before), Endpoint::StageEntry(after))
    }
    /// Resource edges name actual writers/readers, independent of stage sequencing.
    pub fn connect_resource(
        &mut self,
        resource: R,
        writer: KernelId,
        reader: KernelId,
    ) -> Result<(), Error> {
        self.add_dependency(Endpoint::Kernel(writer), Endpoint::Kernel(reader))?;
        let edge = (resource, writer, reader);
        if !self.resource_edges.contains(&edge) {
            self.resource_edges.push(edge);
        }
        Ok(())
    }
    pub fn finalize(self) -> Result<Plan<S, R, G>, Error> {
        if self.stages.values().any(|stage| !stage.bound) {
            return Err(Error::UnboundStage);
        }
        self.topology.validate()?;
        let mut kernels = BTreeMap::new();
        for (&id, &node) in &self.topology.kernels {
            let mut dependencies = BTreeSet::new();
            let mut seen = BTreeSet::new();
            let mut pending = self.topology.dependencies[node].iter().copied().collect::<Vec<_>>();
            while let Some(predecessor) = pending.pop() {
                if !seen.insert(predecessor) {
                    continue;
                }
                if let Some((kernel, _)) = self.topology.nodes[predecessor] {
                    dependencies.insert(kernel);
                } else {
                    pending.extend(&self.topology.dependencies[predecessor]);
                }
            }
            let stage = self.membership[&id];
            kernels.insert(
                id,
                Kernel {
                    stage,
                    group: self.stages[&stage].group,
                    dependencies: dependencies.into_iter().collect(),
                },
            );
        }
        let mut preference = self.topology.kernels.keys().copied().collect::<Vec<_>>();
        preference.sort_by_key(|id| {
            (
                self.stages[&kernels[id].stage].rank,
                self.topology.nodes[self.topology.kernels[id]].map(|(_, rank)| rank),
                *id,
            )
        });
        let order = wyn_graph::topo_sort_by_dependencies(preference, |id, out| {
            out.extend(&kernels[&id].dependencies)
        })
        .map_err(|_| Error::Cycle)?;

        let mut group_ids = self.groups.keys().copied().collect::<Vec<_>>();
        group_ids.sort_by_key(|id| (self.groups[id].rank, *id));
        let indices =
            group_ids.iter().enumerate().map(|(index, &id)| (id, index)).collect::<LookupMap<_, _>>();
        let mut sets = wyn_graph::DisjointSets::new(group_ids.len());
        for kernel in kernels.values() {
            for dependency in &kernel.dependencies {
                if let (Some(a), Some(b)) = (kernel.group, kernels[dependency].group) {
                    if self.groups[&a].coalesce && self.groups[&b].coalesce {
                        sets.merge(indices[&a], indices[&b]);
                    }
                }
            }
        }
        let mut representatives = BTreeMap::new();
        let mut mapping = BTreeMap::new();
        let mut groups = BTreeMap::<G, PublicationGroup<G>>::new();
        for (index, &id) in group_ids.iter().enumerate() {
            let root = sets.representative(index);
            let canonical = *representatives.entry(root).or_insert(id);
            mapping.insert(id, canonical);
            groups
                .entry(canonical)
                .or_insert_with(|| PublicationGroup {
                    id: canonical,
                    members: Vec::new(),
                    kernels: Vec::new(),
                    dependencies: Vec::new(),
                })
                .members
                .push(id);
        }
        for &id in &order {
            let kernel = kernels.get_mut(&id).ok_or(Error::UnknownKernel(id))?;
            if let Some(group) = kernel.group {
                let group = mapping[&group];
                kernel.group = Some(group);
                groups.get_mut(&group).ok_or(Error::UnknownGroup)?.kernels.push(id);
            }
        }
        // Follow unpublished work as well: hiding a stage cannot hide a
        // publication dependency or a cycle introduced by contracting groups.
        for (&id, kernel) in &kernels {
            let Some(group) = kernel.group else {
                continue;
            };
            let mut pending = kernels[&id].dependencies.clone();
            let mut seen = BTreeSet::new();
            while let Some(dependency) = pending.pop() {
                if !seen.insert(dependency) {
                    continue;
                }
                match kernels[&dependency].group {
                    Some(other) if other != group => {
                        let deps = &mut groups.get_mut(&group).ok_or(Error::UnknownGroup)?.dependencies;
                        if !deps.contains(&other) {
                            deps.push(other);
                        }
                    }
                    _ => pending.extend(&kernels[&dependency].dependencies),
                }
            }
        }
        let preferred_groups = group_ids.into_iter().filter(|id| groups.contains_key(id));
        let group_order = wyn_graph::topo_sort_by_dependencies(preferred_groups, |id, out| {
            out.extend(&groups[&id].dependencies)
        })
        .map_err(|_| Error::PublicationCycle)?;
        let groups = group_order
            .into_iter()
            .map(|id| groups.remove(&id).ok_or(Error::UnknownGroup))
            .collect::<Result<_, _>>()?;
        let mut resource_edges = self.resource_edges;
        resource_edges.sort_unstable();
        Ok(Plan {
            order,
            kernels,
            groups,
            resource_edges,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel<S, G> {
    stage: S,
    group: Option<G>,
    dependencies: Vec<KernelId>,
}
impl<S: Copy, G: Copy> Kernel<S, G> {
    pub fn stage(&self) -> S {
        self.stage
    }
    pub fn group(&self) -> Option<G> {
        self.group
    }
    pub fn dependencies(&self) -> &[KernelId] {
        &self.dependencies
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicationGroup<G> {
    id: G,
    members: Vec<G>,
    kernels: Vec<KernelId>,
    dependencies: Vec<G>,
}
impl<G: Copy> PublicationGroup<G> {
    pub fn id(&self) -> G {
        self.id
    }
    pub fn members(&self) -> &[G] {
        &self.members
    }
    pub fn kernels(&self) -> &[KernelId] {
        &self.kernels
    }
    pub fn dependencies(&self) -> &[G] {
        &self.dependencies
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan<S, R, G> {
    order: Vec<KernelId>,
    kernels: BTreeMap<KernelId, Kernel<S, G>>,
    groups: Vec<PublicationGroup<G>>,
    resource_edges: Vec<(R, KernelId, KernelId)>,
}
impl<S: Copy, R, G: Copy> Plan<S, R, G> {
    pub fn kernel_order(&self) -> &[KernelId] {
        &self.order
    }
    pub fn kernel(&self, id: KernelId) -> Option<&Kernel<S, G>> {
        self.kernels.get(&id)
    }
    pub fn groups(&self) -> &[PublicationGroup<G>] {
        &self.groups
    }
    pub fn resource_edges(&self) -> &[(R, KernelId, KernelId)] {
        &self.resource_edges
    }
}

#[cfg(test)]
mod tests;
