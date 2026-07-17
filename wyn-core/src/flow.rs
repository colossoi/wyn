//! Target- and representation-neutral control-flow vocabulary shared by IRs.

use slotmap::new_key_type;
use smallvec::{smallvec, SmallVec};

new_key_type! {
    /// Opaque identity of a control-flow block.
    pub struct BlockId;
}

/// A control-flow terminator parameterized by its value-reference type.
#[derive(Clone, Debug)]
pub enum Terminator<V> {
    Return(Option<V>),
    Branch {
        target: BlockId,
        args: Vec<V>,
    },
    CondBranch {
        cond: V,
        then_target: BlockId,
        then_args: Vec<V>,
        else_target: BlockId,
        else_args: Vec<V>,
    },
    Unreachable,
}

impl<V> Terminator<V> {
    pub fn successors(&self) -> SmallVec<[BlockId; 2]> {
        match self {
            Self::Return(_) | Self::Unreachable => SmallVec::new(),
            Self::Branch { target, .. } => smallvec![*target],
            Self::CondBranch {
                then_target,
                else_target,
                ..
            } => smallvec![*then_target, *else_target],
        }
    }

    pub fn visit_nodes_mut(&mut self, mut visit: impl FnMut(&mut V)) {
        match self {
            Self::Return(value) => value.iter_mut().for_each(visit),
            Self::Branch { args, .. } => args.iter_mut().for_each(visit),
            Self::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                visit(cond);
                then_args.iter_mut().for_each(&mut visit);
                else_args.iter_mut().for_each(visit);
            }
            Self::Unreachable => {}
        }
    }
}

impl<V: Copy> Terminator<V> {
    pub fn referenced_nodes(&self) -> SmallVec<[V; 8]> {
        match self {
            Self::Return(value) => value.iter().copied().collect(),
            Self::Branch { args, .. } => args.iter().copied().collect(),
            Self::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => std::iter::once(*cond)
                .chain(then_args.iter().copied())
                .chain(else_args.iter().copied())
                .collect(),
            Self::Unreachable => SmallVec::new(),
        }
    }

    pub fn for_each_value(&self, mut visit: impl FnMut(V)) {
        match self {
            Self::Return(value) => value.iter().copied().for_each(visit),
            Self::Branch { args, .. } => args.iter().copied().for_each(visit),
            Self::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                visit(*cond);
                then_args.iter().copied().for_each(&mut visit);
                else_args.iter().copied().for_each(visit);
            }
            Self::Unreachable => {}
        }
    }

    pub fn map_values(&self, mut map_value: impl FnMut(V) -> V) -> Self {
        self.try_map::<std::convert::Infallible>(|value| Ok(map_value(value)), Ok)
            .unwrap_or_else(|error| match error {})
    }

    pub fn try_map<E>(
        &self,
        mut map_value: impl FnMut(V) -> Result<V, E>,
        mut map_block: impl FnMut(BlockId) -> Result<BlockId, E>,
    ) -> Result<Self, E> {
        Ok(match self {
            Self::Return(value) => Terminator::Return(value.map(&mut map_value).transpose()?),
            Self::Branch { target, args } => Terminator::Branch {
                target: map_block(*target)?,
                args: args.iter().copied().map(map_value).collect::<Result<_, _>>()?,
            },
            Self::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => Terminator::CondBranch {
                cond: map_value(*cond)?,
                then_target: map_block(*then_target)?,
                then_args: then_args.iter().copied().map(&mut map_value).collect::<Result<_, _>>()?,
                else_target: map_block(*else_target)?,
                else_args: else_args.iter().copied().map(map_value).collect::<Result<_, _>>()?,
            },
            Self::Unreachable => Terminator::Unreachable,
        })
    }
}

/// Structured-control intent attached to a control-flow header block.
#[derive(Debug, Clone)]
pub enum ControlHeader {
    Loop {
        merge: BlockId,
        continue_block: BlockId,
    },
    Selection {
        merge: BlockId,
    },
}

impl ControlHeader {
    pub fn remap(&self, map_block: &impl Fn(BlockId) -> BlockId) -> Self {
        match self {
            Self::Loop {
                merge,
                continue_block,
            } => Self::Loop {
                merge: map_block(*merge),
                continue_block: map_block(*continue_block),
            },
            Self::Selection { merge } => Self::Selection {
                merge: map_block(*merge),
            },
        }
    }
}

/// Target execution environment for an entry point.
#[derive(Debug, Clone)]
pub enum ExecutionModel {
    Vertex,
    Fragment,
    Compute {
        local_size: (u32, u32, u32),
    },
}
