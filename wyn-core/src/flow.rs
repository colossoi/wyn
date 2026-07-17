//! Target- and representation-neutral control-flow vocabulary shared by IRs.

use slotmap::new_key_type;

new_key_type! {
    /// Opaque identity of a control-flow block.
    pub struct BlockId;
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
