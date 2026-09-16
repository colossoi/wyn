use super::super::blocks::Exit;
use super::{AssociatedData, BlockId, FunctionKind, Instruction, Storage};
use std::collections::BTreeSet;

pub(super) fn reachable(data: &AssociatedData) -> BTreeSet<BlockId> {
    let mut pending: Vec<_> = data
        .blocks
        .iter()
        .filter_map(|(&id, block)| {
            block.interface.as_ref().is_some_and(|f| matches!(f.kind, FunctionKind::Entry(_))).then_some(id)
        })
        .collect();
    let mut live = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !live.insert(id) {
            continue;
        }
        let block = &data.blocks[id];
        match &block.exit {
            Exit::Return(_) => {}
            Exit::Jump(edge) => pending.push(edge.target),
            Exit::Branch { yes, no, .. } => pending.extend([yes.target, no.target]),
        }
        for instruction in &data.bodies[block.body].instructions {
            match instruction {
                Instruction::Call { function, .. } => pending.push(*function),
                Instruction::Dispatch(id) => pending.push(data.dispatches[*id].kernel),
                _ => {}
            }
        }
    }
    live
}

pub(super) fn program(data: &AssociatedData) -> String {
    let mut text = include_str!("../blocks.egg").to_owned();
    let mut dispatches = BTreeSet::new();
    let mut buffers = BTreeSet::new();
    for id in reachable(data) {
        let block = &data.blocks[id];
        let b = id.as_u32();
        if let Some(exit) = block.loop_exit {
            text.push_str(&format!("(LoopExit (BlockId {b}) (BlockId {}))\n", exit.as_u32()));
        }
        for region in &block.source_regions {
            text.push_str(&format!(
                "(BlockSourceRegion (BlockId {b}) {})\n",
                region.egglog()
            ));
        }
        if let Some(function) = &block.interface {
            text.push_str(&format!(
                "(Function (BlockId {b}) {} {})\n",
                block.parameters.len(),
                function.results
            ));
            match function.kind {
                FunctionKind::Entry(entry) => text.push_str(&format!(
                    "(Entry {} (BlockId {b}))\n(Host (BlockId {b}))\n",
                    entry.as_u32()
                )),
                FunctionKind::Host => text.push_str(&format!("(Host (BlockId {b}))\n")),
                FunctionKind::Device => text.push_str(&format!("(Device (BlockId {b}))\n")),
                FunctionKind::Kernel([x, y, z]) => {
                    text.push_str(&format!("(Kernel (BlockId {b}) {x} {y} {z})\n"))
                }
            }
        }
        text.push_str(&format!(
            "(Block (BlockId {b}) (BlockId {}) (BodyId {}) {})\n",
            block.function.as_u32(),
            block.body.as_u32(),
            block.parameters.len()
        ));
        match &block.exit {
            Exit::Return(body) => {
                text.push_str(&format!("(Return (BlockId {b}) (BodyId {}))\n", body.as_u32()))
            }
            Exit::Jump(edge) => text.push_str(&format!(
                "(Jump (BlockId {b}) (BlockId {}) (BodyId {}))\n",
                edge.target.as_u32(),
                edge.arguments.as_u32()
            )),
            Exit::Branch { condition, yes, no } => text.push_str(&format!(
                "(Branch (BlockId {b}) (BodyId {}) (BlockId {}) (BodyId {}) (BlockId {}) (BodyId {}))\n",
                condition.as_u32(),
                yes.target.as_u32(),
                yes.arguments.as_u32(),
                no.target.as_u32(),
                no.arguments.as_u32()
            )),
        }
        for (i, instruction) in data.bodies[block.body].instructions.iter().enumerate() {
            match instruction {
                Instruction::Call { function, .. } => text.push_str(&format!(
                    "(Call (BlockId {b}) {i} (BlockId {}))\n",
                    function.as_u32()
                )),
                Instruction::Dispatch(id) => {
                    dispatches.insert(*id);
                    text.push_str(&format!(
                        "(Launch (BlockId {b}) {i} (DispatchId {}))\n",
                        id.as_u32()
                    ));
                }
                Instruction::Allocate(id) => {
                    buffers.insert(*id);
                    text.push_str(&format!(
                        "(Allocate (BlockId {b}) {i} (BufferId {}))\n",
                        id.as_u32()
                    ));
                }
                _ => {}
            }
        }
    }
    for id in dispatches {
        let d = id.as_u32();
        let dispatch = &data.dispatches[id];
        text.push_str(&format!(
            "(Dispatch (DispatchId {d}) (BlockId {}) (GridId {}))\n",
            dispatch.kernel.as_u32(),
            dispatch.grid.as_u32()
        ));
        for before in &dispatch.dependencies {
            text.push_str(&format!(
                "(After (DispatchId {d}) (DispatchId {}))\n",
                before.as_u32()
            ));
        }
        for (role, ids) in [("Read", &dispatch.reads), ("Write", &dispatch.writes)] {
            for id in ids {
                buffers.insert(*id);
                text.push_str(&format!("({role} (DispatchId {d}) (BufferId {}))\n", id.as_u32()));
            }
        }
    }
    for id in buffers {
        let relation = match data.buffers[id].storage {
            Storage::Device => "DeviceBuffer",
            Storage::Function => "LocalBuffer",
            Storage::External(_) => "InputBuffer",
        };
        text.push_str(&format!("({relation} (BufferId {}))\n", id.as_u32()));
    }
    text
}
