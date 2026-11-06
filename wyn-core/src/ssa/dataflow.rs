//! SSA occurrences in the shared execution graph. Clients select storage events;
//! empty instruction runs are contracted without dropping control-flow edges.
use super::types::{BlockId, FuncBody, InstId};
use crate::LookupMap;
use egglog_engine::{EGraph, Value, Write};

pub(super) struct Dataflow {
    pub graph: EGraph,
    pub blocks: LookupMap<BlockId, (Value, Value)>,
    pub instructions: LookupMap<InstId, (Value, Value)>,
    pub terminators: LookupMap<BlockId, (Value, Value)>,
    pub edges: LookupMap<(BlockId, usize), Value>,
}

impl Dataflow {
    pub fn new(
        body: &FuncBody,
        events: &LookupMap<BlockId, Vec<Vec<InstId>>>,
    ) -> Result<Self, egglog_engine::Error> {
        let mut graph = EGraph::default();
        graph.parse_and_run_program(None, include_str!("../egglog/flow.egg"))?;
        let mut blocks = LookupMap::default();
        let mut instructions = LookupMap::default();
        let mut terminators = LookupMap::default();
        let mut edges = LookupMap::default();
        graph.update(|mut sink| {
            let mut next = 0i64;
            for (block, _) in &body.inner.blocks {
                let entry = sink.add("FlowNode", next)?;
                next += 1;
                let mut before = entry;
                for event in &events[&block] {
                    let after = sink.add("FlowNode", next)?;
                    next += 1;
                    sink.add("FlowNext", (before, after))?;
                    for &inst in event {
                        instructions.insert(inst, (before, after));
                    }
                    before = after;
                }
                let exit = sink.add("FlowNode", next)?;
                next += 1;
                sink.add("FlowNext", (before, exit))?;
                terminators.insert(block, (before, exit));
                blocks.insert(block, (entry, exit));
            }
            for (block, data) in &body.inner.blocks {
                for (index, (target, _)) in data.term.edges().into_iter().enumerate() {
                    let edge = sink.add("FlowNode", next)?;
                    next += 1;
                    sink.add("FlowNext", (blocks[&block].1, edge))?;
                    sink.add("FlowNext", (edge, blocks[&target].0))?;
                    sink.add("FlowEdge", (edge, blocks[&target].0))?;
                    edges.insert((block, index), edge);
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        Ok(Self {
            graph,
            blocks,
            instructions,
            terminators,
            edges,
        })
    }
}
