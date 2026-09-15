//! Checked extraction and atomic column selection for one CFG snapshot.
use super::graph_ops::pack_result_values;
use super::ir::{EGraph, Family, FlowValueId, Language, ValueId, ValueKind};
use super::types::{by_value_function_result, PureOp, ResultDestination, WynLanguage};
use crate::flow::{BlockId, Terminator};
use crate::{LookupMap, LookupSet};
use smallvec::smallvec;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, Eq)]
pub(crate) enum IncomingEdge {
    Branch(BlockId),
    Then(BlockId, ValueId),
    Else(BlockId, ValueId),
}

impl PartialEq for IncomingEdge {
    fn eq(&self, other: &Self) -> bool {
        self.source() == other.source() && std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Hash for IncomingEdge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source().hash(state);
        std::mem::discriminant(self).hash(state);
    }
}

impl IncomingEdge {
    pub fn source(self) -> BlockId {
        match self {
            Self::Branch(source) | Self::Then(source, _) | Self::Else(source, _) => source,
        }
    }

    pub fn condition(self) -> Option<ValueId> {
        match self {
            Self::Branch(_) => None,
            Self::Then(_, condition) | Self::Else(_, condition) => Some(condition),
        }
    }

    pub fn arguments_mut<C, R>(self, term: &mut Terminator<C, FlowValueId, R>) -> &mut Vec<FlowValueId> {
        match (self, term) {
            (Self::Branch(_), Terminator::Branch { args, .. }) => args,
            (Self::Then(_, _), Terminator::CondBranch { then_args, .. }) => then_args,
            (Self::Else(_, _), Terminator::CondBranch { else_args, .. }) => else_args,
            _ => unreachable!("interface topology is immutable during column selection"),
        }
    }
}

pub(crate) type BlockInterface = wyn_graph::block_interface::Matrix<IncomingEdge, FlowValueId>;
pub(crate) type BlockInterfaces = LookupMap<BlockId, BlockInterface>;

pub(crate) fn extract<P: Family, Lang: Language>(
    graph: &EGraph<P, Lang>,
) -> Result<BlockInterfaces, String> {
    let mut rows =
        graph.skeleton.blocks.keys().map(|block| (block, Vec::new())).collect::<LookupMap<_, _>>();
    let check_value = |value: ValueId| {
        let node = graph.nodes.get(value).ok_or_else(|| format!("missing flow value {value:?}"))?;
        if !P::ALLOWS_MATERIALIZED_FLOW && Lang::contains_materialized_flow(node.ty()) {
            return Err(format!("materialized flow value {value:?} in physical CFG"));
        }
        Ok(())
    };
    for (source, block) in &graph.skeleton.blocks {
        let mut record = |edge: IncomingEdge, target, args: &[FlowValueId]| {
            if let Some(condition) = edge.condition() {
                check_value(condition)?;
            }
            for argument in args {
                check_value(argument.value())?;
            }
            rows.get_mut(&target)
                .ok_or_else(|| format!("branch from {source:?} targets an absent block {target:?}"))?
                .push((edge, args.to_vec()));
            Ok::<_, String>(())
        };
        match &block.term {
            Terminator::Branch { target, args } => record(IncomingEdge::Branch(source), *target, args)?,
            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                record(IncomingEdge::Then(source, *cond), *then_target, then_args)?;
                record(IncomingEdge::Else(source, *cond), *else_target, else_args)?;
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    }
    graph.skeleton.blocks.iter().map(|(block, body)| {
        for (slot, parameter) in body.params.iter().enumerate() {
            check_value(parameter.value())?;
            if !matches!(graph.nodes[parameter.value()].kind(), ValueKind::BlockParam { block: owner, index } if *owner == block && *index == slot) {
                return Err(format!("mismatched block parameter {:?} at {block:?}[{slot}]", parameter.value()));
            }
        }
        let interface = BlockInterface::new(body.params.iter().copied(), rows.remove(&block).unwrap())
            .map_err(|error| format!("invalid block interface {block:?}: {error:?}"))?;
        Ok((block, interface))
    }).collect()
}

/// Select all parameter lists before committing any parameter, edge, or node index.
/// The callback receives only the frozen interface; topology cannot change during the edit.
pub(crate) fn select_columns<P: Family, Lang: Language>(
    graph: &mut EGraph<P, Lang>,
    mut select: impl FnMut(BlockId, &BlockInterface) -> Result<Vec<usize>, String>,
) -> Result<Vec<ValueId>, String> {
    let interfaces = extract(graph)?;
    let mut removed = Vec::new();
    let mut parameters = Vec::new();
    let mut rows = Vec::new();
    for (block, interface) in interfaces {
        let selected = interface
            .select(select(block, &interface)?)
            .map_err(|error| format!("invalid column selection for {block:?}: {error:?}"))?;
        let retained = selected.parameters().collect::<LookupSet<_>>();
        removed.extend(
            interface
                .parameters()
                .filter(|parameter| !retained.contains(parameter))
                .map(FlowValueId::value),
        );
        parameters.push((block, selected.parameters().collect::<Vec<_>>()));
        rows.extend(selected.rows());
    }
    for (block, params) in parameters {
        for (index, parameter) in params.iter().enumerate() {
            graph.nodes[parameter.value()].kind = ValueKind::BlockParam { block, index };
        }
        graph.skeleton.blocks[block].params = params;
    }
    for (edge, arguments) in rows {
        *edge.arguments_mut(&mut graph.skeleton.blocks[edge.source()].term) = arguments;
    }
    Ok(removed)
}

pub(crate) fn dependencies(
    interface: &BlockInterface,
    slot: usize,
) -> impl Iterator<Item = (BlockId, ValueId)> + '_ {
    interface.edges().iter().zip(interface.columns()[slot].arguments()).flat_map(|(edge, value)| {
        std::iter::once((edge.source(), value.value()))
            .chain(edge.condition().map(|condition| (edge.source(), condition)))
    })
}

/// Expand product parameters into independent leaf columns. Existing value
/// references keep their product shape through a reconstructed result binding.
pub(crate) fn split_product_columns<P: Family>(graph: &mut EGraph<P, WynLanguage>) -> Result<(), String> {
    let interfaces = extract(graph)?;
    for (&block, interface) in &interfaces {
        graph.skeleton.blocks[block].params.clear();
        for parameter in interface.parameters() {
            let value = parameter.value();
            let abi = by_value_function_result::<WynLanguage>(graph.nodes[value].ty().clone());
            if abi.is_product() {
                let binding = abi.map_destinations(|ty, _| {
                    ResultDestination::ReturnValue(graph.add_block_param(block, ty.clone()))
                });
                let packed = pack_result_values(graph, &binding)?;
                graph.install_aliases([(value, packed)]);
            } else {
                let index = graph.skeleton.blocks[block].params.len();
                graph.nodes[value].kind = ValueKind::BlockParam { block, index };
                graph.skeleton.blocks[block].params.push(parameter);
            }
        }
    }
    for interface in interfaces.values() {
        for (edge, arguments) in interface.rows() {
            let mut leaves = Vec::new();
            for (parameter, argument) in interface.parameters().zip(arguments) {
                let value = graph.canonical_value(argument.value());
                let abi =
                    by_value_function_result::<WynLanguage>(graph.nodes[parameter.value()].ty().clone());
                for (path, _) in abi.destination_leaves_with_paths() {
                    let mut projected = value;
                    let mut field = abi.clone();
                    for &index in &path {
                        let Some(child) = field.field(index) else {
                            return Err("block argument projection has an invalid result path".into());
                        };
                        field = child;
                        // Keep the interface's declared field type even when a
                        // literal's internal buffer/size variables differ.
                        projected = graph.intern_pure(
                            PureOp::Project { index: index as u32 },
                            smallvec![projected],
                            field.ty().clone(),
                            None,
                        );
                    }
                    leaves.push(projected);
                }
            }
            let arguments = graph.admit_flow_values(leaves);
            *edge.arguments_mut(&mut graph.skeleton.blocks[edge.source()].term) = arguments;
        }
    }
    graph.canonicalize_boundary_operands();
    Ok(())
}

#[cfg(test)]
mod tests;
