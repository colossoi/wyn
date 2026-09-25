//! Select small scheduled helpers for expansion during the SSA handoff.
use crate::egglog::{BlockId, Exit, ExprId, Instruction, Program, Scheduled, Value};
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
#[path = "inline_tests.rs"]
mod tests;

pub(in crate::egglog) fn run(data: &Program<Scheduled>) -> BTreeSet<BlockId> {
    let functions = data.state.blocks.iter().filter_map(|(&id, b)| b.interface.as_ref().map(|_| id));
    let Ok(order) = wyn_graph::topo_sort_by_dependencies(functions, |id, out| {
        for &block in &data.state.blocks[id].interface.as_ref().unwrap().blocks {
            for instruction in &data.state.bodies[data.state.blocks[block].body].instructions {
                if let Instruction::Call { function, .. } = instruction {
                    out.push(*function);
                }
            }
        }
    }) else {
        return BTreeSet::new();
    };
    let mut costs = BTreeMap::new();
    for id in order {
        if let Some(cost) = cost(data, id, &costs) {
            costs.insert(id, cost);
        }
    }
    costs.into_keys().collect()
}

fn cost(data: &Program<Scheduled>, id: BlockId, callees: &BTreeMap<BlockId, usize>) -> Option<usize> {
    let block = &data.state.blocks[id];
    if block.interface.as_ref()?.blocks.len() != 1 {
        return None;
    }
    let Exit::Return(result) = block.exit else {
        return None;
    };
    let mut values = data.state.bodies[result].results.clone();
    let mut cost = 0;
    for instruction in &data.state.bodies[block.body].instructions {
        match instruction {
            Instruction::BindParameter(_, v)
            | Instruction::BindExpression(_, v)
            | Instruction::BindResult(_, v) => values.push(v.clone()),
            Instruction::Call {
                function, arguments, ..
            } => {
                cost += callees.get(function).copied().unwrap_or(1);
                values.extend(arguments.iter().cloned());
            }
            Instruction::Evaluate(op) => {
                cost += 1;
                let mut operands = vec![];
                data.operations[*op].kind.operands(&mut operands, &mut vec![]);
                values.extend(operands.into_iter().map(Value::Source));
            }
            _ => return None,
        }
    }
    let mut seen = BTreeSet::<ExprId>::new();
    while let Some(value) = values.pop() {
        if cost > 128 {
            return None;
        }
        match value {
            Value::Source(e) if seen.insert(e) => {
                cost += 1;
                values.extend(data.expressions[e].kind.children().into_iter().map(Value::Source));
            }
            Value::Source(_) | Value::Local(_) | Value::Int(_) => {}
            Value::Tuple(items) | Value::Primitive(_, items) => {
                cost += 1;
                values.extend(items);
            }
            Value::Field(value, _) => {
                cost += 1;
                values.push(*value);
            }
            Value::Array(array) => array.for_each_value(&mut |e| values.push(Value::Source(e))),
            _ => return None,
        }
    }
    (cost <= 128).then_some(cost)
}
