//! Forward aggregate fields and exact reconstructions, then reuse block-local
//! duplicates and discard dead aggregates. Operand evaluation stays intact.
use crate::{dr, spirv};
use std::collections::{HashMap, HashSet};

fn is_aggregate(instruction: &dr::Instruction) -> bool {
    matches!(
        instruction.class.opcode,
        spirv::Op::CompositeConstruct | spirv::Op::CompositeExtract
    )
}

fn replace(instruction: &mut dr::Instruction, replacements: &HashMap<spirv::Word, spirv::Word>) {
    for operand in &mut instruction.operands {
        if let dr::Operand::IdRef(value) = operand {
            if let Some(&replacement) = replacements.get(value) {
                *value = replacement;
            }
        }
    }
}

pub(super) fn run(module: &mut dr::Module) {
    // Decorations can affect value semantics. Keep decorated (and named)
    // definitions distinct; the use counts below also retain their targets.
    let protected: HashSet<_> = module
        .annotations
        .iter()
        .chain(&module.debug_names)
        .flat_map(|instruction| instruction.operands.iter())
        .filter_map(|operand| match operand {
            dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect();
    loop {
        let vector_widths: HashMap<_, _> = module
            .types_global_values
            .iter()
            .filter_map(|i| match (i.class.opcode, i.result_id, i.operands.as_slice()) {
                (spirv::Op::TypeVector, Some(id), [_, dr::Operand::LiteralBit32(n)]) => {
                    Some((id, *n as usize))
                }
                _ => None,
            })
            .collect();
        let value_types: HashMap<_, _> =
            module.all_inst_iter().filter_map(|i| Some((i.result_id?, i.result_type?))).collect();
        let definitions: HashMap<_, _> = module
            .all_inst_iter()
            .filter_map(|i| {
                if !is_aggregate(i)
                    || protected.contains(&i.result_id?)
                    || i.class.opcode == spirv::Op::CompositeConstruct
                        && i.result_type
                            .and_then(|ty| vector_widths.get(&ty))
                            .is_some_and(|&n| n != i.operands.len())
                {
                    return None;
                }
                Some((i.result_id?, i.clone()))
            })
            .collect();
        let mut replacements = HashMap::new();
        let mut changed = false;
        for function in &mut module.functions {
            for block in &mut function.blocks {
                let mut available = HashMap::new();
                block.instructions.retain_mut(|instruction| {
                    replace(instruction, &replacements);
                    let Some(result) = instruction.result_id else {
                        return true;
                    };
                    if !is_aggregate(instruction) || protected.contains(&result) {
                        return true;
                    }
                    if instruction.class.opcode == spirv::Op::CompositeExtract {
                        while let [dr::Operand::IdRef(source), dr::Operand::LiteralBit32(index), ..] =
                            instruction.operands.as_slice()
                        {
                            let Some(base) = definitions
                                .get(source)
                                .filter(|i| i.class.opcode == spirv::Op::CompositeConstruct)
                            else {
                                break;
                            };
                            let Some(field) = base.operands.get(*index as usize) else {
                                break;
                            };
                            instruction.operands[0] = field.clone();
                            instruction.operands.remove(1);
                            changed = true;
                        }
                        if let [dr::Operand::IdRef(value)] = instruction.operands.as_slice() {
                            replacements.insert(result, *value);
                            return false;
                        }
                    } else if definitions.contains_key(&result) {
                        let source = instruction
                            .operands
                            .iter()
                            .enumerate()
                            .try_fold(None, |source, (index, operand)| {
                                let dr::Operand::IdRef(id) = operand else {
                                    return None;
                                };
                                let projection = definitions
                                    .get(id)
                                    .filter(|i| i.class.opcode == spirv::Op::CompositeExtract)?;
                                let [dr::Operand::IdRef(original), dr::Operand::LiteralBit32(field)] =
                                    projection.operands.as_slice()
                                else {
                                    return None;
                                };
                                (*field as usize == index && source.is_none_or(|s| s == *original))
                                    .then_some(Some(*original))
                            })
                            .flatten();
                        if let Some(source) = source
                            .filter(|source| value_types.get(source) == instruction.result_type.as_ref())
                        {
                            replacements.insert(result, source);
                            return false;
                        }
                    }
                    let key = (
                        instruction.class.opcode,
                        instruction.result_type,
                        instruction.operands.clone(),
                    );
                    if let Some(&previous) = available.get(&key) {
                        replacements.insert(result, previous);
                        false
                    } else {
                        available.insert(key, result);
                        true
                    }
                });
            }
        }
        // Block order need not be dominance order, and phi operands can refer to
        // later blocks. Apply replacements to every use after discovery is complete.
        // Resolve chains created when an earlier-visited block used a later alias.
        for original in replacements.keys().copied().collect::<Vec<_>>() {
            let mut value = replacements[&original];
            while let Some(&next) = replacements.get(&value) {
                value = next;
            }
            replacements.insert(original, value);
        }
        for instruction in module.all_inst_iter_mut() {
            replace(instruction, &replacements);
        }
        if replacements.is_empty() && !changed {
            break;
        }
    }

    let mut uses: HashMap<_, usize> = HashMap::new();
    let mut aggregates = HashMap::new();
    for instruction in module.all_inst_iter() {
        let operands: Vec<_> = instruction
            .operands
            .iter()
            .filter_map(
                |operand| {
                    if let dr::Operand::IdRef(id) = operand {
                        Some(*id)
                    } else {
                        None
                    }
                },
            )
            .collect();
        for &operand in &operands {
            *uses.entry(operand).or_default() += 1;
        }
        if is_aggregate(instruction) {
            if let Some(result) = instruction.result_id {
                aggregates.insert(result, operands);
            }
        }
    }
    let mut pending: Vec<_> = aggregates.keys().copied().filter(|id| !uses.contains_key(id)).collect();
    let mut dead = HashSet::new();
    while let Some(value) = pending.pop() {
        if !dead.insert(value) {
            continue;
        }
        for operand in &aggregates[&value] {
            if let Some(count) = uses.get_mut(operand) {
                *count -= 1;
                if *count == 0 && aggregates.contains_key(operand) {
                    pending.push(*operand);
                }
            }
        }
    }
    for function in &mut module.functions {
        for block in &mut function.blocks {
            block
                .instructions
                .retain(|instruction| instruction.result_id.is_none_or(|id| !dead.contains(&id)));
        }
    }
}
