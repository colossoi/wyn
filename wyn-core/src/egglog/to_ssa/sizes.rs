//! Publish host allocation sizes without materializing scalar expression graphs.
use super::plan::{Buffer, Stage};
use super::{error, Compiler, OptimizeError};
use crate::builtins::catalog;
use crate::egglog::to_ssa::interface;
use crate::host::{
    BufferLen, DispatchLen, DispatchSize, HostSizeInput, HostSizeScalar, IntegerOp, SizeExpr, SizeOp,
};
use crate::interface::EntryInputKind;
use crate::interface::StorageLayout;
use crate::op::BinaryOperator;
use crate::ssa::layout::block_layout;
use crate::ssa::layout::storage_elem_stride;
use crate::ssa::layout::type_byte_size;
use crate::tlc::SoacOp;
use crate::tlc::VarRef;
use crate::tlc::{ArrayExpr, TermKind};
use crate::types::{Type, TypeExt, TypeName};
use crate::LookupSet;
use egglog_engine::Value;

pub(super) fn capacity(compiler: &Compiler<'_, '_>, buffer: &Buffer) -> Result<BufferLen, OptimizeError> {
    let Some(stride) = storage_elem_stride(&interface::storage_type(&buffer.element)?) else {
        return Err(error("buffer element has no layout"));
    };
    let count = match extent(compiler, buffer.extent) {
        Ok(value) => value,
        Err(error) => {
            let mut inputs = Vec::new();
            extent_inputs(compiler, buffer.extent, &mut LookupSet::default(), &mut inputs)?;
            if inputs.is_empty() {
                return Err(error);
            }
            return Ok(BufferLen::HostProvided {
                inputs,
                elem_bytes: stride,
            });
        }
    };
    Ok(match count {
        SizeExpr::Integer(count) => BufferLen::Fixed {
            bytes: (count.max(1) as u64) * u64::from(stride),
        },
        SizeExpr::BufferLength {
            set,
            binding,
            stride: source_stride,
        } if stride <= 4 => BufferLen::LikeInput {
            set,
            binding,
            elem_bytes: stride,
            src_elem_bytes: source_stride,
        },
        value => BufferLen::Computed {
            // Runtime arrays require storage for at least one element, even when
            // their separately tracked logical length is zero.
            bytes: binary(
                SizeOp::Multiply,
                binary(SizeOp::Max, value, SizeExpr::Integer(1)),
                SizeExpr::Integer(stride.into()),
            ),
        },
    })
}
pub(super) fn dispatch(compiler: &Compiler<'_, '_>, stage: &Stage) -> Result<DispatchSize, OptimizeError> {
    if let Some((x, y, z)) = stage.grid {
        return Ok(DispatchSize::Fixed {
            x,
            y,
            z,
            explicit: true,
        });
    }
    let mut extent = stage.extent;
    let mut divisor = stage.width;
    while let Some((name, children)) = compiler.plan.extent(extent) {
        if name != "ChunkCount" {
            break;
        }
        divisor *= compiler.program.graph.value_to_base::<i64>(children[1]) as u32;
        extent = children[0];
    }
    let value = self::extent(compiler, extent);
    match value {
        Ok(SizeExpr::Integer(count)) => Ok(DispatchSize::Fixed {
            x: (count.max(0) as u32).div_ceil(divisor).clamp(1, 65_535),
            y: 1,
            z: 1,
            explicit: true,
        }),
        Ok(SizeExpr::BufferLength { set, binding, stride }) => Ok(DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding {
                set,
                binding,
                elem_bytes: stride,
            },
            workgroup_size: divisor,
        }),
        Ok(SizeExpr::Scalar(HostSizeInput::PushConstant {
            push_constant_offset, ..
        })) => Ok(DispatchSize::DerivedFrom {
            len: DispatchLen::PushConstant {
                offset: push_constant_offset,
            },
            workgroup_size: divisor,
        }),
        _ => {
            let Some(plan) = compiler.plan.group(stage.operation) else {
                return Err(error("host launch has no plan"));
            };
            let buffer = compiler
                .plan
                .members(plan)
                .into_iter()
                .flat_map(|operation| compiler.plan.resources(operation))
                .find_map(|resource| {
                    let backing = compiler.plan.backing(resource).unwrap_or(resource);
                    let buffer = compiler.plan.buffers.get(&backing)?;
                    (buffer.extent == extent).then_some(buffer)
                });
            let Some(buffer) = buffer else {
                return Err(error("host-computed launch has no capacity buffer"));
            };
            let Some(stride) = storage_elem_stride(&buffer.element) else {
                return Err(error("launch capacity has no layout"));
            };
            Ok(DispatchSize::DerivedFrom {
                len: DispatchLen::InputBinding {
                    set: buffer.binding.set,
                    binding: buffer.binding.binding,
                    elem_bytes: stride,
                },
                workgroup_size: divisor,
            })
        }
    }
}
pub(super) fn extent(compiler: &Compiler<'_, '_>, key: Value) -> Result<SizeExpr, OptimizeError> {
    let Some((name, children)) = compiler.plan.extent(key) else {
        return Err(error("host extent missing"));
    };
    match name {
        "Fixed" => Ok(SizeExpr::Integer(
            compiler.program.graph.value_to_base::<i64>(children[0]),
        )),
        "Length" | "Scalar" => {
            let Some(source) = compiler.plan.expr(children[0]) else {
                return Err(error("host extent identity missing"));
            };
            source_size(compiler, source, name == "Length")
        }
        "ChunkCount" => Ok(binary(
            SizeOp::Ceiling,
            extent(compiler, children[0])?,
            SizeExpr::Integer(compiler.program.graph.value_to_base::<i64>(children[1])),
        )),
        "Product" => Ok(binary(
            SizeOp::Multiply,
            extent(compiler, children[0])?,
            extent(compiler, children[1])?,
        )),
        "Stored" => {
            // The counter stores a logical length; its own one-element allocation
            // says nothing about the maximum capacity of the filtered array.
            let capacity = compiler.plan.capacity_sources(key).into_iter().find_map(|resource| {
                let backing = compiler.plan.backing(resource).unwrap_or(resource);
                let buffer = compiler.plan.buffers.get(&backing)?;
                (buffer.extent != key).then_some(buffer.extent)
            });
            let Some(capacity) = capacity else {
                return Err(error("stored logical length has no host-visible array capacity"));
            };
            extent(compiler, capacity)
        }
        _ => Err(error("unsupported host extent")),
    }
}
pub(super) fn source_size(
    compiler: &Compiler<'_, '_>,
    source: Value,
    length: bool,
) -> Result<SizeExpr, OptimizeError> {
    fn array_size(ty: &Type) -> Option<&Type> {
        if let Some(fields) = crate::types::as_soa_tuple(ty) {
            let size = array_size(fields.first()?)?;
            return fields.iter().all(|field| array_size(field) == Some(size)).then_some(size);
        }
        ty.array_size()
    }
    if let Some(actual) = compiler.facts.alias(source) {
        return source_size(compiler, actual, length);
    }
    if length {
        if let Some(Type::Constructed(TypeName::Size(n), _)) =
            compiler.facts.source_type(source).and_then(array_size)
        {
            return Ok(SizeExpr::Integer(*n as i64));
        }
    }
    if let Some(DispatchLen::InputBinding {
        set,
        binding,
        elem_bytes,
    }) = compiler.host_lengths.get(&source)
    {
        return Ok(SizeExpr::BufferLength {
            set: *set,
            binding: *binding,
            stride: *elem_bytes,
        });
    }
    if !length {
        if let Some(value) = input_scalar(compiler, source, 0)? {
            return Ok(value);
        }
    }
    if let Some(symbol) = compiler.facts.global_symbol(source) {
        if let Some(result) =
            compiler.facts.definition(symbol).and_then(|scope| compiler.facts.result(scope))
        {
            return source_size(compiler, result, length);
        }
    }
    if length {
        if let Some(operation) = compiler.facts.operation(source) {
            if let Some(&(term, _)) = compiler.program.identities.origins.get(&source) {
                if matches!(
                    term.kind,
                    TermKind::Soac(SoacOp::Map { .. } | SoacOp::Scan { .. })
                ) {
                    if let Some(input) = compiler.facts.input(operation, 0) {
                        return source_size(compiler, input, true);
                    }
                }
            }
        }
        if let Some((_, start, end)) = compiler.facts.slice(source) {
            return Ok(binary(
                SizeOp::Subtract,
                source_size(compiler, end, false)?,
                source_size(compiler, start, false)?,
            ));
        }
        if let Some(part) = compiler.facts.array_part(source) {
            return source_size(compiler, part, true);
        }
        if let Some((parent, index)) = compiler.facts.projection(source) {
            if let Some(&(term, scope)) = compiler.program.identities.origins.get(&parent) {
                if let TermKind::Tuple(fields) = &term.kind {
                    if let Some(child) = fields
                        .get(index)
                        .and_then(|term| compiler.program.identities.occurrences.get(&(scope, term.id)))
                    {
                        return source_size(compiler, *child, true);
                    }
                }
            }
        }
        if let Some(Type::Constructed(TypeName::Size(n), _)) =
            compiler.facts.source_type(source).and_then(array_size)
        {
            return Ok(SizeExpr::Integer(*n as i64));
        }
        if let Some(resource) = compiler.plan.value_ref(source) {
            let backing = compiler.plan.backing(resource).unwrap_or(resource);
            if let Some(buffer) = compiler.plan.buffers.get(&backing) {
                return extent(compiler, buffer.extent);
            }
            if let Some(original) = compiler.plan.external(backing) {
                if original != source {
                    return source_size(compiler, original, length);
                }
            }
        }
        if let Some(&(array, scope)) = compiler.program.identities.arrays.get(&source) {
            match array {
                ArrayExpr::Literal(values) => return Ok(SizeExpr::Integer(values.len() as i64)),
                ArrayExpr::Range { len, .. } => {
                    let Some(&source) = compiler.program.identities.occurrences.get(&(scope, len.id))
                    else {
                        return Err(error("array extent occurrence missing"));
                    };
                    return source_size(compiler, source, false);
                }
                _ => {}
            }
        }
    }
    if let Some(&(term, scope)) = compiler.program.identities.origins.get(&source) {
        match &term.kind {
            TermKind::Coerce { inner, .. } => {
                let Some(&source) = compiler.program.identities.occurrences.get(&(scope, inner.id)) else {
                    return Err(error("size coercion missing"));
                };
                return source_size(compiler, source, length);
            }
            TermKind::App { func, args } if !length => {
                if let TermKind::BinOp(op) = &func.kind {
                    let integer = |operator| match term.ty {
                        Type::Constructed(TypeName::UInt(32), _) => SizeOp::U32(operator),
                        _ => SizeOp::I32(operator),
                    };
                    let operation = match op.op {
                        BinaryOperator::Add => Some(integer(IntegerOp::Add)),
                        BinaryOperator::Subtract => Some(integer(IntegerOp::Subtract)),
                        BinaryOperator::Multiply => Some(integer(IntegerOp::Multiply)),
                        BinaryOperator::Divide | BinaryOperator::FloorDivide => Some(SizeOp::Floor),
                        _ => None,
                    };
                    if let Some(op) = operation {
                        let values = args
                            .iter()
                            .map(|arg| {
                                let Some(&value) =
                                    compiler.program.identities.occurrences.get(&(scope, arg.id))
                                else {
                                    return Err(error("size operand missing"));
                                };
                                source_size(compiler, value, false)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        if let [a, b] = values.as_slice() {
                            return Ok(binary(op, a.clone(), b.clone()));
                        }
                    }
                }
                if let TermKind::Var(VarRef::Builtin { id, .. }) = &func.kind {
                    if *id == catalog().known().length && args.len() == 1 {
                        let Some(&source) =
                            compiler.program.identities.occurrences.get(&(scope, args[0].id))
                        else {
                            return Err(error("size array missing"));
                        };
                        return source_size(compiler, source, true);
                    }
                }
            }
            TermKind::ArrayExpr(ArrayExpr::Literal(elements)) if length => {
                return Ok(SizeExpr::Integer(elements.len() as i64));
            }
            TermKind::Tuple(fields) if length => {
                if let Some(child) = fields
                    .first()
                    .and_then(|term| compiler.program.identities.occurrences.get(&(scope, term.id)))
                {
                    return source_size(compiler, *child, true);
                }
            }
            TermKind::IntLit(n) => {
                return n.parse().map(SizeExpr::Integer).map_err(|_| error("invalid host integer"));
            }
            TermKind::ArrayExpr(ArrayExpr::Range { len, .. }) if length => {
                let Some(&source) = compiler.program.identities.occurrences.get(&(scope, len.id)) else {
                    return Err(error("range length missing"));
                };
                return source_size(compiler, source, false);
            }
            _ => {}
        }
    }
    Err(error(format!(
        "host size unavailable for {source:?}: {:?}",
        compiler.program.identities.origins.get(&source).map(|(term, _)| &term.kind)
    )))
}
fn binary(op: SizeOp, left: SizeExpr, right: SizeExpr) -> SizeExpr {
    if let (SizeExpr::Integer(a), SizeExpr::Integer(b)) = (&left, &right) {
        let value = match op {
            SizeOp::Multiply => a.checked_mul(*b),
            SizeOp::Add => a.checked_add(*b),
            SizeOp::Subtract => a.checked_sub(*b),
            SizeOp::Ceiling if *b > 0 && *a >= 0 => a.checked_add(*b - 1).map(|n| n / *b),
            _ => None,
        };
        if let Some(value) = value {
            return SizeExpr::Integer(value);
        }
    }

    if matches!(right, SizeExpr::Integer(0))
        && matches!(
            op,
            SizeOp::Add
                | SizeOp::Subtract
                | SizeOp::I32(IntegerOp::Add | IntegerOp::Subtract)
                | SizeOp::U32(IntegerOp::Add | IntegerOp::Subtract)
        )
    {
        return left;
    }
    if matches!(right, SizeExpr::Integer(1))
        && matches!(
            op,
            SizeOp::Multiply | SizeOp::I32(IntegerOp::Multiply) | SizeOp::U32(IntegerOp::Multiply)
        )
    {
        return left;
    }
    SizeExpr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn input_scalar(
    compiler: &Compiler<'_, '_>,
    source: Value,
    mut offset: u32,
) -> Result<Option<SizeExpr>, OptimizeError> {
    let scalar = match compiler.facts.source_type(source) {
        Some(Type::Constructed(TypeName::Int(32), _)) => HostSizeScalar::I32,
        Some(Type::Constructed(TypeName::UInt(32), _)) => HostSizeScalar::U32,
        Some(Type::Constructed(TypeName::Float(32), _)) => HostSizeScalar::F32,
        _ => return Ok(None),
    };
    let mut root = source;
    let mut path = Vec::new();
    while let Some((parent, index)) = compiler.facts.projection(root) {
        let Some(ty) = compiler.facts.source_type(parent) else {
            return Ok(None);
        };
        let (delta, name) = match ty {
            Type::Constructed(TypeName::Vec, fields) => {
                let Some(element) = fields.first() else {
                    return Ok(None);
                };
                let Some(bytes) = type_byte_size(element) else {
                    return Ok(None);
                };
                (
                    bytes * index as u32,
                    ["x", "y", "z", "w"].get(index).copied().unwrap_or("component").to_string(),
                )
            }
            _ => {
                let Some(layout) = block_layout(ty, StorageLayout::Std140) else {
                    return Ok(None);
                };
                let Some(&delta) = layout.member_offsets.get(index) else {
                    return Ok(None);
                };
                let name = if let Type::Constructed(TypeName::Record(names), _) = ty {
                    names.0.get(index).cloned().unwrap_or_else(|| index.to_string())
                } else {
                    index.to_string()
                };
                (delta, name)
            }
        };
        path.push(name);
        offset += delta;
        root = parent;
    }
    let Some(input) = compiler.input_interfaces.get(&root) else {
        return Ok(None);
    };
    path.reverse();
    let name = std::iter::once(input.name.as_str())
        .chain(path.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join("_");
    Ok(match &input.kind {
        EntryInputKind::PushConstant { slot } => Some(SizeExpr::Scalar(HostSizeInput::PushConstant {
            name: name.clone(),
            push_constant_offset: slot.offset + offset,
            scalar,
        })),
        EntryInputKind::Uniform { binding } => Some(SizeExpr::Scalar(HostSizeInput::Uniform {
            name: name.clone(),
            set: binding.set,
            binding: binding.binding,
            offset,
            scalar,
        })),
        _ => None,
    })
}

fn extent_inputs(
    compiler: &Compiler<'_, '_>,
    key: Value,
    seen: &mut LookupSet<Value>,
    inputs: &mut Vec<HostSizeInput>,
) -> Result<(), OptimizeError> {
    let Some((name, children)) = compiler.plan.extent(key) else {
        return Err(error("missing host capacity"));
    };
    match name {
        "Scalar" | "Length" => {
            if let Some(source) = compiler.plan.expr(children[0]) {
                source_inputs(compiler, source, seen, inputs)?;
            }
        }
        "ChunkCount" => extent_inputs(compiler, children[0], seen, inputs)?,
        "Product" => {
            extent_inputs(compiler, children[0], seen, inputs)?;
            extent_inputs(compiler, children[1], seen, inputs)?;
        }
        _ => {}
    }
    Ok(())
}
fn source_inputs(
    compiler: &Compiler<'_, '_>,
    source: Value,
    seen: &mut LookupSet<Value>,
    inputs: &mut Vec<HostSizeInput>,
) -> Result<(), OptimizeError> {
    if !seen.insert(source) {
        return Ok(());
    }
    if let Some(actual) = compiler.facts.alias(source) {
        return source_inputs(compiler, actual, seen, inputs);
    }
    if let Some(SizeExpr::Scalar(input)) = input_scalar(compiler, source, 0)? {
        if !inputs.contains(&input) {
            inputs.push(input);
        }
        return Ok(());
    }
    if let Some((parent, _)) = compiler.facts.projection(source) {
        source_inputs(compiler, parent, seen, inputs)?;
    }
    if let Some(part) = compiler.facts.array_part(source) {
        source_inputs(compiler, part, seen, inputs)?;
    }
    if let Some(resource) = compiler.plan.value_ref(source) {
        let backing = compiler.plan.backing(resource).unwrap_or(resource);
        if let Some(buffer) = compiler.plan.buffers.get(&backing) {
            extent_inputs(compiler, buffer.extent, seen, inputs)?;
        }
    }
    if let Some(&(term, scope)) = compiler.program.identities.origins.get(&source) {
        let mut children = Vec::new();
        term.for_each_child(&mut |child| {
            if let Some(&value) = compiler.program.identities.occurrences.get(&(scope, child.id)) {
                children.push(value);
            }
        });
        for child in children {
            source_inputs(compiler, child, seen, inputs)?;
        }
    }
    if let Some(&(array, scope)) = compiler.program.identities.arrays.get(&source) {
        if let ArrayExpr::Range { len, .. } = array {
            if let Some(&value) = compiler.program.identities.occurrences.get(&(scope, len.id)) {
                source_inputs(compiler, value, seen, inputs)?;
            }
        }
    }
    Ok(())
}
