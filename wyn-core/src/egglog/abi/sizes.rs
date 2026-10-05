//! Publish host allocation sizes without materializing scalar expression graphs.

use crate::egglog::to_ssa::{error, read::Facts, Compiler};
use crate::egglog::OptimizeError;
use crate::host::{
    BufferLen, DispatchLen, DispatchSize, HostSizeInput, HostSizeScalar, IntegerOp, SizeExpr, SizeOp,
};
use crate::interface::{EntryInputKind, StorageLayout};
use crate::ssa::layout::{block_layout, storage_elem_stride};
use crate::types::{Type, TypeExt, TypeName};
use egglog_engine::{RawValues, Term, TermId, Value};
use std::collections::BTreeSet;

pub(super) fn output_capacity(facts: &Facts<'_, '_>, capacity: Value) -> Result<BufferLen, OptimizeError> {
    if let Some(fields) = facts.enode("OutputBytes", capacity) {
        let bytes = u64::try_from(facts.integer(fields[0]))
            .map_err(|_| error("output byte capacity must be nonnegative"))?;
        return Ok(BufferLen::Fixed { bytes });
    }
    if let Some(fields) = facts.enode("OutputLike", capacity) {
        let Some(binding) = facts.enode("InputBinding", fields[0]) else {
            return Err(error("output size input has no binding"));
        };
        return Ok(BufferLen::LikeInput {
            set: facts.unsigned(binding[0], "output size input set")?,
            binding: facts.unsigned(binding[1], "output size input binding")?,
            elem_bytes: facts.positive(fields[1], "output stride")?,
            src_elem_bytes: facts.positive(fields[2], "output size input stride")?,
        });
    }
    if let Some(fields) = facts.enode("OutputDispatch", capacity) {
        return Ok(BufferLen::SameAsDispatch {
            elem_bytes: facts.positive(fields[0], "output stride")?,
        });
    }
    Err(error("unknown selected output capacity"))
}

pub(in crate::egglog) fn dispatch(
    compiler: &Compiler<'_, '_>,
    stage: Value,
) -> Result<DispatchSize, OptimizeError> {
    let selected = super::required(&compiler.program.graph, "SelectedLaunch", (stage,))?;
    if let Some(fields) = compiler.facts.enode("FixedLaunch", selected) {
        let (x, y, z) = compiler.facts.grid(fields[0])?;
        return Ok(DispatchSize::Fixed {
            x,
            y,
            z,
            explicit: compiler.program.graph.value_to_base::<bool>(fields[1]),
        });
    }
    if let Some(fields) = compiler.facts.enode("BufferLaunch", selected) {
        let Some((binding, element, _)) = compiler.plan.buffer(fields[0])? else {
            return Err(error("launch buffer missing"));
        };
        let Some(elem_bytes) = storage_elem_stride(&compiler.facts.physical_type(element, true)?) else {
            return Err(error("launch element stride missing"));
        };
        return Ok(DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding {
                set: binding.set,
                binding: binding.binding,
                elem_bytes,
            },
            workgroup_size: compiler.facts.positive(fields[1], "launch divisor")?,
        });
    }
    let Some(fields) = compiler.facts.enode("ExtentLaunch", selected) else {
        return Err(error("unknown selected launch"));
    };
    let len = match extent(compiler, fields[0])? {
        SizeExpr::Integer(count) => DispatchLen::Fixed {
            count: u32::try_from(count)
                .map_err(|_| error("launch extent must fit an unsigned 32-bit integer"))?,
        },
        SizeExpr::BufferLength { set, binding, stride } => DispatchLen::InputBinding {
            set,
            binding,
            elem_bytes: stride,
        },
        SizeExpr::Scalar(HostSizeInput::PushConstant {
            push_constant_offset, ..
        }) => DispatchLen::PushConstant {
            offset: push_constant_offset,
        },
        _ => return Err(error("selected launch has no supported physical length")),
    };
    Ok(DispatchSize::DerivedFrom {
        len,
        workgroup_size: compiler.facts.positive(fields[1], "launch divisor")?,
    })
}

pub(in crate::egglog) fn capacity(
    compiler: &Compiler<'_, '_>,
    buffer: Value,
) -> Result<BufferLen, OptimizeError> {
    let Some((_, element, extent_value)) = compiler.plan.buffer(buffer)? else {
        return Err(error("allocation missing"));
    };
    let Some(stride) = storage_elem_stride(&compiler.facts.physical_type(element, true)?) else {
        return Err(error("allocation stride missing"));
    };
    let minimum = super::required(&compiler.program.graph, "MinimumArrayCapacity", RawValues(vec![]))?;
    let minimum = compiler.facts.integer(minimum);
    let mut inputs = BTreeSet::new();
    // TODO: Select HostProvided in Egglog using explicit size-expression support
    // facts in the scalar graph, without feeding lowering success or failure
    // back into planning.
    if !extent_inputs(compiler, extent_value, &mut inputs)? {
        if inputs.is_empty() {
            return Err(error("allocation extent has no host representation"));
        }
        return Ok(BufferLen::HostProvided {
            inputs: inputs.into_iter().collect(),
            elem_bytes: stride,
        });
    }
    match extent(compiler, extent_value)? {
        SizeExpr::Integer(n) => Ok(BufferLen::Fixed {
            bytes: n.max(minimum) as u64 * u64::from(stride),
        }),
        SizeExpr::BufferLength {
            set,
            binding,
            stride: src_elem_bytes,
        } => Ok(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes: stride,
            src_elem_bytes,
        }),
        size => Ok(BufferLen::Computed {
            bytes: binary(
                SizeOp::Multiply,
                binary(SizeOp::Max, SizeExpr::Integer(minimum), size),
                SizeExpr::Integer(stride.into()),
            ),
        }),
    }
}

pub(in crate::egglog) fn extent(
    compiler: &Compiler<'_, '_>,
    key: Value,
) -> Result<SizeExpr, OptimizeError> {
    if let Some(bound) = compiler.facts.lookup("HostBound", (key,)) {
        return extent(compiler, bound);
    }
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
        "Difference" => Ok(binary(
            SizeOp::Subtract,
            extent(compiler, children[0])?,
            extent(compiler, children[1])?,
        )),
        _ => Err(error("unsupported host extent")),
    }
}
pub(in crate::egglog) fn source_size(
    compiler: &Compiler<'_, '_>,
    source: Value,
    length: bool,
) -> Result<SizeExpr, OptimizeError> {
    if length {
        let Some(bound) = compiler.plan.view_extent(source) else {
            return Err(error(format!("array {source:?} has no host extent")));
        };
        if let Some(("Length", fields)) = compiler.plan.extent(bound) {
            if compiler.plan.expr(fields[0]) == Some(source) {
                let Some((binding, stride)) = compiler.facts.input_storage(source)? else {
                    return Err(error(format!("array {source:?} has no host length input")));
                };
                return Ok(SizeExpr::BufferLength {
                    set: binding.set,
                    binding: binding.binding,
                    stride,
                });
            }
        }
        return extent(compiler, bound);
    }
    if let Some(actual) = compiler.facts.alias(source) {
        return source_size(compiler, actual, false);
    }
    if let Some(value) = input_scalar(compiler, source, 0)? {
        return Ok(value);
    }
    if let Some(array) = compiler.facts.lookup("SourceLength", (source,)) {
        return source_size(compiler, array, true);
    }
    let Some(context) = compiler.facts.lookup("ScalarSourceContext", (source,)) else {
        return Err(error("host expression has no selected context"));
    };
    let Some(&term) = compiler.program.stage.selected.roots.get(&(context, source)) else {
        return Err(error("host expression root missing"));
    };
    scalar_size(compiler, term)
}

fn scalar_size(compiler: &Compiler<'_, '_>, term: TermId) -> Result<SizeExpr, OptimizeError> {
    let selected = &compiler.program.stage.selected;
    let (name, fields) = selected.app(term)?;
    match name {
        "ScalarLiteral" => Ok(SizeExpr::Integer(
            selected.text(fields[2])?.parse().map_err(|_| error("noninteger host size"))?,
        )),
        "ScalarParameter" => {
            let Some(source) =
                compiler.facts.parameter(selected.values[fields[2]], selected.integer(fields[3])?)
            else {
                return Err(error("host parameter missing"));
            };
            let Some(value) = input_scalar(compiler, source, 0)? else {
                return Err(error("host size parameter has no input"));
            };
            Ok(value)
        }
        "ScalarLeaf" | "ScalarExecute" => source_size(compiler, selected.values[fields[2]], false),
        "ScalarBinary" => {
            let integer = match compiler.facts.ty(selected.values[fields[1]]) {
                Some(Type::Constructed(TypeName::UInt(32), _)) => SizeOp::U32,
                Some(Type::Constructed(TypeName::Int(32), _)) => SizeOp::I32,
                _ => {
                    return Err(error(
                        "host size arithmetic requires a selected 32-bit integer type",
                    ))
                }
            };
            let op = match selected.text(fields[2])? {
                "+" => integer(IntegerOp::Add),
                "-" => integer(IntegerOp::Subtract),
                "*" => integer(IntegerOp::Multiply),
                "/" | "//" => SizeOp::Floor,
                "%" => SizeOp::Mod,
                _ => return Err(error("unsupported selected size operator")),
            };
            Ok(binary(
                op,
                scalar_size(compiler, fields[3])?,
                scalar_size(compiler, fields[4])?,
            ))
        }
        "ScalarCoerce" => scalar_size(compiler, fields[2]),
        "ScalarProject" => {
            let Some(value) = selected_input(compiler, term)? else {
                return Err(error("host projection has no input"));
            };
            Ok(SizeExpr::Scalar(value))
        }
        "ScalarUnary" if selected.text(fields[2])? == "-" => Ok(binary(
            SizeOp::Subtract,
            SizeExpr::Integer(0),
            scalar_size(compiler, fields[3])?,
        )),
        _ => Err(error(format!("unsupported selected size expression {name}"))),
    }
}
fn binary(op: SizeOp, left: SizeExpr, right: SizeExpr) -> SizeExpr {
    SizeExpr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn input_scalar(
    compiler: &Compiler<'_, '_>,
    source: Value,
    _offset: u32,
) -> Result<Option<SizeExpr>, OptimizeError> {
    let mut source = source;
    let mut path = Vec::new();
    loop {
        if let Some(actual) = compiler.facts.alias(source) {
            source = actual;
        } else if let Some((parent, index)) = compiler.facts.projection(source) {
            path.push(index);
            source = parent;
        } else {
            break;
        }
    }
    let Some(region) = compiler.facts.lookup("SourceParameterRegion", (source,)) else {
        return Ok(None);
    };
    let index = super::required(&compiler.program.graph, "SourceParameterIndex", (source,))?;
    path.reverse();
    parameter_input(compiler, region, compiler.facts.integer(index), &path).map(|v| v.map(SizeExpr::Scalar))
}

fn parameter_input(
    compiler: &Compiler<'_, '_>,
    scope: Value,
    index: i64,
    path: &[usize],
) -> Result<Option<HostSizeInput>, OptimizeError> {
    if !compiler.facts.entry_region(scope) {
        return Ok(None);
    }
    let inputs = compiler.facts.parameter_inputs(scope, index)?;
    let [input] = inputs.as_slice() else {
        return Err(error("host scalar has a split parameter ABI"));
    };
    let input = &input.declaration;
    let rules = match input.kind {
        EntryInputKind::Uniform { .. } => StorageLayout::Std140,
        EntryInputKind::PushConstant { .. } => StorageLayout::Std430,
        _ => return Ok(None),
    };
    let mut ty = &input.ty;
    let mut offset = 0;
    let mut name = input.name.clone();
    for &index in path {
        match ty {
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
                let Some(layout) = block_layout(ty, rules) else {
                    return Err(error("host parameter block layout missing"));
                };
                let Some(&delta) = layout.member_offsets.get(index) else {
                    return Err(error("host field offset missing"));
                };
                let Some(field) = fields.get(index) else {
                    return Err(error("host parameter field missing"));
                };
                if let Type::Constructed(TypeName::Record(names), _) = ty {
                    name.push_str(&format!("_{}", names.0[index]));
                } else {
                    name.push_str(&format!("_{index}"));
                }
                offset += delta;
                ty = field;
            }
            Type::Constructed(TypeName::Vec, _) => {
                let Some(n) = ty.vec_size() else {
                    return Err(error("host vector width missing"));
                };
                if index >= n {
                    return Err(error("host vector index out of bounds"));
                }
                let Some(element) = ty.elem_type() else {
                    return Err(error("host vector element missing"));
                };
                let Some(bytes) = crate::ssa::layout::type_byte_size(element) else {
                    return Err(error("host scalar byte size missing"));
                };
                offset += index as u32 * bytes;
                name.push_str(["_x", "_y", "_z", "_w"][index]);
                ty = element;
            }
            _ => return Err(error("host projection has no aggregate representation")),
        }
    }
    let scalar = match ty {
        Type::Constructed(TypeName::Int(32), _) => HostSizeScalar::I32,
        Type::Constructed(TypeName::UInt(32), _) => HostSizeScalar::U32,
        Type::Constructed(TypeName::Float(32), _) => HostSizeScalar::F32,
        _ => return Ok(None),
    };
    Ok(Some(match input.kind {
        EntryInputKind::Uniform { binding } => HostSizeInput::Uniform {
            name,
            set: binding.set,
            binding: binding.binding,
            offset,
            scalar,
        },
        EntryInputKind::PushConstant { slot } => HostSizeInput::PushConstant {
            name,
            push_constant_offset: slot.offset + offset,
            scalar,
        },
        _ => return Err(error("host scalar ABI changed during decoding")),
    }))
}

fn selected_input(
    compiler: &Compiler<'_, '_>,
    mut term: TermId,
) -> Result<Option<HostSizeInput>, OptimizeError> {
    let selected = &compiler.program.stage.selected;
    let mut path = Vec::new();
    loop {
        let (name, fields) = selected.app(term)?;
        match name {
            "ScalarProject" => {
                path.push(selected.integer(fields[3])? as usize);
                term = fields[2];
            }
            "ScalarParameter" => {
                path.reverse();
                return parameter_input(
                    compiler,
                    selected.values[fields[2]],
                    selected.integer(fields[3])?,
                    &path,
                );
            }
            _ => return Ok(None),
        }
    }
}

fn extent_inputs(
    compiler: &Compiler<'_, '_>,
    extent: Value,
    inputs: &mut BTreeSet<HostSizeInput>,
) -> Result<bool, OptimizeError> {
    if let Some(bound) = compiler.facts.lookup("HostBound", (extent,)) {
        return extent_inputs(compiler, bound, inputs);
    }
    let Some((name, fields)) = compiler.plan.extent(extent) else {
        return Err(error("host extent missing"));
    };
    match name {
        "Fixed" => Ok(true),
        "Product" | "Difference" => {
            Ok(extent_inputs(compiler, fields[0], inputs)? & extent_inputs(compiler, fields[1], inputs)?)
        }
        "ChunkCount" => extent_inputs(compiler, fields[0], inputs),
        "Length" | "Scalar" => {
            let Some(source) = compiler.plan.expr(fields[0]) else {
                return Err(error("host extent identity missing"));
            };
            if name == "Length" {
                if compiler.facts.input_storage(source)?.is_some() {
                    return Ok(true);
                }
                let Some(bound) = compiler.plan.view_extent(source) else {
                    return Err(error(format!(
                        "host array length missing for {source:?} ({:?})",
                        compiler.facts.source_type(source)
                    )));
                };
                if bound == extent {
                    return Err(error(format!(
                        "self-referencing host array length for {source:?}"
                    )));
                }
                return extent_inputs(compiler, bound, inputs);
            }
            source_inputs(compiler, source, inputs)
        }
        _ => Err(error("unresolved host extent")),
    }
}

fn source_inputs(
    compiler: &Compiler<'_, '_>,
    source: Value,
    inputs: &mut BTreeSet<HostSizeInput>,
) -> Result<bool, OptimizeError> {
    if let Some(source) = compiler.facts.alias(source) {
        return source_inputs(compiler, source, inputs);
    }
    if let Some(SizeExpr::Scalar(input)) = input_scalar(compiler, source, 0)? {
        let integer = !matches!(
            input,
            HostSizeInput::Uniform {
                scalar: HostSizeScalar::F32,
                ..
            } | HostSizeInput::PushConstant {
                scalar: HostSizeScalar::F32,
                ..
            }
        );
        inputs.insert(input);
        return Ok(integer);
    }
    if let Some(array) = compiler.facts.lookup("SourceLength", (source,)) {
        let Some(extent) = compiler.plan.view_extent(array) else {
            return Err(error("host length expression missing"));
        };
        return extent_inputs(compiler, extent, inputs);
    }
    let context = super::required(&compiler.program.graph, "ScalarSourceContext", (source,))?;
    let Some(&term) = compiler.program.stage.selected.roots.get(&(context, source)) else {
        return Err(error("selected host expression missing"));
    };
    let selected = &compiler.program.stage.selected;
    let mut pending = vec![term];
    let mut seen = BTreeSet::new();
    let mut supported = true;
    while let Some(term) = pending.pop() {
        if !seen.insert(term) {
            continue;
        }
        let Term::App(name, fields) = selected.dag.get(term) else {
            continue;
        };
        if !name.starts_with("Scalar")
            || matches!(
                name.as_str(),
                "ScalarFunction" | "ScalarKernel" | "ScalarDispatch" | "ScalarTemplate"
            )
        {
            continue;
        }
        if let Some(input) = selected_input(compiler, term)? {
            supported &= !matches!(
                input,
                HostSizeInput::Uniform {
                    scalar: HostSizeScalar::F32,
                    ..
                } | HostSizeInput::PushConstant {
                    scalar: HostSizeScalar::F32,
                    ..
                }
            );
            inputs.insert(input);
            continue;
        }
        match name.as_str() {
            "ScalarLiteral" | "ScalarCoerce" | "ScalarCons" | "ScalarNil" => {}
            "ScalarBinary" => {
                supported &= matches!(selected.text(fields[2])?, "+" | "-" | "*" | "/" | "//" | "%")
            }
            "ScalarUnary" => supported &= selected.text(fields[2])? == "-",
            "ScalarLeaf" | "ScalarExecute" => {
                let child = selected.values[fields[2]];
                if child == source {
                    return Err(error("host expression refers to an unavailable device value"));
                }
                supported &= source_inputs(compiler, child, inputs)?;
                continue;
            }
            _ => supported = false,
        }
        pending.extend(fields.iter().copied());
    }
    Ok(supported)
}
