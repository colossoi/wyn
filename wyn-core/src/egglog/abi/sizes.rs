//! Lower selected allocation extents directly into the final host expressions.

use crate::egglog::to_ssa::{error, Compiler};
use crate::egglog::OptimizeError;
use crate::egglog::{facts::Facts, host::lower as host};
use crate::host::{BufferLen, DispatchLen, DispatchSize, Expr, ScalarExpr, ScalarSource};
use crate::ssa::layout::storage_elem_stride;
use egglog_engine::{RawValues, Value};

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
    let selected =
        crate::egglog::query::Query(&compiler.program.graph).required("SelectedLaunch", (stage,))?;
    if let Some(fields) = compiler.facts.enode("FixedLaunch", selected) {
        let (x, y, z) = compiler.facts.grid(fields[0])?;
        return Ok(DispatchSize::Fixed {
            x,
            y,
            z,
            explicit: true,
        });
    }
    if let Some(fields) = compiler.facts.enode("BufferLaunch", selected) {
        let Some((binding, element, _)) = compiler.bindings.buffer(fields[0])? else {
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
        Expr::Integer(count) => DispatchLen::Fixed {
            count: u32::try_from(count)
                .map_err(|_| error("launch extent must fit an unsigned 32-bit integer"))?,
        },
        Expr::BufferLength {
            source: ScalarSource::Binding { set, binding },
            stride,
        } => DispatchLen::InputBinding {
            set,
            binding,
            elem_bytes: stride,
        },
        Expr::Scalar(ScalarExpr::Parameter {
            source: ScalarSource::PushConstant { offset: base, .. },
            offset,
            ..
        }) => DispatchLen::PushConstant {
            offset: base + offset,
        },
        value => {
            return Err(error(format!(
                "selected launch has no supported physical length: {value:?}"
            )))
        }
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
    let Some((_, element, extent_value)) = compiler.bindings.buffer(buffer)? else {
        return Err(error("allocation missing"));
    };
    let Some(stride) = storage_elem_stride(&compiler.facts.physical_type(element, true)?) else {
        return Err(error("allocation stride missing"));
    };
    let minimum = crate::egglog::query::Query(&compiler.program.graph)
        .required("MinimumArrayCapacity", RawValues(vec![]))?;
    let minimum = compiler.facts.integer(minimum);
    match extent(compiler, extent_value)? {
        Expr::Integer(n) => Ok(BufferLen::Fixed {
            bytes: u64::try_from(n.max(minimum))
                .ok()
                .and_then(|n| n.checked_mul(u64::from(stride)))
                .ok_or_else(|| error("allocation size overflow"))?,
        }),
        Expr::BufferLength {
            source: ScalarSource::Binding { set, binding },
            stride: src_elem_bytes,
        } => Ok(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes: stride,
            src_elem_bytes,
        }),
        size => Ok(BufferLen::Computed {
            bytes: Expr::Max(Box::new(Expr::Integer(minimum)), Box::new(size))
                .multiply(Expr::Integer(stride.into())),
        }),
    }
}

pub(in crate::egglog) fn extent(compiler: &Compiler<'_, '_>, key: Value) -> Result<Expr, OptimizeError> {
    if let Some(bound) = compiler.facts.lookup("HostBound", (key,)) {
        return extent(compiler, bound);
    }
    let Some((name, children)) = compiler.facts.extent(key) else {
        return Err(error("host extent missing"));
    };
    match name {
        "Fixed" => Ok(Expr::Integer(
            compiler.program.graph.value_to_base::<i64>(children[0]),
        )),
        "Length" | "Scalar" => {
            let source = children[0];
            source_size(compiler, source, name == "Length")
        }
        "Product" => Ok(extent(compiler, children[0])?.multiply(extent(compiler, children[1])?)),
        "Difference" => Ok(Expr::Subtract(
            Box::new(extent(compiler, children[0])?),
            Box::new(extent(compiler, children[1])?),
        )),
        _ => Err(error("unsupported host extent")),
    }
}
pub(in crate::egglog) fn source_size(
    compiler: &Compiler<'_, '_>,
    source: Value,
    length: bool,
) -> Result<Expr, OptimizeError> {
    if length {
        let Some(bound) = compiler.facts.view_extent(source) else {
            return Err(error(format!("array {source:?} has no host extent")));
        };
        if let Some(("Length", fields)) = compiler.facts.extent(bound) {
            if fields[0] == source {
                let Some((binding, stride)) = compiler.facts.input_storage(source)? else {
                    return Err(error(format!("array {source:?} has no host length input")));
                };
                return Ok(Expr::BufferLength {
                    source: ScalarSource::Binding {
                        set: binding.set,
                        binding: binding.binding,
                    },
                    stride,
                });
            }
        }
        return extent(compiler, bound);
    }
    if let Some(actual) = compiler.facts.alias(source) {
        return source_size(compiler, actual, false);
    }
    if let Some(array) = compiler.facts.lookup("SourceLength", (source,)) {
        return source_size(compiler, array, true);
    }
    let Some(context) = compiler.facts.lookup("ScalarSourceContext", (source,)) else {
        return Err(error("host expression has no selected context"));
    };
    Ok(
        match host::allocation(compiler.program, context, source)
            .map_err(|e| e.required("allocation extent"))?
        {
            ScalarExpr::I32(value) => Expr::Integer(value.into()),
            ScalarExpr::U32(value) => Expr::Integer(value.into()),
            ScalarExpr::BufferLength { source, stride } => Expr::BufferLength { source, stride },
            value => Expr::Scalar(value),
        },
    )
}
