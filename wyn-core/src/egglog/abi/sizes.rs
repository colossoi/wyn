//! Lower selected allocation extents directly into the final host expressions.

use crate::egglog::to_ssa::{error, Compiler};
use crate::egglog::OptimizeError;
use crate::egglog::{facts::Facts, host::lower as host};
use crate::host::{BufferLen, DispatchSize, Expr, ScalarExpr, ScalarSource};
use crate::interface::StorageLayout;
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
    workgroup_width: u32,
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
    let (elements, divisor) = if let Some(fields) = compiler.facts.enode("BufferLaunch", selected) {
        let Some((binding, _element, _)) = compiler.bindings.buffer(fields[0])? else {
            return Err(error("launch buffer missing"));
        };
        let Some(elem_bytes) = storage_elem_stride(&compiler.bindings.buffer_element(fields[0])?) else {
            return Err(error("launch element stride missing"));
        };
        let bound = Expr::BufferLength {
            source: ScalarSource::Binding {
                set: binding.set,
                binding: binding.binding,
            },
            stride: elem_bytes,
        };
        let domain = compiler.phase_extent(stage)?;
        let elements = match host::launch_extent(compiler.program, domain) {
            Ok(ScalarExpr::BufferLength { source, stride }) => Expr::BufferLength { source, stride },
            Ok(value) => Expr::Scalar(value),
            Err(host::Error::Unsupported) => bound,
            Err(host::Error::Invalid(error)) => return Err(error),
        };
        (elements, compiler.facts.positive(fields[1], "launch divisor")?)
    } else if let Some(fields) = compiler.facts.enode("ExtentLaunch", selected) {
        (
            extent(compiler, fields[0])?,
            compiler.facts.positive(fields[1], "launch divisor")?,
        )
    } else {
        return Err(error("unknown selected launch"));
    };
    if workgroup_width == 0 || divisor % workgroup_width != 0 {
        return Err(error("launch divisor must cover whole workgroups"));
    }
    DispatchSize::linear(elements, divisor).map_err(|e| error(e.to_string()))
}

pub(in crate::egglog) fn capacity(
    compiler: &Compiler<'_, '_>,
    buffer: Value,
) -> Result<BufferLen, OptimizeError> {
    let Some((_, _element, extent_value)) = compiler.bindings.buffer(buffer)? else {
        return Err(error("allocation missing"));
    };
    let Some(stride) = storage_elem_stride(&compiler.bindings.buffer_element(buffer)?) else {
        return Err(error("allocation stride missing"));
    };
    let minimum = crate::egglog::query::Query(&compiler.program.graph)
        .required("MinimumArrayCapacity", RawValues(vec![]))?;
    let minimum = compiler.facts.integer(minimum);
    match allocation_bound(compiler, buffer, extent_value, &mut Vec::new())? {
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
    if let Some(count) = compiler.facts.lookup("StaticCapacityBound", (key,)) {
        return Ok(Expr::Integer(compiler.facts.integer(count)));
    }
    if compiler.program.stage.selected.extents.contains_key(&key) {
        return Ok(Expr::Scalar(
            host::allocation(compiler.program, key).map_err(|e| e.required("allocation extent"))?,
        ));
    }
    let Some((name, children)) = compiler.facts.extent(key) else {
        return Err(error("host extent missing"));
    };
    match name {
        "Fixed" => Ok(Expr::Integer(
            compiler.program.graph.value_to_base::<i64>(children[0]),
        )),
        "Length" | "Scalar" => Ok(
            match host::allocation(compiler.program, key).map_err(|e| e.required("allocation extent"))? {
                ScalarExpr::I32(value) => Expr::Integer(value.into()),
                ScalarExpr::U32(value) => Expr::Integer(value.into()),
                ScalarExpr::BufferLength { source, stride } => Expr::BufferLength { source, stride },
                value => Expr::Scalar(value),
            },
        ),
        "Product" => Ok(extent(compiler, children[0])?.multiply(extent(compiler, children[1])?)),
        "Sum" => Ok(Expr::Add(
            Box::new(extent(compiler, children[0])?),
            Box::new(extent(compiler, children[1])?),
        )),
        "Difference" => Ok(Expr::Subtract(
            Box::new(extent(compiler, children[0])?),
            Box::new(extent(compiler, children[1])?),
        )),
        _ => Err(error("unsupported host extent")),
    }
}
fn allocation_bound(
    compiler: &Compiler<'_, '_>,
    buffer: Value,
    fallback: Value,
    active: &mut Vec<Value>,
) -> Result<Expr, OptimizeError> {
    if active.contains(&buffer) {
        // Alias cycles add no capacity beyond their initial storage.
        return Ok(Expr::Integer(0));
    }
    active.push(buffer);
    let result = (|| {
        let mut bounds = compiler.facts.set("AllocationBounds", (buffer,));
        let explicit = !bounds.is_empty();
        if !explicit {
            bounds.push(fallback);
        }
        let mut size = Expr::Integer(0);
        for bound in bounds {
            let backing = compiler
                .facts
                .extent(bound)
                .filter(|(name, _)| *name == "Length")
                .and_then(|(_, fields)| compiler.facts.lookup("ArrayStorage", (fields[0],)))
                .map(|storage| compiler.facts.backing(storage).unwrap_or(storage));
            let value = match backing {
                Some(backing) if backing != buffer => allocation_bound(
                    compiler,
                    backing,
                    compiler.facts.capacity(backing).unwrap_or(bound),
                    active,
                )?,
                Some(_) if explicit => continue,
                _ => extent(compiler, bound)?,
            };
            // Keep direct input lengths eligible for LikeInput allocation.
            if !explicit {
                return Ok(value);
            }
            size = Expr::Max(Box::new(size), Box::new(value));
        }
        Ok(size)
    })();
    active.pop();
    result
}

pub(in crate::egglog) fn published_shape(
    compiler: &Compiler<'_, '_>,
    source: Value,
) -> Result<Expr, OptimizeError> {
    let read = |resource| published_read(compiler, resource);
    host::published_length(compiler.program, source, &read)
        .map(|length| Expr::Max(Box::new(Expr::Scalar(length)), Box::new(Expr::Integer(0))))
        .map_err(|e| e.required("published array shape"))
}

pub(super) fn published_read(
    compiler: &Compiler<'_, '_>,
    resource: Value,
) -> Result<Option<ScalarExpr>, host::Error> {
    let Some(backing) = compiler.facts.backing(resource) else {
        return Ok(None);
    };
    let Some((binding, _, _)) = compiler.bindings.buffer(backing)? else {
        return Ok(None);
    };
    let ty = compiler.bindings.buffer_element(backing)?;
    let index = compiler
        .facts
        .lookup("MetadataIndex", (resource,))
        .map(|value| compiler.facts.integer(value))
        .unwrap_or(0);
    let offset = u32::try_from(index)
        .ok()
        .and_then(|index| index.checked_mul(4))
        .ok_or_else(|| error("shape offset overflow"))?;
    match host::read(
        ScalarSource::Binding {
            set: binding.set,
            binding: binding.binding,
        },
        offset,
        &ty,
        StorageLayout::Std430,
        false,
    ) {
        Ok(value) => Ok(Some(value)),
        Err(host::Error::Unsupported) => Ok(None),
        Err(host::Error::Invalid(error)) => Err(error.into()),
    }
}
