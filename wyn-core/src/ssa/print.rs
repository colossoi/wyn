//! Pretty-printer for SSA programs.

#![deny(clippy::let_underscore_must_use)]

use crate::ast::TypeName;
use crate::builtins;
use crate::flow::ExecutionModel;
use crate::op;
use crate::ssa;
use crate::ssa::types::Program;
use polytype::Type;
use std::fmt;

use super::types::*;

fn format_ref(vr: &ValueRef) -> String {
    match vr {
        ValueRef::Ssa(id) => format!("%{:?}", id),
        ValueRef::Const(ConstantValue::I32(v)) => format!("{}", v),
        ValueRef::Const(ConstantValue::U32(v)) => format!("{}u", v),
        ValueRef::Const(ConstantValue::F32(bits)) => format!("{}", f32::from_bits(*bits)),
        ValueRef::Const(ConstantValue::Bool(b)) => format!("{}", b),
    }
}

fn format_refs(vals: &[ValueRef]) -> String {
    vals.iter().map(format_ref).collect::<Vec<_>>().join(", ")
}

fn format_type(ty: &Type<TypeName>) -> String {
    match ty {
        Type::Constructed(TypeName::Int(bits), _) => format!("i{bits}"),
        Type::Constructed(TypeName::UInt(bits), _) => format!("u{bits}"),
        Type::Constructed(TypeName::Float(bits), _) => format!("f{bits}"),
        Type::Constructed(TypeName::Bool, _) => "bool".to_string(),
        Type::Constructed(TypeName::Unit, _) => "()".to_string(),
        Type::Constructed(TypeName::Tuple(n), args) => {
            let inner: Vec<String> = args.iter().map(format_type).collect();
            if *n == 1 {
                format!("({},)", inner.join(", "))
            } else {
                format!("({})", inner.join(", "))
            }
        }
        Type::Constructed(TypeName::Array, args) if args.len() >= 2 => {
            let elem = format_type(&args[0]);
            // args = [elem, variant, dim_0, ...]; size is args[2] when present.
            let size = if args.len() >= 3 { format_array_size(&args[2]) } else { "?".to_string() };
            format!("[{size}]{elem}")
        }
        Type::Constructed(TypeName::Vec, args) if !args.is_empty() => {
            let elem = format_type(&args[0]);
            let size = if args.len() >= 2 { format_array_size(&args[1]) } else { "?".to_string() };
            format!("vec{size}{elem}")
        }
        Type::Constructed(TypeName::Mat, args) if args.len() >= 3 => {
            let elem = format_type(&args[0]);
            let cols = format_array_size(&args[1]);
            let rows = format_array_size(&args[2]);
            format!("mat{cols}x{rows}{elem}")
        }
        Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
            let param = format_type(&args[0]);
            let ret = format_type(&args[1]);
            format!("{param} -> {ret}")
        }
        _ => format!("{ty:?}"),
    }
}

fn format_array_size(size_ty: &Type<TypeName>) -> String {
    match size_ty {
        Type::Constructed(TypeName::Size(n), _) => n.to_string(),
        _ => "?".to_string(),
    }
}

fn fmt_val(v: ValueId) -> String {
    format!("%{:?}", v)
}

fn fmt_place(p: ssa::types::PlaceId) -> String {
    format!("@{:?}", p)
}

fn fmt_block(b: BlockId) -> String {
    format!("{:?}", b)
}

pub fn format_program<Tag, GlobalContext>(program: &Program<Tag, GlobalContext>) -> String {
    format!("{}", ProgramDisplay(program))
}

struct ProgramDisplay<'a, Tag, GlobalContext>(&'a Program<Tag, GlobalContext>);

impl<Tag, GlobalContext> fmt::Display for ProgramDisplay<'_, Tag, GlobalContext> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let program = self.0;
        let mut needs_separator = false;

        for function in &program.functions {
            if needs_separator {
                writeln!(out)?;
            }
            format_function(out, &function.name, &function.body)?;
            needs_separator = true;
        }

        for entry in &program.entry_points {
            if needs_separator {
                writeln!(out)?;
            }
            let model = match &entry.execution_model {
                ExecutionModel::Vertex => "vertex",
                ExecutionModel::Fragment => "fragment",
                ExecutionModel::Compute { .. } => "compute",
            };
            let local_size = match &entry.execution_model {
                ExecutionModel::Compute { local_size } => {
                    format!(
                        " local_size({}, {}, {})",
                        local_size.0, local_size.1, local_size.2
                    )
                }
                _ => String::new(),
            };
            write!(out, "entry {model}{local_size} ")?;
            format_function(out, &entry.name, &entry.body)?;
            needs_separator = true;
        }

        Ok(())
    }
}

fn format_function(out: &mut fmt::Formatter<'_>, name: &str, body: &FuncBody) -> fmt::Result {
    let params: Vec<String> =
        body.params().map(|(val, ty, _name)| format!("{}: {}", fmt_val(val), format_type(ty))).collect();
    let ret = format_type(&body.return_ty);
    writeln!(out, "func @{name}({}) -> {ret} {{", params.join(", "))?;

    for (bid, block) in &body.inner.blocks {
        // Skip dead blocks
        if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
            continue;
        }

        // Block header
        if block.params.is_empty() {
            writeln!(out, "  {}:", fmt_block(bid))?;
        } else {
            let params: Vec<String> = block
                .params
                .iter()
                .map(|&p| format!("{}: {}", fmt_val(p), format_type(body.inner.value_type(p))))
                .collect();
            writeln!(out, "  {}({}):", fmt_block(bid), params.join(", "))?;
        }

        // Control header as comment
        if let Some(ctrl) = &block.control_header {
            match ctrl {
                ControlHeader::Loop {
                    merge,
                    continue_block,
                } => {
                    writeln!(
                        out,
                        "    # loop merge={} continue={}",
                        fmt_block(*merge),
                        fmt_block(*continue_block)
                    )?;
                }
                ControlHeader::Selection { merge } => {
                    writeln!(out, "    # selection merge={}", fmt_block(*merge))?;
                }
            }
        }

        // Instructions
        for &inst_id in &block.insts {
            let inst = body.get_inst(inst_id);
            write!(out, "    ")?;
            if let Some(result) = inst.result {
                write!(out, "{} = ", fmt_val(result))?;
            }
            format_inst_kind(out, &inst.data)?;
            writeln!(out)?;
        }

        // Terminator
        write!(out, "    ")?;
        format_terminator(out, &block.term)?;
        writeln!(out)?;
        writeln!(out)?;
    }

    writeln!(out, "}}")
}

fn format_inst_kind(out: &mut fmt::Formatter<'_>, kind: &InstKind) -> fmt::Result {
    use crate::op::{OpTag, PureViewSource};
    match kind {
        InstKind::Op { tag, operands } => match tag {
            op::OpTag::ResourceLen(resource) => {
                write!(out, "resource_len({},{})", resource.set, resource.binding)?;
            }
            OpTag::Int(s) => {
                write!(out, "int {s}")?;
            }
            OpTag::Uint(s) => {
                write!(out, "uint {s}")?;
            }
            OpTag::Float(s) => {
                write!(out, "float {s}")?;
            }
            OpTag::Bool(b) => {
                write!(out, "bool {b}")?;
            }
            OpTag::Unit => {
                write!(out, "unit")?;
            }
            OpTag::BinOp(op) => {
                write!(
                    out,
                    "binop {op} {}, {}",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                )?;
            }
            OpTag::UnaryOp(op) => {
                write!(out, "unaryop {op} {}", format_ref(&operands[0]))?;
            }
            OpTag::Tuple(_) => {
                write!(out, "tuple ({})", format_refs(operands))?;
            }
            OpTag::ArrayLit(_) => {
                write!(out, "array [{}]", format_refs(operands))?;
            }
            OpTag::ArrayRange { has_step } => {
                write!(
                    out,
                    "range {}..{}",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                )?;
                if *has_step {
                    write!(out, " step {}", format_ref(&operands[2]))?;
                }
            }
            OpTag::Vector(_) => {
                write!(out, "vector @[{}]", format_refs(operands))?;
            }
            OpTag::Matrix { rows, cols } => {
                write!(out, "matrix @[")?;
                for r in 0..*rows {
                    if r > 0 {
                        write!(out, ", ")?;
                    }
                    let row = &operands[r * cols..(r + 1) * cols];
                    write!(out, "[{}]", format_refs(row))?;
                }
                write!(out, "]")?;
            }
            OpTag::Project { index } => {
                write!(out, "project {}.{index}", format_ref(&operands[0]))?;
            }
            OpTag::Index => {
                write!(
                    out,
                    "index {}[{}]",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                )?;
            }
            OpTag::Call(function) => {
                write!(out, "call @{function}({})", format_refs(operands))?;
            }
            OpTag::Global(global) => {
                write!(out, "global @{global}")?;
            }

            OpTag::Intrinsic { id, overload_idx } => {
                let name = builtins::by_id(*id).dispatch_name();
                write!(out, "intrinsic @{name}#{overload_idx}({})", format_refs(operands))?;
            }
            OpTag::StorageImageLoad(binding) => {
                write!(
                    out,
                    "storage_image_load @({}, {})({})",
                    binding.set,
                    binding.binding,
                    format_ref(&operands[0])
                )?;
            }
            OpTag::StorageImageStore(binding) => {
                write!(
                    out,
                    "storage_image_write @({}, {})({}, {})",
                    binding.set,
                    binding.binding,
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                )?;
            }
            OpTag::StorageView(src) => {
                let src_str = match src {
                    PureViewSource::Storage(br) => {
                        format!("storage({}, {})", br.set, br.binding)
                    }
                    PureViewSource::Inherited => format_ref(&operands[2]),
                    PureViewSource::Workgroup { id, count } => {
                        format!("workgroup({id}, {count})")
                    }
                };
                write!(
                    out,
                    "storage_view {src_str} {} {}",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                )?;
            }
            OpTag::StorageViewLen => {
                write!(out, "storage_view_len {}", format_ref(&operands[0]))?;
            }
            OpTag::Materialize => {
                write!(out, "materialize {}", format_ref(&operands[0]))?;
            }
            OpTag::AddressableConstant(id) => {
                write!(out, "addressable_constant {}", id.0)?;
            }
            OpTag::DynamicExtract => {
                write!(
                    out,
                    "dynamic_extract {}[{}]",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                )?;
            }
        },
        InstKind::Alloca { elem_ty, result } => {
            write!(out, "alloca {} -> {}", format_type(elem_ty), fmt_place(*result))?;
        }
        InstKind::Load { place } => {
            write!(out, "load {}", fmt_place(*place))?;
        }
        InstKind::Store { place, value } => {
            write!(out, "store {}, {}", fmt_place(*place), format_ref(value))?;
        }
        InstKind::Atomic { place, op, values } => {
            let values = values.iter().map(format_ref).collect::<Vec<_>>().join(", ");
            write!(out, "atomic {:?} {}, {}", op, fmt_place(*place), values)?;
        }
        InstKind::ViewIndex { view, index, result } => {
            write!(
                out,
                "view_index {}[{}] -> {}",
                format_ref(view),
                format_ref(index),
                fmt_place(*result)
            )?;
        }
        InstKind::PlaceIndex { place, index, result } => {
            write!(
                out,
                "place_index {}[{}] -> {}",
                fmt_place(*place),
                format_ref(index),
                fmt_place(*result)
            )?;
        }
        InstKind::OutputSlot { index, result } => {
            write!(out, "output_slot {index} -> {}", fmt_place(*result))?;
        }
        InstKind::ControlBarrier => {
            write!(out, "control_barrier")?;
        }
    }
    Ok(())
}

fn format_terminator(out: &mut fmt::Formatter<'_>, term: &Terminator) -> fmt::Result {
    match term {
        Terminator::Branch { target, args } => {
            if args.is_empty() {
                write!(out, "br {}", fmt_block(*target))?;
            } else {
                write!(out, "br {}({})", fmt_block(*target), format_refs(args))?;
            }
        }
        Terminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => {
            let then_args_str =
                if then_args.is_empty() { String::new() } else { format!("({})", format_refs(then_args)) };
            let else_args_str =
                if else_args.is_empty() { String::new() } else { format!("({})", format_refs(else_args)) };
            write!(
                out,
                "br_if {} then {}{then_args_str} else {}{else_args_str}",
                format_ref(cond),
                fmt_block(*then_target),
                fmt_block(*else_target),
            )?;
        }
        Terminator::Return(Some(val)) => {
            write!(out, "return {}", format_ref(val))?;
        }
        Terminator::Return(None) => {
            write!(out, "return ()")?;
        }
        Terminator::Unreachable => {
            write!(out, "unreachable")?;
        }
    }
    Ok(())
}
