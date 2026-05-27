//! Pretty-printer for SSA programs.

use crate::ast::TypeName;
use crate::ssa::types::{ExecutionModel, Program};
use polytype::Type;
use std::fmt::Write;

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
            if *n == 1 { format!("({},)", inner.join(", ")) } else { format!("({})", inner.join(", ")) }
        }
        Type::Constructed(TypeName::Array, args) if args.len() >= 2 => {
            let elem = format_type(&args[0]);
            let size = if args.len() >= 3 { format_array_size(&args[1]) } else { "?".to_string() };
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

fn fmt_place(p: crate::ssa::types::PlaceId) -> String {
    format!("@{:?}", p)
}

fn format_values(vals: &[ValueId]) -> String {
    vals.iter().map(|v| fmt_val(*v)).collect::<Vec<_>>().join(", ")
}

fn fmt_block(b: BlockId) -> String {
    format!("{:?}", b)
}

pub fn format_program(program: &Program) -> String {
    let mut out = String::new();

    for func in &program.functions {
        format_function(&mut out, &func.name, &func.body);
        out.push('\n');
    }

    for ep in &program.entry_points {
        let model_str = match &ep.execution_model {
            ExecutionModel::Vertex => "vertex",
            ExecutionModel::Fragment => "fragment",
            ExecutionModel::Compute { .. } => "compute",
        };
        let local_size_suffix = match &ep.execution_model {
            ExecutionModel::Compute { local_size } => {
                format!(
                    " local_size({}, {}, {})",
                    local_size.0, local_size.1, local_size.2
                )
            }
            _ => String::new(),
        };
        let _ = write!(out, "entry {model_str}{local_size_suffix} ");
        format_function(&mut out, &ep.name, &ep.body);
        out.push('\n');
    }

    while out.ends_with("\n\n") {
        out.pop();
    }

    out
}

fn format_function(out: &mut String, name: &str, body: &FuncBody) {
    let params: Vec<String> = body
        .params
        .iter()
        .map(|(val, ty, _name)| format!("{}: {}", fmt_val(*val), format_type(ty)))
        .collect();
    let ret = format_type(&body.return_ty);
    let _ = writeln!(out, "func @{name}({}) -> {ret} {{", params.join(", "));

    for (bid, block) in &body.inner.blocks {
        // Skip dead blocks
        if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
            continue;
        }

        // Block header
        if block.params.is_empty() {
            let _ = writeln!(out, "  {}:", fmt_block(bid));
        } else {
            let params: Vec<String> = block
                .params
                .iter()
                .map(|&p| format!("{}: {}", fmt_val(p), format_type(body.inner.value_type(p))))
                .collect();
            let _ = writeln!(out, "  {}({}):", fmt_block(bid), params.join(", "));
        }

        // Control header as comment
        if let Some(ctrl) = body.control_headers.get(&bid) {
            match ctrl {
                ControlHeader::Loop {
                    merge,
                    continue_block,
                } => {
                    let _ = writeln!(
                        out,
                        "    # loop merge={} continue={}",
                        fmt_block(*merge),
                        fmt_block(*continue_block)
                    );
                }
                ControlHeader::Selection { merge } => {
                    let _ = writeln!(out, "    # selection merge={}", fmt_block(*merge));
                }
            }
        }

        // Instructions
        for &inst_id in &block.insts {
            let inst = body.get_inst(inst_id);
            let _ = write!(out, "    ");
            if let Some(result) = inst.result {
                let _ = write!(out, "{} = ", fmt_val(result));
            }
            format_inst_kind(out, &inst.data);
            out.push('\n');
        }

        // Terminator
        let _ = write!(out, "    ");
        format_terminator(out, &block.term);
        out.push('\n');

        out.push('\n');
    }

    out.push('}');
    out.push('\n');
}

fn format_inst_kind(out: &mut String, kind: &InstKind) {
    use crate::op::{OpTag, PureViewSource};
    match kind {
        InstKind::Op { tag, operands } => match tag {
            OpTag::Int(s) => {
                let _ = write!(out, "int {s}");
            }
            OpTag::Uint(s) => {
                let _ = write!(out, "uint {s}");
            }
            OpTag::Float(s) => {
                let _ = write!(out, "float {s}");
            }
            OpTag::Bool(b) => {
                let _ = write!(out, "bool {b}");
            }
            OpTag::Unit => {
                let _ = write!(out, "unit");
            }
            OpTag::BinOp(op) => {
                let _ = write!(
                    out,
                    "binop {op} {}, {}",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                );
            }
            OpTag::UnaryOp(op) => {
                let _ = write!(out, "unaryop {op} {}", format_ref(&operands[0]));
            }
            OpTag::Tuple(_) => {
                let _ = write!(out, "tuple ({})", format_refs(operands));
            }
            OpTag::ArrayLit(_) => {
                let _ = write!(out, "array [{}]", format_refs(operands));
            }
            OpTag::ArrayRange { has_step } => {
                let _ = write!(
                    out,
                    "range {}..{}",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                );
                if *has_step {
                    let _ = write!(out, " step {}", format_ref(&operands[2]));
                }
            }
            OpTag::Vector(_) => {
                let _ = write!(out, "vector @[{}]", format_refs(operands));
            }
            OpTag::Matrix { rows, cols } => {
                let _ = write!(out, "matrix @[");
                for r in 0..*rows {
                    if r > 0 {
                        let _ = write!(out, ", ");
                    }
                    let row = &operands[r * cols..(r + 1) * cols];
                    let _ = write!(out, "[{}]", format_refs(row));
                }
                let _ = write!(out, "]");
            }
            OpTag::Project { index } => {
                let _ = write!(out, "project {}.{index}", format_ref(&operands[0]));
            }
            OpTag::Index => {
                let _ = write!(
                    out,
                    "index {}[{}]",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                );
            }
            OpTag::Call(func) => {
                let _ = write!(out, "call @{func}({})", format_refs(operands));
            }
            OpTag::Global(name) => {
                let _ = write!(out, "global @{name}");
            }
            OpTag::Extern(name) => {
                let _ = write!(out, "extern @{name}");
            }
            OpTag::Intrinsic { id, overload_idx } => {
                let name = crate::builtins::by_id(*id).dispatch_name();
                let _ = write!(out, "intrinsic @{name}#{overload_idx}({})", format_refs(operands));
            }
            OpTag::StorageView(src) => {
                let src_str = match src {
                    PureViewSource::Storage { set, binding } => {
                        format!("storage({set}, {binding})")
                    }
                    PureViewSource::Inherited => format_ref(&operands[2]),
                    PureViewSource::Workgroup { id, count } => {
                        format!("workgroup({id}, {count})")
                    }
                };
                let _ = write!(
                    out,
                    "storage_view {src_str} {} {}",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                );
            }
            OpTag::StorageViewLen => {
                let _ = write!(out, "storage_view_len {}", format_ref(&operands[0]));
            }
            OpTag::Materialize => {
                let _ = write!(out, "materialize {}", format_ref(&operands[0]));
            }
            OpTag::DynamicExtract => {
                let _ = write!(
                    out,
                    "dynamic_extract {}[{}]",
                    format_ref(&operands[0]),
                    format_ref(&operands[1])
                );
            }
            OpTag::ViewIndex | OpTag::OutputSlot { .. } => {
                unreachable!(
                    "OpTag::{:?} is EGIR-only and must not appear in InstKind::Op",
                    tag
                );
            }
        },
        InstKind::Alloca { elem_ty, result } => {
            let _ = write!(out, "alloca {} -> {}", format_type(elem_ty), fmt_place(*result));
        }
        InstKind::Load { place } => {
            let _ = write!(out, "load {}", fmt_place(*place));
        }
        InstKind::Store { place, value } => {
            let _ = write!(out, "store {}, {}", fmt_place(*place), format_ref(value));
        }
        InstKind::ViewIndex { view, index, result } => {
            let _ = write!(
                out,
                "view_index {}[{}] -> {}",
                format_ref(view),
                format_ref(index),
                fmt_place(*result)
            );
        }
        InstKind::OutputSlot { index, result } => {
            let _ = write!(out, "output_slot {index} -> {}", fmt_place(*result));
        }
        InstKind::ControlBarrier => {
            let _ = write!(out, "control_barrier");
        }
    }
}

fn format_terminator(out: &mut String, term: &Terminator) {
    match term {
        Terminator::Branch { target, args } => {
            if args.is_empty() {
                let _ = write!(out, "br {}", fmt_block(*target));
            } else {
                let _ = write!(out, "br {}({})", fmt_block(*target), format_values(args));
            }
        }
        Terminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => {
            let then_args_str = if then_args.is_empty() {
                String::new()
            } else {
                format!("({})", format_values(then_args))
            };
            let else_args_str = if else_args.is_empty() {
                String::new()
            } else {
                format!("({})", format_values(else_args))
            };
            let _ = write!(
                out,
                "br_if {} then {}{then_args_str} else {}{else_args_str}",
                fmt_val(*cond),
                fmt_block(*then_target),
                fmt_block(*else_target),
            );
        }
        Terminator::Return(Some(val)) => {
            let _ = write!(out, "return {}", fmt_val(*val));
        }
        Terminator::Return(None) => {
            let _ = write!(out, "return ()");
        }
        Terminator::Unreachable => {
            let _ = write!(out, "unreachable");
        }
    }
}
