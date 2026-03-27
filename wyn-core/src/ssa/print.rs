//! Pretty-printer for SSA programs.

use crate::ast::TypeName;
use crate::ssa::types::{ExecutionModel, Program};
use polytype::Type;
use std::fmt::Write;

use super::types::*;

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
    match kind {
        InstKind::Int(s) => {
            let _ = write!(out, "int {s}");
        }
        InstKind::Float(s) => {
            let _ = write!(out, "float {s}");
        }
        InstKind::Bool(b) => {
            let _ = write!(out, "bool {b}");
        }
        InstKind::Unit => {
            let _ = write!(out, "unit");
        }
        InstKind::String(s) => {
            let _ = write!(out, "string \"{s}\"");
        }
        InstKind::BinOp { op, lhs, rhs } => {
            let _ = write!(out, "binop {op} {}, {}", fmt_val(*lhs), fmt_val(*rhs));
        }
        InstKind::UnaryOp { op, operand } => {
            let _ = write!(out, "unaryop {op} {}", fmt_val(*operand));
        }
        InstKind::Tuple(vals) => {
            let _ = write!(out, "tuple ({})", format_values(vals));
        }
        InstKind::ArrayLit { elements } => {
            let _ = write!(out, "array [{}]", format_values(elements));
        }
        InstKind::ArrayRange { start, len, step } => {
            let _ = write!(out, "range {}..{}", fmt_val(*start), fmt_val(*len));
            if let Some(step) = step {
                let _ = write!(out, " step {}", fmt_val(*step));
            }
        }
        InstKind::Vector(vals) => {
            let _ = write!(out, "vector @[{}]", format_values(vals));
        }
        InstKind::Matrix(rows) => {
            let _ = write!(out, "matrix @[");
            for (i, row) in rows.iter().enumerate() {
                if i > 0 {
                    let _ = write!(out, ", ");
                }
                let _ = write!(out, "[{}]", format_values(row));
            }
            let _ = write!(out, "]");
        }
        InstKind::Project { base, index } => {
            let _ = write!(out, "project {}.{index}", fmt_val(*base));
        }
        InstKind::Index { base, index } => {
            let _ = write!(out, "index {}[{}]", fmt_val(*base), fmt_val(*index));
        }
        InstKind::Call { func, args } => {
            let _ = write!(out, "call @{func}({})", format_values(args));
        }
        InstKind::Global(name) => {
            let _ = write!(out, "global @{name}");
        }
        InstKind::Extern(name) => {
            let _ = write!(out, "extern @{name}");
        }
        InstKind::Intrinsic { name, args } => {
            let _ = write!(out, "intrinsic @{name}({})", format_values(args));
        }
        InstKind::Alloca { elem_ty } => {
            let _ = write!(out, "alloca {}", format_type(elem_ty));
        }
        InstKind::Load { ptr } => {
            let _ = write!(out, "load {}", fmt_val(*ptr));
        }
        InstKind::Store { ptr, value } => {
            let _ = write!(out, "store {}, {}", fmt_val(*ptr), fmt_val(*value));
        }
        InstKind::StorageView { source, offset, len } => {
            let src = match source {
                ViewSource::Storage { set, binding } => format!("storage({set}, {binding})"),
                ViewSource::Inherited { parent } => fmt_val(*parent),
            };
            let _ = write!(out, "storage_view {src} {} {}", fmt_val(*offset), fmt_val(*len));
        }
        InstKind::StorageViewIndex { view, index } => {
            let _ = write!(out, "storage_view_index {}[{}]", fmt_val(*view), fmt_val(*index));
        }
        InstKind::StorageViewLen { view } => {
            let _ = write!(out, "storage_view_len {}", fmt_val(*view));
        }
        InstKind::OutputPtr { index } => {
            let _ = write!(out, "output_ptr {index}");
        }
        InstKind::Soac(soac) => {
            format_soac(out, soac);
        }
        InstKind::Materialize { value } => {
            let _ = write!(out, "materialize {}", fmt_val(*value));
        }
        InstKind::DynamicExtract { base, index } => {
            let _ = write!(out, "dynamic_extract {}[{}]", fmt_val(*base), fmt_val(*index));
        }
    }
}

fn format_soac(out: &mut String, soac: &Soac) {
    match soac {
        Soac::Map {
            func,
            inputs,
            captures,
            ..
        } => {
            let _ = write!(out, "soac.map @{func}({})", format_values(inputs));
            if !captures.is_empty() {
                let _ = write!(out, " captures=[{}]", format_values(captures));
            }
        }
        Soac::Reduce {
            func,
            input,
            init,
            captures,
            ..
        } => {
            let _ = write!(
                out,
                "soac.reduce @{func}({}, {})",
                fmt_val(*input),
                fmt_val(*init)
            );
            if !captures.is_empty() {
                let _ = write!(out, " captures=[{}]", format_values(captures));
            }
        }
        Soac::Scan {
            func,
            input,
            init,
            captures,
            ..
        } => {
            let _ = write!(out, "soac.scan @{func}({}, {})", fmt_val(*input), fmt_val(*init));
            if !captures.is_empty() {
                let _ = write!(out, " captures=[{}]", format_values(captures));
            }
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
