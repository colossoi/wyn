//! Pretty-printer for SSA programs.
//!
//! Formats an `SsaProgram` as a human-readable text representation suitable for
//! debugging and inspection.

use crate::ast::TypeName;
use crate::tlc::to_ssa::{ExecutionModel, SsaProgram};
use polytype::Type;
use std::fmt::Write;

use super::ssa::*;

/// Format a type concisely.
fn format_type(ty: &Type<TypeName>) -> String {
    match ty {
        Type::Constructed(TypeName::Int(bits), _) => format!("i{bits}"),
        Type::Constructed(TypeName::UInt(bits), _) => format!("u{bits}"),
        Type::Constructed(TypeName::Float(bits), _) => format!("f{bits}"),
        Type::Constructed(TypeName::Bool, _) => "bool".to_string(),
        Type::Constructed(TypeName::Str(s), _) => panic!("BUG: unexpected Str type in SSA: {:?}", s),
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

/// Extract array/vector size as a string.
fn format_array_size(size_ty: &Type<TypeName>) -> String {
    match size_ty {
        Type::Constructed(TypeName::Size(n), _) => n.to_string(),
        _ => "?".to_string(),
    }
}

/// Format a list of value IDs as comma-separated.
fn format_values(vals: &[ValueId]) -> String {
    vals.iter().map(|v| format!("%{}", v.0)).collect::<Vec<_>>().join(", ")
}

/// Format an `SsaProgram` as a readable text representation.
pub fn format_program(program: &SsaProgram) -> String {
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

    // Trim trailing newlines to a single one
    while out.ends_with("\n\n") {
        out.pop();
    }

    out
}

/// Format a single function body.
fn format_function(out: &mut String, name: &str, body: &FuncBody) {
    // Function signature
    let params: Vec<String> =
        body.params.iter().map(|(val, ty, _name)| format!("%{}: {}", val.0, format_type(ty))).collect();
    let ret = format_type(&body.return_ty);
    let _ = writeln!(out, "func @{name}({}) -> {ret} {{", params.join(", "));

    // Blocks
    for (i, block) in body.blocks.iter().enumerate() {
        let block_id = BlockId(i as u32);
        if block.is_dead() {
            continue;
        }

        // Block header
        if block.params.is_empty() {
            let _ = writeln!(out, "  {block_id}:");
        } else {
            let params: Vec<String> =
                block.params.iter().map(|p| format!("%{}: {}", p.value.0, format_type(&p.ty))).collect();
            let _ = writeln!(out, "  {block_id}({}):", params.join(", "));
        }

        // Control header as comment
        if let Some(ref ctrl) = block.control {
            match ctrl {
                ControlHeader::Loop {
                    merge,
                    continue_block,
                } => {
                    let _ = writeln!(out, "    # loop merge={merge} continue={continue_block}");
                }
                ControlHeader::Selection { merge } => {
                    let _ = writeln!(out, "    # selection merge={merge}");
                }
            }
        }

        // Instructions
        for &inst_id in &block.insts {
            let inst = body.get_inst(inst_id);
            let _ = write!(out, "    ");
            if let Some(result) = inst.result {
                let _ = write!(out, "%{} = ", result.0);
            }
            format_inst_kind(out, &inst.kind);
            out.push('\n');
        }

        // Terminator
        if let Some(ref term) = block.terminator {
            let _ = write!(out, "    ");
            format_terminator(out, term);
            out.push('\n');
        }

        out.push('\n');
    }

    out.push('}');
    out.push('\n');
}

/// Format an instruction kind.
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
            let _ = write!(out, "binop {op} %{}, %{}", lhs.0, rhs.0);
        }
        InstKind::UnaryOp { op, operand } => {
            let _ = write!(out, "unaryop {op} %{}", operand.0);
        }
        InstKind::Tuple(vals) => {
            let _ = write!(out, "tuple ({})", format_values(vals));
        }
        InstKind::ArrayLit { elements } => {
            let _ = write!(out, "array [{}]", format_values(elements));
        }
        InstKind::ArrayRange { start, len, step } => {
            let _ = write!(out, "range %{}..%{}", start.0, len.0);
            if let Some(step) = step {
                let _ = write!(out, " step %{}", step.0);
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
            let _ = write!(out, "project %{}.{index}", base.0);
        }
        InstKind::Index { base, index } => {
            let _ = write!(out, "index %{}[%{}]", base.0, index.0);
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
        InstKind::Alloca { elem_ty, .. } => {
            let _ = write!(out, "alloca {}", format_type(elem_ty));
        }
        InstKind::Load { ptr, .. } => {
            let _ = write!(out, "load %{}", ptr.0);
        }
        InstKind::Store { ptr, value, .. } => {
            let _ = write!(out, "store %{}, %{}", ptr.0, value.0);
        }
        InstKind::StorageView { source, offset, len } => {
            let src = match source {
                ViewSource::Storage { set, binding } => format!("storage({set}, {binding})"),
                ViewSource::Inherited { parent } => format!("%{}", parent.0),
            };
            let _ = write!(out, "storage_view {src} %{} %{}", offset.0, len.0);
        }
        InstKind::StorageViewIndex { view, index } => {
            let _ = write!(out, "storage_view_index %{}[%{}]", view.0, index.0);
        }
        InstKind::StorageViewLen { view } => {
            let _ = write!(out, "storage_view_len %{}", view.0);
        }
        InstKind::OutputPtr { index } => {
            let _ = write!(out, "output_ptr {index}");
        }
        InstKind::Soac(soac) => {
            format_soac(out, soac);
        }
    }
}

/// Format a SOAC instruction.
fn format_soac(out: &mut String, soac: &SsaSoac) {
    match soac {
        SsaSoac::Map {
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
        SsaSoac::Reduce {
            func,
            input,
            init,
            captures,
            ..
        } => {
            let _ = write!(out, "soac.reduce @{func}(%{}, %{})", input.0, init.0);
            if !captures.is_empty() {
                let _ = write!(out, " captures=[{}]", format_values(captures));
            }
        }
        SsaSoac::Scan {
            func,
            input,
            init,
            captures,
            ..
        } => {
            let _ = write!(out, "soac.scan @{func}(%{}, %{})", input.0, init.0);
            if !captures.is_empty() {
                let _ = write!(out, " captures=[{}]", format_values(captures));
            }
        }
    }
}

/// Format a block terminator.
fn format_terminator(out: &mut String, term: &Terminator) {
    match term {
        Terminator::Branch { target, args } => {
            if args.is_empty() {
                let _ = write!(out, "br {target}");
            } else {
                let _ = write!(out, "br {target}({})", format_values(args));
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
                "br_if %{} then {then_target}{then_args_str} else {else_target}{else_args_str}",
                cond.0
            );
        }
        Terminator::Return(val) => {
            let _ = write!(out, "return %{}", val.0);
        }
        Terminator::ReturnUnit => {
            let _ = write!(out, "return ()");
        }
        Terminator::Unreachable => {
            let _ = write!(out, "unreachable");
        }
    }
}
