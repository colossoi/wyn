//! Import arithmetic syntax once; native demand follows the source graph.
use crate::builtins::{by_id, catalog, Purity};
use crate::egglog::source::Identities;
use crate::egglog::OptimizeError;
use crate::tlc::{TermKind, VarRef};
use crate::types::{Type, TypeName};
use egglog_engine::{sort::VecContainer, Core, FullState, Write};

pub(super) fn syntax(
    mut sink: FullState<'_, '_>,
    identities: &Identities<'_>,
) -> Result<(), OptimizeError> {
    for (&source, &term) in &identities.origins {
        sink.add("ExprSyntax", source)?;
        let Some(token) = identities.types.get(&term.ty) else {
            return Err(OptimizeError::Output("scalar source has no type identity".into()));
        };
        let ty = sink.add("TypeId", token)?;
        let primitive = match &term.ty {
            Type::Constructed(TypeName::Int(bits), _) => {
                sink.add("ExprIntegerType", ty)?;
                Some(format!("i{bits}"))
            }
            Type::Constructed(TypeName::UInt(bits), _) => {
                sink.add("ExprIntegerType", ty)?;
                Some(format!("u{bits}"))
            }
            Type::Constructed(TypeName::Float(32), _) => Some("f32".into()),
            Type::Constructed(TypeName::Bool, _) => Some("bool".into()),
            _ => None,
        };
        if let Some(name) = primitive {
            sink.add("ExprPrimitiveType", (ty, name.as_str()))?;
        }
        let bits = match &term.kind {
            TermKind::IntLit(text) => Some(
                text.parse::<i64>()
                    .or_else(|_| text.parse::<u64>().map(|v| v as i64))
                    .map_err(|_| crate::egglog::output_error("invalid integer literal"))?,
            ),
            TermKind::FloatLit(value) => Some(i64::from(value.to_bits())),
            TermKind::BoolLit(value) => Some(i64::from(*value)),
            _ => None,
        };
        if let Some(bits) = bits {
            sink.add("ExprLiteralBits", (source, bits))?;
        }
        if matches!(term.kind, TermKind::UnitLit) {
            let args = sink.container_to_value(VecContainer {
                data: vec![],
                do_rebuild: true,
            });
            sink.set("SourceArgumentsOf", source, args)?;
            sink.set("SourceConstruct", source, "unit")?;
        }
        let TermKind::App { func, .. } = &term.kind else {
            continue;
        };
        let (operator, retained, safe) = match &func.kind {
            TermKind::BinOp(op) => (op.op.symbol().to_string(), false, op.op.is_speculatable()),
            TermKind::UnOp(op) => (op.op.symbol().to_string(), false, true),
            TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                if *id == catalog().known().length || *id == catalog().known().slice {
                    continue;
                }
                let builtin = by_id(*id);
                let Some(overload) = builtin.overloads().get(*overload_idx) else {
                    return Err(OptimizeError::Output("invalid scalar builtin overload".into()));
                };
                let operator = format!("builtin:{}:{overload_idx}", builtin.dispatch_name());
                if *id == catalog().known().record_with {
                    sink.add("ExprRecordUpdateOperator", operator.as_str())?;
                }
                (
                    operator,
                    builtin.raw.purity != Purity::Pure || !overload.lowering.is_reusable(),
                    overload.lowering.is_speculatable(),
                )
            }
            _ => continue,
        };
        sink.add("ExprSourceOperator", (source, operator.as_str(), retained, safe))?;
    }
    Ok(())
}
