//! Final host expressions for sizes and scalar arguments; never reinserted into egglog.
use super::{error, OptimizeError};
use crate::egglog::bindings::Bindings;
use crate::egglog::query::Query;
use crate::host::{Expr, ScalarExpr, ScalarType};
use egglog_engine::{sort::S, Value};

pub(super) fn index(
    query: Query<'_>,
    values: &Bindings<Value, ScalarExpr>,
    expression: Value,
) -> Result<Expr, OptimizeError> {
    let (kind, args) = query.application(
        expression,
        &[
            "IndexConstant",
            "IndexValue",
            "IndexAdd",
            "IndexSub",
            "IndexMul",
            "IndexDiv",
            "IndexMod",
            "IndexMin",
            "IndexMax",
        ],
    )?;
    if kind == "IndexConstant" {
        return Ok(Expr::Integer(query.0.value_to_base::<i64>(args[0])));
    }
    if kind == "IndexValue" {
        let Some(value) = values.get(&args[0]) else {
            return Err(error(format!(
                "allocation or launch index {:?} has no host value",
                args[0]
            )));
        };
        return Ok(Expr::Scalar(value.clone()));
    }
    let a = Box::new(index(query, values, args[0])?);
    let b = Box::new(index(query, values, args[1])?);
    Ok(match kind {
        "IndexAdd" => Expr::Add(a, b),
        "IndexSub" => Expr::Subtract(a, b),
        "IndexMul" => Expr::Multiply(a, b),
        "IndexDiv" => Expr::Floor(a, b),
        "IndexMin" => Expr::Min(a, b),
        "IndexMax" => Expr::Max(a, b),
        "IndexMod" => Expr::Subtract(
            a.clone(),
            Box::new(Expr::Multiply(Box::new(Expr::Floor(a, b.clone())), b)),
        ),
        _ => unreachable!(),
    })
}

pub(super) fn guarded_index(
    query: Query<'_>,
    values: &Bindings<Value, ScalarExpr>,
    expression: Value,
    guard: Option<&ScalarExpr>,
) -> Result<Expr, OptimizeError> {
    let Some(guard) = guard else {
        return index(query, values, expression);
    };
    let Some(value) = scalar_index(query, values, expression)? else {
        return Err(error("guarded memory index has no host expression"));
    };
    Ok(Expr::Scalar(ScalarExpr::If {
        condition: Box::new(guard.clone()),
        yes: Box::new(value),
        no: Box::new(ScalarExpr::I32(0)),
    }))
}

pub(super) fn scalar_index(
    query: Query<'_>,
    values: &Bindings<Value, ScalarExpr>,
    expression: Value,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let (kind, args) = query.application(
        expression,
        &[
            "IndexConstant",
            "IndexValue",
            "IndexAdd",
            "IndexSub",
            "IndexMul",
            "IndexDiv",
            "IndexMod",
            "IndexMin",
            "IndexMax",
        ],
    )?;
    if kind == "IndexConstant" {
        let n = query.0.value_to_base::<i64>(args[0]);
        return Ok(i32::try_from(n).ok().map(ScalarExpr::I32));
    }
    if kind == "IndexValue" {
        return Ok(values.get(&args[0]).cloned());
    }
    let (Some(a), Some(b)) = (
        scalar_index(query, values, args[0])?,
        scalar_index(query, values, args[1])?,
    ) else {
        return Ok(None);
    };
    if matches!(kind, "IndexMin" | "IndexMax") {
        let condition = ScalarExpr::Apply {
            op: if kind == "IndexMin" { "lt" } else { "gt" }.into(),
            ty: ScalarType::Bool,
            args: vec![a.clone(), b.clone()],
        };
        return Ok(Some(ScalarExpr::If {
            condition: Box::new(condition),
            yes: Box::new(a),
            no: Box::new(b),
        }));
    }
    let op = match kind {
        "IndexAdd" => "add",
        "IndexSub" => "sub",
        "IndexMul" => "mul",
        "IndexDiv" => "div",
        "IndexMod" => "rem",
        _ => unreachable!(),
    };
    Ok(Some(ScalarExpr::Apply {
        op: op.into(),
        ty: ScalarType::I32,
        args: vec![a, b],
    }))
}

pub(super) fn atom(
    query: Query<'_>,
    values: &Bindings<Value, ScalarExpr>,
    value: Value,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    if let Some(var) = query.enode("Var", value)? {
        return Ok(values.get(&var[0]).cloned());
    }
    let literal = query.fields("LiteralBits", value)?;
    let n = query.0.value_to_base::<i64>(literal[1]);
    let (ty, _) = query.application(
        literal[0],
        &[
            "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "F16", "F32", "F64", "Bool", "UnitType",
        ],
    )?;
    Ok(match ty {
        "I32" => Some(ScalarExpr::I32(n as i32)),
        "U32" => Some(ScalarExpr::U32(n as u32)),
        "F32" => Some(ScalarExpr::F32(n as u32)),
        "Bool" => Some(ScalarExpr::Bool(n != 0)),
        _ => None,
    })
}

pub(super) fn remember(
    query: Query<'_>,
    values: &mut Bindings<Value, ScalarExpr>,
    statement: Value,
) -> Result<(), OptimizeError> {
    let statement = query.fields("Let", statement)?;
    if let Some(branch) = query.enode("If", statement[3])? {
        let condition = atom(query, values, branch[0])?;
        let yes = results(query, values, branch[2])?;
        let no = results(query, values, branch[3])?;
        let pattern = query.vector(statement[1])?;
        if pattern.len() != yes.len() || pattern.len() != no.len() {
            return Err(error("host branch result arity mismatch"));
        }
        for ((binding, yes), no) in pattern.into_iter().zip(yes).zip(no) {
            let Some(binding) = query.enode("BindValue", binding)? else {
                continue;
            };
            let (Some(yes), Some(no)) = (yes, no) else {
                continue;
            };
            let value = if yes == no {
                yes
            } else if let Some(condition) = &condition {
                ScalarExpr::If {
                    condition: Box::new(condition.clone()),
                    yes: Box::new(yes),
                    no: Box::new(no),
                }
            } else {
                continue;
            };
            values.insert(binding[0], value);
        }
        return Ok(());
    }
    let pattern = query.vector(statement[1])?;
    let [binding] = pattern.as_slice() else {
        return Ok(());
    };
    let Some(binding) = query.enode("BindValue", *binding)? else {
        return Ok(());
    };
    let expression = statement[3];
    let value = if let Some(fields) = query.enode("IndexOp", expression)? {
        scalar_index(query, values, fields[0])?
    } else if let Some(fields) = query.enode("Use", expression)? {
        if let Some(operand) = query.enode("ValueOperand", fields[0])? {
            atom(query, values, operand[0])?
        } else if let Some(operand) = query.enode("IndexOperand", fields[0])? {
            scalar_index(query, values, operand[0])?
        } else {
            None
        }
    } else if let Some(fields) = query.enode("PrimOp", expression)? {
        let opcode = query.0.value_to_base::<S>(fields[0]);
        let Some(args) = query
            .vector(fields[1])?
            .into_iter()
            .map(|value| atom(query, values, value))
            .collect::<Result<Option<Vec<_>>, _>>()?
        else {
            return Ok(());
        };
        let Some(info) = query.enode("ScalarInfo", binding[1])? else {
            return Ok(());
        };
        let Some(element) = query.enode("ScalarElement", info[0])? else {
            return Ok(());
        };
        let ty = if query.enode("I32", element[0])?.is_some() {
            ScalarType::I32
        } else if query.enode("U32", element[0])?.is_some() {
            ScalarType::U32
        } else if query.enode("F32", element[0])?.is_some() {
            ScalarType::F32
        } else if query.enode("Bool", element[0])?.is_some() {
            ScalarType::Bool
        } else {
            return Ok(());
        };
        let op = match opcode.as_str() {
            "+" => "add",
            "-" if args.len() == 1 => "neg",
            "-" => "sub",
            "*" => "mul",
            "/" => "div",
            "%" => "rem",
            "index-cast" => "to-i32",
            _ => return Ok(()),
        };
        Some(ScalarExpr::Apply {
            op: op.into(),
            ty,
            args,
        })
    } else {
        None
    };
    if let Some(value) = value {
        values.insert(binding[0], value);
    }
    Ok(())
}

pub(super) fn operand(
    query: Query<'_>,
    values: &Bindings<Value, ScalarExpr>,
    value: Value,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    if let Some(operand) = query.enode("ValueOperand", value)? {
        atom(query, values, operand[0])
    } else if let Some(operand) = query.enode("IndexOperand", value)? {
        scalar_index(query, values, operand[0])
    } else {
        Ok(None)
    }
}

fn results(
    query: Query<'_>,
    outer: &Bindings<Value, ScalarExpr>,
    block: Value,
) -> Result<Vec<Option<ScalarExpr>>, OptimizeError> {
    let mut values =
        outer.iter().map(|(&name, value)| (name, value.clone())).collect::<crate::LookupMap<_, _>>().into();
    let block = query.fields("Block", block)?;
    for statement in query.vector(block[0])? {
        remember(query, &mut values, statement)?;
    }
    query
        .vector(block[1])?
        .into_iter()
        .map(|result| {
            let result = query.fields("Yield", result)?;
            if let Some(operand) = query.enode("ValueOperand", result[1])? {
                atom(query, &values, operand[0])
            } else if let Some(operand) = query.enode("IndexOperand", result[1])? {
                scalar_index(query, &values, operand[0])
            } else {
                Ok(None)
            }
        })
        .collect()
}

pub(super) fn scalar_type(ty: &crate::types::Type) -> Option<ScalarType> {
    use crate::types::{Type, TypeName};
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some(ScalarType::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(ScalarType::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(ScalarType::F32),
        Type::Constructed(TypeName::Bool, _) => Some(ScalarType::Bool),
        _ => None,
    }
}
