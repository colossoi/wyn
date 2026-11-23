//! Final host expressions for sizes and scalar arguments; never reinserted into egglog.
use super::{output_error as error, representation, OptimizeError};
use crate::builtins::catalog;
use crate::egglog::bindings::Bindings;
use crate::egglog::dataflow::ir::Code;
use crate::egglog::graph::Formula;
use crate::egglog::query::Query;
use crate::host::{Expr, ScalarExpr, ScalarSource, ScalarType};
use crate::types::{Type, TypeName};
use egglog_engine::Value;

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
        let Some(value) = values.get(&args[0]).cloned() else {
            return Ok(None);
        };
        let ty = representation::value_type(query, query.required("ValueType", args[0])?)?;
        return Ok(Some(if scalar_type(&ty) == Some(ScalarType::U32) {
            ScalarExpr::Apply {
                op: "to-i32".into(),
                ty: ScalarType::U32,
                args: vec![value],
            }
        } else {
            value
        }));
    }
    let (Some(a), Some(b)) = (
        scalar_index(query, values, args[0])?,
        scalar_index(query, values, args[1])?,
    ) else {
        return Ok(None);
    };
    let op = match kind {
        "IndexAdd" => "add",
        "IndexSub" => "sub",
        "IndexMul" => "mul",
        "IndexDiv" => "div",
        "IndexMod" => "rem",
        "IndexMin" => "min",
        "IndexMax" => "max",
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
    let _ = query;
    Ok(values.get(&value).cloned())
}

pub(super) fn remember(
    query: Query<'_>,
    function: Value,
    values: &mut Bindings<Value, ScalarExpr>,
    statement: Value,
) -> Result<(), OptimizeError> {
    let code = query.0.code(statement)?;
    let pattern = query.0.outputs(statement);
    let operands = query.0.inputs(statement);
    let atoms = operands.iter().map(|&name| query.name(name)).collect::<Result<Vec<_>, _>>()?;
    if matches!(code, Code::Branch) {
        let condition = atom(query, values, atoms[0])?;
        let yes = results(query, function, values, query.0.control_region(statement, 0_i64)?)?;
        let no = results(query, function, values, query.0.control_region(statement, 1_i64)?)?;
        if pattern.len() != yes.len() || pattern.len() != no.len() {
            return Err(error("host branch result arity mismatch"));
        }
        for ((binding, yes), no) in pattern.into_iter().zip(yes).zip(no) {
            let Some(name) = query.value_name(binding)? else {
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
            values.insert(name, value);
        }
        return Ok(());
    }
    let [binding] = pattern.as_slice() else {
        return Ok(());
    };
    let Some(name) = query.value_name(*binding)? else {
        return Ok(());
    };
    let info = query.required("ValueType", name)?;
    if matches!(code, Code::Call) {
        if let Some(result) = scalar_call(query, values, statement, &mut 0, &mut vec![function])? {
            values.insert(name, field(result, 0));
        }
        return Ok(());
    }
    let value = match code {
        Code::Constant { ty, bits } => match scalar_type(&ty) {
            Some(ScalarType::I32) => Some(ScalarExpr::I32(bits as i32)),
            Some(ScalarType::U32) => Some(ScalarExpr::U32(bits as u32)),
            Some(ScalarType::F32) => Some(ScalarExpr::F32(bits as u32)),
            Some(ScalarType::Bool) => Some(ScalarExpr::Bool(bits != 0)),
            None => None,
        },
        Code::Product => atoms
            .iter()
            .map(|&name| atom(query, values, name))
            .collect::<Result<Option<Vec<_>>, _>>()?
            .map(ScalarExpr::Tuple),
        Code::Project { index } => atom(query, values, atoms[0])?.map(|value| field(value, index)),
        Code::IndexValue => scalar_index(query, values, query.0.formula(statement, Formula::Index)?)?,
        Code::Scalar(payload) => {
            let opcode = payload.spelling();
            let builtin = opcode
                .strip_prefix("builtin:")
                .and_then(|name| name.rsplit_once(':'))
                .and_then(|(name, _)| catalog().lookup_by_any_name(name));
            let opcode = opcode
                .strip_prefix("builtin:")
                .and_then(|name| name.rsplit_once(':'))
                .map_or(opcode.as_ref(), |(name, _)| {
                    name.rsplit_once('.').map_or(name, |(_, member)| member)
                });

            let Some(args) = atoms
                .iter()
                .copied()
                .map(|value| atom(query, values, value))
                .collect::<Result<Option<Vec<_>>, _>>()?
            else {
                return Ok(());
            };
            if opcode == "vector" || opcode == "tuple" {
                values.insert(name, ScalarExpr::Tuple(args));
                return Ok(());
            }
            if let Some(index) = opcode.strip_prefix("project:").and_then(|index| index.parse().ok()) {
                if let [value] = args.as_slice() {
                    values.insert(name, field(value.clone(), index));
                }
                return Ok(());
            }
            if builtin.is_some_and(|builtin| builtin.id == catalog().known().record_with) {
                let [record, _, replacement] = args.as_slice() else {
                    return Err(error("host record update arity mismatch"));
                };
                let operand = query.0.input(statement, 1_i64)?;
                let Some(index) = query.lookup("ConstantValue", operand)? else {
                    return Ok(());
                };
                let index = query.0.value_to_base::<i64>(index);
                let info = query.fields("LogicalValue", info)?;
                let fields = query.fields("ProductType", info[0])?;
                let fields = query.vector(fields[0])?;
                if index < 0 || index as usize >= fields.len() {
                    return Err(error("host record field index out of bounds"));
                }
                values.insert(
                    name,
                    ScalarExpr::Tuple(
                        (0..fields.len())
                            .map(|i| {
                                if i == index as usize {
                                    replacement.clone()
                                } else {
                                    field(record.clone(), i)
                                }
                            })
                            .collect(),
                    ),
                );
                return Ok(());
            }
            let Some(info) = query.enode("ScalarInfo", info)? else {
                return Ok(());
            };
            let Some(element) = query.enode("ScalarElement", info[0])? else {
                return Ok(());
            };
            let mut ty = if query.enode("I32", element[0])?.is_some() {
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
            let target = super::representation::primitive_type(query, element[0])?;
            let source = atoms.first().map(|&atom| atom_type(query, function, atom)).transpose()?.flatten();
            let conversion = match (&target, &source, builtin) {
                (Type::Constructed(target, _), Some(Type::Constructed(source, _)), Some(builtin)) => {
                    catalog().conversion(target, source) == Some(builtin.id)
                }
                _ => false,
            };
            if conversion && source.as_ref().and_then(scalar_type) == Some(ScalarType::Bool) {
                let (yes, no) = match ty {
                    ScalarType::I32 => (ScalarExpr::I32(1), ScalarExpr::I32(0)),
                    ScalarType::U32 => (ScalarExpr::U32(1), ScalarExpr::U32(0)),
                    ScalarType::F32 => (ScalarExpr::F32(1f32.to_bits()), ScalarExpr::F32(0)),
                    ScalarType::Bool => (ScalarExpr::Bool(true), ScalarExpr::Bool(false)),
                };
                values.insert(
                    name,
                    ScalarExpr::If {
                        condition: Box::new(args[0].clone()),
                        yes: Box::new(yes),
                        no: Box::new(no),
                    },
                );
                return Ok(());
            }
            let op = match opcode {
                _ if conversion => match ty {
                    ScalarType::I32 => "to-i32",
                    ScalarType::U32 => "to-u32",
                    ScalarType::F32 => "to-f32",
                    ScalarType::Bool => return Ok(()),
                },
                "+" => "add",
                "-" if args.len() == 1 => "neg",
                "-" => "sub",
                "*" => "mul",
                "/" => "div",
                "%" => "rem",
                "==" => "eq",
                "!=" => "ne",
                "<" => "lt",
                "<=" => "le",
                ">" => "gt",
                ">=" => "ge",
                "&" | "&&" => "and",
                "|" | "||" => "or",
                "^" => "xor",
                "<<" => "shl",
                ">>" => "shr",
                "!" | "~" => "not",
                "index-cast" => "to-i32",
                "select" => "select",
                _ => return Ok(()),
            };
            if conversion || matches!(op, "to-i32" | "eq" | "ne" | "lt" | "le" | "gt" | "ge") {
                let Some(operand_type) = source.as_ref().and_then(scalar_type) else {
                    return Ok(());
                };
                ty = operand_type;
            }
            Some(ScalarExpr::Apply {
                op: op.into(),
                ty,
                args,
            })
        }
        _ => None,
    };
    if let Some(value) = value {
        values.insert(name, value);
    }
    Ok(())
}

fn atom_type(query: Query<'_>, _function: Value, name: Value) -> Result<Option<Type>, OptimizeError> {
    super::representation::value_type(query, query.required("ValueType", name)?).map(Some)
}

fn results(
    query: Query<'_>,
    function: Value,
    outer: &Bindings<Value, ScalarExpr>,
    region: Value,
) -> Result<Vec<Option<ScalarExpr>>, OptimizeError> {
    let mut values =
        outer.iter().map(|(&name, value)| (name, value.clone())).collect::<crate::StableMap<_, _>>().into();
    for statement in query.operations(region)? {
        remember(query, function, &mut values, statement)?;
    }
    query.0.yield_values(region).into_iter().map(|name| operand(query, &values, name)).collect()
}

pub(super) fn scalar_type(ty: &crate::types::Type) -> Option<ScalarType> {
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some(ScalarType::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(ScalarType::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(ScalarType::F32),
        Type::Constructed(TypeName::Bool, _) => Some(ScalarType::Bool),
        _ => None,
    }
}

/// Lower CPU-only scalar work into the existing host expression language.
/// Every statement is bound once, including partial computations whose result is unused.
pub(super) fn cpu_body(
    query: Query<'_>,
    function: Value,
    outer: &Bindings<Value, ScalarExpr>,
    body: Value,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    scalar_body(query, function, outer, body, &mut 0, &mut vec![function])
}

pub(super) fn cpu_loop(
    query: Query<'_>,
    function: Value,
    outer: &Bindings<Value, ScalarExpr>,
    loop_: Value,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let result = scalar_loop(query, function, outer, loop_, &mut 0, &mut vec![function])?;
    Ok(result.filter(cpu_available))
}

pub(super) fn cpu_available(value: &ScalarExpr) -> bool {
    match value {
        ScalarExpr::Read { .. } => false,
        ScalarExpr::Apply { args, .. } | ScalarExpr::Tuple(args) => args.iter().all(cpu_available),
        ScalarExpr::If { condition, yes, no } => {
            cpu_available(condition) && cpu_available(yes) && cpu_available(no)
        }
        ScalarExpr::Let { bindings, result } => {
            bindings.iter().all(|(_, value)| cpu_available(value)) && cpu_available(result)
        }
        ScalarExpr::Loop {
            initial,
            condition,
            step,
            ..
        } => cpu_available(initial) && cpu_available(condition) && cpu_available(step),
        ScalarExpr::Field { tuple, .. } => cpu_available(tuple),
        ScalarExpr::I32(_)
        | ScalarExpr::U32(_)
        | ScalarExpr::F32(_)
        | ScalarExpr::Bool(_)
        | ScalarExpr::Local(_)
        | ScalarExpr::Parameter { .. }
        | ScalarExpr::BufferLength { .. } => true,
    }
}

pub(super) fn scalar_extent(extent: &Expr) -> Option<ScalarExpr> {
    match extent {
        Expr::Scalar(value) => Some(value.clone()),
        Expr::Integer(n) => i32::try_from(*n).ok().map(ScalarExpr::I32),
        Expr::BufferLength { source, stride } => Some(ScalarExpr::BufferLength {
            source: source.clone(),
            stride: *stride,
        }),
        Expr::ReadScalar { resource, offset, .. } => Some(ScalarExpr::Read {
            source: ScalarSource::Resource(*resource),
            offset: *offset,
            ty: ScalarType::I32,
        }),
        Expr::Min(a, b) | Expr::Max(a, b) => Some(ScalarExpr::Apply {
            op: if matches!(extent, Expr::Min(..)) { "min" } else { "max" }.into(),
            ty: ScalarType::I32,
            args: vec![scalar_extent(a)?, scalar_extent(b)?],
        }),
        _ => None,
    }
}

fn field(tuple: ScalarExpr, index: usize) -> ScalarExpr {
    if let ScalarExpr::Tuple(fields) = &tuple {
        if let Some(selected) = fields.get(index).filter(|_| fields.iter().all(parameter_field)) {
            return selected.clone();
        }
    }
    ScalarExpr::Field {
        tuple: Box::new(tuple),
        index,
    }
}

fn parameter_field(value: &ScalarExpr) -> bool {
    match value {
        ScalarExpr::Parameter { .. } => true,
        ScalarExpr::Tuple(fields) => fields.iter().all(parameter_field),
        _ => false,
    }
}

// Only unconditional subexpressions establish that host launch evaluation preserves a trap.
pub(super) fn evaluates(expression: &ScalarExpr, value: &ScalarExpr) -> bool {
    expression == value
        || match expression {
            ScalarExpr::Apply { args, .. } | ScalarExpr::Tuple(args) => {
                args.iter().any(|arg| evaluates(arg, value))
            }
            ScalarExpr::Field { tuple, .. } => evaluates(tuple, value),
            ScalarExpr::If { condition, .. } => evaluates(condition, value),
            ScalarExpr::Let { bindings, result } => {
                bindings.iter().any(|(_, expression)| evaluates(expression, value))
                    || evaluates(result, value)
            }
            ScalarExpr::Loop { initial, .. } => evaluates(initial, value),
            _ => false,
        }
}

fn local(serial: &mut usize) -> String {
    let name = format!("host-scalar-{}", *serial);
    *serial += 1;
    name
}

fn operand(
    query: Query<'_>,
    values: &Bindings<Value, ScalarExpr>,
    value: Value,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let Some(name) = query.value_name(value)? else {
        return Ok(None);
    };
    atom(query, values, name)
}

fn scalar_body(
    query: Query<'_>,
    function: Value,
    outer: &Bindings<Value, ScalarExpr>,
    body: Value,
    serial: &mut usize,
    calls: &mut Vec<Value>,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let mut values: Bindings<_, _> =
        outer.iter().map(|(&name, value)| (name, value.clone())).collect::<crate::StableMap<_, _>>().into();
    let mut bindings = Vec::new();
    for statement in query.operations(body)? {
        let code = query.0.code(statement)?;
        let pattern = query.0.outputs(statement);
        let mut names = Vec::new();
        for binding in &pattern {
            let Some(name) = query.value_name(*binding)? else {
                return Ok(None);
            };
            names.push(name);
        }
        let result = if matches!(code, Code::Branch) {
            let condition = query.name(query.0.input(statement, 0_i64)?)?;
            let Some(condition) = atom(query, &values, condition)? else {
                return Ok(None);
            };
            let Some(yes) = scalar_body(
                query,
                function,
                &values,
                query.0.control_region(statement, 0_i64)?,
                serial,
                calls,
            )?
            else {
                return Ok(None);
            };
            let Some(no) = scalar_body(
                query,
                function,
                &values,
                query.0.control_region(statement, 1_i64)?,
                serial,
                calls,
            )?
            else {
                return Ok(None);
            };
            ScalarExpr::If {
                condition: Box::new(condition),
                yes: Box::new(yes),
                no: Box::new(no),
            }
        } else if matches!(code, Code::CountedLoop | Code::WhileLoop) {
            let Some(result) = scalar_loop(query, function, &values, statement, serial, calls)? else {
                return Ok(None);
            };
            result
        } else if matches!(code, Code::Call) {
            let Some(result) = scalar_call(query, &values, statement, serial, calls)? else {
                return Ok(None);
            };
            result
        } else {
            if names.is_empty() {
                return Ok(None);
            }
            remember(query, function, &mut values, statement)?;
            let Some(results) =
                names.iter().map(|name| values.get(name).cloned()).collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            ScalarExpr::Tuple(results)
        };
        let name = local(serial);
        bindings.push((name.clone(), result));
        for (i, source) in names.into_iter().enumerate() {
            values.insert(source, field(ScalarExpr::Local(name.clone()), i));
        }
    }
    let mut results = Vec::new();
    for result in query.0.yield_values(body) {
        let Some(value) = operand(query, &values, result)? else {
            return Ok(None);
        };
        results.push(value);
    }
    Ok(Some(ScalarExpr::Let {
        bindings,
        result: Box::new(ScalarExpr::Tuple(results)),
    }))
}

fn scalar_loop(
    query: Query<'_>,
    function: Value,
    outer: &Bindings<Value, ScalarExpr>,
    loop_: Value,
    serial: &mut usize,
    calls: &mut Vec<Value>,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let name = local(serial);
    let state = ScalarExpr::Local(name.clone());
    let counted = query.contains("CountedLoopOperation", loop_)?;
    let region = query.0.control_region(loop_, 0_i64)?;
    let offset = usize::from(counted);
    let mut values: Bindings<_, _> =
        outer.iter().map(|(&name, value)| (name, value.clone())).collect::<crate::StableMap<_, _>>().into();
    let mut initial = if counted { vec![ScalarExpr::I32(0)] } else { vec![] };
    let count = query.0.output_count(loop_) as usize;
    for i in 0..count {
        let parameter = query.0.parameter(region, i as i64)?;
        let Some(binding) = query.value_name(parameter)? else {
            return Ok(None);
        };
        let argument = query.0.argument(loop_, 0_i64, i as i64)?;
        let Some(value) = operand(query, outer, argument)? else {
            return Ok(None);
        };
        initial.push(value);
        values.insert(binding, field(state.clone(), i + offset));
    }
    let apply = |op: &str, args| ScalarExpr::Apply {
        op: op.into(),
        ty: ScalarType::I32,
        args,
    };
    let condition = if counted {
        let Some(count) = scalar_index(query, outer, query.0.formula(loop_, Formula::LoopCount)?)? else {
            return Ok(None);
        };
        values.insert(query.loop_parameter(loop_)?, field(state.clone(), 0));
        apply("lt", vec![field(state.clone(), 0), count])
    } else {
        let condition = query.loop_parameter(loop_)?;
        let Some(condition) = values.get(&condition).cloned() else {
            return Ok(None);
        };
        condition
    };
    let Some(mut step) = scalar_body(query, function, &values, region, serial, calls)? else {
        return Ok(None);
    };
    if counted {
        let step_name = local(serial);
        let mut fields = vec![apply("add", vec![field(state.clone(), 0), ScalarExpr::I32(1)])];
        fields.extend((0..count).map(|i| field(ScalarExpr::Local(step_name.clone()), i)));
        step = ScalarExpr::Let {
            bindings: vec![(step_name, step)],
            result: Box::new(ScalarExpr::Tuple(fields)),
        };
    }
    let repeated = ScalarExpr::Loop {
        name,
        initial: Box::new(ScalarExpr::Tuple(initial)),
        condition: Box::new(condition),
        step: Box::new(step),
    };
    let result_name = local(serial);
    let results = (0..count).map(|i| field(ScalarExpr::Local(result_name.clone()), i + offset)).collect();
    Ok(Some(ScalarExpr::Let {
        bindings: vec![(result_name, repeated)],
        result: Box::new(ScalarExpr::Tuple(results)),
    }))
}

fn scalar_call(
    query: Query<'_>,
    values: &Bindings<Value, ScalarExpr>,
    call: Value,
    serial: &mut usize,
    calls: &mut Vec<Value>,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let callee = query.0.call_target(call)?;
    let Some(root) = query.0.root_of(callee) else {
        return Ok(None);
    };
    if calls.contains(&callee) {
        return Ok(None);
    }
    let params = query.0.parameters(root);
    let arguments = query.0.inputs(call);
    if params.len() != arguments.len() {
        return Err(error("scalar call arity mismatch"));
    }
    let mut arguments_values = Bindings::default();
    for (parameter, argument) in params.into_iter().zip(arguments) {
        let Some(name) = query.value_name(parameter)? else {
            return Ok(None);
        };
        let Some(value) = operand(query, values, argument)? else {
            return Ok(None);
        };
        arguments_values.insert(name, value);
    }
    calls.push(callee);
    let result = scalar_body(query, callee, &arguments_values, root, serial, calls);
    calls.pop();
    result
}
