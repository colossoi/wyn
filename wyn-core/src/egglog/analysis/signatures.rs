//! Check signatures and capability boundaries on the selected graph.
use super::{count, typed, Query};
use crate::egglog::graph::CollectiveInput;
use crate::egglog::{output_error, OptimizeError};
use crate::LookupSet;
use egglog_engine::Value;

pub(super) fn function(query: Query<'_>, function: Value) -> Result<(), OptimizeError> {
    if query.0.is_external(function) {
        return Ok(());
    }
    let root = query.0.function_root(function)?;
    let results = count(query, "FunctionResultCount", function)?;
    if query.0.yield_count(root) != results {
        return Err(output_error("function result count differs from its signature"));
    }
    for slot in 0..results {
        let name = query.0.yield_value(root, slot)?;
        returns(
            query,
            name,
            query.required("FunctionResultType", (function, slot))?,
        )?;
        if query.0.entry_order(function).is_some() && query.enode("AccNameRef", name)?.is_some() {
            return Err(output_error(
                "entry result cannot expose an accumulator capability",
            ));
        }
    }
    Ok(())
}

pub(super) fn operation(query: Query<'_>, operation: Value) -> Result<(), OptimizeError> {
    if query.contains("CallOperation", operation)? {
        let target = query.0.call_target(operation)?;
        let parameters = if query.0.is_external(target) {
            query.0.external_parameters(target)?.into_iter().map(|param| (param.info, param.diet)).collect()
        } else {
            let root = query.0.function_root(target)?;
            query
                .0
                .parameters(root)
                .into_iter()
                .enumerate()
                .map(|(slot, name)| {
                    Ok((
                        typed(query, name)?,
                        query.required("ParameterDiet", (root, slot as i64))?,
                    ))
                })
                .collect::<Result<Vec<_>, OptimizeError>>()?
        };
        let arguments = query.0.inputs(operation);
        if arguments.len() != parameters.len() {
            return Err(output_error("call argument count differs from its signature"));
        }
        for (slot, (argument, (expected, diet))) in arguments.into_iter().zip(parameters).enumerate() {
            if !compatible(query, typed(query, argument)?, expected)? {
                return Err(output_error("call argument type differs from its signature"));
            }
            if query.required("InputDiet", (operation, slot as i64))? != diet {
                return Err(output_error("call argument diet differs from its signature"));
            }
        }
        let results = count(query, "FunctionResultCount", target)?;
        if query.0.output_count(operation) != results {
            return Err(output_error("call result count differs from its signature"));
        }
        for slot in 0..results {
            let expected = query.required("FunctionResultType", (target, slot))?;
            let output = query.0.output(operation, slot)?;
            if query.required("CallReturnType", (operation, slot))? != expected {
                return Err(output_error("call result type differs from its signature"));
            }
            returns(query, output, expected)?;
        }
    }
    if query.contains("BranchOperation", operation)? {
        boolean(query, query.0.input(operation, 0_i64)?)?;
        for edge in 0_i64..2 {
            let child = query.0.control_region(operation, edge)?;
            if query.0.yield_count(child) != query.0.output_count(operation) {
                return Err(output_error("branch result count differs from its signature"));
            }
            for slot in 0..query.0.output_count(operation) {
                let result = query.0.output(operation, slot)?;
                returns(query, result, query.required("RegionReturnType", (child, slot))?)?;
            }
        }
    }
    if query.contains("WhileLoopOperation", operation)? {
        let condition = query.loop_parameter(operation)?;
        let Some(condition) = query.constructor("ValueNameRef", condition)? else {
            return Err(output_error("loop predicate has no value identity"));
        };
        boolean(query, condition)?;
    }
    if query.contains("ParallelOperation", operation)? {
        let domain = query.0.operation_domain(operation)?;
        let rank = query.0.domain_rank(domain)?;
        let lane = query.0.control_region(operation, 0_i64)?;
        if query.0.parameter_count(lane) != rank + 1 {
            return Err(output_error("parallel parameters differ from its domain"));
        }
        let ordinal = query.name(query.0.parameter(lane, 0_i64)?)?;
        if query.0.domain_ordinal(domain)? != ordinal {
            return Err(output_error("parallel ordinal is not its lane parameter"));
        }
        for slot in 0..rank {
            let axis = query.name(query.0.parameter(lane, slot + 1)?)?;
            if query.0.domain_axis(domain, slot)? != axis {
                return Err(output_error("parallel coordinate is not its lane parameter"));
            }
        }
        let clauses = query.0.result_count(operation);
        for slot in 0..query.0.output_count(operation) {
            let clause = query.0.output_clause(operation, slot)?;
            if clause < 0 || clause >= clauses {
                return Err(output_error("parallel output refers to a missing result clause"));
            }
            let field = query.0.output_field(operation, slot)?;
            if field < 0
                || field >= query.0.collective_input_count(operation, clause, CollectiveInput::Contribution)
            {
                return Err(output_error("parallel output refers to a missing result field"));
            }
        }
        for clause in 0..clauses {
            let kind = query.0.result_kind(operation, clause)?;
            let predicate = query.0.collective_input_at(operation, clause, CollectiveInput::Predicate, 0);
            if query.enode("Compacted", kind)?.is_some() && predicate.is_none() {
                return Err(output_error("compacted result has no predicate"));
            }
            if let Some(predicate) = predicate {
                let Some(predicate) = query.constructor("ValueNameRef", predicate)? else {
                    return Err(output_error("compact predicate has no value identity"));
                };
                boolean(query, predicate)?;
            }
            if query.enode("Folded", kind)?.is_some() {
                let reducer = query.0.reducer(operation, clause)?;
                let neutral = query.0.collective_input_count(operation, clause, CollectiveInput::Neutral);
                if rank == 0
                    || query.0.parameter_count(reducer) != 2 * neutral
                    || query.0.yield_count(reducer) != neutral
                    || query.0.collective_input_count(operation, clause, CollectiveInput::Contribution)
                        != neutral
                {
                    return Err(output_error(
                        "reducer must take two accumulator tuples and preserve their arity",
                    ));
                }
                for slot in 0..neutral {
                    let initial =
                        query.0.collective_input(operation, clause, CollectiveInput::Neutral, slot)?;
                    let parameter = query.0.parameter(reducer, slot)?;
                    let other = query.0.parameter(reducer, slot + neutral)?;
                    if !compatible(query, typed(query, initial)?, typed(query, parameter)?)?
                        || !compatible(query, typed(query, parameter)?, typed(query, other)?)?
                    {
                        return Err(output_error(
                            "reducer parameter type differs from its accumulator",
                        ));
                    }
                }
            }
        }
    }
    if query.contains("AccumulatorOperation", operation)? {
        let body = query.0.control_region(operation, 0_i64)?;
        let groups = query.0.accumulator_group_count(operation);
        if query.0.parameter_count(body) < groups || query.0.yield_count(body) < groups {
            return Err(output_error("accumulator scope loses a capability"));
        }
        for group in 0..groups {
            let result = query.0.yield_value(body, group)?;
            let token = query.fields("AccNameRef", result)?[0];
            let Some(capability) = query.lookup("TokenCapability", token)? else {
                return Err(output_error(
                    "accumulator operand outside its scope or missing its capability",
                ));
            };
            let family = query.fields("TokenBindingAt", capability)?;
            if family[0] != operation || query.0.value_to_base::<i64>(family[1]) != group {
                return Err(output_error("accumulator scope returns a different capability"));
            }
        }
    }
    Ok(())
}

pub(super) fn region(query: Query<'_>, region: Value) -> Result<(), OptimizeError> {
    let mut orders = LookupSet::default();
    for operation in query.operations(region)? {
        let order = query.0.operation_order(operation)?;
        if !orders.insert(order) {
            return Err(output_error("two operation occurrences share an execution order"));
        }
    }
    for parameter in query.0.parameters(region) {
        if let Some(token) = query.enode("AccNameRef", parameter)? {
            if let Some(info) =
                query.enode("AccumulatorInfo", query.required("AccumulatorType", token[0])?)?
            {
                query.for_each("ShapeUse", |row| {
                    if row[0] == info[0] {
                        let Some(name) = query.constructor("ValueNameRef", row[1])? else {
                            return Err(output_error("accumulator shape has no value identity"));
                        };
                        if !query.contains("AvailableIn", (region, name))? {
                            return Err(output_error(
                                "accumulator parameter shape is unavailable at entry",
                            ));
                        }
                    }
                    Ok(())
                })?;
            }
        }
    }
    let mut tokens = LookupSet::default();
    for slot in 0..query.0.yield_count(region) {
        let name = query.0.yield_value(region, slot)?;
        if let Some(token) = query.enode("AccNameRef", name)? {
            if !tokens.insert(token[0]) {
                return Err(output_error("body returns the same accumulator capability twice"));
            }
        }
        returns(query, name, query.required("RegionReturnType", (region, slot))?)?;
    }
    Ok(())
}

fn boolean(query: Query<'_>, name: Value) -> Result<(), OptimizeError> {
    let info = typed(query, name)?;
    if let Some(info) = query.enode("ScalarInfo", info)? {
        if let Some(element) = query.enode("ScalarElement", info[0])? {
            if query.enode("Bool", element[0])?.is_some() {
                return Ok(());
            }
        }
    }
    Err(output_error("control predicate is not Boolean"))
}

fn returns(query: Query<'_>, name: Value, signature: Value) -> Result<(), OptimizeError> {
    let info = typed(query, name)?;
    for (declared, actual) in [
        ("ReturnScalar", "ScalarInfo"),
        ("ReturnResource", "ResourceInfo"),
        ("LogicalReturn", "LogicalValue"),
    ] {
        if let Some(expected) = query.enode(declared, signature)? {
            if let Some(actual) = query.enode(actual, info)? {
                if actual[0] == expected[0]
                    || (declared == "LogicalReturn" && logical_compatible(query, actual[0], expected[0])?)
                {
                    return Ok(());
                }
            }
        }
    }
    if let Some(expected) = query.enode("ReturnAccumulator", signature)? {
        if query.enode("AccNameRef", name)?.is_some() && info == expected[0] {
            return Ok(());
        }
    }
    Err(output_error("result binding type differs from its signature"))
}

// Signature slots preserve element structure and rank; ownership and lengths have separate proofs.
pub(super) fn compatible(query: Query<'_>, a: Value, b: Value) -> Result<bool, OptimizeError> {
    if a == b {
        return Ok(true);
    }
    let (Some(a), Some(b)) = (query.enode("LogicalValue", a)?, query.enode("LogicalValue", b)?) else {
        return Ok(false);
    };
    logical_compatible(query, a[0], b[0])
}

fn logical_compatible(query: Query<'_>, a: Value, b: Value) -> Result<bool, OptimizeError> {
    if a == b {
        return Ok(true);
    }
    if let (Some(a), Some(b)) = (query.enode("ArrayType", a)?, query.enode("ArrayType", b)?) {
        return Ok(query.vector(a[1])?.len() == query.vector(b[1])?.len()
            && logical_compatible(query, a[2], b[2])?);
    }
    if let (Some(a), Some(b)) = (query.enode("ProductType", a)?, query.enode("ProductType", b)?) {
        let a = query.vector(a[0])?;
        let b = query.vector(b[0])?;
        if a.len() != b.len() {
            return Ok(false);
        }
        for (a, b) in a.into_iter().zip(b) {
            if !logical_compatible(query, a, b)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    Ok(false)
}
