//! Shared representation, validation, and construction of canonical SOAC lambdas.
//!
//! Semantic construction and the explicit physical-call adapter use the same
//! identity/region contract and preserve logical result boundaries.

use crate::egir;
use polytype::Type;
use smallvec::smallvec;

use crate::ast::{Span, TypeName};
use crate::egir::program::{fresh_region_name, Func, ProgramIdentities};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, EGraph, GraphResource, OperandRef,
    Parameters, PureOp, ResultBinding, SegBody, Semantic, SkeletonTerminator, ValueId, ValueKind,
    WynLanguage,
};
use crate::flow::BlockId;
use crate::FunctionId;

mod physical;
pub(crate) use physical::{emit_physical_call, PhysicalCallables};

#[cfg(test)]
#[path = "lambda_tests.rs"]
mod tests;

/// The implementation of a canonical SOAC lambda.
#[derive(Clone, Debug)]
pub enum LambdaBody {
    /// The lambda returns its parameters unchanged and has no concrete region.
    Identity,
    /// Executable scalar dataflow with explicit captures.
    Region(SegBody),
}

/// A first-order lambda shared by Screma, Filter, and Hist.
///
/// This is deliberately distinct from `tlc::Lambda`: its higher-order meaning
/// has already been eliminated, captures are explicit, and an identity lambda
/// need not allocate a synthetic EGIR region.
#[derive(Clone, Debug)]
pub struct Lambda {
    pub body: LambdaBody,
    pub parameter_types: Vec<Type<TypeName>>,
    pub result_types: Vec<Type<TypeName>>,
}

impl Lambda {
    pub fn identity(types: Vec<Type<TypeName>>) -> Self {
        Self {
            parameter_types: types.clone(),
            result_types: types,
            body: LambdaBody::Identity,
        }
    }

    pub fn region(
        body: SegBody,
        parameter_types: Vec<Type<TypeName>>,
        result_types: Vec<Type<TypeName>>,
    ) -> Self {
        Self {
            body: LambdaBody::Region(body),
            parameter_types,
            result_types,
        }
    }

    pub fn is_identity(&self) -> bool {
        matches!(self.body, LambdaBody::Identity)
    }

    pub fn seg_body(&self) -> Option<&SegBody> {
        match &self.body {
            LambdaBody::Identity => None,
            LambdaBody::Region(body) => Some(body),
        }
    }

    pub fn seg_body_mut(&mut self) -> Option<&mut SegBody> {
        match &mut self.body {
            LambdaBody::Identity => None,
            LambdaBody::Region(body) => Some(body),
        }
    }

    pub(crate) fn captures(&self) -> &[OperandRef] {
        match &self.body {
            LambdaBody::Identity => &[],
            LambdaBody::Region(body) => &body.captures,
        }
    }

    pub(crate) fn capture_count(&self) -> usize {
        self.captures().len()
    }
    pub(crate) fn validate(&self, role: &str) -> Result<(), String> {
        if self.is_identity() && self.parameter_types != self.result_types {
            return Err(format!(
                "{role} identity lambda has signature {:?} -> {:?}",
                self.parameter_types, self.result_types
            ));
        }
        Ok(())
    }

    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        for ty in &mut self.parameter_types {
            visit(ty);
        }
        for ty in &mut self.result_types {
            visit(ty);
        }
    }

    /// Validate an associative operator's componentwise (a, a) -> a contract.
    pub(crate) fn validate_operator(&self, role: &str, neutral_count: usize) -> Result<(), String> {
        if self.is_identity() {
            return Err(format!("{role} operator is identity"));
        }
        if neutral_count == 0 {
            return Err(format!("{role} has no neutral values"));
        }
        if self.result_types.len() != neutral_count {
            return Err(format!(
                "{role} has {neutral_count} neutral values but returns {} values",
                self.result_types.len()
            ));
        }
        if self.parameter_types.len() != neutral_count * 2 {
            return Err(format!(
                "{role} operator must have {} parameters, found {}",
                neutral_count * 2,
                self.parameter_types.len()
            ));
        }
        let (left, right) = self.parameter_types.split_at(neutral_count);
        if left != right || left != self.result_types {
            return Err(format!(
                "{role} operator must have type (a, a) -> a, found ({left:?}, {right:?}) -> {:?}",
                self.result_types
            ));
        }
        Ok(())
    }
}

pub(crate) fn named_parameters<R: GraphResource>(
    types: &[Type<TypeName>],
    prefix: &str,
) -> Parameters<R, Type<TypeName>> {
    Parameters::from_ordered(
        types.iter().enumerate().map(|(index, ty)| {
            callable_parameter::<R, WynLanguage>(format!("{prefix}_{index}"), ty.clone())
        }),
    )
}

pub(crate) fn function_parameters<R: GraphResource>(
    graph: &mut EGraph<Semantic<R>>,
    params: &Parameters<R, Type<TypeName>>,
) -> Vec<OperandRef> {
    params
        .iter_with_ids()
        .map(|(id, parameter)| graph.add_parameter(id, parameter.representation()))
        .collect()
}
pub(crate) fn result_type(types: &[Type<TypeName>]) -> Type<TypeName> {
    match types {
        [ty] => ty.clone(),
        _ => Type::Constructed(TypeName::Tuple(types.len()), types.to_vec()),
    }
}

pub(crate) fn pack_results<P: egir::ir::Family>(
    graph: &mut EGraph<P>,
    results: &[ValueId],
    types: &[Type<TypeName>],
) -> ValueId {
    debug_assert_eq!(results.len(), types.len());
    match results {
        [result] => *result,
        _ => {
            let ty = result_type(types);
            let binding = ResultBinding::product(
                ty.clone(),
                results.iter().zip(types).map(|(&value, ty)| {
                    egir::graph_ops::bind_physical_result_value(graph, ty.clone(), value)
                }),
            );
            let value = graph.intern_pure(
                PureOp::Tuple(results.len()),
                results.iter().copied().collect(),
                ty,
                None,
            );
            egir::graph_ops::register_result_origin_tree(graph, value, &binding);
            value
        }
    }
}

pub(crate) fn unpack_results<P: egir::ir::Family>(
    graph: &mut EGraph<P>,
    result: ValueId,
    types: &[Type<TypeName>],
) -> Vec<ValueId> {
    match types {
        [_] => vec![result],
        _ => types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                let op = PureOp::Project { index: index as u32 };
                let operands = smallvec![result];
                graph
                    .try_algebraic_fold(&op, &operands, ty)
                    .unwrap_or_else(|| graph.intern_pure(op, operands, ty.clone(), None))
            })
            .collect(),
    }
}

pub(crate) fn logical_result_fields(
    result: &ResultBinding<Type<TypeName>>,
    result_types: &[Type<TypeName>],
) -> Vec<ResultBinding<Type<TypeName>>> {
    match result_types {
        [] => Vec::new(),
        [_] => vec![result.clone()],
        _ => {
            let fields = result.top_level_fields();
            assert_eq!(fields.len(), result_types.len());
            fields
        }
    }
}

pub(crate) fn result_argument_values<P: egir::ir::Family>(
    graph: &mut EGraph<P>,
    results: &[ResultBinding<Type<TypeName>>],
) -> Vec<ValueId> {
    results
        .iter()
        .map(|result| {
            egir::graph_ops::result_argument_value(graph, result)
                .expect("lambda result must have an argument representation")
        })
        .collect()
}

/// Emit a lambda application whose region name has already been resolved.
///
/// Identity lambdas do not have a callable region and simply return their
/// arguments. Region-lambda callers must append captures to `arguments`.
pub(crate) fn emit_call<R: GraphResource>(
    graph: &mut EGraph<Semantic<R>>,
    block: BlockId,
    lambda: &Lambda,
    callee: Option<&Func<Semantic<R>>>,
    arguments: Vec<OperandRef>,
) -> Vec<ResultBinding<Type<TypeName>>> {
    if lambda.is_identity() {
        debug_assert_eq!(arguments.len(), lambda.result_types.len());
        return arguments
            .into_iter()
            .zip(&lambda.result_types)
            .map(|(argument, ty)| {
                let value = argument.value().expect("identity lambda arguments are values or views");
                let abi = by_value_function_result::<WynLanguage>(ty.clone());
                egir::graph_ops::bind_by_value_result(graph, &abi, value)
            })
            .collect();
    }
    let function = lambda.seg_body().expect("region lambda has no callable body").region;
    let callee = callee.expect("region lambda call requires its canonical function boundary");
    assert_eq!(
        callee.region, function,
        "lambda and function boundary disagree on region identity"
    );
    let (_, result) = graph
        .emit_call(
            block,
            function,
            callee.params(),
            callee.result(),
            arguments,
            callee.effects(),
            None,
            None,
        )
        .expect("lambda call must match its canonical function boundary");
    logical_result_fields(&result, &lambda.result_types)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_function<R: GraphResource>(
    mut graph: EGraph<Semantic<R>>,
    return_block: BlockId,
    region: FunctionId,
    name: String,
    span: Span,
    params: Parameters<R, Type<TypeName>>,
    result_types: &[Type<TypeName>],
    results: &[ValueId],
) -> Func<Semantic<R>> {
    let result = pack_results(&mut graph, results, result_types);
    let result_abi = by_value_function_result::<WynLanguage>(result_type(result_types));
    let result = egir::graph_ops::bind_by_value_result(&mut graph, &result_abi, result);
    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(Some(result));
    let effects = if graph.has_ordered_effects() { CallEffects::General } else { CallEffects::Pure };
    Func::<Semantic<R>>::new(region, name, span, None, params, result_abi, effects, graph)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_region_lambda<R: GraphResource>(
    identities: &mut ProgramIdentities,
    scope: &str,
    label: &str,
    span: Span,
    graph: EGraph<Semantic<R>>,
    return_block: BlockId,
    params: Parameters<R, Type<TypeName>>,
    captures: Vec<OperandRef>,
    parameter_types: Vec<Type<TypeName>>,
    result_types: Vec<Type<TypeName>>,
    results: Vec<ValueId>,
    fold_identity: bool,
) -> (Lambda, Option<Func<Semantic<R>>>) {
    let is_identity = fold_identity
        && captures.is_empty()
        && params.len() == parameter_types.len()
        && result_types == parameter_types
        && results.iter().enumerate().all(|(index, result)| {
            matches!(
                graph.nodes.get(*result).map(|node| &node.kind),
                Some(ValueKind::FuncParam { parameter })
                    if params.id_at_abi_position(index) == Some(*parameter)
            )
        });
    if is_identity {
        return (Lambda::identity(parameter_types), None);
    }

    let name = fresh_region_name(identities, &format!("{scope}_{label}"));
    let region = identities.alloc_function(name.clone());
    let function = finish_function(
        graph,
        return_block,
        region,
        name,
        span,
        params,
        &result_types,
        &results,
    );
    (
        Lambda::region(SegBody::new(region, captures), parameter_types, result_types),
        Some(function),
    )
}
