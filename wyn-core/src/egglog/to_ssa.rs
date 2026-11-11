//! Read the final memory program once to emit SSA and its public interfaces.
use super::query::Query;
use super::{execution::Execution, OptimizeError, Optimized, Program};
use crate::ast::Span;
use crate::ssa::builder::BuilderError;
use crate::ssa::stage::Elaborated;
use crate::ssa::types::{Function, ValueRef};
use crate::types::{self, Type, TypeName};
use crate::{CodegenTarget, FunctionId, LookupMap};
use egglog_engine::{sort::S, Value};
use wyn_base::IdSource;

mod body;
mod entry;
mod host;
mod kernels;
use body::Body;

pub(super) fn lower(
    program: &Program<'_, Optimized>,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let query = Query(&program.graph);
    let Some(root) = query.row("CurrentProgram", |_| true)? else {
        return Err(error("missing memory program"));
    };
    let fields = query.fields("MemoryProgram", root[0])?;
    let mut definitions = LookupMap::default();
    for function in query.vector(fields[0])? {
        let fields = query.fields("DefineFunction", function)?;
        definitions.insert(fields[0], function);
    }
    let mut externals = LookupMap::default();
    for external in query.vector(fields[1])? {
        let fields = query.fields("DeclareExternal", external)?;
        externals.insert(fields[0], external);
    }
    let mut compiler = Compiler {
        program,
        definitions,
        externals,
        execution: Execution::select(&program.graph)?,
        block_ids: LookupMap::default(),
        functions: Vec::new(),
        callable_ids: LookupMap::default(),
        function_ids: IdSource::new(),
    };
    entry::lower(&mut compiler, query.vector(fields[2])?, target)
}

pub(super) struct Compiler<'a, 'source> {
    pub program: &'a Program<'source, Optimized>,
    pub definitions: LookupMap<Value, Value>,
    externals: LookupMap<Value, Value>,
    pub functions: Vec<Function>,
    pub execution: Execution,
    block_ids: LookupMap<Value, i32>,
    callable_ids: LookupMap<Value, (FunctionId, Type)>,
    function_ids: IdSource<FunctionId>,
}

impl Compiler<'_, '_> {
    fn function(&mut self, name: Value) -> Result<(FunctionId, Type), OptimizeError> {
        if let Some(selected) = self.callable_ids.get(&name) {
            return Ok(selected.clone());
        }
        let query = Query(&self.program.graph);
        let (definition, fields, external) = if let Some(&definition) = self.definitions.get(&name) {
            (definition, query.fields("DefineFunction", definition)?, false)
        } else if let Some(&external) = self.externals.get(&name) {
            (external, query.fields("DeclareExternal", external)?, true)
        } else {
            return Err(error("unknown memory function"));
        };
        let name_fields = query.fields("FunctionId", name)?;
        let text = self.program.graph.value_to_base::<S>(name_fields[1]).to_string();
        let params = query.vector(fields[1])?.into_iter().try_fold(Vec::new(), |mut out, param| {
            let fields = query.fields("Parameter", param)?;
            if query.enode("BindBlock", fields[0])?.is_some()
                || query.enode("BindAccumulator", fields[0])?.is_some()
            {
                return Err(error(
                    "memory-bearing shader helper must be inlined before emission",
                ));
            }
            if let Some(binding) = runtime_binding(query, fields[0])? {
                out.push(binding);
            }
            Ok::<_, OptimizeError>(out)
        })?;
        let result_types = query
            .vector(fields[2])?
            .into_iter()
            .filter_map(|info| {
                match query.application(
                    info,
                    &[
                        "ReturnScalar",
                        "ReturnMemory",
                        "ReturnArray",
                        "ReturnAccumulator",
                        "ReturnResource",
                    ],
                ) {
                    Ok(("ReturnScalar", fields)) => Some(element_type(query, fields[0])),
                    Ok(("ReturnMemory", _)) => Some(Err(error(
                        "memory-returning shader helper must be inlined before emission",
                    ))),
                    Ok(("ReturnArray" | "ReturnAccumulator", _)) => None,
                    Ok(("ReturnResource", fields)) => Some(resource_type(query, fields[0])),
                    Ok(_) => unreachable!(),
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result_type =
            if let [ty] = result_types.as_slice() { ty.clone() } else { types::tuple(result_types) };
        let id = self.function_ids.next_id();
        self.callable_ids.insert(name, (id, result_type.clone()));
        let mut body = Body::new(self, definition, params, result_type.clone());
        let body = if external {
            body.builder.terminate(crate::ssa::types::Terminator::Unreachable).map_err(builder_error)?;
            body.builder.finish().map_err(builder_error)?
        } else {
            let values = body.block(fields[3])?;
            let result = if values.len() == 1 {
                values.into_iter().next().ok_or_else(|| error("call result missing"))?
            } else {
                body.tuple(values)?
            };
            body.finish(result)?
        };
        self.functions.push(Function {
            id,
            name: text.clone(),
            body,
            span: Span::generated(),
            linkage_name: external.then_some(text),
        });
        Ok((id, result_type))
    }
}

/// Arrays are compile-time descriptors. Only their explicit witnesses and block
/// operands cross SSA control-flow edges; accumulator tokens are compile-time capabilities.
fn runtime_binding(query: Query<'_>, binding: Value) -> Result<Option<(Value, Type)>, OptimizeError> {
    let (kind, fields) = query.application(binding, &["BindValue", "BindBlock", "BindAccumulator"])?;
    match kind {
        "BindBlock" => Ok(Some((fields[0], types::i32()))),
        "BindAccumulator" => Ok(None),
        "BindValue" => {
            let (kind, info) =
                query.application(fields[1], &["ScalarInfo", "ArrayInfo", "ResourceInfo"])?;
            match kind {
                "ScalarInfo" => Ok(Some((fields[0], element_type(query, info[0])?))),
                "ArrayInfo" => Ok(None),
                "ResourceInfo" => Ok(Some((fields[0], resource_type(query, info[0])?))),
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }
}

fn resource_type(query: Query<'_>, resource: Value) -> Result<Type, OptimizeError> {
    let (kind, fields) = query.application(resource, &["TextureType", "SamplerType", "ImageType"])?;
    Ok(match kind {
        "TextureType" => Type::Constructed(TypeName::Texture2D, vec![]),
        "SamplerType" => Type::Constructed(TypeName::Sampler, vec![]),
        "ImageType" => {
            let set = u32::try_from(query.0.value_to_base::<i64>(fields[0]))
                .map_err(|_| error("invalid image descriptor set"))?;
            let slot = u32::try_from(query.0.value_to_base::<i64>(fields[1]))
                .map_err(|_| error("invalid image descriptor slot"))?;
            Type::Constructed(
                TypeName::StorageTexture,
                vec![types::buffer_tag(crate::BindingRef::new(set, slot))],
            )
        }
        _ => unreachable!(),
    })
}

fn element_type(query: Query<'_>, element: Value) -> Result<Type, OptimizeError> {
    if let Some(fields) = query.enode("MatrixElement", element)? {
        let rows = usize::try_from(query.0.value_to_base::<i64>(fields[1]))
            .map_err(|_| error("invalid matrix rows"))?;
        let columns = usize::try_from(query.0.value_to_base::<i64>(fields[2]))
            .map_err(|_| error("invalid matrix columns"))?;
        return Ok(types::mat(rows, columns, primitive_type(query, fields[0])?));
    }
    if let Some(fields) = query.enode("ProductElement", element)? {
        return Ok(types::tuple(
            query
                .vector(fields[0])?
                .into_iter()
                .map(|field| element_type(query, field))
                .collect::<Result<_, _>>()?,
        ));
    }
    let (name, fields) = query.application(element, &["ScalarElement", "VectorElement"])?;
    let primitive = primitive_type(query, fields[0])?;
    if name == "ScalarElement" {
        return Ok(primitive);
    }
    let width = usize::try_from(query.0.value_to_base::<i64>(fields[1]))
        .map_err(|_| error("invalid vector width"))?;
    Ok(Type::Constructed(
        TypeName::Vec,
        vec![primitive, Type::Constructed(TypeName::Size(width), vec![])],
    ))
}

fn primitive_type(query: Query<'_>, primitive: Value) -> Result<Type, OptimizeError> {
    for (name, ty) in [
        ("I8", TypeName::Int(8)),
        ("I16", TypeName::Int(16)),
        ("I32", TypeName::Int(32)),
        ("I64", TypeName::Int(64)),
        ("U8", TypeName::UInt(8)),
        ("U16", TypeName::UInt(16)),
        ("U32", TypeName::UInt(32)),
        ("U64", TypeName::UInt(64)),
        ("F16", TypeName::Float(16)),
        ("F32", TypeName::Float(32)),
        ("F64", TypeName::Float(64)),
        ("Bool", TypeName::Bool),
        ("UnitType", TypeName::Unit),
    ] {
        if query.enode(name, primitive)?.is_some() {
            return Ok(Type::Constructed(ty, vec![]));
        }
    }
    Err(error("unknown memory primitive type"))
}

#[derive(Clone)]
struct Typed {
    value: ValueRef,
    ty: Type,
}

pub(super) fn error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(format!("SSA lowering: {}", message.into()))
}
fn builder_error(error: BuilderError) -> OptimizeError {
    OptimizeError::Output(error.to_string())
}

#[cfg(test)]
#[path = "to_ssa_tests.rs"]
mod tests;
