//! Emit selected egglog expressions and scheduled execution directly into SSA.
use super::{timing, OptimizeError, Optimized, Program};
use crate::ssa::builder::BuilderError;
use crate::ssa::context::BackendGlobal;
use crate::ssa::stage::Elaborated;
use crate::ssa::types::{Function, ValueRef};
use crate::tlc::{extract_lambda_params_ref, DefMeta};
use crate::types::Type;
use crate::{CodegenTarget, FunctionId, LookupMap};
use egglog_engine::Value;
use wyn_base::IdSource;

mod body;
mod host;
mod interface;
mod kernels;
mod plan;
mod publication;
mod read;
mod sizes;
use body::Body;
use read::Facts;

use crate::host::DispatchLen;
use crate::host::ScalarTask;
use crate::interface::EntryInput;
use crate::interface::EntryKind;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::ConstantValue;
use crate::ssa::types::Terminator;
use crate::tlc::TermKind;
use crate::types::TypeName;
use crate::EntryId;
use crate::SymbolId;
pub(super) fn prepare(program: &mut Program<'_, Optimized>) -> Result<(), OptimizeError> {
    program
        .graph
        .parse_and_run_program(Some("ssa-access.egg".into()), include_str!("to_ssa/access.egg"))?;
    Ok(())
}

pub(super) fn lower(
    program: &Program<'_, Optimized>,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let _timing = timing::span("egglog to SSA");
    let facts = Facts { program };
    let mut compiler = Compiler {
        program,
        facts,
        plan: plan::Plan::read(program)?,
        entry_origins: LookupMap::default(),
        host_lengths: LookupMap::default(),
        input_interfaces: LookupMap::default(),
        host_tasks: Vec::new(),
        entry_names: Default::default(),
        functions: Vec::new(),
        function_ids: IdSource::new(),
        entry_ids: IdSource::new(),
        specializations: LookupMap::default(),
    };
    let mut lengths = Ok(());
    program.graph.constructor_enodes("AbiStorage", |row| {
        if lengths.is_err() {
            return;
        }
        lengths = (|| {
            let Some(binding) = compiler.facts.enode("InputBinding", row.children[1]) else {
                return Err(error("ABI storage has no input binding"));
            };
            let Some(expression) = compiler.facts.enode("AbiExpr", row.children[0]) else {
                return Err(error("ABI storage has no source expression"));
            };
            let source = *program.identities.values.resolve(compiler.facts.integer(expression[0]));
            compiler.host_lengths.insert(
                source,
                DispatchLen::InputBinding {
                    set: compiler.facts.integer(binding[0]) as u32,
                    binding: compiler.facts.integer(binding[1]) as u32,
                    elem_bytes: compiler.facts.integer(row.children[2]) as u32,
                },
            );
            Ok(())
        })();
    })?;
    lengths?;
    let mut entries = Vec::new();
    for definition in &program.source.defs {
        if let DefMeta::EntryPoint(entry) = &definition.meta {
            let Some(scope) = compiler.facts.definition(definition.name) else {
                return Err(error("entry scope is missing"));
            };
            let (source, parameters) = extract_lambda_params_ref(&definition.body);
            for stage in compiler
                .plan
                .stages
                .iter()
                .filter(|stage| stage.owner == definition.name)
                .cloned()
                .collect::<Vec<_>>()
            {
                entries.push(interface::entry(
                    &mut compiler,
                    scope,
                    source,
                    &parameters,
                    entry,
                    definition.name,
                    Some(&stage),
                )?);
            }
            let needs_finish = entry.declaration.entry_kind != EntryKind::Compute
                || !compiler.plan.stages.iter().any(|stage| stage.owner == definition.name)
                || compiler.plan.outputs.iter().any(|output| {
                    output.owner == definition.name && output.copy && output.writer.is_none()
                });
            if needs_finish {
                entries.push(interface::entry(
                    &mut compiler,
                    scope,
                    source,
                    &parameters,
                    entry,
                    definition.name,
                    None,
                )?);
            }
        }
    }
    let (pipeline, physical_kernels) = publication::publish(&compiler, &mut entries)?;
    Ok(Elaborated::from_parts(
        compiler.functions,
        entries,
        Vec::new(),
        BackendGlobal {
            pipeline,
            target,
            physical_kernels,
        },
    ))
}

struct Compiler<'a, 'source> {
    program: &'a Program<'source, Optimized>,
    facts: Facts<'a, 'source>,
    plan: plan::Plan<'a, 'source>,
    entry_origins: LookupMap<EntryId, (SymbolId, Option<plan::Stage>)>,
    host_lengths: LookupMap<Value, DispatchLen>,
    input_interfaces: LookupMap<Value, EntryInput>,
    host_tasks: Vec<ScalarTask>,
    entry_names: std::collections::BTreeSet<String>,
    functions: Vec<Function>,
    function_ids: IdSource<FunctionId>,
    entry_ids: IdSource<EntryId>,
    specializations: LookupMap<(Value, Vec<(Type, Option<ConstantValue>)>), FunctionId>,
}

impl Compiler<'_, '_> {
    fn function(&mut self, scope: Value, arguments: &[Typed]) -> Result<FunctionId, OptimizeError> {
        let signature: Vec<_> = arguments.iter().map(|argument| argument.ty.clone()).collect();
        let key = (
            scope,
            arguments.iter().map(|argument| (argument.ty.clone(), argument.value.as_const())).collect(),
        );
        if let Some(&id) = self.specializations.get(&key) {
            return Ok(id);
        }
        let Some(&(_, Some(source))) = self.program.identities.scopes.get(&scope) else {
            return Err(error("callable has no source body"));
        };
        let id = self.function_ids.next_id();
        self.specializations.insert(key, id);
        let name = self
            .facts
            .definition_name(scope)
            .and_then(|symbol| self.program.source.symbols.get(symbol))
            .cloned()
            .unwrap_or_else(|| format!("helper_{}", self.functions.len()));
        if let TermKind::Extern(linkage) = &source.kind {
            let mut result = &source.ty;
            for _ in &signature {
                let Type::Constructed(TypeName::Arrow, parts) = result else {
                    return Err(error("external signature is not callable"));
                };
                let Some(next) = parts.last() else {
                    return Err(error("external return type missing"));
                };
                result = next;
            }
            let mut builder = FuncBuilder::new(
                signature.into_iter().enumerate().map(|(i, ty)| (ty, format!("arg{i}"))).collect(),
                result.clone(),
            );
            builder.terminate(Terminator::Unreachable).map_err(builder_error)?;
            self.functions.push(Function {
                id,
                name,
                body: builder.finish().map_err(builder_error)?,
                span: source.span,
                linkage_name: Some(linkage.clone()),
            });
            return Ok(id);
        }
        let mut lower = Body::new(self, scope, signature, source.ty.clone())?;
        for (i, argument) in arguments.iter().enumerate() {
            if argument.value.as_const().is_some() {
                let Some(formal) = lower.compiler.facts.parameter(scope, i as i64) else {
                    return Err(error("specialized parameter missing"));
                };
                lower.values.insert(formal, argument.clone());
            }
        }
        let result = lower
            .source(scope, source)
            .map_err(|err| error(format!("function {name}, {} arguments: {err}", arguments.len())))?;
        let body = lower.finish(result)?;
        self.functions.push(Function {
            id,
            name,
            body,
            span: source.span,
            linkage_name: None,
        });
        Ok(id)
    }
}

#[derive(Clone)]
struct Typed {
    value: ValueRef,
    ty: Type,
}

fn error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(format!("SSA lowering: {}", message.into()))
}
fn builder_error(error: BuilderError) -> OptimizeError {
    OptimizeError::Output(error.to_string())
}

#[cfg(test)]
#[path = "to_ssa_tests.rs"]
mod tests;
