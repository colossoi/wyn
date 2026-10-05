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
pub(super) mod host;
mod interface;
mod kernels;
mod loops;
pub(super) mod plan;
pub(super) mod read;
use body::Body;
use read::Facts;

use crate::ssa::builder::FuncBuilder;
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
        plan: plan::Plan::new(program)?,
        placements: super::scalar::placement::Placement::new(program)?,
        entry_origins: LookupMap::default(),
        entry_names: Default::default(),
        functions: Vec::new(),
        callable_ids: LookupMap::default(),
        function_ids: IdSource::new(),
        entry_ids: IdSource::new(),
    };
    let mut declarations = Vec::new();
    for definition in &program.source.defs {
        if let DefMeta::EntryPoint(_) = &definition.meta {
            let Some(symbol) = program.identities.symbols.get(&definition.name) else {
                return Err(error("entry symbol missing"));
            };
            for stage in compiler.plan.stages(definition.name)? {
                let entry = super::abi::entry(&mut compiler, definition.name, Some(stage), &declarations)?;
                declarations.push(entry);
            }
            if compiler.facts.contains("EmitOriginalEntry", (symbol,))
                || compiler.facts.contains("FinishEntry", (symbol,))
                || compiler.facts.contains("InterfaceOnlyEntry", (symbol,))
            {
                let entry = super::abi::entry(&mut compiler, definition.name, None, &declarations)?;
                declarations.push(entry);
            }
        }
    }
    let (pipeline, physical_kernels) = super::abi::publication::publish(&mut compiler, &mut declarations)?;
    let mut entries = Vec::new();
    for metadata in declarations {
        let (owner, stage) = compiler.entry_origins[&metadata.id].clone();
        let Some(definition) = program.source.defs.iter().find(|d| d.name == owner) else {
            return Err(error("entry definition missing"));
        };
        let DefMeta::EntryPoint(entry) = &definition.meta else {
            return Err(error("entry declaration missing"));
        };
        let Some(scope) = compiler.facts.definition(owner) else {
            return Err(error("entry scope missing"));
        };
        let (source, parameters) = extract_lambda_params_ref(&definition.body);
        entries.push(interface::entry(
            &mut compiler,
            scope,
            source,
            &parameters,
            entry,
            owner,
            stage,
            &metadata,
        )?);
    }
    entries.retain(|entry| {
        let (owner, _) = &compiler.entry_origins[&entry.id];
        !compiler
            .program
            .identities
            .symbols
            .get(owner)
            .is_some_and(|symbol| compiler.facts.contains("InterfaceOnlyEntry", (symbol,)))
    });
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

pub(super) struct Compiler<'a, 'source> {
    placements: super::scalar::placement::Placement,
    pub(super) program: &'a Program<'source, Optimized>,
    pub(super) facts: Facts<'a, 'source>,
    pub(super) plan: plan::Plan<'a, 'source>,
    pub(super) entry_origins: LookupMap<EntryId, (SymbolId, Option<Value>)>,
    pub(super) entry_names: std::collections::BTreeSet<String>,
    pub(super) functions: Vec<Function>,
    pub(super) function_ids: IdSource<FunctionId>,
    callable_ids: LookupMap<(Value, Vec<Type>), FunctionId>,
    pub(super) entry_ids: IdSource<EntryId>,
}

impl Compiler<'_, '_> {
    fn function(&mut self, scope: Value, arguments: &[Typed]) -> Result<FunctionId, OptimizeError> {
        let signature: Vec<_> = arguments.iter().map(|argument| argument.ty.clone()).collect();
        if let Some(&id) = self.callable_ids.get(&(scope, signature.clone())) {
            return Ok(id);
        }
        let Some(&(_, Some(source))) = self.program.identities.scopes.get(&scope) else {
            return Err(error("callable has no source body"));
        };
        let id = self.function_ids.next_id();
        self.callable_ids.insert((scope, signature.clone()), id);
        let name = if let Some(symbol) = self.facts.definition_name(scope) {
            let Some(name) = self.program.source.symbols.get(symbol) else {
                return Err(error("callable definition name missing"));
            };
            name.clone()
        } else {
            format!("helper_{}", self.functions.len())
        };
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
        let Some(result) = self.facts.result(scope) else {
            return Err(error("callable has no result fact"));
        };
        let mut lower = Body::new(self, scope, signature, source.ty.clone())?;
        let result = lower
            .value(scope, result)
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

pub(super) fn error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(format!("SSA lowering: {}", message.into()))
}
fn builder_error(error: BuilderError) -> OptimizeError {
    OptimizeError::Output(error.to_string())
}

#[cfg(test)]
#[path = "to_ssa_tests.rs"]
mod tests;
