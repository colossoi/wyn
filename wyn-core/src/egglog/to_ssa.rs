//! Lower selected egglog expressions and scheduled execution into SSA and host interfaces.
use super::scalar::evaluation::Evaluation;
use super::{timing, OptimizeError, Optimized, Program};
use crate::ast::Span;
use crate::interface::publish::EntryPoint;
use crate::ssa::builder::BuilderError;
use crate::ssa::types::{Function, ValueRef};
use crate::tlc::DefMeta;
use crate::types::Type;
use crate::{CodegenTarget, FunctionId, LookupMap};
use egglog_engine::{sort::S, Value};
use wyn_base::IdSource;

mod body;
mod entry;
mod kernels;
use super::facts::Facts;
use body::Body;

use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::Terminator;
use crate::types::TypeName;
use crate::EntryId;
use crate::SymbolId;
pub(super) fn lower(
    program: &Program<'_, Optimized>,
    target: CodegenTarget,
) -> Result<crate::ssa::stage::Elaborated, OptimizeError> {
    let evaluations = Evaluation::new(program)?;
    let _timing = timing::span("egglog to SSA");
    let facts = Facts { program };
    let mut compiler = Compiler {
        program,
        facts,
        bindings: super::abi::bindings::Bindings::new(program)?,
        evaluations,
        entry_origins: LookupMap::default(),
        entry_names: Default::default(),
        functions: Vec::new(),
        callable_ids: LookupMap::default(),
        function_ids: IdSource::new(),
        entry_ids: IdSource::new(),
    };
    let mut entries = Vec::new();
    for definition in &program.source.defs {
        if let DefMeta::EntryPoint(_) = &definition.meta {
            let first_entry = entries.len();
            let Some(symbol) = program.identities.symbols.get(&definition.name) else {
                return Err(error("entry symbol missing"));
            };
            for stage in program.stage.execution.entries.get(&symbol).into_iter().flatten().copied() {
                let entry = entry::entry(&mut compiler, definition.name, Some(stage), &entries)?;
                entries.push(entry);
            }
            if compiler.facts.contains("EmitOriginalEntry", (symbol,))
                || compiler.facts.contains("FinishEntry", (symbol,))
                || compiler.facts.contains("InterfaceOnlyEntry", (symbol,))
            {
                let entry = entry::entry(&mut compiler, definition.name, None, &entries)?;
                entries.push(entry);
            }
            // Each stage's accesses were lowered with its interface. Form this
            // source pipeline's layout union once from those final entries.
            let mut accesses = LookupMap::<crate::BindingRef, crate::ResourceAccess>::default();
            for entry in &entries[first_entry..] {
                for (&binding, &access) in &entry.stage_descriptor_storage_accesses {
                    accesses.entry(binding).and_modify(|old| *old = old.merge(access)).or_insert(access);
                }
            }
            for entry in &mut entries[first_entry..] {
                entry.pipeline_storage_accesses.clone_from(&accesses);
            }
        }
    }
    let pipeline = super::abi::publication::publish(&mut compiler, &entries)?;
    entries.retain(|entry| {
        let (owner, _) = &compiler.entry_origins[&entry.id];
        !program
            .identities
            .symbols
            .get(owner)
            .is_some_and(|symbol| compiler.facts.contains("InterfaceOnlyEntry", (symbol,)))
    });
    Ok(crate::ssa::stage::Elaborated::from_parts(
        compiler.functions,
        entries.into_iter().map(EntryPoint::into_shader).collect(),
        crate::ssa::context::BackendGlobal { pipeline, target },
    ))
}

pub(super) struct Compiler<'a, 'source> {
    evaluations: Evaluation,
    pub(super) program: &'a Program<'source, Optimized>,
    pub(super) facts: Facts<'a, 'source>,
    pub(super) bindings: super::abi::bindings::Bindings<'a, 'source>,
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
        let span = self.program.identities.spans.get(&scope).copied().unwrap_or_else(Span::generated);
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
        if let Some(fields) = self.facts.record("SourceExternal", "ExternalInfo", (scope,)) {
            let Some(mut result) = self.facts.ty(fields[0]) else {
                return Err(error("external signature type missing"));
            };
            let linkage = self.program.graph.value_to_base::<S>(fields[1]).to_string();
            for _ in &signature {
                let Type::Constructed(TypeName::Arrow, parts) = result else {
                    return Err(error("external signature is not callable"));
                };
                let Some(next) = parts.last() else {
                    return Err(error("external return type missing"));
                };
                result = next;
            }
            let mut builder = FuncBuilder::new(signature, result.clone());
            builder.terminate(Terminator::Unreachable).map_err(builder_error)?;
            self.functions.push(Function {
                id,
                name,
                body: builder.finish().map_err(builder_error)?,
                span,
                linkage_name: Some(linkage),
            });
            return Ok(id);
        }
        let Some(result) = self.facts.result(scope) else {
            return Err(error("callable has no result fact"));
        };
        let Some(ty) = self.facts.source_type(result).cloned() else {
            return Err(error("callable result type missing"));
        };
        let parameters = self.facts.bind_arguments(scope, signature)?;
        let mut lower = Body::new(self, scope, parameters, ty)?;
        let result = lower
            .value(scope, result)
            .map_err(|err| error(format!("function {name}, {} arguments: {err}", arguments.len())))?;
        let body = lower.finish(result)?;
        self.functions.push(Function {
            id,
            name,
            body,
            span,
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
