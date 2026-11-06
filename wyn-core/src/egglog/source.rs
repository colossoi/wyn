//! Import execution scopes and direct source facts; egglog resolves forwarding.
//! Demanded arithmetic is admitted later into the same graph.
use super::{fusion, planning, OptimizeError};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_access, extract_storage_binding,
    extract_storage_image_binding, extract_texture_binding, extract_uniform_binding,
};
use crate::interface::{EntryKind, IoDecoration, StorageAccess};
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::stage::OwnershipApplied;
use crate::tlc::{
    self, extract_lambda_params_ref, ArrayExpr, DefMeta, Lambda, LoopKind, SoacBody, SoacOp, TermId,
    TermKind, VarRef,
};
use crate::types::{as_soa_tuple, Type, TypeExt, TypeName};
use crate::{LookupMap, LookupSet, SymbolId};
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, EGraph, FullState, RawValues, Value, Write};
use wyn_base::{IdSource, Interner};

mod facts;
mod interface;
mod loops;
mod structure;
#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
use super::bindings::Bindings;

pub(super) type Term = tlc::Term<ExplicitClosurePayload, ExplicitCapturesPayload>;
type OperatorBody = SoacBody<ExplicitClosurePayload, ExplicitCapturesPayload>;

pub(super) fn import(source: &OwnershipApplied) -> Result<(EGraph, Identities<'_>), OptimizeError> {
    let mut graph = fusion::new_graph()?;
    planning::load(&mut graph)?;
    graph.parse_and_run_program(Some("source.egg".into()), include_str!("source.egg"))?;
    graph.parse_and_run_program(Some("analysis.egg".into()), include_str!("analysis.egg"))?;
    graph.parse_and_run_program(
        Some("publication-schema.egg".into()),
        include_str!("publication-schema.egg"),
    )?;
    let mut identities = Identities::default();
    graph.update(|sink| {
        let mut import = Import {
            sink,
            identities: &mut identities,
            definitions: source.defs.iter().map(|d| (d.name, &d.body)).collect(),
            globals: source.defs.iter().map(|definition| definition.name).collect(),
            bindings: Bindings::default(),
            regions: IdSource::new(),
            operations: IdSource::new(),
            outputs: IdSource::new(),
            arrays: IdSource::new(),
            imported_types: LookupMap::default(),
            source_ordinals: LookupMap::default(),
            work: LookupMap::default(),
            operations_by_value: LookupMap::default(),
            operation_regions: LookupSet::default(),
            lengths: LookupMap::default(),
            owners: LookupMap::default(),
            ownership: &source.global_context.ownership,
        };
        // Native writes report engine errors; source validation also has its
        // own error type. On either failure the whole new graph is discarded.
        Ok(import.definitions(source))
    })??;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate flow-forward source-forwarding) flow-projections source-projections)",
    )?;
    Ok((graph, identities))
}

/// Tokens assigned by this graph map back to opaque TLC identities.
#[derive(Default)]
pub(super) struct Identities<'source> {
    terms: Interner<i64, TermId>,
    pub(super) symbols: Interner<i64, SymbolId>,
    pub(super) types: Interner<i64, Type>,
    pub(super) scopes: LookupMap<Value, (Option<Value>, Option<&'source Term>)>,
    pub(super) origins: LookupMap<Value, &'source Term>,
}

impl<'source> Identities<'source> {
    fn term(&mut self, term: &'source Term) -> Result<i64, OptimizeError> {
        if term.id == TermId::SYNTHETIC {
            return Err(OptimizeError::Output(
                "TLC source term has no stable identity".into(),
            ));
        }
        Ok(self.terms.intern(&term.id))
    }
}

/// Walk state only. Executable structure is recorded in egglog, not mirrored here.
struct Scope {
    key: Value,
    position: i64,
    previous_operation: Option<Value>,
    work: i64,
    evaluation: Option<Value>,
}

struct Import<'graph, 'db, 'ids, 'source> {
    sink: FullState<'graph, 'db>,
    identities: &'ids mut Identities<'source>,
    globals: LookupSet<SymbolId>,
    definitions: LookupMap<SymbolId, &'source Term>,
    bindings: Bindings<SymbolId, Value>,
    regions: IdSource<i64>,
    operations: IdSource<i64>,
    outputs: IdSource<i64>,
    arrays: IdSource<i64>,
    imported_types: LookupMap<i64, Value>,
    source_ordinals: LookupMap<Value, i64>,
    work: LookupMap<Value, i64>,
    operations_by_value: LookupMap<Value, Value>,
    operation_regions: LookupSet<Value>,
    lengths: LookupMap<Value, Value>,
    owners: LookupMap<Value, Value>,
    ownership: &'source tlc::ownership::OwnershipFacts,
}

impl<'source> Import<'_, '_, '_, 'source> {
    fn definitions(&mut self, source: &'source OwnershipApplied) -> Result<(), OptimizeError> {
        let counter = self.ty(&Type::Constructed(TypeName::UInt(32), vec![]))?;
        self.sink.add("CounterType", counter)?;
        let length = self.ty(&Type::Constructed(TypeName::Int(32), vec![]))?;
        self.sink.add("LengthType", length)?;
        for (ordinal, definition) in source.defs.iter().enumerate() {
            let (body, parameters) = extract_lambda_params_ref(&definition.body);
            let mut scope = self.scope(Some(body), None)?;
            let symbol = self.identities.symbols.intern(&definition.name);
            self.sink.set("SourceDefinition", symbol, scope.key)?;
            let global = self.sink.add("SourceGlobal", symbol)?;
            self.value_type(global, &definition.ty)?;
            self.flags(global, false, true, true, true, 0)?;
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                self.sink.add("SourceEntryOrder", (symbol, ordinal as i64))?;
                let compute = entry.declaration.entry_kind == EntryKind::Compute;
                self.sink.add("SourceOriginalEntry", (symbol, scope.key, compute))?;
                let grid = match entry.declaration.compute_dispatch {
                    Some(grid) => self.sink.add(
                        "FixedGrid",
                        (i64::from(grid.x), i64::from(grid.y), i64::from(grid.z)),
                    )?,
                    None => self.sink.add("AutomaticGrid", RawValues(vec![]))?,
                };
                self.sink.set("EntryGrid", symbol, grid)?;
            }
            let checkpoint = self.bindings.checkpoint();
            self.parameters(&parameters, &scope)?;
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                let family = if let Some(group) = &entry.declaration.graphics_group {
                    let Some(name) = source.symbols.get(group.root) else {
                        return Err(OptimizeError::Output("graphics source name missing".into()));
                    };
                    name
                } else {
                    entry
                        .declaration
                        .source_entry
                        .as_ref()
                        .map_or(&entry.declaration.name, |entry| &entry.name)
                };
                self.sink.add("SourceInterfaceFamily", (symbol, family.as_str()))?;
                self.interface(entry, symbol, &parameters)?;
                for binding in entry.data.param_bindings.iter().flatten() {
                    self.entry_binding(binding)?;
                }
                for (index, param) in entry.declaration.params.iter().enumerate() {
                    let bound = entry.data.param_bindings.get(index).is_some_and(Option::is_some)
                        || extract_storage_binding(param).is_some()
                        || extract_uniform_binding(param).is_some()
                        || extract_texture_binding(param).is_some()
                        || extract_sampler_binding(param).is_some()
                        || extract_storage_image_binding(param).is_some();
                    let builtin = matches!(extract_io_decoration(param), Some(IoDecoration::BuiltIn(_)));
                    self.sink.add("SourceEntryParameter", (scope.key, index as i64, bound, builtin))?;
                    if let Some(access) = extract_storage_access(param) {
                        let access: i64 = match access {
                            StorageAccess::ReadOnly => 1,
                            StorageAccess::WriteOnly => 2,
                            StorageAccess::ReadWrite => 3,
                        };
                        self.sink.set("SourceParameterStorageAccess", (scope.key, index as i64), access)?;
                    } else {
                        self.sink.set("SourceParameterStorageAccess", (scope.key, index as i64), 1i64)?;
                    }
                    if matches!(
                        extract_storage_access(param),
                        Some(StorageAccess::ReadWrite | StorageAccess::WriteOnly)
                    ) {
                        let binding = entry
                            .data
                            .param_bindings
                            .get(index)
                            .and_then(Option::as_ref)
                            .map(|binding| binding.first_buffer().0)
                            .or_else(|| extract_storage_binding(param));
                        if let Some(binding) = binding {
                            self.sink.add(
                                "SourceMutableBinding",
                                (i64::from(binding.set), i64::from(binding.binding)),
                            )?;
                        }
                    }
                }
            }
            let result = self.finish_body(body, &mut scope)?;
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                if entry.declaration.buffer_demand {
                    self.sink.add("SourceBufferDemand", symbol)?;
                    self.output_leaf(symbol, 0, result, &body.ty, None)?;
                } else {
                    self.output(symbol, result, body, &entry.declaration.outputs)?;
                }
            }
            self.bindings.restore(checkpoint);
        }
        Ok(())
    }

    fn scope(
        &mut self,
        body: Option<&'source Term>,
        parent: Option<Value>,
    ) -> Result<Scope, OptimizeError> {
        let key = self.sink.add("RegionId", self.regions.next_id())?;
        self.identities.scopes.insert(key, (parent, body));
        if let Some(parent) = parent {
            self.sink.add("SourceParent", (key, parent))?;
        } else {
            self.sink.add("SourceRootRegion", key)?;
        }
        let execution = self.sink.add("FlowRegion", key)?;
        self.flags(execution, false, true, true, true, 0)?;
        Ok(Scope {
            key,
            position: 0,
            previous_operation: None,
            work: 0,
            evaluation: None,
        })
    }

    fn source_value(&mut self, term: &'source Term, scope: Value) -> Result<Value, OptimizeError> {
        let id = self.identities.term(term)?;
        let value = self.sink.add("SourceTerm", (scope, id))?;
        self.identities.origins.insert(value, term);
        self.value_type(value, &term.ty)?;
        Ok(value)
    }

    fn bind(&mut self, symbol: SymbolId, value: Value, scope: &Scope) -> Result<(), OptimizeError> {
        self.bindings.insert(symbol, value);
        self.owners.entry(value).or_insert(scope.key);
        if self.ownership.reusable.contains(&symbol) {
            let region = self.owners[&value];
            self.sink.add("SourceReusePermission", (value, region))?;
        }
        Ok(())
    }

    fn formal(&mut self, symbol: SymbolId, ty: &Type, scope: &Scope) -> Result<Value, OptimizeError> {
        let name = self.identities.symbols.intern(&symbol);
        let value = self.sink.add("SourceFormal", (scope.key, name))?;
        self.value_type(value, ty)?;
        self.flags(value, false, true, true, true, 0)?;
        self.bind(symbol, value, scope)?;
        Ok(value)
    }

    fn parameters(&mut self, parameters: &[(SymbolId, Type)], scope: &Scope) -> Result<(), OptimizeError> {
        for (index, (symbol, ty)) in parameters.iter().enumerate() {
            let value = self.formal(*symbol, ty, scope)?;
            self.sink.set("SourceParameter", (scope.key, index as i64), value)?;
        }
        Ok(())
    }

    fn resolve(&mut self, symbol: SymbolId) -> Result<Value, OptimizeError> {
        if let Some(value) = self.bindings.get(&symbol).copied() {
            return Ok(value);
        }
        if self.globals.contains(&symbol) {
            let symbol = self.identities.symbols.intern(&symbol);
            return Ok(self.sink.add("SourceGlobal", symbol)?);
        }
        Err(OptimizeError::Output(format!(
            "unbound TLC symbol {symbol:?} during source import"
        )))
    }

    fn finish_body(&mut self, body: &'source Term, scope: &mut Scope) -> Result<Value, OptimizeError> {
        let (result, _) = self.visit(body, scope)?;
        self.sink.set("SourceResult", scope.key, result)?;
        self.sink.set("SourceRegionWork", scope.key, scope.work)?;
        Ok(result)
    }

    fn nested(
        &mut self,
        body: &'source Term,
        parent: &mut Scope,
        owner: Value,
    ) -> Result<Value, OptimizeError> {
        let mut scope = self.scope(Some(body), Some(parent.key))?;
        self.sink.add("SourceEnters", (owner, scope.key))?;
        let checkpoint = self.bindings.checkpoint();
        self.finish_body(body, &mut scope)?;
        self.bindings.restore(checkpoint);
        Ok(scope.key)
    }

    fn operand(
        &mut self,
        term: &'source Term,
        owner: Value,
        scope: &mut Scope,
    ) -> Result<Value, OptimizeError> {
        let parent = scope.evaluation.replace(owner);
        let result = self.visit(term, scope);
        scope.evaluation = parent;
        let (value, work) = result?;
        self.sink.add("SourceDirectUse", (owner, value))?;
        let total = self.work.entry(owner).or_default();
        *total = total.saturating_add(work);
        Ok(value)
    }

    fn operation(&mut self, value: Value, kind: &str, scope: &Scope) -> Result<Value, OptimizeError> {
        let operation = self.sink.add("OperationId", self.operations.next_id())?;
        let info = self.sink.add("OperationInfo", (scope.key, kind))?;
        self.sink.set("SourceOperation", operation, info)?;
        self.sink.set("SourceOperationValue", operation, value)?;
        self.operations_by_value.insert(value, operation);
        self.operation_regions.insert(scope.key);
        if matches!(kind, "control" | "call" | "index" | "global") {
            self.sink.set("SourceInputCount", operation, 0i64)?;
        }
        Ok(operation)
    }

    fn visit(&mut self, term: &'source Term, scope: &mut Scope) -> Result<(Value, i64), OptimizeError> {
        // Lexical names and let syntax do not introduce semantic values.
        match &term.kind {
            TermKind::Var(VarRef::Symbol(symbol)) => {
                let target = self.resolve(*symbol)?;
                self.sink.add("SourceRegionUse", (scope.key, target))?;
                self.free_reference(target, scope.key)?;
                if let Some(owner) = scope.evaluation {
                    self.sink.add("FlowReference", (owner, target))?;
                }
                return Ok((target, 0));
            }
            TermKind::Let { name, rhs, body, .. } => {
                let (rhs, work) = self.visit(rhs, scope)?;
                let checkpoint = self.bindings.checkpoint();
                self.bind(*name, rhs, scope)?;
                let result = self.visit(body, scope);
                self.bindings.restore(checkpoint);
                let (value, body) = result?;
                return Ok((value, work.saturating_add(body)));
            }
            _ => {}
        }
        let value = self.expression(term, scope)?;
        Ok((value, self.work[&value]))
    }

    fn expression(&mut self, term: &'source Term, scope: &mut Scope) -> Result<Value, OptimizeError> {
        let value = self.source_value(term, scope.key)?;
        self.properties(term, value)?;
        scope.work = scope.work.saturating_add(self.work[&value]);
        let owner = match scope.evaluation {
            Some(owner) => owner,
            None => self.sink.add("FlowRegion", scope.key)?,
        };
        self.sink.add("FlowEvaluation", (owner, value))?;
        match &term.kind {
            TermKind::Var(VarRef::Symbol(_)) | TermKind::Let { .. } => {
                return Err(OptimizeError::Output(
                    "lexical binding passed to expression importer".into(),
                ));
            }
            TermKind::Var(VarRef::Builtin { .. })
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {}
            TermKind::IntLit(text) => {
                if matches!(
                    &term.ty,
                    Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _)
                ) {
                    let integer = text
                        .parse::<i64>()
                        .map_err(|_| OptimizeError::Output("invalid 32-bit integer literal".into()))?;
                    self.sink.set("SourceInteger32", value, integer)?;
                }
            }
            TermKind::Lambda(lambda) => {
                let mut inner = self.scope(Some(&lambda.body), Some(scope.key))?;
                self.sink.set("SourceCallable", value, inner.key)?;
                let checkpoint = self.bindings.checkpoint();
                self.parameters(&lambda.params, &inner)?;
                self.finish_body(&lambda.body, &mut inner)?;
                self.bindings.restore(checkpoint);
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let condition = self.operand(cond, value, scope)?;
                let yes = self.nested(then_branch, scope, value)?;
                let no = self.nested(else_branch, scope, value)?;
                let info = self.sink.add("BranchInfo", (condition, yes, no))?;
                self.sink.set("SourceBranch", value, info)?;
                if self.operation_regions.contains(&yes)
                    || self.operation_regions.contains(&no)
                    || (term.ty.is_array()
                        && !matches!(
                            term.ty.array_size(),
                            Some(Type::Constructed(TypeName::Size(_), _))
                        ))
                {
                    self.operation(value, "control", scope)?;
                }
            }
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                self.loop_body(
                    value,
                    *loop_var,
                    loop_var_ty,
                    init,
                    init_bindings,
                    kind,
                    body,
                    scope,
                )?;
                self.operation(value, "control", scope)?;
            }
            TermKind::Soac(soac) => self.soac(value, soac, scope)?,
            TermKind::ArrayExpr(array) => {
                let array = self.array(array, value, scope)?;
                self.sink.set("SourceAlias", value, array)?;
            }
            TermKind::App { func, args } => {
                let function = self.operand(func, value, scope)?;
                if matches!(
                    func.kind,
                    TermKind::Var(VarRef::Symbol(_)) | TermKind::Lambda(_) | TermKind::Closure(_)
                ) {
                    self.sink.add("SourceApply", (value, function))?;
                }
                let arguments = args
                    .iter()
                    .map(|argument| self.operand(argument, value, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                self.application(value, func, &arguments)?;
                if let Some((_, index, _)) = tlc::record_update_parts(term) {
                    self.sink.add(
                        "SourceFieldUpdate",
                        (value, arguments[0], index as i64, arguments[2]),
                    )?;
                }
                self.arguments(value, arguments)?;
                if !facts::scalar_application(func, args, &term.ty) {
                    self.operation(value, "call", scope)?;
                }
            }
            TermKind::Index { array, index } => {
                let input = self.operand(array, value, scope)?;
                self.sink.set("SourceIndexedArray", value, input)?;
                let index = self.operand(index, value, scope)?;
                self.sink.set("SourceIndex", value, index)?;
                self.operation(value, "index", scope)?;
            }
            TermKind::Closure(closure) => {
                let code = self.resolve(closure.code)?;
                self.sink.add("SourceDirectUse", (value, code))?;
                self.sink.add("SourceClosureCode", (value, code))?;
                for capture in &closure.captures {
                    self.operand(capture, value, scope)?;
                }
            }
            TermKind::Coerce { inner, .. } => {
                let input = self.operand(inner, value, scope)?;
                self.sink.set("SourceAlias", value, input)?;
            }
            TermKind::TupleProj { tuple, idx } => {
                let input = self.operand(tuple, value, scope)?;
                self.sink.add("SourceSelect", (value, input, *idx as i64))?;
            }
            TermKind::Tuple(fields) | TermKind::VecLit(fields) => {
                let mut arguments = Vec::with_capacity(fields.len());
                self.sink.set(
                    "SourceConstruct",
                    value,
                    if matches!(term.kind, TermKind::Tuple(_)) { "tuple" } else { "vector" },
                )?;
                for (index, field) in fields.iter().enumerate() {
                    let field = self.operand(field, value, scope)?;
                    arguments.push(field);
                    self.sink.add("SourceField", (value, index as i64, field))?;
                    if as_soa_tuple(&term.ty).is_some() {
                        self.sink.add("SourceArrayPart", (value, index as i64, field))?;
                    }
                }
                self.arguments(value, arguments)?;
            }
        }
        self.finish_value(value, scope)?;
        self.sink.set("SourceEvaluationPosition", value, scope.position)?;
        scope.position += 1;
        Ok(value)
    }

    fn loop_body(
        &mut self,
        owner: Value,
        variable: SymbolId,
        variable_ty: &Type,
        initial: &'source Term,
        bindings: &'source [(SymbolId, Type, Term)],
        kind: &'source LoopKind<ExplicitClosurePayload, ExplicitCapturesPayload>,
        body: &'source Term,
        parent: &mut Scope,
    ) -> Result<(), OptimizeError> {
        let initial = self.operand(initial, owner, parent)?;
        match kind {
            LoopKind::For { iter, .. } => {
                let extent = self.operand(iter, owner, parent)?;
                let form = self.sink.add("ForEach", extent)?;
                self.sink.set("SourceLoopForm", owner, form)?;
            }
            LoopKind::ForRange { bound, var_ty, .. } => {
                let extent = self.operand(bound, owner, parent)?;
                let ty = self.ty(var_ty)?;
                let form = self.sink.add("ForCount", (extent, ty))?;
                self.sink.set("SourceLoopForm", owner, form)?;
            }
            LoopKind::While { .. } => {}
        }
        let mut header = self.scope(None, Some(parent.key))?;
        let mut iteration = self.scope(Some(body), Some(header.key))?;
        self.sink.add("SourceEnters", (owner, header.key))?;
        self.sink.add("SourceEnters", (owner, iteration.key))?;
        let info = self.sink.add("LoopInfo", (header.key, iteration.key))?;
        self.sink.set("SourceLoop", owner, info)?;
        let checkpoint = self.bindings.checkpoint();
        let accumulator = self.formal(variable, variable_ty, &header)?;
        let info = self.sink.add("LoopInitialInfo", (accumulator, initial))?;
        self.sink.set("SourceLoopInitial", header.key, info)?;
        for (symbol, _, extraction) in bindings {
            let (value, _) = self.visit(extraction, &mut header)?;
            self.bind(*symbol, value, &header)?;
        }
        match kind {
            LoopKind::While { cond } => {
                let (condition, _) = self.visit(cond, &mut header)?;
                let form = self.sink.add("WhileCondition", condition)?;
                self.sink.set("SourceLoopForm", owner, form)?;
            }
            LoopKind::For { var, var_ty, .. } | LoopKind::ForRange { var, var_ty, .. } => {
                let variable = self.formal(*var, var_ty, &iteration)?;
                self.sink.set("SourceIterationValue", iteration.key, variable)?;
            }
        }
        let result = self.finish_body(body, &mut iteration)?;
        self.sink.add("SourceLoopBackedge", (header.key, accumulator, result))?;
        if matches!(kind, LoopKind::ForRange { .. } | LoopKind::While { .. })
            && loops::supported(variable_ty)
        {
            self.sink.add("SourceDispatchLoop", owner)?;
            let domain =
                self.loop_carries(owner, variable_ty, initial, accumulator, result, owner, &mut 0)?;
            self.sink.set("SourceLoopDomain", owner, domain)?;
            self.sink.set(
                "SourceLoopDirectCarry",
                owner,
                matches!(kind, LoopKind::ForRange { .. }) && variable_ty.is_array(),
            )?;
        }
        self.sink.set("SourceRegionWork", header.key, header.work)?;
        self.bindings.restore(checkpoint);
        Ok(())
    }

    fn operator_body(
        &mut self,
        body: &'source OperatorBody,
        operation: Value,
        owner: Value,
        parent: &mut Scope,
    ) -> Result<(), OptimizeError> {
        let captures = body
            .data
            .captures
            .iter()
            .map(|(symbol, ty, term)| self.operand(term, owner, parent).map(|value| (*symbol, ty, value)))
            .collect::<Result<Vec<_>, _>>()?;
        let Lambda { params, body, ret_ty } = &body.lam;
        let mut scope = self.scope(Some(body), Some(parent.key))?;
        self.sink.set("SourceOperatorBody", operation, scope.key)?;
        self.sink.add("SourceEnters", (owner, scope.key))?;
        let checkpoint = self.bindings.checkpoint();
        let mut captured = Vec::new();
        for (symbol, ty, argument) in captures {
            let parameter = self.formal(symbol, ty, &scope)?;
            captured.push(parameter);
            self.sink.set("SourceCapture", (scope.key, parameter), argument)?;
        }
        self.parameters(params, &scope)?;
        if matches!(&body.kind, TermKind::Var(VarRef::Symbol(symbol)) if self.globals.contains(symbol)) {
            let (function, _) = self.visit(body, &mut scope)?;
            let mut arguments =
                params.iter().map(|(symbol, _)| self.resolve(*symbol)).collect::<Result<Vec<_>, _>>()?;
            arguments.extend(captured);
            let arguments = self.sink.container_to_value(VecContainer {
                data: arguments,
                do_rebuild: true,
            });
            let result = self.sink.add("SourceInvocation", (scope.key, function, arguments))?;
            self.value_type(result, ret_ty)?;
            self.sink.set("SourceResult", scope.key, result)?;
            self.sink.set("SourceRegionWork", scope.key, scope.work)?;
        } else {
            self.finish_body(body, &mut scope)?;
        }
        self.bindings.restore(checkpoint);
        Ok(())
    }

    fn array(
        &mut self,
        array: &'source ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload>,
        owner: Value,
        scope: &mut Scope,
    ) -> Result<Value, OptimizeError> {
        let value = if let ArrayExpr::Var(VarRef::Symbol(symbol), _) = array {
            self.resolve(*symbol)?
        } else {
            let value = self.sink.add("SourceArrayAtom", (owner, self.arrays.next_id()))?;
            self.value_type(value, &array.array_type())?;
            self.flags(value, false, true, true, true, 0)?;
            self.owners.insert(value, scope.key);
            value
        };
        self.sink.add("SourceDirectUse", (owner, value))?;
        let extent = match array {
            ArrayExpr::Var(VarRef::Symbol(_), _) => self.sink.add("Length", value)?,
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => {
                return Err(OptimizeError::Output("builtin used as a TLC array input".into()));
            }
            ArrayExpr::Zip(arrays) => {
                self.sink.set("SourceConstruct", value, "zip")?;
                let mut arguments = Vec::with_capacity(arrays.len());
                for (index, array) in arrays.iter().enumerate() {
                    let child = self.array(array, value, scope)?;
                    self.sink.add("SourceArrayPart", (value, index as i64, child))?;
                    arguments.push(child);
                }
                self.arguments(value, arguments)?;
                self.sink.add("Length", value)?
            }
            ArrayExpr::Literal(elements) => {
                self.flags(value, false, true, true, true, 0)?;
                self.sink.set("SourceConstruct", value, "array")?;
                let arguments = elements
                    .iter()
                    .map(|element| self.operand(element, value, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                self.arguments(value, arguments)?;
                self.sink.add("Fixed", elements.len() as i64)?
            }
            ArrayExpr::Range { start, len, step } => {
                self.flags(value, false, true, true, true, 0)?;
                let start = self.operand(start, value, scope)?;
                let length = self.operand(len, value, scope)?;
                self.sink.set("SourceConstruct", value, "range")?;
                let mut arguments = vec![start, length];
                if let Some(step) = step {
                    arguments.push(self.operand(step, value, scope)?);
                } else {
                    self.sink.add("SourceUnitRange", (value, start, length))?;
                }
                self.arguments(value, arguments)?;
                self.sink.add("Scalar", length)?
            }
        };
        self.sink.set("SourceExtent", value, extent)?;
        self.free_reference(value, scope.key)?;
        self.sink.add("SourceRegionUse", (scope.key, value))?;
        self.use_value(owner, value, !matches!(array, ArrayExpr::Var(_, _)))?;
        Ok(value)
    }

    fn input(
        &mut self,
        array: &'source ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload>,
        operation: Value,
        index: i64,
        owner: Value,
        scope: &mut Scope,
    ) -> Result<(), OptimizeError> {
        let value = self.array(array, owner, scope)?;
        self.sink.set("SourceInput", (operation, index), value)?;

        Ok(())
    }

    fn soac(
        &mut self,
        value: Value,
        soac: &'source SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>,
        scope: &mut Scope,
    ) -> Result<(), OptimizeError> {
        let (kind, body) = match soac {
            SoacOp::Map { lam, .. } => ("map", lam),
            SoacOp::Reduce { op, .. } => ("reduce", op),
            SoacOp::Scan { op, .. } => ("scan", op),
            SoacOp::Filter { pred, .. } => ("filter", pred),
            SoacOp::Scatter { lam, .. } => ("scatter", lam),
            SoacOp::BucketScatter { lam, .. } => ("bucket-scatter", lam),
            SoacOp::ReduceByIndex { op, .. } => ("reduce-by-index", op),
        };
        let operation = self.operation(value, kind, scope)?;
        self.operator_body(body, operation, value, scope)?;
        let count = match soac {
            SoacOp::Map { inputs, .. }
            | SoacOp::Scatter { inputs, .. }
            | SoacOp::BucketScatter { inputs, .. } => inputs.len(),
            SoacOp::Reduce { .. } | SoacOp::Scan { .. } | SoacOp::Filter { .. } => 1,
            SoacOp::ReduceByIndex { .. } => 2,
        };
        self.sink.set("SourceInputCount", operation, count as i64)?;
        match soac {
            SoacOp::Map { inputs, .. } => {
                for (index, input) in inputs.iter().enumerate() {
                    self.input(input, operation, index as i64, value, scope)?;
                }
            }
            SoacOp::Reduce { ne, input, .. } | SoacOp::Scan { ne, input, .. } => {
                let neutral = self.operand(ne, value, scope)?;
                self.sink.set("SourceNeutral", operation, neutral)?;
                self.input(input, operation, 0, value, scope)?;
            }
            SoacOp::Filter { input, .. } => self.input(input, operation, 0, value, scope)?,
            SoacOp::Scatter { dest, inputs, .. } | SoacOp::BucketScatter { dest, inputs, .. } => {
                let destination = self.resolve(dest.id)?;
                self.sink.add("SourceDirectUse", (value, destination))?;
                self.sink.set("SourceDestination", operation, destination)?;
                self.use_value(value, destination, false)?;
                for (index, input) in inputs.iter().enumerate() {
                    self.input(input, operation, index as i64, value, scope)?;
                }
            }
            SoacOp::ReduceByIndex {
                dest,
                ne,
                indices,
                values,
                ..
            } => {
                let destination = self.resolve(dest.id)?;
                self.sink.add("SourceDirectUse", (value, destination))?;
                self.sink.set("SourceDestination", operation, destination)?;
                self.use_value(value, destination, false)?;
                let neutral = self.operand(ne, value, scope)?;
                self.sink.set("SourceNeutral", operation, neutral)?;
                self.input(indices, operation, 0, value, scope)?;
                self.input(values, operation, 1, value, scope)?;
            }
        }
        self.collective_metadata(operation, soac)?;
        Ok(())
    }
}
