//! Higher-order-function specialization within defunctionalization.
//!
//! Callable values are inspected directly in the tree. Specialization caches
//! and generated-definition indexes are derived working state and remain
//! private to this pass.

use super::{ClosureConverted, Defunctionalized};
use crate::ast::{Span, TypeName};
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosure, ExplicitClosurePayload};
use crate::tlc::{
    self, apply_type_substitution, ArrayExpr, Def, DefMeta, LoopKind, Program, RewriteDecision, SoacBody,
    SoacOp, Term, TermId, TermIdSource, TermKind, TermRewriter, TypeSubstitution, VarRef,
};
use crate::{LookupMap, LookupSet, SymbolId, SymbolTable};
use polytype::Type;

// =============================================================================
// Verifier and HOF detection
// =============================================================================

#[derive(Debug)]
pub enum HofSpecializeError {
    FunctionTypedParam {
        def: SymbolId,
        param_index: usize,
    },
}

pub fn verify_hof_specialized<S>(program: &Program<S>) -> Result<(), HofSpecializeError>
where
    S: tlc::Stage<Family = ClosureConverted>,
{
    for def in &program.defs {
        let mut current = &def.ty;
        let mut param_index = 0;
        while let Type::Constructed(TypeName::Arrow, args) = current {
            if args.len() != 2 {
                break;
            }
            if is_arrow_type(&args[0]) {
                return Err(HofSpecializeError::FunctionTypedParam {
                    def: def.name,
                    param_index,
                });
            }
            current = &args[1];
            param_index += 1;
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct HofInfo {
    func_param_indices: Vec<usize>,
    def: Def<ClosureConverted>,
}

pub(super) fn is_arrow_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Arrow, _))
}

pub(super) fn extract_param_types(ty: &Type<TypeName>) -> Vec<Type<TypeName>> {
    let mut params = Vec::new();
    let mut current = ty;
    while let Type::Constructed(TypeName::Arrow, args) = current {
        if args.len() != 2 {
            break;
        }
        params.push(args[0].clone());
        current = &args[1];
    }
    params
}

fn detect_hofs(defs: &[Def<ClosureConverted>]) -> LookupMap<SymbolId, HofInfo> {
    let mut result = LookupMap::new();
    for def in defs {
        let func_param_indices = extract_param_types(&def.ty)
            .iter()
            .enumerate()
            .filter_map(|(index, ty)| is_arrow_type(ty).then_some(index))
            .collect::<Vec<_>>();
        if !func_param_indices.is_empty() {
            result.insert(
                def.name,
                HofInfo {
                    func_param_indices,
                    def: def.clone(),
                },
            );
        }
    }
    result
}

// =============================================================================
// Type substitution
// =============================================================================

pub(super) fn build_type_subst(
    polymorphic: &Type<TypeName>,
    concrete: &Type<TypeName>,
    subst: &mut TypeSubstitution,
) {
    match (polymorphic, concrete) {
        (Type::Variable(id), concrete) => {
            if let Some(existing) = subst.get(id) {
                assert_eq!(
                    existing, concrete,
                    "BUG: inconsistent type substitution for {id}: {existing:?} vs {concrete:?}"
                );
            } else {
                subst.insert(*id, concrete.clone());
            }
        }
        (Type::Constructed(_, left), Type::Constructed(_, right)) => {
            for (left, right) in left.iter().zip(right) {
                build_type_subst(left, right, subst);
            }
        }
        _ => {}
    }
}

pub(super) fn format_type_for_key(ty: &Type<TypeName>) -> String {
    format!("{ty:?}")
}

pub(super) fn apply_type_subst_to_term(
    term: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    subst: &TypeSubstitution,
    term_ids: &mut TermIdSource,
) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
    let mut result = term.clone();
    result.rewrite_types(term_ids, &mut |ty| apply_type_substitution(ty, subst));
    result
}

// =============================================================================
// Binder-aware term substitution
// =============================================================================

pub(super) fn get_func_param_sym(def: &Def<ClosureConverted>, param_idx: usize) -> SymbolId {
    let mut body = &def.body;
    let mut index = 0;
    while let TermKind::Lambda(lambda) = &body.kind {
        for (param, _) in &lambda.params {
            if index == param_idx {
                return *param;
            }
            index += 1;
        }
        body = &lambda.body;
    }
    panic!("BUG: parameter index {param_idx} is out of bounds");
}

fn substitute_term(
    term: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    old_symbol: SymbolId,
    replacement: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    term_ids: &mut TermIdSource,
) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
    let mut result = tlc::clone_term_with_fresh_ids(term, term_ids);
    substitute_in_place(&mut result, old_symbol, replacement, term_ids);
    result
}

fn substitute_in_place(
    term: &mut Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    old_symbol: SymbolId,
    replacement: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    term_ids: &mut TermIdSource,
) {
    match &mut term.kind {
        TermKind::Var(VarRef::Symbol(symbol)) if *symbol == old_symbol => {
            *term = tlc::clone_term_with_fresh_ids(replacement, term_ids);
        }
        TermKind::Lambda(lambda) => {
            if !lambda.params.iter().any(|(param, _)| *param == old_symbol) {
                substitute_in_place(&mut lambda.body, old_symbol, replacement, term_ids);
            }
        }
        TermKind::Let { name, rhs, body, .. } => {
            substitute_in_place(rhs, old_symbol, replacement, term_ids);
            if *name != old_symbol {
                substitute_in_place(body, old_symbol, replacement, term_ids);
            }
        }
        TermKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            substitute_in_place(init, old_symbol, replacement, term_ids);
            let shadows =
                *loop_var == old_symbol || init_bindings.iter().any(|(name, _, _)| *name == old_symbol);
            if !shadows {
                for (_, _, value) in init_bindings {
                    substitute_in_place(value, old_symbol, replacement, term_ids);
                }
                match kind {
                    LoopKind::For { var, iter, .. } | LoopKind::ForRange { var, bound: iter, .. } => {
                        substitute_in_place(iter, old_symbol, replacement, term_ids);
                        if *var != old_symbol {
                            substitute_in_place(body, old_symbol, replacement, term_ids);
                        }
                    }
                    LoopKind::While { cond } => {
                        substitute_in_place(cond, old_symbol, replacement, term_ids);
                        substitute_in_place(body, old_symbol, replacement, term_ids);
                    }
                }
            }
        }
        TermKind::Soac(soac) => {
            substitute_in_soac(soac, old_symbol, replacement, term_ids);
        }
        _ => term
            .for_each_child_mut(&mut |child| substitute_in_place(child, old_symbol, replacement, term_ids)),
    }
}

fn substitute_in_soac(
    soac: &mut SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>,
    old_symbol: SymbolId,
    replacement: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    term_ids: &mut TermIdSource,
) {
    fn body(
        body: &mut SoacBody<ExplicitClosurePayload, ExplicitCapturesPayload>,
        old_symbol: SymbolId,
        replacement: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
        term_ids: &mut TermIdSource,
    ) {
        if !body.lam.params.iter().any(|(param, _)| *param == old_symbol) {
            substitute_in_place(&mut body.lam.body, old_symbol, replacement, term_ids);
        }
        for (_, _, capture) in &mut body.data.captures {
            substitute_in_place(capture, old_symbol, replacement, term_ids);
        }
    }

    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            body(lam, old_symbol, replacement, term_ids);
            for input in inputs {
                substitute_in_array(input, old_symbol, replacement, term_ids);
            }
        }
        SoacOp::Reduce { op, ne, input } | SoacOp::Scan { op, ne, input, .. } => {
            body(op, old_symbol, replacement, term_ids);
            substitute_in_place(ne, old_symbol, replacement, term_ids);
            substitute_in_array(input, old_symbol, replacement, term_ids);
        }
        SoacOp::Filter { pred, input, .. } => {
            body(pred, old_symbol, replacement, term_ids);
            substitute_in_array(input, old_symbol, replacement, term_ids);
        }
        SoacOp::Scatter { lam, inputs, .. } => {
            body(lam, old_symbol, replacement, term_ids);
            for input in inputs {
                substitute_in_array(input, old_symbol, replacement, term_ids);
            }
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            body(op, old_symbol, replacement, term_ids);
            substitute_in_place(ne, old_symbol, replacement, term_ids);
            substitute_in_array(indices, old_symbol, replacement, term_ids);
            substitute_in_array(values, old_symbol, replacement, term_ids);
        }
    }
}

fn substitute_in_array(
    array: &mut ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload>,
    old_symbol: SymbolId,
    replacement: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    term_ids: &mut TermIdSource,
) {
    match array {
        ArrayExpr::Var(var, ty) => {
            if matches!(var, VarRef::Symbol(symbol) if *symbol == old_symbol) {
                let TermKind::Var(replacement_var) = &replacement.kind else {
                    panic!("a closure value cannot occupy a named array-input edge");
                };
                *var = *replacement_var;
                *ty = replacement.ty.clone();
            }
        }
        ArrayExpr::Zip(parts) => {
            for part in parts {
                substitute_in_array(part, old_symbol, replacement, term_ids);
            }
        }
        ArrayExpr::Literal(terms) => {
            for term in terms {
                substitute_in_place(term, old_symbol, replacement, term_ids);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            substitute_in_place(start, old_symbol, replacement, term_ids);
            substitute_in_place(len, old_symbol, replacement, term_ids);
            if let Some(step) = step {
                substitute_in_place(step, old_symbol, replacement, term_ids);
            }
        }
    }
}

// =============================================================================
// Specialization
// =============================================================================

#[derive(Debug, Clone)]
struct ResolvedCallable {
    code: SymbolId,
    captures: Vec<Term<ExplicitClosurePayload, ExplicitCapturesPayload>>,
    param_count: usize,
}

fn resolve_callable(
    term: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    top_level: &LookupSet<SymbolId>,
) -> Option<ResolvedCallable> {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(code)) if top_level.contains(code) && is_arrow_type(&term.ty) => {
            Some(ResolvedCallable {
                code: *code,
                captures: Vec::new(),
                param_count: extract_param_types(&term.ty).len(),
            })
        }
        TermKind::Closure(closure) => Some(ResolvedCallable {
            code: closure.code,
            captures: closure.captures.clone(),
            param_count: closure.param_count,
        }),
        _ => None,
    }
}

fn build_specialized_call(
    specialized_symbol: SymbolId,
    function_param_index: usize,
    arguments: &[Term<ExplicitClosurePayload, ExplicitCapturesPayload>],
    captures: &[Term<ExplicitClosurePayload, ExplicitCapturesPayload>],
    ty: Type<TypeName>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
    let mut call_arguments = arguments
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != function_param_index)
        .map(|(_, argument)| tlc::clone_term_with_fresh_ids(argument, term_ids))
        .collect::<Vec<_>>();
    call_arguments.extend(captures.iter().map(|capture| tlc::clone_term_with_fresh_ids(capture, term_ids)));
    tlc::build_app_call(specialized_symbol, call_arguments, ty, span, term_ids)
}

#[derive(Debug, Clone)]
struct ClosureSpecialization {
    symbol: SymbolId,
    environment_params: Vec<Vec<(SymbolId, Type<TypeName>)>>,
}

struct HofSpecializer<'a> {
    symbols: &'a mut SymbolTable,
    top_level: LookupSet<SymbolId>,
    hof_info: LookupMap<SymbolId, HofInfo>,
    specialized_defs: Vec<Def<ClosureConverted>>,
    specialization_cache: LookupMap<(SymbolId, usize, SymbolId, Vec<String>), SymbolId>,
    closure_spec_cache: LookupMap<(SymbolId, Vec<SymbolId>), ClosureSpecialization>,
    defs_by_sym: LookupMap<SymbolId, Def<ClosureConverted>>,
    specialization_counter: usize,
    term_ids: &'a mut TermIdSource,
}

impl TermRewriter<ExplicitClosurePayload, ExplicitCapturesPayload> for HofSpecializer<'_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(
        &mut self,
        term: &mut Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> RewriteDecision {
        if self.maybe_specialize_hof_call(term) {
            RewriteDecision::Changed
        } else {
            RewriteDecision::Unchanged
        }
    }
}

impl HofSpecializer<'_> {
    fn maybe_specialize_hof_call(
        &mut self,
        term: &mut Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> bool {
        let TermKind::App { func, args } = &term.kind else {
            return false;
        };
        let TermKind::Var(VarRef::Symbol(hof_symbol)) = &func.kind else {
            return false;
        };
        let Some(hof) = self.hof_info.get(hof_symbol).cloned() else {
            return false;
        };
        for function_param_index in hof.func_param_indices.iter().copied() {
            let Some(argument) = args.get(function_param_index) else {
                continue;
            };
            let Some(callable) = resolve_callable(argument, &self.top_level) else {
                continue;
            };
            *term = self.specialize_call(
                *hof_symbol,
                &hof.def,
                function_param_index,
                &callable,
                args,
                term.ty.clone(),
                term.span,
            );
            return true;
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn specialize_call(
        &mut self,
        hof_symbol: SymbolId,
        hof_def: &Def<ClosureConverted>,
        function_param_index: usize,
        callable: &ResolvedCallable,
        arguments: &[Term<ExplicitClosurePayload, ExplicitCapturesPayload>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
        let key = (
            hof_symbol,
            function_param_index,
            callable.code,
            arguments.iter().map(|argument| format_type_for_key(&argument.ty)).collect(),
        );
        if let Some(symbol) = self.specialization_cache.get(&key).copied() {
            return build_specialized_call(
                symbol,
                function_param_index,
                arguments,
                &callable.captures,
                ty,
                span,
                self.term_ids,
            );
        }

        let mut type_subst = TypeSubstitution::new();
        for (polymorphic, argument) in extract_param_types(&hof_def.ty).iter().zip(arguments) {
            build_type_subst(polymorphic, &argument.ty, &mut type_subst);
        }

        let hof_name = self.symbols.get(hof_symbol).expect("BUG: HOF symbol not in table");
        let specialized_name = format!("{hof_name}${}", self.specialization_counter);
        self.specialization_counter += 1;
        let specialized_symbol = self.symbols.alloc(specialized_name.clone());
        self.specialization_cache.insert(key, specialized_symbol);

        let function_param = get_func_param_sym(hof_def, function_param_index);
        let (params, inner_body) = tlc::extract_lambda_params(&hof_def.body);
        let mut new_params = params
            .into_iter()
            .enumerate()
            .filter_map(|(index, param)| (index != function_param_index).then_some(param))
            .collect::<Vec<_>>();

        let mut closure_captures = Vec::new();
        for capture in &callable.captures {
            let outer_symbol = match &capture.kind {
                TermKind::Var(VarRef::Symbol(symbol)) => *symbol,
                _ => panic!("BUG: closure capture is not a variable"),
            };
            let outer_name = crate::symbol_name_or_bug(self.symbols, outer_symbol);
            let fresh_symbol = self.symbols.alloc(format!("{specialized_name}__cap_{outer_name}"));
            new_params.push((fresh_symbol, capture.ty.clone()));
            closure_captures.push(Term {
                id: self.term_ids.next_id(),
                ty: capture.ty.clone(),
                span: capture.span,
                kind: TermKind::Var(VarRef::Symbol(fresh_symbol)),
            });
        }
        let callable_term = Term {
            id: self.term_ids.next_id(),
            ty: arguments[function_param_index].ty.clone(),
            span: arguments[function_param_index].span,
            kind: if closure_captures.is_empty() {
                TermKind::Var(VarRef::Symbol(callable.code))
            } else {
                TermKind::Closure(ExplicitClosure {
                    code: callable.code,
                    captures: closure_captures,
                    param_count: callable.param_count,
                })
            },
        };
        let mut body = substitute_term(&inner_body, function_param, &callable_term, self.term_ids);
        self.rewrite_tracked(&mut body);
        let body = apply_type_subst_to_term(&body, &type_subst, self.term_ids);
        let new_params = new_params
            .into_iter()
            .map(|(symbol, ty)| (symbol, apply_type_substitution(&ty, &type_subst)))
            .collect::<Vec<_>>();
        let body = tlc::rebuild_nested_lam(&new_params, body, hof_def.body.span, self.term_ids);
        let arity = new_params.len();
        self.specialized_defs.push(Def {
            data: (),
            name: specialized_symbol,
            ty: body.ty.clone(),
            body,
            meta: DefMeta::Function,
            arity,
            param_diets: vec![crate::types::Diet::observing(); arity],
            return_diet: crate::types::Diet::observing(),
        });
        self.top_level.insert(specialized_symbol);

        build_specialized_call(
            specialized_symbol,
            function_param_index,
            arguments,
            &callable.captures,
            ty,
            span,
            self.term_ids,
        )
    }

    fn cascade_specialize_term(
        &mut self,
        term: &mut Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> bool {
        let mut changed = false;
        term.for_each_child_mut(&mut |child| {
            changed |= self.cascade_specialize_term(child);
        });
        if let TermKind::Soac(soac) = &mut term.kind {
            changed |= self.cascade_specialize_soac(soac);
        }
        if changed {
            term.id = self.term_ids.next_id();
        }
        changed
    }

    fn cascade_specialize_soac(
        &mut self,
        soac: &mut SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> bool {
        match soac {
            SoacOp::Map { lam, .. } | SoacOp::Scatter { lam, .. } => self.cascade_specialize_soac_body(lam),
            SoacOp::Reduce { op, .. } | SoacOp::Scan { op, .. } | SoacOp::ReduceByIndex { op, .. } => {
                self.cascade_specialize_soac_body(op)
            }
            SoacOp::Filter { pred, .. } => self.cascade_specialize_soac_body(pred),
        }
    }

    fn cascade_specialize_soac_body(
        &mut self,
        body: &mut SoacBody<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> bool {
        let callable_indices = body
            .data
            .captures
            .iter()
            .enumerate()
            .filter_map(|(index, (_, ty, term))| {
                (is_arrow_type(ty) && resolve_callable(term, &self.top_level).is_some()).then_some(index)
            })
            .collect::<Vec<_>>();
        if callable_indices.is_empty() {
            return false;
        }
        let TermKind::Var(VarRef::Symbol(lifted_symbol)) = &body.lam.body.kind else {
            return false;
        };
        let lifted_symbol = *lifted_symbol;
        if !self.defs_by_sym.contains_key(&lifted_symbol) {
            return false;
        }

        let callables = callable_indices
            .iter()
            .map(|index| {
                resolve_callable(&body.data.captures[*index].2, &self.top_level)
                    .expect("filtered to callable captures")
            })
            .collect::<Vec<_>>();
        let key = (
            lifted_symbol,
            callables.iter().map(|callable| callable.code).collect(),
        );
        let specialization = if let Some(existing) = self.closure_spec_cache.get(&key) {
            existing.clone()
        } else {
            self.specialize_closure(lifted_symbol, &callable_indices, &callables, &body.data.captures)
        };

        let specialized_ty = self
            .defs_by_sym
            .get(&specialization.symbol)
            .expect("new specialization must be indexed")
            .ty
            .clone();
        *body.lam.body = Term {
            id: self.term_ids.next_id(),
            ty: specialized_ty,
            span: body.lam.body.span,
            kind: TermKind::Var(VarRef::Symbol(specialization.symbol)),
        };

        let mut callable_group = 0;
        let captures = std::mem::take(&mut body.data.captures);
        for (index, capture) in captures.into_iter().enumerate() {
            if callable_indices.contains(&index) {
                let callable = resolve_callable(&capture.2, &self.top_level)
                    .expect("selected capture must remain callable");
                let params = &specialization.environment_params[callable_group];
                callable_group += 1;
                for ((formal, ty), actual) in params.iter().zip(callable.captures) {
                    body.data.captures.push((*formal, ty.clone(), actual));
                }
            } else {
                body.data.captures.push(capture);
            }
        }
        true
    }

    fn specialize_closure(
        &mut self,
        lifted_symbol: SymbolId,
        callable_indices: &[usize],
        callables: &[ResolvedCallable],
        captures: &[(
            SymbolId,
            Type<TypeName>,
            Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
        )],
    ) -> ClosureSpecialization {
        let lifted_def = self
            .defs_by_sym
            .get(&lifted_symbol)
            .expect("lifted def was checked before specialization")
            .clone();
        let (params, inner_body) = tlc::extract_lambda_params(&lifted_def.body);
        let dropped = callable_indices.iter().map(|index| captures[*index].0).collect::<LookupSet<_>>();
        let mut new_params =
            params.into_iter().filter(|(symbol, _)| !dropped.contains(symbol)).collect::<Vec<_>>();
        let mut body = inner_body;
        let mut environment_params = Vec::new();

        for ((index, callable), ordinal) in callable_indices.iter().zip(callables).zip(0usize..) {
            let local_callable = captures[*index].0;
            let mut params_for_callable = Vec::new();
            let mut closure_captures = Vec::new();
            for (capture_ordinal, capture) in callable.captures.iter().enumerate() {
                let symbol = self.symbols.alloc(format!(
                    "_w_closure_env_{}_{}_{}",
                    self.specialization_counter, ordinal, capture_ordinal
                ));
                let param = (symbol, capture.ty.clone());
                new_params.push(param.clone());
                params_for_callable.push(param);
                closure_captures.push(Term {
                    id: self.term_ids.next_id(),
                    ty: capture.ty.clone(),
                    span: capture.span,
                    kind: TermKind::Var(VarRef::Symbol(symbol)),
                });
            }
            let replacement = Term {
                id: self.term_ids.next_id(),
                ty: captures[*index].1.clone(),
                span: captures[*index].2.span,
                kind: if closure_captures.is_empty() {
                    TermKind::Var(VarRef::Symbol(callable.code))
                } else {
                    TermKind::Closure(ExplicitClosure {
                        code: callable.code,
                        captures: closure_captures,
                        param_count: callable.param_count,
                    })
                },
            };
            body = substitute_term(&body, local_callable, &replacement, self.term_ids);
            environment_params.push(params_for_callable);
        }

        self.cascade_specialize_term(&mut body);
        let body = tlc::rebuild_nested_lam(&new_params, body, lifted_def.body.span, self.term_ids);
        let name = crate::symbol_name_or_bug(self.symbols, lifted_symbol).to_string();
        let symbol = self.symbols.alloc(format!("{name}${}", self.specialization_counter));
        self.specialization_counter += 1;
        self.top_level.insert(symbol);
        let arity = new_params.len();
        let def = Def {
            data: (),
            name: symbol,
            ty: body.ty.clone(),
            body,
            meta: lifted_def.meta,
            arity,
            param_diets: vec![crate::types::Diet::observing(); arity],
            return_diet: crate::types::Diet::observing(),
        };
        self.defs_by_sym.insert(symbol, def.clone());
        self.specialized_defs.push(def);

        let specialization = ClosureSpecialization {
            symbol,
            environment_params,
        };
        self.closure_spec_cache.insert(
            (
                lifted_symbol,
                callables.iter().map(|callable| callable.code).collect(),
            ),
            specialization.clone(),
        );
        specialization
    }
}

pub(super) fn run(program: &mut Program<Defunctionalized>) {
    let hof_info = detect_hofs(&program.defs);
    let top_level = program.defs.iter().map(|def| def.name).collect();

    let mut specializer = HofSpecializer {
        symbols: &mut program.symbols,
        top_level,
        hof_info,
        specialized_defs: Vec::new(),
        specialization_cache: LookupMap::new(),
        closure_spec_cache: LookupMap::new(),
        defs_by_sym: LookupMap::new(),
        specialization_counter: 0,
        term_ids: &mut program.term_ids,
    };

    for def in &mut program.defs {
        specializer.rewrite_tracked(&mut def.body);
    }
    for def in program.defs.iter() {
        specializer.defs_by_sym.insert(def.name, def.clone());
    }
    for def in &specializer.specialized_defs {
        specializer.defs_by_sym.insert(def.name, def.clone());
    }
    for def in &mut program.defs {
        specializer.cascade_specialize_term(&mut def.body);
    }
    let mut initially_specialized = std::mem::take(&mut specializer.specialized_defs);
    for def in &mut initially_specialized {
        specializer.cascade_specialize_term(&mut def.body);
    }
    let cascade_specialized = std::mem::take(&mut specializer.specialized_defs);
    program.defs.extend(initially_specialized);
    program.defs.extend(cascade_specialized);
}

#[cfg(test)]
#[path = "hof_specialize_tests.rs"]
mod hof_specialize_tests;
