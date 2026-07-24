//! Monomorphization of TLC definitions.
//!
//! The pass starts from entry points and emits one reachable definition for
//! each concrete instantiation it encounters. Type schemes are consumed from
//! their owning definitions; specialization indexes and the worklist are
//! derived state private to this run.
//!
//! **Representation variants.** Type substitution replaces type variables but
//! deliberately preserves `ArrayVariantAbstract`. It is a first-class
//! representation-polymorphic variant, not a placeholder. EGIR lowering
//! chooses the concrete representation, and `egir::verify_no_abstract` guards
//! the backend boundary.

use super::data::Empty;
use super::run::Polymorphic;
use super::soa::SoaNormalized;
use super::{
    apply_type_substitution, ArrayExpr, Def, DefMeta, Program, RewriteDecision, Term, TermId, TermIdSource,
    TermKind, TermRewriter, TypeSubstitution, VarRef,
};
use crate::ast::TypeName;
use crate::types::{TypeExt, TypeScheme};
use crate::{LookupMap, LookupSet, SymbolId, SymbolTable};
use polytype::Type;
use std::collections::VecDeque;

/// Monomorphic TLC stores no per-definition payload; monomorphization consumes
/// the source schemes while constructing this family.
#[derive(Debug, Clone, Copy, Default)]
pub struct Monomorphic;

impl super::Family for Monomorphic {
    type DefinitionData = ();
    type EntryData = ();
    type ClosureData = Empty;
    type SoacBodyData = Empty;
}

/// TLC after intrinsic specialization and reachable user-function
/// monomorphization.
#[derive(Debug, Clone, Copy, Default)]
pub struct Monomorphized;

impl super::Stage for Monomorphized {
    type Family = Monomorphic;
    type GlobalContext = super::context::RewriteGlobal;
}

/// Specialize intrinsic calls, then consume the polymorphic definition graph
/// into its reachable monomorphic graph.
pub fn run(mut program: Program<SoaNormalized>) -> Program<Monomorphized> {
    super::specialize::run(&mut program);

    let Program {
        defs,
        mut symbols,
        def_syms,
        mut term_ids,
        global_context,
    } = program;
    let defs = Monomorphizer::new(&mut symbols, defs, &mut term_ids).monomorphize();
    let program = Program::from_parts(defs, symbols, def_syms, term_ids, global_context);
    program.assert_flat_apps();
    program
}

struct Monomorphizer<'symbols, 'ids> {
    symbols: &'symbols mut SymbolTable,
    /// Original definitions retained as specialization templates.
    definitions: LookupMap<SymbolId, Def<Polymorphic>>,
    /// Small pass-local signature index. Keeping this separate lets ordinary
    /// monomorphic definitions move into the result without cloning their
    /// trees merely to answer later reference queries.
    definition_info: LookupMap<SymbolId, DefinitionInfo>,
    mono_functions: Vec<Def<Monomorphic>>,
    specializations: LookupMap<(SymbolId, SpecKey), SymbolId>,
    worklist: VecDeque<WorkItem>,
    processed: LookupSet<(SymbolId, SpecKey)>,
    term_ids: &'ids mut TermIdSource,
}

#[derive(Clone)]
struct DefinitionInfo {
    scheme: Option<TypeScheme>,
    ty: Type<TypeName>,
}

impl DefinitionInfo {
    fn polymorphic_type(&self) -> &Type<TypeName> {
        self.scheme.as_ref().map(unwrap_scheme).unwrap_or(&self.ty)
    }

    fn may_need_as_template(&self) -> bool {
        matches!(&self.scheme, Some(TypeScheme::Polytype { .. })) || !self.ty.vars().is_empty()
    }
}

struct WorkItem {
    original_sym: SymbolId,
    spec_key: SpecKey,
    output_sym: SymbolId,
}

/// Deterministic, hashable form of a type substitution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SubstKey(Vec<(usize, Type<TypeName>)>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SpecKey {
    type_subst: SubstKey,
}

impl SubstKey {
    fn from_subst(subst: &TypeSubstitution) -> Self {
        let mut items: Vec<_> = subst.iter().map(|(variable, ty)| (*variable, ty.clone())).collect();
        items.sort_by_key(|(variable, _)| *variable);
        Self(items)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn to_subst(&self) -> TypeSubstitution {
        self.0.iter().cloned().collect()
    }
}

impl SpecKey {
    fn empty() -> Self {
        Self {
            type_subst: SubstKey(Vec::new()),
        }
    }

    fn new(subst: &TypeSubstitution) -> Self {
        Self {
            type_subst: SubstKey::from_subst(subst),
        }
    }

    fn needs_specialization(&self) -> bool {
        !self.type_subst.is_empty()
    }
}

fn unwrap_scheme(scheme: &TypeScheme) -> &Type<TypeName> {
    match scheme {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { body, .. } => unwrap_scheme(body),
    }
}

fn split_function_type(ty: &Type<TypeName>) -> (Vec<Type<TypeName>>, Type<TypeName>) {
    let mut params = Vec::new();
    let mut current = ty.clone();
    loop {
        let Type::Constructed(TypeName::Arrow, args) = &current else {
            break;
        };
        if args.len() != 2 {
            break;
        }
        params.push(args[0].clone());
        current = args[1].clone();
    }
    (params, current)
}

impl<'symbols, 'ids> Monomorphizer<'symbols, 'ids> {
    fn new(
        symbols: &'symbols mut SymbolTable,
        defs: Vec<Def<Polymorphic>>,
        term_ids: &'ids mut TermIdSource,
    ) -> Self {
        let mut definitions = LookupMap::new();
        let mut definition_info = LookupMap::new();
        let mut worklist = VecDeque::new();

        for def in defs {
            let symbol = def.name;
            if matches!(&def.meta, DefMeta::EntryPoint(_)) {
                worklist.push_back(WorkItem {
                    original_sym: symbol,
                    spec_key: SpecKey::empty(),
                    output_sym: symbol,
                });
            }
            definition_info.insert(
                symbol,
                DefinitionInfo {
                    scheme: def.data.scheme.clone(),
                    ty: def.ty.clone(),
                },
            );
            definitions.insert(symbol, def);
        }

        Self {
            symbols,
            definitions,
            definition_info,
            mono_functions: Vec::new(),
            specializations: LookupMap::new(),
            worklist,
            processed: LookupSet::new(),
            term_ids,
        }
    }

    fn monomorphize(mut self) -> Vec<Def<Monomorphic>> {
        while let Some(work_item) = self.worklist.pop_front() {
            let key = (work_item.original_sym, work_item.spec_key.clone());
            if !self.processed.insert(key) {
                continue;
            }

            let def = self.materialize_work_item(&work_item);
            let def = self.process_def(def);
            self.mono_functions.push(def);
        }
        self.mono_functions
    }

    fn materialize_work_item(&mut self, work_item: &WorkItem) -> Def<Polymorphic> {
        let info = self
            .definition_info
            .get(&work_item.original_sym)
            .expect("BUG: monomorphization work item has no definition info");
        let must_keep_template = work_item.spec_key.needs_specialization() || info.may_need_as_template();
        let mut def = if must_keep_template {
            self.definitions
                .get(&work_item.original_sym)
                .expect("BUG: specialization template is missing")
                .clone()
        } else {
            self.definitions
                .remove(&work_item.original_sym)
                .expect("BUG: monomorphic work item was already consumed")
        };

        if work_item.spec_key.needs_specialization() {
            let subst = work_item.spec_key.type_subst.to_subst();
            def.name = work_item.output_sym;
            def.ty = apply_type_substitution(&def.ty, &subst);
            def.body.rewrite_types(self.term_ids, &mut |ty| apply_type_substitution(ty, &subst));
        }
        def
    }

    fn process_def(&mut self, def: Def<Polymorphic>) -> Def<Monomorphic> {
        let Def {
            data: _,
            name,
            ty,
            body,
            meta,
            arity,
            param_diets,
            return_diet,
        } = def;
        Def {
            data: (),
            name,
            ty,
            body: body.rewrite(self),
            meta,
            arity,
            param_diets,
            return_diet,
        }
    }

    fn ensure_in_worklist(&mut self, symbol: SymbolId) {
        let spec_key = SpecKey::empty();
        let key = (symbol, spec_key.clone());
        if self.processed.contains(&key)
            || self
                .worklist
                .iter()
                .any(|work| work.original_sym == symbol && !work.spec_key.needs_specialization())
        {
            return;
        }
        self.worklist.push_back(WorkItem {
            original_sym: symbol,
            spec_key,
            output_sym: symbol,
        });
    }

    fn rewrite_symbol_reference(
        &mut self,
        symbol: SymbolId,
        concrete_type: &Type<TypeName>,
    ) -> Option<SymbolId> {
        let info = self.definition_info.get(&symbol)?.clone();
        let Some(subst) = self.infer_var_substitution(&info, concrete_type) else {
            self.ensure_in_worklist(symbol);
            return None;
        };
        let spec_key = SpecKey::new(&subst);
        if spec_key.needs_specialization() {
            Some(self.get_or_create_specialization(symbol, &spec_key))
        } else {
            self.ensure_in_worklist(symbol);
            None
        }
    }

    fn rewrite_array_references(&mut self, array: &mut ArrayExpr<Empty, Empty>) -> bool {
        match array {
            ArrayExpr::Var(VarRef::Symbol(symbol), ty) => {
                let Some(specialized) = self.rewrite_symbol_reference(*symbol, ty) else {
                    return false;
                };
                *symbol = specialized;
                true
            }
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => false,
            ArrayExpr::Zip(inputs) => {
                let mut changed = false;
                for input in inputs {
                    changed |= self.rewrite_array_references(input);
                }
                changed
            }
            // Their term children have already been handled by TermRewriter.
            ArrayExpr::Literal(_) | ArrayExpr::Range { .. } => false,
        }
    }

    fn infer_substitution(&self, info: &DefinitionInfo, arg_types: &[Type<TypeName>]) -> TypeSubstitution {
        let mut subst = TypeSubstitution::new();
        let (param_types, _) = split_function_type(info.polymorphic_type());
        for (param_ty, arg_ty) in param_types.iter().zip(arg_types) {
            Self::unify_for_subst(param_ty, arg_ty, &mut subst);
        }
        subst
    }

    fn infer_var_substitution(
        &self,
        info: &DefinitionInfo,
        concrete_type: &Type<TypeName>,
    ) -> Option<TypeSubstitution> {
        let mut subst = TypeSubstitution::new();
        Self::unify_for_subst(info.polymorphic_type(), concrete_type, &mut subst);
        (!subst.is_empty()).then_some(subst)
    }

    fn unify_for_subst(expected: &Type<TypeName>, actual: &Type<TypeName>, subst: &mut TypeSubstitution) {
        match (expected, actual) {
            (Type::Variable(id), concrete) => {
                subst.insert(*id, concrete.clone());
            }
            (
                Type::Constructed(expected_name, expected_args),
                Type::Constructed(actual_name, actual_args),
            ) if expected_name == actual_name && expected_args.len() == actual_args.len() => {
                for (expected, actual) in expected_args.iter().zip(actual_args) {
                    Self::unify_for_subst(expected, actual, subst);
                }
            }
            _ => {}
        }
    }

    fn get_or_create_specialization(&mut self, function: SymbolId, spec_key: &SpecKey) -> SymbolId {
        let cache_key = (function, spec_key.clone());
        if let Some(specialized) = self.specializations.get(&cache_key) {
            return *specialized;
        }

        let subst = spec_key.type_subst.to_subst();
        let function_name = self.symbols.get(function).expect("BUG: function symbol is missing");
        let suffix = format_subst(&subst);
        let specialized_name =
            if suffix.is_empty() { function_name.to_string() } else { format!("{function_name}${suffix}") };
        let specialized = self.symbols.alloc(specialized_name);
        self.worklist.push_back(WorkItem {
            original_sym: function,
            spec_key: spec_key.clone(),
            output_sym: specialized,
        });
        self.specializations.insert(cache_key, specialized);
        specialized
    }
}

impl TermRewriter<Empty, Empty> for Monomorphizer<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node_before_children(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        let TermKind::App { func, args } = &mut term.kind else {
            return RewriteDecision::Unchanged;
        };
        let TermKind::Var(VarRef::Symbol(symbol)) = &func.kind else {
            return RewriteDecision::Unchanged;
        };
        let symbol = *symbol;
        let Some(info) = self.definition_info.get(&symbol).cloned() else {
            return RewriteDecision::Unchanged;
        };

        let arg_types = args.iter().map(|arg| arg.ty.clone()).collect::<Vec<_>>();
        let subst = self.infer_substitution(&info, &arg_types);
        let spec_key = SpecKey::new(&subst);
        if !spec_key.needs_specialization() {
            self.ensure_in_worklist(symbol);
            return RewriteDecision::Unchanged;
        }

        let specialized = self.get_or_create_specialization(symbol, &spec_key);
        func.kind = TermKind::Var(VarRef::Symbol(specialized));
        func.id = self.term_ids.next_id();
        RewriteDecision::Changed
    }

    fn rewrite_node(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        match &mut term.kind {
            TermKind::Var(VarRef::Symbol(symbol)) => {
                let Some(specialized) = self.rewrite_symbol_reference(*symbol, &term.ty) else {
                    return RewriteDecision::Unchanged;
                };
                *symbol = specialized;
                RewriteDecision::Changed
            }
            TermKind::ArrayExpr(array) => {
                if self.rewrite_array_references(array) {
                    RewriteDecision::Changed
                } else {
                    RewriteDecision::Unchanged
                }
            }
            _ => RewriteDecision::Unchanged,
        }
    }
}

fn format_subst(subst: &TypeSubstitution) -> String {
    let mut items: Vec<_> = subst.iter().collect();
    items.sort_by_key(|(variable, _)| *variable);
    items.iter().map(|(_, ty)| format_type_compact(ty)).collect::<Vec<_>>().join("_")
}

fn format_type_compact(ty: &Type<TypeName>) -> String {
    match ty {
        Type::Variable(id) => format!("v{id}"),
        Type::Constructed(TypeName::Size(n), _) => format!("n{n}"),
        Type::Constructed(TypeName::Bool, _) => "bool".to_string(),
        _ if ty.is_array() => {
            format!(
                "arr{}_{}{}",
                format_type_compact(ty.elem_type().expect("Array has elem")),
                format_type_compact(ty.array_size().expect("Array has size")),
                format_type_compact(ty.array_variant().expect("Array has variant"))
            )
        }
        Type::Constructed(TypeName::Tuple(arity), args) => {
            let args = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("tup{arity}_{args}")
        }
        _ if ty.is_vec() => {
            let elem = format_type_compact(ty.elem_type().expect("Vec has elem"));
            let size = format_type_compact(ty.vec_size_type().expect("Vec has size"));
            format!("vec_{elem}_{size}")
        }
        Type::Constructed(TypeName::Float(bits), _) => format!("f{bits}"),
        Type::Constructed(TypeName::Int(bits), _) => format!("i{bits}"),
        Type::Constructed(TypeName::UInt(bits), _) => format!("u{bits}"),
        Type::Constructed(TypeName::Arrow, args) => {
            let args = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("fn_{args}")
        }
        Type::Constructed(TypeName::Unit, _) => "unit".to_string(),
        Type::Constructed(TypeName::Named(name), args) if args.is_empty() => name.clone(),
        Type::Constructed(TypeName::Named(name), args) => {
            let args = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("{name}_{args}")
        }
        Type::Constructed(TypeName::ArrayVariantView, _) => "array_view".to_string(),
        Type::Constructed(TypeName::Buffer(binding), _) => {
            format!("buffer_s{}_b{}", binding.set, binding.binding)
        }
        Type::Constructed(TypeName::NoBuffer, _) => "no_buffer".to_string(),
        Type::Constructed(TypeName::ArrayVariantComposite, _) => "array_composite".to_string(),
        Type::Constructed(TypeName::ArrayVariantVirtual, _) => "array_virtual".to_string(),
        Type::Constructed(TypeName::ArrayVariantBounded, _) => "array_bounded".to_string(),
        Type::Constructed(TypeName::ArrayVariantAbstract, _) => "array_abstract".to_string(),
        Type::Constructed(name, args) => {
            let args = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            if args.is_empty() {
                format!("{name:?}")
            } else {
                format!("{name:?}_{args}")
            }
        }
    }
}

#[cfg(test)]
#[path = "monomorphize_tests.rs"]
mod monomorphize_tests;
