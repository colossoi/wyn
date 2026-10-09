//! Specialize reachable TLC definitions by type and array representation.
//!
//! One cache and worklist own all instantiations. Producer facts are local to
//! the definition being rewritten and follow let bindings, including those
//! introduced while instantiating local polymorphic lambdas.

use super::data::{Empty, PolymorphicDefinition};
use super::pin_entry_buffers::BuffersPinned;
use super::{
    apply_type_substitution, curried_function_type, extend_type_substitution, ArrayExpr, Def, DefMeta,
    Program, RewriteDecision, SoacOp, Term, TermId, TermIdSource, TermKind, TermRewriter, TypeSubstitution,
    VarRef,
};
use crate::ast::TypeName;
use crate::error::CompilerError;
use crate::types::{TypeExt, TypeScheme};
use crate::{LookupMap, SymbolId, SymbolTable};
use polytype::Type;
use std::collections::VecDeque;
use std::fmt::Debug;

type Specializable<E> = super::TreeFamily<PolymorphicDefinition, E, Empty, Empty>;

/// Monomorphic TLC stores no per-definition payload; specialization consumes
/// the source schemes while constructing this family.
pub type Monomorphic = super::TreeFamily<(), super::data::PinnedEntry, Empty, Empty>;

/// TLC after intrinsic, type, and producer-derived representation specialization.
#[derive(Debug, Clone, Copy)]
pub enum MonomorphizedTag {}
pub type Monomorphized = super::Program<MonomorphizedTag, Monomorphic, super::context::RewriteGlobal>;

/// Type specialization leaves descriptor regions unresolved until final stages exist.
#[derive(Debug, Clone, Copy)]
pub enum TypesSpecializedTag {}
pub type TypesSpecialized =
    super::Program<TypesSpecializedTag, super::run::UnpinnedPolymorphic, super::context::TransformedGlobal>;

/// Specialize unified roots without erasing regions that final stages will bind.
pub fn specialize_types(
    mut program: super::stage::PartialEvaled,
) -> Result<TypesSpecialized, CompilerError> {
    super::specialize::specialize_intrinsics(&mut program);
    program.defs = Monomorphizer::new(&mut program.symbols, program.defs, &mut program.term_ids, false)
        .monomorphize()?;
    program.assert_flat_apps();
    Ok(program.retag())
}

/// Finish representation and buffer specialization after physical interfaces are known.
pub fn monomorphize(mut program: BuffersPinned) -> std::result::Result<Monomorphized, CompilerError> {
    super::specialize::specialize_intrinsics(&mut program);
    let Program {
        defs,
        mut symbols,
        mut term_ids,
        global_context,
        state: _,
    } = program;
    let defs = Monomorphizer::new(&mut symbols, defs, &mut term_ids, true).monomorphize()?;
    let defs = defs
        .into_iter()
        .map(|def| Def {
            data: (),
            name: def.name,
            package: def.package,
            ty: def.ty,
            body: def.body,
            meta: def.meta,
            arity: def.arity,
            param_diets: def.param_diets,
            return_diet: def.return_diet,
        })
        .collect();
    let program = Program::from_parts(defs, symbols, term_ids, global_context);
    program.assert_flat_apps();
    Ok(program)
}

struct Monomorphizer<'symbols, 'ids, E: Clone + Debug> {
    regions_pinned: bool,
    symbols: &'symbols mut SymbolTable,
    definitions: LookupMap<SymbolId, DefinitionRecord<E>>,
    specializations: LookupMap<(SymbolId, SpecKey), Specialization>,
    worklist: VecDeque<WorkItem>,
    producer_variants: ProducerVariants,
    term_ids: &'ids mut TermIdSource,
}

struct DefinitionRecord<E: Clone + Debug> {
    info: DefinitionInfo,
    /// Ordinary definitions are moved into the output; specializable ones
    /// retain their template for further instantiations.
    template: Option<Def<Specializable<E>>>,
}

impl<E: Clone + Debug> DefinitionRecord<E> {
    fn new(definition: Def<Specializable<E>>) -> Self {
        let info = DefinitionInfo {
            scheme: definition.data.scheme.clone(),
            ty: definition.ty.clone(),
            specialize_representations: matches!(
                definition.meta,
                DefMeta::Function | DefMeta::LiftedLambda
            ),
        };
        Self {
            info,
            template: Some(definition),
        }
    }

    fn materialize(&mut self, retain_template: bool) -> Option<Def<Specializable<E>>> {
        if retain_template {
            self.template.clone()
        } else {
            self.template.take()
        }
    }
}

#[derive(Clone)]
struct DefinitionInfo {
    scheme: Option<TypeScheme>,
    ty: Type<TypeName>,
    specialize_representations: bool,
}

impl DefinitionInfo {
    fn polymorphic_type(&self) -> &Type<TypeName> {
        self.scheme.as_ref().map(unwrap_scheme).unwrap_or(&self.ty)
    }

    fn may_need_as_template(&self) -> bool {
        matches!(&self.scheme, Some(TypeScheme::Polytype { .. }))
            || !self.ty.vars().is_empty()
            || (self.specialize_representations && type_has_specializable_array_variant(&self.ty))
    }
}

#[derive(Clone)]
struct Specialization {
    symbol: SymbolId,
    ty: Type<TypeName>,
}

struct WorkItem {
    original_sym: SymbolId,
    spec_key: SpecKey,
    output: Specialization,
}

/// Deterministic, hashable form of a type substitution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SubstKey(Vec<(usize, Type<TypeName>)>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SpecKey {
    type_subst: SubstKey,
    /// Trailing empty slots are omitted, so an all-empty key is canonical.
    representations: Vec<Option<ConcreteVariant>>,
}

impl SubstKey {
    fn from_subst(subst: &TypeSubstitution) -> Self {
        let mut items: Vec<_> = subst.iter().map(|(variable, ty)| (*variable, ty.clone())).collect();
        items.sort_by_key(|(variable, _)| *variable);
        Self(items)
    }

    fn to_subst(&self) -> TypeSubstitution {
        self.0.iter().cloned().collect()
    }
}

impl SpecKey {
    fn empty() -> Self {
        Self::new(&TypeSubstitution::new())
    }

    fn new(subst: &TypeSubstitution) -> Self {
        Self {
            type_subst: SubstKey::from_subst(subst),
            representations: Vec::new(),
        }
    }

    fn needs_specialization(&self) -> bool {
        !self.type_subst.0.is_empty() || !self.representations.is_empty()
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

impl<'symbols, 'ids, E: Clone + Debug> Monomorphizer<'symbols, 'ids, E> {
    fn new(
        symbols: &'symbols mut SymbolTable,
        defs: Vec<Def<Specializable<E>>>,
        term_ids: &'ids mut TermIdSource,
        regions_pinned: bool,
    ) -> Self {
        let entries: Vec<_> = defs
            .iter()
            .filter(|def| matches!(def.meta, DefMeta::EntryPoint(_)))
            .map(|def| def.name)
            .collect();
        let mut this = Self {
            regions_pinned,
            symbols,
            definitions: defs.into_iter().map(|def| (def.name, DefinitionRecord::new(def))).collect(),
            specializations: LookupMap::new(),
            worklist: VecDeque::new(),
            producer_variants: LookupMap::new(),
            term_ids,
        };
        for entry in entries {
            this.get_or_create_specialization(entry, SpecKey::empty());
        }
        this
    }

    fn monomorphize(mut self) -> std::result::Result<Vec<Def<Specializable<E>>>, CompilerError> {
        let mut defs = Vec::new();
        while let Some(work_item) = self.worklist.pop_front() {
            self.producer_variants.clear();
            let def = self.materialize_work_item(&work_item)?;
            defs.push(self.process_def(def));
        }
        Ok(defs)
    }

    fn materialize_work_item(
        &mut self,
        work_item: &WorkItem,
    ) -> std::result::Result<Def<Specializable<E>>, CompilerError> {
        let Some(definition) = self.definitions.get_mut(&work_item.original_sym) else {
            return Err(CompilerError::Internal(format!(
                "monomorphization work item refers to missing definition {:?}",
                work_item.original_sym
            )));
        };
        let retain_template =
            work_item.spec_key.needs_specialization() || definition.info.may_need_as_template();
        let Some(mut def) = definition.materialize(retain_template) else {
            return Err(CompilerError::Internal(format!(
                "monomorphic definition {:?} was queued after its body was consumed",
                work_item.original_sym
            )));
        };
        def.name = work_item.output.symbol;
        def.ty = work_item.output.ty.clone();
        if work_item.spec_key.needs_specialization() {
            let subst = work_item.spec_key.type_subst.to_subst();
            def.body.rewrite_types(self.term_ids, &mut |ty| apply_type_substitution(ty, &subst));
        }
        if !work_item.spec_key.representations.is_empty() {
            specialize_lambda_params(
                &mut def.body,
                &work_item.spec_key.representations,
                &mut 0,
                &mut self.producer_variants,
                self.term_ids,
            );
        }
        Ok(def)
    }

    fn process_def(&mut self, mut def: Def<Specializable<E>>) -> Def<Specializable<E>> {
        def.data.scheme = None;
        def.body = def.body.rewrite(self);
        def
    }

    fn rewrite_symbol_reference(
        &mut self,
        symbol: SymbolId,
        concrete_type: &Type<TypeName>,
    ) -> Option<SymbolId> {
        let info = &self.definitions.get(&symbol)?.info;
        let mut subst = TypeSubstitution::new();
        extend_type_substitution(info.polymorphic_type(), concrete_type, &mut subst);
        if self.regions_pinned {
            normalize_buffer_substitutions(info.polymorphic_type(), &mut subst);
        }
        let specialized = self.get_or_create_specialization(symbol, SpecKey::new(&subst));
        (specialized.symbol != symbol).then_some(specialized.symbol)
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

    fn infer_call_key(
        &self,
        info: &DefinitionInfo,
        args: &[Term<Empty, Empty>],
        callee_ty: &Type<TypeName>,
    ) -> SpecKey {
        let mut subst = TypeSubstitution::new();
        // The instantiated callee includes result-only variables and opened existential inputs.
        extend_type_substitution(info.polymorphic_type(), callee_ty, &mut subst);
        let (params, _) = split_function_type(info.polymorphic_type());
        for (param, arg) in params.iter().zip(args) {
            extend_type_substitution(param, &arg.ty, &mut subst);
        }
        if self.regions_pinned {
            normalize_buffer_substitutions(info.polymorphic_type(), &mut subst);
        }
        let mut key = SpecKey::new(&subst);
        if info.specialize_representations {
            key.representations = params
                .iter()
                .zip(args)
                .map(|(param, arg)| {
                    if type_has_specializable_array_variant(&apply_type_substitution(param, &subst)) {
                        if let TermKind::Var(VarRef::Symbol(symbol)) = &arg.kind {
                            return self.producer_variants.get(symbol).copied();
                        }
                    }
                    None
                })
                .collect();
            while key.representations.last() == Some(&None) {
                key.representations.pop();
            }
        }
        key
    }

    fn get_or_create_specialization(&mut self, function: SymbolId, spec_key: SpecKey) -> Specialization {
        let cache_key = (function, spec_key.clone());
        if let Some(specialized) = self.specializations.get(&cache_key) {
            return specialized.clone();
        }
        let subst = spec_key.type_subst.to_subst();
        let symbol = if spec_key.needs_specialization() {
            let function_name = self.symbols.get(function).expect("BUG: function symbol is missing");
            let suffix = format_subst(&subst);
            let rep_suffix: String = spec_key
                .representations
                .iter()
                .enumerate()
                .filter_map(|(index, variant)| {
                    variant.map(|variant| format!("_p{index}{}", variant.key_str()))
                })
                .collect();
            self.symbols.alloc(format!("{function_name}${suffix}{rep_suffix}"))
        } else {
            function
        };
        let mut ty = apply_type_substitution(&self.definitions[&function].info.ty, &subst);
        if !spec_key.representations.is_empty() {
            let (mut params, result) = split_function_type(&ty);
            for (param, variant) in params.iter_mut().zip(&spec_key.representations) {
                if let Some(variant) = variant {
                    *param = substitute_specializable_variant_in_type(param, *variant);
                }
            }
            ty = curried_function_type(params.iter(), &result);
        }
        let specialized = Specialization { symbol, ty };
        // Register both the symbol and ABI before its body can request a recursive instance.
        self.specializations.insert(cache_key, specialized.clone());
        self.worklist.push_back(WorkItem {
            original_sym: function,
            spec_key,
            output: specialized.clone(),
        });
        specialized
    }
}

impl<E: Clone + Debug> TermRewriter<Empty, Empty> for Monomorphizer<'_, '_, E> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node_before_children(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        if let TermKind::Let { name, rhs, body, .. } = &term.kind {
            if matches!(rhs.kind, TermKind::Lambda(_)) && !rhs.ty.vars().is_empty() {
                let body = super::subst::substitute_with(
                    (**body).clone(),
                    *name,
                    &mut |occurrence, ids| {
                        let mut subst = TypeSubstitution::new();
                        extend_type_substitution(&rhs.ty, &occurrence.ty, &mut subst);
                        let mut instance = super::clone_term_with_fresh_ids(rhs, ids);
                        instance.rewrite_types(ids, &mut |ty| apply_type_substitution(ty, &subst));
                        instance
                    },
                    self.term_ids,
                );
                *term = body;
                self.rewrite_node_before_children(term);
                return RewriteDecision::Changed;
            }
            // Let bodies are visited after their RHS; unique symbols let facts
            // propagate through aliases without a separate analysis or scope stack.
            if let Some(variant) = detect_producer_variant(rhs, &self.producer_variants) {
                self.producer_variants.insert(*name, variant);
            }
        }
        let TermKind::App { func, args } = &mut term.kind else {
            return RewriteDecision::Unchanged;
        };
        let TermKind::Var(VarRef::Symbol(symbol)) = &func.kind else {
            return RewriteDecision::Unchanged;
        };
        let symbol = *symbol;
        let Some(definition) = self.definitions.get(&symbol) else {
            return RewriteDecision::Unchanged;
        };
        let key = self.infer_call_key(&definition.info, args, &func.ty);
        let specialize_abi = !key.representations.is_empty();
        let specialized = self.get_or_create_specialization(symbol, key);
        if specialized.symbol == symbol {
            return RewriteDecision::Unchanged;
        }
        func.kind = TermKind::Var(VarRef::Symbol(specialized.symbol));
        if specialize_abi {
            func.ty = specialized.ty;
        }
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

/// Change only leading lambda parameter ABIs and carry their producer facts
/// into the body. Logical result sizes and unrelated arrays stay unchanged.
fn specialize_lambda_params(
    term: &mut Term<Empty, Empty>,
    representations: &[Option<ConcreteVariant>],
    index: &mut usize,
    variants: &mut ProducerVariants,
    term_ids: &mut TermIdSource,
) {
    if let TermKind::Lambda(lambda) = &mut term.kind {
        for (symbol, ty) in &mut lambda.params {
            if let Some(variant) = representations.get(*index).copied().flatten() {
                *ty = substitute_specializable_variant_in_type(ty, variant);
                variants.insert(*symbol, variant);
            }
            *index += 1;
        }
        specialize_lambda_params(&mut lambda.body, representations, index, variants, term_ids);
        lambda.ret_ty = lambda.body.ty.clone();
        term.ty = curried_function_type(lambda.params.iter().map(|(_, ty)| ty), &lambda.ret_ty);
        term.id = term_ids.next_id();
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
        Type::Constructed(TypeName::Array, args) => {
            let Some(tensor) = ty.as_tensor() else {
                let args = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
                return format!("Array_{args}");
            };
            let Some(storage) = ty.array_storage() else {
                return format!(
                    "Array_{}",
                    args.iter().map(format_type_compact).collect::<Vec<_>>().join("_")
                );
            };
            let dims = tensor.dims.iter().map(format_type_compact).collect::<Vec<_>>().join("x");
            format!(
                "arr{}_{}{}",
                format_type_compact(tensor.elem),
                dims,
                format_type_compact(storage.variant)
            )
        }
        Type::Constructed(TypeName::Tuple(arity), args) => {
            let args = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("tup{arity}_{args}")
        }
        Type::Constructed(TypeName::Vec, args) => {
            let Some(tensor) = ty.as_tensor() else {
                let args = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
                return format!("Vec_{args}");
            };
            let Some(size) = tensor.dim(0) else {
                return format!(
                    "Vec_{}",
                    args.iter().map(format_type_compact).collect::<Vec<_>>().join("_")
                );
            };
            let elem = format_type_compact(tensor.elem);
            let size = format_type_compact(size);
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

/// An unresolved buffer identity is not a distinct implementation. Otherwise
/// a chain of calls creates one specialization per fresh result-buffer variable,
/// even though their element types, extents and representations are identical.
/// Concrete storage bindings remain distinct; unresolved buffer variables
/// alone do not distinguish implementations.
fn normalize_buffer_substitutions(ty: &Type<TypeName>, subst: &mut TypeSubstitution) {
    if let Type::Constructed(name, fields) = ty {
        if *name == TypeName::Array {
            if let Some(Type::Variable(id)) = fields.last() {
                if matches!(subst.get(id), Some(Type::Variable(_))) {
                    subst.insert(*id, Type::Constructed(TypeName::NoBuffer, vec![]));
                }
            }
        }
        for field in fields {
            normalize_buffer_substitutions(field, subst);
        }
    }
}

/// Concrete array representation selected from a known producer.
///
/// `Bounded` carries the producer's static capacity because its consumer ABI
/// must also expose that capacity. The other variants leave the array's size
/// slot unchanged.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[allow(dead_code)]
enum ConcreteVariant {
    Bounded {
        capacity: usize,
    },
    View,
    Composite,
    Virtual,
}

impl ConcreteVariant {
    fn variant_type(self) -> Type<TypeName> {
        let name = match self {
            Self::Bounded { .. } => TypeName::ArrayVariantBounded,
            Self::View => TypeName::ArrayVariantView,
            Self::Composite => TypeName::ArrayVariantComposite,
            Self::Virtual => TypeName::ArrayVariantVirtual,
        };
        Type::Constructed(name, vec![])
    }

    fn size_type(self) -> Option<Type<TypeName>> {
        match self {
            Self::Bounded { capacity } => Some(Type::Constructed(TypeName::Size(capacity), vec![])),
            _ => None,
        }
    }

    fn key_str(self) -> String {
        match self {
            Self::Bounded { capacity } => format!("bounded{capacity}"),
            Self::View => "view".to_string(),
            Self::Composite => "composite".to_string(),
            Self::Virtual => "virtual".to_string(),
        }
    }
}

type ProducerVariants = LookupMap<SymbolId, ConcreteVariant>;

/// Recognize let-bound producers whose representation follows from the tree.
///
/// A filter over a statically-sized input produces `Bounded`; otherwise it
/// produces `View`. Simple aliases propagate the source binding's fact.
fn detect_producer_variant(
    rhs: &Term<Empty, Empty>,
    variants: &ProducerVariants,
) -> Option<ConcreteVariant> {
    match &rhs.kind {
        TermKind::Soac(SoacOp::Filter { input, .. }) => {
            let input_ty = input.array_type();
            let size = array_size(&input_ty)?;
            Some(match size {
                Type::Constructed(TypeName::Size(capacity), _) => {
                    ConcreteVariant::Bounded { capacity: *capacity }
                }
                _ => ConcreteVariant::View,
            })
        }
        // `open_existential` introduces an alias after the filter-producing
        // ANF binding. Carry the representation through that alias.
        TermKind::Var(VarRef::Symbol(source)) => variants.get(source).copied(),
        _ => None,
    }
}

fn array_size(ty: &Type<TypeName>) -> Option<&Type<TypeName>> {
    if let Type::Constructed(TypeName::Array, args) = ty {
        return args.get(2);
    }
    None
}

/// Return whether any representation-polymorphic array variant appears in the
/// type. `ArrayVariantAbstract` comes from a filter existential; a variable in
/// the variant slot comes from an ordinary representation-polymorphic helper.
fn type_has_specializable_array_variant(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => false,
        Type::Constructed(TypeName::Array, args) if args.len() >= 4 => {
            matches!(
                &args[1],
                Type::Constructed(TypeName::ArrayVariantAbstract, _) | Type::Variable(_)
            ) || args.iter().any(type_has_specializable_array_variant)
        }
        Type::Constructed(_, args) => args.iter().any(type_has_specializable_array_variant),
    }
}

/// Replace representation-polymorphic array variants with `target`.
///
/// A bounded representation also supplies a static capacity when the existing
/// size slot is non-literal.
fn substitute_specializable_variant_in_type(
    ty: &Type<TypeName>,
    target: ConcreteVariant,
) -> Type<TypeName> {
    match ty {
        Type::Variable(_) => ty.clone(),
        Type::Constructed(TypeName::Array, args) if args.len() >= 4 => {
            let mut new_args: Vec<Type<TypeName>> =
                args.iter().map(|arg| substitute_specializable_variant_in_type(arg, target)).collect();
            if matches!(
                &args[1],
                Type::Constructed(TypeName::ArrayVariantAbstract, _) | Type::Variable(_)
            ) {
                new_args[1] = target.variant_type();
                if let Some(size_ty) = target.size_type() {
                    if !matches!(&new_args[2], Type::Constructed(TypeName::Size(_), _)) {
                        new_args[2] = size_ty;
                    }
                }
            }
            Type::Constructed(TypeName::Array, new_args)
        }
        Type::Constructed(name, args) => Type::Constructed(
            name.clone(),
            args.iter().map(|arg| substitute_specializable_variant_in_type(arg, target)).collect(),
        ),
    }
}
