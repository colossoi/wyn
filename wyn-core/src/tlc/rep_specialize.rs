//! Representation specialization (Phase 2 of the array-variant-abstract
//! project).
//!
//! `ArrayVariantAbstract` is a representation-polymorphic variant introduced
//! for `filter`'s existential return. A call that passes a filter result to an
//! abstract-array parameter must target a definition whose parameter ABI uses
//! the producer's concrete representation.
//!
//! This pass analyzes producer-derived representation facts within each
//! definition, clones a callee for each required representation key, and
//! rewrites the corresponding call edges. Generated definitions and rewritten
//! call edges are the durable tree output. The template index, specialization
//! cache, and producer-fact maps are derived working state private to this run.

use super::data::Empty;
use super::monomorphize::{Monomorphic, Monomorphized};
use super::{
    clone_term_with_fresh_ids, curried_function_type, extract_lambda_params_ref, rebuild_nested_lam, Def,
    DefMeta, Program, RewriteDecision, SoacOp, Term, TermId, TermIdSource, TermKind, TermRewriter, VarRef,
};
use crate::ast::TypeName;
use crate::{LookupMap, SymbolId, SymbolTable};
use polytype::Type;

#[derive(Debug, Clone, Copy, Default)]
pub struct RepSpecialized;

impl super::Stage for RepSpecialized {
    type Family = Monomorphic;
    type GlobalContext = super::context::RewriteGlobal;
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

type SpecKey = Vec<Option<ConcreteVariant>>;
type ProducerVariants = LookupMap<SymbolId, ConcreteVariant>;

#[derive(Clone)]
struct Specialization {
    symbol: SymbolId,
    ty: Type<TypeName>,
}

struct RepSpecializer<'a> {
    symbols: &'a mut SymbolTable,
    term_ids: &'a mut TermIdSource,
    /// Original candidate definitions retained as specialization templates.
    templates: LookupMap<SymbolId, Def<Monomorphic>>,
    /// `(original callee, representation key) -> generated definition`.
    specializations: LookupMap<(SymbolId, SpecKey), Specialization>,
    /// Definitions generated during recursive call-edge rewriting.
    new_defs: Vec<Def<Monomorphic>>,
    /// Producer facts for the definition currently being rewritten.
    producer_variants: ProducerVariants,
}

/// Specialize representation-polymorphic call edges and consume the
/// monomorphic checkpoint into the representation-specialized checkpoint.
pub fn run(mut program: Program<Monomorphized>) -> Program<RepSpecialized> {
    // Only definitions with a specializable parameter need a retained
    // template. Ordinary definitions can be rewritten in place without
    // cloning their trees.
    let templates = program
        .defs
        .iter()
        .filter(|def| is_specialization_template(def))
        .map(|def| (def.name, def.clone()))
        .collect();

    let new_defs = {
        let mut specializer = RepSpecializer {
            symbols: &mut program.symbols,
            term_ids: &mut program.term_ids,
            templates,
            specializations: LookupMap::new(),
            new_defs: Vec::new(),
            producer_variants: LookupMap::new(),
        };
        for def in &mut program.defs {
            specializer.rewrite_definition_body(&mut def.body, LookupMap::new());
        }
        specializer.new_defs
    };

    program.defs.extend(new_defs);
    program.into_stage()
}

fn is_specialization_template(def: &Def<Monomorphic>) -> bool {
    if !matches!(def.meta, DefMeta::Function | DefMeta::LiftedLambda) {
        return false;
    }
    let (_, params) = extract_lambda_params_ref(&def.body);
    params.iter().any(|(_, ty)| type_has_specializable_array_variant(ty))
}

// =============================================================================
// Producer analysis
// =============================================================================

/// Derive the representation carried by every relevant binding in one
/// definition. `SymbolId`s are unique within the definition, so lexical scopes
/// do not require a stack: an out-of-scope binding cannot be referenced by a
/// different symbol.
fn analyze_producer_variants(
    term: &Term<Empty, Empty>,
    mut variants: ProducerVariants,
) -> ProducerVariants {
    fn analyze(term: &Term<Empty, Empty>, variants: &mut ProducerVariants) {
        if let TermKind::Let { name, rhs, body, .. } = &term.kind {
            analyze(rhs, variants);
            if let Some(variant) = detect_producer_variant(rhs, variants) {
                variants.insert(*name, variant);
            }
            analyze(body, variants);
            return;
        }
        term.for_each_child(&mut |child| analyze(child, variants));
    }

    analyze(term, &mut variants);
    variants
}

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

// =============================================================================
// Call-edge rewriting and definition generation
// =============================================================================

impl RepSpecializer<'_> {
    fn rewrite_definition_body(
        &mut self,
        body: &mut Term<Empty, Empty>,
        initial_variants: ProducerVariants,
    ) {
        let variants = analyze_producer_variants(body, initial_variants);
        let previous = std::mem::replace(&mut self.producer_variants, variants);
        self.rewrite_tracked(body);
        self.producer_variants = previous;
    }

    /// If `term` calls an abstract-array callee with a producer whose concrete
    /// representation is known, generate or reuse the matching specialization
    /// and repoint the function child.
    fn maybe_specialize_call(&mut self, term: &mut Term<Empty, Empty>) -> bool {
        let span = term.span;
        let TermKind::App { func, args } = &mut term.kind else {
            return false;
        };
        let TermKind::Var(VarRef::Symbol(callee)) = &func.kind else {
            return false;
        };
        let callee = *callee;

        let mut key: SpecKey = args
            .iter()
            .map(|argument| match &argument.kind {
                TermKind::Var(VarRef::Symbol(symbol)) => self.producer_variants.get(symbol).copied(),
                _ => None,
            })
            .collect();
        if key.iter().all(Option::is_none) {
            return false;
        }

        let Some(template) = self.templates.get(&callee) else {
            return false;
        };
        let (_, params) = extract_lambda_params_ref(&template.body);
        for (index, slot) in key.iter_mut().enumerate() {
            let Some((_, param_ty)) = params.get(index) else {
                *slot = None;
                continue;
            };
            if slot.is_some() && !type_has_specializable_array_variant(param_ty) {
                *slot = None;
            }
        }
        if key.iter().all(Option::is_none) {
            return false;
        }

        let specialization = self.get_or_create_specialization(callee, &key);
        **func = Term {
            id: self.term_ids.next_id(),
            ty: specialization.ty,
            span,
            kind: TermKind::Var(VarRef::Symbol(specialization.symbol)),
        };
        true
    }

    fn get_or_create_specialization(&mut self, callee: SymbolId, key: &SpecKey) -> Specialization {
        let cache_key = (callee, key.clone());
        if let Some(existing) = self.specializations.get(&cache_key) {
            return existing.clone();
        }

        let original_name = self.symbols.get(callee).cloned().unwrap_or_else(|| format!("def_{callee:?}"));
        let suffix: String = key
            .iter()
            .enumerate()
            .filter_map(|(index, variant)| variant.map(|variant| format!("_p{index}{}", variant.key_str())))
            .collect();
        let symbol = self.symbols.alloc(format!("{original_name}_rep{suffix}"));

        let (
            original_params,
            specialized_params,
            mut inner_body,
            body_span,
            arity,
            param_diets,
            return_diet,
        ) = {
            let template =
                self.templates.get(&callee).expect("representation specialization template disappeared");
            let (inner_template, original_params) = extract_lambda_params_ref(&template.body);
            let specialized_params = specialize_params(&original_params, key);
            let inner_body = clone_term_with_fresh_ids(inner_template, self.term_ids);
            (
                original_params,
                specialized_params,
                inner_body,
                template.body.span,
                template.arity,
                template.param_diets.clone(),
                template.return_diet.clone(),
            )
        };

        // Register the symbol and its ABI before rewriting the cloned body.
        // Recursive calls with the same key can therefore resolve the
        // in-progress specialization without cloning forever or falling back
        // to the original abstract function type.
        let ty = curried_function_type(specialized_params.iter().map(|(_, ty)| ty), &inner_body.ty);
        let specialization = Specialization { symbol, ty };
        self.specializations.insert(cache_key, specialization.clone());

        let initial_variants = original_params
            .iter()
            .enumerate()
            .filter_map(|(index, (symbol, _))| {
                key.get(index).copied().flatten().map(|variant| (*symbol, variant))
            })
            .collect();
        self.rewrite_definition_body(&mut inner_body, initial_variants);

        let body = rebuild_nested_lam(&specialized_params, inner_body, body_span, self.term_ids);
        debug_assert_eq!(body.ty, specialization.ty);
        self.new_defs.push(Def {
            data: (),
            name: symbol,
            ty: body.ty.clone(),
            body,
            meta: DefMeta::Function,
            arity,
            param_diets,
            return_diet,
        });
        specialization
    }
}

impl TermRewriter<Empty, Empty> for RepSpecializer<'_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        if self.maybe_specialize_call(term) {
            RewriteDecision::Changed
        } else {
            RewriteDecision::Unchanged
        }
    }
}

fn specialize_params(
    params: &[(SymbolId, Type<TypeName>)],
    key: &SpecKey,
) -> Vec<(SymbolId, Type<TypeName>)> {
    params
        .iter()
        .enumerate()
        .map(|(index, (symbol, ty))| {
            let ty = match key.get(index).copied().flatten() {
                Some(variant) => substitute_specializable_variant_in_type(ty, variant),
                None => ty.clone(),
            };
            (*symbol, ty)
        })
        .collect()
}

// =============================================================================
// Type-level representation operations
// =============================================================================

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
