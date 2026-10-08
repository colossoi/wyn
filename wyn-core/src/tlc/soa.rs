//! Structure-of-Arrays (SoA) lowering for TLC.
//!
//! This pass normalizes the concrete shapes exposed by specialization and inlining:
//!
//! 1. **SoA transform**: Rewrites `[n](A,B)` (array of tuples) into `([n]A, [n]B)`
//!    (tuple of arrays) for local arrays. Operations on array-of-tuple types
//!    (index, array_with, array_lit, uninit, length) operate on the distributed components.
//!
//! 2. Converts standalone `zip(a, b)` into tuple construction. Map inputs and
//!    callback parameters already correspond when the Map is constructed.
//!
//! Runs once, after monomorphization and helper inlining, before fusion and
//! defunctionalization. Storage-backed arrays retain their declared layout.

use super::bindings::flatten_nested_let;
use super::data::Empty;
use super::if_over_producer::ConditionalProducersCanonicalized;
use super::{
    clone_term_with_fresh_ids, ArrayExpr, Bindings, RewriteDecision, Term, TermId, TermIdSource, TermKind,
    TermRewriter, VarRef,
};
use crate::ast::{Span, TypeName};
use crate::builtins::catalog;
use crate::tlc;
use crate::types::TypeExt;
use crate::SymbolTable;
use polytype::Type;

/// Monomorphic TLC after SoA lowering and SOAC binding normalization.
#[derive(Debug, Clone, Copy)]
pub enum SoacsAnfNormalizedTag {}
pub type SoacsAnfNormalized =
    super::Program<SoacsAnfNormalizedTag, super::monomorphize::Monomorphic, super::context::RewriteGlobal>;

// =============================================================================
// Type rewriting
// =============================================================================

/// Recursively rewrite types so that `Array[Tuple(n)(T1..Tn), v, s]`
/// becomes `Tuple(n)(Array[soa(T1), v, s], …, Array[soa(Tn), v, s])`.
///
/// Other types are rewritten recursively but their top-level structure is preserved.
pub fn soa_type(ty: &Type<TypeName>) -> Type<TypeName> {
    match ty {
        // Entry-backed arrays retain their declared AoS representation. A
        // single storage resource cannot be rewritten into a tuple of arrays
        // without also changing its ABI into multiple buffers. Generated and
        // function-local composite arrays remain eligible for SoA.
        _ if is_storage_backed_array(ty) => ty.clone(),
        _ if ty.is_array() => {
            let Some(tensor) = ty.as_tensor() else {
                return ty.clone();
            };
            let Some(storage) = ty.array_storage() else {
                return ty.clone();
            };
            let elem = soa_type(tensor.elem);
            let dims = tensor.dims.to_vec();
            let variant = match storage.variant {
                // Resolve unresolved variant variables to Composite when distributing.
                Type::Variable(_) => Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
                v => v.clone(),
            };

            // If the element type is a tuple, distribute the array into each component.
            // Recursively apply soa_type to each distributed array, so nested
            // array-of-tuples like [n](int, vec3) get further distributed.
            if let Type::Constructed(TypeName::Tuple(n), ref component_types) = elem {
                let distributed: Vec<Type<TypeName>> = component_types
                    .iter()
                    .map(|ct| {
                        soa_type(&array_type(
                            ct.clone(),
                            variant.clone(),
                            &dims,
                            storage.region.clone(),
                        ))
                    })
                    .collect();
                Type::Constructed(TypeName::Tuple(n), distributed)
            } else {
                array_type(elem, variant, &dims, storage.region.clone())
            }
        }
        Type::Constructed(name, args) => {
            // For other constructed types (Vec, Mat, scalars, etc.), recurse into args
            let rewritten: Vec<Type<TypeName>> = args.iter().map(soa_type).collect();
            Type::Constructed(name.clone(), rewritten)
        }
        Type::Variable(_) => ty.clone(),
    }
}

fn array_type(
    elem: Type<TypeName>,
    variant: Type<TypeName>,
    dims: &[Type<TypeName>],
    region: Type<TypeName>,
) -> Type<TypeName> {
    let mut args = Vec::with_capacity(dims.len() + 3);
    args.push(elem);
    args.push(variant);
    args.extend_from_slice(dims);
    args.push(region);
    Type::Constructed(TypeName::Array, args)
}

fn is_storage_backed_array(ty: &Type<TypeName>) -> bool {
    matches!(ty.array_buffer(), Some(Type::Constructed(TypeName::Buffer(_), _)))
}

/// Rebuild operations from already-lowered children. Generated terms carry
/// their final types and are never fed back through the source transformation.
struct SoaTransformer<'a, 'ids> {
    term_ids: &'ids mut TermIdSource,
    symbols: &'a mut SymbolTable,
}

impl SoaTransformer<'_, '_> {
    fn structural_replacement(&mut self, term: &Term) -> Option<Term> {
        let span = term.span;
        let known = catalog().known();
        match &term.kind {
            TermKind::Index { array, index }
                if matches!(array.ty, Type::Constructed(TypeName::Tuple(_), _)) =>
            {
                let mut bindings = Bindings::new();
                let array = bindings.name((**array).clone(), "_soa_array", self.symbols, self.term_ids);
                let index = bindings.name((**index).clone(), "_soa_index", self.symbols, self.term_ids);
                let result = self.distribute_index(&array, &index, &term.ty, span);
                Some(bindings.finish(result, self.term_ids))
            }
            TermKind::App { func, args } => {
                let id = tlc::var_term_builtin_id(func, self.symbols)?;
                if (id == known.array_with || id == known.array_with_in_place)
                    && args.len() == 3
                    && matches!(args[0].ty, Type::Constructed(TypeName::Tuple(_), _))
                {
                    let mut bindings = Bindings::new();
                    let array = bindings.name(args[0].clone(), "_soa_array", self.symbols, self.term_ids);
                    let index = bindings.name(args[1].clone(), "_soa_index", self.symbols, self.term_ids);
                    let value = bindings.name(args[2].clone(), "_soa_value", self.symbols, self.term_ids);
                    let result = self.distribute_update(&array, &index, &value, span);
                    return Some(bindings.finish(result, self.term_ids));
                }
                if id == known.length
                    && args.len() == 1
                    && matches!(args[0].ty, Type::Constructed(TypeName::Tuple(_), _))
                {
                    let mut bindings = Bindings::new();
                    let mut array =
                        bindings.name(args[0].clone(), "_soa_array", self.symbols, self.term_ids);
                    while matches!(array.ty, Type::Constructed(TypeName::Tuple(_), _)) {
                        array = self.project(&array, 0, span);
                    }
                    let result = self.builtin_call(known.length, vec![array], term.ty.clone(), span);
                    return Some(bindings.finish(result, self.term_ids));
                }
                None
            }
            TermKind::Var(VarRef::Builtin { id, .. })
                if *id == known.uninit && matches!(term.ty, Type::Constructed(TypeName::Tuple(_), _)) =>
            {
                Some(self.distribute_uninit(&term.ty, span))
            }
            TermKind::ArrayExpr(array @ ArrayExpr::Zip(_)) => Some(self.array_value(array, &term.ty, span)),
            TermKind::ArrayExpr(ArrayExpr::Literal(elements))
                if matches!(term.ty, Type::Constructed(TypeName::Tuple(_), _)) =>
            {
                Some(self.literal_value(elements, &term.ty, span))
            }
            _ => None,
        }
    }

    /// Distribute through the target layout, including nested tuple components.
    /// The operands are references, so distribution never repeats a computation.
    fn distribute_index(
        &mut self,
        array: &Term,
        index: &Term,
        result_ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        if let Type::Constructed(TypeName::Tuple(_), fields) = result_ty {
            // A storage-backed array of tuples remains an array and is indexed
            // directly, even when its element/result type is a tuple.
            if matches!(array.ty, Type::Constructed(TypeName::Tuple(_), _)) {
                let fields = fields
                    .iter()
                    .enumerate()
                    .map(|(i, field_ty)| {
                        let component = self.project(array, i, span);
                        self.distribute_index(&component, index, field_ty, span)
                    })
                    .collect();
                return self.tuple(fields, span);
            }
        }
        let array = clone_term_with_fresh_ids(array, self.term_ids);
        let index = clone_term_with_fresh_ids(index, self.term_ids);
        self.term(
            result_ty.clone(),
            span,
            TermKind::Index {
                array: Box::new(array),
                index: Box::new(index),
            },
        )
    }

    fn distribute_update(&mut self, array: &Term, index: &Term, value: &Term, span: Span) -> Term {
        if let Type::Constructed(TypeName::Tuple(n), _) = &array.ty {
            let fields = (0..*n)
                .map(|i| {
                    let array = self.project(array, i, span);
                    let value = self.project(value, i, span);
                    self.distribute_update(&array, index, &value, span)
                })
                .collect();
            return self.tuple(fields, span);
        }
        let args = [array, index, value]
            .into_iter()
            .map(|term| clone_term_with_fresh_ids(term, self.term_ids))
            .collect();
        self.builtin_call(catalog().known().array_with, args, array.ty.clone(), span)
    }

    fn literal_value(&mut self, elements: &[Term], result_ty: &Type<TypeName>, span: Span) -> Term {
        let mut bindings = Bindings::new();
        let elements: Vec<_> = elements
            .iter()
            .map(|element| bindings.name(element.clone(), "_soa_element", self.symbols, self.term_ids))
            .collect();
        let result = self.distribute_literal(&elements, result_ty, span);
        bindings.finish(result, self.term_ids)
    }

    fn distribute_literal(&mut self, elements: &[Term], result_ty: &Type<TypeName>, span: Span) -> Term {
        if let Type::Constructed(TypeName::Tuple(_), fields) = result_ty {
            let fields = fields
                .iter()
                .enumerate()
                .map(|(i, field_ty)| {
                    let projected: Vec<_> =
                        elements.iter().map(|element| self.project(element, i, span)).collect();
                    self.distribute_literal(&projected, field_ty, span)
                })
                .collect();
            return self.tuple(fields, span);
        }
        let elements =
            elements.iter().map(|element| clone_term_with_fresh_ids(element, self.term_ids)).collect();
        self.term(
            result_ty.clone(),
            span,
            TermKind::ArrayExpr(ArrayExpr::Literal(elements)),
        )
    }

    fn distribute_uninit(&mut self, ty: &Type<TypeName>, span: Span) -> Term {
        if let Type::Constructed(TypeName::Tuple(_), fields) = ty {
            let fields = fields.iter().map(|ty| self.distribute_uninit(ty, span)).collect();
            self.tuple(fields, span)
        } else {
            self.builtin_call(catalog().known().uninit, vec![], ty.clone(), span)
        }
    }

    /// Turn a standalone array atom into a value with its own component type.
    /// Its child terms and named-atom types have already been lowered.
    fn array_value(&mut self, array: &ArrayExpr, ty: &Type<TypeName>, span: Span) -> Term {
        match array {
            ArrayExpr::Var(reference, ty) => self.term(ty.clone(), span, TermKind::Var(*reference)),
            ArrayExpr::Zip(inputs) => {
                let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
                    panic!("lowered zip type")
                };
                assert_eq!(inputs.len(), fields.len(), "zip layout arity");
                let values = inputs
                    .iter()
                    .zip(fields)
                    .map(|(input, ty)| self.array_value(input, ty, span))
                    .collect();
                self.tuple(values, span)
            }
            ArrayExpr::Literal(elements) if matches!(ty, Type::Constructed(TypeName::Tuple(_), _)) => {
                self.literal_value(elements, ty, span)
            }
            _ => self.term(ty.clone(), span, TermKind::ArrayExpr(array.clone())),
        }
    }

    fn term(&mut self, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
        Term::fresh(self.term_ids, ty, span, kind)
    }

    fn tuple(&mut self, fields: Vec<Term>, span: Span) -> Term {
        let ty = Type::Constructed(
            TypeName::Tuple(fields.len()),
            fields.iter().map(|field| field.ty.clone()).collect(),
        );
        self.term(ty, span, TermKind::Tuple(fields))
    }

    fn project(&mut self, tuple: &Term, idx: usize, span: Span) -> Term {
        let Type::Constructed(TypeName::Tuple(_), fields) = &tuple.ty else {
            panic!("tuple projection input")
        };
        let ty = fields[idx].clone();
        let tuple = Box::new(clone_term_with_fresh_ids(tuple, self.term_ids));
        self.term(ty, span, TermKind::TupleProj { tuple, idx })
    }

    fn builtin_call(
        &mut self,
        id: crate::builtins::BuiltinId,
        args: Vec<Term>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let ty = tlc::curried_function_type(args.iter().map(|arg| &arg.ty), &result_ty);
        let func = self.term(ty, span, TermKind::Var(VarRef::Builtin { id, overload_idx: 0 }));
        if args.is_empty() {
            return func;
        }
        self.term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                args,
            },
        )
    }
}

impl TermRewriter<Empty, Empty> for SoaTransformer<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node_before_children(&mut self, term: &mut Term) -> RewriteDecision {
        // A nullary uninit denotes the value itself. Consume its call edge
        // before lowering so distribution happens at the value position,
        // rather than turning the callee into a tuple of component values.
        if let TermKind::App { func, args } = &term.kind {
            if args.is_empty()
                && tlc::var_term_builtin_id(func, self.symbols) == Some(catalog().known().uninit)
            {
                term.kind = func.kind.clone();
                return RewriteDecision::Changed;
            }
        }
        RewriteDecision::Unchanged
    }

    fn rewrite_owned_node(&mut self, mut term: Term) -> (Term, RewriteDecision) {
        // Child expressions are already in the target representation. Rewrite
        // this node's annotations and construct its target operation together.
        term.rewrite_node_types(&mut soa_type);
        if let Some(replacement) = self.structural_replacement(&term) {
            term = replacement;
        }
        if let TermKind::App { args, .. } = &mut term.kind {
            if args.iter().any(|arg| matches!(arg.kind, TermKind::Soac(_))) {
                let mut bindings = Bindings::new();
                crate::map_in_place(args, |arg| {
                    if matches!(arg.kind, TermKind::Soac(_)) {
                        bindings.name(arg, "_anf", self.symbols, self.term_ids)
                    } else {
                        arg
                    }
                });
                // The application becomes a child of the new lets, so the
                // walker's root-ID refresh no longer covers it.
                term.id = self.next_term_id();
                term = bindings.finish(term, self.term_ids);
            }
        }
        let (term, _) = flatten_nested_let(term, self.term_ids);
        (term, RewriteDecision::Changed)
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Normalize concrete array shapes after specialization and helper expansion.
///
/// 1. Rewrites `[n](A,B)` types to `([n]A, [n]B)` and adjusts all operations
///    that touch array-of-tuple types.
/// 2. Converts standalone Zip to tuple construction.
/// 3. Names SOAC application arguments and flattens nested let RHSs.
pub fn normalize_soacs(mut program: ConditionalProducersCanonicalized) -> SoacsAnfNormalized {
    let mut transformer = SoaTransformer {
        term_ids: &mut program.term_ids,
        symbols: &mut program.symbols,
    };
    crate::map_in_place(&mut program.defs, |mut def| {
        def.ty = soa_type(&def.ty);
        def.body = def.body.rewrite_owned(&mut transformer);
        def
    });
    let program = program.retag();
    debug_assert!(
        verify_flattened(&program).is_ok(),
        "SOAC normalization left a nested let rhs"
    );
    program
}

fn verify_flattened(program: &SoacsAnfNormalized) -> Result<(), ()> {
    fn walk(term: &Term) -> Result<(), ()> {
        if matches!(&term.kind, TermKind::Let { rhs, .. } if matches!(rhs.kind, TermKind::Let { .. })) {
            return Err(());
        }
        let mut result = Ok(());
        term.for_each_child(&mut |child| {
            if result.is_ok() {
                result = walk(child);
            }
        });
        result
    }
    program.defs.iter().try_for_each(|def| walk(&def.body))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "soa_transform_tests.rs"]
mod soa_transform_tests;
