//! Monomorphization pass for TLC.
//!
//! This pass takes polymorphic functions (with size/type variables) and creates
//! specialized monomorphic copies for each concrete instantiation that's actually called.
//!
//! This happens at the TLC level, before MIR lowering, so that all functions are
//! monomorphic by the time we reach codegen.
//!
//! Example:
//!   def sum [n] (arr:[n]f32) : f32 = ...
//!
//! When called with [4]f32, creates:
//!   def sum$n4 (arr:[4]f32) : f32 = ...
//!
//! **Representation variants.** `apply_subst` substitutes type *variables*
//! but does not collapse array *representation* variants. In particular,
//! `ArrayVariantAbstract` survives substitution unchanged — it is a
//! first-class representation-polymorphic variant in the lattice, not a
//! placeholder. A helper specialized against an Abstract-typed argument
//! gets `Abstract` in its parameter type; the concrete representation is
//! chosen by the producer's EGIR lowering (`egir/from_tlc.rs`). Any
//! `Array[_, Abstract, _, _]` surviving past EGIR is caught by
//! `egir::verify_no_abstract` before backend emission.

use super::VarRef;
use super::{ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::ast::TypeName;
use crate::types::TypeExt;
use crate::types::TypeScheme;
use crate::{LookupMap, LookupSet};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::VecDeque;

/// A substitution mapping type variables to concrete types
type Substitution = LookupMap<usize, Type<TypeName>>;

/// Monomorphize a TLC program.
///
/// This walks through all definitions starting from entry points, finds calls
/// to polymorphic functions, and creates specialized versions with concrete types.
pub fn run(program: Program, schemes: &LookupMap<SymbolId, TypeScheme>) -> Program {
    let mono = Monomorphizer::new(program, schemes);
    let result = mono.run();
    result.assert_flat_apps();
    result
}

struct Monomorphizer<'a> {
    /// Symbol table for name lookup and allocation
    symbols: SymbolTable,
    /// Original polymorphic functions by symbol
    poly_functions: LookupMap<SymbolId, Def>,
    /// Generated monomorphic functions
    mono_functions: Vec<Def>,
    /// Map from (function_sym, spec_key) to specialized symbol
    specializations: LookupMap<(SymbolId, SpecKey), SymbolId>,
    /// Worklist of functions to process
    worklist: VecDeque<WorkItem>,
    /// Processed (original_sym, spec_key) pairs
    processed: LookupSet<(SymbolId, SpecKey)>,
    /// Type schemes for polymorphic functions (keyed by SymbolId)
    schemes: &'a LookupMap<SymbolId, TypeScheme>,
    /// Term ID source for creating new terms
    term_ids: TermIdSource,
    /// Canonical function name → def SymbolId mapping (passed through unchanged)
    def_syms: LookupMap<String, crate::SymbolId>,
}

struct WorkItem {
    /// Original function symbol (before specialization)
    original_sym: SymbolId,
    /// Specialization key (empty for monomorphic functions)
    spec_key: SpecKey,
    /// The function definition
    def: Def,
}

/// A key for looking up specializations
/// We use a sorted Vec instead of LookupMap for deterministic ordering
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SubstKey(Vec<(usize, TypeKey)>);

/// Combined specialization key: type substitution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SpecKey {
    /// Type substitution for specialization
    type_subst: SubstKey,
}

/// A simplified representation of types for use as hash keys
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum TypeKey {
    Var(usize),
    Size(usize),
    Constructed(String, Vec<TypeKey>),
    Record(Vec<(String, TypeKey)>),
    Sum(Vec<(String, Vec<TypeKey>)>),
    Existential(Vec<String>, Box<TypeKey>),
}

impl SubstKey {
    fn from_subst(subst: &Substitution) -> Self {
        let mut items: Vec<_> = subst.iter().map(|(k, v)| (*k, TypeKey::from_type(v))).collect();
        items.sort();
        SubstKey(items)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Convert back to a Substitution for use in specialize_def
    fn to_subst(&self) -> Substitution {
        self.0.iter().map(|(k, v)| (*k, v.to_type())).collect()
    }
}

impl SpecKey {
    /// Create an empty spec key (for monomorphic functions)
    fn empty() -> Self {
        SpecKey {
            type_subst: SubstKey(Vec::new()),
        }
    }

    fn new(subst: &Substitution) -> Self {
        SpecKey {
            type_subst: SubstKey::from_subst(subst),
        }
    }

    /// Returns true if this represents a non-trivial specialization
    fn needs_specialization(&self) -> bool {
        !self.type_subst.is_empty()
    }
}

impl TypeKey {
    fn from_type(ty: &Type<TypeName>) -> Self {
        match ty {
            Type::Variable(id) => TypeKey::Var(*id),
            Type::Constructed(name, args) => {
                // Handle types with nested structure that need full representation
                match name {
                    TypeName::Size(n) => return TypeKey::Size(*n),
                    TypeName::Record(fields) => {
                        // Field names are in RecordFields, field types are in args
                        let mut key_fields: Vec<_> = fields
                            .iter()
                            .zip(args.iter())
                            .map(|(k, v)| (k.clone(), TypeKey::from_type(v)))
                            .collect();
                        key_fields.sort_by(|a, b| a.0.cmp(&b.0));
                        return TypeKey::Record(key_fields);
                    }
                    TypeName::Sum(variants) => {
                        let key_variants: Vec<_> = variants
                            .iter()
                            .map(|(name, types)| {
                                (name.clone(), types.iter().map(TypeKey::from_type).collect())
                            })
                            .collect();
                        return TypeKey::Sum(key_variants);
                    }
                    TypeName::Existential(vars) => {
                        let inner = &args[0];
                        return TypeKey::Existential(vars.clone(), Box::new(TypeKey::from_type(inner)));
                    }
                    _ => {}
                }

                // For other constructed types, use a string name + args
                let name_str = match name {
                    TypeName::Bool => "bool".to_string(),
                    TypeName::Float(bits) => format!("f{}", bits),
                    TypeName::UInt(bits) => format!("u{}", bits),
                    TypeName::Int(bits) => format!("i{}", bits),
                    TypeName::Array => "array".to_string(),
                    TypeName::Vec => "vec".to_string(),
                    TypeName::Mat => "mat".to_string(),
                    TypeName::SizePlaceholder => {
                        panic!("SizePlaceholder should be resolved before monomorphization")
                    }
                    TypeName::Arrow => "arrow".to_string(),
                    TypeName::SizeVar(s) => format!("sizevar_{}", s),
                    TypeName::UserVar(s) => format!("uservar_{}", s),
                    TypeName::Named(s) => s.clone(),
                    TypeName::Unique => {
                        return TypeKey::Constructed(
                            "unique".to_string(),
                            args.iter().map(TypeKey::from_type).collect(),
                        );
                    }
                    TypeName::Unit => "unit".to_string(),
                    TypeName::Tuple(n) => format!("tuple{}", n),
                    TypeName::Pointer => "ptr".to_string(),
                    TypeName::ArrayVariantComposite => "array_composite".to_string(),
                    TypeName::ArrayVariantView => "array_view".to_string(),
                    TypeName::ArrayVariantVirtual => "array_virtual".to_string(),
                    TypeName::ArrayVariantBounded => "array_bounded".to_string(),
                    TypeName::ArrayVariantAbstract => "array_abstract".to_string(),
                    TypeName::Region(b) => format!("region_s{}_b{}", b.set, b.binding),
                    TypeName::NoRegion => "no_region".to_string(),
                    TypeName::Texture2D => "texture2d".to_string(),
                    TypeName::Sampler => "sampler".to_string(),
                    TypeName::AddressPlaceholder => {
                        panic!("AddressPlaceholder should be resolved before monomorphization")
                    }
                    _ => unreachable!("Should have been handled above: {:?}", name),
                };
                TypeKey::Constructed(name_str, args.iter().map(TypeKey::from_type).collect())
            }
        }
    }

    /// Convert back to a Type. Used for reconstructing substitutions.
    fn to_type(&self) -> Type<TypeName> {
        match self {
            TypeKey::Var(id) => Type::Variable(*id),
            TypeKey::Size(n) => Type::Constructed(TypeName::Size(*n), vec![]),
            TypeKey::Record(fields) => {
                let field_names: Vec<_> = fields.iter().map(|(k, _)| k.clone()).collect();
                let field_types: Vec<_> = fields.iter().map(|(_, v)| v.to_type()).collect();
                Type::Constructed(TypeName::Record(field_names.into()), field_types)
            }
            TypeKey::Sum(variants) => {
                let variant_types: Vec<_> = variants
                    .iter()
                    .map(|(name, types)| (name.clone(), types.iter().map(|t| t.to_type()).collect()))
                    .collect();
                Type::Constructed(TypeName::Sum(variant_types), vec![])
            }
            TypeKey::Existential(vars, inner) => {
                Type::Constructed(TypeName::Existential(vars.clone()), vec![inner.to_type()])
            }
            TypeKey::Constructed(name, args) => {
                let type_args: Vec<_> = args.iter().map(|a| a.to_type()).collect();
                let type_name = match name.as_str() {
                    "f16" => TypeName::Float(16),
                    "f32" => TypeName::Float(32),
                    "f64" => TypeName::Float(64),
                    "u8" => TypeName::UInt(8),
                    "u16" => TypeName::UInt(16),
                    "u32" => TypeName::UInt(32),
                    "u64" => TypeName::UInt(64),
                    "i8" => TypeName::Int(8),
                    "i16" => TypeName::Int(16),
                    "i32" => TypeName::Int(32),
                    "i64" => TypeName::Int(64),
                    "array" => TypeName::Array,
                    "vec" => TypeName::Vec,
                    "mat" => TypeName::Mat,
                    "arrow" => TypeName::Arrow,
                    "unit" => TypeName::Unit,
                    "ptr" => TypeName::Pointer,
                    "unique" => TypeName::Unique,
                    "array_view" => TypeName::ArrayVariantView,
                    "array_composite" => TypeName::ArrayVariantComposite,
                    "array_virtual" => TypeName::ArrayVariantVirtual,
                    "array_bounded" => TypeName::ArrayVariantBounded,
                    "array_abstract" => TypeName::ArrayVariantAbstract,
                    "texture2d" => TypeName::Texture2D,
                    "sampler" => TypeName::Sampler,
                    s if s.starts_with("tuple") => {
                        let n: usize = s[5..]
                            .parse()
                            .unwrap_or_else(|_| panic!("BUG: invalid tuple arity in mangled name: {}", s));
                        TypeName::Tuple(n)
                    }
                    s if s.starts_with("sizevar_") => TypeName::SizeVar(s[8..].to_string()),
                    s if s.starts_with("uservar_") => TypeName::UserVar(s[8..].to_string()),
                    s if s.starts_with("region_s") => {
                        let rest = &s["region_s".len()..];
                        let (set_s, bind_s) = rest
                            .split_once("_b")
                            .unwrap_or_else(|| panic!("BUG: invalid mangled region name: {}", s));
                        let set =
                            set_s.parse().unwrap_or_else(|_| panic!("BUG: invalid region set in: {}", s));
                        let binding = bind_s
                            .parse()
                            .unwrap_or_else(|_| panic!("BUG: invalid region binding in: {}", s));
                        TypeName::Region(crate::BindingRef::new(set, binding))
                    }
                    "no_region" => TypeName::NoRegion,
                    s => TypeName::Named(s.to_string()),
                };
                Type::Constructed(type_name, type_args)
            }
        }
    }
}

// =============================================================================
// Scheme Instantiation Helpers
// =============================================================================

/// Unwrap a TypeScheme to get the inner monotype.
/// The scheme's bound variables are already unique within the scheme,
/// so we can use them directly for unification and substitution.
fn unwrap_scheme(scheme: &TypeScheme) -> &Type<TypeName> {
    match scheme {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { body, .. } => unwrap_scheme(body),
    }
}

/// Split a function type into (param_types, return_type).
/// (A -> B -> C) becomes ([A, B], C)
fn split_function_type(ty: &Type<TypeName>) -> (Vec<Type<TypeName>>, Type<TypeName>) {
    let mut params = Vec::new();
    let mut current = ty.clone();

    while let Type::Constructed(TypeName::Arrow, ref args) = current {
        if args.len() == 2 {
            params.push(args[0].clone());
            current = args[1].clone();
        } else {
            break;
        }
    }

    (params, current)
}

impl<'a> Monomorphizer<'a> {
    fn new(program: Program, schemes: &'a LookupMap<SymbolId, TypeScheme>) -> Self {
        // Build function map and collect entry points
        let mut poly_functions = LookupMap::new();
        let mut entry_points = Vec::new();

        for def in program.defs.iter() {
            let sym = def.name;

            // For entry points, add to worklist
            if matches!(&def.meta, DefMeta::EntryPoint(_)) {
                entry_points.push(WorkItem {
                    original_sym: sym,
                    spec_key: SpecKey::empty(),
                    def: def.clone(),
                });
            }

            poly_functions.insert(sym, def.clone());
        }

        let mut worklist = VecDeque::new();
        worklist.extend(entry_points);

        Monomorphizer {
            symbols: program.symbols,
            poly_functions,
            mono_functions: Vec::new(),
            specializations: LookupMap::new(),
            worklist,
            processed: LookupSet::new(),
            schemes,
            term_ids: TermIdSource::new(),
            def_syms: program.def_syms,
        }
    }

    fn run(mut self) -> Program {
        while let Some(work_item) = self.worklist.pop_front() {
            let key = (work_item.original_sym, work_item.spec_key.clone());
            if self.processed.contains(&key) {
                continue;
            }
            self.processed.insert(key);

            // Process this function: look for calls and specialize callees
            let def = self.process_def(work_item.def);
            self.mono_functions.push(def);
        }

        Program {
            defs: self.mono_functions,
            symbols: self.symbols,
            def_syms: self.def_syms,
        }
    }

    /// Ensure a definition is in the worklist (for monomorphic callees and constants)
    fn ensure_in_worklist(&mut self, sym: SymbolId, def: Def) {
        let key = (sym, SpecKey::empty());
        if !self.processed.contains(&key) {
            // Check if it's already in the worklist
            let already_queued =
                self.worklist.iter().any(|w| w.original_sym == sym && !w.spec_key.needs_specialization());
            if !already_queued {
                self.worklist.push_back(WorkItem {
                    original_sym: sym,
                    spec_key: SpecKey::empty(),
                    def,
                });
            }
        }
    }

    fn process_def(&mut self, def: Def) -> Def {
        let new_body = self.process_term(&def.body);
        Def {
            body: new_body,
            ..def
        }
    }

    /// Process a term, rewriting calls to polymorphic functions.
    fn process_term(&mut self, term: &Term) -> Term {
        let kind = match &term.kind {
            TermKind::App { func, args } => {
                // Check if func is a variable referencing a known function
                if let TermKind::Var(VarRef::Symbol(sym)) = &func.kind {
                    let sym = *sym;
                    if let Some(poly_def) = self.poly_functions.get(&sym).cloned() {
                        // Infer substitution from argument types
                        let arg_types: Vec<_> = args.iter().map(|a| a.ty.clone()).collect();
                        let subst = self.infer_substitution(&poly_def, &arg_types);
                        let spec_key = SpecKey::new(&subst);

                        if spec_key.needs_specialization() {
                            // Get or create specialized version
                            let specialized_sym =
                                self.get_or_create_specialization(sym, &spec_key, &poly_def);
                            let new_func = Term {
                                id: self.term_ids.next_id(),
                                ty: func.ty.clone(),
                                span: func.span,
                                kind: TermKind::Var(VarRef::Symbol(specialized_sym)),
                            };
                            let processed_args: Vec<_> =
                                args.iter().map(|a| self.process_term(a)).collect();
                            return Term {
                                id: self.term_ids.next_id(),
                                ty: term.ty.clone(),
                                span: term.span,
                                kind: TermKind::App {
                                    func: Box::new(self.process_term(&new_func)),
                                    args: processed_args,
                                },
                            };
                        } else {
                            // Monomorphic call - ensure callee is in worklist
                            self.ensure_in_worklist(sym, poly_def);
                        }
                    }
                }

                // Default: just process recursively
                let processed_func = self.process_term(func);
                let processed_args: Vec<_> = args.iter().map(|a| self.process_term(a)).collect();
                TermKind::App {
                    func: Box::new(processed_func),
                    args: processed_args,
                }
            }

            TermKind::Var(VarRef::Symbol(sym)) => {
                let sym = *sym;
                // Check if this is a reference to a polymorphic function
                // This handles cases like `let f = some_poly_fn in ...`
                if let Some(poly_def) = self.poly_functions.get(&sym).cloned() {
                    // Try to infer specialization from the term's type
                    if let Some(subst) = self.infer_var_substitution(&poly_def, &term.ty) {
                        let spec_key = SpecKey::new(&subst);
                        if spec_key.needs_specialization() {
                            let specialized_sym =
                                self.get_or_create_specialization(sym, &spec_key, &poly_def);
                            return Term {
                                id: self.term_ids.next_id(),
                                ty: term.ty.clone(),
                                span: term.span,
                                kind: TermKind::Var(VarRef::Symbol(specialized_sym)),
                            };
                        } else {
                            self.ensure_in_worklist(sym, poly_def);
                        }
                    } else {
                        self.ensure_in_worklist(sym, poly_def);
                    }
                }
                TermKind::Var(VarRef::Symbol(sym))
            }

            // Catalog builtin reference: passes through unchanged.
            TermKind::Var(v @ VarRef::Builtin { .. }) => TermKind::Var(*v),

            TermKind::Lambda(Lambda { params, body, ret_ty }) => TermKind::Lambda(Lambda {
                params: params.clone(),
                body: Box::new(self.process_term(body)),
                ret_ty: ret_ty.clone(),
            }),

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => TermKind::Let {
                name: name.clone(),
                name_ty: name_ty.clone(),
                rhs: Box::new(self.process_term(rhs)),
                body: Box::new(self.process_term(body)),
            },

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TermKind::If {
                cond: Box::new(self.process_term(cond)),
                then_branch: Box::new(self.process_term(then_branch)),
                else_branch: Box::new(self.process_term(else_branch)),
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init_bindings = init_bindings
                    .iter()
                    .map(|(name, ty, expr)| (name.clone(), ty.clone(), self.process_term(expr)))
                    .collect();
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        iter: Box::new(self.process_term(iter)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        bound: Box::new(self.process_term(bound)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.process_term(cond)),
                    },
                };
                TermKind::Loop {
                    loop_var: loop_var.clone(),
                    loop_var_ty: loop_var_ty.clone(),
                    init: Box::new(self.process_term(init)),
                    init_bindings: new_init_bindings,
                    kind: new_kind,
                    body: Box::new(self.process_term(body)),
                }
            }

            // Leaves unchanged
            k @ (TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_)) => k.clone(),

            TermKind::Coerce { inner, target_ty } => TermKind::Coerce {
                inner: Box::new(self.process_term(inner)),
                target_ty: target_ty.clone(),
            },

            TermKind::Soac(ref soac) => TermKind::Soac(self.process_soac(soac)),

            TermKind::ArrayExpr(ref ae) => TermKind::ArrayExpr(self.process_array_expr(ae)),

            TermKind::Tuple(ref parts) => {
                TermKind::Tuple(parts.iter().map(|p| self.process_term(p)).collect())
            }
            TermKind::TupleProj { ref tuple, idx } => TermKind::TupleProj {
                tuple: Box::new(self.process_term(tuple)),
                idx: *idx,
            },
            TermKind::Index { ref array, ref index } => TermKind::Index {
                array: Box::new(self.process_term(array)),
                index: Box::new(self.process_term(index)),
            },
            TermKind::VecLit(ref parts) => {
                TermKind::VecLit(parts.iter().map(|p| self.process_term(p)).collect())
            }
            TermKind::OutputSlotStore {
                slot_index,
                ref value,
            } => TermKind::OutputSlotStore {
                slot_index: *slot_index,
                value: Box::new(self.process_term(value)),
            },
        };

        Term {
            id: self.term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind,
        }
    }

    fn process_lambda(&mut self, lam: &Lambda) -> Lambda {
        Lambda {
            params: lam.params.clone(),
            body: Box::new(self.process_term(&lam.body)),
            ret_ty: lam.ret_ty.clone(),
        }
    }

    fn process_soac_body(&mut self, sb: &super::SoacBody) -> super::SoacBody {
        super::SoacBody {
            lam: self.process_lambda(&sb.lam),
            captures: sb.captures.iter().map(|(s, ty, t)| (*s, ty.clone(), self.process_term(t))).collect(),
        }
    }

    fn process_soac(&mut self, soac: &SoacOp) -> SoacOp {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => SoacOp::Map {
                lam: self.process_soac_body(lam),
                inputs: inputs.iter().map(|ae| self.process_array_expr(ae)).collect(),
                destination: *destination,
            },
            SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
                op: self.process_soac_body(op),
                ne: Box::new(self.process_term(ne)),
                input: self.process_array_expr(input),
            },
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                destination,
            } => SoacOp::Scan {
                op: self.process_soac_body(op),
                reduce_op: self.process_soac_body(reduce_op),
                ne: Box::new(self.process_term(ne)),
                input: self.process_array_expr(input),
                destination: *destination,
            },
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => SoacOp::Filter {
                pred: self.process_soac_body(pred),
                input: self.process_array_expr(input),
                destination: *destination,
            },
            SoacOp::Scatter { dest, lam, inputs } => SoacOp::Scatter {
                dest: dest.clone(),
                lam: self.process_soac_body(lam),
                inputs: inputs.iter().map(|ae| self.process_array_expr(ae)).collect(),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => SoacOp::ReduceByIndex {
                dest: dest.clone(),
                op: self.process_soac_body(op),
                ne: Box::new(self.process_term(ne)),
                indices: self.process_array_expr(indices),
                values: self.process_array_expr(values),
            },
            SoacOp::Screma {
                map_lams,
                accumulators,
                inputs,
                map_input_indices,
            } => SoacOp::Screma {
                map_lams: map_lams.iter().map(|body| self.process_soac_body(body)).collect(),
                accumulators: accumulators
                    .iter()
                    .map(|acc| super::ScremaAccumulatorSpec {
                        kind: acc.kind,
                        step_lam: self.process_soac_body(&acc.step_lam),
                        reduce_op: self.process_soac_body(&acc.reduce_op),
                        ne: Box::new(self.process_term(&acc.ne)),
                    })
                    .collect(),
                inputs: inputs.iter().map(|ae| self.process_array_expr(ae)).collect(),
                map_input_indices: map_input_indices.clone(),
            },
        }
    }

    fn process_array_expr(&mut self, ae: &ArrayExpr) -> ArrayExpr {
        match ae {
            // Route the named input through `process_term`: a var referencing a
            // top-level constant must reach `ensure_in_worklist` so the constant
            // survives monomorphization's worklist-driven DCE, and a reference to
            // a polymorphic def gets its symbol rewritten to the specialization.
            ArrayExpr::Var(vr, ty) => {
                let t = super::atom_var_term(*vr, ty.clone(), &mut self.term_ids);
                super::term_as_input_atom(self.process_term(&t))
            }
            ArrayExpr::Zip(exprs) => {
                ArrayExpr::Zip(exprs.iter().map(|e| self.process_array_expr(e)).collect())
            }
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.iter().map(|t| self.process_term(t)).collect())
            }
            ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
                start: Box::new(self.process_term(start)),
                len: Box::new(self.process_term(len)),
                step: step.as_ref().map(|s| Box::new(self.process_term(s))),
            },
            ArrayExpr::StorageView(sv) => ArrayExpr::StorageView(super::StorageView {
                binding: sv.binding,
                offset: Box::new(self.process_term(&sv.offset)),
                len: Box::new(self.process_term(&sv.len)),
                elem_ty: sv.elem_ty.clone(),
            }),
        }
    }

    /// Infer the substitution needed for a polymorphic function call.
    fn infer_substitution(&self, poly_def: &Def, arg_types: &[Type<TypeName>]) -> Substitution {
        let mut subst = Substitution::new();

        // Get the type scheme for this function
        if let Some(scheme) = self.schemes.get(&poly_def.name) {
            let func_type = unwrap_scheme(scheme);
            let (param_types, _ret_type) = split_function_type(func_type);

            // Unify parameter types with argument types
            for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                self.unify_for_subst(param_ty, arg_ty, &mut subst);
            }
        } else {
            // No scheme - try using the def's function type directly
            let (param_types, _ret_type) = split_function_type(&poly_def.ty);
            for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                self.unify_for_subst(param_ty, arg_ty, &mut subst);
            }
        }

        subst
    }

    /// Infer substitution for a variable reference based on its concrete type.
    fn infer_var_substitution(
        &self,
        poly_def: &Def,
        concrete_type: &Type<TypeName>,
    ) -> Option<Substitution> {
        let mut subst = Substitution::new();

        // Get the polymorphic type from scheme or def
        let poly_type = if let Some(scheme) = self.schemes.get(&poly_def.name) {
            unwrap_scheme(scheme).clone()
        } else {
            poly_def.ty.clone()
        };

        // Unify the polymorphic type with the concrete type
        self.unify_for_subst(&poly_type, concrete_type, &mut subst);
        if !subst.is_empty() {
            return Some(subst);
        }

        None
    }

    /// Unify two types to build a substitution.
    fn unify_for_subst(
        &self,
        expected: &Type<TypeName>,
        actual: &Type<TypeName>,
        subst: &mut Substitution,
    ) {
        match (expected, actual) {
            (Type::Variable(id), concrete) => {
                // Bind the scheme variable to the concrete type
                subst.insert(*id, concrete.clone());
            }
            (Type::Constructed(name1, args1), Type::Constructed(name2, args2)) => {
                // Recurse into constructed types only if type constructors match exactly
                if name1 != name2 {
                    return; // Type mismatch - just no binding
                }
                if args1.len() != args2.len() {
                    return; // Arity mismatch
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    self.unify_for_subst(a1, a2, subst);
                }
            }
            _ => {} // Other cases - no binding
        }
    }

    /// Get or create a specialized version of a function
    fn get_or_create_specialization(
        &mut self,
        func_sym: SymbolId,
        spec_key: &SpecKey,
        poly_def: &Def,
    ) -> SymbolId {
        let cache_key = (func_sym, spec_key.clone());

        if let Some(specialized_sym) = self.specializations.get(&cache_key) {
            return *specialized_sym;
        }

        // Build substitution from type_subst
        let subst = spec_key.type_subst.to_subst();

        // Create new specialized name from type substitution
        let func_name = self.symbols.get(func_sym).expect("BUG: func symbol not in table");
        let type_suffix = format_subst(&subst);
        let specialized_name = if type_suffix.is_empty() {
            func_name.to_string()
        } else {
            format!("{}${}", func_name, type_suffix)
        };
        let specialized_sym = self.symbols.alloc(specialized_name);

        // Clone and specialize the function
        let specialized_def = self.specialize_def(poly_def.clone(), &subst, specialized_sym);

        // Add to worklist to process its body
        self.worklist.push_back(WorkItem {
            original_sym: func_sym,
            spec_key: spec_key.clone(),
            def: specialized_def,
        });

        self.specializations.insert(cache_key, specialized_sym);
        specialized_sym
    }

    /// Create a specialized version of a function by applying substitution.
    fn specialize_def(&mut self, def: Def, subst: &Substitution, new_sym: SymbolId) -> Def {
        Def {
            name: new_sym,
            ty: apply_subst(&def.ty, subst),
            body: self.apply_subst_term(&def.body, subst),
            meta: def.meta,
            arity: def.arity,
        }
    }

    /// Apply a substitution to a term, recursively updating all types.
    fn apply_subst_term(&mut self, term: &Term, subst: &Substitution) -> Term {
        let new_ty = apply_subst(&term.ty, subst);
        let new_kind = match &term.kind {
            TermKind::Var(v) => TermKind::Var(*v),
            TermKind::IntLit(s) => TermKind::IntLit(s.clone()),
            TermKind::FloatLit(f) => TermKind::FloatLit(*f),
            TermKind::BoolLit(b) => TermKind::BoolLit(*b),
            TermKind::UnitLit => TermKind::UnitLit,
            TermKind::Coerce { inner, target_ty } => TermKind::Coerce {
                inner: Box::new(self.apply_subst_term(inner, subst)),
                target_ty: apply_subst(target_ty, subst),
            },
            TermKind::BinOp(op) => TermKind::BinOp(op.clone()),
            TermKind::UnOp(op) => TermKind::UnOp(op.clone()),
            TermKind::Extern(s) => TermKind::Extern(s.clone()),

            TermKind::Soac(ref soac) => TermKind::Soac(self.apply_subst_soac(soac, subst)),

            TermKind::ArrayExpr(ref ae) => TermKind::ArrayExpr(self.apply_subst_array_expr(ae, subst)),

            TermKind::Tuple(parts) => {
                TermKind::Tuple(parts.iter().map(|p| self.apply_subst_term(p, subst)).collect())
            }
            TermKind::TupleProj { tuple, idx } => TermKind::TupleProj {
                tuple: Box::new(self.apply_subst_term(tuple, subst)),
                idx: *idx,
            },
            TermKind::Index { array, index } => TermKind::Index {
                array: Box::new(self.apply_subst_term(array, subst)),
                index: Box::new(self.apply_subst_term(index, subst)),
            },
            TermKind::VecLit(parts) => {
                TermKind::VecLit(parts.iter().map(|p| self.apply_subst_term(p, subst)).collect())
            }
            TermKind::OutputSlotStore { slot_index, value } => TermKind::OutputSlotStore {
                slot_index: *slot_index,
                value: Box::new(self.apply_subst_term(value, subst)),
            },

            TermKind::Lambda(Lambda { params, body, ret_ty }) => TermKind::Lambda(Lambda {
                params: params.iter().map(|(p, ty)| (*p, apply_subst(ty, subst))).collect(),
                body: Box::new(self.apply_subst_term(body, subst)),
                ret_ty: apply_subst(ret_ty, subst),
            }),

            TermKind::App { func, args } => TermKind::App {
                func: Box::new(self.apply_subst_term(func, subst)),
                args: args.iter().map(|a| self.apply_subst_term(a, subst)).collect(),
            },

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => TermKind::Let {
                name: name.clone(),
                name_ty: apply_subst(name_ty, subst),
                rhs: Box::new(self.apply_subst_term(rhs, subst)),
                body: Box::new(self.apply_subst_term(body, subst)),
            },

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TermKind::If {
                cond: Box::new(self.apply_subst_term(cond, subst)),
                then_branch: Box::new(self.apply_subst_term(then_branch, subst)),
                else_branch: Box::new(self.apply_subst_term(else_branch, subst)),
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init_bindings = init_bindings
                    .iter()
                    .map(|(name, ty, expr)| {
                        (
                            name.clone(),
                            apply_subst(ty, subst),
                            self.apply_subst_term(expr, subst),
                        )
                    })
                    .collect();
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var: var.clone(),
                        var_ty: apply_subst(var_ty, subst),
                        iter: Box::new(self.apply_subst_term(iter, subst)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var: var.clone(),
                        var_ty: apply_subst(var_ty, subst),
                        bound: Box::new(self.apply_subst_term(bound, subst)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.apply_subst_term(cond, subst)),
                    },
                };
                TermKind::Loop {
                    loop_var: loop_var.clone(),
                    loop_var_ty: apply_subst(loop_var_ty, subst),
                    init: Box::new(self.apply_subst_term(init, subst)),
                    init_bindings: new_init_bindings,
                    kind: new_kind,
                    body: Box::new(self.apply_subst_term(body, subst)),
                }
            }
        };

        Term {
            id: self.term_ids.next_id(),
            ty: new_ty,
            span: term.span,
            kind: new_kind,
        }
    }

    fn apply_subst_lambda(&mut self, lam: &Lambda, subst: &Substitution) -> Lambda {
        Lambda {
            params: lam.params.iter().map(|(p, ty)| (*p, apply_subst(ty, subst))).collect(),
            body: Box::new(self.apply_subst_term(&lam.body, subst)),
            ret_ty: apply_subst(&lam.ret_ty, subst),
        }
    }

    fn apply_subst_soac_body(&mut self, sb: &super::SoacBody, subst: &Substitution) -> super::SoacBody {
        super::SoacBody {
            lam: self.apply_subst_lambda(&sb.lam, subst),
            captures: sb
                .captures
                .iter()
                .map(|(s, ty, t)| (*s, apply_subst(ty, subst), self.apply_subst_term(t, subst)))
                .collect(),
        }
    }

    fn apply_subst_soac(&mut self, soac: &SoacOp, subst: &Substitution) -> SoacOp {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => SoacOp::Map {
                lam: self.apply_subst_soac_body(lam, subst),
                inputs: inputs.iter().map(|ae| self.apply_subst_array_expr(ae, subst)).collect(),
                destination: *destination,
            },
            SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
                op: self.apply_subst_soac_body(op, subst),
                ne: Box::new(self.apply_subst_term(ne, subst)),
                input: self.apply_subst_array_expr(input, subst),
            },
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                destination,
            } => SoacOp::Scan {
                op: self.apply_subst_soac_body(op, subst),
                reduce_op: self.apply_subst_soac_body(reduce_op, subst),
                ne: Box::new(self.apply_subst_term(ne, subst)),
                input: self.apply_subst_array_expr(input, subst),
                destination: *destination,
            },
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => SoacOp::Filter {
                pred: self.apply_subst_soac_body(pred, subst),
                input: self.apply_subst_array_expr(input, subst),
                destination: *destination,
            },
            SoacOp::Scatter { dest, lam, inputs } => SoacOp::Scatter {
                dest: dest.clone(),
                lam: self.apply_subst_soac_body(lam, subst),
                inputs: inputs.iter().map(|ae| self.apply_subst_array_expr(ae, subst)).collect(),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => SoacOp::ReduceByIndex {
                dest: dest.clone(),
                op: self.apply_subst_soac_body(op, subst),
                ne: Box::new(self.apply_subst_term(ne, subst)),
                indices: self.apply_subst_array_expr(indices, subst),
                values: self.apply_subst_array_expr(values, subst),
            },
            SoacOp::Screma {
                map_lams,
                accumulators,
                inputs,
                map_input_indices,
            } => SoacOp::Screma {
                map_lams: map_lams.iter().map(|body| self.apply_subst_soac_body(body, subst)).collect(),
                accumulators: accumulators
                    .iter()
                    .map(|acc| super::ScremaAccumulatorSpec {
                        kind: acc.kind,
                        step_lam: self.apply_subst_soac_body(&acc.step_lam, subst),
                        reduce_op: self.apply_subst_soac_body(&acc.reduce_op, subst),
                        ne: Box::new(self.apply_subst_term(&acc.ne, subst)),
                    })
                    .collect(),
                inputs: inputs.iter().map(|ae| self.apply_subst_array_expr(ae, subst)).collect(),
                map_input_indices: map_input_indices.clone(),
            },
        }
    }

    fn apply_subst_array_expr(&mut self, ae: &ArrayExpr, subst: &Substitution) -> ArrayExpr {
        match ae {
            ArrayExpr::Var(vr, ty) => ArrayExpr::Var(*vr, apply_subst(ty, subst)),
            ArrayExpr::Zip(exprs) => {
                ArrayExpr::Zip(exprs.iter().map(|e| self.apply_subst_array_expr(e, subst)).collect())
            }
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.iter().map(|t| self.apply_subst_term(t, subst)).collect())
            }
            ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
                start: Box::new(self.apply_subst_term(start, subst)),
                len: Box::new(self.apply_subst_term(len, subst)),
                step: step.as_ref().map(|s| Box::new(self.apply_subst_term(s, subst))),
            },
            ArrayExpr::StorageView(sv) => ArrayExpr::StorageView(super::StorageView {
                binding: sv.binding,
                offset: Box::new(self.apply_subst_term(&sv.offset, subst)),
                len: Box::new(self.apply_subst_term(&sv.len, subst)),
                elem_ty: sv.elem_ty.clone(),
            }),
        }
    }
}

/// Apply a substitution to a type
pub(crate) fn apply_subst(ty: &Type<TypeName>, subst: &Substitution) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Type::Constructed(name, args) => {
            // Recursively apply substitution to type arguments
            let new_args = args.iter().map(|arg| apply_subst(arg, subst)).collect();

            // Also apply substitution to types nested inside TypeName
            let new_name = match name {
                TypeName::Record(fields) => TypeName::Record(fields.clone()),
                TypeName::Sum(variants) => {
                    let new_variants = variants
                        .iter()
                        .map(|(name, types)| {
                            (
                                name.clone(),
                                types.iter().map(|t| apply_subst(t, subst)).collect(),
                            )
                        })
                        .collect();
                    TypeName::Sum(new_variants)
                }
                TypeName::Existential(vars) => TypeName::Existential(vars.clone()),
                _ => name.clone(),
            };

            Type::Constructed(new_name, new_args)
        }
    }
}

/// Format a substitution for use in specialized function names
fn format_subst(subst: &Substitution) -> String {
    let mut items: Vec<_> = subst.iter().collect();
    items.sort_by_key(|(k, _)| *k);

    items.iter().map(|(_, ty)| format_type_compact(ty)).collect::<Vec<_>>().join("_")
}

fn format_type_compact(ty: &Type<TypeName>) -> String {
    match ty {
        Type::Variable(id) => format!("v{}", id),
        Type::Constructed(TypeName::Size(n), _) => format!("n{}", n),
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
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("tup{}_{}", arity, args_str)
        }
        _ if ty.is_vec() => {
            let elem = format_type_compact(ty.elem_type().expect("Vec has elem"));
            let size = format_type_compact(ty.vec_size_type().expect("Vec has size"));
            format!("vec_{}_{}", elem, size)
        }
        Type::Constructed(TypeName::Float(bits), _) => format!("f{}", bits),
        Type::Constructed(TypeName::Int(bits), _) => format!("i{}", bits),
        Type::Constructed(TypeName::UInt(bits), _) => format!("u{}", bits),
        Type::Constructed(TypeName::Arrow, args) => {
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("fn_{}", args_str)
        }
        Type::Constructed(TypeName::Unit, _) => "unit".to_string(),
        Type::Constructed(TypeName::Named(name), args) if args.is_empty() => name.clone(),
        Type::Constructed(TypeName::Named(name), args) => {
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("{}_{}", name, args_str)
        }
        Type::Constructed(TypeName::ArrayVariantView, _) => "array_view".to_string(),
        Type::Constructed(TypeName::Region(b), _) => format!("region_s{}_b{}", b.set, b.binding),
        Type::Constructed(TypeName::NoRegion, _) => "no_region".to_string(),
        Type::Constructed(TypeName::ArrayVariantComposite, _) => "array_composite".to_string(),
        Type::Constructed(TypeName::ArrayVariantVirtual, _) => "array_virtual".to_string(),
        Type::Constructed(TypeName::ArrayVariantBounded, _) => "array_bounded".to_string(),
        Type::Constructed(TypeName::ArrayVariantAbstract, _) => "array_abstract".to_string(),
        Type::Constructed(name, args) => {
            // Fallback for other constructed types
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            if args_str.is_empty() {
                format!("{:?}", name)
            } else {
                format!("{:?}_{}", name, args_str)
            }
        }
    }
}

#[cfg(test)]
#[path = "monomorphize_tests.rs"]
mod monomorphize_tests;
