//! Buffer specialization pass for TLC.
//!
//! Runs after monomorphize, before inline. Rewrites functions that take
//! view-array parameters so that the buffer identity (set, binding) is
//! baked into the function body. After this pass, no `DefMeta::Function`
//! should have view-array parameters — only entry points retain them.
//!
//! When `first(data)` is called where `data` is bound to storage buffer
//! (set=0, binding=0), this creates a specialized copy:
//!
//! ```text
//! def first_buf0_0(offset: u32, len: u32) =
//!     _w_intrinsic_storage_index(0, 0, offset + 0)
//! ```

use super::VarRef;
use super::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind,
};
use crate::ast::{self, Span, TypeName};
use crate::builtins::catalog;
use crate::interface;
use crate::interface::EntryParamBindingKind;
use crate::tlc::var_term_builtin_id;
use crate::types::TypeExt;
use crate::{BindingRef, SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashMap;

/// Buffer binding info for a view-array parameter.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BufferBinding {
    binding: BindingRef,
    elem_ty: Type<TypeName>,
}

/// Provenance for a view-bearing symbol: a compile-time `binding` plus the
/// `u32` `offset`/`len` *expressions* into that buffer. The single environment
/// the unified view-op walker resolves against. Entry params and specialized
/// function params differ only in what `offset`/`len` they're seeded with —
/// `0`/`storage_len(binding)` for an entry's whole-buffer view, the split
/// `(offset, len)` param vars for a specialized function — not in any tag here.
/// (Cloned per use site; TLC tolerates duplicate term ids, so no enum/recipe
/// indirection is needed.)
#[derive(Debug, Clone)]
struct ViewProv {
    binding: BindingRef,
    elem_ty: Type<TypeName>,
    offset: Term,
    len: Term,
}

/// A specialization key: for each parameter position, `Some(binding)` if it's
/// a buffer-backed view, `None` otherwise.
type SpecKey = Vec<Option<BindingRef>>;

/// Buffer specializer state.
struct BufferSpecializer {
    symbols: SymbolTable,
    term_ids: TermIdSource,
    /// Cache: (original_def_sym, spec_key) → specialized_def_sym
    specializations: HashMap<(SymbolId, SpecKey), SymbolId>,
    /// Newly generated defs
    new_defs: Vec<Def>,
    /// All defs by symbol for lookup
    def_map: HashMap<SymbolId, Def>,
}

/// Populate `EntryDecl::param_bindings` on every entry-point def.
/// Done once at the start of buffer specialization; the result is
/// read by this pass, `tlc::parallelize`, and `egir::from_tlc`.
fn populate_entry_param_bindings(program: &mut Program) {
    for def in program.defs.iter_mut() {
        let DefMeta::EntryPoint(entry) = &mut def.meta else {
            continue;
        };
        let (params, _) = extract_params(&def.body);
        entry.param_bindings = crate::binding_layout::compute_entry_binding_layout(
            &params,
            entry,
            crate::egir::from_tlc::AUTO_STORAGE_SET,
        );
    }
}

/// Check if a type is a view array (unsized, ArrayVariantView).
fn is_view_array(ty: &Type<TypeName>) -> bool {
    if !ty.is_array() {
        return false;
    }
    let is_view = ty
        .array_variant()
        .map(|v| matches!(v, Type::Constructed(TypeName::ArrayVariantView, _)))
        .unwrap_or(false);
    let is_unsized = ty.array_size().map(|s| matches!(s, Type::Variable(_))).unwrap_or(false);
    is_view && is_unsized
}

/// Extract element type from an array type.
fn array_elem_type(ty: &Type<TypeName>) -> Type<TypeName> {
    ty.elem_type().expect("array has elem type").clone()
}

/// If `ty` is a fixed-size *composite* array `[N]elem`, return `N`.
fn composite_array_len(ty: &Type<TypeName>) -> Option<usize> {
    let is_composite = ty
        .array_variant()
        .map(|v| matches!(v, Type::Constructed(TypeName::ArrayVariantComposite, _)))
        .unwrap_or(false);
    if !is_composite {
        return None;
    }
    match ty.array_size()? {
        Type::Constructed(TypeName::Size(n), _) => Some(*n),
        _ => None,
    }
}

/// Whether a term is the literal integer `0`. A whole-buffer view's offset is
/// `0`; recognizing it lets the index/length lowering drop the no-op `0 + i`
/// and the index coercion, keeping the entry body's normal form byte-identical
/// to what downstream passes (`parallelize`, size-hint) pattern-match.
fn is_int_lit_zero(t: &Term) -> bool {
    matches!(&t.kind, TermKind::IntLit(s) if s == "0")
}

/// Extract the parameters of a function by walking nested lambdas.
fn extract_params(term: &Term) -> (Vec<(SymbolId, Type<TypeName>)>, &Term) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            let (mut inner_params, inner_body) = extract_params(&lam.body);
            let mut params: Vec<(SymbolId, Type<TypeName>)> = lam.params.clone();
            params.append(&mut inner_params);
            (params, inner_body)
        }
        _ => (vec![], term),
    }
}

/// Run buffer specialization on a TLC program. Before doing anything
/// else, populate `EntryDecl::param_bindings` on every entry — the
/// auto-storage binding layout becomes data that every downstream
/// consumer (this pass, `parallelize`, `from_tlc`) reads instead of
/// re-deriving.
pub fn run(mut program: Program) -> Program {
    populate_entry_param_bindings(&mut program);

    let mut specializer = BufferSpecializer {
        symbols: program.symbols,
        term_ids: TermIdSource::new(),
        specializations: HashMap::new(),
        new_defs: Vec::new(),
        def_map: HashMap::new(),
    };

    // Build def_map
    for def in &program.defs {
        specializer.def_map.insert(def.name, def.clone());
    }

    // Process each def. Compute entries seed the view-provenance environment
    // from their view-typed parameters; the single view-op walker lowers
    // reads/length/slices and specializes any function or lifted lambda
    // reached through a view arg. This is essential for the defunctionalized
    // lambda case, where a lifted lambda takes a buffer arg and calls a helper
    // through it — without walking bodies, a runtime-sized view reaches
    // `materialize` (→ SPIR-V can't type that).
    let mut processed_defs: Vec<Def> = Vec::new();
    for def in &program.defs {
        match &def.meta {
            DefMeta::EntryPoint(entry) => {
                // Compute entries lower their view params against a whole-buffer
                // provenance environment. Vertex/fragment entries have no view
                // parameters, but their bodies still go through the walker for
                // consistency.
                let processed = if matches!(entry.entry_type, interface::Attribute::Compute) {
                    specializer.process_entry_point(def, entry)
                } else {
                    specializer.rewrite_def_body(def)
                };
                processed_defs.push(processed);
            }
            DefMeta::Function | DefMeta::LiftedLambda => {
                let processed = specializer.specialize_function_body(def);
                processed_defs.push(processed);
            }
        }
    }

    // Append newly generated specialized defs
    processed_defs.extend(specializer.new_defs.drain(..));

    let result = Program {
        defs: processed_defs,
        symbols: specializer.symbols,
        ..program
    };
    result.assert_flat_apps();
    result
}

impl BufferSpecializer {
    /// Process a compute entry point: compute buffer bindings for its params,
    /// then walk the body rewriting calls to functions that receive buffer args.
    fn process_entry_point(&mut self, def: &Def, entry: &interface::EntryDecl) -> Def {
        // Seed the view-provenance environment from this entry's view-typed
        // params. Each is a whole-buffer view (offset 0, len storage_len);
        // the single view-op walker lowers reads/length/slices against it.
        //
        // Two sources:
        //
        // 1. Auto-bound (no explicit attribute): the layout pass in
        //    `populate_entry_param_bindings` assigned a binding number
        //    and cached it as a `Single` entry on `entry.param_bindings`.
        //
        // 2. Explicit-bound (`#[storage(set, binding, ...)]`): the layout
        //    pass deliberately left these as `None` in `param_bindings`
        //    so the auto-binding counter doesn't reserve their slot. We
        //    still need their provenance here, so read it back off the
        //    AST attribute on `entry.params`.
        //
        // Tuple-of-views entries are skipped because the body rewriter
        // handles only bare `Var(sym)` references, not `t.0` / `t.1`.
        let span = def.body.span;
        let mut view_params: HashMap<SymbolId, ViewProv> = HashMap::new();
        for param_binding in entry.param_bindings.iter().flatten() {
            let EntryParamBindingKind::Single { binding, elem_ty, .. } = &param_binding.kind else {
                continue;
            };
            let prov = self.whole_buffer_prov(*binding, elem_ty.clone(), span);
            view_params.insert(param_binding.param_sym, prov);
        }
        let (body_params, _) = extract_params(&def.body);
        for (i, (sym, ty)) in body_params.iter().enumerate() {
            let Some(pat) = entry.params.get(i) else {
                continue;
            };
            let Some(binding) = crate::binding_layout::extract_storage_binding(pat) else {
                continue;
            };
            if !is_view_array(ty) {
                continue;
            }
            let prov = self.whole_buffer_prov(binding, array_elem_type(ty), span);
            view_params.insert(*sym, prov);
        }

        let new_body = self.rewrite_specialized_body(&def.body, &view_params);

        Def {
            body: new_body,
            ..def.clone()
        }
    }

    /// Rewrite a def body with no view params — used for vertex/fragment
    /// entries that don't have buffer-backed view parameters.
    fn rewrite_def_body(&mut self, def: &Def) -> Def {
        let new_body = self.rewrite_specialized_body(&def.body, &HashMap::new());
        Def {
            body: new_body,
            ..def.clone()
        }
    }

    /// Apply buffer specialization to a non-entry `DefMeta::Function` or
    /// `DefMeta::LiftedLambda` body. `defunctionalize` lifts entry-point
    /// lambdas into plain functions, and a function that calls another
    /// through a view arg (e.g. `helper(view, i)`) still needs that call
    /// specialized to `_w_intrinsic_storage_index(...)`.
    fn specialize_function_body(&mut self, def: &Def) -> Def {
        // A view-param function is specialized per buffer at its call sites
        // (`specialize_call` / the SOAC capture path), each clone baking in a
        // real `(set, binding)`. We rewrite the original against an *empty*
        // provenance environment: its own view params have no binding to bake
        // — fabricating one (the old `BindingRef::new(0, n)`) is what made a
        // mis-routed read silently land in descriptor `(0,0)`. With no
        // provenance, any view op left on an unspecialized param simply isn't
        // lowered and fails downstream rather than reading the wrong buffer.
        // (When every call site specializes, this original is dead.)
        let new_body = self.rewrite_specialized_body(&def.body, &HashMap::new());
        Def {
            body: new_body,
            ..def.clone()
        }
    }

    fn try_specialize_soac_view_captures(
        &mut self,
        sb: &super::SoacBody,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> Option<super::SoacBody> {
        let TermKind::Var(VarRef::Symbol(lifted_sym)) = &sb.lam.body.kind else {
            return None;
        };
        let lifted_sym = *lifted_sym;
        let target_def = self.def_map.get(&lifted_sym)?.clone();
        if !matches!(target_def.meta, DefMeta::Function | DefMeta::LiftedLambda) {
            return None;
        }

        // closure_convert appends captures after the original lambda
        // params, so capture i corresponds to lifted-def param
        // `n_lam_params + i`.
        let (target_params, _) = extract_params(&target_def.body);
        let n_lam_params = sb.lam.params.len();
        if target_params.len() != n_lam_params + sb.captures.len() {
            return None;
        }

        let mut buffer_args: Vec<Option<BufferBinding>> = vec![None; target_params.len()];
        let mut cap_views: Vec<Option<crate::tlc::StorageView>> =
            (0..target_params.len()).map(|_| None).collect();
        let mut any_buffer = false;
        for (i, (_, _, cap_term)) in sb.captures.iter().enumerate() {
            let TermKind::Var(VarRef::Symbol(_)) = &cap_term.kind else {
                continue;
            };
            let Some(view) = self.try_resolve_view_expr(cap_term, view_params) else {
                continue;
            };
            buffer_args[n_lam_params + i] = Some(BufferBinding {
                binding: view.binding,
                elem_ty: view.elem_ty.clone(),
            });
            cap_views[n_lam_params + i] = Some(view);
            any_buffer = true;
        }
        if !any_buffer {
            return None;
        }

        // Share the `(orig_sym, spec_key) → spec_sym` cache with the App
        // path so a buffer reached via both forms gets one specialized
        // copy. SOAC sites need the offset/len param symbols too — those
        // come back as `split_syms` on a cache miss; on a hit, recover
        // them by walking the cached def's param list.
        let spec_key: SpecKey = buffer_args.iter().map(|b| b.as_ref().map(|bb| bb.binding)).collect();
        let (spec_sym, split_syms) =
            if let Some(&cached) = self.specializations.get(&(lifted_sym, spec_key.clone())) {
                (cached, self.recover_split_syms(cached, &buffer_args))
            } else {
                let (s, splits) =
                    self.create_specialized_def(lifted_sym, &target_def, &buffer_args, sb.lam.body.span);
                self.specializations.insert((lifted_sym, spec_key), s);
                (s, splits)
            };

        let new_lam_body = Term {
            id: self.term_ids.next_id(),
            kind: TermKind::Var(VarRef::Symbol(spec_sym)),
            ..(*sb.lam.body).clone()
        };
        let new_lam = Lambda {
            params: sb.lam.params.clone(),
            body: Box::new(new_lam_body),
            ret_ty: sb.lam.ret_ty.clone(),
        };

        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut new_captures: Vec<(SymbolId, Type<TypeName>, Term)> =
            Vec::with_capacity(sb.captures.len() + 1);
        for (i, (cap_sym, cap_ty, cap_term)) in sb.captures.iter().enumerate() {
            let pos = n_lam_params + i;
            match (&cap_views[pos], &split_syms[pos]) {
                (Some(view), Some((offset_sym, len_sym))) => {
                    // Forward the source view's runtime offset/len into the
                    // child def's split params — whole-buffer (0/storage_len)
                    // for an entry capture, the param's own bounds for a
                    // view-param capture.
                    new_captures.push((*offset_sym, u32_ty.clone(), (*view.offset).clone()));
                    new_captures.push((*len_sym, u32_ty.clone(), (*view.len).clone()));
                }
                _ => {
                    new_captures.push((
                        *cap_sym,
                        cap_ty.clone(),
                        self.rewrite_specialized_body(cap_term, view_params),
                    ));
                }
            }
        }

        Some(super::SoacBody {
            lam: new_lam,
            captures: new_captures,
        })
    }

    /// Given a cached specialized def and the buffer_args that produced
    /// it, walk the spec def's params to recover the `(offset_sym, len_sym)`
    /// allocated for each view position. Used when the specialization
    /// cache hits and we need the symbols at a new caller site.
    fn recover_split_syms(
        &self,
        spec_sym: SymbolId,
        buffer_args: &[Option<BufferBinding>],
    ) -> Vec<Option<(SymbolId, SymbolId)>> {
        let def = self
            .new_defs
            .iter()
            .find(|d| d.name == spec_sym)
            .expect("BUG: cached spec sym not in new_defs");
        let (params, _) = extract_params(&def.body);
        let mut splits: Vec<Option<(SymbolId, SymbolId)>> = vec![None; buffer_args.len()];
        let mut p = 0usize;
        for (i, slot) in buffer_args.iter().enumerate() {
            if slot.is_some() {
                splits[i] = Some((params[p].0, params[p + 1].0));
                p += 2;
            } else {
                p += 1;
            }
        }
        splits
    }

    /// Create a specialized version of a function for specific buffer bindings.
    /// At the call site, replace `f(buf)` with `f_specialized(0, storage_len(set, binding))`.
    fn specialize_call(
        &mut self,
        func_sym: SymbolId,
        target_def: &Def,
        args: &[&Term],
        buffer_args: &[Option<BufferBinding>],
        original_term: &Term,
    ) -> Term {
        let span = original_term.span;

        // Build specialization key
        let spec_key: SpecKey = buffer_args.iter().map(|b| b.as_ref().map(|bb| bb.binding)).collect();

        // Check if we already have this specialization. App call sites
        // build their new arg list inline below; the `split_syms` side
        // channel from `create_specialized_def` is unused on this path
        // (it exists for SOAC capture rewriting, where positions matter).
        let spec_sym = if let Some(&sym) = self.specializations.get(&(func_sym, spec_key.clone())) {
            sym
        } else {
            let (spec_sym, _splits) = self.create_specialized_def(func_sym, target_def, buffer_args, span);
            self.specializations.insert((func_sym, spec_key), spec_sym);
            spec_sym
        };

        // Build the call site: replace buffer args with (offset=0, len=storage_len(set,binding))
        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut new_args: Vec<Term> = Vec::new();

        for (arg, buf) in args.iter().zip(buffer_args.iter()) {
            if let Some(binding) = buf {
                // Replace with (0, _w_intrinsic_storage_len(set, binding))
                let zero = self.make_int_lit("0", u32_ty.clone(), span);
                let storage_len = super::storage_len_call(binding.binding, span, &mut self.term_ids);
                new_args.push(zero);
                new_args.push(storage_len);
            } else {
                // Args arrive already rewritten by the caller (the specialized
                // App arm); a non-buffer arg passes through unchanged.
                new_args.push((*arg).clone());
            }
        }

        // Build the application: spec_sym(new_args...)
        let func_ref = Term {
            id: self.term_ids.next_id(),
            ty: target_def.ty.clone(), // approximate — not critical
            span,
            kind: TermKind::Var(VarRef::Symbol(spec_sym)),
        };

        if new_args.is_empty() {
            func_ref
        } else {
            Term {
                id: self.term_ids.next_id(),
                ty: original_term.ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(func_ref),
                    args: new_args,
                },
            }
        }
    }

    /// Provenance for an entry's whole-buffer view: the descriptor `binding`,
    /// `offset = 0`, `len = storage_len(binding)`. The `0` offset is what the
    /// view-op lowering recognizes to keep the entry body's normal form intact.
    fn whole_buffer_prov(&mut self, binding: BindingRef, elem_ty: Type<TypeName>, span: Span) -> ViewProv {
        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
        let offset = self.make_int_lit("0", u32_ty, span);
        let len = super::storage_len_call(binding, span, &mut self.term_ids);
        ViewProv {
            binding,
            elem_ty,
            offset,
            len,
        }
    }

    /// A fresh `Var(sym)` term of type `ty`. Used to seed `ViewProv` offset/len.
    fn var_term(&mut self, sym: SymbolId, ty: Type<TypeName>, span: Span) -> Term {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind: TermKind::Var(VarRef::Symbol(sym)),
        }
    }

    /// Create a specialized copy of a function where view params are replaced
    /// with (offset: u32, len: u32) and array ops use storage intrinsics.
    /// Returns the new def's `SymbolId` plus, for each `buffer_args` entry,
    /// the `(offset_sym, len_sym)` allocated for that position (or `None`
    /// for non-view positions). Callers reading captures-by-position
    /// (SOACs) need these to label the new offset/len terms.
    fn create_specialized_def(
        &mut self,
        _func_sym: SymbolId,
        target_def: &Def,
        buffer_args: &[Option<BufferBinding>],
        _span: Span,
    ) -> (SymbolId, Vec<Option<(SymbolId, SymbolId)>>) {
        let orig_name = self.symbols.get(target_def.name).expect("BUG: symbol not in table").clone();

        // Build suffix from buffer bindings
        let suffix: String = buffer_args
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                b.as_ref().map(|bb| format!("_b{}s{}b{}", i, bb.binding.set, bb.binding.binding))
            })
            .collect();
        let spec_name = format!("{}{}", orig_name, suffix);
        let spec_sym = self.symbols.alloc(spec_name);

        // Extract the original function params and body
        let (orig_params, orig_body) = extract_params(&target_def.body);

        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);

        // Build new params and a mapping from old view params to their buffer info
        let mut new_params: Vec<(SymbolId, Type<TypeName>)> = Vec::new();
        // Maps old param sym → (offset_sym, len_sym, set, binding, elem_ty)
        let mut view_param_map: HashMap<SymbolId, ViewProv> = HashMap::new();

        let mut split_syms: Vec<Option<(SymbolId, SymbolId)>> =
            (0..buffer_args.len()).map(|_| None).collect();
        for (i, (sym, ty)) in orig_params.iter().enumerate() {
            if let Some(Some(binding)) = buffer_args.get(i) {
                // Replace view param with offset + len
                let offset_sym = self.symbols.alloc(format!("{}_offset", self.symbols.get(*sym).unwrap()));
                let len_sym = self.symbols.alloc(format!("{}_len", self.symbols.get(*sym).unwrap()));
                new_params.push((offset_sym, u32_ty.clone()));
                new_params.push((len_sym, u32_ty.clone()));
                let offset = self.var_term(offset_sym, u32_ty.clone(), _span);
                let len = self.var_term(len_sym, u32_ty.clone(), _span);
                view_param_map.insert(
                    *sym,
                    ViewProv {
                        binding: binding.binding,
                        elem_ty: binding.elem_ty.clone(),
                        offset,
                        len,
                    },
                );
                split_syms[i] = Some((offset_sym, len_sym));
            } else {
                new_params.push((*sym, ty.clone()));
            }
        }

        // Rewrite the body: replace _w_index(arr, i) and _w_intrinsic_length(arr)
        // where arr maps to a buffer param.
        let new_body = self.rewrite_specialized_body(orig_body, &view_param_map);

        // Wrap body in nested lambdas for the new params
        let final_body = wrap_in_lambdas(new_body, &new_params, &mut self.term_ids);

        let spec_def = Def {
            name: spec_sym,
            ty: target_def.ty.clone(), // TODO: could refine this
            body: final_body,
            meta: DefMeta::Function,
            arity: new_params.len(),
        };

        self.new_defs.push(spec_def);
        (spec_sym, split_syms)
    }

    /// Rewrite the body of a specialized function, replacing operations on
    /// view params with storage intrinsics.
    fn rewrite_specialized_body(&mut self, term: &Term, view_params: &HashMap<SymbolId, ViewProv>) -> Term {
        match &term.kind {
            TermKind::App { func, args } => {
                // `length(arr_expr)` where arr_expr resolves to a view.
                if var_term_builtin_id(func, &self.symbols) == Some(catalog().known().length)
                    && args.len() == 1
                {
                    if let Some(view) = self.try_resolve_view_expr(&args[0], view_params) {
                        if view.len.ty == term.ty {
                            return *view.len;
                        }
                        return self.make_app("i32.u32", vec![*view.len], term.ty.clone(), term.span);
                    }
                }

                // `_w_intrinsic_slice(view, start, end)` whose *result* is a
                // composite array is a slice→composite materialization (e.g.
                // `xs[0..3]` passed to a `[N]elem`-typed function). Lower it here
                // to an N-element composite of storage reads at `offset + k`, so
                // the view param's buffer provenance is baked in rather than
                // surviving as a bare `Var(view_param)` that reaches SPIR-V as an
                // undefined global. (A slice that stays a *view* is consumed by
                // Index/length/Let above and is left untouched — recursing into
                // its operand here would hit the bare-Var case.)
                if var_term_builtin_id(func, &self.symbols) == Some(catalog().known().slice) {
                    if let Some(n) = composite_array_len(&term.ty) {
                        if let Some(view) = self.try_resolve_view_expr(term, view_params) {
                            let span = term.span;
                            let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
                            // Bind the slice's base offset once; each element reads
                            // `offset + k`.
                            let off_sym = self.symbols.alloc("_slice_off".to_string());
                            let binding = view.binding;
                            let elem_ty = view.elem_ty.clone();
                            let mut reads: Vec<Term> = Vec::with_capacity(n);
                            for k in 0..n {
                                let off_ref = Term {
                                    id: self.term_ids.next_id(),
                                    ty: u32_ty.clone(),
                                    span,
                                    kind: TermKind::Var(VarRef::Symbol(off_sym)),
                                };
                                let k_lit = self.make_int_lit(&k.to_string(), u32_ty.clone(), span);
                                let idx = self.make_binop_app(
                                    ast::BinaryOp { op: "+".to_string() },
                                    off_ref,
                                    k_lit,
                                    u32_ty.clone(),
                                    span,
                                );
                                reads.push(super::storage_index_call(
                                    binding,
                                    idx,
                                    elem_ty.clone(),
                                    span,
                                    &mut self.term_ids,
                                ));
                            }
                            let array = Term {
                                id: self.term_ids.next_id(),
                                ty: term.ty.clone(),
                                span,
                                kind: TermKind::ArrayExpr(ArrayExpr::Literal(reads)),
                            };
                            return Term {
                                id: self.term_ids.next_id(),
                                ty: term.ty.clone(),
                                span,
                                kind: TermKind::Let {
                                    name: off_sym,
                                    name_ty: u32_ty,
                                    rhs: view.offset,
                                    body: Box::new(array),
                                },
                            };
                        }
                    }
                }

                // User-defined function specialization (only meaningful
                // for SymbolId-bound user functions).
                if let TermKind::Var(VarRef::Symbol(sym)) = &func.kind {
                    // Check for calls to functions that themselves take view args
                    // (recursive specialization)
                    let inner_buffer_args: Vec<Option<BufferBinding>> = args
                        .iter()
                        .map(|a| {
                            if let TermKind::Var(VarRef::Symbol(s)) = &a.kind {
                                // A view-typed arg is one of this scope's view
                                // params; build a BufferBinding from its provenance.
                                view_params.get(s).map(|prov| BufferBinding {
                                    binding: prov.binding,
                                    elem_ty: prov.elem_ty.clone(),
                                })
                            } else {
                                None
                            }
                        })
                        .collect();

                    let has_inner_buf = inner_buffer_args.iter().any(|b| b.is_some());
                    if has_inner_buf {
                        if let Some(target_def) = self.def_map.get(sym).cloned() {
                            if matches!(target_def.meta, DefMeta::Function | DefMeta::LiftedLambda) {
                                let rewritten_args: Vec<Term> = args
                                    .iter()
                                    .map(|a| self.rewrite_specialized_body(a, view_params))
                                    .collect();
                                let rewritten_refs: Vec<&Term> = rewritten_args.iter().collect();
                                return self.specialize_call(
                                    *sym,
                                    &target_def,
                                    &rewritten_refs,
                                    &inner_buffer_args,
                                    term,
                                );
                            }
                        }
                    }
                }

                // Default: recurse into func and args.
                let new_func = self.rewrite_specialized_body(func, view_params);
                let new_args: Vec<Term> =
                    args.iter().map(|a| self.rewrite_specialized_body(a, view_params)).collect();
                Term {
                    kind: TermKind::App {
                        func: Box::new(new_func),
                        args: new_args,
                    },
                    ..term.clone()
                }
            }

            TermKind::Lambda(lam) => {
                let new_body = self.rewrite_specialized_body(&lam.body, view_params);
                Term {
                    kind: TermKind::Lambda(Lambda {
                        params: lam.params.clone(),
                        body: Box::new(new_body),
                        ret_ty: lam.ret_ty.clone(),
                    }),
                    ..term.clone()
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                // Check if the RHS is a view expression (e.g. a slice of a buffer param).
                // If so, bind fresh offset/len symbols and extend view_params for the body.
                if let Some(view) = self.try_resolve_view_expr(rhs, view_params) {
                    let span = term.span;
                    let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);

                    // Fresh symbols for the let-bound view's offset and len
                    let base_name = crate::symbol_name_or_bug(&self.symbols, *name).to_string();
                    let offset_sym = self.symbols.alloc(format!("{}_offset", base_name));
                    let len_sym = self.symbols.alloc(format!("{}_len", base_name));

                    // Extend view_params so the body knows `name` is a view.
                    // The body refers to the let-bound offset/len symbols below.
                    let mut extended = view_params.clone();
                    extended.insert(
                        *name,
                        ViewProv {
                            binding: view.binding,
                            elem_ty: view.elem_ty.clone(),
                            offset: self.var_term(offset_sym, u32_ty.clone(), span),
                            len: self.var_term(len_sym, u32_ty.clone(), span),
                        },
                    );

                    let new_body = self.rewrite_specialized_body(body, &extended);

                    // Wrap body in let-bindings:
                    //   let offset_sym = <view.offset> in
                    //   let len_sym = <view.len> in
                    //   <new_body>
                    let inner = Term {
                        id: self.term_ids.next_id(),
                        ty: term.ty.clone(),
                        span,
                        kind: TermKind::Let {
                            name: len_sym,
                            name_ty: u32_ty.clone(),
                            rhs: view.len,
                            body: Box::new(new_body),
                        },
                    };
                    return Term {
                        id: self.term_ids.next_id(),
                        ty: term.ty.clone(),
                        span,
                        kind: TermKind::Let {
                            name: offset_sym,
                            name_ty: u32_ty,
                            rhs: view.offset,
                            body: Box::new(inner),
                        },
                    };
                }

                let new_rhs = self.rewrite_specialized_body(rhs, view_params);
                let new_body = self.rewrite_specialized_body(body, view_params);
                Term {
                    kind: TermKind::Let {
                        name: *name,
                        name_ty: name_ty.clone(),
                        rhs: Box::new(new_rhs),
                        body: Box::new(new_body),
                    },
                    ..term.clone()
                }
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let new_cond = self.rewrite_specialized_body(cond, view_params);
                let new_then = self.rewrite_specialized_body(then_branch, view_params);
                let new_else = self.rewrite_specialized_body(else_branch, view_params);
                Term {
                    kind: TermKind::If {
                        cond: Box::new(new_cond),
                        then_branch: Box::new(new_then),
                        else_branch: Box::new(new_else),
                    },
                    ..term.clone()
                }
            }

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init = self.rewrite_specialized_body(init, view_params);
                let new_bindings: Vec<_> = init_bindings
                    .iter()
                    .map(|(s, t, e)| (*s, t.clone(), self.rewrite_specialized_body(e, view_params)))
                    .collect();
                let new_kind = self.rewrite_specialized_loop_kind(kind, view_params);
                let new_body = self.rewrite_specialized_body(body, view_params);
                Term {
                    kind: TermKind::Loop {
                        loop_var: *loop_var,
                        loop_var_ty: loop_var_ty.clone(),
                        init: Box::new(new_init),
                        init_bindings: new_bindings,
                        kind: new_kind,
                        body: Box::new(new_body),
                    },
                    ..term.clone()
                }
            }

            TermKind::Soac(soac) => {
                let new_soac = self.rewrite_specialized_soac(soac, view_params);
                Term {
                    kind: TermKind::Soac(new_soac),
                    ..term.clone()
                }
            }

            TermKind::ArrayExpr(ae) => {
                let new_ae = self.rewrite_specialized_array_expr(ae, view_params);
                Term {
                    kind: TermKind::ArrayExpr(new_ae),
                    ..term.clone()
                }
            }

            // Leaves
            TermKind::Var(VarRef::Symbol(_)) => {
                // Reachable for two distinct shapes:
                //   1. `Var(view_param)` as a user-function call argument.
                //      The App arm above already computed `inner_buffer_args`
                //      from the *original* args and routes through
                //      `specialize_call`, which discards the rewritten arg for
                //      view-typed positions and inserts `(0, storage_len)`
                //      instead. Our clone here is throwaway in that path.
                //   2. `Var(sym)` where `sym` isn't a view param at all — a
                //      scalar / non-view local — passing through unchanged.
                //
                // The case the rewriter does NOT cover: a `let alias = expr in
                // …` where `expr` is view-typed but isn't recognized by
                // `try_resolve_view_expr` (which only handles bare
                // `Var(view_param)`, slice intrinsics, and a few App shapes).
                // For e.g. `let alias = (if cond then board else board) in
                // length(alias)`, the Let arm fails to extend `view_params`
                // with `alias`, so downstream `length(alias)` / `alias[i]`
                // sees a bare view-typed Var that never gets resolved.
                // No production source exercises that today; see the parked
                // `bare_var_view_alias_via_if_not_resolved` test in
                // `buffer_specialize_tests.rs`.
                term.clone()
            }
            TermKind::Var(VarRef::Builtin { .. })
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => term.clone(),

            TermKind::Coerce { inner, target_ty } => Term {
                kind: TermKind::Coerce {
                    inner: Box::new(self.rewrite_specialized_body(inner, view_params)),
                    target_ty: target_ty.clone(),
                },
                ..term.clone()
            },

            TermKind::Tuple(parts) => Term {
                kind: TermKind::Tuple(
                    parts.iter().map(|p| self.rewrite_specialized_body(p, view_params)).collect(),
                ),
                ..term.clone()
            },
            TermKind::TupleProj { tuple, idx } => Term {
                kind: TermKind::TupleProj {
                    tuple: Box::new(self.rewrite_specialized_body(tuple, view_params)),
                    idx: *idx,
                },
                ..term.clone()
            },
            TermKind::Index { array, index } => {
                // View-param index: rewrite to storage_index(set, binding, offset + i).
                if let Some(view) = self.try_resolve_view_expr(array, view_params) {
                    let span = term.span;
                    let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
                    // Whole-buffer view (offset 0): index directly, no `0 + i`
                    // and no coercion, matching the entry-body normal form.
                    if is_int_lit_zero(&view.offset) {
                        let idx = self.rewrite_specialized_body(index, view_params);
                        return super::storage_index_call(
                            view.binding,
                            idx,
                            view.elem_ty,
                            span,
                            &mut self.term_ids,
                        );
                    }
                    let idx = self.rewrite_specialized_body(index, view_params);
                    let idx = if idx.ty == u32_ty {
                        idx
                    } else {
                        self.make_app("u32.i32", vec![idx], u32_ty.clone(), span)
                    };
                    let add_result = self.make_binop_app(
                        ast::BinaryOp { op: "+".to_string() },
                        *view.offset,
                        idx,
                        u32_ty,
                        span,
                    );
                    return super::storage_index_call(
                        view.binding,
                        add_result,
                        view.elem_ty,
                        span,
                        &mut self.term_ids,
                    );
                }
                Term {
                    kind: TermKind::Index {
                        array: Box::new(self.rewrite_specialized_body(array, view_params)),
                        index: Box::new(self.rewrite_specialized_body(index, view_params)),
                    },
                    ..term.clone()
                }
            }
            TermKind::VecLit(parts) => Term {
                kind: TermKind::VecLit(
                    parts.iter().map(|p| self.rewrite_specialized_body(p, view_params)).collect(),
                ),
                ..term.clone()
            },
            TermKind::OutputSlotStore { slot_index, value } => Term {
                kind: TermKind::OutputSlotStore {
                    slot_index: *slot_index,
                    value: Box::new(self.rewrite_specialized_body(value, view_params)),
                },
                ..term.clone()
            },
        }
    }

    fn rewrite_specialized_loop_kind(
        &mut self,
        kind: &LoopKind,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> LoopKind {
        match kind {
            LoopKind::For { var, var_ty, iter } => LoopKind::For {
                var: *var,
                var_ty: var_ty.clone(),
                iter: Box::new(self.rewrite_specialized_body(iter, view_params)),
            },
            LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                var: *var,
                var_ty: var_ty.clone(),
                bound: Box::new(self.rewrite_specialized_body(bound, view_params)),
            },
            LoopKind::While { cond } => LoopKind::While {
                cond: Box::new(self.rewrite_specialized_body(cond, view_params)),
            },
        }
    }

    fn rewrite_specialized_soac(
        &mut self,
        soac: &SoacOp,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> SoacOp {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => SoacOp::Map {
                lam: self.rewrite_specialized_soac_body(lam, view_params),
                inputs: inputs
                    .iter()
                    .map(|ae| self.rewrite_specialized_array_expr(ae, view_params))
                    .collect(),
                destination: *destination,
            },
            SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
                op: self.rewrite_specialized_soac_body(op, view_params),
                ne: Box::new(self.rewrite_specialized_body(ne, view_params)),
                input: self.rewrite_specialized_array_expr(input, view_params),
            },
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                destination,
            } => SoacOp::Scan {
                op: self.rewrite_specialized_soac_body(op, view_params),
                reduce_op: self.rewrite_specialized_soac_body(reduce_op, view_params),
                ne: Box::new(self.rewrite_specialized_body(ne, view_params)),
                input: self.rewrite_specialized_array_expr(input, view_params),
                destination: *destination,
            },
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => SoacOp::Filter {
                pred: self.rewrite_specialized_soac_body(pred, view_params),
                input: self.rewrite_specialized_array_expr(input, view_params),
                destination: *destination,
            },
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => SoacOp::Scatter {
                dest: self.rewrite_specialized_place(dest, view_params),
                indices: self.rewrite_specialized_array_expr(indices, view_params),
                values: self.rewrite_specialized_array_expr(values, view_params),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => SoacOp::ReduceByIndex {
                dest: self.rewrite_specialized_place(dest, view_params),
                op: self.rewrite_specialized_soac_body(op, view_params),
                ne: Box::new(self.rewrite_specialized_body(ne, view_params)),
                indices: self.rewrite_specialized_array_expr(indices, view_params),
                values: self.rewrite_specialized_array_expr(values, view_params),
            },
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
            } => SoacOp::Redomap {
                op: self.rewrite_specialized_soac_body(op, view_params),
                reduce_op: self.rewrite_specialized_soac_body(reduce_op, view_params),
                ne: Box::new(self.rewrite_specialized_body(ne, view_params)),
                inputs: inputs
                    .iter()
                    .map(|ae| self.rewrite_specialized_array_expr(ae, view_params))
                    .collect(),
            },
        }
    }

    fn rewrite_specialized_lambda(
        &mut self,
        lam: &Lambda,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> Lambda {
        Lambda {
            params: lam.params.clone(),
            body: Box::new(self.rewrite_specialized_body(&lam.body, view_params)),
            ret_ty: lam.ret_ty.clone(),
        }
    }

    fn rewrite_specialized_soac_body(
        &mut self,
        sb: &super::SoacBody,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> super::SoacBody {
        // Same capture-specialization the entry path uses, resolved against the
        // local view_params: a SOAC whose lambda captures one of this
        // specialized function's view params must specialize the lifted def too.
        if let Some(new_sb) = self.try_specialize_soac_view_captures(sb, view_params) {
            return new_sb;
        }

        super::SoacBody {
            lam: self.rewrite_specialized_lambda(&sb.lam, view_params),
            captures: sb
                .captures
                .iter()
                .map(|(s, t, e)| (*s, t.clone(), self.rewrite_specialized_body(e, view_params)))
                .collect(),
        }
    }

    fn rewrite_specialized_array_expr(
        &mut self,
        ae: &ArrayExpr,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => {
                // A bare `Var` pointing to a view param used as a whole array
                // (e.g. a SOAC input). A specialized-function view param (offset
                // is a `Var`) no longer exists as a value — reconstitute it as an
                // explicit `StorageView`. An entry whole-buffer view (offset 0)
                // still denotes a live param; leave the `Var` so egir/parallelize
                // resolve its binding and size hint from the entry layout — the
                // normal form those passes pattern-match.
                if let TermKind::Var(VarRef::Symbol(sym)) = &t.kind {
                    if let Some(prov) = view_params.get(sym) {
                        if !is_int_lit_zero(&prov.offset) {
                            return ArrayExpr::StorageView(crate::tlc::StorageView {
                                binding: prov.binding,
                                offset: Box::new(prov.offset.clone()),
                                len: Box::new(prov.len.clone()),
                                elem_ty: prov.elem_ty.clone(),
                            });
                        }
                    }
                }
                ArrayExpr::Ref(Box::new(self.rewrite_specialized_body(t, view_params)))
            }
            ArrayExpr::Zip(aes) => ArrayExpr::Zip(
                aes.iter().map(|a| self.rewrite_specialized_array_expr(a, view_params)).collect(),
            ),
            ArrayExpr::Soac(op) => {
                ArrayExpr::Soac(Box::new(self.rewrite_specialized_soac(op, view_params)))
            }
            ArrayExpr::Literal(terms) => ArrayExpr::Literal(
                terms.iter().map(|t| self.rewrite_specialized_body(t, view_params)).collect(),
            ),
            ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
                start: Box::new(self.rewrite_specialized_body(start, view_params)),
                len: Box::new(self.rewrite_specialized_body(len, view_params)),
                step: step.as_ref().map(|s| Box::new(self.rewrite_specialized_body(s, view_params))),
            },
            ArrayExpr::StorageView(sv) => ArrayExpr::StorageView(crate::tlc::StorageView {
                binding: sv.binding,
                offset: Box::new(self.rewrite_specialized_body(&sv.offset, view_params)),
                len: Box::new(self.rewrite_specialized_body(&sv.len, view_params)),
                elem_ty: sv.elem_ty.clone(),
            }),
        }
    }

    fn rewrite_specialized_place(
        &mut self,
        place: &Place,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> Place {
        match place {
            Place::BufferSlice {
                base,
                offset,
                shape,
                elem_ty,
            } => Place::BufferSlice {
                base: Box::new(self.rewrite_specialized_body(base, view_params)),
                offset: Box::new(self.rewrite_specialized_body(offset, view_params)),
                shape: shape.clone(),
                elem_ty: elem_ty.clone(),
            },
            Place::LocalArray { id, shape, elem_ty } => Place::LocalArray {
                id: *id,
                shape: shape.clone(),
                elem_ty: elem_ty.clone(),
            },
        }
    }

    // =========================================================================
    // View expression resolution
    // =========================================================================

    /// Try to resolve an expression to a `StorageView` — i.e. determine whether
    /// it refers (directly or via slicing) to a known buffer-backed view.
    ///
    /// Returns `Some(StorageView)` for:
    /// - `Var(sym)` where sym is in `view_params`
    /// - `_w_intrinsic_slice(expr, start, end)` where expr resolves to a view
    fn try_resolve_view_expr(
        &mut self,
        term: &Term,
        view_params: &HashMap<SymbolId, ViewProv>,
    ) -> Option<crate::tlc::StorageView> {
        match &term.kind {
            TermKind::Var(VarRef::Symbol(sym)) => {
                let prov = view_params.get(sym)?;
                Some(crate::tlc::StorageView {
                    binding: prov.binding,
                    offset: Box::new(prov.offset.clone()),
                    len: Box::new(prov.len.clone()),
                    elem_ty: prov.elem_ty.clone(),
                })
            }
            TermKind::App { func, args } => {
                // Check for _w_intrinsic_slice(expr, start, end). The
                // func may arrive as either `Var(Symbol)` (synthesized
                // paths) or `Var(Builtin)` (post-NameResolution user
                // code) — `var_term_builtin_id` handles both.
                if var_term_builtin_id(func, &self.symbols) == Some(catalog().known().slice) {
                    if args.len() == 3 {
                        let parent = self.try_resolve_view_expr(&args[0], view_params)?;
                        let span = term.span;
                        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);

                        // Rewrite start and end so any nested view refs are resolved
                        let start = self.rewrite_specialized_body(&args[1], view_params);
                        let end = self.rewrite_specialized_body(&args[2], view_params);

                        // new_offset = parent.offset + start
                        let new_offset = self.make_binop_app(
                            ast::BinaryOp { op: "+".to_string() },
                            *parent.offset,
                            start.clone(),
                            u32_ty.clone(),
                            span,
                        );

                        // new_len = end - start
                        let new_len = self.make_binop_app(
                            ast::BinaryOp { op: "-".to_string() },
                            end,
                            start,
                            u32_ty,
                            span,
                        );

                        return Some(crate::tlc::StorageView {
                            binding: parent.binding,
                            offset: Box::new(new_offset),
                            len: Box::new(new_len),
                            elem_ty: parent.elem_ty,
                        });
                    }
                }
                None
            }
            _ => None,
        }
    }

    // =========================================================================
    // Helper term constructors
    // =========================================================================

    fn make_int_lit(&mut self, value: &str, ty: Type<TypeName>, span: Span) -> Term {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind: TermKind::IntLit(value.to_string()),
        }
    }

    fn make_app(
        &mut self,
        intrinsic_name: &str,
        args: Vec<Term>,
        ret_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Catalog targets are emitted as `Var(Builtin(id))`. The assert
        // requires single-overload entries — multi-overload targets
        // need an explicit overload index from the caller.
        let func_var = if let Some(def) = catalog().lookup_by_any_name(intrinsic_name) {
            assert_eq!(
                def.overloads().len(),
                1,
                "buffer_specialize::make_app({:?}) targets a multi-overload catalog entry; \
                 caller must specify overload_idx explicitly",
                intrinsic_name
            );
            VarRef::Builtin {
                id: def.id,
                overload_idx: 0,
            }
        } else {
            VarRef::Symbol(self.symbols.alloc(intrinsic_name.to_string()))
        };
        self.build_app_call(func_var, args, ret_ty, span)
    }

    /// Shared App-builder for `make_app` (and previously `make_app_by_id`
    /// before storage-intrinsic emission moved to `tlc::storage_index_call` /
    /// `tlc::storage_len_call`).
    fn build_app_call(
        &mut self,
        func_var: crate::tlc::VarRef,
        args: Vec<Term>,
        ret_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_term = Term {
            id: self.term_ids.next_id(),
            ty: ret_ty.clone(), // approximate
            span,
            kind: TermKind::Var(func_var),
        };

        if args.is_empty() {
            func_term
        } else {
            Term {
                id: self.term_ids.next_id(),
                ty: ret_ty,
                span,
                kind: TermKind::App {
                    func: Box::new(func_term),
                    args,
                },
            }
        }
    }

    fn make_binop_app(
        &mut self,
        op: ast::BinaryOp,
        lhs: Term,
        rhs: Term,
        ret_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let op_term = Term {
            id: self.term_ids.next_id(),
            ty: ret_ty.clone(),
            span,
            kind: TermKind::BinOp(op),
        };
        Term {
            id: self.term_ids.next_id(),
            ty: ret_ty,
            span,
            kind: TermKind::App {
                func: Box::new(op_term),
                args: vec![lhs, rhs],
            },
        }
    }
}

/// Wrap a body term in a single flat lambda for the given parameter list.
fn wrap_in_lambdas(body: Term, params: &[(SymbolId, Type<TypeName>)], term_ids: &mut TermIdSource) -> Term {
    if params.is_empty() {
        return body;
    }

    let ret_ty = body.ty.clone();
    let mut lam_ty = ret_ty.clone();
    for (_, ty) in params.iter().rev() {
        lam_ty = Type::Constructed(TypeName::Arrow, vec![ty.clone(), lam_ty]);
    }
    Term {
        id: term_ids.next_id(),
        ty: lam_ty,
        span: body.span,
        kind: TermKind::Lambda(Lambda {
            params: params.to_vec(),
            body: Box::new(body),
            ret_ty,
        }),
    }
}

#[cfg(test)]
#[path = "buffer_specialize_tests.rs"]
mod buffer_specialize_tests;
