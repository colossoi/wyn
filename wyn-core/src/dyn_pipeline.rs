//! Test-only dynamic TLC pass driver.
//!
//! The production pipeline encodes pass order in the typestate (`TlcPartialEvaled
//! → TlcSoaNormalized → …`), which is great for guaranteeing the *real* order
//! but makes it impossible to try a *different* order without rewriting the
//! state machine. This driver throws that away: it holds the whole state bag in
//! one mutable struct and exposes one method per pass that calls the underlying
//! `tlc::<pass>::run` directly, so a test can sequence passes in any order.
//!
//! It is NOT a substitute for the typestate — it deliberately drops the
//! compile-time ordering guarantees so we can prototype reorderings (e.g. "what
//! if defunctionalize + monomorphize ran before fusion?"). Fallible passes
//! `.expect(...)` rather than propagate, since a panic *is* the signal we want
//! when probing an illegal order.

use crate::tlc::{self, Program};
use crate::{cached_compiler_init, types, Compiler, IdSource, LookupMap, LookupSet, SymbolId, TypeTable};

/// All state any TLC pass needs, in one flat bag. Mirrors the union of
/// `TlcEarlyInner` / `TlcDefunctionalized` / `TlcLateInner` fields.
#[allow(dead_code)] // exploratory harness: not every field/pass is wired into every probe
pub(crate) struct Driver {
    pub tlc: Program,
    pub type_table: TypeTable,
    pub known_defs: LookupSet<String>,
    pub schemes: LookupMap<SymbolId, types::TypeScheme>,
    pub auto_storage_binding_ids: IdSource<u32>,
    pub disable_parallelize: bool,
}

#[allow(dead_code)] // exploratory harness: passes are wired in per-probe
impl Driver {
    /// Run the front-end (parse → resolve → fold → type-check → to_tlc →
    /// pin_entry_regions) and capture the state bag. The result sits at the
    /// `TlcRegionsPinned` boundary — the same starting point `optimize_for_test`
    /// uses — but with no typestate constraining what runs next.
    pub fn from_source(src: &str) -> Self {
        let (mut node_counter, mut module_manager) = cached_compiler_init();
        let type_checked = Compiler::parse(src, &mut node_counter)
            .expect("parse")
            .resolve(&module_manager)
            .expect("resolve")
            .fold_ast_constants()
            .type_check(&mut module_manager)
            .expect("type_check");
        let pinned = type_checked
            .to_tlc(&module_manager, false)
            .pin_entry_regions()
            .expect("pin_entry_regions");
        let inner = pinned.0;
        Driver {
            tlc: inner.tlc,
            type_table: inner.type_table,
            known_defs: inner.known_defs,
            schemes: inner.schemes,
            auto_storage_binding_ids: inner.auto_storage_binding_ids,
            disable_parallelize: true,
        }
    }

    // --- Program → Program passes ------------------------------------------

    pub fn partial_eval(mut self) -> Self {
        self.tlc = tlc::partial_eval::PartialEvaluator::partial_eval(self.tlc);
        self
    }

    /// SoA transform + SOAC normalization (the `normalize_soacs` typestate step).
    pub fn normalize_soacs(mut self) -> Self {
        self.tlc = tlc::soa::run(self.tlc);
        self
    }

    pub fn force_inline_soac_helpers(mut self) -> Self {
        self.tlc = tlc::inline::run_force_soac_helpers(self.tlc);
        self
    }

    pub fn inline_small(mut self) -> Self {
        self.tlc = tlc::inline::run_small(self.tlc);
        self
    }

    /// Fold compiler-generated lambda defs back at call sites (`run_large`).
    pub fn fold_generated_lambdas(mut self) -> Self {
        self.tlc = tlc::inline::run_large(self.tlc);
        self
    }

    pub fn reachable(mut self) -> Self {
        self.tlc = tlc::inline::run_reachable(self.tlc);
        self
    }

    pub fn rep_specialize(mut self) -> Self {
        self.tlc = tlc::rep_specialize::run(self.tlc);
        self
    }

    pub fn if_over_producer(mut self) -> Self {
        self.tlc = tlc::if_over_producer::run(self.tlc);
        self
    }

    pub fn fuse_maps(mut self) -> Self {
        self.tlc = tlc::fusion::run(self.tlc);
        self
    }

    pub fn expose_entry_producers(mut self) -> Self {
        self.tlc = tlc::materialize_entry_soacs::run(self.tlc);
        self
    }

    pub fn fuse_static_indices(mut self) -> Self {
        self.tlc = tlc::static_index_fusion::run(self.tlc);
        self
    }

    pub fn float_runtime_index_producers(mut self) -> Self {
        self.tlc = tlc::runtime_index_producers::run(self.tlc);
        self
    }

    pub fn lift_gathers(mut self) -> Self {
        self.tlc = tlc::lift_gathers::run(self.tlc, &mut self.auto_storage_binding_ids);
        self
    }

    // --- Multi-pass / state-threading steps --------------------------------

    /// Three-phase defunctionalization. Preserves `SoacOp` envelopes (operator
    /// lambdas stay inline with captures threaded) — it does NOT lower SOACs to
    /// loops, so fusion can still run after this.
    pub fn defunctionalize(mut self) -> Self {
        let (cc, info) = tlc::closure_convert::run(self.tlc, &self.known_defs);
        let hof_free = tlc::hof_specialize::run(cc, &info);
        self.tlc = tlc::closure_calls_lower::run(hof_free, &info);
        self
    }

    /// Specialize polymorphic intrinsics, then monomorphize user functions
    /// (region-specialized). After this no `Type::Variable` remains.
    pub fn monomorphize(mut self) -> Self {
        let specialized = tlc::specialize::run(self.tlc);
        self.tlc = tlc::monomorphize::run(specialized, &self.schemes);
        self
    }

    // --- Fallible passes (panic on error — that's the probe signal) ---------

    pub fn normalize_outputs(mut self) -> Self {
        self.tlc = tlc::normalize_outputs::run(self.tlc).expect("normalize_outputs");
        self
    }

    pub fn apply_ownership(mut self) -> Self {
        self.tlc = tlc::ownership::apply_ownership(self.tlc).expect("apply_ownership");
        self
    }

    /// Parallelize SOACs. Discards the produced `PipelineDescriptor`/plans —
    /// this driver only carries the TLC `Program` forward.
    pub fn parallelize(mut self) -> Self {
        self.tlc = tlc::if_over_producer::run(self.tlc);
        let result = tlc::parallelize::run(self.tlc, self.disable_parallelize, &mut self.auto_storage_binding_ids)
            .expect("parallelize");
        self.tlc = result.program;
        self
    }

    // --- Inspection --------------------------------------------------------

    pub fn program(&self) -> &Program {
        &self.tlc
    }

    /// Names of all defs (for assertions / debugging).
    pub fn def_names(&self) -> Vec<String> {
        self.tlc
            .defs
            .iter()
            .map(|d| self.tlc.symbols.get(d.name).cloned().unwrap_or_default())
            .collect()
    }
}

/// Shared test helper: run the front-end + the canonical `optimize_for_test`
/// pipeline to `TlcReachable`. Consolidates the per-pass test files' previously-
/// bespoke front-end-plus-chain helpers, so a pipeline reorder only has to touch
/// `optimize_for_test`, not a dozen copies. `disable_parallelize = true` is the
/// common case; parallelizer / egir tests pass `false`.
pub(crate) fn compile_to_reachable(src: &str, disable_parallelize: bool) -> crate::TlcReachable {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    let type_checked = Compiler::parse(src, &mut node_counter)
        .expect("parse")
        .resolve(&module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked
        .to_tlc(&module_manager, false)
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .optimize_for_test(disable_parallelize)
}

/// As [`compile_to_reachable`], returning just the final TLC program.
pub(crate) fn compile_to_tlc_program(src: &str) -> Program {
    compile_to_reachable(src, true).0.tlc
}

#[cfg(test)]
#[path = "dyn_pipeline_tests.rs"]
mod dyn_pipeline_tests;
