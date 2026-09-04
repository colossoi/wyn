//! Shared test entry point for the TLC pipeline.
//!
//! One front-end + canonical TLC chain that every per-pass test module
//! calls, so a pipeline reorder only has to touch one helper rather
//! than a dozen bespoke front-end-plus-chain copies.

use crate::ast_type_holes;
use crate::compile_thru_frontend;
use crate::error::{CompilerError, LoadModulesError};
use crate::frontend::{ParsedModules, WynFrontend};
use crate::optimize_tlc_for_test;
use crate::optimize_tlc_for_test_thru_soac_normalization;
use crate::semantic_modules::SemanticModules;
use crate::tlc;
use crate::{ast::NodeCounter, CompilerOptions};
use wyn_module_graph::{BuildError, ModulePath, PackageIdentity, PackagePlan};

pub(crate) fn load_test_modules(source: &str, options: CompilerOptions) -> ParsedModules {
    try_load_test_modules(source, options).expect("test source graph should load")
}

pub(crate) fn try_load_test_modules(
    source: &str,
    options: CompilerOptions,
) -> Result<ParsedModules, CompilerError> {
    let plan = test_package_plan(source);
    match ParsedModules::load(plan, options) {
        Ok(modules) => Ok(modules),
        Err(LoadModulesError::Prelude(error)) => Err(error),
        Err(LoadModulesError::Modules(failure)) => match failure.error() {
            BuildError::Parse {
                source: CompilerError::ParseError(message, span),
                ..
            } => Err(CompilerError::ParseError(message.clone(), *span)),
            error => Err(CompilerError::Internal(format!(
                "test source module failed to load: {error}"
            ))),
        },
    }
}

pub(crate) fn load_test_modules_with_state(
    source: &str,
    options: CompilerOptions,
    mut node_ids: NodeCounter,
    semantic_modules: SemanticModules,
) -> ParsedModules {
    let plan = test_package_plan(source);
    let mut frontend = WynFrontend::new(&mut node_ids, options);
    let graph = plan.load(&mut frontend).expect("test source graph should load");
    ParsedModules {
        options,
        graph,
        node_ids,
        semantic_modules,
    }
}

fn test_package_plan(source: &str) -> PackagePlan {
    let identity = PackageIdentity::new("test/root", "v0.0.0").expect("valid package identity");
    let root_path = ModulePath::new("main.wyn").expect("valid root module path");
    PackagePlan::single_source(identity, root_path, source)
}

/// Front-end (parse → resolve → type-check → to_tlc → validate_ownership)
/// shared by every `compile_*` helper, so they differ only in how far down the
/// canonical chain they run.
fn front_end(src: &str) -> tlc::stage::OwnershipValidated {
    let type_checked = compile_thru_frontend(src).expect("type_check");
    let program = ast_type_holes::reject_type_holes(type_checked).expect("type holes");
    let program = tlc::lower_from_ast(program).expect("lower_from_ast");
    tlc::validate_ownership(program).expect("validate_ownership")
}

/// Run the front-end + the canonical TLC pipeline to `tlc::stage::Reachable`.
pub(crate) fn compile_to_reachable(src: &str) -> tlc::stage::Reachable {
    optimize_tlc_for_test(front_end(src)).expect("TLC optimization")
}

// Stage-boundary helpers for source-normalization pass tests. Each
// returns the program at the input boundary of the next pass, so a test can run
// that pass itself and observe its effect (rather than re-running the whole
// pipeline, which already ran the pass — and everything after it).

/// Through source-level SOAC ANF normalization, immediately before nested
/// runtime-index producers are floated.
pub(crate) fn compile_thru_expose_producers(src: &str) -> tlc::stage::SoacsAnfNormalized {
    optimize_tlc_for_test_thru_soac_normalization(front_end(src)).expect("TLC optimization")
}

/// Compatibility name for tests whose subject starts at the same SOAC ANF
/// boundary and then runs runtime-index producer exposure directly.
pub(crate) fn compile_thru_static_index(src: &str) -> tlc::stage::SoacsAnfNormalized {
    optimize_tlc_for_test_thru_soac_normalization(front_end(src)).expect("TLC optimization")
}
