//! Public-library integration setup for contracts not exposed by the CLI.
use wyn_core::{
    ast_type_holes, egglog, tlc, CodegenTarget, CompilerOptions, ParsedModules, PipelineTopologyPolicy,
};
use wyn_module_graph::{ModulePath, PackageIdentity, PackagePlan};

pub fn source(source: &str) -> tlc::stage::OwnershipApplied {
    let plan = PackagePlan::single_source(
        PackageIdentity::new("test/integration", "v0.0.0").unwrap(),
        ModulePath::new("main.wyn").unwrap(),
        source,
    );
    let checked =
        ParsedModules::load(plan, CompilerOptions { graphics: true }).unwrap().type_check().unwrap();
    let program = ast_type_holes::reject_type_holes(checked).unwrap();
    let program = tlc::lower_from_ast(program).unwrap();
    let program = tlc::validate_ownership(program).unwrap();
    let program = tlc::partial_eval(program);
    let program = tlc::specialize_types(program).unwrap();
    let program = tlc::inline_unified_helpers(program);
    let program = tlc::extract_stages(program).unwrap();
    let program = tlc::pin_entry_buffers(program).unwrap();
    let program = tlc::monomorphize(program).unwrap();
    let program = tlc::canonicalize_conditional_producers(program);
    let program = tlc::normalize_soacs(program);
    let program = tlc::float_runtime_index_nested_producers(program);
    let program = tlc::defunctionalize(program);
    let program = tlc::fold_generated_lambdas(program);
    tlc::apply_ownership(program)
}

pub fn lower(
    source: &tlc::stage::OwnershipApplied,
    target: CodegenTarget,
) -> wyn_core::ssa::stage::Elaborated {
    lower_with_options(
        source,
        target,
        egglog::ScalarOptimization::Basic,
        PipelineTopologyPolicy::AllowGenerated,
    )
}

pub fn lower_with_options(
    source: &tlc::stage::OwnershipApplied,
    target: CodegenTarget,
    policy: egglog::ScalarOptimization,
    topology: PipelineTopologyPolicy,
) -> wyn_core::ssa::stage::Elaborated {
    let program = egglog::from_tlc(source).unwrap();
    let program = egglog::fuse(program).unwrap();
    let program = egglog::place(program, topology).unwrap();
    let program = egglog::schedule(program).unwrap();
    let program = egglog::prepare_interfaces(program).unwrap();
    let program = egglog::optimize_with_policy(program, policy).unwrap();
    let program = egglog::prepare_host(program).unwrap();
    egglog::to_ssa(program, target).unwrap()
}
