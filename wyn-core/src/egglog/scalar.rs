//! Demand-driven scalar admission followed by an egglog optimization fixed point.
//! The structural graph remains authoritative for effects and dispatch placement.
use super::query::Query;
use super::{parse_program, source, timing, OptimizeError, ScalarOptimization};
use crate::LookupMap;
use egglog_engine::{EGraph, TermDag, TermId, Value};

pub(super) mod extract;
mod fold;
mod import;
pub(super) mod placement;
mod read;
mod scopes;
use import::Importer;

#[cfg(test)]
#[path = "scalar/analysis_tests.rs"]
mod analysis_tests;

/// Egglog's own extracted DAG, retained unchanged for direct SSA emission.
pub(super) struct Selected {
    pub dag: TermDag,
    pub values: Vec<Value>,
    pub roots: LookupMap<(Value, Value), TermId>,
}

const RULES: &str = concat!(
    include_str!("scalar/schema.egg"),
    "\n",
    include_str!("scalar/analysis.egg"),
    "\n",
    include_str!("scalar/inline.egg"),
    "\n",
    include_str!("scalar/micro.egg"),
);

pub(super) fn run(
    graph: &mut EGraph,
    identities: &source::Identities<'_>,
    policy: ScalarOptimization,
) -> Result<(Selected, EGraph), OptimizeError> {
    let _timing = timing::span("egglog scalar optimization");
    fold::register(graph);
    graph.parse_and_run_program(Some("scalar rules".into()), RULES)?;
    graph.parse_and_run_program(
        Some("scalar admission".into()),
        include_str!("scalar/admission.egg"),
    )?;
    graph.parse_and_run_program(Some("host preferences".into()), include_str!("host.egg"))?;
    graph.parse_and_run_program(
        Some("publication dependencies".into()),
        include_str!("publication.egg"),
    )?;
    graph.parse_and_run_program(None, "(run-schedule (saturate scalar-admission) (saturate scalar-contexts) (saturate scalar-demands) (saturate host-targets) (run host-captures) (saturate scalar-import) (run readout) (saturate physical-abi) (run physical-bindings) (saturate publication-uses) (saturate (seq publication-access physical-abi)) (saturate publication-order))")?;
    let schedule = parse_program(
        "scalar fixed point",
        match policy {
            ScalarOptimization::Basic => include_str!("scalar/basic.egg"),
            ScalarOptimization::Full => include_str!("scalar/schedule.egg"),
        },
    )?;
    let facts = Query(graph);
    for &region in identities.scopes.keys() {
        facts.required("ScalarScopeContext", (region,))?;
    }
    let mut scalars = timing::time("egglog scalar / graph projection", || {
        // Native identities are shared with planning; arithmetic optimization
        // runs in its own graph without structural rule matches.
        let mut scalars = graph.clone();
        for name in scalars.get_function_names() {
            if !name.starts_with("Scalar")
                && !matches!(
                    name.as_str(),
                    "RegionId"
                        | "OperationId"
                        | "TypeId"
                        | "SourceTerm"
                        | "SourceFormal"
                        | "SourceGlobal"
                        | "SourceArrayAtom"
                        | "SourceProjected"
                        | "SourceInvocation"
                        | "SourceLoop"
                        | "SourceReadOnly"
                        | "SourceRegionReadOnly"
                        | "FusionSource"
                        | "Joined"
                )
            {
                scalars.clear_function(&name)?;
            }
        }
        Ok::<_, OptimizeError>(scalars)
    })?;
    let mut cache = LookupMap::default();
    let mut templates = LookupMap::default();
    timing::time("egglog scalar / admission", || {
        scalars.update(|sink| {
            let mut importer = Importer::new(sink, identities, facts, &mut cache, &mut templates, policy);
            Ok(facts.for_each("ScalarImportRoot", |r| {
                importer.root(r[0], r[1], r[2], graph.value_to_base::<bool>(r[3]))
            }))
        })?
    })?;
    timing::time("egglog scalar / fixed point", || scalars.run_program(schedule))?;
    timing::time("egglog scalar / effect summaries", || {
        scalars.parse_and_run_program(None, "(run-schedule (saturate scalar-effects))")
    })?;
    scalars.parse_and_run_program(
        Some("scalar host preferences".into()),
        include_str!("scalar/host.egg"),
    )?;
    scalars.clear_function("ScalarActive")?;
    let selected = placement::select(&mut scalars, policy)?;
    if timing::enabled() {
        eprintln!("egglog scalar admitted values: {}", cache.len());
    }
    Ok((selected, scalars))
}
