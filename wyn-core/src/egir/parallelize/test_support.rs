use super::*;

pub(crate) const FILTER_SCAN_GROUPS: u32 = model::FILTER_SCAN_GROUPS;
pub(crate) const REDUCE_PHASE1_WIDTH: u32 = model::REDUCE_PHASE1_WIDTH;

pub(crate) fn preflight_fallback_reasons(
    inner: &AllocatedProgram,
) -> std::result::Result<Vec<&'static str>, String> {
    planning::preflight_fallback_reasons(inner)
        .map(|reasons| reasons.into_iter().map(fallback_name).collect())
        .map_err(|error| error.to_string())
}

pub(crate) fn planned_callable_names(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> std::result::Result<Vec<String>, String> {
    let plan = build_parallel_plan(inner, effect_ids).map_err(|error| error.to_string())?;
    Ok(plan.generated_callables().map(|function| function.name.clone()).collect())
}

fn fallback_name(reason: FallbackReason) -> &'static str {
    match reason {
        FallbackReason::SequentialPolicy => "sequential policy",
        FallbackReason::UnsupportedPlacement => "unsupported placement",
        FallbackReason::UnsupportedCaptures => "unsupported captures",
        FallbackReason::UnsupportedViewShape => "unsupported view shape",
        FallbackReason::UnsupportedDestination => "unsupported destination",
        FallbackReason::UnsupportedScratchLayout => "unsupported scratch layout",
        FallbackReason::UnsupportedOperationShape => "unsupported operation shape",
    }
}
