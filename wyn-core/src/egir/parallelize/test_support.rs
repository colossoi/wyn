use super::*;

pub(crate) const FILTER_SCAN_GROUPS: u32 = model::FILTER_SCAN_GROUPS;
pub(crate) const REDUCE_PHASE1_WIDTH: u32 = model::REDUCE_PHASE1_WIDTH;

pub(crate) fn planned_callable_names(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> std::result::Result<Vec<String>, String> {
    let plan = build_parallel_plan(inner, effect_ids).map_err(|error| error.to_string())?;
    Ok(plan.generated_callables().map(|function| function.name.clone()).collect())
}
