//! Run execution placement and structural dispatch planning on native facts.
use super::{timing, OptimizeError};
use crate::PipelineTopologyPolicy;
use egglog_engine::{EGraph, RawValues, Write};

pub(super) const RULES: &str = concat!(
    include_str!("planning.egg"),
    "\n",
    include_str!("execution.egg"),
    "\n",
    include_str!("schedule.egg"),
    "\n",
    include_str!("residency.egg"),
    "\n",
    include_str!("allocation.egg"),
    "\n",
    include_str!("reuse.egg"),
    "\n",
    include_str!("dispatch.egg"),
    "\n",
    include_str!("epilogues.egg"),
    "\n",
);
pub(super) const KEYS: &str = "(datatype TypeKey (TypeId i64))\n";

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("planning keys".into()), KEYS)?;
    graph.parse_and_run_program(Some("structural planning rules".into()), RULES)?;
    Ok(())
}

pub(super) fn place(graph: &mut EGraph, topology: PipelineTopologyPolicy) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("planning bridge".into()),
        concat!(
            include_str!("planning/import.egg"),
            "\n",
            include_str!("planning/abi.egg"),
            "\n",
            include_str!("planning/representation.egg")
        ),
    )?;
    let policy = match topology {
        PipelineTopologyPolicy::AllowGenerated => "AllowGeneratedTopology",
        PipelineTopologyPolicy::AuthoredOnly => "AuthoredTopology",
    };
    graph.update(|mut sink| sink.add(policy, RawValues(Vec::new())))?;
    timing::time("egglog placement / import", || {
        graph.parse_and_run_program(None, "(run-schedule (saturate planning-import))")
    })?;
    timing::time("egglog placement / order", || {
        graph.parse_and_run_program(Some("execution order".into()), include_str!("planning/order.egg"))
    })?;
    let mut pending = false;
    graph.constructor_enodes("OrderPending", |_| pending = true)?;
    if pending {
        return Err(OptimizeError::Extraction(
            "cycle in the fused execution graph".into(),
        ));
    }
    graph.parse_and_run_program(
        Some("execution placement".into()),
        include_str!("planning/place.egg"),
    )?;
    Ok(())
}

pub(super) fn schedule(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("structural scheduling".into()),
        include_str!("planning/run.egg"),
    )?;
    if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
        for output in graph.parse_and_run_program(None, "(print-stats) (print-size)")? {
            eprintln!("{output}");
        }
    }
    Ok(())
}
