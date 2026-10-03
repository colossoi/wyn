//! Resolve target support once while constructing the final host program.
use super::{
    to_ssa::{host, read::Facts},
    OptimizeError, Optimized, Program,
};

pub(super) struct Capture {
    pub value: crate::host::ScalarExpr,
    pub ty: crate::host::ScalarType,
    pub stages: Vec<egglog_engine::Value>,
}

pub(super) fn prepare(program: &mut Program<'_, Optimized>) -> Result<(), OptimizeError> {
    let selected = &program.stage.selected;
    let mut captures = crate::LookupMap::<egglog_engine::TermId, Capture>::default();
    super::query::Query(&program.stage.scalars).for_each("ScalarPhaseSource", |row| {
        let Some(&root) = selected.roots.get(&(row[1], row[2])) else {
            return Err(OptimizeError::Output("phase scalar root missing".into()));
        };
        let mut pending = vec![root];
        let mut seen = crate::LookupSet::default();
        while let Some(term) = pending.pop() {
            if !seen.insert(term) {
                continue;
            }
            let (name, fields) = selected.app(term)?;
            if let Some(capture) = captures.get_mut(&term) {
                if !capture.stages.contains(&row[0]) {
                    capture.stages.push(row[0]);
                }
                continue;
            }
            if super::query::Query(&program.stage.scalars)
                .contains("ScalarPreferHost", (selected.values[term],))?
            {
                if let Some(ty) =
                    (Facts { program }).ty(selected.values[fields[1]]).and_then(host::scalar_type)
                {
                    match host::selected(program, term) {
                        Ok(value) => {
                            captures.insert(
                                term,
                                Capture {
                                    value,
                                    ty,
                                    stages: vec![row[0]],
                                },
                            );
                            continue;
                        }
                        Err(host::Error::Unsupported) => {}
                        Err(host::Error::Invalid(error)) => return Err(error),
                    }
                }
            }
            pending.extend(super::scalar::extract::operands(name, fields));
        }
        Ok(())
    })?;
    program.stage.captures = captures;
    let mut roots = Vec::new();
    program.graph.constructor_enodes("HostTarget", |row| roots.push(row.children[0]))?;
    roots.sort();
    roots.dedup();
    for root in roots {
        let executor = super::abi::required(&program.graph, "PreferredExecutor", (root,))?;
        if super::abi::fields(&program.graph, "GpuExecutor", executor)?.is_some() {
            continue;
        }
        if super::abi::fields(&program.graph, "CpuExecutor", executor)?.is_none() {
            return Err(OptimizeError::Output("unknown preferred executor".into()));
        }
        let mut targets = Vec::new();
        program.graph.constructor_enodes("HostTarget", |row| {
            if row.children[0] == root {
                targets.push((row.children[1], row.children[2]));
            }
        })?;
        let mut lowered = Vec::new();
        for (source, context) in targets {
            if (Facts { program }).source_type(source).and_then(host::scalar_type).is_none() {
                lowered.clear();
                break;
            }
            match host::expression(program, context, source) {
                Ok(value) => lowered.push((source, value)),
                Err(host::Error::Unsupported) => {
                    lowered.clear();
                    break;
                }
                Err(host::Error::Invalid(error)) => return Err(error),
            }
        }
        for (source, value) in lowered {
            program.stage.host.insert((root, source), value);
        }
    }
    Ok(())
}
