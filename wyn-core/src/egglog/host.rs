//! Resolve target support once while constructing the final host program.
use super::{facts::Facts, OptimizeError, Optimized, Program};

pub(super) mod lower;

pub(super) struct Capture {
    pub value: crate::host::ScalarExpr,
    pub ty: crate::host::ScalarType,
    pub stages: Vec<egglog_engine::Value>,
}

pub(super) fn prepare(program: &mut Program<'_, Optimized>) -> Result<(), OptimizeError> {
    let selected = &program.stage.selected;
    let mut captures = crate::LookupMap::<egglog_engine::TermId, Capture>::default();
    super::query::Query(&program.stage.scalars).for_each("HostEvalPhaseRoot", |row| {
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
                .contains("HostEvalPreferred", (selected.values[term],))?
            {
                if let Some(ty) =
                    (Facts { program }).ty(selected.values[fields[1]]).and_then(lower::scalar_type)
                {
                    match lower::selected(program, term, false) {
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
                        Err(lower::Error::Unsupported) => {}
                        Err(lower::Error::Invalid(error)) => return Err(error),
                    }
                }
            }
            pending.extend(super::scalar::extract::operands(name, fields));
        }
        Ok(())
    })?;
    program.stage.captures = captures;
    let mut targets = std::collections::BTreeMap::<_, Vec<_>>::new();
    program.graph.constructor_enodes("HostTarget", |row| {
        targets.entry(row.children[0]).or_default().push((row.children[1], row.children[2]));
    })?;
    for (root, targets) in targets {
        let mut lowered = Vec::new();
        for (source, context) in targets {
            if (Facts { program }).source_type(source).and_then(lower::scalar_type).is_none() {
                lowered.clear();
                break;
            }
            match lower::expression(program, context, source) {
                Ok(value) => lowered.push((source, value)),
                Err(lower::Error::Unsupported) => {
                    lowered.clear();
                    break;
                }
                Err(lower::Error::Invalid(error)) => return Err(error),
            }
        }
        for (source, value) in lowered {
            program.stage.host.insert((root, source), value);
        }
    }
    Ok(())
}
