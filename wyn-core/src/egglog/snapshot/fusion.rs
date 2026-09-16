//! Role-tagged uses and observable outputs needed only by fusion.
use super::super::visit::OperandRole;
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::egglog) enum Role {
    Input,
    Capture,
    Neutral,
    Argument,
    Length,
}
impl Role {
    pub(in crate::egglog) fn egglog(self) -> &'static str {
        match self {
            Self::Input => "(Input)",
            Self::Capture => "(Capture)",
            Self::Neutral => "(Neutral)",
            Self::Argument => "(Argument)",
            Self::Length => "(Length)",
        }
    }
}
impl From<OperandRole> for Role {
    fn from(role: OperandRole) -> Self {
        match role {
            OperandRole::Input => Self::Input,
            OperandRole::Capture => Self::Capture,
            OperandRole::Neutral => Self::Neutral,
            OperandRole::Argument => Self::Argument,
        }
    }
}

pub(in crate::egglog) struct Fusion {
    pub execution: Snapshot,
    pub uses: BTreeSet<(OperationId, OperationId, Role)>,
    pub observed: BTreeSet<OperationId>,
}

pub(in crate::egglog) fn analyze(data: &AssociatedData) -> Fusion {
    let mut analysis = Analysis::new(data);
    let visitor = &mut analysis.visitor;
    let mut uses = BTreeSet::new();
    for &consumer in &analysis.snapshot.live {
        let kind = &data.operations[consumer].kind;
        let mut values = BTreeMap::<Role, References>::new();
        let mut region_uses = BTreeSet::new();
        match kind {
            OperationKind::Call { function, args }
                if super::super::fusion::length_source(data, kind).is_some() =>
            {
                values.insert(Role::Argument, visitor.expression(*function));
                values.insert(Role::Length, visitor.expressions(args));
            }
            _ => {
                kind.for_each_operand(&mut |operand| match operand {
                    Operand::Value(role, e) => {
                        let refs = visitor.expression(e);
                        values.entry(role.into()).or_default().extend(&refs, &mut visitor.sets);
                    }
                    Operand::Region(r) => {
                        region_uses.insert(r);
                    }
                });
            }
        }
        for (role, refs) in values {
            let deps = refs.dependencies(&analysis.external, &mut visitor.sets);
            uses.extend(visitor.sets.iter(deps).map(|p| (OperationId::from(p), consumer, role)));
        }
        for region in region_uses {
            uses.extend(
                visitor
                    .sets
                    .iter(analysis.external[&region])
                    .map(|p| (OperationId::from(p), consumer, Role::Capture)),
            );
        }
    }
    let mut observed = BTreeSet::new();
    for region in analysis.active {
        let deps = analysis.results[&region].dependencies(&analysis.external, &mut visitor.sets);
        observed.extend(visitor.sets.iter(deps).map(OperationId::from));
    }
    Fusion {
        execution: analysis.snapshot,
        uses,
        observed,
    }
}
