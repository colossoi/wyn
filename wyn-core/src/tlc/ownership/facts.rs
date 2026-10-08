//! Stable ownership permission, independent of use order and SOAC identity.

use super::analysis::{build, Origin};
use crate::tlc::{Family, Program};
use crate::{LookupSet, SymbolId};

/// Permission to reuse a value's backing storage. This is not a claim that the
/// value is dead, nor that a particular operation can overwrite it safely.
#[derive(Debug, Clone, Default)]
pub struct OwnershipFacts {
    pub(crate) reusable: LookupSet<SymbolId>,
}

pub(super) fn collect<Tag, F: Family, G>(program: &Program<Tag, F, G>) -> OwnershipFacts {
    let model = build(program);
    let reusable = model
        .var_to_owner
        .iter()
        .filter_map(|(&symbol, &owner)| {
            matches!(
                model.origin(owner),
                Some(Origin::Fresh | Origin::UniqueParam | Origin::Entry)
            )
            .then_some(symbol)
        })
        .collect();
    OwnershipFacts { reusable }
}
