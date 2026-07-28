//! Structural capability contract for physical Screma recipes.
//!
//! Graph-dependent checks remain with each recipe analyzer and may still
//! select serial fallback.

use crate::egir::soac::screma;
use crate::egir::types::WynSoacPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScremaShape {
    Map,
    Reduce,
    Scan,
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScremaRecipeClass {
    Map,
    Reduce,
    Scan,
    Serial,
}

#[derive(Clone, Copy)]
struct RecipeFacts {
    shape: ScremaShape,
    identity_post: bool,
    parallel_scan_post: bool,
    closed_combines: bool,
    input_count: usize,
    operator_count: usize,
    routed_maps: bool,
    fresh_operators: bool,
    routed_operators: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScremaRecipeCapabilities {
    shape: ScremaShape,
    recipe: ScremaRecipeClass,
}

impl ScremaRecipeCapabilities {
    pub(super) fn analyze<P: WynSoacPhase>(op: &screma::Op<P>) -> Self {
        let shape = if op.is_map() {
            ScremaShape::Map
        } else if op.is_reduce() {
            ScremaShape::Reduce
        } else if op.is_scan_only() {
            ScremaShape::Scan
        } else {
            ScremaShape::Mixed
        };
        classify(RecipeFacts {
            shape,
            identity_post: !op.has_post_map() && op.hidden_scan_outputs.is_empty(),
            parallel_scan_post: op.lanes().maps.is_empty()
                && !op.post_maps.is_empty()
                && op.post_maps.iter().all(|map| {
                    map.input_indices == [screma::InputId(0)] && map.destination.is_output_view()
                })
                && op.hidden_scan_outputs == [0],
            // Associativity belongs to the Screma operator contract. Ordered
            // recipes deliberately do not require commutativity.
            closed_combines: op.operators().iter().all(|op| op.combine.captures.is_empty()),
            input_count: op.lanes().inputs.len(),
            operator_count: op.operators().len(),
            routed_maps: op.lanes().maps.iter().all(|map| map.destination.is_output_view()),
            fresh_operators: op.operators().iter().all(|op| op.destination.is_unplaced_fresh()),
            routed_operators: op.operators().iter().all(|op| op.destination.is_output_view()),
        })
    }

    pub(super) fn shape(self) -> ScremaShape {
        self.shape
    }

    pub(super) fn recipe_class(self) -> ScremaRecipeClass {
        self.recipe
    }
}

fn classify(facts: RecipeFacts) -> ScremaRecipeCapabilities {
    let recipe = match facts.shape {
        ScremaShape::Map if facts.identity_post => ScremaRecipeClass::Map,
        ScremaShape::Reduce
            if facts.identity_post
                && facts.closed_combines
                && facts.input_count != 0
                && facts.routed_maps
                && facts.fresh_operators =>
        {
            ScremaRecipeClass::Reduce
        }
        ScremaShape::Scan
            if (facts.identity_post || facts.parallel_scan_post)
                && facts.closed_combines
                && facts.input_count == 1
                && facts.operator_count == 1
                && facts.routed_maps
                && (facts.routed_operators || facts.parallel_scan_post) =>
        {
            ScremaRecipeClass::Scan
        }
        _ => ScremaRecipeClass::Serial,
    };
    ScremaRecipeCapabilities {
        shape: facts.shape,
        recipe,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    enum Mutation {
        None,
        ParallelScanPost,
        UnsupportedPost,
        CapturedCombine,
        NoInput,
        SecondInput,
        SecondOperator,
        UnroutedMap,
        RoutedReduction,
        UnroutedScan,
    }

    fn facts(shape: ScremaShape, mutation: Mutation) -> RecipeFacts {
        let mut facts = RecipeFacts {
            shape,
            identity_post: true,
            parallel_scan_post: false,
            closed_combines: true,
            input_count: 1,
            operator_count: usize::from(shape != ScremaShape::Map),
            routed_maps: true,
            fresh_operators: shape == ScremaShape::Reduce,
            routed_operators: shape == ScremaShape::Scan,
        };
        match mutation {
            Mutation::None => {}
            Mutation::ParallelScanPost => {
                facts.identity_post = false;
                facts.parallel_scan_post = true;
                facts.routed_operators = false;
            }
            Mutation::UnsupportedPost => facts.identity_post = false,
            Mutation::CapturedCombine => facts.closed_combines = false,
            Mutation::NoInput => facts.input_count = 0,
            Mutation::SecondInput => facts.input_count = 2,
            Mutation::SecondOperator => facts.operator_count = 2,
            Mutation::UnroutedMap => facts.routed_maps = false,
            Mutation::RoutedReduction => facts.fresh_operators = false,
            Mutation::UnroutedScan => facts.routed_operators = false,
        }
        facts
    }

    #[test]
    fn canonical_screma_recipe_contract_is_table_driven() {
        use Mutation::*;
        use ScremaRecipeClass::{Map, Reduce, Scan, Serial};
        use ScremaShape::{Map as MapShape, Mixed, Reduce as ReduceShape, Scan as ScanShape};

        #[rustfmt::skip]
        let cases = [
            ("map",                       MapShape,    None,             Map),
            ("map post",                  MapShape,    UnsupportedPost,  Serial),
            ("reduction",                 ReduceShape, None,             Reduce),
            ("reduction combine capture", ReduceShape, CapturedCombine,  Serial),
            ("reduction without input",   ReduceShape, NoInput,          Serial),
            ("reduction map routing",     ReduceShape, UnroutedMap,      Serial),
            ("reduction output routing",  ReduceShape, RoutedReduction,  Serial),
            ("scan",                      ScanShape,   None,             Scan),
            ("multiple scans",            ScanShape,   SecondOperator,   Serial),
            ("scan combine capture",      ScanShape,   CapturedCombine,  Serial),
            ("scan input arity",          ScanShape,   SecondInput,      Serial),
            ("scan map routing",          ScanShape,   UnroutedMap,      Serial),
            ("scan output routing",       ScanShape,   UnroutedScan,     Serial),
            ("scan post-map",             ScanShape,   ParallelScanPost, Scan),
            ("unsupported scan post-map", ScanShape,   UnsupportedPost,  Serial),
            ("mixed operators",           Mixed,       None,             Serial),
        ];

        for (name, shape, mutation, expected) in cases {
            let capabilities = classify(facts(shape, mutation));
            assert_eq!(capabilities.shape(), shape, "{name}: shape");
            assert_eq!(capabilities.recipe_class(), expected, "{name}: recipe");
        }
    }
}
