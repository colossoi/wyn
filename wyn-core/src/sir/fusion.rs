//! SIR Fusion Pass
//!
//! Performs SOAC fusion optimizations on SIR programs.
//! These transforms reduce kernel launch overhead and improve memory locality.
//!
//! Supported fusions:
//! - Map-map fusion: `map(f, map(g, xs))` → `map(f ∘ g, xs)`
//! - Zip elimination: `map(f, zip(xs, ys))` → multi-input map
//! - Reduce-map fusion: `reduce(op, ne, map(f, xs))` → `reduce(op, ne, f, xs)`

use std::collections::HashMap;

use super::{Def, Lambda, Map, Op, Program, Soac, Statement, StatementId, VarId};

// =============================================================================
// Fusion Statistics
// =============================================================================

/// Statistics from the fusion pass.
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Number of map-map fusions performed.
    pub map_map_fusions: u32,
    /// Number of zip eliminations performed.
    pub zip_eliminations: u32,
    /// Number of reduce-map fusions performed.
    pub reduce_map_fusions: u32,
    /// Total statements before fusion.
    pub statements_before: u32,
    /// Total statements after fusion.
    pub statements_after: u32,
}

impl FusionStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any fusion was performed.
    pub fn any_fusions(&self) -> bool {
        self.map_map_fusions > 0 || self.zip_eliminations > 0 || self.reduce_map_fusions > 0
    }
}

// =============================================================================
// Fusion Context
// =============================================================================

/// The fusion pass.
pub struct SirFusion {
    /// Statistics about performed fusions.
    stats: FusionStats,
    /// Map from VarId to the statement that produces it.
    /// Used for finding producer-consumer relationships.
    #[allow(dead_code)]
    producers: HashMap<VarId, StatementId>,
}

impl SirFusion {
    /// Create a new fusion pass.
    pub fn new() -> Self {
        SirFusion {
            stats: FusionStats::new(),
            producers: HashMap::new(),
        }
    }

    /// Run fusion on a SIR program.
    ///
    /// Returns the transformed program and fusion statistics.
    /// Currently this is a pass-through that records statistics.
    /// TODO: Implement actual fusion transforms.
    pub fn fuse(&mut self, program: Program) -> (Program, FusionStats) {
        // Count statements before fusion
        self.stats.statements_before = self.count_statements(&program);

        // TODO: Build producer map
        // TODO: Find fusible map-map pairs
        // TODO: Perform fusions
        // TODO: Remove dead code

        // For now, return unchanged program
        self.stats.statements_after = self.stats.statements_before;

        (program, self.stats.clone())
    }

    /// Count total statements in the program.
    fn count_statements(&self, program: &Program) -> u32 {
        let mut count = 0;
        for def in &program.defs {
            count += self.count_statements_in_def(def);
        }
        count
    }

    /// Count statements in a definition.
    fn count_statements_in_def(&self, def: &Def) -> u32 {
        match def {
            Def::Function { body, .. } => self.count_statements_in_body(&body.statements),
            Def::EntryPoint { body, .. } => self.count_statements_in_body(&body.statements),
            Def::Constant { .. } => 0,
            Def::Uniform { .. } => 0,
            Def::Storage { .. } => 0,
        }
    }

    /// Count statements in a body.
    fn count_statements_in_body(&self, stmts: &[Statement]) -> u32 {
        let mut count = stmts.len() as u32;
        for stmt in stmts {
            // Count statements in nested lambdas
            count += self.count_statements_in_statement(stmt);
        }
        count
    }

    /// Count statements nested inside a statement (in lambdas).
    fn count_statements_in_statement(&self, stmt: &Statement) -> u32 {
        match &stmt.exp {
            super::Exp::Op(Op::Soac(soac)) => self.count_statements_in_soac(soac),
            super::Exp::If {
                then_body, else_body, ..
            } => {
                self.count_statements_in_body(&then_body.statements)
                    + self.count_statements_in_body(&else_body.statements)
            }
            _ => 0,
        }
    }

    /// Count statements in SOAC lambdas.
    fn count_statements_in_soac(&self, soac: &Soac) -> u32 {
        match soac {
            Soac::Map(Map { f, .. }) => self.count_statements_in_lambda(f),
            Soac::Reduce(r) => self.count_statements_in_lambda(&r.f),
            Soac::Scan(s) => self.count_statements_in_lambda(&s.f),
            Soac::SegMap(sm) => self.count_statements_in_lambda(&sm.f),
            Soac::SegReduce(sr) => self.count_statements_in_lambda(&sr.f),
            Soac::SegScan(ss) => self.count_statements_in_lambda(&ss.f),
            Soac::Iota { .. } | Soac::Replicate { .. } | Soac::Reshape { .. } => 0,
        }
    }

    /// Count statements in a lambda body.
    fn count_statements_in_lambda(&self, lambda: &Lambda) -> u32 {
        self.count_statements_in_body(&lambda.body.statements)
    }

    /// Check if a statement is a map that can be fused.
    #[allow(dead_code)]
    fn is_fusible_map<'a>(&self, stmt: &'a Statement) -> Option<&'a Map> {
        if let super::Exp::Op(Op::Soac(Soac::Map(map))) = &stmt.exp {
            // For now, all maps are potentially fusible
            // TODO: Add checks for side effects, multiple uses, etc.
            Some(map)
        } else {
            None
        }
    }
}

impl Default for SirFusion {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Fusion Helpers
// =============================================================================

/// Compose two lambdas: (f ∘ g)(x) = f(g(x))
///
/// This creates a new lambda that applies g first, then f.
#[allow(dead_code)]
fn compose_lambdas(_outer: &Lambda, _inner: &Lambda) -> Lambda {
    // TODO: Implement lambda composition
    // This requires:
    // 1. Create new lambda with inner's params
    // 2. Substitute inner's body result into outer's params
    // 3. Concatenate statements from both lambdas
    // 4. Use outer's return value
    unimplemented!("Lambda composition not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_stats_default() {
        let stats = FusionStats::new();
        assert!(!stats.any_fusions());
    }

    #[test]
    fn test_fusion_stats_any() {
        let mut stats = FusionStats::new();
        stats.map_map_fusions = 1;
        assert!(stats.any_fusions());
    }
}
