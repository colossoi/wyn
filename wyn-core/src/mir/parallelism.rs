//! Parallelism analysis for SOAC (Second-Order Array Combinator) operations.
//!
//! This module analyzes MIR to identify SOAC intrinsics and classify them
//! based on their parallelization characteristics.

use super::{Body, Expr, ExprId};

/// Classification of a SOAC's parallelization characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismKind {
    /// Each output element can be computed independently.
    /// Examples: map, zip
    Independent,
    /// Output elements have data dependencies on each other.
    /// Examples: reduce, scan, filter, scatter, hist_1d
    Dependent,
}

/// Information about a SOAC found in the MIR.
#[derive(Debug, Clone)]
pub struct SoacInfo {
    /// The expression ID of the SOAC call.
    pub expr_id: ExprId,
    /// The intrinsic name (e.g., "_w_intrinsic_map").
    pub soac_name: String,
    /// The parallelism classification.
    pub parallelism: ParallelismKind,
}

/// Result of parallelism analysis on a MIR body.
#[derive(Debug)]
pub struct ParallelismAnalysis {
    /// All SOACs found in the body.
    pub soacs: Vec<SoacInfo>,
}

impl ParallelismAnalysis {
    /// Analyze a MIR body to find and classify all SOAC intrinsics.
    pub fn analyze(body: &Body) -> Self {
        let mut soacs = Vec::new();

        for (idx, expr) in body.exprs.iter().enumerate() {
            if let Expr::Call { func, .. } = expr {
                if let Some(parallelism) = classify_soac(func) {
                    soacs.push(SoacInfo {
                        expr_id: ExprId(idx as u32),
                        soac_name: func.clone(),
                        parallelism,
                    });
                }
            }
        }

        ParallelismAnalysis { soacs }
    }

    /// Returns true if all SOACs in the body are independent (embarrassingly parallel).
    pub fn all_independent(&self) -> bool {
        self.soacs.iter().all(|s| s.parallelism == ParallelismKind::Independent)
    }

    /// Returns the count of independent vs dependent SOACs.
    pub fn counts(&self) -> (usize, usize) {
        let independent =
            self.soacs.iter().filter(|s| s.parallelism == ParallelismKind::Independent).count();
        let dependent = self.soacs.len() - independent;
        (independent, dependent)
    }
}

/// Classify a function name as a SOAC intrinsic and determine its parallelism kind.
/// Returns None if the function is not a SOAC intrinsic.
fn classify_soac(name: &str) -> Option<ParallelismKind> {
    use ParallelismKind::*;

    match name {
        // Independent: each output element can be computed without knowledge of others
        "_w_intrinsic_map" => Some(Independent),
        "_w_intrinsic_zip" => Some(Independent),

        // Dependent: output elements depend on other elements or require coordination
        "_w_intrinsic_reduce" => Some(Dependent), // accumulation to single value
        "_w_intrinsic_scan" => Some(Dependent),   // prefix sum, each depends on prior
        "_w_intrinsic_filter" => Some(Dependent), // output size is data-dependent
        "_w_intrinsic_scatter" => Some(Dependent), // multiple writes may target same index
        "_w_intrinsic_hist_1d" => Some(Dependent), // reduce-by-index, same bin conflicts

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_soac() {
        assert_eq!(
            classify_soac("_w_intrinsic_map"),
            Some(ParallelismKind::Independent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_zip"),
            Some(ParallelismKind::Independent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_reduce"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_scan"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_filter"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_scatter"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(
            classify_soac("_w_intrinsic_hist_1d"),
            Some(ParallelismKind::Dependent)
        );
        assert_eq!(classify_soac("some_other_function"), None);
    }
}
