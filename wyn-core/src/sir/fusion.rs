//! SIR Fusion Pass
//!
//! Performs SOAC fusion optimizations on SIR programs.
//! These transforms reduce kernel launch overhead and improve memory locality.
//!
//! Supported fusions:
//! - Map-map fusion: `map(f, map(g, xs))` → `map(f ∘ g, xs)`
//! - Reduce-map fusion: `reduce(op, ne, map(f, xs))` → `reduce(op ∘ f, ne, xs)`

use std::collections::{HashMap, HashSet};

use crate::IdSource;

use super::{
    Body, Def, Exp, Lambda, LambdaId, Map, Op, Param, Pat, PatElem, Program, Reduce, Soac,
    Statement, StatementId, VarId,
};

// =============================================================================
// Fusion Statistics
// =============================================================================

/// Statistics from the fusion pass.
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Number of map-map fusions performed.
    pub map_map_fusions: u32,
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
        self.map_map_fusions > 0 || self.reduce_map_fusions > 0
    }
}

// =============================================================================
// Fusion Context
// =============================================================================

/// The fusion pass.
pub struct SirFusion {
    /// Statistics about performed fusions.
    stats: FusionStats,
    /// Map from VarId to the statement index that produces it.
    producers: HashMap<VarId, usize>,
    /// Use count for each variable (how many times it's referenced).
    use_counts: HashMap<VarId, usize>,
    /// ID sources for fresh allocation during fusion.
    var_source: IdSource<VarId>,
    stm_source: IdSource<StatementId>,
    lambda_source: IdSource<LambdaId>,
}

impl SirFusion {
    /// Create a new fusion pass.
    pub fn new() -> Self {
        SirFusion {
            stats: FusionStats::new(),
            producers: HashMap::new(),
            use_counts: HashMap::new(),
            var_source: IdSource::new(),
            stm_source: IdSource::new(),
            lambda_source: IdSource::new(),
        }
    }

    /// Initialize ID sources to continue from existing program IDs.
    fn init_id_sources(&mut self, program: &Program) {
        let mut max_var = 0u32;
        let mut max_stm = 0u32;
        let mut max_lambda = 0u32;

        for def in &program.defs {
            self.scan_def_for_ids(def, &mut max_var, &mut max_stm, &mut max_lambda);
        }

        self.var_source = IdSource::starting_from(max_var + 1);
        self.stm_source = IdSource::starting_from(max_stm + 1);
        self.lambda_source = IdSource::starting_from(max_lambda + 1);
    }

    fn scan_def_for_ids(&self, def: &Def, max_var: &mut u32, max_stm: &mut u32, max_lambda: &mut u32) {
        match def {
            Def::Function { params, body, .. } => {
                for p in params {
                    *max_var = (*max_var).max(p.var.0);
                }
                self.scan_body_for_ids(body, max_var, max_stm, max_lambda);
            }
            Def::EntryPoint { inputs, body, .. } => {
                for inp in inputs {
                    *max_var = (*max_var).max(inp.var.0);
                }
                self.scan_body_for_ids(body, max_var, max_stm, max_lambda);
            }
            Def::Constant { body, .. } => {
                self.scan_body_for_ids(body, max_var, max_stm, max_lambda);
            }
            Def::Uniform { .. } | Def::Storage { .. } => {}
        }
    }

    fn scan_body_for_ids(&self, body: &Body, max_var: &mut u32, max_stm: &mut u32, max_lambda: &mut u32) {
        for stmt in &body.statements {
            *max_stm = (*max_stm).max(stmt.id.0);
            for bind in &stmt.pat.binds {
                *max_var = (*max_var).max(bind.var.0);
            }
            self.scan_exp_for_ids(&stmt.exp, max_var, max_stm, max_lambda);
        }
    }

    fn scan_exp_for_ids(&self, exp: &Exp, max_var: &mut u32, max_stm: &mut u32, max_lambda: &mut u32) {
        match exp {
            Exp::Op(Op::Soac(soac)) => {
                self.scan_soac_for_ids(soac, max_var, max_stm, max_lambda);
            }
            Exp::If { then_body, else_body, .. } => {
                self.scan_body_for_ids(then_body, max_var, max_stm, max_lambda);
                self.scan_body_for_ids(else_body, max_var, max_stm, max_lambda);
            }
            Exp::Loop { params, body, .. } => {
                for p in params {
                    *max_var = (*max_var).max(p.var.0);
                }
                self.scan_body_for_ids(body, max_var, max_stm, max_lambda);
            }
            _ => {}
        }
    }

    fn scan_soac_for_ids(&self, soac: &Soac, max_var: &mut u32, max_stm: &mut u32, max_lambda: &mut u32) {
        match soac {
            Soac::Map(Map { f, .. }) => self.scan_lambda_for_ids(f, max_var, max_stm, max_lambda),
            Soac::Reduce(r) => self.scan_lambda_for_ids(&r.f, max_var, max_stm, max_lambda),
            Soac::Scan(s) => self.scan_lambda_for_ids(&s.f, max_var, max_stm, max_lambda),
            Soac::SegMap(sm) => self.scan_lambda_for_ids(&sm.f, max_var, max_stm, max_lambda),
            Soac::SegReduce(sr) => self.scan_lambda_for_ids(&sr.f, max_var, max_stm, max_lambda),
            Soac::SegScan(ss) => self.scan_lambda_for_ids(&ss.f, max_var, max_stm, max_lambda),
            Soac::Iota { .. } | Soac::Replicate { .. } | Soac::Reshape { .. } => {}
        }
    }

    fn scan_lambda_for_ids(&self, lambda: &Lambda, max_var: &mut u32, max_stm: &mut u32, max_lambda: &mut u32) {
        *max_lambda = (*max_lambda).max(lambda.id.0);
        for p in &lambda.params {
            *max_var = (*max_var).max(p.var.0);
        }
        self.scan_body_for_ids(&lambda.body, max_var, max_stm, max_lambda);
    }

    /// Run fusion on a SIR program.
    ///
    /// Returns the transformed program and fusion statistics.
    pub fn fuse(&mut self, mut program: Program) -> (Program, FusionStats) {
        // Count statements before fusion
        self.stats.statements_before = self.count_statements(&program);

        // Initialize ID sources
        self.init_id_sources(&program);

        // Process each definition
        for def in &mut program.defs {
            self.fuse_def(def);
        }

        // Count statements after fusion
        self.stats.statements_after = self.count_statements(&program);

        (program, self.stats.clone())
    }

    /// Fuse within a definition.
    fn fuse_def(&mut self, def: &mut Def) {
        match def {
            Def::Function { body, .. } | Def::EntryPoint { body, .. } | Def::Constant { body, .. } => {
                self.fuse_body(body);
            }
            Def::Uniform { .. } | Def::Storage { .. } => {}
        }
    }

    /// Fuse within a body.
    fn fuse_body(&mut self, body: &mut Body) {
        // Build producer and use count maps for this body
        self.build_analysis_maps(&body.statements);

        // Iterate until no more fusions can be performed
        let mut changed = true;
        while changed {
            changed = false;

            // Try to find and perform one fusion
            if let Some((consumer_idx, producer_idx)) = self.find_map_map_fusion(&body.statements) {
                self.perform_map_map_fusion(body, consumer_idx, producer_idx);
                // Rebuild analysis maps after modification
                self.build_analysis_maps(&body.statements);
                changed = true;
            } else if let Some((consumer_idx, producer_idx)) = self.find_reduce_map_fusion(&body.statements) {
                self.perform_reduce_map_fusion(body, consumer_idx, producer_idx);
                self.build_analysis_maps(&body.statements);
                changed = true;
            }
        }

        // Recursively fuse in nested lambdas
        for stmt in &mut body.statements {
            self.fuse_exp(&mut stmt.exp);
        }
    }

    /// Fuse within an expression (for nested lambdas).
    fn fuse_exp(&mut self, exp: &mut Exp) {
        match exp {
            Exp::Op(Op::Soac(soac)) => {
                self.fuse_soac(soac);
            }
            Exp::If { then_body, else_body, .. } => {
                self.fuse_body(then_body);
                self.fuse_body(else_body);
            }
            Exp::Loop { body, .. } => {
                self.fuse_body(body);
            }
            _ => {}
        }
    }

    /// Fuse within a SOAC's lambda bodies.
    fn fuse_soac(&mut self, soac: &mut Soac) {
        match soac {
            Soac::Map(Map { f, .. }) => self.fuse_body(&mut f.body),
            Soac::Reduce(r) => self.fuse_body(&mut r.f.body),
            Soac::Scan(s) => self.fuse_body(&mut s.f.body),
            Soac::SegMap(sm) => self.fuse_body(&mut sm.f.body),
            Soac::SegReduce(sr) => self.fuse_body(&mut sr.f.body),
            Soac::SegScan(ss) => self.fuse_body(&mut ss.f.body),
            Soac::Iota { .. } | Soac::Replicate { .. } | Soac::Reshape { .. } => {}
        }
    }

    /// Build producer map and use counts for a statement list.
    fn build_analysis_maps(&mut self, stmts: &[Statement]) {
        self.producers.clear();
        self.use_counts.clear();

        for (idx, stmt) in stmts.iter().enumerate() {
            // Record producers
            for bind in &stmt.pat.binds {
                self.producers.insert(bind.var, idx);
            }

            // Count uses
            self.count_uses_in_exp(&stmt.exp);
        }
    }

    /// Count variable uses in an expression.
    fn count_uses_in_exp(&mut self, exp: &Exp) {
        match exp {
            Exp::Var(v) => {
                *self.use_counts.entry(*v).or_insert(0) += 1;
            }
            Exp::Prim(prim) => self.count_uses_in_prim(prim),
            Exp::Op(Op::Soac(soac)) => self.count_uses_in_soac(soac),
            Exp::Op(Op::Launch(launch)) => {
                for v in &launch.inputs {
                    *self.use_counts.entry(*v).or_insert(0) += 1;
                }
            }
            Exp::If { cond, then_body, else_body } => {
                *self.use_counts.entry(*cond).or_insert(0) += 1;
                self.count_uses_in_body(then_body);
                self.count_uses_in_body(else_body);
            }
            Exp::Loop { init, body, .. } => {
                for v in init {
                    *self.use_counts.entry(*v).or_insert(0) += 1;
                }
                self.count_uses_in_body(body);
            }
            Exp::Apply { args, .. } => {
                for v in args {
                    *self.use_counts.entry(*v).or_insert(0) += 1;
                }
            }
            Exp::Tuple(vars) => {
                for v in vars {
                    *self.use_counts.entry(*v).or_insert(0) += 1;
                }
            }
            Exp::TupleProj { tuple, .. } => {
                *self.use_counts.entry(*tuple).or_insert(0) += 1;
            }
        }
    }

    fn count_uses_in_prim(&mut self, prim: &super::Prim) {
        use super::Prim::*;
        match prim {
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Mod(a, b)
            | Eq(a, b) | Ne(a, b) | Lt(a, b) | Le(a, b) | Gt(a, b) | Ge(a, b)
            | And(a, b) | Or(a, b) => {
                *self.use_counts.entry(*a).or_insert(0) += 1;
                *self.use_counts.entry(*b).or_insert(0) += 1;
            }
            Neg(v) | Not(v) => {
                *self.use_counts.entry(*v).or_insert(0) += 1;
            }
            Index { arr, idx } => {
                *self.use_counts.entry(*arr).or_insert(0) += 1;
                *self.use_counts.entry(*idx).or_insert(0) += 1;
            }
            Intrinsic { args, .. } => {
                for v in args {
                    *self.use_counts.entry(*v).or_insert(0) += 1;
                }
            }
            ConstBool(_) | ConstI32(_) | ConstI64(_) | ConstU32(_) | ConstU64(_)
            | ConstF32(_) | ConstF64(_) => {}
        }
    }

    fn count_uses_in_soac(&mut self, soac: &Soac) {
        match soac {
            Soac::Map(Map { f, arrs, .. }) => {
                for v in arrs {
                    *self.use_counts.entry(*v).or_insert(0) += 1;
                }
                self.count_uses_in_lambda(f);
            }
            Soac::Reduce(r) => {
                *self.use_counts.entry(r.arr).or_insert(0) += 1;
                *self.use_counts.entry(r.neutral).or_insert(0) += 1;
                self.count_uses_in_lambda(&r.f);
            }
            Soac::Scan(s) => {
                *self.use_counts.entry(s.arr).or_insert(0) += 1;
                *self.use_counts.entry(s.neutral).or_insert(0) += 1;
                self.count_uses_in_lambda(&s.f);
            }
            Soac::SegMap(sm) => {
                *self.use_counts.entry(sm.segs).or_insert(0) += 1;
                for v in &sm.arrs {
                    *self.use_counts.entry(*v).or_insert(0) += 1;
                }
                self.count_uses_in_lambda(&sm.f);
            }
            Soac::SegReduce(sr) => {
                *self.use_counts.entry(sr.segs).or_insert(0) += 1;
                *self.use_counts.entry(sr.arr).or_insert(0) += 1;
                *self.use_counts.entry(sr.neutral).or_insert(0) += 1;
                self.count_uses_in_lambda(&sr.f);
            }
            Soac::SegScan(ss) => {
                *self.use_counts.entry(ss.segs).or_insert(0) += 1;
                *self.use_counts.entry(ss.arr).or_insert(0) += 1;
                *self.use_counts.entry(ss.neutral).or_insert(0) += 1;
                self.count_uses_in_lambda(&ss.f);
            }
            Soac::Replicate { value, .. } => {
                *self.use_counts.entry(*value).or_insert(0) += 1;
            }
            Soac::Reshape { arr, .. } => {
                *self.use_counts.entry(*arr).or_insert(0) += 1;
            }
            Soac::Iota { .. } => {}
        }
    }

    fn count_uses_in_lambda(&mut self, lambda: &Lambda) {
        for v in &lambda.captures {
            *self.use_counts.entry(*v).or_insert(0) += 1;
        }
        self.count_uses_in_body(&lambda.body);
    }

    fn count_uses_in_body(&mut self, body: &Body) {
        for stmt in &body.statements {
            self.count_uses_in_exp(&stmt.exp);
        }
        for v in &body.result {
            *self.use_counts.entry(*v).or_insert(0) += 1;
        }
    }

    /// Find a map-map fusion opportunity.
    /// Returns (consumer_index, producer_index) if found.
    fn find_map_map_fusion(&self, stmts: &[Statement]) -> Option<(usize, usize)> {
        for (consumer_idx, stmt) in stmts.iter().enumerate() {
            if let Exp::Op(Op::Soac(Soac::Map(consumer_map))) = &stmt.exp {
                // Check if any input array is produced by a single-use map
                for arr in &consumer_map.arrs {
                    if let Some(&producer_idx) = self.producers.get(arr) {
                        if producer_idx < consumer_idx {
                            if let Exp::Op(Op::Soac(Soac::Map(_))) = &stmts[producer_idx].exp {
                                // Check that the intermediate is only used once
                                if self.use_counts.get(arr).copied().unwrap_or(0) == 1 {
                                    // Check that the producer has a single output
                                    if stmts[producer_idx].pat.binds.len() == 1 {
                                        return Some((consumer_idx, producer_idx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Find a reduce-map fusion opportunity.
    /// Returns (reduce_index, map_index) if found.
    fn find_reduce_map_fusion(&self, stmts: &[Statement]) -> Option<(usize, usize)> {
        for (reduce_idx, stmt) in stmts.iter().enumerate() {
            if let Exp::Op(Op::Soac(Soac::Reduce(reduce))) = &stmt.exp {
                if let Some(&producer_idx) = self.producers.get(&reduce.arr) {
                    if producer_idx < reduce_idx {
                        if let Exp::Op(Op::Soac(Soac::Map(_))) = &stmts[producer_idx].exp {
                            // Check that the intermediate is only used once
                            if self.use_counts.get(&reduce.arr).copied().unwrap_or(0) == 1 {
                                if stmts[producer_idx].pat.binds.len() == 1 {
                                    return Some((reduce_idx, producer_idx));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Perform map-map fusion.
    fn perform_map_map_fusion(&mut self, body: &mut Body, consumer_idx: usize, producer_idx: usize) {
        // Get the producer map's output variable
        let producer_output_var = body.statements[producer_idx].pat.binds[0].var;

        // Clone the relevant data before mutating
        let (producer_map, consumer_map) = {
            let producer_stmt = &body.statements[producer_idx];
            let consumer_stmt = &body.statements[consumer_idx];

            let producer_map = if let Exp::Op(Op::Soac(Soac::Map(m))) = &producer_stmt.exp {
                m.clone()
            } else {
                return;
            };

            let consumer_map = if let Exp::Op(Op::Soac(Soac::Map(m))) = &consumer_stmt.exp {
                m.clone()
            } else {
                return;
            };

            (producer_map, consumer_map)
        };

        // Find which input position in consumer uses the producer's output
        let input_pos = consumer_map.arrs.iter().position(|&v| v == producer_output_var);
        let input_pos = match input_pos {
            Some(pos) => pos,
            None => return,
        };

        // Compose the lambdas: consumer ∘ producer
        let fused_lambda = self.compose_map_lambdas(
            &consumer_map.f,
            &producer_map.f,
            input_pos,
        );

        // Build new input array list: replace the intermediate with producer's inputs
        let mut new_arrs = Vec::new();
        for (i, arr) in consumer_map.arrs.iter().enumerate() {
            if i == input_pos {
                // Replace with producer's input arrays
                new_arrs.extend(producer_map.arrs.iter().copied());
            } else {
                new_arrs.push(*arr);
            }
        }

        // Create the fused map
        let fused_map = Map {
            w: producer_map.w.clone(), // Use producer's width (both should be same)
            f: fused_lambda,
            arrs: new_arrs,
        };

        // Replace consumer statement with fused map
        body.statements[consumer_idx].exp = Exp::Op(Op::Soac(Soac::Map(fused_map)));

        // Mark producer statement as dead (will be removed by DCE)
        // For now, replace with a no-op that will be cleaned up
        self.mark_statement_dead(body, producer_idx);

        // Remove dead statements
        self.remove_dead_statements(body);

        self.stats.map_map_fusions += 1;
    }

    /// Perform reduce-map fusion.
    fn perform_reduce_map_fusion(&mut self, body: &mut Body, reduce_idx: usize, map_idx: usize) {
        // Clone the relevant data
        let (producer_map, reduce_op) = {
            let map_stmt = &body.statements[map_idx];
            let reduce_stmt = &body.statements[reduce_idx];

            let producer_map = if let Exp::Op(Op::Soac(Soac::Map(m))) = &map_stmt.exp {
                m.clone()
            } else {
                return;
            };

            let reduce_op = if let Exp::Op(Op::Soac(Soac::Reduce(r))) = &reduce_stmt.exp {
                r.clone()
            } else {
                return;
            };

            (producer_map, reduce_op)
        };

        // Compose the reduction lambda with the map lambda
        // reduce(op, ne, map(f, xs)) becomes reduce(op', ne, xs)
        // where op'(acc, x) = op(acc, f(x))
        let fused_lambda = self.compose_reduce_with_map(&reduce_op.f, &producer_map.f);

        // Create the fused reduce
        let fused_reduce = Reduce {
            w: producer_map.w.clone(),
            f: fused_lambda,
            neutral: reduce_op.neutral,
            arr: producer_map.arrs[0], // Assume single input for now
            assoc: reduce_op.assoc,
        };

        // Replace reduce statement with fused reduce
        body.statements[reduce_idx].exp = Exp::Op(Op::Soac(Soac::Reduce(fused_reduce)));

        // Mark map statement as dead
        self.mark_statement_dead(body, map_idx);
        self.remove_dead_statements(body);

        self.stats.reduce_map_fusions += 1;
    }

    /// Compose two map lambdas: consumer ∘ producer
    /// Creates a lambda that applies producer first, then consumer.
    fn compose_map_lambdas(
        &mut self,
        consumer: &Lambda,
        producer: &Lambda,
        consumer_input_pos: usize,
    ) -> Lambda {
        // Build substitution map for renaming producer's variables
        let mut subst: HashMap<VarId, VarId> = HashMap::new();

        // Create fresh parameters for the composed lambda (from producer)
        let mut new_params = Vec::new();
        for param in &producer.params {
            let fresh_var = self.var_source.next_id();
            subst.insert(param.var, fresh_var);
            new_params.push(Param {
                name_hint: param.name_hint.clone(),
                var: fresh_var,
                ty: param.ty.clone(),
                span: param.span,
            });
        }

        // Add consumer's other parameters (those not replaced by producer output)
        for (i, param) in consumer.params.iter().enumerate() {
            if i != consumer_input_pos {
                let fresh_var = self.var_source.next_id();
                subst.insert(param.var, fresh_var);
                new_params.push(Param {
                    name_hint: param.name_hint.clone(),
                    var: fresh_var,
                    ty: param.ty.clone(),
                    span: param.span,
                });
            }
        }

        // Clone and rename producer's body statements
        let mut new_statements = Vec::new();
        for stmt in &producer.body.statements {
            let renamed_stmt = self.rename_statement(stmt, &mut subst);
            new_statements.push(renamed_stmt);
        }

        // Map consumer's input parameter (at consumer_input_pos) to producer's output
        if !producer.body.result.is_empty() {
            let producer_result = producer.body.result[0];
            let renamed_result = *subst.get(&producer_result).unwrap_or(&producer_result);
            subst.insert(consumer.params[consumer_input_pos].var, renamed_result);
        }

        // Clone and rename consumer's body statements
        for stmt in &consumer.body.statements {
            let renamed_stmt = self.rename_statement(stmt, &mut subst);
            new_statements.push(renamed_stmt);
        }

        // Compute result
        let new_result: Vec<VarId> = consumer.body.result
            .iter()
            .map(|v| *subst.get(v).unwrap_or(v))
            .collect();

        // Merge captures (union of both, excluding now-internal vars)
        let internal_vars: HashSet<VarId> = subst.values().copied().collect();
        let mut new_captures = Vec::new();
        for v in &producer.captures {
            if !internal_vars.contains(v) && !new_captures.contains(v) {
                new_captures.push(*v);
            }
        }
        for v in &consumer.captures {
            if !internal_vars.contains(v) && !new_captures.contains(v) {
                new_captures.push(*v);
            }
        }

        Lambda {
            id: self.lambda_source.next_id(),
            params: new_params,
            captures: new_captures,
            body: Body {
                statements: new_statements,
                result: new_result,
            },
            ret_tys: consumer.ret_tys.clone(),
            span: consumer.span,
        }
    }

    /// Compose a reduce lambda with a map lambda.
    /// reduce(op, ne, map(f, xs)) → reduce(op', ne, xs)
    /// where op'(acc, x) = op(acc, f(x))
    fn compose_reduce_with_map(&mut self, reduce_op: &Lambda, map_f: &Lambda) -> Lambda {
        let mut subst: HashMap<VarId, VarId> = HashMap::new();

        // The reduce op takes (acc, elem), map_f takes (elem) -> transformed
        // New op takes (acc, original_elem) and computes op(acc, f(original_elem))

        // Create new accumulator parameter
        let acc_var = self.var_source.next_id();
        subst.insert(reduce_op.params[0].var, acc_var);

        // Create new element parameter (original, before map)
        let elem_var = self.var_source.next_id();
        subst.insert(map_f.params[0].var, elem_var);

        let new_params = vec![
            Param {
                name_hint: reduce_op.params[0].name_hint.clone(),
                var: acc_var,
                ty: reduce_op.params[0].ty.clone(),
                span: reduce_op.params[0].span,
            },
            Param {
                name_hint: map_f.params[0].name_hint.clone(),
                var: elem_var,
                ty: map_f.params[0].ty.clone(),
                span: map_f.params[0].span,
            },
        ];

        // First, apply map_f to the element
        let mut new_statements = Vec::new();
        for stmt in &map_f.body.statements {
            let renamed_stmt = self.rename_statement(stmt, &mut subst);
            new_statements.push(renamed_stmt);
        }

        // Map_f's result becomes the second argument to reduce_op
        if !map_f.body.result.is_empty() {
            let map_result = map_f.body.result[0];
            let renamed_result = *subst.get(&map_result).unwrap_or(&map_result);
            // reduce_op's second param (the element) should be map_f's result
            subst.insert(reduce_op.params[1].var, renamed_result);
        }

        // Then apply reduce_op
        for stmt in &reduce_op.body.statements {
            let renamed_stmt = self.rename_statement(stmt, &mut subst);
            new_statements.push(renamed_stmt);
        }

        let new_result: Vec<VarId> = reduce_op.body.result
            .iter()
            .map(|v| *subst.get(v).unwrap_or(v))
            .collect();

        // Merge captures
        let internal_vars: HashSet<VarId> = subst.values().copied().collect();
        let mut new_captures = Vec::new();
        for v in &map_f.captures {
            if !internal_vars.contains(v) && !new_captures.contains(v) {
                new_captures.push(*v);
            }
        }
        for v in &reduce_op.captures {
            if !internal_vars.contains(v) && !new_captures.contains(v) {
                new_captures.push(*v);
            }
        }

        Lambda {
            id: self.lambda_source.next_id(),
            params: new_params,
            captures: new_captures,
            body: Body {
                statements: new_statements,
                result: new_result,
            },
            ret_tys: reduce_op.ret_tys.clone(),
            span: reduce_op.span,
        }
    }

    /// Rename variables in a statement according to substitution map.
    fn rename_statement(&mut self, stmt: &Statement, subst: &mut HashMap<VarId, VarId>) -> Statement {
        // Rename pattern bindings
        let new_pat = Pat {
            binds: stmt.pat.binds.iter().map(|bind| {
                let fresh_var = self.var_source.next_id();
                subst.insert(bind.var, fresh_var);
                PatElem {
                    var: fresh_var,
                    ty: bind.ty.clone(),
                    name_hint: bind.name_hint.clone(),
                }
            }).collect(),
        };

        let new_exp = self.rename_exp(&stmt.exp, subst);

        Statement {
            id: self.stm_source.next_id(),
            pat: new_pat,
            exp: new_exp,
            ty: stmt.ty.clone(),
            span: stmt.span,
        }
    }

    /// Rename variables in an expression.
    fn rename_exp(&mut self, exp: &Exp, subst: &HashMap<VarId, VarId>) -> Exp {
        match exp {
            Exp::Var(v) => Exp::Var(*subst.get(v).unwrap_or(v)),
            Exp::Prim(prim) => Exp::Prim(self.rename_prim(prim, subst)),
            Exp::Op(Op::Soac(soac)) => Exp::Op(Op::Soac(self.rename_soac(soac, subst))),
            Exp::Op(Op::Launch(launch)) => {
                Exp::Op(Op::Launch(super::Launch {
                    kind: launch.kind.clone(),
                    inputs: launch.inputs.iter().map(|v| *subst.get(v).unwrap_or(v)).collect(),
                    outputs: launch.outputs.clone(),
                    body: launch.body.clone(),
                }))
            }
            Exp::If { cond, then_body, else_body } => Exp::If {
                cond: *subst.get(cond).unwrap_or(cond),
                then_body: self.rename_body(then_body, subst),
                else_body: self.rename_body(else_body, subst),
            },
            Exp::Loop { params, init, body } => Exp::Loop {
                params: params.clone(),
                init: init.iter().map(|v| *subst.get(v).unwrap_or(v)).collect(),
                body: self.rename_body(body, subst),
            },
            Exp::Apply { func, args } => Exp::Apply {
                func: func.clone(),
                args: args.iter().map(|v| *subst.get(v).unwrap_or(v)).collect(),
            },
            Exp::Tuple(vars) => Exp::Tuple(
                vars.iter().map(|v| *subst.get(v).unwrap_or(v)).collect()
            ),
            Exp::TupleProj { tuple, index } => Exp::TupleProj {
                tuple: *subst.get(tuple).unwrap_or(tuple),
                index: *index,
            },
        }
    }

    fn rename_prim(&self, prim: &super::Prim, subst: &HashMap<VarId, VarId>) -> super::Prim {
        use super::Prim::*;
        let get = |v: &VarId| *subst.get(v).unwrap_or(v);
        match prim {
            ConstBool(b) => ConstBool(*b),
            ConstI32(n) => ConstI32(*n),
            ConstI64(n) => ConstI64(*n),
            ConstU32(n) => ConstU32(*n),
            ConstU64(n) => ConstU64(*n),
            ConstF32(n) => ConstF32(*n),
            ConstF64(n) => ConstF64(*n),
            Add(a, b) => Add(get(a), get(b)),
            Sub(a, b) => Sub(get(a), get(b)),
            Mul(a, b) => Mul(get(a), get(b)),
            Div(a, b) => Div(get(a), get(b)),
            Mod(a, b) => Mod(get(a), get(b)),
            Eq(a, b) => Eq(get(a), get(b)),
            Ne(a, b) => Ne(get(a), get(b)),
            Lt(a, b) => Lt(get(a), get(b)),
            Le(a, b) => Le(get(a), get(b)),
            Gt(a, b) => Gt(get(a), get(b)),
            Ge(a, b) => Ge(get(a), get(b)),
            And(a, b) => And(get(a), get(b)),
            Or(a, b) => Or(get(a), get(b)),
            Neg(v) => Neg(get(v)),
            Not(v) => Not(get(v)),
            Index { arr, idx } => Index { arr: get(arr), idx: get(idx) },
            Intrinsic { name, args } => Intrinsic {
                name: name.clone(),
                args: args.iter().map(get).collect(),
            },
        }
    }

    fn rename_soac(&mut self, soac: &Soac, subst: &HashMap<VarId, VarId>) -> Soac {
        let get = |v: &VarId| *subst.get(v).unwrap_or(v);
        match soac {
            Soac::Map(Map { w, f, arrs }) => Soac::Map(Map {
                w: w.clone(),
                f: self.rename_lambda(f, subst),
                arrs: arrs.iter().map(get).collect(),
            }),
            Soac::Reduce(r) => Soac::Reduce(Reduce {
                w: r.w.clone(),
                f: self.rename_lambda(&r.f, subst),
                neutral: get(&r.neutral),
                arr: get(&r.arr),
                assoc: r.assoc,
            }),
            Soac::Scan(s) => Soac::Scan(super::Scan {
                w: s.w.clone(),
                f: self.rename_lambda(&s.f, subst),
                neutral: get(&s.neutral),
                arr: get(&s.arr),
                assoc: s.assoc,
            }),
            Soac::SegMap(sm) => Soac::SegMap(super::SegMap {
                segs: get(&sm.segs),
                f: self.rename_lambda(&sm.f, subst),
                arrs: sm.arrs.iter().map(get).collect(),
            }),
            Soac::SegReduce(sr) => Soac::SegReduce(super::SegReduce {
                segs: get(&sr.segs),
                f: self.rename_lambda(&sr.f, subst),
                neutral: get(&sr.neutral),
                arr: get(&sr.arr),
                assoc: sr.assoc,
            }),
            Soac::SegScan(ss) => Soac::SegScan(super::SegScan {
                segs: get(&ss.segs),
                f: self.rename_lambda(&ss.f, subst),
                neutral: get(&ss.neutral),
                arr: get(&ss.arr),
                assoc: ss.assoc,
            }),
            Soac::Iota { n, elem_ty } => Soac::Iota { n: n.clone(), elem_ty: *elem_ty },
            Soac::Replicate { n, value } => Soac::Replicate { n: n.clone(), value: get(value) },
            Soac::Reshape { new_shape, arr } => Soac::Reshape {
                new_shape: new_shape.clone(),
                arr: get(arr),
            },
        }
    }

    fn rename_lambda(&mut self, lambda: &Lambda, outer_subst: &HashMap<VarId, VarId>) -> Lambda {
        // Create a new substitution that includes outer scope
        let mut inner_subst = outer_subst.clone();

        // Rename captures
        let new_captures: Vec<VarId> = lambda.captures
            .iter()
            .map(|v| *outer_subst.get(v).unwrap_or(v))
            .collect();

        // Create fresh variables for params (shadow outer scope)
        let new_params: Vec<Param> = lambda.params.iter().map(|p| {
            let fresh = self.var_source.next_id();
            inner_subst.insert(p.var, fresh);
            Param {
                name_hint: p.name_hint.clone(),
                var: fresh,
                ty: p.ty.clone(),
                span: p.span,
            }
        }).collect();

        let new_body = self.rename_body(&lambda.body, &inner_subst);

        Lambda {
            id: self.lambda_source.next_id(),
            params: new_params,
            captures: new_captures,
            body: new_body,
            ret_tys: lambda.ret_tys.clone(),
            span: lambda.span,
        }
    }

    fn rename_body(&mut self, body: &Body, subst: &HashMap<VarId, VarId>) -> Body {
        let mut inner_subst = subst.clone();
        let mut new_statements = Vec::new();

        for stmt in &body.statements {
            // Rename pattern bindings (creates new mappings)
            let new_pat = Pat {
                binds: stmt.pat.binds.iter().map(|bind| {
                    let fresh = self.var_source.next_id();
                    inner_subst.insert(bind.var, fresh);
                    PatElem {
                        var: fresh,
                        ty: bind.ty.clone(),
                        name_hint: bind.name_hint.clone(),
                    }
                }).collect(),
            };

            let new_exp = self.rename_exp(&stmt.exp, &inner_subst);

            new_statements.push(Statement {
                id: self.stm_source.next_id(),
                pat: new_pat,
                exp: new_exp,
                ty: stmt.ty.clone(),
                span: stmt.span,
            });
        }

        let new_result: Vec<VarId> = body.result
            .iter()
            .map(|v| *inner_subst.get(v).unwrap_or(v))
            .collect();

        Body {
            statements: new_statements,
            result: new_result,
        }
    }

    /// Mark a statement as dead by replacing it with a unit expression.
    fn mark_statement_dead(&self, body: &mut Body, idx: usize) {
        // We'll use a marker that remove_dead_statements will recognize
        body.statements[idx].exp = Exp::Tuple(vec![]); // Empty tuple = unit = dead
    }

    /// Remove statements marked as dead.
    fn remove_dead_statements(&self, body: &mut Body) {
        body.statements.retain(|stmt| {
            // Keep if not a dead marker (empty tuple with pattern that's not used)
            !matches!(&stmt.exp, Exp::Tuple(v) if v.is_empty())
        });
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
            Def::Constant { body, .. } => self.count_statements_in_body(&body.statements),
            Def::Uniform { .. } => 0,
            Def::Storage { .. } => 0,
        }
    }

    /// Count statements in a body.
    fn count_statements_in_body(&self, stmts: &[Statement]) -> u32 {
        let mut count = stmts.len() as u32;
        for stmt in stmts {
            count += self.count_statements_in_statement(stmt);
        }
        count
    }

    /// Count statements nested inside a statement (in lambdas).
    fn count_statements_in_statement(&self, stmt: &Statement) -> u32 {
        match &stmt.exp {
            Exp::Op(Op::Soac(soac)) => self.count_statements_in_soac(soac),
            Exp::If { then_body, else_body, .. } => {
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
}

impl Default for SirFusion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Span, TypeName};
    use crate::sir::{Body, Size, SirType};
    use crate::sir::builder::SirBuilder;
    use polytype::Type;

    fn dummy_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn i32_type() -> SirType {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn i32_array_type() -> SirType {
        Type::Constructed(TypeName::Array, vec![i32_type()])
    }

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

    #[test]
    fn test_map_map_fusion_detection() {
        // Build a simple program with map(f, map(g, xs))
        let mut builder = SirBuilder::new();

        // xs = input array
        let xs_var = builder.fresh_var();

        // Inner map: ys = map(|x| x + 1, xs)
        let inner_lambda = builder.lambda_with_fresh_params(
            vec![("x", i32_type())],
            vec![],
            |params| {
                let x = params[0];
                let mut b = SirBuilder::new();
                // We need fresh IDs that don't conflict
                let one_pat = b.pat1("one", i32_type());
                let one_var = one_pat.binds[0].var;
                let one_stm = Statement {
                    id: StatementId(100),
                    pat: one_pat,
                    exp: Exp::Prim(super::super::Prim::ConstI32(1)),
                    ty: i32_type(),
                    span: dummy_span(),
                };

                let result_pat = b.pat1("result", i32_type());
                let result_var = result_pat.binds[0].var;
                let result_stm = Statement {
                    id: StatementId(101),
                    pat: result_pat,
                    exp: Exp::Prim(super::super::Prim::Add(x, one_var)),
                    ty: i32_type(),
                    span: dummy_span(),
                };

                Body {
                    statements: vec![one_stm, result_stm],
                    result: vec![result_var],
                }
            },
            vec![i32_type()],
            dummy_span(),
        );

        let ys_var = builder.fresh_var();
        let inner_map_stm = Statement {
            id: StatementId(0),
            pat: Pat::single(ys_var, i32_array_type(), "ys".into()),
            exp: Exp::Op(Op::Soac(Soac::Map(Map {
                w: Size::Const(10),
                f: inner_lambda,
                arrs: vec![xs_var],
            }))),
            ty: i32_array_type(),
            span: dummy_span(),
        };

        // Outer map: zs = map(|y| y * 2, ys)
        let outer_lambda = builder.lambda_with_fresh_params(
            vec![("y", i32_type())],
            vec![],
            |params| {
                let y = params[0];
                let mut b = SirBuilder::new();

                let two_pat = b.pat1("two", i32_type());
                let two_var = two_pat.binds[0].var;
                let two_stm = Statement {
                    id: StatementId(200),
                    pat: two_pat,
                    exp: Exp::Prim(super::super::Prim::ConstI32(2)),
                    ty: i32_type(),
                    span: dummy_span(),
                };

                let result_pat = b.pat1("result", i32_type());
                let result_var = result_pat.binds[0].var;
                let result_stm = Statement {
                    id: StatementId(201),
                    pat: result_pat,
                    exp: Exp::Prim(super::super::Prim::Mul(y, two_var)),
                    ty: i32_type(),
                    span: dummy_span(),
                };

                Body {
                    statements: vec![two_stm, result_stm],
                    result: vec![result_var],
                }
            },
            vec![i32_type()],
            dummy_span(),
        );

        let zs_var = builder.fresh_var();
        let outer_map_stm = Statement {
            id: StatementId(1),
            pat: Pat::single(zs_var, i32_array_type(), "zs".into()),
            exp: Exp::Op(Op::Soac(Soac::Map(Map {
                w: Size::Const(10),
                f: outer_lambda,
                arrs: vec![ys_var], // Uses output of inner map
            }))),
            ty: i32_array_type(),
            span: dummy_span(),
        };

        let body = Body {
            statements: vec![inner_map_stm, outer_map_stm],
            result: vec![zs_var],
        };

        let program = Program {
            defs: vec![Def::Function {
                id: crate::ast::NodeId::new(1),
                name: "test".into(),
                params: vec![],
                ret_ty: i32_array_type(),
                body,
                span: dummy_span(),
            }],
            lambdas: HashMap::new(),
        };

        // Run fusion
        let mut fusion = SirFusion::new();
        let (fused_program, stats) = fusion.fuse(program);

        // Should have performed one map-map fusion
        assert_eq!(stats.map_map_fusions, 1);

        // Should have fewer statements after fusion
        assert!(stats.statements_after < stats.statements_before);

        // The fused program should have only one map statement in the body
        if let Def::Function { body, .. } = &fused_program.defs[0] {
            assert_eq!(body.statements.len(), 1);
            // And that statement should be a map
            assert!(matches!(&body.statements[0].exp, Exp::Op(Op::Soac(Soac::Map(_)))));
        } else {
            panic!("Expected function def");
        }
    }
}
