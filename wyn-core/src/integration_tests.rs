#![cfg(test)]
//! Integration tests for the full compilation pipeline.
//!
//! These tests verify that source code compiles correctly through all stages:
//! parse → desugar → resolve → type_check → alias_check → TLC → monomorphize → SSA
//!
//! All tests include entry points to ensure monomorphization can find reachable code.

use crate::Compiler;
use crate::SymbolTable;
use crate::ssa::types::Program;
use crate::tlc::TermKind;
use crate::tlc::VarRef;
use crate::tlc::extract_lambda_params;

/// Run source through the pipeline up to SSA.
fn compile_to_ssa(input: &str) -> Program {
    crate::compile_thru_ssa(input).expect("compile to SSA").ssa
}

/// Helper to check that code fails type checking (for testing error cases).
fn should_fail_type_check(input: &str) -> bool {
    crate::compile_thru_frontend(input).is_err()
}

/// Helper to compile up through TLC fusion (stops before defunctionalization).
/// Off-milestone stop — drives the typestate API directly so the same
/// `module_manager` covers both `type_check` and `to_tlc`.
fn compile_to_fused_tlc(input: &str) -> crate::tlc::Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let type_checked = Compiler::parse(input, &mut node_counter)
        .expect("parse")
        .resolve(&module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    let tlc = type_checked.to_tlc(&module_manager, false);
    let fused =
        tlc.partial_eval().normalize_soacs().fuse_maps().apply_ownership().expect("apply_ownership");
    fused.0.tlc
}

/// Assert that a compute `reduce`-over-`map`-of-range `src` parallelizes and
/// that phase1's per-thread loop trip-count transitively depends on
/// `thread_id` — i.e. each thread reduces only its *chunk* of the range.
///
/// The bug: phase1 looped the full `0..n` per invocation (quadratic — every
/// thread redoes the whole reduction; `thread_id` was used only as the
/// partials output slot, never to bound the loop). With the bug the
/// trip-count is the raw input length `n` (thread-independent), so
/// `thread_id` is unreachable from the loop condition and this fails.
fn assert_phase1_loop_depends_on_thread_id(src: &str) {
    use crate::builtins::catalog;
    use crate::op::OpTag;
    use crate::ssa::types::{ControlHeader, FuncBody, InstKind, Terminator, ValueId};
    use std::collections::{HashMap, HashSet, VecDeque};

    let program = compile_to_ssa(src);
    let thread_id_builtin = catalog().known().thread_id;

    // The phase1 entry is the parallelized worker — the one that reads
    // `thread_id` (the per-thread partials slot). phase2 is single-threaded.
    // (The EGIR redomap path mutates the original entry in place, so phase1
    // keeps the source entry name rather than gaining a `_phase1` suffix.)
    let has_thread_id = |body: &FuncBody| -> bool {
        body.inner.blocks.iter().any(|(_, block)| {
            block.insts.iter().any(|&i| {
                matches!(&body.get_inst(i).data,
                    InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if *id == thread_id_builtin)
            })
        })
    };
    let phase1 = program.entry_points.iter().find(|e| has_thread_id(&e.body)).unwrap_or_else(|| {
        panic!(
            "expected a parallelized phase1 entry (one using thread_id); entries: {:?}",
            program.entry_points.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
        )
    });
    let body = &phase1.body;

    // The two-phase reduce must have a phase2 that combines the partials into
    // the result — otherwise the partials are written but never reduced (an
    // incomplete program; the descriptor would reference a phantom entry).
    assert!(
        program.entry_points.iter().any(|e| e.name.contains("phase2") || e.name.contains("combine")),
        "missing phase2 combine entry — partials are never reduced to a result; entries: {:?}",
        program.entry_points.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
    );

    // Map each SSA result to its operand values; locate the `thread_id` result.
    let mut def: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    let mut thread_id_val: Option<ValueId> = None;
    for (_bid, block) in &body.inner.blocks {
        for &inst_id in &block.insts {
            let inst = body.get_inst(inst_id);
            let Some(result) = inst.result else { continue };
            def.insert(result, inst.data.ssa_uses());
            if let InstKind::Op {
                tag: OpTag::Intrinsic { id, .. },
                ..
            } = &inst.data
            {
                if *id == thread_id_builtin {
                    thread_id_val = Some(result);
                }
            }
        }
    }
    let thread_id_val = thread_id_val.expect("phase1 must compute thread_id");

    // Loop-header condition value(s).
    let cond_vals: Vec<ValueId> = body
        .inner
        .blocks
        .iter()
        .filter(|(bid, _)| matches!(body.control_headers.get(bid), Some(ControlHeader::Loop { .. })))
        .filter_map(|(_, block)| match &block.term {
            Terminator::CondBranch { cond, .. } => Some(*cond),
            _ => None,
        })
        .collect();
    assert!(!cond_vals.is_empty(), "phase1 must contain a loop");

    // Is `thread_id` reachable from a loop condition via def→operand edges?
    let reaches_tid = |start: ValueId| -> bool {
        let mut seen = HashSet::new();
        let mut q = VecDeque::from([start]);
        while let Some(v) = q.pop_front() {
            if v == thread_id_val {
                return true;
            }
            if seen.insert(v) {
                if let Some(ops) = def.get(&v) {
                    q.extend(ops.iter().copied());
                }
            }
        }
        false
    };
    assert!(
        cond_vals.iter().any(|&c| reaches_tid(c)),
        "phase1's loop trip-count is independent of thread_id — every thread \
         reduces the full input (quadratic) instead of a per-thread chunk"
    );

    // End-to-end: the parallelized program (including phase2 reducing the
    // partials and storing the result) must still reach SPIR-V.
    compile_to_spirv(src).expect("parallelized reduce-over-range must compile to SPIR-V");
}

/// Baseline: a scalar reduce over a mapped range chunks phase1 correctly.
#[test]
fn compute_scalar_reduce_over_range_chunks_phase1() {
    assert_phase1_loop_depends_on_thread_id(
        r#"
#[compute]
entry mn(n: u32) u32 =
  let cands = map(|i: u32| i * 2654435761u32, 0u32..<n) in
  reduce(|a: u32, b: u32| if a < b then a else b, 4294967295u32, cands)
"#,
    );
}

/// The miner's shape: a reduce whose element is an AoS `(scalar, array)`
/// tuple. Routed (like scalars) through the EGIR `phase1_transform_redomap`
/// chunking — phase1 chunks the range and phase2 combines the partials.
#[test]
fn compute_soa_tuple_reduce_over_range_chunks_phase1() {
    assert_phase1_loop_depends_on_thread_id(
        r#"
#[compute]
entry mn(n: u32) (u32, [4]u32) =
  let cands = map(|i: u32| (i, [i, i, i, i]), 0u32..<n) in
  reduce(
    |a: (u32, [4]u32), b: (u32, [4]u32)| if a.0 < b.0 then a else b,
    (4294967295u32, [0u32, 0u32, 0u32, 0u32]),
    cands)
"#,
    );
}

/// The reduce phase2 is a workgroup-parallel tree reduce: a `LocalSize(W,1,1)`
/// `*_phase2_combine` entry that uses `local_id`, a workgroup-shared array, and
/// `ControlBarrier`s — not a single-threaded combine loop.
#[test]
fn phase2_reduce_is_workgroup_parallel_tree() {
    use crate::builtins::catalog;
    use crate::op::{OpTag, PureViewSource};
    use crate::ssa::types::{ExecutionModel, InstKind};

    let program = compile_to_ssa(
        r#"
#[compute]
entry sum(xs: []f32) f32 =
  reduce(|a: f32, b: f32| a + b, 0.0, xs)
"#,
    );

    let phase2 = program
        .entry_points
        .iter()
        .find(|e| e.name.ends_with("_phase2_combine"))
        .expect("a *_phase2_combine entry");

    match &phase2.execution_model {
        ExecutionModel::Compute { local_size } => {
            assert_eq!(local_size.0, crate::egir::parallelize::PHASE2_WIDTH);
            assert_eq!((local_size.1, local_size.2), (1, 1));
        }
        other => panic!("phase2 not compute: {:?}", other),
    }

    let body = &phase2.body;
    let insts = || body.inner.blocks.iter().flat_map(|(_, b)| b.insts.iter().map(|&i| body.get_inst(i)));

    let barriers = insts().filter(|n| matches!(n.data, InstKind::ControlBarrier)).count();
    assert_eq!(barriers, 2, "grid-stride + tree-step barriers");

    assert!(
        insts().any(|n| matches!(
            &n.data,
            InstKind::Op {
                tag: OpTag::StorageView(PureViewSource::Workgroup { .. }),
                ..
            }
        )),
        "phase2 must use a workgroup-shared array"
    );

    let local_id = catalog().known().local_id;
    assert!(
        insts().any(|n| matches!(
            &n.data,
            InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if *id == local_id
        )),
        "phase2 must read local_id"
    );
}

// =============================================================================
// Phase 2 regression: unbound `Var(Symbol(sym))` references through TLC passes
// =============================================================================

/// Walk a term and assert that every `TermKind::Var(VarRef::Symbol(sym))`
/// references a sym that is either:
/// - bound by an enclosing Let / Lambda param / Loop var / SOAC element
///   parameter, or
/// - a top-level def name in `top_level`.
///
/// On violation, panics with the offending sym, its symbol-table name,
/// and the pipeline stage name.
fn assert_no_unbound_var_refs(program: &crate::tlc::Program, stage: &str) {
    use crate::SymbolId;
    use crate::tlc::{ArrayExpr, Lambda, LoopKind, SoacOp, Term, TermKind};
    use std::collections::HashSet;

    fn walk(term: &Term, bound: &HashSet<SymbolId>, symbols: &SymbolTable, stage: &str, def_name: &str) {
        match &term.kind {
            TermKind::Var(VarRef::Symbol(sym)) => {
                assert!(
                    bound.contains(sym),
                    "[{stage}] def `{def_name}`: unbound Var(sym{:?}) name={:?}",
                    sym.0,
                    symbols.get(*sym)
                );
            }
            TermKind::Var(VarRef::Builtin { .. })
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {}
            TermKind::Coerce { inner, .. } => walk(inner, bound, symbols, stage, def_name),
            TermKind::App { func, args } => {
                walk(func, bound, symbols, stage, def_name);
                for a in args {
                    walk(a, bound, symbols, stage, def_name);
                }
            }
            TermKind::Lambda(Lambda { params, body, .. }) => {
                let mut inner = bound.clone();
                for (p, _) in params {
                    inner.insert(*p);
                }
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::Let { name, rhs, body, .. } => {
                walk(rhs, bound, symbols, stage, def_name);
                let mut inner = bound.clone();
                inner.insert(*name);
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                walk(cond, bound, symbols, stage, def_name);
                walk(then_branch, bound, symbols, stage, def_name);
                walk(else_branch, bound, symbols, stage, def_name);
            }
            TermKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                walk(init, bound, symbols, stage, def_name);
                for (_, _, e) in init_bindings {
                    walk(e, bound, symbols, stage, def_name);
                }
                match kind {
                    LoopKind::For { iter, .. } => walk(iter, bound, symbols, stage, def_name),
                    LoopKind::ForRange { bound: bnd, .. } => {
                        walk(bnd, bound, symbols, stage, def_name);
                    }
                    LoopKind::While { cond } => walk(cond, bound, symbols, stage, def_name),
                }
                let mut inner = bound.clone();
                inner.insert(*loop_var);
                if let LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } = kind {
                    inner.insert(*var);
                }
                for (n, _, _) in init_bindings {
                    inner.insert(*n);
                }
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::Soac(soac) => walk_soac(soac, bound, symbols, stage, def_name),
            TermKind::ArrayExpr(ae) => walk_array_expr(ae, bound, symbols, stage, def_name),

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                for p in parts {
                    walk(p, bound, symbols, stage, def_name);
                }
            }
            TermKind::TupleProj { tuple, .. } => walk(tuple, bound, symbols, stage, def_name),
            TermKind::Index { array, index } => {
                walk(array, bound, symbols, stage, def_name);
                walk(index, bound, symbols, stage, def_name);
            }
        }
    }

    fn walk_lambda(
        lam: &crate::tlc::Lambda,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        let mut inner = bound.clone();
        for (p, _) in &lam.params {
            inner.insert(*p);
        }
        walk(&lam.body, &inner, symbols, stage, def_name);
    }

    fn walk_soac(
        soac: &SoacOp,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match soac {
            SoacOp::Map { lam, inputs, .. } => {
                for i in inputs {
                    walk_array_expr(i, bound, symbols, stage, def_name);
                }
                walk_lambda(&lam.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Reduce { op, ne, input, .. } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Scan { op, ne, input, .. } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => {
                walk(ne, bound, symbols, stage, def_name);
                for i in inputs {
                    walk_array_expr(i, bound, symbols, stage, def_name);
                }
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
                walk_lambda(&reduce_op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Filter { pred, input, .. } => {
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&pred.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Scatter { indices, values, .. } => {
                walk_array_expr(indices, bound, symbols, stage, def_name);
                walk_array_expr(values, bound, symbols, stage, def_name);
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(indices, bound, symbols, stage, def_name);
                walk_array_expr(values, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
        }
    }

    fn walk_array_expr(
        ae: &ArrayExpr,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match ae {
            ArrayExpr::Ref(t) => walk(t, bound, symbols, stage, def_name),
            ArrayExpr::Zip(arrs) => {
                for a in arrs {
                    walk_array_expr(a, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::Soac(soac) => walk_soac(soac, bound, symbols, stage, def_name),
            ArrayExpr::Literal(elems) => {
                for e in elems {
                    walk(e, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::Range { start, len, step } => {
                walk(start, bound, symbols, stage, def_name);
                walk(len, bound, symbols, stage, def_name);
                if let Some(s) = step {
                    walk(s, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::StorageView(crate::tlc::StorageView { offset, len, .. }) => {
                walk(offset, bound, symbols, stage, def_name);
                walk(len, bound, symbols, stage, def_name);
            }
        }
    }

    // `bound` = everything the TLC name-resolver considers a top-level
    // symbol, not just things with bodies. `def_syms` holds the
    // pre-allocated SymbolId for every prelude/user/sig top-level name.
    let mut top_level: std::collections::HashSet<SymbolId> = std::collections::HashSet::new();
    for sym in program.def_syms.values() {
        top_level.insert(*sym);
    }
    // Catch any defs whose own SymbolId isn't already in `def_syms` —
    // shouldn't happen by construction, but assert via union.
    for d in &program.defs {
        top_level.insert(d.name);
    }
    for def in &program.defs {
        let def_name = program.symbols.get(def.name).cloned().unwrap_or_default();
        walk(&def.body, &top_level, &program.symbols, stage, &def_name);
    }
}

/// Regression: under Phase 2, `let x = arr in body[x]` was leaving a
/// free `Var(Symbol)` in the post-`partial_eval` TLC because `apply`'s
/// `Var(Builtin)` and catch-all arms cloned the original term instead
/// of residualizing through the eval'd args. The fix lives in
/// `tlc::partial_eval::residualize_call`. This test compiles the
/// minimal repro through the full canonical TLC pipeline and asserts
/// no `Var(Symbol(sym))` references an unbound symbol.
#[test]
fn let_binding_substitution_survives_partial_eval() {
    let source = r#"
#[fragment]
entry frag() #[location(0)] vec4f32 =
    let range = [1, 2, 3, 4] in
    @[f32.i32(range[0]), 0.0, 0.0, 1.0]
"#;
    let tlc = crate::compile_thru_tlc(source).expect("compile_thru_tlc");
    assert_no_unbound_var_refs(&tlc.tlc, "compile_thru_tlc");
}

// =============================================================================
// SOAC Fusion Integration Tests
// =============================================================================

#[test]
fn consuming_map_compiles_end_to_end() {
    // `*[N]T` map whose input is dead-after: the ownership pass
    // sets `destination = SoacDestination::InputBuffer`, EGIR
    // conversion threads it through, and `soac_expand` emits the
    // in-place loop. Compiling end-to-end through SSA exercises
    // every layer.
    let _ssa = compile_to_ssa(
        r#"
def f(a: *[8]i32) [8]i32 = map(|x: i32| x + 1, a)
"#,
    );
}

/// Count `_w_intrinsic_uninit` calls across the entire SSA program
/// (functions + entry points). `Fresh` Map destinations introduce
/// one per allocation; the `InputBuffer` destination should
/// introduce zero. Aggregating across all bodies sidesteps
/// inlining choices that move the map's body between functions.
fn count_uninit_in_program(ssa: &Program) -> usize {
    let mut count = 0;
    let bodies = ssa
        .functions
        .iter()
        .map(|f| &f.body.inner.insts)
        .chain(ssa.entry_points.iter().map(|e| &e.body.inner.insts));
    for insts in bodies {
        for (_id, inst) in insts {
            if let crate::ssa::types::InstKind::Op { tag, .. } = &inst.data {
                match tag {
                    crate::op::OpTag::Call(f) => {
                        if f == "_w_intrinsic_uninit" {
                            count += 1;
                        }
                    }
                    crate::op::OpTag::Intrinsic { id, .. } => {
                        let name = crate::builtins::by_id(*id).raw.surface_name;
                        if name == "_w_intrinsic_uninit" {
                            count += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    count
}

#[test]
fn consuming_map_skips_fresh_allocation() {
    // For an input-side DPS Map, the loop's carried buffer starts
    // as the input parameter — no `_w_intrinsic_uninit` call
    // should be emitted. Compare against a non-consuming Map
    // (caller-borrowed input) which falls back to Fresh and
    // emits at least one uninit allocation.
    let consuming_ssa = compile_to_ssa(
        r#"
def bump(a: *[8]i32) [8]i32 = map(|x: i32| x + 1, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = bump([1, 2, 3, 4, 5, 6, 7, 8]) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&consuming_ssa),
        0,
        "consuming map (`*[N]T` input, dead-after) should not allocate a fresh buffer",
    );

    let borrowing_ssa = compile_to_ssa(
        r#"
def bump(a: [8]i32) [8]i32 = map(|x: i32| x + 1, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = bump([1, 2, 3, 4, 5, 6, 7, 8]) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert!(
        count_uninit_in_program(&borrowing_ssa) >= 1,
        "non-consuming map (caller-borrowed input) should allocate a fresh buffer",
    );
}

#[test]
fn consuming_scan_compiles_end_to_end() {
    // Parallel of `consuming_map_compiles_end_to_end` for Scan: `*[N]T`
    // input that's dead-after; ownership sets
    // `destination = SoacDestination::InputBuffer`, EGIR threads it through,
    // and `soac_expand` runs the destination-passing loop. Pre-S4 this hit a
    // panic.
    let _ssa = compile_to_ssa(
        r#"
def cumsum(a: *[8]i32) [8]i32 = scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    );
}

#[test]
fn consuming_scan_skips_fresh_allocation() {
    // Same allocation-count assertion as the Map version: an
    // input-side DPS Scan carries the input buffer through the loop
    // and writes back to it, so no `_w_intrinsic_uninit` is needed.
    let consuming_ssa = compile_to_ssa(
        r#"
def cumsum(a: *[8]i32) [8]i32 = scan(|acc: i32, x: i32| acc + x, 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = cumsum([1, 2, 3, 4, 5, 6, 7, 8]) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&consuming_ssa),
        0,
        "consuming scan (`*[N]T` input, dead-after) should not allocate a fresh buffer",
    );

    let borrowing_ssa = compile_to_ssa(
        r#"
def cumsum(a: [8]i32) [8]i32 = scan(|acc: i32, x: i32| acc + x, 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = cumsum([1, 2, 3, 4, 5, 6, 7, 8]) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert!(
        count_uninit_in_program(&borrowing_ssa) >= 1,
        "non-consuming scan (caller-borrowed input) should allocate a fresh buffer",
    );
}

#[test]
fn consuming_scan_compute_entry_compiles_to_spirv() {
    // Compute entry with `*[]T` param. Exercises the Scan-DPS path
    // end-to-end through SPIR-V emission. Regression guards:
    //
    // 1. Type-checker: `*[]T` on a compute-entry param must constrain
    //    the array variant to `View`, not default to `Composite`.
    //    Otherwise `polytype_to_spirv` panics with "Composite variant
    //    unsized arrays not supported".
    //
    // 2. SPIR-V backend: views threaded through loop block params
    //    (`%phi = phi(orig_view, array_with_inplace_result)`) must
    //    keep their buffer provenance. `view_buffer_id` is propagated
    //    across branch edges and through `array_with_inplace`, so
    //    `ViewIndex` can resolve the backing buffer without
    //    extracting buffer_id from a runtime struct field.
    let spv = compile_to_spirv(
        r#"
#[compute]
entry scan_inplace(a: *[]i32) []i32 =
  scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    )
    .expect("compute scan_inplace should compile end-to-end");
    assert!(!spv.is_empty(), "compute scan_inplace emitted empty SPIR-V");
}

#[test]
fn parallel_scan_emits_swap_wrapper_with_swapped_args() {
    // Phase 3 of parallel scan reads `off = block_offsets[tid]` and
    // applies `op(off, elem)` to each element of `output[chunk]`. Map's
    // body-call convention is `func(elem, ...captures)`, which would
    // give `op(elem, off)` — sound for commutative ops, silently wrong
    // for non-commutative ones. EGIR plumbs around this by synthesizing
    // a swap-args wrapper function `\(a, b) -> op(b, a)` and routing
    // phase 3 through it; this test pins that wiring in SSA.
    let ssa = compile_to_ssa(
        r#"
#[compute]
entry parallel_scan(a: []i32) []i32 = scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    );

    let wrapper = ssa
        .functions
        .iter()
        .find(|f| f.name.ends_with("_scan_op_swap"))
        .expect("parallel scan should synthesize a swap wrapper EgirFunc");

    assert_eq!(
        wrapper.body.params.len(),
        2,
        "swap wrapper must take exactly two params"
    );
    let a_id = wrapper.body.params[0].0;
    let b_id = wrapper.body.params[1].0;

    let call = wrapper
        .body
        .inner
        .insts
        .values()
        .find_map(|inst| match &inst.data {
            crate::ssa::types::InstKind::Op {
                tag: crate::op::OpTag::Call(name),
                operands,
            } => Some((name.clone(), operands.clone())),
            _ => None,
        })
        .expect("swap wrapper body must contain a Call");

    assert!(
        !call.0.ends_with("_scan_op_swap"),
        "swap wrapper should call the underlying op, not itself: got {}",
        call.0
    );
    let operands: Vec<_> = call.1.iter().map(|v| v.as_ssa()).collect();
    assert_eq!(
        operands,
        vec![Some(b_id), Some(a_id)],
        "swap wrapper must call inner(b, a), not inner(a, b); got operands {:?} vs params [a={:?}, b={:?}]",
        operands,
        a_id,
        b_id,
    );
}

#[test]
fn consuming_filter_compiles_end_to_end() {
    // `*[N]T` filter whose input is dead-after: ownership sets
    // `destination = SoacDestination::InputBuffer`, EGIR threads it
    // through, and `build_filter_loop` carries the input array as the
    // destination buffer.
    let _ssa = compile_to_ssa(
        r#"
def keep_pos(a: *[8]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)
"#,
    );
}

#[test]
fn consuming_filter_skips_fresh_allocation() {
    // Parallel of the Map / Scan allocation-count tests. Consuming
    // filter carries the input as the destination buffer — no
    // `_w_intrinsic_uninit` call. Non-consuming filter still
    // allocates the destination.
    let consuming_ssa = compile_to_ssa(
        r#"
def keep_pos(a: *[8]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = keep_pos([1, -2, 3, -4, 5, -6, 7, -8]) in
    @[f32.i32(r[0]), 0.0, 0.0, 1.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&consuming_ssa),
        0,
        "consuming filter (`*[N]T` input, dead-after) should not allocate a fresh buffer",
    );

    let borrowing_ssa = compile_to_ssa(
        r#"
def keep_pos(a: [8]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = keep_pos([1, -2, 3, -4, 5, -6, 7, -8]) in
    @[f32.i32(r[0]), 0.0, 0.0, 1.0]
"#,
    );
    assert!(
        count_uninit_in_program(&borrowing_ssa) >= 1,
        "non-consuming filter (caller-borrowed input) should allocate a fresh buffer",
    );
}

/// Multiset of `(category, identifier)` pairs across every instruction
/// in `ssa.functions` + `ssa.entry_points`. Used by structural-equivalence
/// tests that need to compare two SSA programs while ignoring value-id
/// renumbering, block-ordering, and other low-level details.
fn inst_signature_multiset(ssa: &Program) -> std::collections::BTreeMap<String, usize> {
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;
    use std::collections::BTreeMap;

    let signature = |kind: &InstKind| -> String {
        match kind {
            InstKind::Alloca { .. } => "Alloca".to_string(),
            InstKind::Load { .. } => "Load".to_string(),
            InstKind::Store { .. } => "Store".to_string(),
            InstKind::ViewIndex { .. } => "ViewIndex".to_string(),
            InstKind::OutputSlot { .. } => "OutputSlot".to_string(),
            InstKind::ControlBarrier => "ControlBarrier".to_string(),
            InstKind::Op { tag, .. } => format!(
                "Op:{}",
                match tag {
                    OpTag::Call(name) => format!("Call({})", name),
                    OpTag::Intrinsic { id, .. } => {
                        let name = crate::builtins::by_id(*id).raw.surface_name;
                        format!("Intrinsic({})", name)
                    }
                    OpTag::BinOp(s) => format!("BinOp({})", s),
                    OpTag::UnaryOp(s) => format!("UnaryOp({})", s),
                    // Literal values intentionally NOT included in the
                    // signature — a constant-folding refactor shouldn't
                    // make the test flake. Variant name alone is the
                    // structural signal.
                    OpTag::Int(_) => "Int".to_string(),
                    OpTag::Uint(_) => "Uint".to_string(),
                    OpTag::Float(_) => "Float".to_string(),
                    OpTag::Bool(_) => "Bool".to_string(),
                    OpTag::Unit => "Unit".to_string(),
                    OpTag::Global(_) => "Global".to_string(),
                    OpTag::Extern(_) => "Extern".to_string(),
                    OpTag::Tuple(_) => "Tuple".to_string(),
                    OpTag::Vector(_) => "Vector".to_string(),
                    OpTag::Matrix { .. } => "Matrix".to_string(),
                    OpTag::ArrayLit(_) => "ArrayLit".to_string(),
                    OpTag::ArrayRange { .. } => "ArrayRange".to_string(),
                    OpTag::Project { .. } => "Project".to_string(),
                    OpTag::Index => "Index".to_string(),
                    OpTag::Materialize => "Materialize".to_string(),
                    OpTag::DynamicExtract => "DynamicExtract".to_string(),
                    OpTag::StorageView(_) => "StorageView".to_string(),
                    OpTag::StorageViewLen => "StorageViewLen".to_string(),
                    OpTag::ViewIndex => "ViewIndex(pure)".to_string(),
                    OpTag::OutputSlot { .. } => "OutputSlot(pure)".to_string(),
                }
            ),
        }
    };

    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    let bodies = ssa
        .functions
        .iter()
        .map(|f| &f.body.inner.insts)
        .chain(ssa.entry_points.iter().map(|e| &e.body.inner.insts));
    for insts in bodies {
        for (_id, inst) in insts {
            *out.entry(signature(&inst.data)).or_insert(0) += 1;
        }
    }
    out
}

#[test]
fn consuming_filter_length_matches_borrowing() {
    // Sharpening `consuming_filter_skips_fresh_allocation`: that test
    // only asserts on the uninit count, not on the indexed reads or
    // `length(r)` extraction path. If the bounded result's `len` field
    // gets the input's static capacity instead of the runtime
    // write-cursor count, `length(r)` silently returns N rather than
    // the actual positive-count — r[0] and r[1] still produce the
    // first two kept elements (those slots were written), masking the
    // bug. Anything that iterates `0..length(r)` would then read
    // garbage past the real count.
    //
    // Test shape: same fragment against consuming (`*[N]T`) and
    // borrowing (`[N]T`) inputs. Assert the lowered SSA instruction
    // multisets are equal modulo the one extra `_w_intrinsic_uninit`
    // call in the borrowing variant. If the consuming bounded wrapper
    // short-circuits the length lookup to a constant, the consuming
    // side will be missing the `_w_intrinsic_length` intrinsic the
    // borrowing side has — multisets diverge, test fails.
    let consuming = compile_to_ssa(
        r#"
def keep_pos(a: *[4]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = keep_pos([1, -2, 3, -4]) in
    @[f32.i32(length(r)), f32.i32(r[0]), f32.i32(r[1]), 1.0]
"#,
    );
    let borrowing = compile_to_ssa(
        r#"
def keep_pos(a: [4]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = keep_pos([1, -2, 3, -4]) in
    @[f32.i32(length(r)), f32.i32(r[0]), f32.i32(r[1]), 1.0]
"#,
    );

    let mut c_tags = inst_signature_multiset(&consuming);
    let b_tags = inst_signature_multiset(&borrowing);

    // Mod out the expected allocation-strategy difference: borrowing
    // has one extra `_w_intrinsic_uninit` call.
    let uninit = "Op:Intrinsic(_w_intrinsic_uninit)".to_string();
    let c_uninit = c_tags.get(&uninit).copied().unwrap_or(0);
    let b_uninit = b_tags.get(&uninit).copied().unwrap_or(0);
    assert!(
        b_uninit > c_uninit,
        "expected borrowing to allocate more buffers than consuming \
         (c={}, b={}) — consuming-filter probably regressed",
        c_uninit,
        b_uninit,
    );
    *c_tags.entry(uninit).or_insert(0) += b_uninit - c_uninit;

    assert_eq!(
        c_tags, b_tags,
        "consuming vs borrowing filter SSA differs in non-allocation ops — \
         likely a bounded-wrapper / length-extraction soundness bug. \
         The two programs differ only in whether the filter result aliases \
         the input buffer; the user-visible computation should be identical."
    );
}

#[test]
fn test_map_reduce_fusion_end_to_end() {
    let source = r#"
def globalArr: [4]f32 = [10.0, 20.0, 30.0, 40.0]

def myMap(ro: f32, rd: f32) [4]f32 =
  map(|x: f32| x + ro + rd, globalArr)

def myReduce(hits: [4]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, 999.0, hits)

#[fragment]
entry fragment_main() #[location(0)] vec4f32 =
  let hits = myMap(1.0, 2.0) in
  let closest = myReduce(hits) in
  @[closest, 0.0, 0.0, 1.0]
"#;

    let tlc = compile_to_fused_tlc(source);

    // After fusion, check that myMap's body is no longer a standalone map
    // or that some def contains a fused reduce
    let _my_map_has_map = tlc.defs.iter().any(|def| {
        let name = tlc.symbols.get(def.name).cloned().unwrap_or_default();
        if name != "myMap" {
            return false;
        }
        let (_, body) = extract_lambda_params(&def.body);
        has_soac_kind(&body, "Map")
    });

    let _any_has_reduce = tlc.defs.iter().any(|def| {
        let (_, body) = extract_lambda_params(&def.body);
        has_soac_kind(&body, "Reduce")
    });

    // Check fragment_main: does it contain a fused Reduce?
    let fragment_main = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("fragment_main"))
        .expect("fragment_main not found");

    let (_, frag_body) = extract_lambda_params(&fragment_main.body);
    let frag_has_redomap = has_soac_kind(&frag_body, "Redomap");
    let frag_has_reduce = has_soac_kind(&frag_body, "Reduce");
    let frag_has_map = has_soac_kind(&frag_body, "Map");

    eprintln!("fragment_main has Redomap: {}", frag_has_redomap);
    eprintln!("fragment_main has Reduce: {}", frag_has_reduce);
    eprintln!("fragment_main has Map: {}", frag_has_map);
    eprintln!(
        "fragment_main body: {:?}",
        std::mem::discriminant(&frag_body.kind)
    );

    // Print the Let chain structure
    fn print_term(term: &crate::tlc::Term, syms: &SymbolTable, depth: usize) {
        let indent = "  ".repeat(depth);
        match &term.kind {
            TermKind::Let { name, rhs, body, .. } => {
                let n = syms.get(*name).cloned().unwrap_or_else(|| format!("{:?}", name));
                eprintln!("{indent}let {n} = ...");
                print_term(rhs, syms, depth + 1);
                print_term(body, syms, depth);
            }
            TermKind::Soac(soac) => {
                eprintln!("{indent}SOAC {:?}", std::mem::discriminant(soac));
            }
            TermKind::App { func, args } => {
                eprintln!("{indent}App:");
                print_term(func, syms, depth + 1);
                for a in args {
                    print_term(a, syms, depth + 1);
                }
            }
            TermKind::Var(VarRef::Symbol(s)) => {
                let n = syms.get(*s).cloned().unwrap_or_else(|| format!("{:?}", s));
                eprintln!("{indent}Var({n})");
            }
            other => {
                eprintln!("{indent}{:?}", std::mem::discriminant(other));
            }
        }
    }
    print_term(&frag_body, &tlc.symbols, 0);

    // The fusion should have replaced the let chain with a fused SOAC
    // or at minimum the fragment_main should contain a Reduce
    assert!(
        frag_has_redomap || frag_has_reduce,
        "Expected fragment_main to contain a fused Redomap or Reduce after interprocedural fusion"
    );
}

fn has_soac_kind(term: &crate::tlc::Term, kind: &str) -> bool {
    use crate::tlc::{SoacOp, TermKind};
    match &term.kind {
        TermKind::Soac(SoacOp::Map { .. }) if kind == "Map" => true,
        TermKind::Soac(SoacOp::Reduce { .. }) if kind == "Reduce" => true,
        TermKind::Soac(SoacOp::Redomap { .. }) if kind == "Redomap" => true,
        TermKind::Let { rhs, body, .. } => has_soac_kind(rhs, kind) || has_soac_kind(body, kind),
        TermKind::Lambda(lam) => has_soac_kind(&lam.body, kind),
        TermKind::App { func, args } => {
            has_soac_kind(func, kind) || args.iter().any(|a| has_soac_kind(a, kind))
        }
        TermKind::Tuple(parts) | TermKind::VecLit(parts) => parts.iter().any(|p| has_soac_kind(p, kind)),
        TermKind::TupleProj { tuple, .. } => has_soac_kind(tuple, kind),
        TermKind::Index { array, index } => has_soac_kind(array, kind) || has_soac_kind(index, kind),
        _ => false,
    }
}

// =============================================================================
// Multi-output compute entries
// =============================================================================

/// Regression: a compute entry returning a tuple of >1 runtime-sized
/// array used to panic in EGIR lowering — the SOAC→OutputView rewrite
/// that streams a runtime-sized result into its bound storage view only
/// fired for the single-output case. For a tuple result, each field
/// reached `emit_compute_output_stores` as a plain value and hit the
/// `is_unsized_array` guard (`from_tlc.rs` "must rewrite to OutputView
/// upstream"). Each tuple field's producing Map/Scan must be retargeted
/// to its own output view.
#[test]
fn test_multi_output_compute_runtime_sized_arrays() {
    let _ssa = compile_to_ssa(
        r#"
#[compute]
entry gen(src: []f32) ([]f32, []f32) =
    (map(|x: f32| x * 2.0, src), map(|x: f32| x * 3.0, src))
"#,
    );
    // Compilation success (no panic) is the test.
}

// =============================================================================
// Basic Expressions
// =============================================================================

#[test]
fn test_basic_expressions() {
    // Tests: functions, let bindings, if expressions, binary/unary ops
    let _ssa = compile_to_ssa(
        r#"
def add(x: i32, y: i32) i32 = x + y

def with_let(a: i32, b: i32) i32 =
    let x = a in
    let y = b in
    x + y

def with_if(x: bool) i32 = if x then 1 else 0

def with_ops(x: i32, y: i32) i32 = x * y + x / y - (-x)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = add(1, 2) in
    let c = with_let(3, 4) in
    let d = with_if(true) in
    let e = with_ops(5, 6) in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );

    // Compilation success is the test (partial eval may inline simple functions)
}

// =============================================================================
// Data Structures
// =============================================================================

#[test]
fn test_data_structures() {
    // Tests: arrays, tuples, records, tuple patterns
    let _ssa = compile_to_ssa(
        r#"
def arr = [1, 2, 3]

def record = {x: 1, y: 2}

def tuple_destruct: i32 =
    let (a, b) = (1, 2) in a + b

def nested_tuple: i32 =
    let ((a, b), c) = ((1, 2), 3) in a + b + c

def array_index(arr: [4]i32, i: i32) i32 = arr[i]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = arr[0] in
    let b = record.x in
    let c = tuple_destruct in
    let d = nested_tuple in
    let e = array_index([1, 2, 3, 4], 0) in
    @[f32.i32(a + b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Tuple Positional Access
// =============================================================================

#[test]
fn test_tuple_positional_access() {
    // Tests: .0, .1 on tuples, chained access, in expressions
    let _ssa = compile_to_ssa(
        r#"
def first(t: (i32, f32)) i32 = t.0

def second(t: (i32, f32)) f32 = t.1

def sum_pair(t: (i32, i32)) i32 = t.0 + t.1

def nested(t: ((i32, i32), f32)) i32 =
    let inner = t.0 in inner.0 + inner.1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let t = (42, 3.14) in
    let a = first(t) in
    let b = sum_pair((1, 2)) in
    let c = nested(((10, 20), 1.0)) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
}

// =============================================================================
// Loops
// =============================================================================

#[test]
fn test_loops() {
    // Tests: while loops, for-range loops, for-in loops
    let _ssa = compile_to_ssa(
        r#"
def while_loop: i32 =
    loop x = 0 while x < 10 do x + 1

def for_range_loop: i32 =
    loop acc = 0 for i < 10 do acc + i

def for_in_loop(arr: [5]i32) i32 =
    loop acc = 0 for x in arr do acc + x

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = while_loop in
    let b = for_range_loop in
    let c = for_in_loop([1, 2, 3, 4, 5]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Lambdas and Closures
// =============================================================================

#[test]
fn test_lambdas_and_closures() {
    // Tests: lambdas with captures, nested lambdas, direct calls, tuple params
    let _ssa = compile_to_ssa(
        r#"
def with_capture(y: i32) i32 =
    let f = |x: i32| x + y in
    f(10)

def nested_lambda(x: i32) i32 =
    let outer = |a: i32|
        let inner = |b: i32| a + b + x in
        inner(a)
    in
    outer(5)

def tuple_param_lambda: i32 =
    let add = |(x, y): (i32, i32)| x + y in
    add((1, 2))

def direct_call: i32 =
    let inc = |x: i32| x + 1 in
    inc(5)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = with_capture(10) in
    let c = nested_lambda(100) in
    let d = tuple_param_lambda in
    let e = direct_call in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Higher-Order Functions (map, reduce, filter)
// =============================================================================

#[test]
fn test_higher_order_functions() {
    // Tests: map, reduce, filter with lambdas and named functions
    let _ssa = compile_to_ssa(
        r#"
def double(x: i32) i32 = x * 2

def map_named(arr: [4]i32) [4]i32 = map(double, arr)

def map_lambda(arr: [4]i32) [4]i32 = map(|x: i32| x + 1, arr)

def map_with_capture(arr: [4]i32, offset: i32) [4]i32 =
    map(|x: i32| x + offset, arr)

def reduce_sum(arr: [4]f32) f32 =
    reduce(|acc: f32, x: f32| acc + x, 0.0, arr)

def reduce_tuple(hits: [4](f32, i32)) (f32, i32) =
    reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
             if t1 < t2 then (t1, m1) else (t2, m2),
           (1000.0, 0),
           hits)

def is_positive(x: i32) bool = x > 0

def filter_positive(arr: [5]i32) ?k. [k]i32 =
    filter(is_positive, arr)

def filter_lambda(arr: [4]i32) ?k. [k]i32 =
    filter(|x: i32| x % 2 == 0, arr)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = map_named([1, 2, 3, 4]) in
    let b = map_lambda([1, 2, 3, 4]) in
    let c = map_with_capture([1, 2, 3, 4], 10) in
    let d = reduce_sum([1.0, 2.0, 3.0, 4.0]) in
    let (t, _) = reduce_tuple([(1.0, 0), (2.0, 1), (0.5, 2), (3.0, 3)]) in
    let e = filter_positive([1, -2, 3, -4, 5]) in
    let f = filter_lambda([1, 2, 3, 4]) in
    @[d + t, f32.i32(a[0] + b[0] + c[0] + length(e) + length(f)), 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Defunctionalization Scenarios
// =============================================================================

#[test]
fn test_defunctionalization() {
    // Tests various defunctionalization scenarios
    let _ssa = compile_to_ssa(
        r#"
def different_captures(x: i32, y: i32, arr: [4]i32) ([4]i32, [4]i32) =
    let result1 = map(|e: i32| e + x, arr) in
    let result2 = map(|e: i32| e * y, arr) in
    (result1, result2)

def nested_capture(x: i32, arr: [4]i32) [4]i32 =
    let outer = |y: i32|
        let inner = |z: i32| x + y + z in
        inner(y)
    in
    map(outer, arr)

def reused_lambda(x: i32, arr1: [4]i32, arr2: [4]i32) ([4]i32, [4]i32) =
    let adder = |e: i32| e + x in
    let result1 = map(adder, arr1) in
    let result2 = map(adder, arr2) in
    (result1, result2)

def hof_chain(scale: i32, offset: i32, arr: [4]i32) i32 =
    let scaled = map(|x: i32| x * scale, arr) in
    let shifted = map(|x: i32| x + offset, scaled) in
    reduce(|a: i32, b: i32| a + b, 0, shifted)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let (a, b) = different_captures(1, 2, [1, 2, 3, 4]) in
    let c = nested_capture(10, [1, 2, 3, 4]) in
    let (d, e) = reused_lambda(5, [1, 2, 3, 4], [5, 6, 7, 8]) in
    let f = hof_chain(2, 10, [1, 2, 3, 4]) in
    @[f32.i32(a[0] + b[0] + c[0] + d[0] + e[0] + f), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Type Checking Errors
// =============================================================================

#[test]
fn test_type_errors() {
    // These fail during type checking, before monomorphization, so no entry points needed

    // Arrays of functions are not permitted
    assert!(
        should_fail_type_check(
            r#"
def test: [2](i32 -> i32) =
    [|x: i32| x + 1, |x: i32| x * 2]
"#
        ),
        "Should reject arrays of functions"
    );

    // Function from if expression
    assert!(
        should_fail_type_check(
            r#"
def choose(b: bool) (i32 -> i32) =
    if b then |x: i32| x + 1 else |x: i32| x * 2
"#
        ),
        "Should reject function returned from if expression"
    );

    // Loop parameter cannot be a function
    assert!(
        should_fail_type_check(
            r#"
def test: (i32 -> i32) =
    loop f = |x: i32| x while false do f
"#
        ),
        "Should reject function as loop parameter"
    );
}

/// Companion to `test_spirv_loop_carrying_map_over_iota`, with the
/// loop initialized from a composite ARRAY LITERAL rather than a
/// `map(…, iota(…))` call. Before the fix these had asymmetric
/// outcomes: literal-init produced a Composite variant, map-init
/// produced a Virtual variant, and the loop back-edge unification
/// failed with "Loop body type must match loop variable type:
/// Failure(Virtual, Composite)". Now that `map`'s output variant is
/// pinned to Composite, both styles compile.
#[test]
fn test_spirv_loop_carrying_literal_init() {
    let source = r#"
def f(seed: f32) [4]f32 =
    let init: [4]f32 = [seed, seed, seed, seed] in
    let (_, out) =
        loop (i, arr) = (0, init) while i < 2 do
            let arr' = map(|j: i32| arr[j] + 1.0, iota(4))
            in (i + 1, arr')
    in out

#[compute]
entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop back-edge carrying a literal-init array across a map(…, iota(…)) body \
         should compile; both init and body variants are Composite",
    );
}

/// Regression: `lift_graphical_invariant_soacs` used to look only at the
/// direct free vars of a candidate SOAC for entry-param refs. A
/// fragment-shader-local `let uv = fragCoord.x` introduces `uv` as a
/// fresh symbol that's *not* an entry param but transitively depends on
/// one. A reduce/redomap whose body reads `uv` would then be wrongly
/// classified as graphical-invariant and hoisted into a compute prepass
/// that references `@uv` as an unbound global — SPIR-V codegen panics
/// with `Unknown global: uv`. The check needs to follow let bindings
/// transitively.
#[test]
fn test_no_overhoist_redomap_through_let_bound_dependency() {
    let source = r#"
def cands: [12]i32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32)
  #[location(0)] vec4f32 =
  let uv = fragCoord.x in
  let glows = map(|i: i32| uv + f32.i32(i), cands) in
  let total = reduce(|a: f32, b: f32| a + b, 0.0, glows) in
  @[total, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect(
        "redomap whose body reads a let-bound local that transitively \
         depends on an entry param must remain in the fragment shader; \
         the lift pass must not classify it as graphical-invariant",
    );
}

/// Companion to the over-hoist test above: a reduce whose only
/// non-constant dependency is a `#[uniform]` param IS graphical-invariant
/// (a uniform is constant across invocations), so it must lift into a
/// compute pre-pass. Because `#[uniform]` is entry-param-only and the
/// lift's taint set treats every entry param as per-invocation, the lift
/// has to explicitly exempt uniform params — otherwise it silently stops
/// firing for the common uniform-driven case. And since the pre-pass is a
/// separate entry, it must re-declare the uniform as its own `#[uniform]`
/// param; without that, codegen panics with `Unknown global: iTime`.
#[test]
fn test_uniform_driven_reduce_lifts_into_prepass() {
    let source = r#"
#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32)
  #[builtin(position)] vec4f32 =
  @[-1.0, -1.0, 0.0, 1.0]

#[fragment]
entry fragment_main(
  #[uniform(set=1, binding=0)] iTime: f32,
  #[builtin(position)] fragCoord: vec4f32
) #[location(0)] vec4f32 =
  let samples = map(|i: i32| f32.cos(iTime + f32.i32(i)), 0..<64) in
  let breath = reduce(|a: f32, b: f32| a + b, 0.0, samples) in
  @[breath, 0.0, 0.0, 1.0]
"#;

    // The lift must fire: the uniform-driven reduce becomes a compute
    // pre-pass (multi-staged into phase1 + phase2 compute entries). Before
    // the uniform exemption there were no compute entries at all.
    let ssa = compile_to_ssa_with_modules(source);
    let compute_entries: Vec<&str> = ssa
        .entry_points
        .iter()
        .filter(|e| {
            matches!(
                e.execution_model,
                crate::ssa::types::ExecutionModel::Compute { .. }
            )
        })
        .map(|e| e.name.as_str())
        .collect();
    assert!(
        compute_entries.iter().any(|n| n.contains("prepass")),
        "uniform-driven reduce should lift into a compute pre-pass; \
         compute entries were {:?}",
        compute_entries
    );

    // ...and the lifted result must still compile: the pre-pass re-declares
    // iTime as its own uniform, so codegen resolves it (no `Unknown global`).
    compile_to_spirv(source).expect(
        "a fragment reduce depending only on a uniform must lift into a \
         compute pre-pass that re-declares the uniform and compiles to SPIR-V",
    );
}

// =============================================================================
// Materialization Optimization
// =============================================================================

#[test]
fn test_materialization_optimization() {
    // Tests that materialization hoisting works correctly
    let _ssa = compile_to_ssa(
        r#"
def identity(arr: [3]i32) [3]i32 = arr

def no_redundant_complex(arr: [3]i32, i: i32) i32 =
    if true then (identity(arr))[i] else (identity(arr))[i]

def no_materialize_tuple(x: i32) i32 =
    let pair = (x, x + 1) in
    let (a, b) = pair in
    a + b

def no_materialize_loop_tuple(arr: [10]i32) i32 =
    let (sum, _) = loop (acc, i) = (0, 0) while i < 10 do
        (acc + arr[i], i + 1)
    in sum

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = no_redundant_complex([1, 2, 3], 0) in
    let b = no_materialize_tuple(5) in
    let c = no_materialize_loop_tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Math Functions and Conversions
// =============================================================================

#[test]
fn test_math_and_conversions() {
    // Tests: f32 conversions, math operations, qualified names
    let _ssa = compile_to_ssa(
        r#"
def conversions(x: i32, y: i64) f32 =
    let f1 = f32.i32(x) in
    let f2 = f32.i64(y) in
    f1 + f2

def math_ops(x: f32) f32 =
    let a = f32.sin(x) in
    let b = f32.cos(x) in
    let c = f32.sqrt(a) in
    let d = f32.exp(b) in
    let e = f32.log(c) in
    let f = d ** 2.0f32 in
    let g = f32.sinh(x) in
    let h = f32.asinh(g) in
    let i = f32.atan2(x, a) in
    f32.fma(f, e, i)

def vector_length(v: vec2f32) f32 =
    f32.sqrt(v.x * v.x + v.y * v.y)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = conversions(1, 2i64) in
    let b = math_ops(1.0) in
    let c = vector_length(@[3.0, 4.0]) in
    @[a + b + c, 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Matrix Operations
// =============================================================================

#[test]
fn test_matrix_operations() {
    // Tests: mul overloads (mat*mat, mat*vec, vec*mat)
    let _ssa = compile_to_ssa(
        r#"
def test_mul(m1: mat4f32, m2: mat4f32, v: vec4f32) vec4f32 =
    let mat_result = mul(m1, m2) in
    let vec_result1 = mul(mat_result, v) in
    let vec_result2 = mul(v, m1) in
    vec_result1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let m = @[[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]] in
    let v = @[1.0, 2.0, 3.0, 1.0] in
    test_mul(m, m, v)
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Complex Shader Integration
// =============================================================================

#[test]
fn test_complex_shader() {
    // Full shader with uniforms, matrices, map, multiple functions
    let _ssa = compile_to_ssa(
        r#"
def verts: [3]vec4f32 =
    [@[-1.0, -1.0, 0.0, 1.0],
     @[3.0, -1.0, 0.0, 1.0],
     @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32) #[builtin(position)] vec4f32 =
    verts[vertex_id]

def translation(p: vec3f32) mat4f32 =
    @[[1.0f32, 0.0f32, 0.0f32, p.x],
      [0.0f32, 1.0f32, 0.0f32, p.y],
      [0.0f32, 0.0f32, 1.0f32, p.z],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def rotation_y(angle: f32) mat4f32 =
    let s = f32.sin(angle) in
    let c = f32.cos(angle) in
    @[[c, 0.0f32, s, 0.0f32],
      [0.0f32, 1.0f32, 0.0f32, 0.0f32],
      [0.0 - s, 0.0f32, c, 0.0f32],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def cube_corners: [8]vec3f32 =
    [@[-1.0, -1.0, 1.0], @[-1.0, 1.0, 1.0],
     @[1.0, 1.0, 1.0], @[1.0, -1.0, 1.0],
     @[-1.0, -1.0, -1.0], @[-1.0, 1.0, -1.0],
     @[1.0, 1.0, -1.0], @[1.0, -1.0, -1.0]]

def main_image(res: vec2f32, time: f32, fragCoord: vec2f32) vec4f32 =
    let cam = translation(@[0.0, 0.0, 10.0]) in
    let rot = rotation_y(time) in
    let mat = rot * cam in
    let v4s = map(|v: vec3f32| @[v.x, v.y, v.z, 1.0] * mat, cube_corners) in
    v4s[0]

#[fragment]
entry fragment_main(#[uniform(set=1, binding=0)] iResolution: vec2f32, #[uniform(set=1, binding=1)] iTime: f32, #[builtin(frag_coord)] pos: vec4f32) #[location(0)] vec4f32 =
    main_image(@[iResolution.x, iResolution.y], iTime, @[pos.x, pos.y])
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Full Pipeline to SPIR-V
// =============================================================================

#[test]
fn test_function_call_with_array_arg() {
    // Test calling a function with an array literal argument
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let result = sum_first_two([1, 2, 3, 4]) in
    @[f32.i32(result), 0.0, 0.0, 1.0]
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn test_compute_shader_with_storage_slice() {
    // Test compute shader with storage buffer slice
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn test_full_pipeline_to_spirv() {
    // Verify the full pipeline compiles successfully to SPIR-V
    let source = r#"
def compute(x: f32, y: f32) f32 =
    let a = f32.sin(x) in
    let b = f32.cos(y) in
    a + b

#[fragment]
entry fragment_main(#[uniform(set=1, binding=0)] iTime: f32, #[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let s = compute(pos.x, pos.y) in
    @[s + iTime, 0.0, 0.0, 1.0]
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn mul_all_three_overloads_compile_to_spirv() {
    // `mul` has three overloads with three different `PrimOp`s
    // (MatrixTimesMatrix / MatrixTimesVector / VectorTimesMatrix).
    // `tlc::specialize` rewrites every `mul(a, b)` call into
    // `BinOp("*")(a, b)`; the BinOp lowering then picks the right
    // SPIR-V op based on operand shapes. This pins the wiring end-to-
    // end: a single shader exercising all three call shapes must
    // compile through to valid SPIR-V.
    let source = r#"
def m1 = @[[1.0f32, 0.0f32], [0.0f32, 1.0f32]]
def m2 = @[[2.0f32, 0.0f32], [0.0f32, 2.0f32]]

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let a: mat2f32 = m1 in
    let b: mat2f32 = m2 in
    let v: vec2f32 = @[pos.x, pos.y] in
    let mm: mat2f32 = mul(a, b) in
    let mv: vec2f32 = mul(mm, v) in
    let vm: vec2f32 = mul(v, mm) in
    @[mv.x, mv.y, vm.x, vm.y]
"#;
    let result = crate::compile_thru_spirv(source);
    assert!(
        result.is_ok(),
        "all three mul overloads should compile to SPIR-V: {:?}",
        result.err()
    );
}

// =============================================================================
// Array Variant Monomorphization
// =============================================================================

#[test]
fn test_array_variant_monomorphization() {
    // Slicing a view with constant bounds produces a Composite array (materialized at
    // the call site), so both call sites use the same Composite variant of sum_first_two.
    let ssa = compile_to_ssa(
        r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#,
    );

    // Collect all sum_first_two variants (including buffer-specialized)
    let sum_versions: Vec<_> =
        ssa.functions.iter().filter(|f| f.name.starts_with("sum_first_two")).collect();

    eprintln!("sum_first_two SSA functions:");
    for f in &sum_versions {
        eprintln!("  {}", f.name);
        // Show param types
        for (val, ty, name) in &f.body.params {
            eprintln!("    param {} ({:?}) :: {:?}", name, val, ty);
        }
        // Show all instructions that involve indexing or storage views
        for inst in f.body.inner.insts.values() {
            match &inst.data {
                crate::ssa::types::InstKind::Op {
                    tag: crate::op::OpTag::Index,
                    ..
                } => {
                    eprintln!("    inst {:?}: Index", inst.result);
                }
                crate::ssa::types::InstKind::Op {
                    tag: crate::op::OpTag::StorageView(_),
                    ..
                } => {
                    eprintln!("    inst {:?}: StorageView", inst.result);
                }
                crate::ssa::types::InstKind::ViewIndex { .. } => {
                    eprintln!("    inst {:?}: ViewIndex", inst.result);
                }
                _ => {}
            }
        }
    }

    // After TLC-level inlining and DCE, sum_first_two may be fully inlined
    // at all call sites and eliminated. The important thing is that the program
    // compiles successfully to SSA — buffer specialization is tested implicitly.
    // (The function may or may not survive depending on inlining thresholds.)
}

// =============================================================================
// SPIR-V Block Param / Phi Node Tests
// =============================================================================

/// Compile source all the way through SPIR-V and return Ok/Err.
fn compile_to_spirv(input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(crate::compile_thru_spirv(input)?.spirv)
}

/// Helper: compile source through SSA. Same as `compile_to_ssa`; kept as
/// a separate name because some legacy tests distinguished module-bearing
/// programs from non-module ones — `compile_thru_frontend` handles both.
fn compile_to_ssa_with_modules(input: &str) -> Program {
    crate::compile_thru_ssa(input).expect("compile to SSA").ssa
}

/// Verify that nested if/else chains compile to SPIR-V.
#[test]
fn test_spirv_nested_if_else_block_params() {
    let source = r#"
def choose(a: f32, b: f32, c: f32, sel1: i32, sel2: i32) f32 =
    let x = if sel1 == 0 then a
            else if sel1 == 1 then b
            else c in
    let y = if sel2 == 0 then a
            else if sel2 == 1 then c
            else b in
    x + y

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = choose(pos.x, pos.y, pos.z, 1, 2) in
    @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Nested if/else should compile to SPIR-V");
}

/// Verify many conditional branches producing block params compile to SPIR-V.
#[test]
fn test_spirv_many_conditional_block_params() {
    let source = r#"
def process(a: f32, b: f32, c: f32, d: f32, flag: i32) (f32, f32, f32, f32) =
    let x = if flag == 0 then a + b else a - b in
    let y = if flag == 1 then b + c else b * c in
    let z = if flag == 2 then c + d else c - d in
    let w = if flag == 0 then d * a else d + a in
    (x, y, z, w)

def combine(t: (f32, f32, f32, f32)) f32 =
    let (a, b, c, d) = t in
    a + b + c + d

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let result = process(pos.x, pos.y, pos.z, pos.w, 1) in
    let s = combine(result) in
    @[s, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Many conditionals should compile to SPIR-V");
}

/// Verify maps over small arrays with nested conditionals compile to SPIR-V.
#[test]
fn test_spirv_map_with_nested_conditionals() {
    let source = r#"
def selectValue(x: f32, flag: i32) f32 =
    if flag == 0 then x * 2.0
    else if flag == 1 then x + 1.0
    else x - 1.0

#[compute]
entry compute_main(data: [8]f32) [8]f32 =
    map(|x| selectValue(x, 1), data)
"#;
    compile_to_spirv(source).expect("Map with nested conditionals should compile to SPIR-V");
}

/// Verify multiple maps followed by a reduce compile to SPIR-V.
#[test]
fn test_spirv_multiple_maps_and_reduce() {
    let source = r#"
#[compute]
entry compute_main(data: [8](f32, f32)) [8]f32 =
    let first = map(|t| let (a, _) = t in a, data) in
    let second = map(|t| let (_, b) = t in b, data) in
    let combined = map(|(a, b): (f32, f32)| a + b, zip(first, second)) in
    let total = reduce(|a: f32, b: f32| a + b, 0.0, combined) in
    map(|x| x + total, combined)
"#;
    compile_to_spirv(source).expect("Multiple maps + reduce should compile to SPIR-V");
}

/// Verify conditional array element selection compiles to SPIR-V
/// (the finalOrigins/finalDirs pattern from raytrace.wyn).
#[test]
fn test_spirv_conditional_array_construction() {
    let source = r#"
def build(a: [4]f32, b: [4]f32, flags: [4]i32) [4]f32 =
    [
        if flags[0] == 1 then b[0] else a[0],
        if flags[1] == 1 then b[1] else a[1],
        if flags[2] == 1 then b[2] else a[2],
        if flags[3] == 1 then b[3] else a[3]
    ]

#[compute]
entry compute_main(data: [4]f32) [4]f32 =
    let doubled = map(|x| x * 2.0, data) in
    let flags = [1, 0, 1, 0] in
    build(data, doubled, flags)
"#;
    compile_to_spirv(source).expect("Conditional array construction should compile to SPIR-V");
}

/// Regression: mapping a lambda whose return type is a mixed
/// scalar/vector tuple. The SoA transform rewrites the output
/// `[N](f32, i32, vec3f32)` into a tuple-of-arrays before EGIR
/// conversion; `egir::soac_expand` must split the per-iteration
/// ArrayWith into per-component ArrayWith calls + a Tuple repack.
/// Without the split, SPIR-V lowering's runtime-index path hit a
/// cache miss and failed with "element type not found".
#[test]
fn test_spirv_map_array_of_mixed_tuple() {
    let source = r#"
def build(xs: [8]f32) [8](f32, i32, vec3f32) =
    map(|x: f32| (x + 1.0, 0, @[x, x, x]), xs)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) in
    let (a, _, v) = arr[3] in
    @[a, v.x, v.y, v.z]
"#;
    compile_to_spirv(source).expect("map over [N](f32, i32, vec3f32) should compile to SPIR-V");
}

/// Regression: nested SoA. The element type itself contains a
/// composite array, so the SoA transform produces a tuple of
/// arrays whose components are themselves arrays — exercising
/// `emit_write_element`'s recursion through `soa_element_type`.
#[test]
fn test_spirv_map_array_of_nested_tuple() {
    let source = r#"
def build(xs: [4]f32) [4](f32, [3]f32) =
    map(|x: f32| (x + 1.0, [x, x, x]), xs)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0]) in
    let (a, inner) = arr[2] in
    @[a, inner[0], inner[1], inner[2]]
"#;
    compile_to_spirv(source).expect("map over [N](f32, [M]f32) should compile to SPIR-V");
}

/// Regression: a loop that carries an array whose next-iteration
/// value comes from `map(…, iota(N))`.
///
/// Before the fix, `map`'s type scheme claimed the output variant
/// was the same as the input's. Since `iota` returns a Virtual
/// array, `map`'s output was typed Virtual — but `egir::soac_expand`
/// actually materializes the result via `_w_intrinsic_uninit` +
/// `_w_intrinsic_array_with_inplace`, which is a Composite
/// representation. That type/representation mismatch surfaced as
/// a SPIR-V "ArrayWith: element type not found" cache miss when
/// the loop back-edge carried what SSA thought was a Virtual
/// array. Fix: pin `map` / `scan` / `filter` output variant to
/// Composite in the type scheme, matching the runtime
/// representation.
#[test]
fn test_spirv_loop_carrying_map_over_iota() {
    let source = r#"
def f(seed: f32) [4]f32 =
  let init: [4]f32 = map(|j: i32| seed + f32.i32(j), iota(4)) in
  let (_, out) =
    loop (i, arr) = (0, init) while i < 2 do
      let arr' = map(|j: i32| arr[j] + 1.0, iota(4))
      in (i + 1, arr')
  in out

#[compute]
entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop carrying `map(..., iota(N))` should compile; currently fails with \
         ArrayWith cache miss because the back-edge array is Virtual variant",
    );
}

/// Test the specific raytrace.wyn file compiles to SPIR-V.
#[test]
fn test_spirv_raytrace() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../testfiles/playground/raytrace.wyn"
    ))
    .expect("Could not read testfiles/playground/raytrace.wyn");
    compile_to_spirv(&source).expect("raytrace.wyn should compile to SPIR-V");
}

/// Regression: if/else before interprocedural map+reduce fusion caused
/// UnterminatedBlock in soac_lower. The if/else creates a dead Unreachable
/// block in SSA, which soac_lower's rebuild would pre-create but finish()
/// rejected because Unreachable doubles as the "unterminated" sentinel.
#[test]
fn test_interproc_fusion_if_before_fused_reduce() {
    let source = r#"
def maxDist: f32 = 100.0
def globalData: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

def producer(x: f32) [12]f32 =
  map(|a: f32| a * x, globalData)

def consumer(arr: [12]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, maxDist, arr)

def scene(x: f32, y: f32) f32 =
  let ground = if y > 0.0 then y else maxDist in
  let hits = producer(x) in
  let closest = consumer(hits) in
  closest + ground

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let r = scene(1.0, 0.5) in
  @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("if-before-interproc-fusion should compile");
}

/// Verify raytrace.wyn compiles through SSA to SPIR-V without errors.
/// This exercises the RPO block emission and incremental array literal
/// lowering that were needed for complex cross-block value references.
/// (test_spirv_raytrace covers this; this test verifies the SSA is well-formed
/// enough that compile_to_ssa_with_modules succeeds.)
#[test]
fn test_ssa_raytrace_well_formed() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../testfiles/playground/raytrace.wyn"
    ))
    .expect("Could not read testfiles/playground/raytrace.wyn");

    let ssa = compile_to_ssa_with_modules(&source);

    // Verify key functions exist
    assert!(
        ssa.functions.iter().any(|f| f.name == "trace"),
        "trace should be in SSA output"
    );
}

// =============================================================================
// Constant Inlining
// =============================================================================

/// Constants that reference other constants should be fully inlined.
#[test]
fn test_constant_referencing_constant() {
    let source = r#"
def PI: f32 = 3.14159265
def TAU: f32 = PI * 2.0
def QUARTER_TAU: f32 = TAU / 4.0

#[fragment]
entry frag(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
  @[QUARTER_TAU, PI, TAU, 1.0]
"#;
    compile_to_spirv(source).expect("constants referencing constants should compile");
}

// `test_soa_eliminates_extraction_maps` was deleted: it inspected post-SSA for
// `InstKind::Soac(Soac::Map { .. })`, but SOACs no longer exist as SSA
// instructions — they're expanded inside EGIR (`egir::soac_expand`) before
// elaboration produces any SSA. The underlying optimization concern (extraction
// maps after a zip-map should fold to tuple projections) is now an EGIR-level
// question and would need a different shape of test.

/// Pipeline that includes TLC inline_small (the new pass).
fn compile_to_ssa_with_inline_small(input: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(input, &mut node_counter).expect("Parsing failed");
    let type_checked = parsed
        .resolve(&mut module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("Type checking failed");

    type_checked
        .to_tlc(&module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .ssa
}

#[test]
fn test_constant_inlining_global_ref() {
    // Minimal repro: a constant def used by a function, going through inline_small.
    // This should NOT produce an unresolved Global("PI") in SSA.
    let ssa = compile_to_ssa_with_inline_small(
        r#"
def PI: f32 = 3.141592

def use_pi(x: f32) f32 = x * PI

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = use_pi(pos.x) in
    @[r, 0.0, 0.0, 1.0]
"#,
    );

    // Dump what we got.
    eprintln!("{}", crate::ssa::print::format_program(&ssa));

    // Check that no Global("PI") instruction exists — it should have been inlined.
    for func in &ssa.functions {
        for (_id, inst) in &func.body.inner.insts {
            if let crate::ssa::types::InstKind::Op {
                tag: crate::op::OpTag::Global(name),
                ..
            } = &inst.data
            {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in function '{}'",
                    func.name
                );
            }
        }
    }
    for ep in &ssa.entry_points {
        for (_id, inst) in &ep.body.inner.insts {
            if let crate::ssa::types::InstKind::Op {
                tag: crate::op::OpTag::Global(name),
                ..
            } = &inst.data
            {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in entry '{}'",
                    ep.name
                );
            }
        }
    }
}

// ============================================================================
// `--fill-holes`: type-hole default fill
// ============================================================================

/// Compile through TLC with `fill_holes = true`; return the resulting
/// `TlcTransformed` so tests can inspect `fill_hole_errors` and the
/// program shape.
fn compile_tlc_with_fill_holes(input: &str) -> crate::TlcTransformed {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(input, &mut node_counter).expect("parse");
    let type_checked = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked.to_tlc(&module_manager, true)
}

#[test]
fn fill_holes_numeric_scalars_compile_clean() {
    // Scalar holes (i32 / f32 / bool) default to 0 / 0.0 / false and
    // compile through with no fill-hole errors.
    for src in ["def x: i32 = ???", "def y: f32 = ???", "def z: bool = ???"] {
        let tlc = compile_tlc_with_fill_holes(src);
        assert!(
            tlc.0.fill_hole_errors.is_empty(),
            "scalar hole in `{}` should fill cleanly: {:?}",
            src,
            tlc.0.fill_hole_errors
        );
    }
}

#[test]
fn fill_holes_vec_compiles_clean() {
    let tlc = compile_tlc_with_fill_holes("def v: vec3f32 = ???");
    assert!(
        tlc.0.fill_hole_errors.is_empty(),
        "vec3 hole should fill cleanly: {:?}",
        tlc.0.fill_hole_errors
    );
}

#[test]
fn fill_holes_rejects_function_type() {
    let tlc = compile_tlc_with_fill_holes("def f: i32 -> i32 = ???");
    assert!(
        !tlc.0.fill_hole_errors.is_empty(),
        "function-typed hole should surface a fill-hole error"
    );
    let msg = format!("{:?}", tlc.0.fill_hole_errors[0]);
    assert!(
        msg.contains("function value") || msg.contains("Arrow"),
        "error should mention function type: {}",
        msg
    );
}

#[test]
fn fill_holes_respects_inferred_type_from_context() {
    // Hole's type is inferred from the enclosing context (array
    // element type here). Default-fill fires at the inferred type.
    let tlc = compile_tlc_with_fill_holes("def arr: [3]i32 = [1i32, ???, 3i32]");
    assert!(
        tlc.0.fill_hole_errors.is_empty(),
        "hole in i32 array should fill as i32 cleanly: {:?}",
        tlc.0.fill_hole_errors
    );
}

// =============================================================================
// Known-failing tests for higher-order-function defunctionalization gaps
// =============================================================================
//
// Both currently fail. Keep them as guards: when the underlying bugs in
// defunctionalization / SPIR-V lowering are fixed, these tests start
// passing and the cleanup will be obvious.

/// Bug 1: A closure with captured free variables, passed to a user-
/// defined HOF, produces SPIR-V that fails `spirv-val` with
/// `OpFunctionCall Function <id>'s parameter count does not match the
/// argument count`. Defunc/mono specializes both the HOF and the
/// closure body correctly (each gets the captured-env params it
/// needs), but the call-sites for the function-typed parameter inside
/// the HOF body still pass only the original arguments — the captured
/// env never gets threaded through.
///
/// Minimal repro: a 2-arg HOF that calls its f-arg twice, with a
/// closure capturing two scalars from the entry's params.
#[test]
fn hof_closure_with_captures_lowers_to_valid_spirv() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

#[compute]
entry test(a: f32, b: f32) f32 =
  let g = |y: f32| y * y + a + b in
  apply2(g, 1.0f32, 2.0f32)
"#;
    let spirv = compile_to_spirv(src).expect("compile");
    assert_spirv_call_arities_match(&spirv);
}

/// Bug 2: An inline lambda with no captures, called from inside a
/// `map` body, panics in SPIR-V lowering with "BUG: Unknown type
/// reached lowering: Ignored". The HOF passed the lambda survives
/// type checking but the elaborated function's signature carries an
/// `Ignored` placeholder where the lambda's argument type should be —
/// presumably defunctionalization or the elaborator is leaving the
/// inner call's type unresolved.
///
/// Minimal repro: a HOF taking `f32 -> f32`, called inside a `map`
/// over `[]f32`, with a closure-free inline lambda.
#[test]
fn hof_no_capture_lambda_in_map_body_lowers_without_panic() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

#[compute]
entry test(in_arr: []f32) []f32 =
  map(|x: f32| apply2(|y: f32| y * y, x, x + 1.0f32), in_arr)
"#;
    // catch_unwind because the bug surfaces as a panic in
    // spirv/mod.rs (not a Result::Err).
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_to_spirv(src)));
    let bytes = result.expect("compilation panicked — see Bug 2 docstring").expect("compile returned Err");
    assert_spirv_call_arities_match(&bytes);
}

// =============================================================================
// Sum-type lowering (Phase C) integration tests
// =============================================================================

/// Build a sum value with one constructor and select on it. Exercises
/// constructor-expression → flattened-tuple lowering and match →
/// tag-checked if-chain lowering, end-to-end through SPIR-V.
#[test]
fn sum_type_lowering_compiles_to_spirv() {
    let src = r#"
def pick(v: #left(f32) | #right(f32)) f32 =
    match v
    case #left(x) -> x + 1.0f32
    case #right(y) -> y * 2.0f32

#[fragment]
entry main() #[location(0)] vec4f32 =
    let a = pick(#left(0.5f32)) in
    let b = pick(#right(0.25f32)) in
    @[a, b, 0.0f32, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("sum-type program should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Multi-payload constructor with mixed arities: `#point(f32, f32)`
/// and a nullary `#origin`. Verifies that the flattened-no-sharing
/// layout zero-fills dead slots in the nullary case.
#[test]
fn sum_type_multi_payload_compiles_to_spirv() {
    let src = r#"
def length_sq(p: #point(f32, f32) | #origin) f32 =
    match p
    case #point(x, y) -> x * x + y * y
    case #origin -> 0.0f32

#[fragment]
entry main() #[location(0)] vec4f32 =
    let a = length_sq(#point(3.0f32, 4.0f32)) in
    let b = length_sq(#origin) in
    @[a, b, 0.0f32, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("multi-payload sum should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

// =============================================================================
// Swizzle-with lowering (Phase C) integration tests
// =============================================================================

/// Plain `=` swizzle update: write `e` into v.yz, leaving v.x intact.
#[test]
fn swizzle_with_plain_assign_compiles_to_spirv() {
    let src = r#"
def update(v: vec3f32, e: vec2f32) vec3f32 = v with .yz = e

#[fragment]
entry main() #[location(0)] vec4f32 =
    let v = update(@[1.0f32, 2.0f32, 3.0f32], @[20.0f32, 30.0f32]) in
    @[v.x, v.y, v.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("plain swizzle-with should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Compound `*=` swizzle update: vec2 × mat2 multiply, written into
/// v.yz. Exercises the binary-op path inside transform_vec_with.
#[test]
fn swizzle_with_compound_mul_compiles_to_spirv() {
    let src = r#"
def update(v: vec3f32, m: mat2f32) vec3f32 = v with .yz *= m

#[fragment]
entry main() #[location(0)] vec4f32 =
    let m: mat2f32 = @[[1.0f32, 0.0f32], [0.0f32, 1.0f32]] in
    let v = update(@[1.0f32, 2.0f32, 3.0f32], m) in
    @[v.x, v.y, v.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("compound swizzle-with should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Four chained `with .swizzle *= mat2` rotations on a direction vector.
#[test]
fn swizzle_with_chained_rotations_compiles_to_spirv() {
    let src = r#"
def rot(a: f32) mat2f32 =
    let c = f32.cos(a) in
    let s = f32.sin(a) in
    @[[c, s], [0.0f32 - s, c]]

def transform(dir0: vec3f32, mx: f32, my: f32) vec3f32 =
    let d1 = dir0 with .yz *= rot(my) in
    let d2 = d1 with .xz *= rot(mx) in
    let d3 = d2 with .yz *= rot(my) in
    d3 with .xz *= rot(mx)

#[fragment]
entry main() #[location(0)] vec4f32 =
    let d = transform(@[0.0f32, 0.0f32, 1.0f32], 0.5f32, 0.3f32) in
    @[d.x, d.y, d.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("chained swizzle-with rotations should compile");
    assert_spirv_call_arities_match(&spirv);
}

/// Walk every `OpFunctionCall` in a SPIR-V module and assert each
/// call's argument count matches the called function's declared
/// parameter count. The arity-mismatch class of bug above produces
/// SPIR-V that round-trips through rspirv but fails this invariant.
fn assert_spirv_call_arities_match(spirv_words: &[u32]) {
    use rspirv::binary::parse_words;
    use rspirv::dr::{Loader, Operand};
    use rspirv::spirv::Op;

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    // Map function id → declared parameter count.
    let mut arities: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for func in &module.functions {
        if let Some(def) = &func.def {
            if let Some(Operand::IdRef(_)) = def.result_id.map(Operand::IdRef) {}
            if let Some(id) = def.result_id {
                arities.insert(id, func.parameters.len());
            }
        }
    }

    // Walk every block's instructions for OpFunctionCall.
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == Op::FunctionCall {
                    // Operands: [0] callee IdRef, [1..] argument IdRefs.
                    let callee = match inst.operands.first() {
                        Some(Operand::IdRef(id)) => *id,
                        _ => continue,
                    };
                    let arg_count = inst.operands.len() - 1;
                    let expected = match arities.get(&callee) {
                        Some(n) => *n,
                        None => continue, // external call (e.g. GlslExt)
                    };
                    assert_eq!(
                        arg_count, expected,
                        "OpFunctionCall to function %{} passes {} args but the function declares {} parameters",
                        callee, arg_count, expected
                    );
                }
            }
        }
    }
}

// =============================================================================
// EGIR-side Map parallelization regression tests
// =============================================================================

/// A compute entry whose body is `map(f, xs)` should emit a kernel that
/// loads `gl_GlobalInvocationID` (lane-indexed access) — not a serial
/// driver loop over `0..N`.
#[test]
fn compute_map_loads_global_invocation_id() {
    use rspirv::binary::parse_words;
    use rspirv::dr::{Loader, Operand};
    use rspirv::spirv::Op;
    let src = r#"
#[compute]
entry sq(xs: []f32) []f32 = map(|x: f32| x * x, xs)
"#;
    let spirv = compile_to_spirv(src).expect("map compute compiles");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    // Find the gl_GlobalInvocationID input variable's id from the
    // EntryPoint interface (3rd-and-later operands of OpEntryPoint).
    let entry = module.entry_points.iter().find(|i| {
        if let Some(Operand::LiteralString(name)) = i.operands.get(2) { name == "sq" } else { false }
    });
    let entry = entry.expect("entry sq present");
    let func_id = match entry.operands.get(1) {
        Some(Operand::IdRef(id)) => *id,
        _ => panic!("entry has function id"),
    };

    // gl_GlobalInvocationID is the Input variable decorated with
    // BuiltIn GlobalInvocationId.
    // OpDecorate operand layout: [target_id, Decoration, *literals].
    // For BuiltIn the literal at operand 2 is the BuiltIn kind.
    let gid_var = module
        .annotations
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::Decorate
                && matches!(
                    inst.operands.get(2),
                    Some(Operand::BuiltIn(rspirv::spirv::BuiltIn::GlobalInvocationId))
                )
        })
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        })
        .expect("gl_GlobalInvocationID decoration present");

    // The entry function body must contain an OpLoad whose pointer is
    // the gl_GlobalInvocationID variable.
    let func = module
        .functions
        .iter()
        .find(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(func_id))
        .expect("entry function present");
    let loads_gid = func.blocks.iter().any(|b| {
        b.instructions.iter().any(|inst| {
            inst.class.opcode == Op::Load
                && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if *id == gid_var)
        })
    });
    assert!(
        loads_gid,
        "compute Map entry must OpLoad gl_GlobalInvocationID; got serial loop instead"
    );
}

/// A compute entry whose body is `map(f, xs)` should not contain an
/// `OpLoopMerge` — the parallel kernel is a single guarded scalar
/// branch. Inner function loops (e.g. raymarch) are not affected.
#[test]
fn compute_map_has_no_full_serial_loop() {
    use rspirv::binary::parse_words;
    use rspirv::dr::{Loader, Operand};
    use rspirv::spirv::Op;
    let src = r#"
#[compute]
entry sq(xs: []f32) []f32 = map(|x: f32| x * x, xs)
"#;
    let spirv = compile_to_spirv(src).expect("map compute compiles");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let entry = module
        .entry_points
        .iter()
        .find(|i| matches!(i.operands.get(2), Some(Operand::LiteralString(n)) if n == "sq"));
    let func_id = match entry.and_then(|i| i.operands.get(1)) {
        Some(Operand::IdRef(id)) => *id,
        _ => panic!("entry sq not found"),
    };
    let func = module
        .functions
        .iter()
        .find(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(func_id))
        .expect("entry function present");
    let has_loop_merge =
        func.blocks.iter().any(|b| b.instructions.iter().any(|inst| inst.class.opcode == Op::LoopMerge));
    assert!(
        !has_loop_merge,
        "compute Map entry must NOT contain OpLoopMerge — the parallel kernel is loop-free"
    );
}

/// Compile `source` through the full *parallelized* pipeline (matching the
/// production driver, which always parallelizes compute) and return the
/// lowered SPIR-V + pipeline descriptor.
fn compile_parallel(source: &str) -> crate::Lowered {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let type_checked = Compiler::parse(source, &mut node_counter)
        .expect("parse")
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked
        .to_tlc(&module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        // `parallelize_soacs` takes a *disable* flag; `false` enables it,
        // matching the production driver's default (non-`--single-stage`).
        .parallelize_soacs(false)
        .expect("parallelize_soacs")
        .filter_reachable()
        .to_egraph()
        .expect("to_egraph")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .lower()
        .expect("lower")
}

/// Full storage-buffer descriptors of a compute pipeline.
fn compute_storage_buffers(
    pipeline: &crate::pipeline_descriptor::PipelineDescriptor,
    entry: &str,
) -> Vec<crate::pipeline_descriptor::Binding> {
    use crate::pipeline_descriptor::{Binding, Pipeline};
    pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(cp) if cp.entry_point == entry => Some(cp),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no compute pipeline named {entry}"))
        .bindings
        .iter()
        .filter(|b| matches!(b, Binding::StorageBuffer { .. }))
        .cloned()
        .collect()
}

/// Gathering a *computed* array (a `map` result) at a runtime index used to
/// panic in SPIR-V emission ("Composite variant unsized arrays not
/// supported"). `lift_gathers` splits the producer `map` into its own
/// `gen_gather_0` compute stage writing a storage buffer, and rewrites the
/// consumer's `counts[i]` into a load from that buffer. This pins the
/// end-to-end wiring: both stages agree on the gather buffer's binding, it's a
/// compiler-managed Intermediate (not host I/O), it doesn't collide with the
/// consumer's own input/output, and it carries a `LikeInput` sizing policy so
/// the host allocates it from `bh`'s length (a `map` preserves element count;
/// `[]vec4f32` → `[]i32` is 4 of 16 bytes per element).
#[test]
fn gather_computed_array_materializes_to_shared_intermediate() {
    use crate::pipeline_descriptor::{Access, Binding, BufferLen, BufferUsage};
    let src = "\
#[compute]
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  map(|i:i32| counts[i % 256], iota(6144))
";
    let lowered = compile_parallel(src);
    assert!(!lowered.spirv.is_empty(), "lowering produced no SPIR-V");

    let gather_entry = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            crate::pipeline_descriptor::Pipeline::Compute(cp) if cp.entry_point.contains("_gather_") => {
                Some(cp.entry_point.clone())
            }
            _ => None,
        })
        .expect("a gather pre-pass compute stage must exist");

    let gather_bufs = compute_storage_buffers(&lowered.pipeline, &gather_entry);
    let consumer_bufs = compute_storage_buffers(&lowered.pipeline, "gen");

    // The pre-pass writes exactly one Intermediate (the gather buffer), sized
    // LikeInput of `bh` (binding 0): one i32 (4B) per vec4f32 (16B) element.
    let producer_intermediates: Vec<&Binding> = gather_bufs
        .iter()
        .filter(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        producer_intermediates.len(),
        1,
        "pre-pass writes one gather intermediate: {gather_bufs:?}"
    );
    let Binding::StorageBuffer {
        binding: gather_binding,
        access,
        length,
        ..
    } = producer_intermediates[0]
    else {
        unreachable!()
    };
    assert_eq!(*access, Access::WriteOnly, "pre-pass writes the gather buffer");
    assert_eq!(
        length.as_ref(),
        Some(&BufferLen::LikeInput {
            set: 0,
            binding: 0,
            elem_bytes: 4,
            src_elem_bytes: 16,
        }),
        "gather buffer must be sized from its input array's element count"
    );

    // The consumer reads that same binding as a read-only Intermediate.
    let reads_gather = consumer_bufs.iter().any(|b| {
        matches!(b, Binding::StorageBuffer { binding, usage: BufferUsage::Intermediate, access: Access::ReadOnly, .. } if binding == gather_binding)
    });
    assert!(
        reads_gather,
        "consumer must read the gather buffer (binding {gather_binding}) read-only: {consumer_bufs:?}"
    );

    // The consumer's own output goes to a different binding — no collision.
    let consumer_outputs: Vec<u32> = consumer_bufs
        .iter()
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Output,
                ..
            } => Some(*binding),
            _ => None,
        })
        .collect();
    assert_eq!(
        consumer_outputs.len(),
        1,
        "consumer writes one output: {consumer_bufs:?}"
    );
    assert_ne!(
        consumer_outputs[0], *gather_binding,
        "consumer output must not collide with the gather buffer"
    );
}

// ---------------------------------------------------------------------------
// Multi-consumer gather regression
// ---------------------------------------------------------------------------
//
// `lift_gathers` handles a computed array `counts = map(...)` consumed by a
// single downstream SOAC/gather. Whenever `counts` has *two or more* downstream
// consumers (e.g. a `reduce` plus a `scan`, or a `scan` plus a direct
// `counts[i % N]` gather in the same lambda), the lift currently leaves the
// in-register Composite array in place and the SPIR-V backend panics at
// `spirv/mod.rs:374` ("Composite variant unsized arrays not supported"). The
// single-consumer controls below pin the working baseline; the
// `#[ignore]`-marked tests capture the multi-consumer bug — run with
// `cargo test -- --ignored` to verify the panic and remove the `#[ignore]`
// once the lift threads the same intermediate buffer to every consumer.

/// Control: a single `scan` consumer of a computed `counts` map lifts cleanly.
#[test]
fn single_consumer_scan_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts  = map(|x: i32| x * 2, xs) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| offsets[i % 8], iota(64))
",
    )
    .expect("single-consumer scan-over-map must lift + compile");
}

/// Control: a single `reduce` consumer of a computed `counts` map lifts cleanly.
#[test]
fn single_consumer_reduce_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts = map(|x: i32| x * 2, xs) in
  let total  = reduce(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| total, iota(64))
",
    )
    .expect("single-consumer reduce-over-map must lift + compile");
}

/// When `counts` is consumed by **both** a `reduce` and a `scan` (and a
/// downstream gather then reads the scan result), `lift_gathers` materializes
/// `counts` into one shared gather buffer that both downstream SOACs read from.
#[test]
fn multi_consumer_scan_plus_reduce_lifts() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts  = map(|x: i32| x * 2, xs) in
  let total   = reduce(|a: i32, b: i32| a + b, 0, counts) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| offsets[i % 8] + total, iota(64))
",
    )
    .expect("multi-consumer (reduce + scan over the same counts) should lift + compile");
}

/// `counts` consumed by both `scan(counts)` and a direct random gather
/// `counts[i % 8]`. The scan's input is a SOAC edge in the producer graph;
/// the `counts[i % 8]` reference inside the outer map's lambda body is *not*
/// a SOAC edge. With the use-count fix in `producer_graph` counting every
/// `Var(counts)` reference (not just SOAC edges), fusion sees `counts` as
/// multi-use and declines to fuse + drop the let, so `lift_gathers` handles
/// it as a normal multi-consumer let-bound producer.
#[test]
fn multi_consumer_scan_plus_gather_lifts() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts  = map(|x: i32| x * 2, xs) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| offsets[i % 8] + counts[i % 8], iota(64))
",
    )
    .expect("multi-consumer (scan + direct gather of the same counts) should lift + compile");
}

/// Bisected min-repro: a fused scan whose op-lambda calls a user helper
/// function (`box_count`) and whose result is randomly indexed
/// (`offsets[nb - 1]`). Prior to the fix, `lift_gathers::free_symbol_vars`
/// passed empty `known_defs` to `collect_free_vars`, so the top-level def
/// reference to `box_count` appeared as a "free var" — failing the
/// entry-param predicate and declining the lift. The scan then stayed in
/// `gen`'s body where `parallelize::make_scan_plan` fell back to the serial
/// pipeline, and EGIR's `is_handleable_soac` rejected the resulting
/// `Scan { destination: Fresh, input: View }` combination → unexpanded
/// `PendingSoac` panic at `egir/elaborate.rs:314`. With the fix, top-level
/// defs are filtered out of the predicate, the scan lifts into a gather
/// pre-pass, and the rest of the pipeline handles it normally.
#[test]
fn fused_scan_helper_call_then_indexed_read_compiles() {
    compile_to_spirv(
        "\
def win_count(hw: f32) i32 =
  let span = 2.0 * hw - 1.0 in
  let fit  = i32.f32(floor(span / 2.4)) in
  if fit < 0 then 0 else if fit > 3 then 3 else fit

def box_count(hw: f32) i32 = 8 + 5 * win_count(hw)

#[compute]
entry gen(bh: []vec4f32, #[uniform(set=1,binding=0)] nb: i32) [1]i32 =
  let counts  = map(|h: vec4f32| box_count(h.x), bh) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  [if nb <= 0 then 0 else offsets[nb - 1]]
",
    )
    .expect("fused-scan-of-helper-mapping with indexed scan read should compile");
}

/// A `scan` producer gathers the same way a `map` does: it's lifted into its
/// own pre-pass (here a multi-stage parallel scan) writing the gather buffer,
/// which the consumer reads via `storage_index`. The forced-output binding is
/// honored uniformly across SOAC kinds, so the scan's final output lands on
/// the buffer the consumer reads, and the scan's own intermediates (block
/// sums/offsets) sit above it without collision.
#[test]
fn gather_scan_producer_materializes_to_shared_intermediate() {
    use crate::pipeline_descriptor::{Access, Binding, BufferLen, BufferUsage, Pipeline};
    let src = "\
#[compute]
entry g(xs: []i32) []i32 =
  let o = scan(|a:i32,b:i32| a+b, 0, xs) in
  map(|i:i32| o[i % 256], iota(6144))
";
    let lowered = compile_parallel(src);
    assert!(!lowered.spirv.is_empty(), "lowering produced no SPIR-V");

    // The consumer reads the gather buffer as a read-only Intermediate sized
    // LikeInput of `xs` (scan preserves element count and type: i32 → i32).
    let consumer_bufs = compute_storage_buffers(&lowered.pipeline, "g");
    let gather = consumer_bufs
        .iter()
        .find_map(|b| match b {
            Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Intermediate,
                access: Access::ReadOnly,
                length: Some(len),
                ..
            } => Some((*binding, len.clone())),
            _ => None,
        })
        .expect("consumer must read a sized gather intermediate");
    assert_eq!(
        gather.1,
        BufferLen::LikeInput {
            set: 0,
            binding: 0,
            elem_bytes: 4,
            src_elem_bytes: 4,
        }
    );

    // The gather pre-pass is a multi-stage parallel scan that writes that same
    // binding as its result, with its block-sum/offset intermediates above it.
    let scan_pipeline = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::MultiCompute(mc) if mc.stages.iter().any(|s| s.entry_point.contains("_gather_")) => {
                Some(mc)
            }
            _ => None,
        })
        .expect("scan gather pre-pass must be a multi_compute pipeline");
    let writes_gather = scan_pipeline
        .bindings
        .iter()
        .any(|b| matches!(b, Binding::StorageBuffer { binding, .. } if *binding == gather.0));
    assert!(
        writes_gather,
        "scan pre-pass must write the gather buffer (binding {})",
        gather.0
    );
    // No other binding in the scan pipeline collides with the gather output.
    let dup = scan_pipeline
        .bindings
        .iter()
        .filter(|b| matches!(b, Binding::StorageBuffer { binding, .. } if *binding == gather.0))
        .count();
    assert_eq!(
        dup, 1,
        "exactly one scan binding is the gather buffer: {:?}",
        scan_pipeline.bindings
    );
}

/// Multiple gathers of the *same* computed array coalesce to one buffer: the
/// lift keys on the let-bound symbol and rewrites every `arr[..]` use to the
/// same storage binding, so a single pre-pass materializes `arr` once no
/// matter how many times (or in how many consumer maps) it's indexed.
#[test]
fn gather_same_array_coalesces_to_one_buffer() {
    use crate::pipeline_descriptor::{Binding, BufferUsage, Pipeline};
    let src = "\
#[compute]
entry gen(bh: []i32) []i32 =
  let arr = map(|x:i32| x + 1, bh) in
  map(|i:i32| arr[i % 256] + arr[(i + 1) % 256], iota(6144))
";
    let lowered = compile_parallel(src);
    assert!(!lowered.spirv.is_empty());

    // Exactly one gather pre-pass, despite two `arr[..]` uses.
    let gather_prepasses = lowered
        .pipeline
        .pipelines
        .iter()
        .filter(|p| matches!(p, Pipeline::Compute(cp) if cp.entry_point.contains("_gather_")))
        .count();
    assert_eq!(
        gather_prepasses, 1,
        "two uses of one computed array must share one gather pre-pass"
    );

    // The consumer references exactly one gather intermediate.
    let consumer_intermediates = compute_storage_buffers(&lowered.pipeline, "gen")
        .into_iter()
        .filter(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        consumer_intermediates, 1,
        "both gathers must read the same intermediate buffer"
    );
}

// =============================================================================
// Stage 0: multi-dimensional fixed-size composite locals
// =============================================================================
//
// `[N][M]T` composite literal declared as a local in a compute entry and
// indexed both with constants (→ OpCompositeExtract in SPIR-V) and with
// runtime values (→ OpAccessChain). Tests both the type-system + lowering
// path (already implemented across `parser → checker → tlc → egir → ssa →
// spirv/wgsl`) and the const-fold path that produces a single nested
// `OpConstantComposite`.

#[test]
fn multidim_composite_local_const_and_runtime_index() {
    use crate::ssa::types::InstKind;

    let src = r#"
        #[compute]
        entry pick_const() i32 =
            let m: [3][2]i32 = [[1, 2], [3, 4], [5, 6]] in
            m[1][0]

        #[compute]
        entry pick_runtime(i: i32, j: i32) i32 =
            let m: [3][2]i32 = [[1, 2], [3, 4], [5, 6]] in
            m[i][j]
    "#;

    let program = compile_to_ssa(src);

    // Both entries survive — neither got DCE'd or fused away.
    let entry_names: Vec<&str> = program.entry_points.iter().map(|e| e.name.as_str()).collect();
    assert!(
        entry_names.contains(&"pick_const") && entry_names.contains(&"pick_runtime"),
        "expected both entries in SSA; got {entry_names:?}",
    );

    // The const-index entry should produce zero `Op::Index` insts in its
    // body — folding reduces `m[1][0]` to the literal scalar `3`. The
    // runtime-index entry MUST still carry `Op::Index` operations, since
    // `i` and `j` are entry parameters and can't be folded away.
    let body_dyn_extracts = |ep_name: &str| -> usize {
        let ep = program.entry_points.iter().find(|e| e.name == ep_name).unwrap();
        ep.body
            .inner
            .insts
            .iter()
            .filter(|(_, inst)| {
                matches!(
                    &inst.data,
                    InstKind::Op {
                        tag: crate::op::OpTag::DynamicExtract,
                        ..
                    }
                )
            })
            .count()
    };
    assert_eq!(
        body_dyn_extracts("pick_const"),
        0,
        "const-index `m[1][0]` should fold; saw DynamicExtract ops remaining"
    );
    assert_eq!(
        body_dyn_extracts("pick_runtime"),
        2,
        "runtime-index `m[i][j]` should emit two chained DynamicExtract ops"
    );

    // End-to-end smoke: the SPIR-V backend accepts the program.
    let _ = crate::compile_thru_spirv(src).expect("compile_thru_spirv should succeed");
}

// Stage 2: runtime-outer / fixed-inner storage view. Pins the descriptor
// shape — the per-element byte count must reflect the full inner sub-array
// size (`[4]u32` → 16 B), not the innermost scalar (4 B), because the
// dispatch length is `byte_size / elem_bytes` and the buffer holds one
// `[4]u32` per dispatched thread.
#[test]
fn multidim_view_inner_fixed_carries_subarray_elem_bytes() {
    use crate::pipeline_descriptor::{BufferLen, DispatchLen, DispatchSize, Pipeline};
    let src = r#"
        #[compute]
        entry row_sums(buf: []([4]u32)) []u32 =
            map(|row: [4]u32| row[0] + row[1] + row[2] + row[3], buf)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("compile_thru_spirv");
    let Pipeline::Compute(cp) = lowered.pipeline.pipelines.first().expect("one pipeline") else {
        panic!("expected single-compute pipeline");
    };
    match &cp.dispatch_size {
        DispatchSize::DerivedFrom { len, .. } => match len {
            DispatchLen::InputBinding {
                set,
                binding,
                elem_bytes,
            } => {
                assert_eq!(*set, 0);
                assert_eq!(*binding, 0);
                assert_eq!(
                    *elem_bytes, 16,
                    "buf: []([4]u32) — each iterated element is [4]u32 (16 bytes), not 4"
                );
            }
            other => panic!("expected InputBinding dispatch length, got {other:?}"),
        },
        other => panic!("expected DerivedFrom dispatch size, got {other:?}"),
    }
    // The output `[]u32` is one u32 per dispatched thread.
    let output_len = cp.bindings.iter().find_map(|b| match b {
        crate::pipeline_descriptor::Binding::StorageBuffer { name, length, .. }
            if name == "row_sums_output" =>
        {
            length.clone()
        }
        _ => None,
    });
    match output_len {
        Some(BufferLen::SameAsDispatch { elem_bytes }) => assert_eq!(elem_bytes, 4),
        other => panic!("output should be SameAsDispatch{{elem_bytes: 4}}, got {other:?}"),
    }
}
