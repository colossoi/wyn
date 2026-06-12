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
    let tlc = type_checked.to_tlc(&module_manager, false).pin_entry_regions().expect("pin_entry_regions");
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
            TermKind::OutputSlotStore { value, .. } => walk(value, bound, symbols, stage, def_name),
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

/// Spec §x binop y: `f32 ** i32` (float base, integer exponent) must
/// type-check, lower to valid SPIR-V, and route through `OpConvertSToF`
/// before `GLSL Pow`. Use exponent `9` to skip the EGIR fold's
/// `2..8` constant-power-to-mul-chain rewrite, forcing the
/// backend-conversion path.
#[test]
fn pow_float_base_int_exp_lowers_via_convert_then_pow() {
    use rspirv::binary::parse_words;
    use rspirv::dr::Loader;
    use rspirv::spirv::Op;

    let spirv = compile_to_spirv(
        "\
#[compute]
entry e(xs: []f32) []f32 = map(|x: f32| x ** 9, xs)
",
    )
    .expect("f32 ** i32 (exp=9) compiles to SPIR-V");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut converts = 0;
    let mut pows = 0;
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst.class.opcode {
                    Op::ConvertSToF => converts += 1,
                    Op::ExtInst => {
                        // GLSL.std.450 Pow = opcode 26 (operand index 1).
                        if let Some(rspirv::dr::Operand::LiteralExtInstInteger(26)) = inst.operands.get(1) {
                            pows += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(
        converts >= 1,
        "expected at least one OpConvertSToF to coerce i32 exponent, found {converts}"
    );
    assert!(pows >= 1, "expected at least one GLSL Pow ext-inst, found {pows}");
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

/// Single-stage equivalent of `compile_to_spirv` — disables
/// `parallelize_soacs`, matching the CLI's `--single-stage` mode.
fn compile_to_spirv_single_stage(input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(crate::compile_thru_spirv_single_stage(input)?.spirv)
}

/// Helper: compile source through SSA. Same as `compile_to_ssa`; kept as
/// a separate name because some legacy tests distinguished module-bearing
/// programs from non-module ones — `compile_thru_frontend` handles both.
fn compile_to_ssa_with_modules(input: &str) -> Program {
    crate::compile_thru_ssa(input).expect("compile to SSA").ssa
}

// =========================================================================
// Backend gaps (aspirational, #[ignore]d)
//
// Each test asserts the *desired* code-gen outcome for a construct the SPIR-V
// backend currently can't handle; `#[ignore]`d so the suite stays green.
// Surfaced while building the lib/ statistics generators. Drop the `#[ignore]`
// when the gap is closed.
// =========================================================================

/// Gap: returning a runtime-sized `[]f32` from a (non-entry) function and then
/// indexing it panics the backend with "Composite variant unsized arrays not
/// supported" (`spirv/mod.rs`), instead of lowering the result as a
/// runtime-length array. A let-bound map + index, and a reduce over the same
/// array, both lower fine — it's specifically a function *return* of an
/// unsized Composite array that the type lowering rejects.
#[test]
#[ignore = "gap: returning a runtime-sized array from a function panics SPIR-V type lowering"]
fn returning_runtime_sized_array_from_fn_lowers() {
    let source = r#"
def g(n: i32) []f32 = map(|i: i32| f32.i32(i), 0i32 ..< n)
#[compute]
entry e() [1]f32 = [g(256)[3]]
"#;
    compile_to_spirv(source).expect("returning a runtime-sized array should lower to SPIR-V");
}

/// Gap: a runtime-sized array with *two or more* consumers panics the backend
/// with "Composite variant unsized arrays not supported" (`spirv/mod.rs`).
/// A single-consumer runtime-sized array fuses into its consumer and never
/// materializes (`f32.sum(map(…, 0..<n))` lowers fine), but binding it and
/// reading it more than once forces materialization of an unsized Composite
/// array, which the type lowering rejects. This is what blocks the lib/ `Stats`
/// gatherer, whose sample array feeds `sum`, a deviation `map`, `minimum`, and
/// `maximum`. Distinct from `returning_runtime_sized_array_from_fn_lowers`,
/// which is about *returning* such an array.
#[test]
#[ignore = "gap: a runtime-sized array with 2+ consumers must materialize as an unsized Composite array"]
fn runtime_sized_array_with_multiple_consumers_lowers() {
    let source = r#"
def g(n: i32) f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< n) in
    f32.sum(xs) + f32.maximum(xs)
#[compute]
entry e() f32 = g(256)
"#;
    compile_to_spirv(source)
        .expect("a runtime-sized array read by multiple consumers should lower to SPIR-V");
}

/// The reified operator members on the builtin numeric modules — `i32.(+)`,
/// `f32.(<)`, `u32.(<<)`, etc. — resolve as real module functions and lower.
/// The `(op)` form is the function reification of the primitive infix BinOp;
/// its body is the BinOp itself, so it lowers through the same path as `a + b`.
#[test]
fn reified_numeric_operators_lower_to_spirv() {
    let source = r#"
#[compute]
entry arith() i32 =
    i32.(+)(i32.(*)(i32.(-)(10i32, 3i32), 4i32), i32.(%)(9i32, 5i32))
#[compute]
entry bits() u32 = u32.(>>)(u32.(<<)(u32.(^)(u32.(&)(255u32, 15u32), 8u32), 2u32), 1u32)
#[compute]
entry cmp(x: f32, y: f32) i32 =
    let a = if f32.(==)(x, y) then 1i32 else 0i32 in
    let b = if f32.(!=)(x, y) then 2i32 else 0i32 in
    let c = if f32.(<)(x, y) then 4i32 else 0i32 in
    let d = if f32.(>)(x, y) then 8i32 else 0i32 in
    let e = if f32.(<=)(x, y) then 16i32 else 0i32 in
    let f = if f32.(>=)(x, y) then 32i32 else 0i32 in
    a + b + c + d + e + f
"#;
    compile_to_spirv(source).expect("reified numeric operator members should lower to SPIR-V");
}

/// The payoff of reifying operators into real module members: an operator can
/// be passed as a first-class value to a higher-order function (Wyn forbids
/// partial application, but operator members are saturated function references,
/// which it does support). Before reification `i32.(+)` had no referent.
#[test]
fn reified_operator_passed_as_first_class_value() {
    let source = r#"
#[compute]
entry sum(xs: [16]i32) i32 = reduce(i32.(+), 0i32, xs)
"#;
    compile_to_spirv(source).expect("a reified operator member should be passable to a HOF");
}

/// The `numeric` whole-array reductions `sum`/`product`/`minimum`/`maximum` are
/// implemented (for the float modules) as `reduce` over the per-type operator
/// and its neutral, so they lower to real SPIR-V reduction loops.
#[test]
fn numeric_array_reductions_lower_to_spirv() {
    let source = r#"
def N: i32 = 256
#[compute]
entry e() [4]f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< N) in
    [f32.sum(xs), f32.product(xs), f32.minimum(xs), f32.maximum(xs)]
"#;
    compile_to_spirv(source).expect("numeric array-reductions should lower to SPIR-V");
}

/// The statistics-gatherer shape that motivated the reductions: reduce a sample
/// stream to `[count, mean, variance, stddev, min, max]` using `f32.sum`,
/// `f32.minimum`, `f32.maximum`. This is the lib/ `Stats` summarize body.
#[test]
fn statistics_gatherer_lowers() {
    let source = r#"
def N: i32 = 256
#[compute]
entry summarize() [6]f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< N) in
    let n = f32.i32(N) in
    let mean = f32.sum(xs) / n in
    let sq = map(|v: f32| (v - mean) * (v - mean), xs) in
    let variance = f32.sum(sq) / n in
    [n, mean, variance, f32.sqrt(variance), f32.minimum(xs), f32.maximum(xs)]
"#;
    compile_to_spirv(source).expect("the statistics gatherer should lower to SPIR-V");
}

/// Pre-existing SPIR-V codegen bug (not operator-related — pure infix
/// reproduces it). A three-deep dependent `let` chain whose final binding is a
/// *bitwise* op, consumed by a following `if` condition, mislowers with
/// "place ... has no pointer — its defining instruction was not lowered (or ran
/// after a consumer)". Narrowing: a two-`let` chain lowers fine, and a
/// three-`let` chain ending in an *arithmetic* op lowers fine — it takes both
/// the chain depth and the bitwise final op; the `if` branch contents are
/// irrelevant. Smells like an EGIR/SSA scheduling issue where the bitwise
/// place is ordered after its consumer in the conditional.
#[test]
#[ignore = "pre-existing: bitwise op at the end of a 3-deep let-chain feeding an if mislowers"]
fn bitwise_in_deep_let_chain_feeding_if_lowers() {
    let source = r#"
#[compute]
entry t() i32 =
    let s = 2i32 + 3i32 in
    let p = s * 4i32 in
    let x = p & 12i32 in
    if x < 100i32 then x else 0i32
"#;
    compile_to_spirv(source)
        .expect("bitwise result threaded through a deep let-chain into an if should lower");
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
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
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

/// Walk every `OpAccessChain` into a `StorageBuffer`-class `OpVariable`
/// and assert the access-chain's result pointer points at the same
/// element type the variable's runtime array carries. spirv-val rejects
/// a mismatch with `OpAccessChain result type ... does not match the
/// type that results from indexing into the base`; the in-process check
/// catches the same shape so tests don't need `spirv-val` on $PATH.
fn assert_spirv_storage_access_chain_pointee_types_match(spirv_words: &[u32]) {
    use rspirv::binary::parse_words;
    use rspirv::dr::{Loader, Operand};
    use rspirv::spirv::{Op, StorageClass};
    use std::collections::HashMap;

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    // First pass: index every type-defining instruction by its result id.
    let mut types: HashMap<u32, &rspirv::dr::Instruction> = HashMap::new();
    for inst in module.types_global_values.iter() {
        if let Some(id) = inst.result_id {
            types.insert(id, inst);
        }
    }

    // Helper: drill `OpTypePointer Class %inner` → `inner`.
    let ptr_pointee = |type_id: u32| -> Option<u32> {
        let inst = types.get(&type_id)?;
        if inst.class.opcode != Op::TypePointer {
            return None;
        }
        // Operands: [0] StorageClass, [1] inner type IdRef.
        match inst.operands.get(1) {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        }
    };

    // Helper: drill `OpTypeStruct %member0 %member1 ...` → first member.
    // For wyn's storage buffer blocks this is the `OpTypeRuntimeArray`.
    let struct_first_member = |type_id: u32| -> Option<u32> {
        let inst = types.get(&type_id)?;
        if inst.class.opcode != Op::TypeStruct {
            return None;
        }
        match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        }
    };

    // Helper: drill `OpTypeRuntimeArray %elem` → `elem`. (Also accepts
    // OpTypeArray.)
    let array_elem = |type_id: u32| -> Option<u32> {
        let inst = types.get(&type_id)?;
        if !matches!(inst.class.opcode, Op::TypeRuntimeArray | Op::TypeArray) {
            return None;
        }
        match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        }
    };

    // Collect each StorageBuffer-class OpVariable's element type.
    let mut storage_var_elem: HashMap<u32, u32> = HashMap::new();
    for inst in module.types_global_values.iter() {
        if inst.class.opcode != Op::Variable {
            continue;
        }
        let class = match inst.operands.first() {
            Some(Operand::StorageClass(c)) => *c,
            _ => continue,
        };
        if class != StorageClass::StorageBuffer {
            continue;
        }
        let var_id = match inst.result_id {
            Some(id) => id,
            None => continue,
        };
        // Variable's result_type is `OpTypePointer StorageBuffer %struct`.
        let var_ptr_ty = match inst.result_type {
            Some(id) => id,
            None => continue,
        };
        let struct_ty = match ptr_pointee(var_ptr_ty) {
            Some(id) => id,
            None => continue,
        };
        let runtime_arr = match struct_first_member(struct_ty) {
            Some(id) => id,
            None => continue,
        };
        let elem = match array_elem(runtime_arr) {
            Some(id) => id,
            None => continue,
        };
        storage_var_elem.insert(var_id, elem);
    }

    // Walk every function body for OpAccessChain into such a variable.
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::AccessChain {
                    continue;
                }
                // Operands: [0] base IdRef, [1..] index IdRefs.
                let base = match inst.operands.first() {
                    Some(Operand::IdRef(id)) => *id,
                    _ => continue,
                };
                let Some(expected_elem) = storage_var_elem.get(&base).copied() else {
                    continue;
                };
                let result_ptr_ty = inst.result_type.expect("OpAccessChain has result type");
                let actual_pointee =
                    ptr_pointee(result_ptr_ty).expect("OpAccessChain result type must be OpTypePointer");
                assert_eq!(
                    actual_pointee, expected_elem,
                    "OpAccessChain into StorageBuffer var %{base}: result pointer pointee \
                     %{actual_pointee} does not match the variable's array element type \
                     %{expected_elem} (chain result_id %{:?})",
                    inst.result_id
                );
            }
        }
    }
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
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .fold_generated_lambdas()
        .inline_small()
        .materialize_entry_soacs()
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
    // The pre-pass writes the gather binding as its EntryOutput — that
    // shows up as `WriteOnly` in the producer pipeline. `publish` then
    // promotes any `ReadOnly` consumer reference of a module-written
    // binding to `ReadWrite` (see the consumer assertion below), so
    // wgpu/Naga sees consistent module-level access. `WriteOnly` is
    // already `read_only=false` at the wgpu layer, so the producer side
    // doesn't need widening.
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

    // The consumer reads that same binding as an Intermediate. Access is
    // `ReadWrite` because of the module-level promotion described above —
    // not because the consumer itself writes (it doesn't).
    let reads_gather = consumer_bufs.iter().any(|b| {
        matches!(b, Binding::StorageBuffer { binding, usage: BufferUsage::Intermediate, access: Access::ReadWrite, .. } if binding == gather_binding)
    });
    assert!(
        reads_gather,
        "consumer must read the gather buffer (binding {gather_binding}) as ReadWrite intermediate: {consumer_bufs:?}"
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

/// A scalar bound in an outer SOAC lambda and captured by a *nested* SOAC
/// lambda must be threaded as a capture, not left as a free global. This is
/// the N-body inner-force-sum shape — `map(|i| … reduce(+, …, map(|j| f(xs[j],
/// xs[i]), …)))` — where the inner `map` over `j` closes over `pi = xs[i]`.
#[test]
fn nested_soac_captures_outer_scalar() {
    compile_to_spirv(
        "\
#[compute]
entry t(xs: []f32) []f32 =
  map(|i: i32|
        let pi = xs[i] in
        reduce(|a: f32, b: f32| a + b, 0.0,
               map(|j: i32| xs[j] - pi, 0i32 ..< 4)),
      0i32 ..< 4)
",
    )
    .expect("nested SOAC capturing an outer scalar must compile");
}

// ---- `filter` SOAC composition (gaps surfaced exploring the N-body port) ----
//
// `filter(pred, xs)` needs a statically-sized `xs` and returns the existential
// `?k. [k]T`, which a `let` opens before use. These pin which compositions of
// that result the compiler accepts. (None of these can be *executed* in unit
// tests — there's no GPU adapter here — so they assert compilation only.)

/// The supported shape: fixed-size input, existential result opened by `let`,
/// consumed by `length`. Compiles end to end.
#[test]
fn filter_in_subroutine_length_compiles() {
    compile_to_spirv(
        "\
def evens(arr: [8]i32) ?k. [k]i32 = filter(|x: i32| x % 2i32 == 0i32, arr)

#[compute]
entry filt_count() i32 =
  let e = evens([1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32, 8i32]) in
  length(e)
",
    )
    .expect("filter in a subroutine, opened by `let`, consumed by `length` must compile");
}

/// Runtime-sized `filter` consumed by `length`: the input is an entry-param
/// view (`[]i32`), so `filter` compacts kept elements into a reserved scratch
/// storage buffer and yields a runtime-length view; `length` reads the view's
/// `len` operand (the surviving count).
#[test]
fn filter_runtime_length_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry filt_count(xs: []i32) i32 =
  let e = filter(|x: i32| x % 2i32 == 0i32, xs) in
  length(e)
",
    )
    .expect("runtime-sized filter consumed by length must compile");
}

/// A runtime-sized `filter` inside a **subroutine** that the entry calls. This
/// is the safety net for the scratch-binding home: `filter` compacts into a
/// reserved storage buffer, and only a compute *entry* owns a descriptor set +
/// binding namespace to host it (an `EgirFunc` does not — see the guard in
/// `from_tlc::convert_function`). This compiles because `evens` is **inlined**
/// into `filt_count` before EGIR conversion, so `convert_soac_filter` runs in
/// the entry's converter and the scratch buffer lands at a non-colliding entry
/// binding.
///
/// IF THIS TEST STARTS FAILING with "runtime `filter` in function `evens`
/// reserved a scratch storage buffer …": the inlining invariant broke — a
/// function whose result is a runtime filter survived to EGIR as a standalone
/// `EgirFunc`. The scratch buffer then has no descriptor-set home. To fix,
/// either (a) restore inlining of filter-returning functions before `from_tlc`,
/// or (b) thread a caller-reserved scratch binding into the function's
/// signature (like an extra param / interface entry) so the buffer is declared
/// and sized on a real descriptor set. Do NOT relax the `convert_function`
/// guard to emit anyway — that mis-numbers the binding and silently drops its
/// host declaration (wrong-buffer codegen).
#[test]
fn filter_runtime_in_subroutine_compiles() {
    compile_to_spirv(
        "\
def evens(arr: []i32) ?k. [k]i32 = filter(|x: i32| x % 2i32 == 0i32, arr)

#[compute]
entry filt_count(xs: []i32) i32 =
  let e = evens(xs) in
  length(e)
",
    )
    .expect("runtime filter in an (inlined) subroutine must compile");
}

/// Summing a filtered runtime-sized array — `reduce(+, 0, filter(p, xs))` over
/// an entry-param view. `filter` yields a runtime-length scratch view; `reduce`
/// consumes it like any reduce-over-view. (Was a gap: the static-literal form
/// left an unexpanded `PendingSoac` panicking at `elaborate.rs`.)
#[test]
fn filter_into_reduce_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry filt_reduce(xs: []i32) i32 =
  let kept = filter(|x: i32| x > 4i32, xs) in
  reduce(|a: i32, b: i32| a + b, 0i32, kept)
",
    )
    .expect("summing a filtered runtime array (filter → reduce) must compile");
}

/// True iff the pipeline for `entry` is a multi-stage compute (the two-phase
/// shape a parallelized reduce/redomap lowers to: chunk + combine). Used to
/// confirm the masked-redomap fusion fired — a *serial* filter→reduce would be
/// a single-stage `Compute` instead.
fn is_two_phase_compute(pipeline: &crate::pipeline_descriptor::PipelineDescriptor, entry: &str) -> bool {
    use crate::pipeline_descriptor::Pipeline;
    pipeline.pipelines.iter().any(|p| match p {
        Pipeline::MultiCompute(mc) => {
            mc.stages.len() >= 2 && mc.stages.iter().any(|s| s.entry_point == entry)
        }
        _ => false,
    })
}

/// `reduce(op, ne, filter(p, xs))` fuses into a masked redomap — no compacted
/// intermediate array — and parallelizes as a two-phase reduce. Pins that the
/// fusion fired (not the serial scratch-view filter path).
#[test]
fn filter_into_reduce_fuses_to_parallel_redomap() {
    let lowered = crate::compile_thru_spirv(
        "\
#[compute]
entry filt_reduce(xs: []i32) i32 =
  let kept = filter(|x: i32| x > 4i32, xs) in
  reduce(|a: i32, b: i32| a + b, 0i32, kept)
",
    )
    .expect("filter→reduce compiles");
    assert!(
        is_two_phase_compute(&lowered.pipeline, "filt_reduce"),
        "reduce(filter(..)) must fuse to a masked redomap (two-phase compute), not a serial filter",
    );
}

/// The masked-redomap fusion must fire even when `filter` and `reduce` live in
/// **different functions** — TLC fusion sees through the `evens` call via
/// function summaries (no inlining needed at fusion time).
#[test]
fn filter_into_reduce_fuses_across_functions() {
    let lowered = crate::compile_thru_spirv(
        "\
def evens(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 2i32 == 0i32, xs)

#[compute]
entry filt_reduce(xs: []i32) i32 =
  let kept = evens(xs) in
  reduce(|a: i32, b: i32| a + b, 0i32, kept)
",
    )
    .expect("cross-function filter→reduce compiles");
    assert!(
        is_two_phase_compute(&lowered.pipeline, "filt_reduce"),
        "cross-function reduce(evens(xs)) must fuse to a masked redomap via function summaries",
    );
}

/// Cross-function auto-parallelization: a `scan` factored into a helper that
/// `inline_small` will NOT fold (its operator has control flow, so the
/// size/control-flow gate skips it) still parallelizes — `materialize_entry_soacs`
/// exposes it at the entry boundary so `parallelize` produces the same
/// multi-phase pipeline as the in-entry form. (`inline_small` skipping the
/// helper is what makes this exercise the new pass specifically.)
#[test]
fn cross_function_scan_parallelizes() {
    let lowered = crate::compile_thru_spirv(
        "\
def stencil(xs: []i32) []i32 = scan(|a: i32, b: i32| if a > b then a else b, 0i32, xs)
#[compute]
entry e(xs: []i32) []i32 = stencil(xs)
",
    )
    .expect("cross-function scan compiles");
    assert!(
        is_two_phase_compute(&lowered.pipeline, "e"),
        "a scan factored into a (non-inlinable) helper must still parallelize cross-function",
    );
}

/// GAP: a computed array consumed by random index (a gather) whose producer is in
/// a **helper** isn't supported — `lift_gathers` runs before the producer is
/// materialized, so the gather array reaches the backend as a runtime-sized
/// Composite and **panics** (`spirv::polytype_to_spirv`, the unsized-Composite
/// arm). It *should* be a clean compile error instead. The body below asserts
/// that desired behavior; un-ignore when the panic→error fix lands. (Same shape
/// also panics on master via `inline_small`, so this is a pre-existing
/// limitation, not a regression from the materialize pass.)
#[test]
#[ignore = "cross-function gather: runtime-sized composite indexed → backend panic; \
            want a clean error (panic→error fix not yet done)"]
fn cross_function_gather_errors_cleanly() {
    let r = crate::compile_thru_spirv(
        "\
def counts(xs: []i32) []i32 = map(|x: i32| x * 2i32, xs)
#[compute]
entry g(xs: []i32) []i32 =
  let c = counts(xs) in
  map(|i: i32| c[i % 8i32], iota(64))
",
    );
    let err = r.err().expect("cross-function gather must be a compile error, not success/panic");
    let msg = err.to_string();
    assert!(
        msg.contains("runtime-sized") && msg.contains("index"),
        "error should explain the un-lifted runtime-sized gather, got: {msg}",
    );
}

/// Invariant, end to end: a SOAC helper called *per element* inside a `map`
/// lambda must NOT be hoisted and parallelized — the inner reduce stays a
/// serial per-thread loop. The entry parallelizes as a single-stage
/// lane-indexed map, not a multi-phase reduce pipeline.
#[test]
fn per_element_helper_soac_stays_serial() {
    let lowered = crate::compile_thru_spirv(
        "\
def rsum(x: i32) i32 = reduce(|a: i32, b: i32| a + b, 0i32, [x, x, x])
#[compute]
entry e(xs: []i32) []i32 = map(|x: i32| rsum(x), xs)
",
    )
    .expect("per-element helper compiles");
    assert!(
        !is_two_phase_compute(&lowered.pipeline, "e"),
        "a per-element helper reduce must stay serial, not become a parallel reduce pipeline",
    );
}

/// Returning a filtered runtime-sized array from a compute entry. The filter
/// compacts directly into the user-visible output buffer (sized to the input's
/// element count), and its surviving count is written to a paired `len` cell
/// the host reads back. `realize_outputs::retarget_filter_output` wires this.
#[test]
fn filter_result_as_compute_output_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry filt_out(xs: []i32) ?k. [k]i32 =
  filter(|x: i32| x % 2i32 == 0i32, xs)
",
    )
    .expect("returning a filtered array from a compute entry must compile");
}

/// Pins the filter→output host ABI (the paired length buffer). The `filt_out`
/// pipeline must expose: the input, a host-readable **Output** data buffer sized
/// `LikeInput` of the input (capacity n), and a compiler-managed **Intermediate**
/// length cell sized `Fixed { bytes: 4 }` (one u32) holding the surviving count.
#[test]
fn filter_output_descriptor_has_paired_length_buffer() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage};
    let src = "\
#[compute]
entry filt_out(xs: []i32) ?k. [k]i32 =
  filter(|x: i32| x % 2i32 == 0i32, xs)
";
    let lowered = crate::compile_thru_spirv(src).expect("filter→output compiles");
    let bufs = compute_storage_buffers(&lowered.pipeline, "filt_out");

    let output = bufs
        .iter()
        .find(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Output,
                    ..
                }
            )
        })
        .expect("filter→output has a host-readable Output buffer");
    let Binding::StorageBuffer { length: out_len, .. } = output else {
        unreachable!()
    };
    assert!(
        matches!(out_len, Some(BufferLen::LikeInput { .. })),
        "output data buffer is sized to the input element count (capacity n): {output:?}",
    );

    let intermediates: Vec<&Binding> = bufs
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
        intermediates.len(),
        1,
        "exactly one Intermediate (the paired length cell): {bufs:?}",
    );
    let Binding::StorageBuffer { length: len_len, .. } = intermediates[0] else {
        unreachable!()
    };
    assert_eq!(
        *len_len,
        Some(BufferLen::Fixed { bytes: 4 }),
        "the length cell is a single u32 (4 bytes): {:?}",
        intermediates[0],
    );
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

/// Bisected min-repro: a multi-output entry returns a scan result AS one
/// output and also reads that scan by a (constant) index for a second
/// output. Previously panicked at `spirv/mod.rs:375` ("Composite variant
/// unsized arrays not supported") because `realize_outputs` retargeted the
/// scan to `OutputView` for slot 0 but slot 1's `[offsets[0]]` still
/// demanded the (now-vanished) in-register Composite. Fixed in
/// `realize_outputs::rewrite_other_index_consumers_to_loads`: detect the
/// sibling Index consumer, synthesise a `ViewIndex + Load` against slot
/// 0's output view (both backends declare output bindings as read-write),
/// alias the Index NodeId to the load result. Slot 0's binding doubles as
/// the shared buffer.
#[test]
fn multi_output_returns_scan_and_reads_it_by_index() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, [1]i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, [offsets[0]])
",
    )
    .expect("multi-output (scan + indexed read of same scan) should compile");
}

/// Dynamic-index variant of the above — slot 1 reads `offsets[k]` where
/// `k` is a uniform, exercising the path where the rewrite passes the
/// dynamic index NodeId straight through `emit_view_load`.
#[test]
fn multi_output_returns_scan_and_reads_it_by_dynamic_index() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32, #[uniform(set=1,binding=0)] k: i32) ([]i32, [1]i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, [offsets[k]])
",
    )
    .expect("multi-output with dynamic-index sibling read of same scan should compile");
}

/// Map producer variant — slot 0 retargets a Map (not a Scan); slot 1
/// reads it by index. Same mechanism, different SOAC kind.
#[test]
fn multi_output_returns_map_and_reads_it_by_index() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, [1]i32) =
  let doubled = map(|x: i32| x * 2, xs) in
  (doubled, [doubled[0]])
",
    )
    .expect("multi-output (map + indexed read of same map) should compile");
}

/// Returning the same scan in two output slots — `(offsets, offsets)` —
/// can't be served by retargeting the scan twice. Slot 0 retargets to
/// its view; slot 1, also a runtime-sized array, can't retarget the
/// (already-retargeted) SOAC, so it falls through to the existing
/// "runtime-sized array not produced by a retargetable map/scan"
/// `ConvertError::Unsupported`. Pinned here to confirm we surface a
/// clean diagnostic rather than panicking.
#[test]
fn multi_output_returns_scan_in_two_slots_is_rejected() {
    let result = crate::compile_thru_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, []i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, offsets)
",
    );
    let err = match result {
        Ok(_) => panic!("expected returning the same scan in two output slots to fail"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("runtime-sized array"),
        "expected runtime-sized-array diagnostic, got: {msg}"
    );
}

/// Under `--single-stage` mode, a vec4-emitting map that gathers from a
/// derived (map/scan-produced) array must still produce well-formed
/// SPIR-V. The bug was: `lift_gathers` flagged the producer's gather
/// buffer via an Output-role `StorageBindingDecl` on the prepass entry,
/// but `from_tlc::convert` only consulted the parallelize `plans` for
/// `forced_output_binding`. With `parallelize_soacs(disable=true)`
/// `plans` is empty, so `build_entry_outputs` auto-allocated the
/// prepass's output at the next free binding — colliding with the
/// consumer's vec4 output and emitting an `int` store into a `vec4`
/// buffer. spirv-val flagged the OpAccessChain type mismatch.
#[test]
fn single_stage_vec4_map_gather_from_derived_array_repro() {
    let spirv = compile_to_spirv_single_stage(
        "\
#[compute]
entry gen(xs: []i32) []vec4f32 =
  let cs = map(|x: i32| x * 2, xs) in
  map(|i: i32| @[f32.i32(cs[i]), 0.0, 0.0, 1.0], iota(8))
",
    )
    .expect("single-stage vec4-map gathering derived array compiles");
    assert_spirv_storage_access_chain_pointee_types_match(&spirv);
}

/// The descriptor for a compiler-allocated `lift_gathers` intermediate
/// must agree with the SPIR-V module's per-binding writability. SPIR-V's
/// `NonWritable` decoration is module-level, so when a sibling entry in
/// the same module writes the gather buffer, the consumer pipeline's
/// `OpVariable` has no `NonWritable` and Naga reports the storage as
/// `read_write`. Previously the descriptor reported `read_only` based on
/// the consumer's `StorageBindingDecl.role` alone, causing wgpu to fail
/// pipeline creation with `Storage class Storage{LOAD} doesn't match
/// shader Storage{LOAD | STORE}`. The fix promotes any intermediate
/// binding whose `(set, binding)` is also an entry-output target to
/// `Access::ReadWrite`.
#[test]
fn intermediate_buffer_descriptor_access_repro() {
    use crate::pipeline_descriptor::{Access, Binding, BufferUsage, Pipeline};
    use std::collections::HashSet;

    let lowered = crate::compile_thru_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, [1]i32) =
  let counts  = map(|x: i32| x * 2, xs) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  (map(|i: i32| offsets[i % 8], iota(64)),
   [offsets[7]])
",
    )
    .expect("compile to SPIR-V");

    // Collect every (set, binding) any pipeline writes to anywhere in
    // the module. A read-only declaration of one of these in a sibling
    // pipeline is a descriptor↔shader mismatch.
    let written: HashSet<(u32, u32)> = lowered
        .pipeline
        .pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(cp) => cp.bindings.iter().collect::<Vec<_>>(),
            Pipeline::MultiCompute(mc) => mc.bindings.iter().collect::<Vec<_>>(),
            Pipeline::Graphics(gp) => gp.bindings.iter().collect::<Vec<_>>(),
        })
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                set, binding, access, ..
            } if matches!(access, Access::WriteOnly | Access::ReadWrite) => Some((*set, *binding)),
            _ => None,
        })
        .collect();

    let mut violations: Vec<String> = Vec::new();
    for p in &lowered.pipeline.pipelines {
        let (label, bindings): (String, &[Binding]) = match p {
            Pipeline::Compute(cp) => (cp.entry_point.clone(), &cp.bindings),
            Pipeline::MultiCompute(mc) => {
                let names: Vec<&str> = mc.stages.iter().map(|s| s.entry_point.as_str()).collect();
                (format!("multi[{}]", names.join(",")), &mc.bindings)
            }
            Pipeline::Graphics(gp) => {
                let names: Vec<&str> = gp.stages.iter().map(|s| s.entry_point.as_str()).collect();
                (format!("graphics[{}]", names.join(",")), &gp.bindings)
            }
        };
        for b in bindings {
            if let Binding::StorageBuffer {
                set,
                binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Intermediate,
                name,
                ..
            } = b
            {
                if written.contains(&(*set, *binding)) {
                    violations.push(format!(
                        "pipeline {label}: intermediate binding {set}.{binding} ({name}) \
                         declared read_only but written by another entry in the module"
                    ));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "descriptor↔shader access mismatch on lift_gathers intermediates:\n  {}",
        violations.join("\n  ")
    );
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

    // The consumer reads the gather buffer as an Intermediate sized
    // LikeInput of `xs` (scan preserves element count and type: i32 → i32).
    // Access is `ReadWrite`: `publish` promotes any binding written by a
    // sibling entry in the module so the descriptor matches the SPIR-V's
    // module-level access.
    let consumer_bufs = compute_storage_buffers(&lowered.pipeline, "g");
    let gather = consumer_bufs
        .iter()
        .find_map(|b| match b {
            Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Intermediate,
                access: Access::ReadWrite,
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
    // The output `[]u32`'s size variable matches `buf`'s (the
    // type checker unified them — `map(f, buf): [n]u32` for the same
    // `n` as `buf: [n][4]u32`), so the length-policy inference
    // emits `LikeInput` rather than the looser `SameAsDispatch`.
    // Both resolve to the same allocated byte size when the dispatch
    // is itself derived from `buf` (as it is here), but `LikeInput`
    // names the source binding explicitly.
    let output_len = cp.bindings.iter().find_map(|b| match b {
        crate::pipeline_descriptor::Binding::StorageBuffer { name, length, .. }
            if name == "row_sums_output" =>
        {
            length.clone()
        }
        _ => None,
    });
    match output_len {
        Some(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            src_elem_bytes,
        }) => {
            assert_eq!(set, 0);
            assert_eq!(binding, 0);
            assert_eq!(elem_bytes, 4);
            assert_eq!(src_elem_bytes, 16);
        }
        other => panic!(
            "output should be LikeInput {{set:0, binding:0, elem_bytes:4, src_elem_bytes:16}}, got {other:?}"
        ),
    }
}

/// `If`-over-two-retargetable-maps with a runtime-sized output:
/// previously rejected by `realize_outputs::lower_slot` because the
/// merge-block param wasn't a Map/Scan node. After the DPS migration,
/// each branch's `OutputSlotStore` records its own `SlotSource` at
/// its block; `realize_outputs` retargets both Maps into the same
/// output view. Runtime CFG ensures only one fires per execution
/// path.
#[test]
fn compute_if_over_two_maps_compiles_runtime_sized() {
    use crate::pipeline_descriptor::{BufferLen, Pipeline};
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime == 0.0
            then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
            else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("compile_thru_spirv");
    let Pipeline::Compute(cp) = lowered.pipeline.pipelines.first().expect("one pipeline") else {
        panic!("expected single-compute pipeline");
    };
    // Output's size variable matches `prev`'s — the length-inference
    // rule emits `LikeInput` rather than `SameAsDispatch`.
    let output_len = cp.bindings.iter().find_map(|b| match b {
        crate::pipeline_descriptor::Binding::StorageBuffer { name, length, .. }
            if name == "tick_output" =>
        {
            length.clone()
        }
        _ => None,
    });
    match output_len {
        Some(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            ..
        }) => {
            assert_eq!(set, 2);
            assert_eq!(binding, 0);
            assert_eq!(elem_bytes, 8); // vec2f32
        }
        other => panic!("output should be LikeInput, got {other:?}"),
    }
}

/// Nested `If` over retargetable maps. The `convert_slot_store`
/// recursion handles arbitrary nesting; each leaf records its own
/// `SlotSource` against the branch's block. After realization the
/// slot has three sources (one per leaf), all retargeting into the
/// same `OutputView`.
#[test]
fn compute_nested_if_over_three_maps_compiles_runtime_sized() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime < 0.0
            then map(|p: vec2f32| @[0.0f32, 0.0f32], prev)
            else if iTime == 0.0
              then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
              else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;
    crate::compile_thru_spirv(src).expect("nested If over three maps must compile");
}

/// `Let`-wrapped `If` whose body's branches read the let-bound value.
/// `convert_slot_store` recognises the `Let` and binds the RHS at the
/// current block before recursing into the body — so the binding
/// survives the branch fork.
#[test]
fn compute_let_wrapped_if_over_two_maps_compiles_runtime_sized() {
    let src = r#"
        #[compute]
        entry tick<[n]>(#[storage(set=2, binding=0, access=read)] prev: [n]vec2f32,
                        #[uniform(set=1, binding=1)] iTime: f32) [n]vec2f32 =
          let nudge: f32 = iTime * 0.1f32 in
          if iTime == 0.0
            then map(|p: vec2f32| @[nudge, nudge], prev)
            else map(|p: vec2f32| @[p.x + nudge, p.y + nudge], prev)
    "#;
    crate::compile_thru_spirv(src).expect("Let-wrapped If over two maps must compile");
}

/// The user's original case 1 (fixed-size output): both branches map
/// over different sources (`0..<N` vs `prev_pos`) but the output is
/// `[Size(2)]vec2f32` because `N = 2` is a literal. Lands in the
/// fixed-aggregate path of `dispatch::compute_slot_source`. This
/// always worked; the test pins it as regression protection alongside
/// the runtime-sized variants the DPS migration unblocked.
#[test]
fn compute_if_over_two_maps_compiles_fixed_size_different_sources() {
    let src = r#"
        def N: i32 = 2
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev_pos: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime == 0.0 then
            map(|i:i32| if i == 0 then @[2.0, 2.0] else @[15.0, 5.0], 0i32..<N)
          else
            map(|pos:vec2f32| @[pos.x + 1.0, pos.y + 1.0], prev_pos)
    "#;
    crate::compile_thru_spirv(src).expect("fixed-size If-over-maps (different sources) must compile");
}

/// The user's original case 2 (fixed-size output): both branches map
/// over the *same* range source. Output is still `[Size(2)]vec2f32`.
/// Same code path as case 1 — the source-difference doesn't matter for
/// fixed-size aggregates.
#[test]
fn compute_if_over_two_maps_compiles_fixed_size_same_source() {
    let src = r#"
        def N: i32 = 2
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev_pos: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime == 0.0 then
            map(|i:i32| if i == 0 then @[2.0, 2.0] else @[15.0, 5.0], 0i32..<N)
          else
            map(|i:i32| @[f32.i32(i), f32.i32(i)], 0i32..<N)
    "#;
    crate::compile_thru_spirv(src).expect("fixed-size If-over-maps (same source) must compile");
}

/// Multi-output entry whose Tuple components each contain an `If`.
/// `normalize_outputs` decomposes the Tuple into per-slot
/// `OutputSlotStore`s; each then enters the `If`-fork recursion in
/// `convert_slot_store`. Both slots end up multi-source, each
/// retargeting into its own `OutputView`.
#[test]
fn compute_multi_output_tuple_of_ifs_compiles() {
    let src = r#"
        #[compute]
        entry tick<[n]>(#[storage(set=2, binding=0, access=read)] prev_pos: [n]vec2f32,
                        #[uniform(set=1, binding=1)] iTime: f32) ([n]vec2f32, [n]f32) =
          (if iTime == 0.0
             then map(|p: vec2f32| @[0.0f32, 0.0f32], prev_pos)
             else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev_pos),
           if iTime == 0.0
             then map(|p: vec2f32| 0.0f32, prev_pos)
             else map(|p: vec2f32| p.x * p.x + p.y * p.y, prev_pos))
    "#;
    crate::compile_thru_spirv(src).expect("multi-output tuple of Ifs must compile");
}

/// Assert the StorageBuffer variable decorated `(set, binding)` is the base
/// of at least one `OpAccessChain` — i.e. actually read/written, not merely
/// declared. Regression guard for view-array provenance loss, where a lifted
/// lambda's reads went to the (wrong) output descriptor and the input buffer
/// was declared but never accessed.
fn assert_storage_descriptor_is_accessed(spirv_words: &[u32], set: u32, binding: u32) {
    use rspirv::binary::parse_words;
    use rspirv::dr::{Loader, Operand};
    use rspirv::spirv::{Decoration, Op};
    use std::collections::HashMap;

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut sets: HashMap<u32, u32> = HashMap::new();
    let mut binds: HashMap<u32, u32> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        let target = match inst.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => continue,
        };
        match (inst.operands.get(1), inst.operands.get(2)) {
            (Some(Operand::Decoration(Decoration::DescriptorSet)), Some(Operand::LiteralBit32(n))) => {
                sets.insert(target, *n);
            }
            (Some(Operand::Decoration(Decoration::Binding)), Some(Operand::LiteralBit32(n))) => {
                binds.insert(target, *n);
            }
            _ => {}
        }
    }

    let target_var = sets
        .iter()
        .find(|(id, s)| **s == set && binds.get(id) == Some(&binding))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no StorageBuffer variable decorated (set={set}, binding={binding})"));

    let accessed = module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == Op::AccessChain
                    && matches!(inst.operands.first(), Some(Operand::IdRef(base)) if *base == target_var)
            })
        })
    });

    assert!(
        accessed,
        "descriptor (set={set}, binding={binding}) is declared but never reached by an \
         OpAccessChain — view-array provenance was lost (reads went to the wrong buffer)"
    );
}

// View-array slice provenance through a SOAC capture: `xs[0..3]` is fine at
// top level but used to fail inside a `map` lambda body with
// "slice_to_composite: no buffer provenance". After buffer-specialize learned
// the slice→composite case, the lifted lambda's reads must come from `xs`'s
// own descriptor (set=2, binding=0), not the compiler-allocated output buffer.
#[test]
fn slice_view_inside_map_lambda_compiles_to_spirv() {
    let src = r#"
        def gather3(arr: [3]f32) f32 = arr[0] + arr[1] + arr[2]

        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|_:i32| gather3(xs[0..3]), 0i32..<3)
    "#;
    let lowered = crate::compile_thru_spirv(src)
        .expect("view-array slice inside a map lambda must preserve buffer provenance");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

// ---- Buffer-provenance guards ------------------------------------------------
//
// These pin the *correct* descriptor each view read resolves to, so a later
// refactor of buffer_specialize (e.g. unifying `rewrite_term` and
// `rewrite_specialized_body`) can't silently mis-route a read to the wrong
// buffer. `cargo test` green alone does NOT prove this: a wrong-buffer read
// still passes spirv-val as long as the (wrong) descriptor is declared — which
// is exactly the historical bug. Each guard asserts via rspirv that the
// expected `(set, binding)` is actually the base of an `OpAccessChain`.

/// Indexing a *captured* view inside a `map` lambda (→ lifted lambda, the
/// `rewrite_specialized_body` Index arm) must read from the captured buffer.
#[test]
fn view_index_in_map_lambda_reads_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|i: i32| xs[i] + xs[0], 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("captured-view index compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// Two captured views at distinct `(set, binding)` must each be read from
/// their own descriptor — catches a unification that swaps or collapses
/// buffer provenance.
#[test]
fn two_view_captures_read_distinct_buffers() {
    let src = r#"
        #[compute]
        entry tick(
          #[storage(set=2, binding=0, access=read)] xs: []f32,
          #[storage(set=2, binding=1, access=read)] ys: []f32
        ) []f32 =
          map(|i: i32| xs[i] + ys[0], 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("two captured views compile");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 1);
}

/// A captured view passed to a user function that itself indexes it (→
/// recursive per-buffer specialization) must read from the captured buffer.
#[test]
fn view_through_nested_fn_specialization_reads_own_buffer() {
    let src = r#"
        def firstx(zs: []f32) f32 = zs[0]

        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|_: i32| firstx(xs), 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("nested view specialization compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// A view used directly as a `map` *input* (→ the entry walker
/// `rewrite_term` / SOAC-input path, not a capture) must read from its buffer.
#[test]
fn view_as_map_input_reads_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|x: f32| x * 2.0, xs)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("view-as-map-input compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// Assert that some `OpArrayLength` queries the runtime-array length of the
/// `(set, binding)` descriptor — the lowering of `length(view)`. Distinct from
/// `assert_storage_descriptor_is_accessed`: a length query is an `OpArrayLength`
/// on the buffer struct, not an `OpAccessChain` into it.
fn assert_array_length_queried_on_descriptor(spirv_words: &[u32], set: u32, binding: u32) {
    use rspirv::binary::parse_words;
    use rspirv::dr::{Loader, Operand};
    use rspirv::spirv::{Decoration, Op};
    use std::collections::HashMap;

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut sets: HashMap<u32, u32> = HashMap::new();
    let mut binds: HashMap<u32, u32> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        let target = match inst.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => continue,
        };
        match (inst.operands.get(1), inst.operands.get(2)) {
            (Some(Operand::Decoration(Decoration::DescriptorSet)), Some(Operand::LiteralBit32(n))) => {
                sets.insert(target, *n);
            }
            (Some(Operand::Decoration(Decoration::Binding)), Some(Operand::LiteralBit32(n))) => {
                binds.insert(target, *n);
            }
            _ => {}
        }
    }

    let target_var = sets
        .iter()
        .find(|(id, s)| **s == set && binds.get(id) == Some(&binding))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no StorageBuffer variable decorated (set={set}, binding={binding})"));

    let queried = module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == Op::ArrayLength
                    && matches!(inst.operands.first(), Some(Operand::IdRef(s)) if *s == target_var)
            })
        })
    });

    assert!(
        queried,
        "descriptor (set={set}, binding={binding}) is declared but its length is never \
         queried by an OpArrayLength — length(view) provenance was lost"
    );
}

/// `length(view)` on an entry param must query *its* descriptor. The binding is
/// baked into `_w_storage_len(set, binding)` as constants (not the side-map), so
/// this also pins that the right `(set, binding)` reaches the `OpArrayLength`.
#[test]
fn entry_length_queries_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|i: i32| xs[i] * f32.i32(length(xs)), 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("entry length(view) compiles");
    assert_array_length_queried_on_descriptor(&lowered.spirv, 2, 0);
    // and the indexed read still hits the same descriptor
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// A `scan` over a view threads that view through loop-carried block params
/// (scan-DPS phase 1). This is exactly the path `propagate_view_provenance`
/// keeps correct today; the guard pins that the loop-carried read still resolves
/// to the input descriptor, so the Tier-2 deletion of that propagation (binding
/// in the type) can't silently re-route it.
#[test]
fn scan_over_view_reads_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          scan(|a: f32, b: f32| a + b, 0.0, xs)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("scan over a view compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// Merging two views at *distinct* descriptors — `if c then xs else ys` — has
/// no single static binding, so it must not compile. Type inference unifies the
/// two branches' region variables into one; `pin_entry_regions` then tries to
/// pin that one variable to both `Region(2,0)` and `Region(2,1)`, detects the
/// conflict, and rejects it — rather than silently reading the wrong buffer.
#[test]
fn merge_of_distinct_buffers_is_a_type_error() {
    let src = r#"
        #[compute]
        entry tick(
          #[storage(set=2, binding=0, access=read)] xs: []f32,
          #[storage(set=2, binding=1, access=read)] ys: []f32,
          #[uniform(set=1, binding=0)] c: u32
        ) []f32 =
          map(|i: i32| (if c > 0u32 then xs else ys)[i], 0i32..<4)
    "#;
    let err = crate::compile_thru_spirv(src)
        .err()
        .expect("merging xs and ys (distinct descriptors) must not compile");
    let msg = format!("{err}");
    assert!(
        msg.contains("region") || msg.contains("binding") || msg.contains("descriptor"),
        "expected a region/binding-mismatch type error, got: {msg}"
    );
}

// ---- Constructor-style type conversions `T(value)` ----
//
// The `i32(x)` form dispatches via the existing per-type catalog
// entries (`i32.f32`, etc.); the `vec2i32(v)` form additionally
// desugars at `to_tlc` time into a `VecLit` of componentwise scalar
// conversion calls. These tests pin the end-to-end pipeline.

#[test]
fn ctor_scalar_constructor_compiles_to_spirv() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32,
                   #[uniform(set=1, binding=0)] n: u32) []i32 =
          map(|x: f32| i32(x), xs)
    "#;
    crate::compile_thru_spirv(src).expect("i32(f32) constructor must compile to SPIR-V");
}

#[test]
fn ctor_scalar_constructor_matches_legacy_dot_form() {
    let new = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32,
                   #[uniform(set=1, binding=0)] n: u32) []i32 =
          map(|x: f32| i32(x), xs)
    "#;
    let legacy = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32,
                   #[uniform(set=1, binding=0)] n: u32) []i32 =
          map(|x: f32| i32.f32(x), xs)
    "#;
    // Both must compile; backward-compat sanity for the legacy form.
    crate::compile_thru_spirv(new).expect("new T(value) form must compile");
    crate::compile_thru_spirv(legacy).expect("legacy T.source(value) form must still compile");
}

#[test]
fn ctor_vec2_constructor_compiles_to_spirv() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []vec2f32,
                   #[uniform(set=1, binding=0)] n: u32) []vec2i32 =
          map(|v: vec2f32| vec2i32(v), xs)
    "#;
    crate::compile_thru_spirv(src).expect("vec2i32(vec2f32) must compile to SPIR-V");
}

#[test]
fn ctor_vec3_and_vec4_constructors_compile_to_spirv() {
    let v3 = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []vec3i32,
                   #[uniform(set=1, binding=0)] n: u32) []vec3f32 =
          map(|v: vec3i32| vec3f32(v), xs)
    "#;
    let v4 = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []vec4u32,
                   #[uniform(set=1, binding=0)] n: u32) []vec4f32 =
          map(|v: vec4u32| vec4f32(v), xs)
    "#;
    crate::compile_thru_spirv(v3).expect("vec3f32(vec3i32) must compile");
    crate::compile_thru_spirv(v4).expect("vec4f32(vec4u32) must compile");
}

// ---- ArrayVariantAbstract — `filter` → size-polymorphic consumer ----
//
// `filter`'s return scheme is now `?k. Array[a, Abstract, k, no_region]`
// (was `Composite`). The producer's EGIR lowering picks Bounded for
// static-capacity inputs and View for runtime-sized ones; the consumer
// can be a size-polymorphic helper that gets specialized against the
// `Abstract` representation in TLC and resolved at the producer edge in
// EGIR. The backend-boundary verifier (`egir::verify_no_abstract`)
// rejects any residual `Array[_, Abstract, _, _]`.
//
// These pin the canonical patterns; for fusion-shape and runtime-length
// coverage see the older `filter_into_reduce_*` tests above.

#[test]
fn filter_into_user_size_poly_helper_compiles() {
    let src = r#"
def sum<[n]>(xs: [n]f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)

#[compute]
entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) f32 =
  let kept = filter(|x: f32| x > 0.0, xs) in
  sum(kept)
"#;
    crate::compile_thru_spirv(src)
        .expect("`filter` piped through a user-defined size-poly helper must compile to SPIR-V");
}

#[test]
fn filter_into_user_size_poly_helper_static_capacity() {
    // Static-capacity input exercises the Bounded producer path
    // (filter result is `{buffer: [8]f32, len: u32}`).
    let src = r#"
def sum<[n]>(xs: [n]f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)

#[compute]
entry tick() f32 =
  let kept = filter(|x: f32| x > 0.0, [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]) in
  sum(kept)
"#;
    crate::compile_thru_spirv(src)
        .expect("static-capacity filter piped through a size-poly helper must compile");
}

/// Phase 2 of the array-variant-abstract project. Particle simulator
/// the user posted that motivated `tlc::rep_specialize`: `filter` flows
/// into `center` (a user-defined size-poly helper that itself calls
/// `sum`). Both must specialize per-buffer at the call edge — `inline_small`
/// alone doesn't reach this nesting. Verifier-clean compile through
/// to SPIR-V proves the pass closed the gap.
#[test]
fn particle_sim_filter_into_user_helper_compiles() {
    let src = std::fs::read_to_string("../testfiles/playground/particles.wyn")
        .expect("read testfiles/playground/particles.wyn");
    crate::compile_thru_spirv(&src)
        .expect("particle sim with filter→center→sum chain must compile after rep_specialize");
}

/// Regression: an entry returning a tuple where the second output is a
/// fixed-size literal that *indexes into a scan result* used to silently
/// drop the second output's binding from the descriptor JSON. Root
/// cause: `lift_gathers::lift_entry` read `out_count` off `def.ty`'s
/// return slot, but `normalize_outputs` rewrites that to `SideEffect`
/// (the body's tail is an `OutputSlotStore` chain). The old
/// `storage_output_count(SideEffect)` undercounted, so the gather
/// intermediate landed on the binding slot the second output expected.
/// Fix reads `decl.outputs.len()` directly.
#[test]
fn entry_tuple_output_with_scan_indexed_literal_keeps_both_bindings() {
    use crate::pipeline_descriptor::{BufferUsage, Pipeline};
    let lowered = crate::compile_thru_spirv(
        "\
#[compute]
entry gen(xs: []i32, #[uniform(set=1,binding=0)] n: i32) ([]vec4f32, [5]i32) =
  let offsets = scan(|a:i32,b:i32| a+b, 0, xs) in
  (map(|i:i32| @[f32.i32(i),0.0,0.0,1.0], iota(64)),
   [36, offsets[n - 1], 0, 0, 0])
",
    )
    .expect("scan-into-tuple-literal must compile");
    let gen_pipeline = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) if c.entry_point == "gen" => Some(c),
            _ => None,
        })
        .expect("compute pipeline `gen` present");
    let output_names: Vec<&str> = gen_pipeline
        .bindings
        .iter()
        .filter_map(|b| match b {
            crate::pipeline_descriptor::Binding::StorageBuffer {
                usage: BufferUsage::Output,
                name,
                ..
            } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        output_names.contains(&"gen_output_0") && output_names.contains(&"gen_output_1"),
        "both gen_output_0 and gen_output_1 must be present as outputs in the descriptor; got {output_names:?}"
    );
}

/// Infix bitwise/shift operators must lower to the matching SPIR-V ops.
/// `^` → OpBitwiseXor and `<<` → OpShiftLeftLogical; the operands are
/// unsigned so this also pins the (UInt, _) arm of `lower_binop`.
#[test]
fn bitwise_shift_ops_lower_to_spirv() {
    use rspirv::binary::parse_words;
    use rspirv::dr::Loader;
    use rspirv::spirv::Op;

    let spirv = compile_to_spirv(
        "\
#[compute]
entry e(xs: []u32) []u32 = map(|x: u32| (x ^ 5u32) << 1u32, xs)
",
    )
    .expect("infix bitwise/shift compiles to SPIR-V");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let (mut xors, mut shls) = (0, 0);
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst.class.opcode {
                    Op::BitwiseXor => xors += 1,
                    Op::ShiftLeftLogical => shls += 1,
                    _ => {}
                }
            }
        }
    }
    assert!(xors >= 1, "expected at least one OpBitwiseXor, found {xors}");
    assert!(
        shls >= 1,
        "expected at least one OpShiftLeftLogical, found {shls}"
    );
}

/// A function whose body uses bitwise/shift operators with a reused let-local
/// miscompiles when it is inlined BOTH into a captured value hoisted before a
/// SOAC and into the SOAC's lambda: the local leaks as "Unknown global: w"
/// during SPIR-V generation.
///
/// Bisected trigger (all three required):
///   1. bitwise/shift body with a reused let-local: `let w = .. in (w >> _) ^ w`
///   2. the fn called to produce a *captured* value: `let k = f(7u32) in ..`
///   3. the fn *also* called inside the SOAC lambda
/// An arithmetic-only body, a literal (non-call) `k`, or calling `f` only
/// inside the lambda each compile fine. Surfaced by the lib/rng.wyn PCG hash
/// (`pcg` has `let w = .. in (w >> 22) ^ w`, used for the hoisted key and
/// inside the per-element map).
#[test]
fn bitwise_fn_inlined_both_captured_and_in_soac_lambda_lowers() {
    compile_to_spirv(
        "\
def f(v: u32) u32 = let w = v ^ 1u32 in (w >> 1u32) ^ w
#[compute]
entry e() []f32 =
  let k = f(7u32) in
  map(|i: i32| f32.u32(f(k + u32.i32(i))), 0i32 ..< 4)
",
    )
    .expect("bitwise fn inlined both as captured value and in SOAC lambda must lower to SPIR-V");
}

/// `inner` captures `x` transitively through the intermediate lambda `outer`.
/// partial_eval inlines the constant call `nested_lambda(100)`, dissolving the
/// inner `let outer = <lambda>`; `apply_var` must apply the call through that
/// env-bound lambda, otherwise `outer` is left dangling and closure conversion
/// mis-threads its capture (`ArityMismatch`). Regression for the env-bound
/// lambda case of dissolved-let residualization.
#[test]
fn nested_transitive_capture_through_inlined_lambda_lowers() {
    let _ = compile_to_ssa(
        "\
def nested_lambda(x: i32) i32 =
  let outer = |a: i32|
    let inner = |b: i32| a + b + x in
    inner(a)
  in
  outer(5)
#[vertex]
entry v() #[builtin(position)] vec4f32 = @[f32.i32(nested_lambda(100)), 0.0, 0.0, 1.0]
",
    );
}

/// partial_eval folds integer arithmetic; a u32 multiply like `C * K`
/// overflows u32 (and its i128-free product would overflow i64), so the fold
/// must wrap mod 2^32 rather than emit an out-of-range literal
/// ("Invalid u32"). Surfaced by lib/rng.wyn's PCG hash.
#[test]
fn folded_u32_arithmetic_wraps_to_width() {
    compile_to_spirv(
        "\
def C: u32 = 2654435769u32
#[compute]
entry e() []u32 = map(|i: i32| C * 747796405u32 + 2891336453u32, 0i32 ..< 4)
",
    )
    .expect("folded overflowing u32 arithmetic must wrap, not error");
}

/// A deep chain of `let (x0, x1) = mix(x0, x1, ..)` — each step uses the
/// previous result twice — inlined by partial_eval must not duplicate the
/// residual at every use site. Doing so is exponential in the chain depth
/// (the term doubles per step) and at shallower depth also drops a binding
/// ("Unknown global: x0"). partial_eval keeps non-trivial residual `let`s
/// shared instead. Surfaced by lib/rng.wyn's Threefry `block`.
#[test]
fn deep_tuple_let_chain_keeps_sharing() {
    compile_to_spirv(
        "\
module type RA = {
  type key
  sig at(k: key, p: u32) u32
}
module g : RA = {
  type key = (u32, u32)
  def mix(a: u32, b: u32, r: u32) (u32, u32) = let y = a + b in (y, (b << r) ^ y)
  def block(c0: u32, c1: u32, k0: u32, k1: u32) (u32, u32) =
    let (x0, x1) = mix(c0 + k0, c1 + k1, 13u32) in
    let (x0, x1) = mix(x0, x1, 15u32) in
    let (x0, x1) = mix(x0, x1, 26u32) in
    let (x0, x1) = mix(x0, x1, 6u32) in
    let (x0, x1) = mix(x0, x1, 17u32) in
    let (x0, x1) = mix(x0, x1, 29u32) in
    let (x0, x1) = mix(x0, x1, 16u32) in
    let (x0, x1) = mix(x0, x1, 24u32) in
    (x0 + k0, x1 + k1)
  def at(k: key, p: u32) u32 = let (k0, k1) = k in let (r0, _) = block(p, 0u32, k0, k1) in r0
}
#[compute]
entry e() []u32 = map(|i: i32| g.at((0x9e3779b9u32, 0x243f6a88u32), u32.i32(i)), 0i32 ..< 4)
",
    )
    .expect("deep tuple-destructure let-chain must lower without blowup or dangling vars");
}
