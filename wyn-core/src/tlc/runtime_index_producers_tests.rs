use super::run;
use crate::Compiler;
use crate::tlc::{DefMeta, Program, SoacOp, Term, TermKind, VarRef};

fn prepared(source: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).expect("parse");
    let tc = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    tc.to_tlc(&module_manager, false)
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs")
        .expose_entry_producer_helpers()
        .fuse_static_indices()
        .0
        .tlc
        .clone()
}

fn entry_body(program: &Program) -> &Term {
    program
        .defs
        .iter()
        .find(|d| matches!(d.meta, DefMeta::EntryPoint(_)))
        .map(|d| &d.body)
        .expect("entry point")
}

fn walk(term: &Term, f: &mut impl FnMut(&Term)) {
    f(term);
    term.for_each_child(&mut |c| walk(c, f));
}

fn let_bound_runtime_gather(program: &Program, term: &Term) -> bool {
    let mut found = false;
    walk(term, &mut |t| {
        if let TermKind::Let { name, rhs, .. } = &t.kind {
            let is_runtime_name = program
                .symbols
                .get(*name)
                .is_some_and(|n| n == "_runtime_gather");
            if is_runtime_name && is_liftable_producer(rhs) {
                found = true;
            }
        }
    });
    found
}

fn index_reads_runtime_gather(program: &Program, term: &Term) -> bool {
    let mut found = false;
    walk(term, &mut |t| {
        if let TermKind::Index { array, .. } = &t.kind {
            if let TermKind::Var(VarRef::Symbol(sym)) = &array.kind {
                if program
                    .symbols
                    .get(*sym)
                    .is_some_and(|n| n == "_runtime_gather")
                {
                    found = true;
                }
            }
        }
    });
    found
}

fn is_liftable_producer(term: &Term) -> bool {
    match &term.kind {
        TermKind::Let { body, .. } => is_liftable_producer(body),
        TermKind::Soac(SoacOp::Map { .. } | SoacOp::Scan { .. }) => true,
        _ => false,
    }
}

#[test]
fn runtime_index_into_inlined_producer_becomes_let_bound_gather_shape() {
    let program = prepared(
        r#"
def g(n: i32) []f32 = map(|i: i32| f32.i32(i), 0i32 ..< n)
#[compute]
entry e(j: i32) [1]f32 = [g(256)[j]]
"#,
    );
    let floated = run(program);
    let body = entry_body(&floated);
    assert!(
        let_bound_runtime_gather(&floated, body),
        "runtime nested producer should be floated into a let-bound producer: {body:?}"
    );
    assert!(
        index_reads_runtime_gather(&floated, body),
        "the runtime index should read the floated producer by Var: {body:?}"
    );
}
