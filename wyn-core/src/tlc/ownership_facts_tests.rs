use crate::tlc::{self, TermKind, VarRef, WalkDecision};

#[test]
fn permission_survives_later_reads() {
    let source = "entry main(xs:*[]i32) ([]i32,i32) = let ys=map(|x:i32|x+1,xs) in (ys,length(xs))";
    let program = crate::compile_thru_tlc(source).unwrap();
    let entry = program.defs.iter().find(|def| matches!(def.meta, tlc::DefMeta::EntryPoint(_))).unwrap();
    assert!(entry.param_diets[0].is_consuming());
}

#[test]
fn borrowed_parameters_do_not_receive_reuse_permission() {
    let program = crate::compile_thru_tlc("entry main(xs:[]i32) []i32 = map(|x:i32|x+1,xs)").unwrap();
    let entry = program.defs.iter().find(|def| matches!(def.meta, tlc::DefMeta::EntryPoint(_))).unwrap();
    assert!(!entry.param_diets[0].is_consuming());
}

#[test]
fn functional_updates_remain_functional_at_the_tlc_boundary() {
    let program =
        crate::compile_thru_tlc("entry main(xs:*[4]i32, i:i32) [4]i32 = xs with [i] = 7").unwrap();
    let known = crate::builtins::catalog().known();
    let mut functional = 0;
    for def in &program.defs {
        def.body.walk(&mut |term: &tlc::Term<_, _>| {
            if let TermKind::Var(VarRef::Builtin { id, .. }) = term.kind {
                assert_ne!(id, known.array_with_in_place);
                functional += usize::from(id == known.array_with);
            }
            WalkDecision::Recurse
        });
    }
    assert!(functional > 0);
}
