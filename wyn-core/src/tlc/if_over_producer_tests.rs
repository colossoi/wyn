use super::*;

fn entry_map_count(source: &str) -> usize {
    let program = crate::test_pipeline::compile_thru_static_index(source);
    let entry = program.defs.iter().find(|def| matches!(def.meta, tlc::DefMeta::EntryPoint(_))).unwrap();
    fn count(term: &Term) -> usize {
        let mut result = usize::from(matches!(term.kind, TermKind::Soac(SoacOp::Map { .. })));
        term.for_each_child(&mut |child| result += count(child));
        result
    }
    count(&entry.body)
}

#[test]
fn conditional_tuple_maps_share_the_domain_before_soa_lowering() {
    // After SoA lowering the result is a tuple of arrays, so its outer type
    // no longer directly exposes the common static array dimension.
    assert_eq!(
        entry_map_count(
            "entry main(xs: [4]i32, ys: [4]i32, flag: bool) [4](i32,i32) =
             if flag then map(|x|(x,x+1),xs) else map(|y|(y,y+2),ys)"
        ),
        1
    );
}

#[test]
fn conditional_zipped_maps_need_no_parameter_repair() {
    assert_eq!(
        entry_map_count(
            "entry main(xs: [4]i32, ys: [4]i32, flag: bool) [4]i32 =
             if flag then map(|(x,y)|x+y,zip(xs,ys))
             else map(|(y,x)|y-x,zip(ys,xs))"
        ),
        1
    );
}

#[test]
fn conditional_maps_with_unproven_domains_stay_separate() {
    assert_eq!(
        entry_map_count(
            "entry main(xs: []i32, ys: []i32, flag: bool) []i32 =
             if flag then map(|x|x+1,xs) else map(|y|y+2,ys)"
        ),
        2
    );
}
