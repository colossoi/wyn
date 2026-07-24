use super::apply::{OwnershipPatch, OwnershipRewriter};
use crate::ast::{Span, TypeName};
use crate::builtins::catalog;
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::{Term, TermIdSource, TermKind, VarRef};
use crate::LookupMap;
use polytype::Type;

fn unit_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, Vec::new())
}

fn term(
    term_ids: &mut TermIdSource,
    kind: TermKind<ExplicitClosurePayload, ExplicitCapturesPayload>,
) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
    Term {
        id: term_ids.next_id(),
        ty: unit_type(),
        span: Span::dummy(),
        kind,
    }
}

#[test]
fn ownership_rebuilder_consumes_patch_and_refreshes_changed_path() {
    let mut term_ids = TermIdSource::new();
    let func = term(
        &mut term_ids,
        TermKind::Var(VarRef::Builtin {
            id: catalog().known().array_with,
            overload_idx: 0,
        }),
    );
    let original_func_id = func.id;
    let args = (0..3).map(|_| term(&mut term_ids, TermKind::UnitLit)).collect();
    let app = term(
        &mut term_ids,
        TermKind::App {
            func: Box::new(func),
            args,
        },
    );
    let original_app_id = app.id;
    let untouched = term(&mut term_ids, TermKind::UnitLit);
    let untouched_id = untouched.id;
    let root = term(&mut term_ids, TermKind::Tuple(vec![untouched, app]));
    let original_root_id = root.id;

    let mut patches = LookupMap::new();
    patches.insert(original_app_id, OwnershipPatch::PromoteArrayWith);
    let mut rewriter = OwnershipRewriter {
        patches: &mut patches,
        term_ids: &mut term_ids,
    };
    let rewritten = root.rewrite(&mut rewriter);

    assert!(patches.is_empty());
    assert_ne!(rewritten.id, original_root_id);
    let TermKind::Tuple(children) = &rewritten.kind else {
        panic!("expected tuple root");
    };
    assert_eq!(children[0].id, untouched_id);
    assert_ne!(children[1].id, original_app_id);
    let TermKind::App { func, .. } = &children[1].kind else {
        panic!("expected patched App");
    };
    assert_ne!(func.id, original_func_id);
    assert!(matches!(
        &func.kind,
        TermKind::Var(VarRef::Builtin { id, .. })
            if *id == catalog().known().array_with_in_place
    ));
}
