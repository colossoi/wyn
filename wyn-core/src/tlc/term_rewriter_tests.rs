use super::{data, RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter};
use crate::ast::{Span, TypeName};
use polytype::Type;

type TestTerm = Term<data::Empty, data::Empty>;
type TestTermKind = TermKind<data::Empty, data::Empty>;

fn unit_term(term_ids: &mut TermIdSource, kind: TestTermKind) -> TestTerm {
    TestTerm {
        id: term_ids.next_id(),
        ty: Type::Constructed(TypeName::Unit, Vec::new()),
        span: Span::dummy(),
        kind,
    }
}

struct ReplaceTarget<'a> {
    target: TermId,
    term_ids: &'a mut TermIdSource,
    applied: bool,
}

impl TermRewriter<data::Empty, data::Empty> for ReplaceTarget<'_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut TestTerm) -> RewriteDecision {
        if term.id != self.target {
            return RewriteDecision::Unchanged;
        }
        term.kind = TestTermKind::BoolLit(true);
        self.applied = true;
        RewriteDecision::Changed
    }
}

#[test]
fn consuming_rewriter_reuses_untouched_subtrees_and_refreshes_changed_path_ids() {
    let mut term_ids = TermIdSource::new();
    let boxed_leaf = unit_term(&mut term_ids, TestTermKind::UnitLit);
    let untouched = unit_term(
        &mut term_ids,
        TestTermKind::Coerce {
            inner: Box::new(boxed_leaf),
            target_ty: Type::Constructed(TypeName::Unit, Vec::new()),
        },
    );
    let untouched_id = untouched.id;
    let untouched_inner_id = match &untouched.kind {
        TestTermKind::Coerce { inner, .. } => inner.id,
        _ => unreachable!(),
    };
    let untouched_allocation = match &untouched.kind {
        TestTermKind::Coerce { inner, .. } => &**inner as *const TestTerm,
        _ => unreachable!(),
    };

    let target = unit_term(&mut term_ids, TestTermKind::UnitLit);
    let target_id = target.id;
    let root = unit_term(&mut term_ids, TestTermKind::Tuple(vec![untouched, target]));
    let root_id = root.id;
    let root_allocation = match &root.kind {
        TestTermKind::Tuple(children) => children.as_ptr(),
        _ => unreachable!(),
    };

    let mut rewriter = ReplaceTarget {
        target: target_id,
        term_ids: &mut term_ids,
        applied: false,
    };
    let rewritten = root.rewrite(&mut rewriter);
    assert!(rewriter.applied);

    let TestTermKind::Tuple(children) = &rewritten.kind else {
        panic!("expected tuple root");
    };
    assert_eq!(
        children.as_ptr(),
        root_allocation,
        "rebuilding a changed parent should retain its child allocation"
    );
    assert_ne!(rewritten.id, root_id, "changed ancestor needs a fresh ID");
    assert_eq!(children[0].id, untouched_id);
    let TestTermKind::Coerce { inner, .. } = &children[0].kind else {
        panic!("expected untouched coercion");
    };
    assert_eq!(inner.id, untouched_inner_id);
    assert_eq!(
        &**inner as *const TestTerm, untouched_allocation,
        "untouched boxed subtree allocation should be retained"
    );
    assert_ne!(children[1].id, target_id);
    assert!(matches!(children[1].kind, TestTermKind::BoolLit(true)));
}

struct RewriteFirstChildFromParent<'a> {
    parent: TermId,
    term_ids: &'a mut TermIdSource,
}

impl TermRewriter<data::Empty, data::Empty> for RewriteFirstChildFromParent<'_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node_before_children(&mut self, term: &mut TestTerm) -> RewriteDecision {
        if term.id != self.parent {
            return RewriteDecision::Unchanged;
        }
        let TestTermKind::Tuple(children) = &mut term.kind else {
            panic!("expected tuple parent");
        };
        children[0].kind = TestTermKind::BoolLit(true);
        children[0].id = self.term_ids.next_id();
        RewriteDecision::Changed
    }
}

#[test]
fn consuming_rewriter_can_resolve_an_edge_before_visiting_that_child() {
    let mut term_ids = TermIdSource::new();
    let child = unit_term(&mut term_ids, TestTermKind::UnitLit);
    let child_id = child.id;
    let root = unit_term(&mut term_ids, TestTermKind::Tuple(vec![child]));
    let root_id = root.id;

    let mut rewriter = RewriteFirstChildFromParent {
        parent: root_id,
        term_ids: &mut term_ids,
    };
    let rewritten = root.rewrite(&mut rewriter);

    let TestTermKind::Tuple(children) = rewritten.kind else {
        panic!("expected tuple root");
    };
    assert_ne!(rewritten.id, root_id);
    assert_ne!(children[0].id, child_id);
    assert!(matches!(children[0].kind, TestTermKind::BoolLit(true)));
}
