use super::import;
use crate::egglog::query::Query;
use crate::tlc::{self, DefMeta, TermKind};
use crate::{compile_thru_tlc, types};

#[test]
fn source_import_resolves_coerced_tuple_fields_in_their_visible_scope() {
    let mut program = compile_thru_tlc("entry main(x: i32) i32 = x + 1").unwrap();
    let [literal_id, tuple_id, coerced_id, projected_id] = std::array::from_fn(|_| program.next_term_id());
    let entry = program.defs.iter_mut().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).unwrap();
    let TermKind::Lambda(lambda) = &mut entry.body.kind else {
        panic!("entry lambda")
    };
    let scalar = (*lambda.body).clone();
    let scalar_id = scalar.id;
    let ty = scalar.ty.clone();
    let span = scalar.span;
    let tuple_ty = types::Type::Constructed(types::TypeName::Tuple(2), vec![ty.clone(), ty.clone()]);
    let other = tlc::Term {
        id: literal_id,
        ty: ty.clone(),
        span,
        kind: TermKind::IntLit("7".into()),
    };
    let tuple = tlc::Term {
        id: tuple_id,
        ty: tuple_ty.clone(),
        span,
        kind: TermKind::Tuple(vec![other, scalar]),
    };
    let coerced = tlc::Term {
        id: coerced_id,
        ty: tuple_ty.clone(),
        span,
        kind: TermKind::Coerce {
            inner: Box::new(tuple),
            target_ty: tuple_ty,
        },
    };
    let projected = tlc::Term {
        id: projected_id,
        ty,
        span,
        kind: TermKind::TupleProj {
            tuple: Box::new(coerced),
            idx: 1,
        },
    };
    *lambda.body = projected;
    let (graph, identities) = import(&program).unwrap();
    let query = Query(&graph);
    let (&(scope, _), &projected) =
        identities.occurrences.iter().find(|((_, id), _)| *id == projected_id).unwrap();
    let scalar = identities.occurrences[&(scope, scalar_id)];
    assert_eq!(query.required("SourceAlias", (projected,)).unwrap(), scalar);
}
