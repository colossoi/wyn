use super::run;
use crate::ast::{Span, TypeName};
use crate::tlc::{
    ArrayExpr, Def, DefMeta, Lambda, Place, Program, Shape, SoacBody, SoacDestination, SoacOp, Term,
    TermIdSource, TermKind, VarRef,
};
use polytype::Type;

fn input_ae(boxed: Box<crate::tlc::Term>) -> crate::tlc::ArrayExpr {
    use crate::tlc::{ArrayExpr, TermKind};
    let t = *boxed;
    match t.kind {
        TermKind::Var(vr) => ArrayExpr::Var(vr, t.ty),
        TermKind::ArrayExpr(ae) => ae,
        other => panic!("test SOAC input must be a variable or array expr, got {other:?}"),
    }
}

fn span() -> Span {
    Span::dummy()
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn runtime_array_ty(elem: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Variable(0),
            crate::types::no_region(),
        ],
    )
}

fn static_array_ty(elem: Type<TypeName>, n: usize) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
            Type::Constructed(TypeName::Size(n), vec![]),
            crate::types::no_region(),
        ],
    )
}

fn term(kind: TermKind, ty: Type<TypeName>, ids: &mut TermIdSource) -> Term {
    Term {
        id: ids.next_id(),
        ty,
        span: span(),
        kind,
    }
}

fn range_expr(n: usize, ty: Type<TypeName>, ids: &mut TermIdSource) -> Term {
    term(
        TermKind::ArrayExpr(ArrayExpr::Range {
            start: Box::new(term(TermKind::IntLit("0".into()), i32_ty(), ids)),
            len: Box::new(term(TermKind::IntLit(n.to_string()), i32_ty(), ids)),
            step: None,
        }),
        ty,
        ids,
    )
}

/// Drive a source string to the slot just before
/// `float_runtime_index_nested_producers` (after `fuse_static_indices`), the
/// pass under test — so the inlined runtime-indexed producer is still present
/// for `run` to float.
fn prepared(source: &str) -> Program {
    crate::test_pipeline::compile_thru_static_index(source)
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
            let is_runtime_name = program.symbols.get(*name).is_some_and(|n| n == "_runtime_gather");
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
                if program.symbols.get(*sym).is_some_and(|n| n == "_runtime_gather") {
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

#[test]
fn runtime_index_inside_fused_scatter_envelope_becomes_let_bound_gather_shape() {
    let mut symbols = crate::SymbolTable::new();
    let mut ids = TermIdSource::new();

    let main = symbols.alloc("main".to_string());
    let fb = symbols.alloc("fb".to_string());
    let j = symbols.alloc("j".to_string());
    let p = symbols.alloc("p".to_string());
    let v = symbols.alloc("v".to_string());
    let i = symbols.alloc("i".to_string());
    let idx_tmp = symbols.alloc("idx_tmp".to_string());

    let nested_map = term(
        TermKind::Soac(SoacOp::Map {
            lam: SoacBody {
                lam: Lambda {
                    params: vec![(i, i32_ty())],
                    body: Box::new(term(TermKind::Var(VarRef::Symbol(i)), i32_ty(), &mut ids)),
                    ret_ty: i32_ty(),
                },
                captures: vec![],
            },
            inputs: vec![input_ae(Box::new(range_expr(
                8,
                static_array_ty(i32_ty(), 8),
                &mut ids,
            )))],
            destination: SoacDestination::Fresh,
        }),
        runtime_array_ty(i32_ty()),
        &mut ids,
    );
    let indexed = term(
        TermKind::Index {
            array: Box::new(nested_map),
            index: Box::new(term(TermKind::Var(VarRef::Symbol(j)), i32_ty(), &mut ids)),
        },
        i32_ty(),
        &mut ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), f32_ty()]);
    let tuple = term(
        TermKind::Tuple(vec![
            term(TermKind::Var(VarRef::Symbol(idx_tmp)), i32_ty(), &mut ids),
            term(TermKind::Var(VarRef::Symbol(v)), f32_ty(), &mut ids),
        ]),
        tuple_ty.clone(),
        &mut ids,
    );
    let envelope_body = term(
        TermKind::Let {
            name: idx_tmp,
            name_ty: i32_ty(),
            rhs: Box::new(indexed),
            body: Box::new(tuple),
        },
        tuple_ty.clone(),
        &mut ids,
    );
    let scatter = term(
        TermKind::Soac(SoacOp::Scatter {
            dest: Place::LocalArray {
                id: fb,
                shape: Shape(vec![]),
                elem_ty: f32_ty(),
            },
            lam: SoacBody {
                lam: Lambda {
                    params: vec![(p, i32_ty()), (v, f32_ty())],
                    body: Box::new(envelope_body),
                    ret_ty: tuple_ty,
                },
                captures: vec![],
            },
            inputs: vec![
                input_ae(Box::new(range_expr(4, static_array_ty(i32_ty(), 4), &mut ids))),
                input_ae(Box::new(range_expr(4, static_array_ty(f32_ty(), 4), &mut ids))),
            ],
        }),
        runtime_array_ty(f32_ty()),
        &mut ids,
    );
    let program = Program {
        defs: vec![Def {
            name: main,
            ty: unit_ty(),
            body: scatter,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: std::collections::HashMap::new(),
    };

    let floated = run(program);
    let body = &floated.defs[0].body;
    assert!(
        let_bound_runtime_gather(&floated, body),
        "runtime nested producer inside scatter envelope should be floated: {body:?}"
    );
    assert!(
        index_reads_runtime_gather(&floated, body),
        "scatter envelope should read the floated producer by Var: {body:?}"
    );
}
