#![cfg(test)]

use crate::ast::{Span, TypeName};
use crate::tlc::monomorphize::{Monomorphizer, SpecKey, Substitution, format_type_compact};
use crate::tlc::{Term, TermIdSource, TermKind};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

/// Test helper that manages symbol table and term ID generation.
struct TestBuilder {
    symbols: SymbolTable,
    ids: TermIdSource,
}

impl TestBuilder {
    fn new() -> Self {
        TestBuilder {
            symbols: SymbolTable::new(),
            ids: TermIdSource::new(),
        }
    }

    fn sym(&mut self, name: &str) -> SymbolId {
        self.symbols.alloc(name.to_string())
    }

    fn next_id(&mut self) -> super::TermId {
        self.ids.next_id()
    }

    fn span(&self) -> Span {
        Span::dummy()
    }

    fn lookup(&self, sym: SymbolId) -> &str {
        self.symbols.get(sym).expect("BUG: symbol not in table")
    }

    fn finish(self) -> SymbolTable {
        self.symbols
    }
}

#[test]
fn test_format_type_compact() {
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    assert_eq!(format_type_compact(&f32_ty), "f32");

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    assert_eq!(format_type_compact(&i32_ty), "i32");

    let size_ty = Type::Constructed(TypeName::Size(4), vec![]);
    assert_eq!(format_type_compact(&size_ty), "n4");
}

#[test]
fn test_spec_key_empty() {
    let key = SpecKey::empty();
    assert!(!key.needs_specialization());
}

#[test]
fn test_spec_key_with_subst() {
    let mut subst = Substitution::new();
    subst.insert(0, Type::Constructed(TypeName::Float(32), vec![]));

    let key = SpecKey::new(&subst);
    assert!(key.needs_specialization());
}

#[test]
fn test_collect_application_spine() {
    // Build: f(a, b) as App(App(Var(f), a), b)
    let mut b = TestBuilder::new();
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let f_sym = b.sym("f");
    let a_sym = b.sym("a");
    let b_sym = b.sym("b");

    let f_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(f_sym),
    };

    let a_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(a_sym),
    };

    let b_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(b_sym),
    };

    let app1 = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(f_var),
            arg: Box::new(a_var.clone()),
        },
    };

    let (base, args) = Monomorphizer::collect_application_spine(&app1, &b_var);

    // Check base is f
    assert!(matches!(&base.kind, TermKind::Var(sym) if *sym == f_sym));

    // Check args are [a, b]
    assert_eq!(args.len(), 2);
    assert!(matches!(&args[0].kind, TermKind::Var(sym) if *sym == a_sym));
    assert!(matches!(&args[1].kind, TermKind::Var(sym) if *sym == b_sym));
}
