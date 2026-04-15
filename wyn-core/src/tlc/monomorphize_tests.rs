#![cfg(test)]

use super::{Monomorphizer, SpecKey, Substitution, format_type_compact};
use crate::ast::{Span, TypeName};
use crate::tlc::{Term, TermId, TermIdSource, TermKind};
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

    fn next_id(&mut self) -> TermId {
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

// test_collect_application_spine removed: with flat App { func, args },
// spine collection is no longer needed.
