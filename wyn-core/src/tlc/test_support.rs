use crate::ast::Span;
use crate::{SymbolId, SymbolTable};

use super::{TermId, TermIdSource};

/// Shared symbol and term-ID allocator for TLC unit-test fixtures.
pub(crate) struct TestBuilder {
    pub(crate) ids: TermIdSource,
    symbols: SymbolTable,
}

impl TestBuilder {
    pub(crate) fn new() -> Self {
        Self {
            ids: TermIdSource::new(),
            symbols: SymbolTable::new(),
        }
    }

    pub(crate) fn sym(&mut self, name: &str) -> SymbolId {
        self.symbols.alloc(name.to_string())
    }

    pub(crate) fn next_id(&mut self) -> TermId {
        self.ids.next_id()
    }

    pub(crate) fn span(&self) -> Span {
        Span::generated()
    }

    pub(crate) fn finish(self) -> (SymbolTable, TermIdSource) {
        (self.symbols, self.ids)
    }
}
