use std::collections::BTreeMap;
use std::sync::Arc;

use thiserror::Error;

use crate::ModuleId;

/// Source buffers loaded so far during one graph build.
///
/// This remains separate from parsed module payloads so diagnostics can use it
/// even when graph construction stops before producing a `ModuleGraph`.
#[derive(Clone, Debug, Default)]
pub(crate) struct SourceMap {
    files: BTreeMap<ModuleId, SourceFile>,
}

impl SourceMap {
    pub(crate) fn insert(&mut self, module: ModuleId, text: Arc<str>) -> Result<(), SourceTextError> {
        let file = SourceFile::new(module, text)?;
        let previous = self.files.insert(module, file);
        assert!(previous.is_none(), "source map received a duplicate module ID");
        Ok(())
    }

    pub(crate) fn source(&self, module: ModuleId) -> Option<&str> {
        self.files.get(&module).map(SourceFile::text)
    }

    pub(crate) fn snippet(&self, span: Span) -> Result<&str, SpanError> {
        self.file(span.module().ok_or(SpanError::GeneratedSpan)?)?.snippet(span)
    }

    pub(crate) fn location(&self, span: Span) -> Result<SourceLocation, SpanError> {
        self.file(span.module().ok_or(SpanError::GeneratedSpan)?)?.location(span)
    }

    fn file(&self, module: ModuleId) -> Result<&SourceFile, SpanError> {
        self.files.get(&module).ok_or(SpanError::UnknownModule { module })
    }
}

/// A half-open UTF-8 byte range within one source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextRange {
    start: u32,
    end: u32,
}

impl TextRange {
    /// Construct a range when `start <= end`.
    pub const fn new(start: u32, end: u32) -> Result<Self, SpanError> {
        if start <= end {
            Ok(Self { start, end })
        } else {
            Err(SpanError::ReversedRange { start, end })
        }
    }

    /// Start byte, inclusive.
    pub const fn start(self) -> u32 {
        self.start
    }

    /// End byte, exclusive.
    pub const fn end(self) -> u32 {
        self.end
    }

    /// Length in bytes.
    pub const fn len(self) -> u32 {
        self.end - self.start
    }

    /// True when the range contains no bytes.
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// A source range associated with its physical source module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    module: Option<ModuleId>,
    range: TextRange,
}

impl Span {
    pub const fn new(module: ModuleId, range: TextRange) -> Self {
        Self {
            module: Some(module),
            range,
        }
    }

    /// A span for compiler-generated syntax with no physical source range.
    pub const fn generated() -> Self {
        Self {
            module: None,
            range: TextRange { start: 0, end: 0 },
        }
    }

    /// The physical source module, or `None` for generated syntax.
    pub const fn module(self) -> Option<ModuleId> {
        self.module
    }

    pub const fn range(self) -> TextRange {
        self.range
    }

    pub const fn is_generated(self) -> bool {
        self.module.is_none()
    }

    /// Cover two spans belonging to one parsed syntax tree.
    ///
    /// Generated endpoints contribute no source range. Two physical endpoints
    /// must belong to the same module; a mismatch is an internal compiler bug.
    pub fn merge(&self, other: &Self) -> Self {
        if self.is_generated() {
            return *other;
        }
        if other.is_generated() {
            return *self;
        }
        assert_eq!(
            self.module, other.module,
            "cannot merge spans from different modules"
        );
        Self {
            module: self.module,
            range: TextRange {
                start: self.range.start.min(other.range.start),
                end: self.range.end.max(other.range.end),
            },
        }
    }

    pub const fn contains(self, offset: u32) -> bool {
        self.range.start <= offset && offset < self.range.end
    }

    pub const fn size(self) -> u32 {
        self.range.len()
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.module() {
            Some(module) => write!(formatter, "{module:?}:{}..{}", self.range.start, self.range.end),
            None => formatter.write_str("generated"),
        }
    }
}

/// One-based user-facing source position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    pub line: u32,
    pub column: u32,
}

/// Error produced when storing a source buffer.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SourceTextError {
    #[error("source buffer is larger than the supported 4 GiB limit")]
    TooLarge,
}

/// Error produced when querying a source span.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SpanError {
    #[error("generated syntax has no physical source location")]
    GeneratedSpan,
    #[error("source range starts at {start} after its end at {end}")]
    ReversedRange {
        start: u32,
        end: u32,
    },
    #[error("span belongs to {actual:?}, but this source file belongs to {expected:?}")]
    WrongModule {
        expected: ModuleId,
        actual: ModuleId,
    },
    #[error("source range {start}..{end} exceeds the source length of {source_len} bytes")]
    OutOfBounds {
        start: u32,
        end: u32,
        source_len: u32,
    },
    #[error("source range boundary at byte {offset} is not a UTF-8 character boundary")]
    NotCharBoundary {
        offset: u32,
    },
    #[error("module {module:?} does not belong to this source graph")]
    UnknownModule {
        module: ModuleId,
    },
    #[error("source position cannot be represented as a 32-bit line and column")]
    PositionOverflow,
}

/// Source text and its precomputed line index for one module.
#[derive(Clone, Debug)]
pub(crate) struct SourceFile {
    module: ModuleId,
    text: Arc<str>,
    line_starts: Box<[u32]>,
}

impl SourceFile {
    /// Store a source buffer and build its line index.
    pub(crate) fn new(module: ModuleId, text: Arc<str>) -> Result<Self, SourceTextError> {
        let source_len = u32::try_from(text.len()).map_err(|_| SourceTextError::TooLarge)?;
        let mut line_starts = vec![0];
        for (offset, byte) in text.bytes().enumerate() {
            if byte == b'\n' {
                let next = u32::try_from(offset + 1).map_err(|_| SourceTextError::TooLarge)?;
                line_starts.push(next);
            }
        }
        debug_assert!(line_starts.last().copied().unwrap_or_default() <= source_len);

        Ok(Self {
            module,
            text,
            line_starts: line_starts.into_boxed_slice(),
        })
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// Validate and return the source covered by a span.
    pub(crate) fn snippet(&self, span: Span) -> Result<&str, SpanError> {
        self.validate_span(span)?;
        let range = span.range();
        Ok(&self.text[range.start() as usize..range.end() as usize])
    }

    /// Convert the start of a span to a one-based line and character column.
    pub(crate) fn location(&self, span: Span) -> Result<SourceLocation, SpanError> {
        self.validate_span(span)?;
        let offset = span.range().start();
        let line_index = match self.line_starts.binary_search(&offset) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        let line_start = self.line_starts[line_index] as usize;
        let column = self.text[line_start..offset as usize].chars().count() + 1;

        Ok(SourceLocation {
            line: u32::try_from(line_index + 1).map_err(|_| SpanError::PositionOverflow)?,
            column: u32::try_from(column).map_err(|_| SpanError::PositionOverflow)?,
        })
    }

    fn validate_span(&self, span: Span) -> Result<(), SpanError> {
        let actual = span.module().ok_or(SpanError::GeneratedSpan)?;
        if actual != self.module {
            return Err(SpanError::WrongModule {
                expected: self.module,
                actual,
            });
        }

        let range = span.range();
        let source_len = self.source_len();
        if range.end() > source_len {
            return Err(SpanError::OutOfBounds {
                start: range.start(),
                end: range.end(),
                source_len,
            });
        }
        for offset in [range.start(), range.end()] {
            if !self.text.is_char_boundary(offset as usize) {
                return Err(SpanError::NotCharBoundary { offset });
            }
        }
        Ok(())
    }

    fn source_len(&self) -> u32 {
        u32::try_from(self.text.len()).unwrap_or(u32::MAX)
    }
}
