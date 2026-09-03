//! Syntax-independent rendering for source-oriented error messages.
//!
//! The crate deliberately deals only in source text, UTF-8 byte ranges, and
//! human-readable messages. It has no knowledge of tokens, syntax trees, or
//! Wyn's type system.

use std::error::Error;
use std::fmt;
use std::ops::Range;

use unicode_width::UnicodeWidthChar;

const TAB_WIDTH: usize = 4;

/// Render an error associated with a half-open UTF-8 byte range in source text.
pub fn render_error(
    message: &str,
    source_name: &str,
    source: &str,
    range: Range<usize>,
) -> Result<String, RenderError> {
    validate_range(source, &range)?;
    let mut output = render_error_message(message);
    let lines = Lines::new(source);
    let start_line = lines.containing(range.start);
    let end_offset = if range.is_empty() { range.end } else { range.end - 1 };
    let end_line = lines.containing(end_offset);
    let start_column = source[lines.items[start_line].start..range.start].chars().count() + 1;
    let gutter_width = (end_line + 1).to_string().len();

    output.push_str(&format!(
        "\n  --> {}:{}:{}\n{:width$} |",
        source_name,
        start_line + 1,
        start_column,
        "",
        width = gutter_width + 1,
    ));

    for line_index in start_line..=end_line {
        let line = &lines.items[line_index];
        let rendered_line = expand_tabs(line.text(source), TAB_WIDTH);
        output.push_str(&format!(
            "\n {:>width$} | {}",
            line_index + 1,
            rendered_line,
            width = gutter_width,
        ));

        let selected_start =
            if line_index == start_line { range.start.min(line.content_end) } else { line.start };
        let selected_end =
            if line_index == end_line { range.end.min(line.content_end) } else { line.content_end };
        let prefix = &source[line.start..selected_start];
        let selected = &source[selected_start..selected_end];
        let prefix_width = display_width(prefix, 0, TAB_WIDTH);
        let marker_width = display_width(selected, prefix_width, TAB_WIDTH).max(1);

        output.push_str(&format!(
            "\n{:width$} | {}{}",
            "",
            " ".repeat(prefix_width),
            "^".repeat(marker_width),
            width = gutter_width + 1,
        ));
    }

    Ok(output)
}

/// Render an error that has no physical source location.
pub fn render_error_message(message: &str) -> String {
    format!("error: {message}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderError {
    ReversedRange {
        start: usize,
        end: usize,
    },
    OutOfBounds {
        end: usize,
        source_len: usize,
    },
    NotCharBoundary {
        offset: usize,
    },
}

impl fmt::Display for RenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReversedRange { start, end } => {
                write!(formatter, "source range starts at {start} after its end at {end}")
            }
            Self::OutOfBounds { end, source_len } => write!(
                formatter,
                "source range ends at byte {end}, past the source length of {source_len}"
            ),
            Self::NotCharBoundary { offset } => {
                write!(formatter, "byte {offset} is not a UTF-8 character boundary")
            }
        }
    }
}

impl Error for RenderError {}

fn validate_range(source: &str, range: &Range<usize>) -> Result<(), RenderError> {
    if range.start > range.end {
        return Err(RenderError::ReversedRange {
            start: range.start,
            end: range.end,
        });
    }
    if range.end > source.len() {
        return Err(RenderError::OutOfBounds {
            end: range.end,
            source_len: source.len(),
        });
    }
    for offset in [range.start, range.end] {
        if !source.is_char_boundary(offset) {
            return Err(RenderError::NotCharBoundary { offset });
        }
    }
    Ok(())
}

#[derive(Debug)]
struct Line {
    start: usize,
    content_end: usize,
}

impl Line {
    fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.content_end]
    }
}

#[derive(Debug)]
struct Lines {
    items: Vec<Line>,
}

impl Lines {
    fn new(source: &str) -> Self {
        let mut items = Vec::new();
        let mut start = 0;
        for (newline, _) in source.match_indices('\n') {
            let content_end =
                newline - usize::from(source.as_bytes().get(newline.wrapping_sub(1)) == Some(&b'\r'));
            items.push(Line { start, content_end });
            start = newline + 1;
        }
        items.push(Line {
            start,
            content_end: source.len(),
        });
        Self { items }
    }

    fn containing(&self, offset: usize) -> usize {
        match self.items.binary_search_by_key(&offset, |line| line.start) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        }
    }
}

fn expand_tabs(text: &str, tab_width: usize) -> String {
    let mut output = String::new();
    let mut column = 0;
    for ch in text.chars() {
        if ch == '\t' {
            let spaces = tab_width - column % tab_width;
            output.push_str(&" ".repeat(spaces));
            column += spaces;
        } else {
            output.push(ch);
            column += ch.width().unwrap_or(0);
        }
    }
    output
}

fn display_width(text: &str, initial: usize, tab_width: usize) -> usize {
    let mut column = initial;
    for ch in text.chars() {
        if ch == '\t' {
            column += tab_width - column % tab_width;
        } else {
            column += ch.width().unwrap_or(0);
        }
    }
    column - initial
}

#[cfg(test)]
mod tests {
    use super::{render_error, render_error_message, RenderError};

    #[test]
    fn renders_an_error_without_source() {
        assert_eq!(
            render_error_message("failed to initialize the compiler"),
            "error: failed to initialize the compiler"
        );
    }

    #[test]
    fn renders_a_single_line_range() {
        let source = "fn shade() {\n    vec3f32(r, g, b)\n}";
        let start = source.find("vec3f32").unwrap();
        let rendered = render_error(
            "expected `vec4f32`, found `vec3f32`",
            "shader.wyn",
            source,
            start..start + "vec3f32".len(),
        )
        .unwrap();

        assert_eq!(
            rendered,
            "error: expected `vec4f32`, found `vec3f32`\n  --> shader.wyn:2:5\n   |\n 2 |     vec3f32(r, g, b)\n   |     ^^^^^^^"
        );
    }

    #[test]
    fn renders_every_line_in_a_multiline_range() {
        let source = "if ready {\n    first()\n    second()\n}";
        let start = source.find("first").unwrap();
        let end = source.find("second").unwrap() + "second".len();
        let rendered = render_error("incompatible branch types", "shader.wyn", source, start..end).unwrap();

        assert_eq!(
            rendered,
            "error: incompatible branch types\n  --> shader.wyn:2:5\n   |\n 2 |     first()\n   |     ^^^^^^^\n 3 |     second()\n   | ^^^^^^^^^^"
        );
    }

    #[test]
    fn empty_range_gets_one_caret() {
        let rendered = render_error("expected an expression", "shader.wyn", "value", 5..5).unwrap();
        assert!(rendered.ends_with("\n   |      ^"));
    }

    #[test]
    fn tabs_and_wide_unicode_align_markers() {
        let source = "\t界 + value";
        let start = source.find("value").unwrap();
        let rendered = render_error("bad value", "unicode.wyn", source, start..source.len()).unwrap();
        assert!(rendered.ends_with("\n   |          ^^^^^"));
    }

    #[test]
    fn rejects_invalid_ranges() {
        let error = render_error("bad span", "source", "é", 1..2).unwrap_err();
        assert_eq!(error, RenderError::NotCharBoundary { offset: 1 });
    }
}
