use std::io::{self, Read};

use serde_json::Value;
use tree_sitter_highlight::{HighlightConfiguration, Highlighter, HtmlRenderer};

const HIGHLIGHT_NAMES: &[&str] = &[
    "attribute",
    "comment.line",
    "constant.builtin",
    "constant.builtin.boolean",
    "constant.numeric.float",
    "constant.numeric.integer",
    "constructor",
    "function",
    "function.call",
    "function.operator",
    "keyword",
    "namespace",
    "operator",
    "punctuation.bracket",
    "punctuation.delimiter",
    "string",
    "type",
    "type.builtin",
    "type.definition",
    "type.parameter",
    "variable",
    "variable.builtin",
    "variable.other.member",
    "variable.parameter",
];

// mdBook ships Highlight.js theme CSS. Translate Tree-sitter's richer,
// dotted capture names to the closest classes that those themes style.
const HIGHLIGHT_ATTRIBUTES: &[&[u8]] = &[
    b"class=\"hljs-meta\"",     // attribute
    b"class=\"hljs-comment\"",  // comment.line
    b"class=\"hljs-literal\"",  // constant.builtin
    b"class=\"hljs-literal\"",  // constant.builtin.boolean
    b"class=\"hljs-number\"",   // constant.numeric.float
    b"class=\"hljs-number\"",   // constant.numeric.integer
    b"class=\"hljs-symbol\"",   // constructor
    b"class=\"hljs-title\"",    // function
    b"class=\"hljs-title\"",    // function.call
    b"class=\"hljs-title\"",    // function.operator
    b"class=\"hljs-keyword\"",  // keyword
    b"class=\"hljs-title\"",    // namespace
    b"class=\"hljs-keyword\"",  // operator
    b"",                        // punctuation.bracket
    b"",                        // punctuation.delimiter
    b"class=\"hljs-string\"",   // string
    b"class=\"hljs-type\"",     // type
    b"class=\"hljs-type\"",     // type.builtin
    b"class=\"hljs-type\"",     // type.definition
    b"class=\"hljs-type\"",     // type.parameter
    b"class=\"hljs-variable\"", // variable
    b"class=\"hljs-variable\"", // variable.builtin
    b"class=\"hljs-attr\"",     // variable.other.member
    b"class=\"hljs-params\"",   // variable.parameter
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("supports") {
        return Ok(());
    }

    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut protocol: Value = serde_json::from_str(&input)?;
    let book = protocol
        .as_array_mut()
        .and_then(|parts| parts.get_mut(1))
        .ok_or("invalid mdBook preprocessor input")?;

    let config = highlight_config()?;
    highlight_sections(book, &config)?;

    serde_json::to_writer(io::stdout(), book)?;
    Ok(())
}

fn highlight_config() -> Result<HighlightConfiguration, Box<dyn std::error::Error>> {
    let language = tree_sitter_wyn::LANGUAGE.into();
    let mut config =
        HighlightConfiguration::new(language, "wyn", tree_sitter_wyn::HIGHLIGHTS_QUERY, "", "")?;
    assert_eq!(HIGHLIGHT_NAMES.len(), HIGHLIGHT_ATTRIBUTES.len());
    config.configure(HIGHLIGHT_NAMES);
    Ok(config)
}

fn highlight_sections(
    book: &mut Value,
    config: &HighlightConfiguration,
) -> Result<(), Box<dyn std::error::Error>> {
    let items = if book.get("items").is_some() { book.get_mut("items") } else { book.get_mut("sub_items") };
    let Some(sections) = items.and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for section in sections {
        let Some(chapter) = section.get_mut("Chapter") else {
            continue;
        };
        if let Some(content) = chapter.get("content").and_then(Value::as_str) {
            chapter["content"] = Value::String(highlight_markdown(content, config)?);
        }
        highlight_sections(chapter, config)?;
    }
    Ok(())
}

fn highlight_markdown(
    markdown: &str,
    config: &HighlightConfiguration,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::with_capacity(markdown.len());
    let mut lines = markdown.split_inclusive('\n').peekable();

    while let Some(line) = lines.next() {
        if line.trim_end() != "```wyn" {
            output.push_str(line);
            continue;
        }

        let mut source = String::new();
        let mut closed = false;
        for code_line in lines.by_ref() {
            if code_line.trim_end() == "```" {
                closed = true;
                break;
            }
            source.push_str(code_line);
        }
        if !closed {
            output.push_str(line);
            output.push_str(&source);
            break;
        }

        output.push_str("<pre><code class=\"language-wyn hljs\">");
        output.push_str(&highlight_html(&source, config)?);
        output.push_str("</code></pre>\n");
    }

    Ok(output)
}

fn highlight_html(
    source: &str,
    config: &HighlightConfiguration,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut highlighter = Highlighter::new();
    let events = highlighter.highlight(config, source.as_bytes(), None, |_| None)?;
    let mut renderer = HtmlRenderer::new();
    renderer.render(events, source.as_bytes(), &|highlight, output| {
        output.extend_from_slice(HIGHLIGHT_ATTRIBUTES[highlight.0]);
    })?;
    Ok(String::from_utf8(renderer.html)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlights_only_wyn_fences_with_tree_sitter_captures() {
        let config = highlight_config().unwrap();
        let markdown =
            "Before\n\n```wyn\ndef shade(x: f32) f32 = x -- color\n```\n\n```text\ndef plain\n```\n";
        let highlighted = highlight_markdown(markdown, &config).unwrap();

        assert!(highlighted.contains("class=\"language-wyn hljs\""));
        assert!(highlighted.contains("class=\"hljs-keyword\">def</span>"));
        assert!(highlighted.contains("class=\"hljs-title\">shade</span>"));
        assert!(highlighted.contains("class=\"hljs-type\">f32</span>"));
        assert!(highlighted.contains("class=\"hljs-comment\">-- color</span>"));
        assert!(highlighted.contains("```text\ndef plain\n```"));
    }
}
