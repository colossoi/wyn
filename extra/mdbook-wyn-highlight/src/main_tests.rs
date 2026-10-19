use super::*;

#[test]
fn highlights_only_wyn_fences_with_tree_sitter_captures() {
    let config = highlight_config().unwrap();
    let markdown = "Before\n\n```wyn\ndef shade(x: f32) f32 = x -- color\n```\n\n```text\ndef plain\n```\n";
    let highlighted = highlight_markdown(markdown, &config).unwrap();

    assert!(highlighted.contains("class=\"language-wyn hljs\""));
    assert!(highlighted.contains("class=\"hljs-keyword\">def</span>"));
    assert!(highlighted.contains("class=\"hljs-title\">shade</span>"));
    assert!(highlighted.contains("class=\"hljs-type\">f32</span>"));
    assert!(highlighted.contains("class=\"hljs-comment\">-- color</span>"));
    assert!(highlighted.contains("```text\ndef plain\n```"));
}
