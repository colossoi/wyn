use super::*;

#[test]
fn highlights_mdbook_chapters_and_nested_chapters() {
    let config = highlight_config().unwrap();
    for root_key in ["sections", "items"] {
        let mut book = serde_json::json!({
            root_key: [{"Chapter": {
                "content": "```wyn\ndef outer = 1\n```\n",
                "sub_items": [{"Chapter": {
                    "content": "```wyn\ndef inner = 2\n```\n",
                    "sub_items": []
                }}]
            }}]
        });
        highlight_sections(&mut book, &config).unwrap();
        let chapter = &book[root_key][0]["Chapter"];
        for content in [
            &chapter["content"],
            &chapter["sub_items"][0]["Chapter"]["content"],
        ] {
            assert!(content.as_str().unwrap().contains("class=\"hljs-keyword\">def</span>"));
        }
    }
}

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
