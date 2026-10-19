use super::{render_error, render_error_message, render_warning, render_warning_message, RenderError};

#[test]
fn renders_an_error_without_source() {
    assert_eq!(
        render_error_message("failed to initialize the compiler"),
        "error: failed to initialize the compiler"
    );
}

#[test]
fn renders_warnings_with_and_without_source() {
    assert_eq!(render_warning_message("unused value"), "warning: unused value");
    let rendered = render_warning("unused value", "main.wyn", "value", 0..5).unwrap();
    assert!(rendered.starts_with("warning: unused value\n  --> main.wyn:1:1"));
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
            "error: expected `vec4f32`, found `vec3f32`\n  --> shader.wyn:2:5\n   |\n 1 | fn shade() {\n 2 |     vec3f32(r, g, b)\n   |     ^^^^^^^\n 3 | }"
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
            "error: incompatible branch types\n  --> shader.wyn:2:5\n   |\n 1 | if ready {\n 2 |     first()\n   |     ^^^^^^^\n 3 |     second()\n   | ^^^^^^^^^^\n 4 | }"
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
