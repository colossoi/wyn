//! Refutability is checked while binding patterns, including nested patterns.

use crate::compile_thru_frontend;

#[test]
fn accepts_irrefutable_patterns() {
    for source in [
        "def f(n: i32) i32 = let x = n in x",
        "def f(n: i32) i32 = let _ = n in 0",
        "def f(n: i32) i32 = let x: i32 = n in x",
        "def f(p: (i32, i32)) i32 = let (x, y) = p in x + y",
        "def f(p: (i32, (i32, i32))) i32 = let (x, (y, z)) = p in x + y + z",
        "def f(v: vec2i32) i32 = let @[x, y] = v in x + y",
        "def f(u: ()) i32 = let () = u in 0",
        "def f(p: #boxed(i32)) i32 = let #boxed(x) = p in x",
        "def f(p: #boxed(#inner(i32))) i32 = let #boxed(#inner(_)) = p in 0",
    ] {
        assert!(compile_thru_frontend(source).is_ok(), "rejected: {source}");
    }
}

#[test]
fn rejects_nested_refutable_patterns() {
    for (source, reason) in [
        ("def f(n: i32) i32 = let 0 = n in 1", "literal"),
        ("def f(n: i32) i32 = let 0: i32 = n in 1", "literal"),
        ("def f(p: (i32, i32)) i32 = let (0, x) = p in x", "literal"),
        ("def f(p: (i32, i32)) i32 = let (x, 0) = p in x", "literal"),
        ("def f(v: vec2i32) i32 = let @[x, 0] = v in x", "literal"),
        ("def f(p: #a | #b) i32 = let #a = p in 0", "#b"),
        ("def f(p: #boxed(#a | #b)) i32 = let #boxed(#a) = p in 0", "#b"),
        ("def f(p: #boxed(i32)) i32 = let #boxed(0) = p in 0", "literal"),
        ("def f(p: #a | #b | #c) i32 = let #a = p in 0", "#c"),
    ] {
        let error = compile_thru_frontend(source).expect_err(source).to_string();
        assert!(
            error.contains("refutable") && error.contains(reason),
            "{source}: {error}"
        );
    }
}

#[test]
fn rejects_incompatible_pattern_types_before_refutability() {
    for source in [
        "def f(n: i32) i32 = let () = n in 0",
        "def f(n: i32) i32 = let #foo = n in 0",
    ] {
        let error = compile_thru_frontend(source).expect_err(source).to_string();
        assert!(!error.contains("refutable"), "{source}: {error}");
    }
}
