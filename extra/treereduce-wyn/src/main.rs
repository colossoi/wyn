use std::collections::HashMap;

use anyhow::Result;

fn main() -> Result<()> {
    // Candidate substitutions the reducer tries for each node kind.
    // Keys are tree-sitter node kinds from `tree-sitter-wyn`; values
    // are fallback strings the reducer can splice in if the original
    // subtree is more complex than the bug needs.
    //
    // For expression-position slots we use wyn's `???` type hole —
    // it's polymorphic at type-check time, so one placeholder fits
    // every site regardless of expected type. The wyn compiler then
    // exits 2 on a successful type-hole-only compile; the
    // interestingness script treats exit 2 as uninteresting so the
    // reducer only keeps substitutions that STILL trigger the
    // target bug (sqrt panic, exit 101).
    let mut r: HashMap<&'static str, &'static [&'static str]> = HashMap::new();

    let hole: &[&str] = &["???"];
    r.insert("call_expression", hole);
    r.insert("let_expression", hole);
    r.insert("if_expression", hole);
    r.insert("loop_expression", hole);
    r.insert("match_expression", hole);
    r.insert("field_expression", hole);
    r.insert("index_expression", hole);
    r.insert("unary_expression", hole);
    r.insert("binary_expression", hole);
    r.insert("type_ascription", hole);
    r.insert("type_coercion", hole);
    r.insert("array_with", hole);
    r.insert("lambda_expression", hole);
    r.insert("curry_expression", hole);
    r.insert("tuple_expression", hole);
    r.insert("array_literal", hole);
    r.insert("vec_literal", hole);
    r.insert("record_expression", hole);
    r.insert("integer_literal", hole);
    r.insert("float_literal", hole);

    // Top-level declarations: empty string deletes the whole node.
    // Lets the reducer strip dead `def` / `binding` forms that the
    // expression-level hole rewrites can't reach.
    let del: &[&str] = &[""];
    r.insert("def_declaration", del);
    r.insert("binding_declaration", del);

    // Patterns: `_` wildcard is wyn's polymorphic pattern — matches
    // anywhere a binding pattern is expected.
    let wc: &[&str] = &["_"];
    r.insert("tuple_pattern", wc);
    r.insert("record_pattern", wc);
    r.insert("typed_pattern", wc);
    r.insert("attributed_pattern", wc);
    r.insert("constructor_pattern", wc);

    treereduce::cli::main(
        tree_sitter_wyn::LANGUAGE.into(),
        tree_sitter_wyn::NODE_TYPES,
        r,
    )
}
