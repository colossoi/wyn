//! Unit tests for the GLSL identifier sanitizer and validator.

use super::{glsl_mangle, validate_glsl_identifier};

// ---------- glsl_mangle: per-char cases ----------

#[test]
fn mangle_plain_passthrough() {
    assert_eq!(glsl_mangle("foo"), "w_foo");
    assert_eq!(glsl_mangle("a1b2c3"), "w_a1b2c3");
}

#[test]
fn mangle_dot() {
    assert_eq!(glsl_mangle("a.b"), "w_a_Db");
    assert_eq!(
        glsl_mangle("materials.pbrDistributionGGX"),
        "w_materials_DpbrDistributionGGX"
    );
}

#[test]
fn mangle_dollar() {
    assert_eq!(glsl_mangle("foo$0"), "w_foo_S0");
    assert_eq!(glsl_mangle("hof_outer$0"), "w_hof_Uouter_S0");
}

#[test]
fn mangle_underscore() {
    // Inside a name, `_` → `_U`. At the very start of the input, the
    // prefix contracts from `w_` to `w` so we never produce `__` at the
    // prefix/body boundary.
    assert_eq!(glsl_mangle("a_b"), "w_a_Ub");
    assert_eq!(glsl_mangle("_w_intrinsic_foo"), "w_Uw_Uintrinsic_Ufoo");
}

#[test]
fn mangle_mixed() {
    assert_eq!(glsl_mangle("materials.foo$0"), "w_materials_Dfoo_S0");
}

#[test]
fn mangle_empty() {
    // Empty input: no leading char, so no prefix underscore. Prefix alone.
    assert_eq!(glsl_mangle(""), "w");
}

#[test]
fn mangle_fallback_hex() {
    // Non-ASCII char exercises the `_X<hex>_` fallback. Leading fallback
    // also contracts the prefix since `-` is not alphanumeric.
    assert_eq!(glsl_mangle("a-b"), "w_a_X2d_b");
    assert_eq!(glsl_mangle("-ab"), "w_X2d_ab");
}

#[test]
fn mangle_no_double_underscore() {
    // Key invariant: no mangled output ever contains `__`.
    for input in [
        "_w_lambda_13",
        "_foo",
        ".foo",
        "$foo",
        "foo",
        "foo_bar",
        "foo.bar",
        "foo$0",
        "materials.pbrDistributionGGX",
    ] {
        let mangled = glsl_mangle(input);
        assert!(
            !mangled.contains("__"),
            "input '{}' produced '{}' which contains __",
            input,
            mangled
        );
    }
}

// ---------- glsl_mangle: injectivity property ----------

#[test]
fn mangle_injective_smoke() {
    // Pairs that would collide under naive `.`→`_`/`$`→`_` schemes must
    // produce distinct outputs here.
    let inputs = [
        "foo.bar", "foo_bar", "foo$bar", "foo_Dbar", "foo_Sbar", "foo_Ubar",
    ];
    let outputs: Vec<_> = inputs.iter().map(|s| glsl_mangle(s)).collect();
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            assert_ne!(
                outputs[i], outputs[j],
                "inputs '{}' and '{}' mangled to the same output '{}'",
                inputs[i], inputs[j], outputs[i]
            );
        }
    }
}

#[test]
fn mangle_is_idempotent_as_distinct() {
    // Running the mangler on its own output produces a legal but different
    // string — double-mangling is detectable because it differs from the
    // original single-mangle.
    let once = glsl_mangle("foo.bar");
    let twice = glsl_mangle(&once);
    assert_ne!(once, twice);
    assert!(twice.starts_with("w_w"));
}

// ---------- validate_glsl_identifier ----------

#[test]
fn validate_accepts_plain() {
    assert!(validate_glsl_identifier("iResolution").is_ok());
    assert!(validate_glsl_identifier("fragCoord").is_ok());
    assert!(validate_glsl_identifier("_internal").is_ok());
    assert!(validate_glsl_identifier("a1b2").is_ok());
}

#[test]
fn validate_rejects_illegal_chars() {
    assert!(validate_glsl_identifier("foo.bar").is_err());
    assert!(validate_glsl_identifier("foo$0").is_err());
    assert!(validate_glsl_identifier("foo-bar").is_err());
}

#[test]
fn validate_rejects_keywords() {
    assert!(validate_glsl_identifier("void").is_err());
    assert!(validate_glsl_identifier("vec4").is_err());
    assert!(validate_glsl_identifier("uniform").is_err());
    assert!(validate_glsl_identifier("discard").is_err());
}

#[test]
fn validate_rejects_gl_prefix() {
    assert!(validate_glsl_identifier("gl_Position").is_err());
    assert!(validate_glsl_identifier("gl_anything").is_err());
}

#[test]
fn validate_rejects_empty() {
    assert!(validate_glsl_identifier("").is_err());
}

#[test]
fn validate_rejects_leading_digit() {
    assert!(validate_glsl_identifier("7foo").is_err());
}
