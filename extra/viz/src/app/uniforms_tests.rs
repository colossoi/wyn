use super::{dynamic_uniform, DynamicUniform};

#[test]
fn recognizes_shadertoy_and_playground_dynamic_uniform_names() {
    for (name, expected) in [
        ("iResolution", DynamicUniform::Resolution),
        ("resolution", DynamicUniform::Resolution),
        ("iTime", DynamicUniform::Time),
        ("time", DynamicUniform::Time),
        ("iMouse", DynamicUniform::Mouse),
        ("mouse", DynamicUniform::Mouse),
        ("iFrame", DynamicUniform::Frame),
        ("frame", DynamicUniform::Frame),
    ] {
        assert_eq!(dynamic_uniform(name), Some(expected), "name: {name}");
    }
}

#[test]
fn leaves_user_uniforms_for_explicit_initialization() {
    assert_eq!(dynamic_uniform("camera"), None);
}
