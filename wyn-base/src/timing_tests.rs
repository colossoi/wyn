use super::{enabled, span, with_enabled};

#[test]
fn timing_scope_restores_nested_settings_and_unwinds() {
    assert!(!enabled());
    assert!(span("disabled").is_none());
    with_enabled(true, || {
        assert!(enabled());
        with_enabled(false, || assert!(!enabled()));
        assert!(enabled());
        assert!(std::thread::spawn(enabled).join().is_ok_and(|value| !value));
        let result = std::panic::catch_unwind(|| {
            with_enabled(false, || panic!("test timing scope unwind"));
        });
        assert!(result.is_err());
        assert!(enabled());
    });
    assert!(!enabled());
}
