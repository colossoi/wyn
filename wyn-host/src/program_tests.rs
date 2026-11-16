use super::Expr;

#[test]
fn constant_capacities_preserve_checked_host_arithmetic() {
    let integer = |n| Box::new(Expr::Integer(n));
    assert_eq!(
        Expr::Multiply(integer(2), integer(8)).constant().unwrap(),
        Some(16)
    );
    assert_eq!(Expr::Floor(integer(-5), integer(2)).constant().unwrap(), Some(-3));
    assert_eq!(
        Expr::Ceiling(integer(-5), integer(2)).constant().unwrap(),
        Some(-2)
    );
    assert_eq!(Expr::Input("n".into()).constant().unwrap(), None);
    assert!(Expr::Add(integer(i64::MAX), integer(1)).constant().is_err());
    assert!(Expr::Floor(integer(1), integer(0)).constant().is_err());
}
