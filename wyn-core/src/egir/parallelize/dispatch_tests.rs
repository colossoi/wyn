use super::checked_axis_workgroups;

#[test]
fn axis_planning_uses_checked_wide_products_and_target_limits() {
    assert_eq!(checked_axis_workgroups(&[4096, 16], 1), None);
    assert_eq!(checked_axis_workgroups(&[4095, 16], 1), Some(65_520));
    assert_eq!(checked_axis_workgroups(&[u32::MAX, u32::MAX, u32::MAX], 64), None);
    assert_eq!(checked_axis_workgroups(&[2016], 64), Some(32));
}
