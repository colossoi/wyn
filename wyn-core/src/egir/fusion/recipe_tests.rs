use super::*;

fn literal(recipes: &mut Recipes, value: u32) -> LambdaId {
    let mut builder = Builder::new(vec![], vec![]);
    let result = builder.integer(value, TypeId::from(0));
    let lambda = builder.finish(recipes, vec![result]);
    let Code::Expression(id) = lambda.body.unwrap().code else {
        panic!("expected an expression recipe");
    };
    id
}

#[test]
fn candidate_recipes_share_storage_until_written_and_roll_back_on_rejection() {
    let mut recipes = Recipes::default();
    let original = literal(&mut recipes, 1);
    let mut candidate = recipes.clone();
    assert!(Arc::ptr_eq(&recipes.expressions, &candidate.expressions));

    let rejected = literal(&mut candidate, 2);
    assert!(!Arc::ptr_eq(&recipes.expressions, &candidate.expressions));
    assert_eq!(recipes.expressions.len(), 1);
    assert_eq!(candidate.expressions.len(), 2);
    drop(candidate);

    let accepted = literal(&mut recipes, 3);
    assert_eq!(
        rejected, accepted,
        "rejected candidates must not consume recipe IDs"
    );
    for (id, value) in [(original, 1), (accepted, 3)] {
        let expression = &recipes.expressions[id];
        assert!(matches!(
            expression.nodes[expression.results[0]].kind,
            NodeKind::Integer(actual) if actual == value
        ));
    }
}
