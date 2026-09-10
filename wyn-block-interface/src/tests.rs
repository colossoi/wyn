use super::*;

#[test]
fn selection_moves_whole_columns_and_preserves_edge_order() {
    let matrix = Matrix::new(
        [10, 11, 12, 13, 14],
        [('a', vec![1, 2, 3, 4, 5]), ('b', vec![6, 2, 8, 9, 0])],
    )
    .unwrap();
    let selected = matrix.select([4, 1, 3]).unwrap();
    assert_eq!(selected.parameters().collect::<Vec<_>>(), [14, 11, 13]);
    assert_eq!(
        selected.rows().collect::<Vec<_>>(),
        [('a', vec![5, 2, 4]), ('b', vec![0, 2, 9])]
    );
    assert_eq!(selected.columns()[0].common_argument(), None);
    assert_eq!(selected.columns()[1].common_argument(), Some(2));
    assert_eq!(matrix.parameters().collect::<Vec<_>>(), [10, 11, 12, 13, 14]);
}

#[test]
fn empty_dimensions_keep_their_distinct_meanings() {
    let matrix = Matrix::new([1, 2], [(0, vec![3, 4]), (1, vec![3, 5])]).unwrap();
    let empty = matrix.select([]).unwrap();
    assert_eq!(empty.edges(), [0, 1]);
    assert_eq!(empty.rows().collect::<Vec<_>>(), [(0, vec![]), (1, vec![])]);
    let no_predecessors = Matrix::<u8, _>::new([1], []).unwrap();
    assert_eq!(no_predecessors.columns()[0].common_argument(), None);
    assert_eq!(no_predecessors.columns()[0].arguments(), []);
    assert!(Matrix::<u8, u8>::new([], []).unwrap().rows().next().is_none());
}

#[test]
fn malformed_inputs_and_selections_are_rejected() {
    assert_eq!(Matrix::<u8, _>::new([1, 1], []), Err(Error::DuplicateParameter));
    assert_eq!(Matrix::new([], [(0, vec![1])]), Err(Error::Arity));
    assert_eq!(Matrix::new([1], [(0, vec![])]), Err(Error::Arity));
    assert_eq!(
        Matrix::<_, u8>::new([], [(0, vec![]), (0, vec![])]),
        Err(Error::DuplicateEdge)
    );
    let matrix = Matrix::<u8, _>::new([1, 2], []).unwrap();
    assert_eq!(matrix.select([0, 0]), Err(Error::InvalidSelection));
    assert_eq!(matrix.select([2]), Err(Error::InvalidSelection));
}

#[test]
fn chained_selection_is_equivalent_to_composed_selection() {
    let matrix = Matrix::new([10, 11, 12, 13], [(0, vec![0, 1, 2, 3]), (1, vec![4, 5, 6, 7])]).unwrap();
    assert_eq!(
        matrix.select([3, 0, 2]).unwrap().select([2, 0]).unwrap(),
        matrix.select([2, 3]).unwrap()
    );
}
