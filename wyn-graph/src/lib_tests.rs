use super::*;

fn diamond(node: u8, out: &mut Vec<u8>) {
    match node {
        0 => out.extend([1, 2]),
        1 | 2 => out.push(3),
        _ => {}
    }
}

#[test]
fn reachability_handles_duplicates_and_cycles() {
    fn cyclic(node: u8, out: &mut Vec<u8>) {
        match node {
            0 => out.extend([1, 1, 2]),
            1 => out.extend([0, 3]),
            2 => out.push(3),
            _ => {}
        }
    }

    assert_eq!(
        reachable_from_ordered([0], WalkOrder::BreadthFirst, cyclic),
        vec![0, 1, 2, 3]
    );
    assert!(reaches_ordered(0, 3, WalkOrder::BreadthFirst, cyclic));
    assert!(!reaches_ordered(3, 0, WalkOrder::BreadthFirst, cyclic));
}

#[test]
fn ordered_reachability_supports_stack_order() {
    fn tree(node: u8, out: &mut Vec<u8>) {
        match node {
            0 => out.extend([1, 2]),
            1 => out.push(3),
            2 => out.push(4),
            _ => {}
        }
    }

    assert_eq!(
        reachable_from_ordered([0], WalkOrder::BreadthFirst, tree),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(
        reachable_from_ordered([0], WalkOrder::DepthFirst, tree),
        vec![0, 2, 4, 1, 3]
    );
}

#[test]
fn traversal_can_prune_or_break() {
    fn tree(node: u8, out: &mut Vec<u8>) {
        match node {
            0 => out.extend([1, 2]),
            1 => out.push(3),
            2 => out.push(4),
            _ => {}
        }
    }

    let mut visited = Vec::new();
    let _: Option<()> = walk_reachable([0], WalkOrder::BreadthFirst, tree, |node| {
        visited.push(node);
        if node == 1 {
            WalkDecision::Prune
        } else {
            WalkDecision::Continue
        }
    });
    assert_eq!(visited, vec![0, 1, 2, 4]);

    let found = find_map_reachable([0], WalkOrder::DepthFirst, tree, |node| {
        (node == 4).then_some(node * 10)
    });
    assert_eq!(found, Some(40));
    assert!(reachable_set([0], WalkOrder::DepthFirst, tree).contains(&3));
    assert!(reaches_ordered(0, 4, WalkOrder::DepthFirst, tree));
}

#[test]
fn dependency_topological_sort_orders_dependencies_first() {
    fn deps(node: u8, out: &mut Vec<u8>) {
        match node {
            2 => out.extend([0, 1]),
            3 => out.push(2),
            _ => {}
        }
    }

    let order = match topo_sort_by_dependencies([0, 1, 2, 3], deps) {
        Ok(order) => order,
        Err(err) => panic!("dependency graph should be acyclic: {err}"),
    };
    assert_eq!(order, vec![0, 1, 2, 3]);
}

#[test]
fn topological_sort_reports_cycle_members() {
    fn deps(node: u8, out: &mut Vec<u8>) {
        match node {
            0 => out.push(1),
            1 => out.push(0),
            _ => {}
        }
    }

    let err = match topo_sort_by_dependencies([0, 1, 2], deps) {
        Ok(order) => panic!("cycle should be detected, got order {order:?}"),
        Err(err) => err,
    };
    assert_eq!(err.remaining(), &[0, 1]);
}

#[test]
fn dominator_tree_excludes_unreachable_predecessors() {
    fn cfg(node: u8, out: &mut Vec<u8>) {
        match node {
            0 => out.push(1),
            1 => out.extend([2, 3]),
            2 => out.push(1),
            4 => out.push(3),
            _ => {}
        }
    }

    let tree = DominatorTree::build(0, cfg);
    assert_eq!(tree.idom(0), None);
    assert_eq!(tree.idom(1), Some(0));
    assert_eq!(tree.idom(2), Some(1));
    assert_eq!(tree.idom(3), Some(1));
    assert_eq!(tree.idom(4), None);
    assert_eq!(tree.preorder(), &[0, 1, 3, 2]);

    assert!(tree.dominates(1, 3));
    assert!(!tree.dominates(2, 3));
    assert!(!tree.is_reachable(4));
}

#[test]
fn dominator_walk_order_selects_child_and_preorder_order() {
    // `build` discovers depth-first, so `0`'s successors land in the
    // reverse of the order the callback lists them.
    let dfs = DominatorTree::build(0, diamond);
    assert_eq!(dfs.children(0), &[2, 3, 1]);
    assert_eq!(dfs.preorder(), &[0, 2, 3, 1]);

    let bfs = DominatorTree::build_ordered(0, WalkOrder::BreadthFirst, diamond);
    assert_eq!(bfs.children(0), &[1, 2, 3]);
    assert_eq!(bfs.preorder(), &[0, 1, 2, 3]);

    // Only the ordering moves; domination itself is order-independent.
    for tree in [&dfs, &bfs] {
        assert_eq!(tree.idom(1), Some(0));
        assert_eq!(tree.idom(2), Some(0));
        assert_eq!(tree.idom(3), Some(0));
        assert!(tree.dominates(0, 3));
        assert!(!tree.dominates(1, 3));
    }
}
