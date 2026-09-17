//! Persistent sets of arena IDs. Compressed radix branches share unchanged
//! subtrees; each leaf packs 64 adjacent IDs into one word. Set operations skip
//! shared or disjoint subtrees and memoize the remaining node pairs.
use crate::LookupMap;

/// An immutable set handle, valid only in the store that created it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Set(usize);
/// The empty set, valid in every store.
pub const EMPTY: Set = Set(0);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Empty,
    Leaf {
        key: u32,
        bits: u64,
    },
    Branch {
        key: u32,
        bit: u32,
        left: Set,
        right: Set,
    },
}
impl Node {
    fn key(self) -> u32 {
        match self {
            Self::Empty => 0,
            Self::Leaf { key, .. } | Self::Branch { key, .. } => key,
        }
    }
    fn bit(self) -> u32 {
        match self {
            Self::Branch { bit, .. } => bit,
            _ => 0,
        }
    }
}

/// Interned sets of u32 IDs. Old versions remain valid until the store is
/// dropped; equal contents share a handle. Operations reuse unchanged subtrees.
pub struct Sets {
    nodes: Vec<Node>,
    intern: LookupMap<Node, Set>,
    merges: LookupMap<(u8, Set, Set), Set>,
}
impl Default for Sets {
    fn default() -> Self {
        Self {
            nodes: vec![Node::Empty],
            intern: LookupMap::new(),
            merges: LookupMap::new(),
        }
    }
}
impl Sets {
    fn node(&mut self, node: Node) -> Set {
        if let Some(&id) = self.intern.get(&node) {
            return id;
        }
        let id = Set(self.nodes.len());
        self.nodes.push(node);
        self.intern.insert(node, id);
        id
    }
    fn leaf(&mut self, key: u32, bits: u64) -> Set {
        if bits == 0 {
            EMPTY
        } else {
            self.node(Node::Leaf { key, bits })
        }
    }
    fn branch(&mut self, key: u32, bit: u32, left: Set, right: Set) -> Set {
        if left == EMPTY {
            right
        } else if right == EMPTY {
            left
        } else {
            self.node(Node::Branch {
                key,
                bit,
                left,
                right,
            })
        }
    }
    fn join(&mut self, a: Set, b: Set) -> Set {
        let ka = self.nodes[a.0].key();
        let kb = self.nodes[b.0].key();
        let bit = 1u32 << (31 - (ka ^ kb).leading_zeros());
        let key = ka & !(bit | (bit - 1));
        let (left, right) = if ka & bit == 0 { (a, b) } else { (b, a) };
        self.branch(key, bit, left, right)
    }
    pub fn singleton(&mut self, value: u32) -> Set {
        self.leaf(value >> 6, 1u64 << (value & 63))
    }
    pub fn insert(&mut self, set: Set, value: u32) -> Set {
        let value = self.singleton(value);
        self.union(set, value)
    }
    pub fn union(&mut self, a: Set, b: Set) -> Set {
        self.merge(0, a, b)
    }
    pub fn intersection(&mut self, a: Set, b: Set) -> Set {
        self.merge(1, a, b)
    }
    pub fn difference(&mut self, a: Set, b: Set) -> Set {
        self.merge(2, a, b)
    }

    fn disjoint(&mut self, op: u8, a: Set, b: Set) -> Set {
        match op {
            0 => self.join(a, b),
            1 => EMPTY,
            _ => a,
        }
    }
    fn merge(&mut self, op: u8, mut a: Set, mut b: Set) -> Set {
        if a == b {
            return if op == 2 { EMPTY } else { a };
        }
        if a == EMPTY {
            return if op == 0 { b } else { EMPTY };
        }
        if b == EMPTY {
            return if op == 1 { EMPTY } else { a };
        }
        if op != 2 && a > b {
            std::mem::swap(&mut a, &mut b);
        }
        if let Some(&set) = self.merges.get(&(op, a, b)) {
            return set;
        }
        let an = self.nodes[a.0];
        let bn = self.nodes[b.0];
        let result = if an.bit() > bn.bit() {
            self.into_branch(op, a, b, true)
        } else if bn.bit() > an.bit() {
            self.into_branch(op, b, a, false)
        } else if an.key() != bn.key() {
            self.disjoint(op, a, b)
        } else {
            match (an, bn) {
                (Node::Leaf { key, bits: x }, Node::Leaf { bits: y, .. }) => self.leaf(
                    key,
                    match op {
                        0 => x | y,
                        1 => x & y,
                        _ => x & !y,
                    },
                ),
                (
                    Node::Branch {
                        key,
                        bit,
                        left: al,
                        right: ar,
                    },
                    Node::Branch {
                        left: bl, right: br, ..
                    },
                ) => {
                    let left = self.merge(op, al, bl);
                    let right = self.merge(op, ar, br);
                    self.branch(key, bit, left, right)
                }
                _ => unreachable!("nonempty nodes with the same branch bit have the same kind"),
            }
        };
        self.merges.insert((op, a, b), result);
        result
    }
    // Descend only the matching branch; a compressed prefix may prove the two
    // sets disjoint without visiting any member. Preserve operand order for A-B.
    fn into_branch(&mut self, op: u8, tree: Set, other: Set, tree_first: bool) -> Set {
        let Node::Branch {
            key,
            bit,
            mut left,
            mut right,
        } = self.nodes[tree.0]
        else {
            unreachable!("a larger branch bit implies an internal node")
        };
        let other_key = self.nodes[other.0].key();
        if other_key & !(bit | (bit - 1)) != key {
            return if tree_first {
                self.disjoint(op, tree, other)
            } else {
                self.disjoint(op, other, tree)
            };
        }
        let target = if other_key & bit == 0 { &mut left } else { &mut right };
        let merged =
            if tree_first { self.merge(op, *target, other) } else { self.merge(op, other, *target) };
        if op == 1 || (op == 2 && !tree_first) {
            return merged;
        }
        *target = merged;
        self.branch(key, bit, left, right)
    }
    pub fn iter(&self, set: Set) -> impl Iterator<Item = u32> + '_ {
        let mut pending = vec![set];
        let (mut key, mut bits) = (0u32, 0u64);
        std::iter::from_fn(move || loop {
            if bits != 0 {
                let value = (key << 6) | bits.trailing_zeros();
                bits &= bits - 1;
                return Some(value);
            }
            match self.nodes[pending.pop()?.0] {
                Node::Empty => {}
                Node::Leaf { key: k, bits: b } => {
                    key = k;
                    bits = b;
                }
                Node::Branch { left, right, .. } => {
                    pending.push(right);
                    pending.push(left);
                }
            }
        })
    }
}

#[cfg(test)]
#[path = "persistent_sets_tests.rs"]
mod tests;
