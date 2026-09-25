//! Attach declarations to the structured blocks that own them.
//! Instructions stay in place; only their variable declarations move outward.

use crate::op::OpTag;
use crate::ssa::types::{FuncBody, InstKind, PlaceId, ValueId, ValueRef};
use crate::structured::{Body, Node, Scope, ScopeId};
use crate::{LookupMap, LookupSet};

struct Binding {
    definition: ScopeId,
    owner: ScopeId,
    assigned: bool,
    same_scope_uses: bool,
    global: bool,
}

/// Fill each scope's declarations and return values eligible for inlining.
/// The def/use maps are temporary; the emitter only needs the resulting tree.
pub(super) fn place_bindings(body: &FuncBody, tree: &mut Body) -> LookupSet<ValueId> {
    let mut analysis = Analysis {
        body,
        tree,
        bindings: LookupMap::new(),
        uses: Vec::new(),
        place_defs: body
            .inner
            .insts
            .values()
            .filter_map(|node| node.data.place_result().map(|place| (place, &node.data)))
            .collect(),
    };
    for &value in &body.inner.params {
        analysis.define(value, tree.root.id, false, true);
    }
    analysis.walk(&tree.root);
    for (value, scope) in analysis.uses {
        let binding = analysis.bindings.get_mut(&value).expect("structured value has a definition");
        binding.owner = tree.common_scope(binding.owner, scope);
        binding.same_scope_uses &= binding.definition == scope;
    }
    let mut declarations = vec![vec![]; tree.parents.len()];
    let mut inline = LookupSet::new();
    // SSA order makes declaration order deterministic.
    for value in body.inner.values.keys() {
        if let Some(binding) = analysis.bindings.get(&value) {
            if !binding.global && (binding.assigned || binding.definition != binding.owner) {
                declarations[binding.owner.0].push(value);
            } else if binding.global || binding.same_scope_uses {
                inline.insert(value);
            }
        }
    }
    attach(&mut tree.root, &mut declarations);
    inline
}

fn attach(scope: &mut Scope, declarations: &mut [Vec<ValueId>]) {
    scope.declarations = std::mem::take(&mut declarations[scope.id.0]);
    for node in &mut scope.nodes {
        match node {
            Node::If {
                then_body, else_body, ..
            } => {
                attach(then_body, declarations);
                attach(else_body, declarations);
            }
            Node::Loop { body } => attach(body, declarations),
            _ => {}
        }
    }
}

struct Analysis<'a> {
    body: &'a FuncBody,
    tree: &'a Body,
    bindings: LookupMap<ValueId, Binding>,
    uses: Vec<(ValueId, ScopeId)>,
    place_defs: LookupMap<PlaceId, &'a InstKind>,
}

impl Analysis<'_> {
    fn define(&mut self, value: ValueId, scope: ScopeId, assigned: bool, global: bool) {
        self.bindings
            .entry(value)
            .and_modify(|binding| {
                binding.owner = self.tree.common_scope(binding.owner, scope);
                binding.assigned |= assigned;
            })
            .or_insert(Binding {
                definition: scope,
                owner: scope,
                assigned,
                same_scope_uses: true,
                global,
            });
    }

    fn use_value(&mut self, value: ValueRef, scope: ScopeId) {
        if let ValueRef::Ssa(value) = value {
            self.uses.push((value, scope));
        }
    }

    fn use_place(&mut self, place: PlaceId, scope: ScopeId) {
        if let Some(&instruction) = self.place_defs.get(&place) {
            // Addresses are emitted as expressions, so their index/view
            // operands must remain visible wherever the address is used.
            for value in instruction.value_uses() {
                self.use_value(value, scope);
            }
            for parent in instruction.place_uses() {
                self.use_place(parent, scope);
            }
        }
    }

    fn walk(&mut self, scope: &Scope) {
        for node in &scope.nodes {
            match node {
                Node::Inst(id) => {
                    let instruction = self.body.get_inst(*id);
                    if let Some(result) = instruction.result {
                        let global = matches!(
                            instruction.data,
                            InstKind::Op {
                                tag: OpTag::AddressableConstant(_),
                                ..
                            }
                        );
                        self.define(
                            result,
                            if global { self.tree.root.id } else { scope.id },
                            false,
                            global,
                        );
                    }
                    for value in instruction.data.value_uses() {
                        self.use_value(value, scope.id);
                    }
                    for place in instruction.data.place_uses() {
                        self.use_place(place, scope.id);
                    }
                }
                Node::ParallelCopy(copies) => {
                    for &(target, value) in copies {
                        self.define(target, scope.id, true, false);
                        self.use_value(value, scope.id);
                    }
                }
                Node::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    self.use_value(*cond, scope.id);
                    self.walk(then_body);
                    self.walk(else_body);
                }
                Node::Loop { body } => self.walk(body),
                Node::BreakIf { cond, .. } => self.use_value(*cond, scope.id),
                Node::Return(value) => {
                    if let Some(value) = value {
                        self.use_value(*value, scope.id);
                    }
                }
            }
        }
    }
}
