//! Structured control flow IR for textual-shader emission.
//!
//! Converts the SSA CFG (blocks + branches) into a tree that maps directly
//! to the control-flow constructs WGSL exposes: sequential statements,
//! if-else, while loops. SPIR-V emission keeps the SSA CFG directly and
//! doesn't go through this; it's needed only for backends whose output is
//! structured source rather than a basic-block graph.

use crate::flow::{BlockId, ControlHeader};
use crate::ssa::ir::InstId;
use crate::ssa::types::{FuncBody, Terminator, ValueId, ValueRef};
use crate::LookupSet;

/// A node in the structured control flow tree.
#[derive(Debug)]
pub enum Node {
    /// Execute an SSA instruction (becomes a `let` or side-effect statement).
    Inst(InstId),

    /// `if (cond) { then_body } else { else_body }`
    /// `merge_params` are declared before the if; each branch assigns to them.
    If {
        cond: ValueRef,
        then_body: Vec<Node>,
        then_args: Vec<ValueRef>,
        else_body: Vec<Node>,
        else_args: Vec<ValueRef>,
        merge_params: Vec<ValueId>,
    },

    /// `state = init; while (cond) { body; state = continue_values; }`
    Loop {
        /// Header block params (the loop state variables).
        state_vars: Vec<ValueId>,
        /// Initial values for state vars (from the branch into the loop).
        init_args: Vec<ValueRef>,
        /// Code evaluated on every test, including conditional expressions.
        header: Vec<Node>,
        /// The condition value.
        cond: ValueRef,
        /// Whether the condition is "continue when true" (false = invert).
        cond_is_continue: bool,
        /// The loop body.
        body: Vec<Node>,
    },

    /// `return expr;`
    Return(Option<ValueRef>),

    /// Assign a value to a variable (used for loop state updates, etc.).
    Assign {
        target: ValueId,
        value: ValueRef,
    },
}

/// Convert an SSA function body into structured control flow nodes.
pub fn structurize(body: &FuncBody) -> Vec<Node> {
    let ctx = StructCtx {
        body,
        depth: std::cell::Cell::new(0),
    };
    ctx.lower_from(body.inner.entry, &[])
}

struct StructCtx<'a> {
    body: &'a FuncBody,
    depth: std::cell::Cell<usize>,
}

impl<'a> StructCtx<'a> {
    /// Lower starting from a block, producing a sequence of nodes.
    /// `args` are the values to bind to the block's params.
    fn lower_from(&self, block_id: BlockId, args: &[ValueRef]) -> Vec<Node> {
        let mut nodes = Vec::new();
        let mut current = block_id;
        let mut current_args: Vec<ValueRef> = args.to_vec();
        let mut visited = LookupSet::new();

        loop {
            if !visited.insert(current) {
                panic!(
                    "structurize: cycle detected at {:?} — CFG is not structured",
                    current
                );
            }
            let block = &self.body.inner.blocks[current];

            // Bind block params from args (emit as Assign for non-entry blocks).
            // Textual emitters handle these: first occurrence declares, later assigns.
            for (param, arg) in block.params.iter().zip(current_args.iter()) {
                if ValueRef::Ssa(*param) != *arg {
                    nodes.push(Node::Assign {
                        target: *param,
                        value: *arg,
                    });
                }
            }

            // Emit instructions
            for &inst_id in &block.insts {
                nodes.push(Node::Inst(inst_id));
            }

            // Handle terminator
            match &block.term {
                Terminator::Return(val) => {
                    nodes.push(Node::Return(*val));
                    return nodes;
                }

                Terminator::Branch { target, args } => {
                    if let Some(ControlHeader::Loop { merge, .. }) =
                        self.body.inner.blocks[*target].control_header.as_ref()
                    {
                        let merge = *merge;

                        self.emit_loop(*target, args, merge, &mut nodes);
                        // Continue from the merge block
                        current = merge;
                        // Merge block params are set by the loop exit
                        current_args = Vec::new(); // already bound inside emit_loop
                        continue;
                    }

                    // Simple branch — continue to target
                    current = *target;
                    current_args = args.clone();
                }

                Terminator::CondBranch {
                    cond,
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                } => {
                    let merge = self.find_merge(current, *then_target, *else_target);
                    if let Some(merge_id) = merge {
                        self.emit_if(
                            *cond,
                            *then_target,
                            then_args,
                            *else_target,
                            else_args,
                            merge_id,
                            &mut nodes,
                        );
                        // Continue from the merge block
                        current = merge_id;
                        current_args = Vec::new(); // already bound inside emit_if
                        continue;
                    }
                    // Can't find merge — shouldn't happen for structured code
                    nodes.push(Node::Return(None));
                    return nodes;
                }

                Terminator::Unreachable => return nodes,
            }
        }
    }

    fn emit_if(
        &self,
        cond: ValueRef,
        then_target: BlockId,
        then_args: &[ValueRef],
        else_target: BlockId,
        else_args: &[ValueRef],
        merge_id: BlockId,
        nodes: &mut Vec<Node>,
    ) {
        let merge_block = &self.body.inner.blocks[merge_id];
        let merge_params: Vec<ValueId> = merge_block.params.clone();

        // Lower each arm — stops when it reaches the merge block
        let (then_body, then_exit_args) = self.lower_arm(then_target, then_args, merge_id);
        let (else_body, else_exit_args) = self.lower_arm(else_target, else_args, merge_id);

        nodes.push(Node::If {
            cond,
            then_body,
            then_args: then_exit_args,
            else_body,
            else_args: else_exit_args,
            merge_params,
        });
    }

    /// Lower a branch arm, stopping when we reach `stop_at` block.
    /// Returns the body nodes and the args passed to the stop block.
    fn lower_arm(&self, start: BlockId, args: &[ValueRef], stop_at: BlockId) -> (Vec<Node>, Vec<ValueRef>) {
        let mut nodes = Vec::new();
        let mut current = start;
        let mut current_args: Vec<ValueRef> = args.to_vec();
        let mut visited = LookupSet::new();
        let d = self.depth.get();
        self.depth.set(d + 1);
        if d > 100 {
            panic!(
                "lower_arm: depth {} exceeded, start={:?}, stop_at={:?}",
                d, start, stop_at
            );
        }

        let result = loop {
            if current == stop_at {
                break (nodes, current_args);
            }
            if !visited.insert(current) {
                panic!("lower_arm: cycle at {:?}, stop_at={:?}", current, stop_at);
            }

            let block = &self.body.inner.blocks[current];

            // Bind block params
            for (param, arg) in block.params.iter().zip(current_args.iter()) {
                if ValueRef::Ssa(*param) != *arg {
                    nodes.push(Node::Assign {
                        target: *param,
                        value: *arg,
                    });
                }
            }

            // Emit instructions
            for &inst_id in &block.insts {
                nodes.push(Node::Inst(inst_id));
            }

            match &block.term {
                Terminator::Branch { target, args } => {
                    if *target == stop_at {
                        break (nodes, args.clone());
                    }

                    // Check for loop
                    if let Some(ControlHeader::Loop { merge, .. }) =
                        self.body.inner.blocks[*target].control_header.as_ref()
                    {
                        let merge = *merge;

                        self.emit_loop(*target, args, merge, &mut nodes);
                        current = merge;
                        current_args = Vec::new();
                        continue;
                    }

                    current = *target;
                    current_args = args.clone();
                }

                Terminator::CondBranch {
                    cond,
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                } => {
                    let merge = self.find_merge(current, *then_target, *else_target);
                    if let Some(merge_id) = merge {
                        self.emit_if(
                            *cond,
                            *then_target,
                            then_args,
                            *else_target,
                            else_args,
                            merge_id,
                            &mut nodes,
                        );
                        current = merge_id;
                        current_args = Vec::new();
                        continue;
                    }
                    break (nodes, vec![]);
                }

                Terminator::Return(val) => {
                    nodes.push(Node::Return(*val));
                    break (nodes, vec![]);
                }

                Terminator::Unreachable => break (nodes, vec![]),
            }
        };
        self.depth.set(d);
        result
    }

    fn emit_loop(
        &self,
        header_id: BlockId,
        init_args: &[ValueRef],
        merge_id: BlockId,
        nodes: &mut Vec<Node>,
    ) {
        let header = &self.body.inner.blocks[header_id];
        let state_vars: Vec<ValueId> = header.params.clone();
        // A loop test can contain selections or nested loops before reaching
        // the conditional exit. Follow their merges to find that exit block.
        let mut test = header_id;
        let mut visited = LookupSet::new();
        loop {
            assert!(visited.insert(test), "cyclic loop test at {test:?}");
            let block = &self.body.inner.blocks[test];
            if matches!(block.term, Terminator::CondBranch { then_target, else_target, .. }
                if then_target == merge_id || else_target == merge_id)
            {
                break;
            }
            test = match block.control_header {
                Some(ControlHeader::Selection { merge }) => merge,
                Some(ControlHeader::Loop { merge, .. }) if test != header_id => merge,
                _ => match block.term {
                    Terminator::Branch { target, .. } => target,
                    _ => panic!("loop test has no conditional exit at {test:?}"),
                },
            };
        }
        let (mut header_nodes, test_args) = self.lower_arm(header_id, &[], test);
        let condition = &self.body.inner.blocks[test];
        for (&target, &value) in condition.params.iter().zip(&test_args) {
            header_nodes.push(Node::Assign { target, value });
        }
        header_nodes.extend(condition.insts.iter().copied().map(Node::Inst));
        let Terminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } = &condition.term
        else {
            unreachable!("loop test {test:?} must be conditional");
        };
        let cond_is_continue = *else_target == merge_id;
        let (body_target, body_args, exit_args) = if cond_is_continue {
            (*then_target, then_args, else_args)
        } else {
            (*else_target, else_args, then_args)
        };
        let (mut body, continue_args) = self.lower_arm(body_target, body_args, header_id);

        // Add state variable updates from the back-edge args
        for (state_var, arg) in state_vars.iter().zip(continue_args.iter()) {
            body.push(Node::Assign {
                target: *state_var,
                value: *arg,
            });
        }

        nodes.push(Node::Loop {
            state_vars,
            init_args: init_args.to_vec(),
            header: header_nodes,
            cond: *cond,
            cond_is_continue,
            body,
        });

        // Set merge block params from loop exit args (after the loop).
        let merge_block = &self.body.inner.blocks[merge_id];
        for (param, arg) in merge_block.params.iter().zip(exit_args.iter()) {
            nodes.push(Node::Assign {
                target: *param,
                value: *arg,
            });
        }
    }

    /// Find the merge block for an if-else.
    /// Prefers the annotated ControlHeader::Selection merge, falls back to BFS.
    fn find_merge(
        &self,
        source_block: BlockId,
        then_block: BlockId,
        else_block: BlockId,
    ) -> Option<BlockId> {
        // Check annotated selection merge first
        if let Some(ControlHeader::Selection { merge }) =
            self.body.inner.blocks[source_block].control_header.as_ref()
        {
            return Some(*merge);
        }
        // Fallback: BFS for first common target
        let then_reachable = self.reachable_from(then_block);
        let else_reachable: LookupSet<_> = self.reachable_from(else_block).into_iter().collect();
        then_reachable.into_iter().find(|b| else_reachable.contains(b))
    }

    fn reachable_from(&self, start: BlockId) -> Vec<BlockId> {
        let mut visited = LookupSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut result = Vec::new();
        queue.push_back(start);
        while let Some(block) = queue.pop_front() {
            if !visited.insert(block) {
                continue;
            }
            result.push(block);
            let blk = &self.body.inner.blocks[block];
            match &blk.term {
                Terminator::Branch { target, .. } => queue.push_back(*target),
                Terminator::CondBranch {
                    then_target,
                    else_target,
                    ..
                } => {
                    queue.push_back(*then_target);
                    queue.push_back(*else_target);
                }
                _ => {}
            }
        }
        result
    }
}

#[cfg(test)]
#[path = "structured_tests.rs"]
mod tests;
