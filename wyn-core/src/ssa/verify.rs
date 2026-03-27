//! SSA verification pass.
//!
//! Verifies that an SSA function body satisfies the key invariants:
//! - Every value is defined before use
//! - Every value is defined exactly once
//! - Block argument counts match at branch sites
//! - Effect token chains are well-formed

use std::collections::{HashMap, HashSet};

use super::types::{BlockId, EffectToken, FuncBody, Terminator, ValueId};

/// Verification error.
#[derive(Debug, Clone)]
pub enum VerifyError {
    UseBeforeDef {
        value: ValueId,
        use_block: BlockId,
        use_inst: Option<usize>,
    },
    MultipleDef {
        value: ValueId,
        first_def: DefLocation,
        second_def: DefLocation,
    },
    BlockArgCountMismatch {
        branch_block: BlockId,
        target_block: BlockId,
        expected: usize,
        got: usize,
    },
    UndefinedEffectToken {
        token: EffectToken,
        use_block: BlockId,
    },
    DuplicateEffectToken {
        token: EffectToken,
    },
}

/// Location where a value was defined.
#[derive(Debug, Clone)]
pub enum DefLocation {
    Param(usize),
    BlockParam(BlockId, usize),
    Inst(BlockId, usize),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::UseBeforeDef {
                value,
                use_block,
                use_inst,
            } => {
                if let Some(inst) = use_inst {
                    write!(
                        f,
                        "Value {:?} used before definition in block {:?} at instruction {}",
                        value, use_block, inst
                    )
                } else {
                    write!(
                        f,
                        "Value {:?} used before definition in block {:?} (terminator)",
                        value, use_block
                    )
                }
            }
            VerifyError::MultipleDef {
                value,
                first_def,
                second_def,
            } => {
                write!(
                    f,
                    "Value {:?} defined multiple times: {:?} and {:?}",
                    value, first_def, second_def
                )
            }
            VerifyError::BlockArgCountMismatch {
                branch_block,
                target_block,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Branch from {:?} to {:?} passes {} args, but block expects {}",
                    branch_block, target_block, got, expected
                )
            }
            VerifyError::UndefinedEffectToken { token, use_block } => {
                write!(
                    f,
                    "Effect token {} used in block {:?} but never produced",
                    token, use_block
                )
            }
            VerifyError::DuplicateEffectToken { token } => {
                write!(f, "Effect token {} produced multiple times", token)
            }
        }
    }
}

impl std::error::Error for VerifyError {}

/// Verify that a function body satisfies SSA invariants.
pub fn verify_func(body: &FuncBody) -> Result<(), Vec<VerifyError>> {
    let mut verifier = Verifier::new(body);
    verifier.verify();

    if verifier.errors.is_empty() { Ok(()) } else { Err(verifier.errors) }
}

struct Verifier<'a> {
    body: &'a FuncBody,
    value_defs: HashMap<ValueId, DefLocation>,
    effect_defs: HashSet<EffectToken>,
    errors: Vec<VerifyError>,
}

impl<'a> Verifier<'a> {
    fn new(body: &'a FuncBody) -> Self {
        Verifier {
            body,
            value_defs: HashMap::new(),
            effect_defs: HashSet::new(),
            errors: Vec::new(),
        }
    }

    fn verify(&mut self) {
        // Register function parameters as defined
        for (i, (value, _, _)) in self.body.params.iter().enumerate() {
            self.define_value(*value, DefLocation::Param(i));
        }

        // Register entry effect token
        self.effect_defs.insert(self.body.entry_effect());

        // Verify each block
        for (block_id, block) in &self.body.inner.blocks {
            self.verify_block(block_id, block);
        }
    }

    fn verify_block(&mut self, block_id: BlockId, block: &wyn_ssa::BasicBlock) {
        // Register block parameters as defined (skip entry block — its params
        // are already registered as function params)
        if block_id != self.body.entry_block() {
            for (i, &param) in block.params.iter().enumerate() {
                self.define_value(param, DefLocation::BlockParam(block_id, i));
            }
        }

        // Verify each instruction
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = self.body.get_inst(inst_id);

            // Check uses
            for value in inst.data.value_uses() {
                self.check_value_defined(value, block_id, Some(inst_idx));
            }

            // Register result as defined
            if let Some(result) = inst.result {
                self.define_value(result, DefLocation::Inst(block_id, inst_idx));
            }

            // Verify effect tokens (now on InstNode.effects, not in InstKind)
            if let Some((effect_in, effect_out)) = inst.effects {
                if !self.effect_defs.contains(&effect_in) {
                    self.errors.push(VerifyError::UndefinedEffectToken {
                        token: effect_in,
                        use_block: block_id,
                    });
                }
                if !self.effect_defs.insert(effect_out) {
                    self.errors.push(VerifyError::DuplicateEffectToken { token: effect_out });
                }
            }
        }

        // Verify terminator
        self.verify_terminator(block_id, &block.term);
    }

    fn verify_terminator(&mut self, block_id: BlockId, term: &Terminator) {
        match term {
            Terminator::Branch { target, args } => {
                for &arg in args {
                    self.check_value_defined(arg, block_id, None);
                }
                let target_params = self.body.get_block(*target).params.len();
                if args.len() != target_params {
                    self.errors.push(VerifyError::BlockArgCountMismatch {
                        branch_block: block_id,
                        target_block: *target,
                        expected: target_params,
                        got: args.len(),
                    });
                }
            }
            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                self.check_value_defined(*cond, block_id, None);

                for &arg in then_args {
                    self.check_value_defined(arg, block_id, None);
                }
                let then_params = self.body.get_block(*then_target).params.len();
                if then_args.len() != then_params {
                    self.errors.push(VerifyError::BlockArgCountMismatch {
                        branch_block: block_id,
                        target_block: *then_target,
                        expected: then_params,
                        got: then_args.len(),
                    });
                }

                for &arg in else_args {
                    self.check_value_defined(arg, block_id, None);
                }
                let else_params = self.body.get_block(*else_target).params.len();
                if else_args.len() != else_params {
                    self.errors.push(VerifyError::BlockArgCountMismatch {
                        branch_block: block_id,
                        target_block: *else_target,
                        expected: else_params,
                        got: else_args.len(),
                    });
                }
            }
            Terminator::Return(Some(value)) => {
                self.check_value_defined(*value, block_id, None);
            }
            Terminator::Return(None) | Terminator::Unreachable => {}
        }
    }

    fn define_value(&mut self, value: ValueId, loc: DefLocation) {
        if let Some(existing) = self.value_defs.get(&value) {
            self.errors.push(VerifyError::MultipleDef {
                value,
                first_def: existing.clone(),
                second_def: loc,
            });
        } else {
            self.value_defs.insert(value, loc);
        }
    }

    fn check_value_defined(&mut self, value: ValueId, block_id: BlockId, inst_idx: Option<usize>) {
        if !self.value_defs.contains_key(&value) {
            self.errors.push(VerifyError::UseBeforeDef {
                value,
                use_block: block_id,
                use_inst: inst_idx,
            });
        }
    }
}
