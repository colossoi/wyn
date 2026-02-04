//! SSA verification pass.
//!
//! Verifies that an SSA function body satisfies the key invariants:
//! - Every value is defined before use
//! - Every value is defined exactly once
//! - Every block has a terminator
//! - Block argument counts match at branch sites
//! - Effect token chains are well-formed

use std::collections::{HashMap, HashSet};

use super::ssa::{BlockId, EffectToken, FuncBody, InstKind, Terminator, ValueId};

/// Verification error.
#[derive(Debug, Clone)]
pub enum VerifyError {
    /// A value was used before it was defined.
    UseBeforeDef {
        value: ValueId,
        use_block: BlockId,
        use_inst: Option<usize>,
    },

    /// A value was defined multiple times.
    MultipleDef {
        value: ValueId,
        first_def: DefLocation,
        second_def: DefLocation,
    },

    /// A block has no terminator.
    MissingTerminator {
        block: BlockId,
    },

    /// Branch passes wrong number of arguments to target block.
    BlockArgCountMismatch {
        branch_block: BlockId,
        target_block: BlockId,
        expected: usize,
        got: usize,
    },

    /// An effect token was used that wasn't produced.
    UndefinedEffectToken {
        token: EffectToken,
        use_block: BlockId,
    },

    /// An effect token was produced multiple times.
    DuplicateEffectToken {
        token: EffectToken,
    },
}

/// Location where a value was defined.
#[derive(Debug, Clone)]
pub enum DefLocation {
    /// Function parameter.
    Param(usize),
    /// Block parameter.
    BlockParam(BlockId, usize),
    /// Instruction result.
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
                        "Value {} used before definition in block {} at instruction {}",
                        value, use_block, inst
                    )
                } else {
                    write!(
                        f,
                        "Value {} used before definition in block {} (terminator)",
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
                    "Value {} defined multiple times: {:?} and {:?}",
                    value, first_def, second_def
                )
            }
            VerifyError::MissingTerminator { block } => {
                write!(f, "Block {} has no terminator", block)
            }
            VerifyError::BlockArgCountMismatch {
                branch_block,
                target_block,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Branch from {} to {} passes {} args, but block expects {}",
                    branch_block, target_block, got, expected
                )
            }
            VerifyError::UndefinedEffectToken { token, use_block } => {
                write!(
                    f,
                    "Effect token {} used in block {} but never produced",
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
    /// Map from value to where it's defined.
    value_defs: HashMap<ValueId, DefLocation>,
    /// Set of defined effect tokens.
    effect_defs: HashSet<EffectToken>,
    /// Collected errors.
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
        self.effect_defs.insert(self.body.entry_effect);

        // Verify each block
        for (block_idx, block) in self.body.blocks.iter().enumerate() {
            let block_id = BlockId(block_idx as u32);
            self.verify_block(block_id, block);
        }
    }

    fn verify_block(&mut self, block_id: BlockId, block: &super::ssa::Block) {
        // Register block parameters as defined
        for (i, param) in block.params.iter().enumerate() {
            self.define_value(param.value, DefLocation::BlockParam(block_id, i));
        }

        // Verify each instruction
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = self.body.get_inst(inst_id);

            // Check uses in the instruction
            self.verify_inst_uses(block_id, inst_idx, &inst.kind);

            // Register the result as defined
            if let Some(result) = inst.result {
                self.define_value(result, DefLocation::Inst(block_id, inst_idx));
            }

            // Verify effect tokens
            self.verify_inst_effects(block_id, &inst.kind);
        }

        // Verify terminator exists
        match &block.terminator {
            None => {
                self.errors.push(VerifyError::MissingTerminator { block: block_id });
            }
            Some(term) => {
                self.verify_terminator(block_id, term);
            }
        }
    }

    fn verify_inst_uses(&mut self, block_id: BlockId, inst_idx: usize, kind: &InstKind) {
        let uses = self.collect_value_uses(kind);
        for value in uses {
            self.check_value_defined(value, block_id, Some(inst_idx));
        }
    }

    fn verify_inst_effects(&mut self, block_id: BlockId, kind: &InstKind) {
        match kind {
            InstKind::Alloca {
                effect_in,
                effect_out,
                ..
            }
            | InstKind::Load {
                effect_in,
                effect_out,
                ..
            }
            | InstKind::Store {
                effect_in,
                effect_out,
                ..
            } => {
                // Check input token is defined
                if !self.effect_defs.contains(effect_in) {
                    self.errors.push(VerifyError::UndefinedEffectToken {
                        token: *effect_in,
                        use_block: block_id,
                    });
                }

                // Register output token
                if !self.effect_defs.insert(*effect_out) {
                    self.errors.push(VerifyError::DuplicateEffectToken { token: *effect_out });
                }
            }
            _ => {}
        }
    }

    fn verify_terminator(&mut self, block_id: BlockId, term: &Terminator) {
        match term {
            Terminator::Branch { target, args } => {
                // Check all args are defined
                for &arg in args {
                    self.check_value_defined(arg, block_id, None);
                }
                // Check arg count matches target block params
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
                // Check condition is defined
                self.check_value_defined(*cond, block_id, None);

                // Check then args
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

                // Check else args
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
            Terminator::Return(value) => {
                self.check_value_defined(*value, block_id, None);
            }
            Terminator::ReturnUnit | Terminator::Unreachable => {}
        }
    }

    fn collect_value_uses(&self, kind: &InstKind) -> Vec<ValueId> {
        match kind {
            InstKind::Int(_)
            | InstKind::Float(_)
            | InstKind::Bool(_)
            | InstKind::Unit
            | InstKind::String(_)
            | InstKind::Global(_)
            | InstKind::Extern(_) => vec![],

            InstKind::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            InstKind::UnaryOp { operand, .. } => vec![*operand],
            InstKind::Tuple(elems) => elems.clone(),
            InstKind::ArrayLit { elements } => elements.clone(),
            InstKind::ArrayRange { start, len, step } => {
                let mut uses = vec![*start, *len];
                if let Some(s) = step {
                    uses.push(*s);
                }
                uses
            }
            InstKind::Vector(elems) => elems.clone(),
            InstKind::Matrix(rows) => rows.iter().flatten().copied().collect(),
            InstKind::Project { base, .. } => vec![*base],
            InstKind::Index { base, index } => vec![*base, *index],
            InstKind::Call { args, .. } => args.clone(),
            InstKind::Intrinsic { args, .. } => args.clone(),
            InstKind::Alloca { .. } => vec![],
            InstKind::Load { ptr, .. } => vec![*ptr],
            InstKind::Store { ptr, value, .. } => vec![*ptr, *value],
            InstKind::StorageView { offset, len, .. } => vec![*offset, *len],
            InstKind::StorageViewIndex { view, index } => vec![*view, *index],
            InstKind::StorageViewLen { view } => vec![*view],
            InstKind::OutputPtr { .. } => vec![],
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
