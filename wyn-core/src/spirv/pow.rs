//! Compiler-generated SPIR-V helpers for integer exponentiation.
//!
//! GLSL.std.450's `Pow` (op 26) is float-only — `exp(y * log(x))`,
//! transcendental. Wyn supports `i32 ** i32` and `u32 ** u32` by
//! emitting one helper function per signedness at module setup time,
//! implementing [exponentiation by squaring]:
//!
//! ```text
//! pow(base, exp):
//!     result = 1
//!     while exp > 0:
//!         if exp & 1:
//!             result *= base
//!         base *= base
//!         exp >>= 1
//!     return result
//! ```
//!
//! Each `**` use site lowers to a single `OpFunctionCall` against the
//! cached helper; the function id lives in `Constructor::int_pow_functions`.
//! Two functions are emitted unconditionally (~30 instructions each);
//! driver dead-code elimination drops them when unused.
//!
//! Signed vs unsigned differ only in the loop-exit comparison
//! (`OpSGreaterThan` vs `OpUGreaterThan`) and the right-shift
//! (`OpShiftRightArithmetic` vs `OpShiftRightLogical`). Multiplication
//! is `OpIMul` either way (two's-complement / modular).
//!
//! For a negative signed exponent, the first `exp > 0` check fails and
//! the function returns 1 — matching the mathematical convention
//! `x^(-n) = 1` for integers (negative exponents have no integer
//! representation except for ±1).
//!
//! [exponentiation by squaring]:
//!     https://en.wikipedia.org/wiki/Exponentiation_by_squaring

use wspirv::spirv;

use super::Constructor;
use crate::error::Result;

/// Emit both helper functions (`signed` and `unsigned`) and cache their
/// ids in `Constructor::int_pow_functions`. Called once per module,
/// after function forward declarations and before any function body is
/// emitted.
pub(super) fn emit_int_pow_helpers(c: &mut Constructor) -> Result<()> {
    let signed_id = emit_one(c, /* signed = */ true)?;
    c.int_pow_functions.insert(true, signed_id);
    let unsigned_id = emit_one(c, /* signed = */ false)?;
    c.int_pow_functions.insert(false, unsigned_id);
    Ok(())
}

/// Emit one variant. Returns the SPIR-V function id, which the caller
/// stores under the appropriate `signed` key.
fn emit_one(c: &mut Constructor, signed: bool) -> Result<spirv::Word> {
    let int_ty = if signed { c.i32_type } else { c.u32_type };
    let name = if signed { "_w_int_pow_i32" } else { "_w_int_pow_u32" };

    // The Constructor's begin_function opens both a vars block (for
    // OpVariable declarations) and a first code block (for the
    // initialization code); current_block is the code block on return.
    let func_id = c.begin_function(name, &["base", "exp"], &[int_ty, int_ty], int_ty)?;
    let param_base = c.param_ids[0];
    let param_exp = c.param_ids[1];

    // Function-scope mutable state: result, base, exp.
    let result_var = c.declare_variable("result", int_ty)?;
    let base_var = c.declare_variable("base", int_ty)?;
    let exp_var = c.declare_variable("exp", int_ty)?;

    // Constants. The shift amount and the bitwise-and mask are u32
    // because OpShiftRightLogical's "Shift" operand and OpBitwiseAnd's
    // operands are both bit-pattern-typed; using the loop's int_ty
    // keeps everything in lockstep.
    let zero = if signed { c.const_i32(0) } else { c.const_u32(0) };
    let one = if signed { c.const_i32(1) } else { c.const_u32(1) };

    // result = 1; base = arg_base; exp = arg_exp
    c.builder.store(result_var, one, None, [])?;
    c.builder.store(base_var, param_base, None, [])?;
    c.builder.store(exp_var, param_exp, None, [])?;

    // Block labels for the structured loop. SPIR-V requires every
    // structured construct to declare its merge target up front:
    //   header  → OpLoopMerge merge continue + OpBranchConditional
    //   body    → OpSelectionMerge after_if  + OpBranchConditional
    //   if_then → branch unconditionally to after_if
    //   after_if → branch to continue
    //   continue → squaring + shift, branch back to header
    //   merge   → load result, OpReturnValue
    let header = c.builder.id();
    let body = c.builder.id();
    let if_then = c.builder.id();
    let after_if = c.builder.id();
    let cont = c.builder.id();
    let merge = c.builder.id();

    // Entry → header
    c.builder.branch(header)?;

    // header: while (exp > 0) ...
    // SPIR-V requires OpLoopMerge to be the second-to-last instruction
    // in its block — immediately before the terminator. So compute the
    // loop condition first, then emit OpLoopMerge + OpBranchConditional
    // back-to-back.
    c.builder.begin_block(Some(header))?;
    let e_hdr = c.builder.load(int_ty, None, exp_var, None, [])?;
    let more = if signed {
        c.builder.s_greater_than(c.bool_type, None, e_hdr, zero)?
    } else {
        c.builder.u_greater_than(c.bool_type, None, e_hdr, zero)?
    };
    c.builder.loop_merge(merge, cont, spirv::LoopControl::NONE, [])?;
    c.builder.branch_conditional(more, body, merge, [])?;

    // body: if (exp & 1) result *= base
    c.builder.begin_block(Some(body))?;
    let e_body = c.builder.load(int_ty, None, exp_var, None, [])?;
    let odd_bits = c.builder.bitwise_and(int_ty, None, e_body, one)?;
    let odd_b = c.builder.i_not_equal(c.bool_type, None, odd_bits, zero)?;
    c.builder.selection_merge(after_if, spirv::SelectionControl::NONE)?;
    c.builder.branch_conditional(odd_b, if_then, after_if, [])?;

    // if_then: result *= base
    c.builder.begin_block(Some(if_then))?;
    let r = c.builder.load(int_ty, None, result_var, None, [])?;
    let b = c.builder.load(int_ty, None, base_var, None, [])?;
    let new_r = c.builder.i_mul(int_ty, None, r, b)?;
    c.builder.store(result_var, new_r, None, [])?;
    c.builder.branch(after_if)?;

    // after_if: fall through to continue.
    c.builder.begin_block(Some(after_if))?;
    c.builder.branch(cont)?;

    // continue: base *= base; exp >>= 1; branch back to header
    c.builder.begin_block(Some(cont))?;
    let b2 = c.builder.load(int_ty, None, base_var, None, [])?;
    let new_base = c.builder.i_mul(int_ty, None, b2, b2)?;
    c.builder.store(base_var, new_base, None, [])?;
    let e_cont = c.builder.load(int_ty, None, exp_var, None, [])?;
    let new_exp = if signed {
        c.builder.shift_right_arithmetic(int_ty, None, e_cont, one)?
    } else {
        c.builder.shift_right_logical(int_ty, None, e_cont, one)?
    };
    c.builder.store(exp_var, new_exp, None, [])?;
    c.builder.branch(header)?;

    // merge: return result
    c.builder.begin_block(Some(merge))?;
    let final_r = c.builder.load(int_ty, None, result_var, None, [])?;
    c.builder.ret_value(final_r)?;

    c.end_function()?;
    Ok(func_id)
}
