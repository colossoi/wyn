//! `LowerCtx` arithmetic / comparison / primop dispatch
//! (`lower_binop`, `lower_unaryop`, `lower_primop` + helpers).

use super::lower::LowerCtx;
use super::*;

impl<'a, 'b> LowerCtx<'a, 'b> {
    pub(super) fn lower_binop(
        &mut self,
        op: &str,
        lhs: spirv::Word,
        rhs: spirv::Word,
        lhs_ty: &PolyType<TypeName>,
        rhs_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        use PolyType::*;
        use TypeName::*;

        let bool_type = self.constructor.bool_type;

        match (op, lhs_ty, rhs_ty) {
            // Scalar-left mixed-type ops (must precede scalar catch-alls)
            ("*", Constructed(Float(_), _), Constructed(Vec, _)) => {
                Ok(self.constructor.builder.vector_times_scalar(result_ty, None, rhs, lhs)?)
            }
            ("*", Constructed(Float(_), _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.matrix_times_scalar(result_ty, None, rhs, lhs)?)
            }
            ("+" | "-" | "/" | "%", Constructed(Float(_) | Int(_) | UInt(_), _), Constructed(Vec, _)) => {
                let splat = self.splat_scalar(lhs, rhs_ty, result_ty)?;
                self.lower_binop(op, splat, rhs, rhs_ty, rhs_ty, result_ty)
            }

            // Float operations
            ("+", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_not_equal(bool_type, None, lhs, rhs)?)
            }
            ("**", Constructed(Float(_), _), Constructed(Int(_), _)) => {
                // Float base, integer exponent (the spec'd het case). Convert
                // the exponent to the base's float type, then call GLSL Pow.
                let glsl = self.constructor.glsl_ext_inst_id;
                let conv = self.constructor.builder.convert_s_to_f(result_ty, None, rhs)?;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(conv)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }
            ("**", Constructed(Float(_), _), Constructed(UInt(_), _)) => {
                let glsl = self.constructor.glsl_ext_inst_id;
                let conv = self.constructor.builder.convert_u_to_f(result_ty, None, rhs)?;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(conv)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }
            ("**", Constructed(Float(_), _), _) => {
                // Float base, float exponent: GLSL pow (opcode 26).
                let glsl = self.constructor.glsl_ext_inst_id;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(rhs)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }
            ("**", Constructed(Int(_), _), _) => self.emit_int_pow_call(lhs, rhs, result_ty, true),
            ("**", Constructed(UInt(_), _), _) => self.emit_int_pow_call(lhs, rhs, result_ty, false),

            // Integer operations (signed)
            ("+", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Unsigned integer operations
            ("+", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_mod(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Bitwise and shift operations (integer operands)
            ("&", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.bitwise_and(result_ty, None, lhs, rhs)?)
            }
            ("|", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.bitwise_or(result_ty, None, lhs, rhs)?)
            }
            ("^", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.bitwise_xor(result_ty, None, lhs, rhs)?)
            }
            ("<<", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.shift_left_logical(result_ty, None, lhs, rhs)?)
            }
            // Signed `>>` is arithmetic (sign-extending); unsigned is logical.
            (">>", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.shift_right_arithmetic(result_ty, None, lhs, rhs)?)
            }
            (">>", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.shift_right_logical(result_ty, None, lhs, rhs)?)
            }

            // Boolean operations
            ("&&", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_and(bool_type, None, lhs, rhs)?)
            }
            ("||", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_or(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Mixed-type multiplication: mat*mat, mat*vec, vec*mat, vec*scalar, mat*scalar
            ("*", Constructed(Mat, _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.matrix_times_matrix(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Mat, _), Constructed(Vec, _)) => {
                Ok(self.constructor.builder.matrix_times_vector(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Vec, _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.vector_times_matrix(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Vec, _), Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.vector_times_scalar(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Mat, _), Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.matrix_times_scalar(result_ty, None, lhs, rhs)?)
            }

            // Vector operations: dispatch based on element type
            (_, Constructed(Vec, _), _) => {
                // If rhs is scalar (not vec/mat), splat it to match lhs vec
                let rhs = if matches!(rhs_ty, Constructed(Float(_) | Int(_) | UInt(_), _)) {
                    self.splat_scalar(rhs, lhs_ty, result_ty)?
                } else {
                    rhs
                };

                let elem_ty = lhs_ty
                    .elem_type()
                    .ok_or_else(|| crate::err_spirv!("Vec type missing element type: {:?}", lhs_ty))?;
                match (op, elem_ty) {
                    ("+", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_add(result_ty, None, lhs, rhs)?)
                    }
                    ("-", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_sub(result_ty, None, lhs, rhs)?)
                    }
                    ("*", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_mul(result_ty, None, lhs, rhs)?)
                    }
                    ("/", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_div(result_ty, None, lhs, rhs)?)
                    }
                    ("%", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_rem(result_ty, None, lhs, rhs)?)
                    }
                    ("+", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
                    }
                    ("-", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
                    }
                    ("*", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
                    }
                    ("/", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.s_div(result_ty, None, lhs, rhs)?)
                    }
                    ("%", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.s_rem(result_ty, None, lhs, rhs)?)
                    }
                    ("+", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
                    }
                    ("-", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
                    }
                    ("*", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
                    }
                    ("/", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.u_div(result_ty, None, lhs, rhs)?)
                    }
                    ("%", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.u_mod(result_ty, None, lhs, rhs)?)
                    }

                    // Vector equality: componentwise compare → bvec,
                    // then `OpAll` / `OpAny` collapse to scalar bool.
                    // Matches GLSL `all(a == b)` / `any(a != b)`.
                    ("==" | "!=", _) => {
                        let vec_size = lhs_ty
                            .vec_size()
                            .ok_or_else(|| crate::err_spirv!("Vec type missing size: {:?}", lhs_ty))?
                            as u32;
                        let bvec_ty =
                            self.constructor.get_or_create_vec_type(self.constructor.bool_type, vec_size);
                        let cmp = match (op, elem_ty) {
                            ("==", Constructed(Float(_), _)) => {
                                self.constructor.builder.f_ord_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("!=", Constructed(Float(_), _)) => {
                                self.constructor.builder.f_ord_not_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("==", Constructed(Int(_) | UInt(_), _)) => {
                                self.constructor.builder.i_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("!=", Constructed(Int(_) | UInt(_), _)) => {
                                self.constructor.builder.i_not_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("==", Constructed(Bool, _)) => {
                                self.constructor.builder.logical_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("!=", Constructed(Bool, _)) => {
                                self.constructor.builder.logical_not_equal(bvec_ty, None, lhs, rhs)?
                            }
                            _ => bail_spirv!("Unsupported vector {} on element {:?}", op, elem_ty),
                        };
                        if op == "==" {
                            Ok(self.constructor.builder.all(result_ty, None, cmp)?)
                        } else {
                            Ok(self.constructor.builder.any(result_ty, None, cmp)?)
                        }
                    }

                    _ => bail_spirv!(
                        "Unsupported vector binary operation: {} on element {:?}",
                        op,
                        elem_ty
                    ),
                }
            }

            _ => bail_spirv!("Unsupported binary operation: {} on {:?}", op, lhs_ty),
        }
    }

    /// Lower an integer `**` to an `OpFunctionCall` against the
    /// compiler-generated helper emitted by `spirv::pow`. Bridges
    /// `lower_binop`'s inline dispatch to the shared helper used by
    /// `PrimOp::IntPow` in `lower_primop`.
    pub(super) fn emit_int_pow_call(
        &mut self,
        lhs: spirv::Word,
        rhs: spirv::Word,
        result_ty: spirv::Word,
        signed: bool,
    ) -> Result<spirv::Word> {
        let func_id = self
            .constructor
            .int_pow_functions
            .get(&signed)
            .copied()
            .ok_or_else(|| err_spirv!("int_pow helper not emitted (signed={})", signed))?;
        Ok(self.constructor.builder.function_call(result_ty, None, func_id, vec![lhs, rhs])?)
    }

    /// Splat a scalar SPIR-V value into a vector matching `vec_ty`.
    pub(super) fn splat_scalar(
        &mut self,
        scalar: spirv::Word,
        vec_ty: &PolyType<TypeName>,
        vec_spirv_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let n = vec_ty.vec_size().ok_or_else(|| {
            crate::err_spirv!("Cannot splat: vec type has no concrete size: {:?}", vec_ty)
        })?;
        let components = vec![scalar; n];
        Ok(self.constructor.builder.composite_construct(vec_spirv_ty, None, components)?)
    }

    pub(super) fn lower_unaryop(
        &mut self,
        op: &str,
        operand: spirv::Word,
        operand_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        use PolyType::*;
        use TypeName::*;

        match (op, operand_ty) {
            ("-", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_negate(result_ty, None, operand)?)
            }
            ("-", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_negate(result_ty, None, operand)?)
            }
            ("!", Constructed(Bool, _)) => {
                Ok(self.constructor.builder.logical_not(result_ty, None, operand)?)
            }
            // Vector unary operations
            ("-", Constructed(Vec, _)) => {
                let elem_ty = operand_ty
                    .elem_type()
                    .ok_or_else(|| crate::err_spirv!("Vec type missing element type: {:?}", operand_ty))?;
                match elem_ty {
                    Constructed(Float(_), _) => {
                        Ok(self.constructor.builder.f_negate(result_ty, None, operand)?)
                    }
                    Constructed(Int(_), _) => {
                        Ok(self.constructor.builder.s_negate(result_ty, None, operand)?)
                    }
                    _ => bail_spirv!(
                        "Unsupported vector unary operation: {} on element {:?}",
                        op,
                        elem_ty
                    ),
                }
            }
            _ => bail_spirv!("Unsupported unary operation: {} on {:?}", op, operand_ty),
        }
    }

    pub(super) fn lower_primop(
        &mut self,
        prim_op: &PrimOp,
        arg_ids: &[spirv::Word],
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let glsl = self.constructor.glsl_ext_inst_id;
        let operands: Vec<Operand> = arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();

        match prim_op {
            PrimOp::GlslExt(ext_op) => {
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, *ext_op, operands)?)
            }
            PrimOp::IntPow { signed } => {
                if arg_ids.len() != 2 {
                    bail_spirv!("int_pow requires 2 args");
                }
                // Function id was cached by `spirv::pow::emit_int_pow_helpers`
                // during module setup; missing means a backend-init bug.
                let func_id = self
                    .constructor
                    .int_pow_functions
                    .get(signed)
                    .copied()
                    .ok_or_else(|| err_spirv!("int_pow helper not emitted (signed={})", signed))?;
                Ok(self.constructor.builder.function_call(result_ty, None, func_id, arg_ids.to_vec())?)
            }
            PrimOp::Dot => {
                if arg_ids.len() != 2 {
                    bail_spirv!("dot requires 2 args");
                }
                Ok(self.constructor.builder.dot(result_ty, None, arg_ids[0], arg_ids[1])?)
            }
            PrimOp::MatrixTimesMatrix => {
                if arg_ids.len() != 2 {
                    bail_spirv!("matrix × matrix requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_matrix(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::MatrixTimesVector => {
                if arg_ids.len() != 2 {
                    bail_spirv!("matrix × vector requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_vector(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::VectorTimesMatrix => {
                if arg_ids.len() != 2 {
                    bail_spirv!("vector × matrix requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .vector_times_matrix(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::FPToSI => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPToSI requires 1 arg");
                }
                Ok(self.constructor.builder.convert_f_to_s(result_ty, None, arg_ids[0])?)
            }
            PrimOp::FPToUI => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPToUI requires 1 arg");
                }
                Ok(self.constructor.builder.convert_f_to_u(result_ty, None, arg_ids[0])?)
            }
            PrimOp::SIToFP => {
                if arg_ids.len() != 1 {
                    bail_spirv!("SIToFP requires 1 arg");
                }
                Ok(self.constructor.builder.convert_s_to_f(result_ty, None, arg_ids[0])?)
            }
            PrimOp::UIToFP => {
                if arg_ids.len() != 1 {
                    bail_spirv!("UIToFP requires 1 arg");
                }
                Ok(self.constructor.builder.convert_u_to_f(result_ty, None, arg_ids[0])?)
            }
            PrimOp::Bitcast => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Bitcast requires 1 arg");
                }
                Ok(self.constructor.builder.bitcast(result_ty, None, arg_ids[0])?)
            }
            PrimOp::IsNan => {
                if arg_ids.len() != 1 {
                    bail_spirv!("isnan requires 1 arg");
                }
                Ok(self.constructor.builder.is_nan(result_ty, None, arg_ids[0])?)
            }
            PrimOp::IsInf => {
                if arg_ids.len() != 1 {
                    bail_spirv!("isinf requires 1 arg");
                }
                Ok(self.constructor.builder.is_inf(result_ty, None, arg_ids[0])?)
            }
            // Additional arithmetic ops
            PrimOp::FAdd | PrimOp::FSub | PrimOp::FMul | PrimOp::FDiv | PrimOp::FRem | PrimOp::FMod => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Float binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::FAdd => {
                        Ok(self.constructor.builder.f_add(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FSub => {
                        Ok(self.constructor.builder.f_sub(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FMul => {
                        Ok(self.constructor.builder.f_mul(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FDiv => {
                        Ok(self.constructor.builder.f_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FRem => {
                        Ok(self.constructor.builder.f_rem(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FMod => {
                        Ok(self.constructor.builder.f_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    _ => unreachable!(),
                }
            }
            PrimOp::IAdd
            | PrimOp::ISub
            | PrimOp::IMul
            | PrimOp::SDiv
            | PrimOp::UDiv
            | PrimOp::SRem
            | PrimOp::SMod
            | PrimOp::UMod => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Integer binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::IAdd => {
                        Ok(self.constructor.builder.i_add(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::ISub => {
                        Ok(self.constructor.builder.i_sub(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::IMul => {
                        Ok(self.constructor.builder.i_mul(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SDiv => {
                        Ok(self.constructor.builder.s_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::UDiv => {
                        Ok(self.constructor.builder.u_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SRem => {
                        Ok(self.constructor.builder.s_rem(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SMod => {
                        Ok(self.constructor.builder.s_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::UMod => {
                        Ok(self.constructor.builder.u_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    _ => unreachable!(),
                }
            }
            // Comparison ops
            PrimOp::FOrdEqual
            | PrimOp::FOrdNotEqual
            | PrimOp::FOrdLessThan
            | PrimOp::FOrdGreaterThan
            | PrimOp::FOrdLessThanEqual
            | PrimOp::FOrdGreaterThanEqual => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Float comparison requires 2 args");
                }
                match prim_op {
                    PrimOp::FOrdEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdNotEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_not_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdLessThan => Ok(self
                        .constructor
                        .builder
                        .f_ord_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdGreaterThan => Ok(self
                        .constructor
                        .builder
                        .f_ord_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdLessThanEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            PrimOp::IEqual
            | PrimOp::INotEqual
            | PrimOp::SLessThan
            | PrimOp::ULessThan
            | PrimOp::SGreaterThan
            | PrimOp::UGreaterThan
            | PrimOp::SLessThanEqual
            | PrimOp::ULessThanEqual
            | PrimOp::SGreaterThanEqual
            | PrimOp::UGreaterThanEqual => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Integer comparison requires 2 args");
                }
                match prim_op {
                    PrimOp::IEqual => {
                        Ok(self.constructor.builder.i_equal(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::INotEqual => Ok(self
                        .constructor
                        .builder
                        .i_not_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SLessThan => Ok(self
                        .constructor
                        .builder
                        .s_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ULessThan => Ok(self
                        .constructor
                        .builder
                        .u_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SGreaterThan => Ok(self
                        .constructor
                        .builder
                        .s_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::UGreaterThan => Ok(self
                        .constructor
                        .builder
                        .u_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SLessThanEqual => Ok(self
                        .constructor
                        .builder
                        .s_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ULessThanEqual => Ok(self
                        .constructor
                        .builder
                        .u_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .s_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::UGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .u_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            // Bitwise ops
            PrimOp::BitwiseAnd | PrimOp::BitwiseOr | PrimOp::BitwiseXor => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Bitwise binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::BitwiseAnd => Ok(self
                        .constructor
                        .builder
                        .bitwise_and(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::BitwiseOr => {
                        Ok(self.constructor.builder.bitwise_or(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::BitwiseXor => Ok(self
                        .constructor
                        .builder
                        .bitwise_xor(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            PrimOp::Not => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Not requires 1 arg");
                }
                Ok(self.constructor.builder.not(result_ty, None, arg_ids[0])?)
            }
            PrimOp::ShiftLeftLogical | PrimOp::ShiftRightArithmetic | PrimOp::ShiftRightLogical => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Shift op requires 2 args");
                }
                match prim_op {
                    PrimOp::ShiftLeftLogical => Ok(self
                        .constructor
                        .builder
                        .shift_left_logical(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ShiftRightArithmetic => Ok(self
                        .constructor
                        .builder
                        .shift_right_arithmetic(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ShiftRightLogical => Ok(self
                        .constructor
                        .builder
                        .shift_right_logical(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            // Additional type conversions
            PrimOp::FPConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPConvert requires 1 arg");
                }
                Ok(self.constructor.builder.f_convert(result_ty, None, arg_ids[0])?)
            }
            PrimOp::SConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("SConvert requires 1 arg");
                }
                Ok(self.constructor.builder.s_convert(result_ty, None, arg_ids[0])?)
            }
            PrimOp::UConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("UConvert requires 1 arg");
                }
                Ok(self.constructor.builder.u_convert(result_ty, None, arg_ids[0])?)
            }
            // Additional matrix/vector ops
            PrimOp::OuterProduct => {
                if arg_ids.len() != 2 {
                    bail_spirv!("OuterProduct requires 2 args");
                }
                Ok(self.constructor.builder.outer_product(result_ty, None, arg_ids[0], arg_ids[1])?)
            }
            PrimOp::VectorTimesScalar => {
                if arg_ids.len() != 2 {
                    bail_spirv!("VectorTimesScalar requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .vector_times_scalar(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::MatrixTimesScalar => {
                if arg_ids.len() != 2 {
                    bail_spirv!("MatrixTimesScalar requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_scalar(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::DPdx => {
                if arg_ids.len() != 1 {
                    bail_spirv!("DPdx requires 1 arg");
                }
                Ok(self.constructor.builder.d_pdx(result_ty, None, arg_ids[0])?)
            }
            PrimOp::DPdy => {
                if arg_ids.len() != 1 {
                    bail_spirv!("DPdy requires 1 arg");
                }
                Ok(self.constructor.builder.d_pdy(result_ty, None, arg_ids[0])?)
            }
            PrimOp::Fwidth => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Fwidth requires 1 arg");
                }
                Ok(self.constructor.builder.fwidth(result_ty, None, arg_ids[0])?)
            }
        }
    }
}
