//! Direct TLC to EGraph conversion.
//!
//! Converts a TLC program directly to the acyclic e-graph representation,
//! bypassing the sequential SSA construction of `to_ssa`. Pure operations
//! are hash-consed (giving GVN for free), and the result is elaborated
//! back to `FuncBody` via demand-driven scheduling (giving DCE for free).

use std::collections::{HashMap, HashSet};

use crate::ast::TypeName;
use crate::ssa::types::{ControlHeader, EffectToken, FuncBody, Function, InstKind, Program, ValueRef};
use crate::ssa::types::{Soac, ViewSource};
use crate::tlc::{
    ArrayExpr, Def as TlcDef, DefMeta, Lambda, LoopKind, Program as TlcProgram, SoacOp, Term, TermKind,
};
use crate::types::TypeExt;
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use smallvec::{SmallVec, smallvec};
use wyn_ssa::BlockId;

use super::domtree::{DomTree, SkeletonCfgView};
use super::elaborate;
use super::types::*;

// ============================================================================
// Error type
// ============================================================================

#[derive(Debug)]
pub enum ConvertError {
    /// Error during EGraph construction.
    GraphError(String),
    /// Unsupported TLC construct (todo).
    Unsupported(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::GraphError(msg) => write!(f, "EGraph conversion error: {}", msg),
            ConvertError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for ConvertError {}

// ============================================================================
// Public entry point
// ============================================================================

/// Convert a TLC program directly to SSA via EGraph elaboration.
pub fn convert_program(program: &TlcProgram) -> Result<Program, ConvertError> {
    let top_level: HashMap<SymbolId, &TlcDef> = program.defs.iter().map(|d| (d.name, d)).collect();
    let symbols = &program.symbols;

    let constants_by_name: HashMap<String, SymbolId> = program
        .defs
        .iter()
        .filter(|d| d.arity == 0 && matches!(&d.meta, DefMeta::Function))
        .filter_map(|d| symbols.get(d.name).map(|n| (n.clone(), d.name)))
        .collect();

    // Phase 1: detect pure constants (same logic as to_ssa).
    let mut pure_constant_names: HashSet<String> = HashSet::new();
    let mut constants = Vec::new();

    for def in &program.defs {
        if def.arity != 0 || !matches!(&def.meta, DefMeta::Function) {
            continue;
        }
        if matches!(&def.body.kind, TermKind::Extern(_)) {
            continue;
        }
        let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();

        // Build a mini EGraph for the constant body and check if it's purely constant.
        let mut converter = Converter::new(
            &top_level,
            &constants_by_name,
            symbols,
            pure_constant_names.clone(),
        );
        if let Ok(result_nid) = converter.convert_term(&def.body) {
            converter.set_return(Some(result_nid));
            if let Some(body) = converter.elaborate_to_funcbody(&[], def.body.ty.clone()) {
                if is_purely_constant_body(&body) {
                    pure_constant_names.insert(def_name.clone());
                    constants.push(crate::ssa::types::Constant {
                        name: def_name,
                        body,
                        result_ty: def.body.ty.clone(),
                    });
                    continue;
                }
            }
        }
    }

    // Phase 2: convert functions and entry points.
    let mut functions = Vec::new();
    let mut entry_points = Vec::new();

    for def in &program.defs {
        match &def.meta {
            DefMeta::Function => {
                let def_name = symbols.get(def.name).expect("BUG: symbol not in table");
                if pure_constant_names.contains(def_name) {
                    continue;
                }
                let func =
                    convert_function(def, &top_level, &constants_by_name, symbols, &pure_constant_names)?;
                functions.push(func);
            }
            DefMeta::EntryPoint(entry) => {
                let ep = convert_entry_point(
                    def,
                    entry,
                    &top_level,
                    &constants_by_name,
                    symbols,
                    &pure_constant_names,
                )?;
                entry_points.push(ep);
            }
        }
    }

    Ok(Program {
        functions,
        entry_points,
        constants,
        uniforms: program.uniforms.clone(),
        storage: program.storage.clone(),
    })
}

// ============================================================================
// Function conversion
// ============================================================================

fn convert_function(
    def: &TlcDef,
    top_level: &HashMap<SymbolId, &TlcDef>,
    constants_by_name: &HashMap<String, SymbolId>,
    symbols: &SymbolTable,
    pure_constants: &HashSet<String>,
) -> Result<Function, ConvertError> {
    let def_name = symbols.get(def.name).expect("BUG").clone();

    // Extern functions
    if let TermKind::Extern(linkage_name) = &def.body.kind {
        let (param_types, ret_type) = extract_function_signature(&def.ty);
        let params: Vec<(Type<TypeName>, String)> =
            param_types.into_iter().enumerate().map(|(i, ty)| (ty, format!("arg{}", i))).collect();
        let mut builder = crate::ssa::builder::FuncBuilder::new(params, ret_type);
        builder
            .terminate(crate::ssa::types::Terminator::Unreachable)
            .map_err(|e| ConvertError::GraphError(e.to_string()))?;
        let body = builder.finish().map_err(|e| ConvertError::GraphError(e.to_string()))?;
        return Ok(Function {
            name: def_name,
            body,
            span: def.body.span,
            linkage_name: Some(linkage_name.clone()),
        });
    }

    // Regular functions: extract lambda params.
    let (inner_body, params) = extract_lambda_params(&def.body);
    let ret_type = inner_body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> = params
        .iter()
        .map(|(sym, ty)| {
            let name = symbols.get(*sym).unwrap_or(&format!("arg")).clone();
            (ty.clone(), name)
        })
        .collect();

    let mut converter = Converter::new(top_level, constants_by_name, symbols, pure_constants.clone());

    // Register function params.
    for (i, (sym, ty)) in params.iter().enumerate() {
        let nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, nid);
    }

    // Convert body.
    let result = converter.convert_term(inner_body)?;
    converter.set_return(Some(result));

    let body = converter
        .elaborate_to_funcbody(&param_info, ret_type)
        .ok_or_else(|| ConvertError::GraphError("elaboration failed".into()))?;

    Ok(Function {
        name: def_name,
        body,
        span: def.body.span,
        linkage_name: None,
    })
}

/// Entry point conversion delegates to `to_ssa` for now — the GPU I/O logic
/// (storage views, output ptrs, MapInto rewriting) is complex and identical
/// in both paths. The resulting FuncBody can be optionally canonicalized
/// through the EGraph afterward.
fn convert_entry_point(
    def: &TlcDef,
    entry: &crate::ast::EntryDecl,
    top_level: &HashMap<SymbolId, &TlcDef>,
    constants_by_name: &HashMap<String, SymbolId>,
    symbols: &SymbolTable,
    pure_constants: &HashSet<String>,
) -> Result<crate::ssa::types::EntryPoint, ConvertError> {
    crate::egir::entry_points::convert_entry_point_pub(
        def,
        entry,
        top_level,
        constants_by_name,
        symbols,
        pure_constants,
    )
    .map_err(|e| ConvertError::GraphError(format!("entry point via to_ssa: {}", e)))
}

// ============================================================================
// Converter
// ============================================================================

struct Converter<'a> {
    /// The e-graph being built.
    graph: EGraph,
    /// Current skeleton block for side effects and terminators.
    current_block: BlockId,
    /// TLC variable → EGraph node mapping.
    locals: HashMap<SymbolId, NodeId>,
    /// Top-level definitions.
    top_level: &'a HashMap<SymbolId, &'a TlcDef>,
    /// Arity-0 defs indexed by name.
    constants_by_name: &'a HashMap<String, SymbolId>,
    /// Symbol table.
    symbols: &'a SymbolTable,
    /// Cache for inlined constant bodies.
    inlined_constants: HashMap<String, NodeId>,
    /// Names of hoisted pure constants.
    pure_constants: HashSet<String>,
    /// Control headers for structured control flow (SPIR-V).
    control_headers: HashMap<BlockId, ControlHeader>,
    /// Effect token counter.
    next_effect: u32,
}

impl<'a> Converter<'a> {
    fn new(
        top_level: &'a HashMap<SymbolId, &'a TlcDef>,
        constants_by_name: &'a HashMap<String, SymbolId>,
        symbols: &'a SymbolTable,
        pure_constants: HashSet<String>,
    ) -> Self {
        let graph = EGraph::new();
        let entry = graph.skeleton.entry;
        Converter {
            graph,
            current_block: entry,
            locals: HashMap::new(),
            top_level,
            constants_by_name,
            symbols,
            inlined_constants: HashMap::new(),
            pure_constants,
            control_headers: HashMap::new(),
            next_effect: 1,
        }
    }

    fn alloc_effect(&mut self) -> EffectToken {
        let t = EffectToken(self.next_effect);
        self.next_effect += 1;
        t
    }

    /// Set the return terminator on the current block.
    fn set_return(&mut self, result: Option<NodeId>) {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(result);
    }

    /// Elaborate the built EGraph into a FuncBody.
    fn elaborate_to_funcbody(
        self,
        params: &[(Type<TypeName>, String)],
        return_ty: Type<TypeName>,
    ) -> Option<FuncBody> {
        let skel_domtree = DomTree::build(&SkeletonCfgView {
            skeleton: &self.graph.skeleton,
        });

        // Identity map: skeleton blocks map to themselves.
        let identity_map: HashMap<crate::ssa::types::BlockId, BlockId> = self
            .graph
            .skeleton
            .blocks
            .keys()
            .map(|b| {
                // Convert wyn_ssa::BlockId to ssa::types::BlockId.
                // They're the same type (both re-exported from wyn_ssa).
                (b, b)
            })
            .collect();

        Some(elaborate::elaborate(
            &self.graph,
            &skel_domtree,
            params,
            return_ty,
            &self.control_headers,
            &identity_map,
        ))
    }

    // ========================================================================
    // Term conversion
    // ========================================================================

    fn convert_term(&mut self, term: &Term) -> Result<NodeId, ConvertError> {
        let ty = term.ty.clone();

        match &term.kind {
            // --- Literals ---
            TermKind::IntLit(s) => {
                let op = if matches!(&ty, Type::Constructed(TypeName::UInt(_), _)) {
                    PureOp::Uint(s.clone())
                } else {
                    PureOp::Int(s.clone())
                };
                Ok(self.graph.intern_pure(op, smallvec![], ty))
            }
            TermKind::FloatLit(f) => {
                Ok(self.graph.intern_pure(PureOp::Float(f.to_string()), smallvec![], ty))
            }
            TermKind::BoolLit(b) => Ok(self.graph.intern_pure(PureOp::Bool(*b), smallvec![], ty)),
            TermKind::StringLit(s) => {
                Ok(self.graph.intern_pure(PureOp::StringLit(s.clone()), smallvec![], ty))
            }

            // --- Variables ---
            TermKind::Var(sym) => self.convert_var(*sym, ty),

            // --- Let bindings (scope only, no instruction) ---
            TermKind::Let {
                name,
                name_ty: _,
                rhs,
                body,
            } => {
                let rhs_nid = self.convert_term(rhs)?;
                self.locals.insert(*name, rhs_nid);
                let result = self.convert_term(body)?;
                self.locals.remove(name);
                Ok(result)
            }

            // --- Extern ---
            TermKind::Extern(name) => {
                Ok(self.graph.intern_pure(PureOp::Extern(name.clone()), smallvec![], ty))
            }

            // --- Force (pass-through) ---
            TermKind::Force(inner) => self.convert_term(inner),

            // --- If/else (Step 3) ---
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => self.convert_if(cond, then_branch, else_branch, ty),

            // --- Application (Step 2 + 4) ---
            TermKind::App { func, args } => self.convert_app(func, args, ty),

            // --- Loops ---
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => self.convert_loop(*loop_var, loop_var_ty, init, init_bindings, kind, body, ty),

            // --- SOACs ---
            TermKind::Soac(soac) => self.convert_soac(soac, ty),

            // --- Array expressions ---
            TermKind::ArrayExpr(ae) => self.convert_array_expr(ae, ty),

            // --- Should not appear after defunctionalization ---
            TermKind::Lambda(_) => {
                panic!("ICE: bare Lambda in to_egir (should be lifted)")
            }
            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                panic!("ICE: bare operator in to_egir (should be inside App)")
            }
            TermKind::Pack { .. } | TermKind::Unpack { .. } => Err(ConvertError::Unsupported(
                "existentials not yet implemented".into(),
            )),
        }
    }

    // ========================================================================
    // Variable resolution
    // ========================================================================

    fn convert_var(&mut self, sym: SymbolId, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        // Local binding
        if let Some(&nid) = self.locals.get(&sym) {
            return Ok(nid);
        }

        let name = self.symbols.get(sym).expect("BUG: symbol not in table").clone();

        // Cached constant
        if let Some(&nid) = self.inlined_constants.get(&name) {
            return Ok(nid);
        }

        // Hoisted pure constant → Global reference
        if self.pure_constants.contains(&name) {
            return Ok(self.graph.intern_pure(PureOp::Global(name), smallvec![], ty));
        }

        // Arity-0 constant def → inline its body
        let const_def = self
            .top_level
            .get(&sym)
            .filter(|d| d.arity == 0)
            .or_else(|| self.constants_by_name.get(&name).and_then(|def_sym| self.top_level.get(def_sym)))
            .copied();

        if let Some(def) = const_def {
            let body = def.body.clone();
            let nid = self.convert_term(&body)?;
            self.inlined_constants.insert(name, nid);
            return Ok(nid);
        }

        // Function reference → Global
        Ok(self.graph.intern_pure(PureOp::Global(name), smallvec![], ty))
    }

    // ========================================================================
    // Application
    // ========================================================================

    fn convert_app(
        &mut self,
        func: &Term,
        args: &[Term],
        ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        match &func.kind {
            TermKind::BinOp(op) => {
                let lhs = self.convert_term(&args[0])?;
                let rhs = self.convert_term(&args[1])?;
                Ok(self.graph.intern_pure(PureOp::BinOp(op.op.clone()), smallvec![lhs, rhs], ty))
            }
            TermKind::UnOp(op) => {
                let operand = self.convert_term(&args[0])?;
                Ok(self.graph.intern_pure(PureOp::UnaryOp(op.op.clone()), smallvec![operand], ty))
            }
            TermKind::Var(sym) => {
                let name = self.symbols.get(*sym).expect("BUG").clone();
                self.convert_named_app(&name, *sym, args, ty)
            }
            _ => {
                // General application: convert func, then call
                let _func_nid = self.convert_term(func)?;
                let _arg_nids: Vec<NodeId> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                // TODO: emit Call side effect
                Err(ConvertError::Unsupported("general application".into()))
            }
        }
    }

    fn convert_named_app(
        &mut self,
        name: &str,
        sym: SymbolId,
        args: &[Term],
        ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        // Intrinsic patterns
        match name {
            "_w_tuple" => {
                let operands: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::Tuple(n), operands, ty))
            }
            "_w_vec_lit" => {
                let operands: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::Vector(n), operands, ty))
            }
            "_w_tuple_proj" => {
                if args.len() != 2 {
                    return Err(ConvertError::GraphError("_w_tuple_proj expects 2 args".into()));
                }
                let base = self.convert_term(&args[0])?;
                let index = match &args[1].kind {
                    TermKind::IntLit(s) => s.parse::<u32>().unwrap_or(0),
                    _ => {
                        return Err(ConvertError::GraphError(
                            "_w_tuple_proj index must be literal".into(),
                        ));
                    }
                };
                Ok(self.graph.intern_pure(PureOp::Project { index }, smallvec![base], ty))
            }
            "_w_index" => {
                if args.len() != 2 {
                    return Err(ConvertError::GraphError("_w_index expects 2 args".into()));
                }
                let base = self.convert_term(&args[0])?;
                let index = self.convert_term(&args[1])?;
                Ok(self.graph.intern_pure(PureOp::Index, smallvec![base, index], ty))
            }
            "_w_array_lit" => {
                let operands: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::ArrayLit(n), operands, ty))
            }
            "_w_range" => {
                let start = self.convert_term(&args[0])?;
                let len = self.convert_term(&args[1])?;
                Ok(self.graph.intern_pure(
                    PureOp::ArrayRange { has_step: false },
                    smallvec![start, len],
                    ty,
                ))
            }
            name if name.starts_with("_w_intrinsic_") => {
                // Intrinsic call → side effect
                let arg_nids: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let arg_vrefs: Vec<ValueRef> = (0..arg_nids.len())
                    .map(|_| ValueRef::Ssa(crate::ssa::types::ValueId::default()))
                    .collect();
                let result_nid = self.graph.alloc_side_effect_result(ty.clone());
                let effect_in = EffectToken(0);
                let effect_out = self.alloc_effect();
                self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                    kind: InstKind::Intrinsic {
                        name: name.to_string(),
                        args: arg_vrefs,
                    },
                    operand_nodes: arg_nids,
                    result: Some(result_nid),
                    effects: Some((effect_in, effect_out)),
                });
                Ok(result_nid)
            }
            _ => {
                // Function call
                if let Some(def) = self.top_level.get(&sym) {
                    if def.arity == args.len() {
                        let arg_nids: SmallVec<[NodeId; 4]> =
                            args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                        let arg_vrefs: Vec<ValueRef> = (0..arg_nids.len())
                            .map(|_| ValueRef::Ssa(crate::ssa::types::ValueId::default()))
                            .collect();
                        let result_nid = self.graph.alloc_side_effect_result(ty.clone());
                        let effect_in = EffectToken(0);
                        let effect_out = self.alloc_effect();
                        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                            kind: InstKind::Call {
                                func: name.to_string(),
                                args: arg_vrefs,
                            },
                            operand_nodes: arg_nids,
                            result: Some(result_nid),
                            effects: Some((effect_in, effect_out)),
                        });
                        return Ok(result_nid);
                    }
                }
                // Arity-0 constant applied to args? Inline the body.
                let var_nid = self.convert_var(sym, ty.clone())?;
                // If we got a Global, emit a call
                if matches!(self.graph.nodes[var_nid], ENode::Pure { ref op, .. } if matches!(op, PureOp::Global(_)))
                {
                    let arg_nids: SmallVec<[NodeId; 4]> =
                        args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                    let arg_vrefs: Vec<ValueRef> = (0..arg_nids.len())
                        .map(|_| ValueRef::Ssa(crate::ssa::types::ValueId::default()))
                        .collect();
                    let result_nid = self.graph.alloc_side_effect_result(ty);
                    let effect_in = EffectToken(0);
                    let effect_out = self.alloc_effect();
                    self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                        kind: InstKind::Call {
                            func: name.to_string(),
                            args: arg_vrefs,
                        },
                        operand_nodes: arg_nids,
                        result: Some(result_nid),
                        effects: Some((effect_in, effect_out)),
                    });
                    Ok(result_nid)
                } else {
                    Err(ConvertError::Unsupported(format!(
                        "application of non-function: {}",
                        name
                    )))
                }
            }
        }
    }

    // ========================================================================
    // If/else
    // ========================================================================

    fn convert_if(
        &mut self,
        cond: &Term,
        then_branch: &Term,
        else_branch: &Term,
        ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let cond_nid = self.convert_term(cond)?;

        let then_block = self.graph.skeleton.create_block();
        let else_block = self.graph.skeleton.create_block();
        let merge_block = self.graph.skeleton.create_block();

        let result_nid = self.graph.add_block_param(merge_block, 0, ty.clone());
        self.graph.skeleton.blocks[merge_block].params.push(result_nid);

        // Selection header for SPIR-V structured control flow.
        self.control_headers.insert(
            self.current_block,
            ControlHeader::Selection { merge: merge_block },
        );

        // Terminate current block with CondBranch.
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: then_block,
            then_args: vec![],
            else_target: else_block,
            else_args: vec![],
        };

        // Then branch.
        self.current_block = then_block;
        let then_result = self.convert_term(then_branch)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: merge_block,
            args: vec![then_result],
        };

        // Else branch.
        self.current_block = else_block;
        let else_result = self.convert_term(else_branch)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: merge_block,
            args: vec![else_result],
        };

        // Continue from merge.
        self.current_block = merge_block;
        Ok(result_nid)
    }

    // ========================================================================
    // Loops
    // ========================================================================

    fn convert_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        kind: &LoopKind,
        body: &Term,
        _result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        match kind {
            LoopKind::While { cond } => {
                self.convert_while_loop(loop_var, loop_var_ty, init, init_bindings, cond, body)
            }
            LoopKind::ForRange { var, var_ty, bound } => self.convert_for_range_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                *var,
                var_ty,
                bound,
                body,
            ),
            LoopKind::For { var, var_ty, iter } => self.convert_for_in_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                *var,
                var_ty,
                iter,
                body,
            ),
        }
    }

    fn convert_while_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        cond: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        let acc_ty = loop_var_ty.clone();

        // Create blocks: header, body, exit
        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();

        // Header has acc param; exit has result param
        let acc_nid = self.graph.add_block_param(header, 0, acc_ty.clone());
        self.graph.skeleton.blocks[header].params.push(acc_nid);
        let result_nid = self.graph.add_block_param(exit, 0, acc_ty.clone());
        self.graph.skeleton.blocks[exit].params.push(result_nid);

        // Loop header for SPIR-V
        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

        // Init → header
        let init_nid = self.convert_term(init)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![init_nid],
        };

        // Header: bind loop_var, process init_bindings, check cond
        self.current_block = header;
        self.locals.insert(loop_var, acc_nid);
        for (sym, _ty, expr) in init_bindings {
            let val = self.convert_term(expr)?;
            self.locals.insert(*sym, val);
        }
        let cond_nid = self.convert_term(cond)?;
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![acc_nid],
        };

        // Body: convert body, branch back to header
        self.current_block = body_block;
        let new_acc = self.convert_term(body)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![new_acc],
        };

        // Exit
        self.current_block = exit;
        self.locals.remove(&loop_var);
        for (sym, _, _) in init_bindings {
            self.locals.remove(sym);
        }
        Ok(result_nid)
    }

    fn convert_for_range_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        index_var: SymbolId,
        _index_var_ty: &Type<TypeName>,
        bound: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        // Create blocks
        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();

        // Header has (acc, index) params; exit has result param
        let acc_nid = self.graph.add_block_param(header, 0, acc_ty.clone());
        let idx_nid = self.graph.add_block_param(header, 1, i32_ty.clone());
        self.graph.skeleton.blocks[header].params.push(acc_nid);
        self.graph.skeleton.blocks[header].params.push(idx_nid);
        let result_nid = self.graph.add_block_param(exit, 0, acc_ty.clone());
        self.graph.skeleton.blocks[exit].params.push(result_nid);

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

        // Init → header with (init, 0)
        let init_nid = self.convert_term(init)?;
        let bound_nid = self.convert_term(bound)?;
        let zero = self.graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![init_nid, zero],
        };

        // Header: bind vars, check i < bound
        self.current_block = header;
        self.locals.insert(loop_var, acc_nid);
        self.locals.insert(index_var, idx_nid);
        for (sym, _ty, expr) in init_bindings {
            let val = self.convert_term(expr)?;
            self.locals.insert(*sym, val);
        }
        let cond_nid =
            self.graph.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, bound_nid], bool_ty);
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![acc_nid],
        };

        // Body: convert body, increment index, branch back
        self.current_block = body_block;
        let new_acc = self.convert_term(body)?;
        let one = self.graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.graph.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        };

        // Exit
        self.current_block = exit;
        self.locals.remove(&loop_var);
        self.locals.remove(&index_var);
        for (sym, _, _) in init_bindings {
            self.locals.remove(sym);
        }
        Ok(result_nid)
    }

    fn convert_for_in_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        elem_var: SymbolId,
        elem_ty: &Type<TypeName>,
        iter: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        // For-in is like for-range but indexes into the iterator.
        // TODO: SoA-aware soa_length / soa_index
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();

        let acc_nid = self.graph.add_block_param(header, 0, acc_ty.clone());
        let idx_nid = self.graph.add_block_param(header, 1, i32_ty.clone());
        self.graph.skeleton.blocks[header].params.push(acc_nid);
        self.graph.skeleton.blocks[header].params.push(idx_nid);
        let result_nid = self.graph.add_block_param(exit, 0, acc_ty.clone());
        self.graph.skeleton.blocks[exit].params.push(result_nid);

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

        // Init
        let init_nid = self.convert_term(init)?;
        let iter_nid = self.convert_term(iter)?;

        // Length intrinsic
        let len_nid = self.graph.intern_pure(
            PureOp::UnaryOp("_w_intrinsic_length".into()),
            smallvec![iter_nid],
            i32_ty.clone(),
        );
        let zero = self.graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![init_nid, zero],
        };

        // Header
        self.current_block = header;
        self.locals.insert(loop_var, acc_nid);
        for (sym, _ty, expr) in init_bindings {
            let val = self.convert_term(expr)?;
            self.locals.insert(*sym, val);
        }
        let cond_nid =
            self.graph.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, len_nid], bool_ty);
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![acc_nid],
        };

        // Body: index into iterator, bind elem_var
        self.current_block = body_block;
        let elem_nid = self.graph.intern_pure(PureOp::Index, smallvec![iter_nid, idx_nid], elem_ty.clone());
        self.locals.insert(elem_var, elem_nid);

        let new_acc = self.convert_term(body)?;
        let one = self.graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.graph.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        };

        // Exit
        self.current_block = exit;
        self.locals.remove(&loop_var);
        self.locals.remove(&elem_var);
        for (sym, _, _) in init_bindings {
            self.locals.remove(sym);
        }
        Ok(result_nid)
    }

    // ========================================================================
    // SOACs
    // ========================================================================

    fn convert_soac(&mut self, soac: &SoacOp, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        match soac {
            SoacOp::Map { lam, inputs } => self.convert_soac_map(lam, inputs, ty),
            SoacOp::Reduce { op, ne, input, .. } => self.convert_soac_reduce(op, ne, input, ty),
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => self.convert_soac_redomap(op, reduce_op, ne, inputs, ty),
            SoacOp::Scan { op, ne, input } => self.convert_soac_scan(op, ne, input, ty),
            SoacOp::Filter { pred, input } => self.convert_soac_filter(pred, input, ty),
            SoacOp::Scatter { .. } => Err(ConvertError::Unsupported("SOAC scatter".into())),
            SoacOp::ReduceByIndex { .. } => Err(ConvertError::Unsupported("SOAC reduce_by_index".into())),
        }
    }

    fn lambda_fn_name(&self, lam: &Lambda) -> Result<String, ConvertError> {
        match &lam.body.kind {
            TermKind::Var(sym) => Ok(self.symbols.get(*sym).expect("BUG: symbol not in table").clone()),
            _ => Err(ConvertError::GraphError(
                "SOAC lambda body should be a function reference post-defunc".into(),
            )),
        }
    }

    /// Emit a SOAC as a side effect in the skeleton. Returns the result NodeId.
    fn emit_soac(&mut self, kind: InstKind, operands: SmallVec<[NodeId; 4]>, ty: Type<TypeName>) -> NodeId {
        let result_nid = self.graph.alloc_side_effect_result(ty);
        let effect_in = EffectToken(0);
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind,
            operand_nodes: operands,
            result: Some(result_nid),
            effects: Some((effect_in, effect_out)),
        });
        result_nid
    }

    fn convert_soac_map(
        &mut self,
        lam: &Lambda,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let f_name = self.lambda_fn_name(lam)?;
        let capture_nids: Vec<NodeId> =
            lam.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_type(ae)).collect();
        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_elem_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_elem_type(ae)).collect();
        let output_elem_ty = if result_ty.is_array() {
            result_ty.elem_type().expect("Array has elem").clone()
        } else if !input_elem_types.is_empty() {
            input_elem_types[0].clone()
        } else {
            return Err(ConvertError::GraphError(
                "map: cannot determine output elem type".into(),
            ));
        };

        let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
        operands.extend_from_slice(&input_nids);
        operands.extend_from_slice(&capture_nids);

        Ok(self.emit_soac(
            InstKind::Soac(Soac::Map {
                func: f_name,
                inputs: vec![Default::default(); input_nids.len()],
                captures: vec![Default::default(); capture_nids.len()],
                input_array_types: input_arr_types,
                input_elem_types,
                output_elem_type: output_elem_ty,
            }),
            operands,
            result_ty,
        ))
    }

    fn convert_soac_reduce(
        &mut self,
        op: &Lambda,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;
        let capture_nids: Vec<NodeId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;
        let init_nid = self.convert_term(ne)?;

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid, init_nid];
        operands.extend(capture_nids.iter().copied());

        Ok(self.emit_soac(
            InstKind::Soac(Soac::Reduce {
                func: op_name,
                input: Default::default(),
                init: Default::default(),
                captures: vec![Default::default(); capture_nids.len()],
                input_array_type: arr_ty,
                input_elem_type: elem_ty,
            }),
            operands,
            result_ty,
        ))
    }

    fn convert_soac_redomap(
        &mut self,
        op: &Lambda,
        reduce_op: &Lambda,
        ne: &Term,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;
        let reduce_func_name = self.lambda_fn_name(reduce_op)?;
        let capture_nids: Vec<NodeId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let reduce_capture_nids: Vec<NodeId> =
            reduce_op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_type(ae)).collect();
        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_elem_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_elem_type(ae)).collect();
        let init_nid = self.convert_term(ne)?;

        let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
        operands.extend(input_nids.iter().copied());
        operands.push(init_nid);
        operands.extend(capture_nids.iter().copied());
        operands.extend(reduce_capture_nids.iter().copied());

        Ok(self.emit_soac(
            InstKind::Soac(Soac::Redomap {
                func: op_name,
                reduce_func: reduce_func_name,
                inputs: vec![Default::default(); input_nids.len()],
                init: Default::default(),
                captures: vec![Default::default(); capture_nids.len()],
                reduce_captures: vec![Default::default(); reduce_capture_nids.len()],
                input_array_types: input_arr_types,
                input_elem_types,
            }),
            operands,
            result_ty,
        ))
    }

    fn convert_soac_scan(
        &mut self,
        op: &Lambda,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;
        let capture_nids: Vec<NodeId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;
        let init_nid = self.convert_term(ne)?;

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid, init_nid];
        operands.extend(capture_nids.iter().copied());

        Ok(self.emit_soac(
            InstKind::Soac(Soac::Scan {
                func: op_name,
                input: Default::default(),
                init: Default::default(),
                captures: vec![Default::default(); capture_nids.len()],
                input_array_type: arr_ty,
                input_elem_type: elem_ty,
            }),
            operands,
            result_ty,
        ))
    }

    fn convert_soac_filter(
        &mut self,
        pred: &Lambda,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let pred_name = self.lambda_fn_name(pred)?;
        let capture_nids: Vec<NodeId> =
            pred.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let arr_nid = self.convert_array_expr_value(input)?;
        let pred_ref = self.graph.intern_pure(
            PureOp::Global(pred_name),
            smallvec![],
            Type::Constructed(TypeName::Unit, vec![]),
        );

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![pred_ref, arr_nid];
        operands.extend(capture_nids.iter().copied());

        let dummy_vrefs: Vec<ValueRef> =
            (0..operands.len()).map(|_| ValueRef::Ssa(Default::default())).collect();
        Ok(self.emit_soac(
            InstKind::Intrinsic {
                name: "_w_intrinsic_filter".into(),
                args: dummy_vrefs,
            },
            operands,
            result_ty,
        ))
    }

    // ========================================================================
    // ArrayExpr
    // ========================================================================

    fn convert_array_expr(&mut self, ae: &ArrayExpr, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        match ae {
            ArrayExpr::Ref(term) => self.convert_term(term),
            ArrayExpr::Zip(_) => panic!("ArrayExpr::Zip should have been eliminated by soa::normalize"),
            ArrayExpr::Soac(op) => self.convert_soac(op, ty),
            ArrayExpr::Generate { .. } => Err(ConvertError::Unsupported("ArrayExpr::Generate".into())),
            ArrayExpr::Literal(terms) => {
                let operands: SmallVec<[NodeId; 4]> =
                    terms.iter().map(|t| self.convert_term(t)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::ArrayLit(n), operands, ty))
            }
            ArrayExpr::Range { start, len } => {
                let start_nid = self.convert_term(start)?;
                let len_nid = self.convert_term(len)?;
                Ok(self.graph.intern_pure(
                    PureOp::ArrayRange { has_step: false },
                    smallvec![start_nid, len_nid],
                    ty,
                ))
            }
            ArrayExpr::StorageBuffer {
                set,
                binding,
                offset,
                len,
                elem_ty,
            } => {
                let offset_nid = self.convert_term(offset)?;
                let len_nid = self.convert_term(len)?;
                let array_ty = Type::Constructed(
                    TypeName::Array,
                    vec![
                        elem_ty.clone(),
                        Type::Constructed(TypeName::SizePlaceholder, vec![]),
                        Type::Constructed(TypeName::ArrayVariantView, vec![]),
                    ],
                );
                let result_nid = self.graph.alloc_side_effect_result(array_ty);
                let effect_in = EffectToken(0);
                let effect_out = self.alloc_effect();
                self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                    kind: InstKind::StorageView {
                        source: ViewSource::Storage {
                            set: *set,
                            binding: *binding,
                        },
                        offset: ValueRef::Ssa(Default::default()),
                        len: ValueRef::Ssa(Default::default()),
                    },
                    operand_nodes: smallvec![offset_nid, len_nid],
                    result: Some(result_nid),
                    effects: Some((effect_in, effect_out)),
                });
                Ok(result_nid)
            }
        }
    }

    fn convert_array_expr_value(&mut self, ae: &ArrayExpr) -> Result<NodeId, ConvertError> {
        match ae {
            ArrayExpr::Ref(term) => self.convert_term(term),
            _ => {
                let ty = self.array_expr_type(ae);
                self.convert_array_expr(ae, ty)
            }
        }
    }

    fn array_expr_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            ArrayExpr::Ref(t) => t.ty.clone(),
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Generate { elem_ty, .. } => elem_ty.clone(),
            ArrayExpr::Literal(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Range { start, .. } => Type::Constructed(
                TypeName::Array,
                vec![
                    start.ty.clone(),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                ],
            ),
            ArrayExpr::StorageBuffer { elem_ty, .. } => Type::Constructed(
                TypeName::Array,
                vec![
                    elem_ty.clone(),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    Type::Constructed(TypeName::ArrayVariantView, vec![]),
                ],
            ),
        }
    }

    fn array_expr_elem_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            ArrayExpr::Ref(t) => match &t.ty {
                Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
                _ => t.ty.clone(),
            },
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Generate { elem_ty, .. } => elem_ty.clone(),
            ArrayExpr::Literal(terms) => {
                terms.first().map(|t| t.ty.clone()).unwrap_or(Type::Constructed(TypeName::Unit, vec![]))
            }
            ArrayExpr::Range { start, .. } => start.ty.clone(),
            ArrayExpr::StorageBuffer { elem_ty, .. } => elem_ty.clone(),
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Check whether a FuncBody contains only purely constant instructions.
fn is_purely_constant_body(body: &FuncBody) -> bool {
    body.inner.insts.values().all(|inst| {
        matches!(
            &inst.data,
            InstKind::Int(_)
                | InstKind::Float(_)
                | InstKind::Bool(_)
                | InstKind::Unit
                | InstKind::String(_)
                | InstKind::Tuple(_)
                | InstKind::Vector(_)
                | InstKind::Matrix(_)
                | InstKind::ArrayLit { .. }
                | InstKind::Global(_)
        )
    })
}

/// Extract parameter types and return type from an arrow type.
fn extract_function_signature(ty: &Type<TypeName>) -> (Vec<Type<TypeName>>, Type<TypeName>) {
    let mut params = Vec::new();
    let mut current = ty.clone();
    while let Type::Constructed(TypeName::Arrow, ref args) = current {
        if args.len() == 2 {
            params.push(args[0].clone());
            current = args[1].clone();
        } else {
            break;
        }
    }
    (params, current)
}

/// Walk through nested Lambdas to extract parameters and the inner body.
fn extract_lambda_params(term: &Term) -> (&Term, Vec<(SymbolId, Type<TypeName>)>) {
    let mut params = Vec::new();
    let mut current = term;
    while let TermKind::Lambda(lam) = &current.kind {
        params.extend(lam.params.iter().cloned());
        current = &lam.body;
    }
    (current, params)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile a source string through the full TLC pipeline, then convert via EGraph.
    fn compile_via_egir(src: &str) -> Program {
        let mut frontend = crate::cached_frontend();
        let parsed = crate::Compiler::parse(src, &mut frontend.node_counter).expect("Parsing failed");
        let alias_checked = parsed
            .desugar(&mut frontend.node_counter)
            .expect("Desugaring failed")
            .resolve(&mut frontend.module_manager)
            .expect("Name resolution failed")
            .fold_ast_constants()
            .type_check(&mut frontend.module_manager, &mut frontend.schemes)
            .expect("Type checking failed")
            .alias_check()
            .expect("Alias checking failed");

        let tlc = alias_checked
            .to_tlc(&frontend.schemes, &frontend.module_manager)
            .partial_eval()
            .normalize_soacs()
            .fuse_maps()
            .defunctionalize()
            .monomorphize()
            .buffer_specialize()
            .inline()
            .inline_small()
            .parallelize_soacs()
            .filter_reachable();

        convert_program(&tlc.tlc).expect("egir::from_tlc conversion failed")
    }

    use crate::ast::Span;
    use crate::tlc::TermIdSource;

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn mk_term(ty: Type<TypeName>, kind: TermKind) -> Term {
        Term {
            id: TermIdSource::new().next_id(),
            ty,
            span: Span::dummy(),
            kind,
        }
    }

    /// Build a minimal TLC def and convert it via EGraph.
    fn convert_simple_def(body: Term, params: Vec<(SymbolId, Type<TypeName>)>) -> FuncBody {
        let symbols = crate::SymbolTable::new();
        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let ret_ty = body.ty.clone();
        let param_info: Vec<(Type<TypeName>, String)> =
            params.iter().enumerate().map(|(i, (_, ty))| (ty.clone(), format!("p{}", i))).collect();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        for (i, (sym, ty)) in params.iter().enumerate() {
            let nid = converter.graph.add_func_param(i, ty.clone());
            converter.locals.insert(*sym, nid);
        }
        let result = converter.convert_term(&body).expect("conversion failed");
        converter.set_return(Some(result));
        converter.elaborate_to_funcbody(&param_info, ret_ty).expect("elaboration failed")
    }

    #[test]
    fn test_int_literal_roundtrip() {
        let body = mk_term(i32_ty(), TermKind::IntLit("42".into()));
        let func = convert_simple_def(body, vec![]);
        let entry = func.get_block(func.entry_block());
        // Should have one Int instruction.
        assert!(
            entry
                .insts
                .iter()
                .any(|&iid| { matches!(&func.get_inst(iid).data, InstKind::Int(s) if s == "42") })
        );
    }

    #[test]
    fn test_add_roundtrip() {
        let mut symbols = crate::SymbolTable::new();
        let a_sym = symbols.alloc("a".into());
        let b_sym = symbols.alloc("b".into());

        // Build: a + b
        let a_var = mk_term(i32_ty(), TermKind::Var(a_sym));
        let b_var = mk_term(i32_ty(), TermKind::Var(b_sym));
        let add_op = mk_term(
            i32_ty(), // simplified — real type would be arrow
            TermKind::BinOp(crate::ast::BinaryOp { op: "+".into() }),
        );
        let app = mk_term(
            i32_ty(),
            TermKind::App {
                func: Box::new(add_op),
                args: vec![a_var, b_var],
            },
        );

        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        let a_nid = converter.graph.add_func_param(0, i32_ty());
        converter.locals.insert(a_sym, a_nid);
        let b_nid = converter.graph.add_func_param(1, i32_ty());
        converter.locals.insert(b_sym, b_nid);

        let result = converter.convert_term(&app).expect("conversion failed");
        converter.set_return(Some(result));

        let params = vec![(i32_ty(), "a".into()), (i32_ty(), "b".into())];
        let func = converter.elaborate_to_funcbody(&params, i32_ty()).expect("elaboration failed");

        let entry = func.get_block(func.entry_block());
        // Should have a BinOp(+) instruction.
        assert!(
            entry.insts.iter().any(|&iid| {
                matches!(&func.get_inst(iid).data, InstKind::BinOp { op, .. } if op == "+")
            })
        );
    }

    #[test]
    fn test_gvn_via_let() {
        // let x = 42 in let y = 42 in (x, y)
        // GVN should deduplicate the two 42 constants into a single node.
        // (A `+` would be constant-folded to `84`, erasing the evidence.)
        use polytype::Type;
        let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);

        let lit42 = mk_term(i32_ty(), TermKind::IntLit("42".into()));
        let lit42b = mk_term(i32_ty(), TermKind::IntLit("42".into()));

        let mut symbols = crate::SymbolTable::new();
        let x_sym = symbols.alloc("x".into());
        let y_sym = symbols.alloc("y".into());
        let tuple_sym = symbols.alloc("_w_tuple".into());

        let tuple_op = mk_term(
            Type::Constructed(
                TypeName::Arrow,
                vec![
                    i32_ty(),
                    Type::Constructed(TypeName::Arrow, vec![i32_ty(), pair_ty.clone()]),
                ],
            ),
            TermKind::Var(tuple_sym),
        );
        let x_ref = mk_term(i32_ty(), TermKind::Var(x_sym));
        let y_ref = mk_term(i32_ty(), TermKind::Var(y_sym));
        let pair_app = mk_term(
            pair_ty.clone(),
            TermKind::App {
                func: Box::new(tuple_op),
                args: vec![x_ref, y_ref],
            },
        );

        let inner_let = mk_term(
            pair_ty.clone(),
            TermKind::Let {
                name: y_sym,
                name_ty: i32_ty(),
                rhs: Box::new(lit42b),
                body: Box::new(pair_app),
            },
        );
        let outer_let = mk_term(
            pair_ty.clone(),
            TermKind::Let {
                name: x_sym,
                name_ty: i32_ty(),
                rhs: Box::new(lit42),
                body: Box::new(inner_let),
            },
        );

        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        let result = converter.convert_term(&outer_let).expect("conversion failed");
        converter.set_return(Some(result));

        let func = converter.elaborate_to_funcbody(&[], pair_ty).expect("elaboration failed");

        let entry = func.get_block(func.entry_block());
        // GVN: should have only ONE Int("42") instruction, not two.
        let const_count = entry
            .insts
            .iter()
            .filter(|&&iid| matches!(&func.get_inst(iid).data, InstKind::Int(s) if s == "42"))
            .count();
        assert_eq!(
            const_count, 1,
            "GVN should deduplicate: found {} copies of 42",
            const_count
        );
    }

    #[test]
    fn test_if_else_roundtrip() {
        // if cond then 1 else 0
        let mut symbols = crate::SymbolTable::new();
        let c_sym = symbols.alloc("c".into());
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        let cond = mk_term(bool_ty.clone(), TermKind::Var(c_sym));
        let then_br = mk_term(i32_ty(), TermKind::IntLit("1".into()));
        let else_br = mk_term(i32_ty(), TermKind::IntLit("0".into()));
        let if_term = mk_term(
            i32_ty(),
            TermKind::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_br),
                else_branch: Box::new(else_br),
            },
        );

        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        let c_nid = converter.graph.add_func_param(0, bool_ty);
        converter.locals.insert(c_sym, c_nid);

        let result = converter.convert_term(&if_term).expect("conversion failed");
        converter.set_return(Some(result));

        let params = vec![(Type::Constructed(TypeName::Bool, vec![]), "c".into())];
        let func = converter.elaborate_to_funcbody(&params, i32_ty()).expect("elaboration failed");

        // Should have 4 blocks: entry, then, else, merge
        assert_eq!(func.inner.blocks.len(), 4, "if/else should produce 4 blocks");

        // Entry should end with CondBranch
        let entry = func.get_block(func.entry_block());
        assert!(
            matches!(&entry.term, wyn_ssa::Terminator::CondBranch { .. }),
            "Entry should end with CondBranch, got {:?}",
            entry.term
        );
    }

    // ====================================================================
    // Full pipeline integration tests
    // ====================================================================

    #[test]
    fn test_full_pipeline_simple() {
        let program = compile_via_egir(
            r#"
def add(a: i32, b: i32) i32 = a + b

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = add(1, 2) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
        );
        // 'add' may be inlined by TLC passes — just verify the program is valid
        assert!(!program.entry_points.is_empty(), "Should have entry points");
    }

    #[test]
    fn test_full_pipeline_if_else() {
        let program = compile_via_egir(
            r#"
def pick(c: bool, a: i32, b: i32) i32 = if c then a else b

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = pick(true, 1, 2) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
        );
        // 'pick' may be inlined — just verify compilation succeeds
        assert!(!program.entry_points.is_empty(), "Should have entry points");
    }

    #[test]
    fn test_full_pipeline_loop() {
        let program = compile_via_egir(
            r#"
def sum_to(n: i32) i32 =
    loop acc = 0 for i < n do acc + i

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = sum_to(10) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
        );
        // 'sum_to' may be inlined — just verify compilation succeeds
        assert!(!program.entry_points.is_empty(), "Should have entry points");
    }
}
