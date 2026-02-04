//! Direct TLC to SSA conversion.
//!
//! This module converts a lifted TLC program directly to SSA form,
//! bypassing the intermediate MIR representation.

use std::collections::HashMap;

use crate::ast::{self, NodeId, Span, TypeName};
use crate::mir::ssa::{FuncBody, InstKind, Terminator, ValueId};
use crate::mir::ssa_builder::FuncBuilder;
use polytype::Type;

use super::{Def as TlcDef, DefMeta, LoopKind as TlcLoopKind, Program as TlcProgram, Term, TermKind};

/// Extract parameter types and return type from a curried function type.
/// For `A -> B -> C`, returns `([A, B], C)`.
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

/// Check if a type is an unsized array (runtime-sized storage buffer).
fn is_unsized_array(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            matches!(&args[2], Type::Variable(_))
        }
        _ => false,
    }
}

/// Result of converting a TLC program to SSA.
#[derive(Debug, Clone)]
pub struct SsaProgram {
    /// Function definitions with their SSA bodies.
    pub functions: Vec<SsaFunction>,
    /// Entry point definitions.
    pub entry_points: Vec<SsaEntryPoint>,
    /// Uniform declarations.
    pub uniforms: Vec<ast::UniformDecl>,
    /// Storage buffer declarations.
    pub storage: Vec<ast::StorageDecl>,
}

/// A converted function.
#[derive(Debug, Clone)]
pub struct SsaFunction {
    pub name: String,
    pub body: FuncBody,
    pub span: Span,
    /// For extern functions, the linkage name.
    pub linkage_name: Option<String>,
}

/// A converted entry point.
#[derive(Debug, Clone)]
pub struct SsaEntryPoint {
    pub name: String,
    pub body: FuncBody,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub span: Span,
}

/// Execution model for entry points.
#[derive(Debug, Clone)]
pub enum ExecutionModel {
    Vertex,
    Fragment,
    Compute {
        local_size: (u32, u32, u32),
    },
}

/// Input to an entry point.
#[derive(Debug, Clone)]
pub struct EntryInput {
    pub name: String,
    pub ty: Type<TypeName>,
    pub decoration: Option<IoDecoration>,
    pub size_hint: Option<u32>,
    pub storage_binding: Option<(u32, u32)>,
}

/// Output from an entry point.
#[derive(Debug, Clone)]
pub struct EntryOutput {
    pub ty: Type<TypeName>,
    pub decoration: Option<IoDecoration>,
    /// For compute shaders with unsized array outputs: (set, binding).
    pub storage_binding: Option<(u32, u32)>,
}

/// I/O decoration for entry point parameters.
#[derive(Debug, Clone)]
pub enum IoDecoration {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
}

/// Error during TLC to SSA conversion.
#[derive(Debug, Clone)]
pub enum ConvertError {
    /// Unknown variable reference.
    UnknownVariable(String),
    /// Builder error.
    BuilderError(String),
    /// Invalid intrinsic call.
    InvalidIntrinsic(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::UnknownVariable(name) => write!(f, "Unknown variable: {}", name),
            ConvertError::BuilderError(msg) => write!(f, "Builder error: {}", msg),
            ConvertError::InvalidIntrinsic(msg) => write!(f, "Invalid intrinsic: {}", msg),
        }
    }
}

impl std::error::Error for ConvertError {}

/// Convert a TLC program to SSA.
pub fn convert_program(program: &TlcProgram) -> Result<SsaProgram, ConvertError> {
    let top_level: HashMap<String, &TlcDef> = program.defs.iter().map(|d| (d.name.clone(), d)).collect();

    let mut functions = Vec::new();
    let mut entry_points = Vec::new();

    for def in &program.defs {
        match &def.meta {
            DefMeta::Function => {
                let func = convert_function(def, &top_level)?;
                functions.push(func);
            }
            DefMeta::EntryPoint(entry) => {
                let ep = convert_entry_point(def, entry, &top_level)?;
                entry_points.push(ep);
            }
        }
    }

    Ok(SsaProgram {
        functions,
        entry_points,
        uniforms: program.uniforms.clone(),
        storage: program.storage.clone(),
    })
}

/// Convert a function definition to SSA.
fn convert_function(
    def: &TlcDef,
    top_level: &HashMap<String, &TlcDef>,
) -> Result<SsaFunction, ConvertError> {
    // Check if this is an extern function
    if let TermKind::Extern(linkage_name) = &def.body.kind {
        let (param_types, ret_type) = extract_function_signature(&def.ty);
        let params: Vec<(Type<TypeName>, String)> =
            param_types.into_iter().enumerate().map(|(i, ty)| (ty, format!("arg{}", i))).collect();

        let mut builder = FuncBuilder::new(params, ret_type);

        // Extern functions just return immediately - the actual call happens at call sites
        builder
            .terminate(Terminator::Unreachable)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let body = builder.finish().map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        return Ok(SsaFunction {
            name: def.name.clone(),
            body,
            span: def.body.span,
            linkage_name: Some(linkage_name.clone()),
        });
    }

    // Extract parameters from nested Lams
    let (params, inner_body) = extract_params(&def.body);

    // Build parameter list for FuncBuilder
    let param_list: Vec<(Type<TypeName>, String)> =
        params.iter().map(|(name, ty, _)| (ty.clone(), name.clone())).collect();

    let ret_type = inner_body.ty.clone();
    let builder = FuncBuilder::new(param_list, ret_type.clone());

    // Create converter with variable mappings
    let mut converter = Converter::new(builder, top_level);

    // Map parameters to their ValueIds
    for (i, (name, _, _)) in params.iter().enumerate() {
        let value = converter.builder.get_param(i);
        converter.locals.insert(name.clone(), value);
    }

    // Convert the body
    let result = converter.convert_term(inner_body)?;

    // Terminate with return
    if matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
        converter
            .builder
            .terminate(Terminator::ReturnUnit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    } else {
        converter
            .builder
            .terminate(Terminator::Return(result))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    }

    let body = converter.finish()?;

    Ok(SsaFunction {
        name: def.name.clone(),
        body,
        span: def.body.span,
        linkage_name: None,
    })
}

/// Convert an entry point definition to SSA.
fn convert_entry_point(
    def: &TlcDef,
    entry: &ast::EntryDecl,
    top_level: &HashMap<String, &TlcDef>,
) -> Result<SsaEntryPoint, ConvertError> {
    // Extract parameters from nested Lams
    let (params, inner_body) = extract_params(&def.body);

    let is_compute = matches!(entry.entry_type, ast::Attribute::Compute);

    // Build inputs with decorations
    let mut inputs = Vec::new();
    let mut binding_num = 0u32;

    for (i, (name, ty, _span)) in params.iter().enumerate() {
        let decoration = entry.params.get(i).and_then(|p| extract_io_decoration(p));
        let size_hint = entry.params.get(i).and_then(|p| extract_size_hint(p));

        let storage_binding = if is_compute && is_unsized_array(ty) {
            let binding = (0, binding_num);
            binding_num += 1;
            Some(binding)
        } else {
            None
        };

        inputs.push(EntryInput {
            name: name.clone(),
            ty: ty.clone(),
            decoration,
            size_hint,
            storage_binding,
        });
    }

    // Build parameter list for FuncBuilder
    let param_list: Vec<(Type<TypeName>, String)> =
        params.iter().map(|(name, ty, _)| (ty.clone(), name.clone())).collect();

    let ret_type = inner_body.ty.clone();
    let builder = FuncBuilder::new(param_list, ret_type.clone());

    // Create converter
    let mut converter = Converter::new(builder, top_level);

    // Map parameters
    for (i, (name, _, _)) in params.iter().enumerate() {
        let value = converter.builder.get_param(i);
        converter.locals.insert(name.clone(), value);
    }

    // Convert the body
    let result = converter.convert_term(inner_body)?;

    // Convert execution model (needed to decide how to handle outputs)
    let execution_model = match &entry.entry_type {
        ast::Attribute::Vertex => ExecutionModel::Vertex,
        ast::Attribute::Fragment => ExecutionModel::Fragment,
        ast::Attribute::Compute => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
        _ => panic!("Invalid entry type attribute: {:?}", entry.entry_type),
    };

    // Build outputs (pass binding_num so outputs continue from where inputs left off)
    let outputs = build_entry_outputs(entry, &inner_body.ty, is_compute, binding_num);

    // For vertex/fragment shaders with outputs, emit explicit stores to output variables
    let is_compute = matches!(execution_model, ExecutionModel::Compute { .. });
    let is_unit_return = matches!(ret_type, Type::Constructed(TypeName::Unit, _));

    if !is_compute && !is_unit_return && !outputs.is_empty() {
        // Get effect token for stores
        let mut effect = converter.builder.entry_effect();
        let span = inner_body.span;
        let node_id = NodeId(0); // Dummy node ID for generated code

        if outputs.len() == 1 {
            // Single output: store result directly
            let ptr_ty = Type::Constructed(TypeName::Pointer, vec![outputs[0].ty.clone()]);
            let ptr = converter
                .builder
                .push_output_ptr(0, ptr_ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            effect = converter
                .builder
                .push_store(ptr, result, effect, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        } else {
            // Tuple output: extract and store each component
            for (i, output) in outputs.iter().enumerate() {
                let component = converter
                    .builder
                    .push_project(result, i as u32, output.ty.clone(), span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                let ptr_ty = Type::Constructed(TypeName::Pointer, vec![output.ty.clone()]);
                let ptr = converter
                    .builder
                    .push_output_ptr(i, ptr_ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                effect = converter
                    .builder
                    .push_store(ptr, component, effect, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            }
        }
        let _ = effect; // Silence unused warning

        // End with ReturnUnit since stores are explicit
        converter
            .builder
            .terminate(Terminator::ReturnUnit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    } else if is_unit_return {
        converter
            .builder
            .terminate(Terminator::ReturnUnit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    } else {
        // Compute shaders or other cases: use Return
        converter
            .builder
            .terminate(Terminator::Return(result))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    }

    let body = converter.finish()?;

    Ok(SsaEntryPoint {
        name: def.name.clone(),
        body,
        execution_model,
        inputs,
        outputs,
        span: def.body.span,
    })
}

/// Extract curried parameters from nested Lams.
fn extract_params(term: &Term) -> (Vec<(String, Type<TypeName>, Span)>, &Term) {
    match &term.kind {
        TermKind::Lam {
            param,
            param_ty,
            body,
        } => {
            let (mut params, inner) = extract_params(body);
            params.insert(0, (param.clone(), param_ty.clone(), term.span));
            (params, inner)
        }
        _ => (vec![], term),
    }
}

/// Extract I/O decoration from a pattern.
fn extract_io_decoration(pattern: &ast::Pattern) -> Option<IoDecoration> {
    match &pattern.kind {
        ast::PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                match attr {
                    ast::Attribute::BuiltIn(builtin) => {
                        return Some(IoDecoration::BuiltIn(*builtin));
                    }
                    ast::Attribute::Location(loc) => {
                        return Some(IoDecoration::Location(*loc));
                    }
                    _ => {}
                }
            }
            extract_io_decoration(inner)
        }
        ast::PatternKind::Typed(inner, _) => extract_io_decoration(inner),
        _ => None,
    }
}

/// Extract size hint from a pattern.
fn extract_size_hint(pattern: &ast::Pattern) -> Option<u32> {
    match &pattern.kind {
        ast::PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let ast::Attribute::SizeHint(n) = attr {
                    return Some(*n);
                }
            }
            extract_size_hint(inner)
        }
        ast::PatternKind::Typed(inner, _) => extract_size_hint(inner),
        _ => None,
    }
}

/// Build entry outputs from AST entry declaration.
/// For compute shaders, unsized array outputs get storage bindings starting from `binding_start`.
fn build_entry_outputs(
    entry: &ast::EntryDecl,
    ret_type: &Type<TypeName>,
    is_compute: bool,
    binding_start: u32,
) -> Vec<EntryOutput> {
    let mut binding_num = binding_start;

    let mut storage_binding_for = |ty: &Type<TypeName>| -> Option<(u32, u32)> {
        if is_compute && is_unsized_array(ty) {
            let binding = (0, binding_num);
            binding_num += 1;
            Some(binding)
        } else {
            None
        }
    };

    if entry.outputs.iter().all(|o| o.attribute.is_none()) && entry.outputs.len() == 1 {
        if !matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
            vec![EntryOutput {
                ty: ret_type.clone(),
                decoration: None,
                storage_binding: storage_binding_for(ret_type),
            }]
        } else {
            vec![]
        }
    } else if let Type::Constructed(TypeName::Tuple(_), component_types) = ret_type {
        entry
            .outputs
            .iter()
            .zip(component_types.iter())
            .map(|(output, ty)| EntryOutput {
                ty: ty.clone(),
                decoration: output.attribute.as_ref().and_then(|a| convert_to_io_decoration(a)),
                storage_binding: storage_binding_for(ty),
            })
            .collect()
    } else {
        vec![EntryOutput {
            ty: ret_type.clone(),
            decoration: entry
                .outputs
                .first()
                .and_then(|o| o.attribute.as_ref())
                .and_then(|a| convert_to_io_decoration(a)),
            storage_binding: storage_binding_for(ret_type),
        }]
    }
}

/// Convert AST attribute to IoDecoration.
fn convert_to_io_decoration(attr: &ast::Attribute) -> Option<IoDecoration> {
    match attr {
        ast::Attribute::BuiltIn(builtin) => Some(IoDecoration::BuiltIn(*builtin)),
        ast::Attribute::Location(loc) => Some(IoDecoration::Location(*loc)),
        _ => None,
    }
}

/// State for converting TLC to SSA.
struct Converter<'a> {
    /// The SSA function builder.
    builder: FuncBuilder,
    /// Mapping from variable names to SSA values.
    locals: HashMap<String, ValueId>,
    /// Top-level definitions for resolving function calls.
    top_level: &'a HashMap<String, &'a TlcDef>,
}

impl<'a> Converter<'a> {
    fn new(builder: FuncBuilder, top_level: &'a HashMap<String, &'a TlcDef>) -> Self {
        Converter {
            builder,
            locals: HashMap::new(),
            top_level,
        }
    }

    /// Finish building and return the function body.
    fn finish(self) -> Result<FuncBody, ConvertError> {
        self.builder.finish().map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    /// Convert a TLC term to SSA, returning the result ValueId.
    fn convert_term(&mut self, term: &Term) -> Result<ValueId, ConvertError> {
        let ty = term.ty.clone();
        let span = term.span;
        let node_id = NodeId(0);

        match &term.kind {
            TermKind::Var(name) => {
                if let Some(&value) = self.locals.get(name) {
                    Ok(value)
                } else {
                    // Global reference
                    self.builder
                        .push_global(name, ty, span, node_id)
                        .map_err(|e| ConvertError::BuilderError(e.to_string()))
                }
            }

            TermKind::IntLit(s) => self
                .builder
                .push_int(s, ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),

            TermKind::FloatLit(f) => self
                .builder
                .push_float(&f.to_string(), ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),

            TermKind::BoolLit(b) => self
                .builder
                .push_bool(*b, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),

            TermKind::StringLit(s) => self
                .builder
                .push_inst(InstKind::String(s.clone()), ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),

            TermKind::Let {
                name,
                name_ty: _,
                rhs,
                body,
            } => {
                let rhs_value = self.convert_term(rhs)?;
                self.locals.insert(name.clone(), rhs_value);
                let result = self.convert_term(body)?;
                self.locals.remove(name);
                Ok(result)
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => self.convert_if(cond, then_branch, else_branch, ty, span, node_id),

            TermKind::App { func, arg } => self.convert_app(func, arg, ty, span, node_id),

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => self.convert_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
                ty,
                span,
                node_id,
            ),

            TermKind::Lam { .. } => {
                panic!(
                    "Unexpected lambda in TLC to SSA conversion at {:?}. \
                     All lambdas should have been lifted to top-level.",
                    span
                )
            }

            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                panic!(
                    "Unexpected bare operator in TLC to SSA conversion at {:?}. \
                     Operators should always be applied to arguments.",
                    span
                )
            }

            TermKind::Extern(linkage_name) => self
                .builder
                .push_inst(InstKind::Extern(linkage_name.clone()), ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),
        }
    }

    /// Convert an if-then-else to SSA blocks.
    fn convert_if(
        &mut self,
        cond: &Term,
        then_branch: &Term,
        else_branch: &Term,
        result_ty: Type<TypeName>,
        _span: Span,
        _node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let cond_value = self.convert_term(cond)?;

        let if_blocks = self.builder.create_if_then_else(result_ty);

        // Mark current block as selection header for SPIR-V structured control flow.
        self.builder
            .mark_selection_header(if_blocks.merge_block)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        self.builder
            .terminate(Terminator::CondBranch {
                cond: cond_value,
                then_target: if_blocks.then_block,
                then_args: vec![],
                else_target: if_blocks.else_block,
                else_args: vec![],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Convert then branch
        self.builder
            .switch_to_block(if_blocks.then_block)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let then_result = self.convert_term(then_branch)?;
        self.builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![then_result],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Convert else branch
        self.builder
            .switch_to_block(if_blocks.else_block)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let else_result = self.convert_term(else_branch)?;
        self.builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![else_result],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Continue from merge
        self.builder
            .switch_to_block(if_blocks.merge_block)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(if_blocks.result)
    }

    /// Collect the spine of a nested application chain.
    fn collect_application_spine<'t>(func: &'t Term, arg: &'t Term) -> (&'t Term, Vec<&'t Term>) {
        let mut args = vec![arg];
        let mut current = func;

        loop {
            match &current.kind {
                TermKind::App {
                    func: inner_func,
                    arg: inner_arg,
                } => {
                    args.push(inner_arg.as_ref());
                    current = inner_func.as_ref();
                }
                _ => {
                    args.reverse();
                    return (current, args);
                }
            }
        }
    }

    /// Convert an application.
    fn convert_app(
        &mut self,
        func: &Term,
        arg: &Term,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let (base_term, args) = Self::collect_application_spine(func, arg);

        match &base_term.kind {
            TermKind::BinOp(op) => {
                assert!(args.len() == 2, "BinOp requires exactly 2 arguments");
                let lhs = self.convert_term(args[0])?;
                let rhs = self.convert_term(args[1])?;
                self.builder
                    .push_binop(&op.op, lhs, rhs, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))
            }

            TermKind::UnOp(op) => {
                assert!(args.len() == 1, "UnOp requires exactly 1 argument");
                let operand = self.convert_term(args[0])?;
                self.builder
                    .push_unary(&op.op, operand, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))
            }

            TermKind::Var(name) => self.convert_var_application(name, &args, ty, span, node_id),

            _ => {
                // This shouldn't happen after defunctionalization
                Err(ConvertError::BuilderError(format!(
                    "Computed function application not supported: {:?}",
                    base_term.kind
                )))
            }
        }
    }

    /// Convert a variable application (function call or intrinsic).
    fn convert_var_application(
        &mut self,
        name: &str,
        args: &[&Term],
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        // Handle HOF intrinsics BEFORE converting args (they handle function args specially)
        match name {
            "_w_intrinsic_reduce" => {
                return self.convert_reduce(args, ty, span, node_id);
            }
            "_w_intrinsic_map" => {
                return self.convert_map(args, ty, span, node_id);
            }
            _ => {}
        }

        let arg_values: Vec<ValueId> =
            args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;

        // Handle special intrinsics
        match name {
            "_w_vec_lit" => {
                return self
                    .builder
                    .push_inst(InstKind::Vector(arg_values), ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_array_lit" => {
                return self
                    .builder
                    .push_inst(InstKind::ArrayLit { elements: arg_values }, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_tuple" => {
                return self
                    .builder
                    .push_tuple(arg_values, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_range" if arg_values.len() == 3 => {
                // _w_range(start, len, kind) -> ArrayRange
                return self
                    .builder
                    .push_inst(
                        InstKind::ArrayRange {
                            start: arg_values[0],
                            len: arg_values[1],
                            step: None,
                        },
                        ty,
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_range_step" if arg_values.len() == 4 => {
                // _w_range_step(start, step, len, kind) -> ArrayRange
                return self
                    .builder
                    .push_inst(
                        InstKind::ArrayRange {
                            start: arg_values[0],
                            len: arg_values[2],
                            step: Some(arg_values[1]),
                        },
                        ty,
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_index" if arg_values.len() == 2 => {
                return self
                    .builder
                    .push_index(arg_values[0], arg_values[1], ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_tuple_proj" if args.len() == 2 => {
                // The second argument must be a constant integer literal
                let base = self.convert_term(args[0])?;
                let index = match &args[1].kind {
                    TermKind::IntLit(s) => s.parse::<u32>().map_err(|_| {
                        ConvertError::BuilderError(format!(
                            "_w_tuple_proj index '{}' is not a valid u32",
                            s
                        ))
                    })?,
                    _ => {
                        return Err(ConvertError::BuilderError(
                            "_w_tuple_proj requires constant integer index".to_string(),
                        ));
                    }
                };
                return self
                    .builder
                    .push_project(base, index, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            _ => {}
        }

        // Check if it's a known function
        if let Some(def) = self.top_level.get(name) {
            let arity = def.arity;
            if arg_values.len() == arity {
                return self
                    .builder
                    .push_call(name, arg_values, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            } else if arg_values.len() < arity {
                panic!(
                    "Partial application not supported: {} expects {} args, got {}",
                    name,
                    arity,
                    arg_values.len()
                );
            } else {
                panic!(
                    "Too many arguments for function {}: expected {}, got {}",
                    name,
                    arity,
                    arg_values.len()
                );
            }
        }

        // Intrinsic or unknown
        if name.starts_with("_w_") {
            self.builder
                .push_intrinsic(name, arg_values, ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
        } else {
            self.builder
                .push_call(name, arg_values, ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
        }
    }

    /// Convert a loop to SSA.
    fn convert_loop(
        &mut self,
        loop_var: &str,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(String, Type<TypeName>, Term)],
        kind: &TlcLoopKind,
        body: &Term,
        _result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        match kind {
            TlcLoopKind::While { cond } => self.convert_while_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                cond,
                body,
                span,
                node_id,
            ),
            TlcLoopKind::ForRange { var, var_ty, bound } => self.convert_for_range_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                var,
                var_ty,
                bound,
                body,
                span,
                node_id,
            ),
            TlcLoopKind::For { var, var_ty, iter } => self.convert_for_in_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                var,
                var_ty,
                iter,
                body,
                span,
                node_id,
            ),
        }
    }

    fn convert_while_loop(
        &mut self,
        loop_var: &str,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(String, Type<TypeName>, Term)],
        cond: &Term,
        body: &Term,
        _span: Span,
        _node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = loop_var_ty.clone();

        let loop_blocks = self.builder.create_while_loop(acc_ty);

        // Convert initial value
        let init_value = self.convert_term(init)?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header: map loop_var to acc parameter
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.locals.insert(loop_var.to_string(), loop_blocks.acc);

        // Process init_bindings
        for (name, _ty, expr) in init_bindings {
            let value = self.convert_term(expr)?;
            self.locals.insert(name.clone(), value);
        }

        // Convert condition
        let cond_value = self.convert_term(cond)?;

        self.builder
            .terminate(Terminator::CondBranch {
                cond: cond_value,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let new_acc = self.convert_term(body)?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Clean up
        self.locals.remove(loop_var);
        for (name, _, _) in init_bindings {
            self.locals.remove(name);
        }

        Ok(loop_blocks.result)
    }

    fn convert_for_range_loop(
        &mut self,
        loop_var: &str,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(String, Type<TypeName>, Term)],
        index_var: &str,
        _index_var_ty: &Type<TypeName>,
        bound: &Term,
        body: &Term,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

        let loop_blocks = self.builder.create_for_range_loop(acc_ty);

        // Convert initial value and bound
        let init_value = self.convert_term(init)?;
        let bound_value = self.convert_term(bound)?;

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.locals.insert(loop_var.to_string(), loop_blocks.acc);
        self.locals.insert(index_var.to_string(), loop_blocks.index);

        // Process init_bindings
        for (name, _ty, expr) in init_bindings {
            let value = self.convert_term(expr)?;
            self.locals.insert(name.clone(), value);
        }

        // Check i < bound
        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, bound_value, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let new_acc = self.convert_term(body)?;
        let one = self
            .builder
            .push_int("1", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop("+", loop_blocks.index, one, i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Clean up
        self.locals.remove(loop_var);
        self.locals.remove(index_var);
        for (name, _, _) in init_bindings {
            self.locals.remove(name);
        }

        Ok(loop_blocks.result)
    }

    fn convert_for_in_loop(
        &mut self,
        loop_var: &str,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(String, Type<TypeName>, Term)],
        elem_var: &str,
        elem_ty: &Type<TypeName>,
        iter: &Term,
        body: &Term,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

        let loop_blocks = self.builder.create_for_range_loop(acc_ty);

        // Convert initial value and iterator
        let init_value = self.convert_term(init)?;
        let iter_value = self.convert_term(iter)?;

        // Get length
        let len = self
            .builder
            .push_intrinsic(
                "_w_intrinsic_length",
                vec![iter_value],
                i32_ty.clone(),
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.locals.insert(loop_var.to_string(), loop_blocks.acc);

        // Get element at index
        let elem = self
            .builder
            .push_index(iter_value, loop_blocks.index, elem_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.locals.insert(elem_var.to_string(), elem);

        // Process init_bindings
        for (name, _ty, expr) in init_bindings {
            let value = self.convert_term(expr)?;
            self.locals.insert(name.clone(), value);
        }

        // Check i < len
        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let new_acc = self.convert_term(body)?;
        let one = self
            .builder
            .push_int("1", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop("+", loop_blocks.index, one, i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Clean up
        self.locals.remove(loop_var);
        self.locals.remove(elem_var);
        for (name, _, _) in init_bindings {
            self.locals.remove(name);
        }

        Ok(loop_blocks.result)
    }

    /// Expand _w_intrinsic_reduce to an explicit loop.
    fn convert_reduce(
        &mut self,
        args: &[&Term],
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        if args.len() < 3 {
            return Err(ConvertError::InvalidIntrinsic(
                "_w_intrinsic_reduce requires 3 arguments (op, init, arr)".to_string(),
            ));
        }

        let op_term = args[0];
        let init_term = args[1];
        let arr_term = args[2];

        // Get function name from op
        let op_name = match &op_term.kind {
            TermKind::Var(name) => name.clone(),
            _ => {
                return Err(ConvertError::InvalidIntrinsic(
                    "_w_intrinsic_reduce: operator must be a function reference".to_string(),
                ));
            }
        };

        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
        let acc_ty = result_ty.clone();

        // Get element type
        let elem_ty = match &arr_term.ty {
            Type::Constructed(TypeName::Array, type_args) if !type_args.is_empty() => type_args[0].clone(),
            _ => acc_ty.clone(),
        };

        // Convert array and init
        let arr_value = self.convert_term(arr_term)?;
        let init_value = self.convert_term(init_term)?;

        // Get length
        let len = self
            .builder
            .push_intrinsic(
                "_w_intrinsic_length",
                vec![arr_value],
                i32_ty.clone(),
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Create loop
        let loop_blocks = self.builder.create_for_range_loop(acc_ty.clone());

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let elem = self
            .builder
            .push_index(arr_value, loop_blocks.index, elem_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let new_acc = self
            .builder
            .push_call(&op_name, vec![loop_blocks.acc, elem], acc_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let one = self
            .builder
            .push_int("1", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop("+", loop_blocks.index, one, i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(loop_blocks.result)
    }

    /// Expand _w_intrinsic_map to an explicit loop.
    fn convert_map(
        &mut self,
        args: &[&Term],
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        if args.len() < 2 {
            return Err(ConvertError::InvalidIntrinsic(
                "_w_intrinsic_map requires 2 arguments (f, arr)".to_string(),
            ));
        }

        let f_term = args[0];
        let arr_term = args[1];
        let capture_terms = &args[2..]; // Captured variables from defunctionalization

        // Get function name
        let f_name = match &f_term.kind {
            TermKind::Var(name) => name.clone(),
            _ => {
                return Err(ConvertError::InvalidIntrinsic(
                    "_w_intrinsic_map: function must be a function reference".to_string(),
                ));
            }
        };

        // Convert captures to SSA values (these are passed as extra args to the lifted function)
        let capture_values: Vec<ValueId> =
            capture_terms.iter().map(|t| self.convert_term(t)).collect::<Result<_, _>>()?;

        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

        // Get element types and array size from the type
        let (input_elem_ty, array_size) = match &arr_term.ty {
            Type::Constructed(TypeName::Array, type_args) if type_args.len() >= 3 => {
                let elem = type_args[0].clone();
                let size = match &type_args[2] {
                    Type::Constructed(TypeName::Size(n), _) => Some(*n),
                    _ => None,
                };
                (elem, size)
            }
            _ => {
                return Err(ConvertError::InvalidIntrinsic(
                    "_w_intrinsic_map: second argument must be an array".to_string(),
                ));
            }
        };
        let output_elem_ty = match &result_ty {
            Type::Constructed(TypeName::Array, type_args) if !type_args.is_empty() => type_args[0].clone(),
            _ => input_elem_ty.clone(),
        };

        // Convert array
        let arr_value = self.convert_term(arr_term)?;

        // Get length - use compile-time size for fixed arrays
        let len = match array_size {
            Some(n) => self
                .builder
                .push_int(&n.to_string(), i32_ty.clone(), span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,
            None => self
                .builder
                .push_intrinsic(
                    "_w_intrinsic_length",
                    vec![arr_value],
                    i32_ty.clone(),
                    span,
                    node_id,
                )
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,
        };

        // Create uninitialized result array
        let init_array = self
            .builder
            .push_call("_w_intrinsic_uninit", vec![], result_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Create loop
        let loop_blocks = self.builder.create_for_range_loop(result_ty.clone());

        // Branch to header
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_array, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let input_elem = self
            .builder
            .push_index(arr_value, loop_blocks.index, input_elem_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Call function with element + captures
        let mut call_args = vec![input_elem];
        call_args.extend(capture_values.iter().copied());
        let output_elem = self
            .builder
            .push_call(&f_name, call_args, output_elem_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let new_arr = self
            .builder
            .push_call(
                "_w_intrinsic_array_with",
                vec![loop_blocks.acc, loop_blocks.index, output_elem],
                result_ty,
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let one = self
            .builder
            .push_int("1", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop("+", loop_blocks.index, one, i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_arr, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(loop_blocks.result)
    }
}
