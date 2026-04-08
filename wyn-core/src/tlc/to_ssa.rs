//! Direct TLC to SSA conversion.
//!
//! This module converts a lifted TLC program directly to SSA form,
//! bypassing the intermediate MIR representation.

use std::collections::{HashMap, HashSet};

use crate::ast::{self, TypeName};
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{FuncBody, InstKind, Soac, Terminator, ValueId, ValueRef, ViewSource};
use crate::types::TypeExt;
use crate::{SymbolId, SymbolTable};
use polytype::Type;

use super::{
    ArrayExpr, Def as TlcDef, DefMeta, Lambda, LoopKind as TlcLoopKind, Program as TlcProgram, SoacOp,
    Term, TermKind,
};

/// Extract parameter types and return type from an arrow type.
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
    ty.array_size().map(|s| matches!(s, Type::Variable(_))).unwrap_or(false)
}

/// Check if a type is an SoA tuple-of-arrays (result of SoA transform on `[n](A,B)` → `([n]A, [n]B)`).
/// Returns the component types if so.
/// Components can be plain arrays or nested SoA tuples (for nested array-of-tuples).
fn is_soa_tuple(ty: &Type<TypeName>) -> Option<&[Type<TypeName>]> {
    match ty {
        Type::Constructed(TypeName::Tuple(_), component_types) if !component_types.is_empty() => {
            let all_soa = component_types.iter().all(|ct| {
                matches!(ct, Type::Constructed(TypeName::Array, args) if args.len() == 3)
                    || is_soa_tuple(ct).is_some()
            });
            if all_soa { Some(component_types) } else { None }
        }
        _ => None,
    }
}

/// Given an SoA tuple type `([n]A, [n]B)`, return the element tuple type `(A, B)`.
/// Handles nested SoA: `(([n]A, [n]B), [n]C)` → `((A, B), C)`.
fn soa_elem_type(soa_ty: &Type<TypeName>) -> Type<TypeName> {
    match soa_ty {
        Type::Constructed(TypeName::Tuple(n), component_types) => {
            let elem_types: Vec<Type<TypeName>> = component_types
                .iter()
                .map(|ct| match ct {
                    ty if ty.is_array() => ty.elem_type().expect("Array has elem").clone(),
                    ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
                    _ => ct.clone(),
                })
                .collect();
            Type::Constructed(TypeName::Tuple(*n), elem_types)
        }
        _ => panic!("BUG: soa_elem_type called on non-SoA type: {:?}", soa_ty),
    }
}

use crate::ssa::types::{
    Constant, EntryInput, EntryOutput, EntryPoint, ExecutionModel, Function, IoDecoration, Program,
};

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
pub fn convert_program(program: &TlcProgram) -> Result<Program, ConvertError> {
    let top_level: HashMap<SymbolId, &TlcDef> = program.defs.iter().map(|d| (d.name, d)).collect();
    let symbols = &program.symbols;

    // Build name-indexed map for arity-0 defs so we can inline constants even
    // when the reference uses a different SymbolId (e.g. after specialize).
    let constants_by_name: HashMap<String, SymbolId> = program
        .defs
        .iter()
        .filter(|d| d.arity == 0 && matches!(&d.meta, DefMeta::Function))
        .filter_map(|d| symbols.get(d.name).map(|n| (n.clone(), d.name)))
        .collect();

    // Phase 1: Identify which arity-0 defs are purely constant by converting
    // their bodies and checking the resulting SSA.
    let mut pure_constant_names: HashSet<String> = HashSet::new();
    let mut constants: Vec<Constant> = Vec::new();

    for def in &program.defs {
        if def.arity != 0 || !matches!(&def.meta, DefMeta::Function) {
            continue;
        }
        if matches!(&def.body.kind, TermKind::Extern(_)) {
            continue;
        }
        let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();

        // Convert the body in isolation to check if it's purely constant.
        // Pass current pure_constants so chained constant defs emit Global refs.
        let ret_type = def.body.ty.clone();
        let builder = FuncBuilder::new(vec![], ret_type.clone());
        let mut converter = Converter::new(
            builder,
            &top_level,
            &constants_by_name,
            symbols,
            pure_constant_names.clone(),
        );
        if let Ok(result_val) = converter.convert_term(&def.body) {
            if let Ok(()) = converter
                .builder
                .terminate(Terminator::Return(Some(result_val)))
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
            {
                if let Ok(body) = converter.finish() {
                    if is_purely_constant_body(&body) {
                        pure_constant_names.insert(def_name.clone());
                        constants.push(Constant {
                            name: def_name,
                            body,
                            result_ty: ret_type,
                        });
                        continue;
                    }
                }
            }
        }
    }

    // Phase 2: Convert functions and entry points with hoisted constant refs.
    let mut functions = Vec::new();
    let mut entry_points = Vec::new();

    for def in &program.defs {
        match &def.meta {
            DefMeta::Function => {
                let def_name = symbols.get(def.name).expect("BUG: symbol not in table");
                if pure_constant_names.contains(def_name) {
                    continue; // already emitted as Constant
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

/// Check whether an SSA function body contains only pure constant instructions.
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

/// Convert a function definition to SSA.
fn convert_function(
    def: &TlcDef,
    top_level: &HashMap<SymbolId, &TlcDef>,
    constants_by_name: &HashMap<String, SymbolId>,
    symbols: &SymbolTable,
    pure_constants: &HashSet<String>,
) -> Result<Function, ConvertError> {
    let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();

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

        return Ok(Function {
            name: def_name,
            body,
            span: def.body.span,
            linkage_name: Some(linkage_name.clone()),
        });
    }

    // Extract parameters from nested Lams
    let (params, inner_body) = extract_params(&def.body, symbols);

    // Build parameter list for FuncBuilder
    let param_list: Vec<(Type<TypeName>, String)> =
        params.iter().map(|(name, ty)| (ty.clone(), name.clone())).collect();

    let ret_type = inner_body.ty.clone();
    let builder = FuncBuilder::new(param_list, ret_type.clone());

    // Create converter with variable mappings
    let mut converter = Converter::new(
        builder,
        top_level,
        constants_by_name,
        symbols,
        pure_constants.clone(),
    );

    // Map parameters to their ValueIds
    for (i, (name, _)) in params.iter().enumerate() {
        let value = converter.builder.get_param(i);
        converter.locals.insert(name.clone(), value);
    }

    // Wrap view-array parameters in inherited StorageView instructions so
    // the SPIR-V backend can resolve buffer identity by walking the chain
    // back to the caller's StorageView.

    let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
    for (i, (name, ty)) in params.iter().enumerate() {
        let is_view = ty
            .array_variant()
            .map(|v| matches!(v, Type::Constructed(TypeName::ArrayVariantView, _)))
            .unwrap_or(false);
        if is_view {
            let param_val = converter.builder.get_param(i);
            // The param is an opaque view struct; create an inherited view that
            // references it so the backend can chase the chain.
            let zero = converter
                .builder
                .push_int("0", u32_ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            let view_len = converter
                .builder
                .push_intrinsic("_w_intrinsic_view_len", vec![param_val], u32_ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            let view = converter
                .builder
                .emit_inherited_view(param_val, zero, view_len, ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            converter.locals.insert(name.clone(), view);
        }
    }

    // Convert the body
    let result = converter.convert_term(inner_body)?;

    // Terminate with return
    if matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
        converter
            .builder
            .terminate(Terminator::Return(None))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    } else {
        converter
            .builder
            .terminate(Terminator::Return(Some(result)))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    }

    let body = converter.finish()?;

    Ok(Function {
        name: def_name,
        body,
        span: def.body.span,
        linkage_name: None,
    })
}

/// Convert an entry point definition to SSA.
fn convert_entry_point(
    def: &TlcDef,
    entry: &ast::EntryDecl,
    top_level: &HashMap<SymbolId, &TlcDef>,
    constants_by_name: &HashMap<String, SymbolId>,
    symbols: &SymbolTable,
    pure_constants: &HashSet<String>,
) -> Result<EntryPoint, ConvertError> {
    let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();

    // Extract parameters from nested Lams
    let (params, inner_body) = extract_params(&def.body, symbols);

    let is_compute = matches!(entry.entry_type, ast::Attribute::Compute);

    // Build inputs with decorations
    let mut inputs = Vec::new();
    let mut binding_num = 0u32;
    let mut pc_offset = 0u32;

    for (i, (name, ty)) in params.iter().enumerate() {
        let decoration = entry.params.get(i).and_then(|p| extract_io_decoration(p));
        let size_hint = entry.params.get(i).and_then(|p| extract_size_hint(p));

        let storage_binding = if is_compute && is_unsized_array(ty) {
            let binding = (0, binding_num);
            binding_num += 1;
            Some(binding)
        } else {
            None
        };

        // Compute shader inputs that are not storage buffers and not builtins
        // become push constants.
        let push_constant_offset = if is_compute
            && storage_binding.is_none()
            && !matches!(&decoration, Some(IoDecoration::BuiltIn(_)))
        {
            let offset = pc_offset;
            pc_offset += crate::ssa::layout::type_byte_size(ty).unwrap_or(4);
            Some(offset)
        } else {
            None
        };

        inputs.push(EntryInput {
            name: name.clone(),
            ty: ty.clone(),
            decoration,
            size_hint,
            storage_binding,
            push_constant_offset,
        });
    }

    // Build parameter list for FuncBuilder
    let param_list: Vec<(Type<TypeName>, String)> =
        params.iter().map(|(name, ty)| (ty.clone(), name.clone())).collect();

    let ret_type = inner_body.ty.clone();
    let builder = FuncBuilder::new(param_list, ret_type.clone());

    // Create converter
    let mut converter = Converter::new(
        builder,
        top_level,
        constants_by_name,
        symbols,
        pure_constants.clone(),
    );

    // Map parameters
    for (i, (name, _)) in params.iter().enumerate() {
        let value = converter.builder.get_param(i);
        converter.locals.insert(name.clone(), value);
    }

    // Wrap storage buffer parameters in whole-buffer views so that any direct
    // use of the parameter (e.g. slicing) sees a StorageView rather than a raw
    // ValueId that has no SPIR-V representation. The parallelizer replaces the
    // entire body when it applies, so these views are harmless in that case.
    let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
    for input in &inputs {
        if let Some((set, binding)) = input.storage_binding {
            let view = converter
                .builder
                .emit_storage_view(set, binding, input.ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            converter.locals.insert(input.name.clone(), view);
        }
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

    if is_unit_return {
        converter
            .builder
            .terminate(Terminator::Return(None))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    } else if is_compute && !outputs.is_empty() {
        // Compute shader with non-unit return: write result to output storage buffer
        // using the same StorageView + StorageViewIndex + Store helpers as the parallelizer.
        let mut effect = converter.builder.entry_effect();

        for (i, output) in outputs.iter().enumerate() {
            let (set, binding) =
                output.storage_binding.expect("BUG: compute output without storage binding");
            let value = if outputs.len() == 1 {
                result
            } else {
                converter
                    .builder
                    .push_project(result, i as u32, output.ty.clone())
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            };

            // Check if the output is a fixed-size array — if so, store elements individually
            let fixed_size = output.ty.array_size().and_then(|s| {
                if let Type::Constructed(TypeName::Size(n), _) = s { Some(*n) } else { None }
            });
            let elem_ty = output.ty.elem_type().cloned();

            if let (Some(n), Some(et)) = (fixed_size, elem_ty) {
                // Fixed-size array: unpack and store each element
                let view = converter
                    .builder
                    .emit_storage_view(set, binding, et.clone())
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                for j in 0..n {
                    let elem = converter
                        .builder
                        .push_project(value, j as u32, et.clone())
                        .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                    let idx = converter
                        .builder
                        .push_int(&j.to_string(), u32_ty.clone())
                        .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                    effect = converter
                        .builder
                        .emit_storage_store(view, idx, elem, et.clone(), effect)
                        .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                }
            } else {
                // Scalar or other: store directly at index 0
                let view = converter
                    .builder
                    .emit_storage_view(set, binding, output.ty.clone())
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                let idx_zero = converter
                    .builder
                    .push_int("0", u32_ty.clone())
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                effect = converter
                    .builder
                    .emit_storage_store(view, idx_zero, value, output.ty.clone(), effect)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            }
        }
        let _ = effect;

        converter
            .builder
            .terminate(Terminator::Return(None))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    } else if !is_compute && !is_unit_return && !outputs.is_empty() {
        // Vertex/fragment shader with outputs: emit OutputPtr + Store
        let mut effect = converter.builder.entry_effect();

        if outputs.len() == 1 {
            // Single output: store result directly
            let ptr_ty = Type::Constructed(
                TypeName::Pointer,
                vec![
                    outputs[0].ty.clone(),
                    Type::Constructed(TypeName::PointerOutput, vec![]),
                ],
            );
            let ptr = converter
                .builder
                .push_output_ptr(0, ptr_ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            effect = converter
                .builder
                .push_store(ptr, result, effect)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        } else {
            // Tuple output: extract and store each component
            for (i, output) in outputs.iter().enumerate() {
                let component = converter
                    .builder
                    .push_project(result, i as u32, output.ty.clone())
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                let ptr_ty = Type::Constructed(
                    TypeName::Pointer,
                    vec![
                        output.ty.clone(),
                        Type::Constructed(TypeName::PointerOutput, vec![]),
                    ],
                );
                let ptr = converter
                    .builder
                    .push_output_ptr(i, ptr_ty)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                effect = converter
                    .builder
                    .push_store(ptr, component, effect)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            }
        }
        let _ = effect;

        converter
            .builder
            .terminate(Terminator::Return(None))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    } else {
        // Non-entry or no outputs: use Return
        converter
            .builder
            .terminate(Terminator::Return(Some(result)))
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
    }

    let body = converter.finish()?;

    Ok(EntryPoint {
        name: def_name,
        body,
        execution_model,
        inputs,
        outputs,
        span: def.body.span,
    })
}

/// Extract parameters from a Lambda term.
/// Returns parameter names as Strings (looked up from symbol table) for SSA construction.
fn extract_params<'a>(term: &'a Term, symbols: &SymbolTable) -> (Vec<(String, Type<TypeName>)>, &'a Term) {
    match &term.kind {
        TermKind::Lambda(Lambda {
            params: lam_params,
            body,
            ..
        }) => {
            let (mut params, inner) = extract_params(body, symbols);
            // Insert in reverse order so first param ends up first
            for (p, ty) in lam_params.iter().rev() {
                let param_name = symbols.get(*p).expect("BUG: symbol not in table").clone();
                params.insert(0, (param_name, ty.clone()));
            }
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
/// For compute shaders, all non-unit outputs get storage bindings starting from `binding_start`.
fn build_entry_outputs(
    entry: &ast::EntryDecl,
    ret_type: &Type<TypeName>,
    is_compute: bool,
    binding_start: u32,
) -> Vec<EntryOutput> {
    let mut binding_num = binding_start;

    let mut storage_binding_for = |ty: &Type<TypeName>, is_compute: bool| -> Option<(u32, u32)> {
        if is_compute && !matches!(ty, Type::Constructed(TypeName::Unit, _)) {
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
                storage_binding: storage_binding_for(ret_type, is_compute),
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
                storage_binding: storage_binding_for(ty, is_compute),
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
            storage_binding: storage_binding_for(ret_type, is_compute),
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
    top_level: &'a HashMap<SymbolId, &'a TlcDef>,
    /// Arity-0 defs indexed by name (for cross-SymbolId lookup after specialize).
    constants_by_name: &'a HashMap<String, SymbolId>,
    /// Symbol table for name lookup.
    symbols: &'a SymbolTable,
    /// Cache for inlined constant defs, keyed by name for cross-SymbolId hits.
    inlined_constants: HashMap<String, ValueId>,
    /// Names of pure constant defs hoisted to program scope.
    /// References emit `Global(name)` instead of inlining the body.
    pure_constants: HashSet<String>,
}

impl<'a> Converter<'a> {
    fn new(
        builder: FuncBuilder,
        top_level: &'a HashMap<SymbolId, &'a TlcDef>,
        constants_by_name: &'a HashMap<String, SymbolId>,
        symbols: &'a SymbolTable,
        pure_constants: HashSet<String>,
    ) -> Self {
        Converter {
            builder,
            locals: HashMap::new(),
            top_level,
            constants_by_name,
            symbols,
            inlined_constants: HashMap::new(),
            pure_constants,
        }
    }

    /// Finish building and return the function body.
    fn finish(self) -> Result<FuncBody, ConvertError> {
        self.builder.finish().map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    /// Convert a TLC term to SSA, returning the result ValueId.
    fn convert_term(&mut self, term: &Term) -> Result<ValueId, ConvertError> {
        let ty = term.ty.clone();

        match &term.kind {
            TermKind::Var(sym) => {
                let name = self.symbols.get(*sym).expect("BUG: symbol not in table");
                if let Some(&value) = self.locals.get(name) {
                    Ok(value)
                } else if let Some(&cached) = self.inlined_constants.get(name) {
                    Ok(cached)
                } else if self.pure_constants.contains(name) {
                    // Hoisted constant — emit a Global reference instead of inlining
                    self.builder
                        .push_global(name, ty)
                        .map_err(|e| ConvertError::BuilderError(e.to_string()))
                } else {
                    // Check if this is an arity-0 constant def (by SymbolId or by name).
                    let const_def = self.top_level.get(sym).filter(|d| d.arity == 0).or_else(|| {
                        self.constants_by_name.get(name).and_then(|def_sym| self.top_level.get(def_sym))
                    });
                    if let Some(def) = const_def {
                        let body = def.body.clone();
                        let value = self.convert_term(&body)?;
                        let name = self.symbols.get(*sym).expect("BUG").clone();
                        self.inlined_constants.insert(name, value);
                        Ok(value)
                    } else {
                        // Global reference
                        self.builder
                            .push_global(name, ty)
                            .map_err(|e| ConvertError::BuilderError(e.to_string()))
                    }
                }
            }

            TermKind::IntLit(s) => {
                self.builder.push_int(s, ty).map_err(|e| ConvertError::BuilderError(e.to_string()))
            }

            TermKind::FloatLit(f) => self
                .builder
                .push_float(&f.to_string(), ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),

            TermKind::BoolLit(b) => {
                self.builder.push_bool(*b).map_err(|e| ConvertError::BuilderError(e.to_string()))
            }

            TermKind::StringLit(s) => self
                .builder
                .push_inst(InstKind::String(s.clone()), ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),

            TermKind::Let {
                name: name_sym,
                name_ty: _,
                rhs,
                body,
            } => {
                let name = self.symbols.get(*name_sym).expect("BUG: symbol not in table").clone();
                let rhs_value = self.convert_term(rhs)?;
                self.locals.insert(name.clone(), rhs_value);
                let result = self.convert_term(body)?;
                self.locals.remove(&name);
                Ok(result)
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => self.convert_if(cond, then_branch, else_branch, ty),

            TermKind::App { func, args } => self.convert_app(func, args, ty),

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                // Look up names from symbol table for loop variables
                let loop_var_name = self.symbols.get(*loop_var).expect("BUG: symbol not in table");
                let resolved_bindings: Vec<(String, Type<TypeName>, &Term)> = init_bindings
                    .iter()
                    .map(|(sym, ty, expr)| {
                        let name = self.symbols.get(*sym).expect("BUG: symbol not in table").clone();
                        (name, ty.clone(), expr)
                    })
                    .collect();
                self.convert_loop(
                    loop_var_name,
                    loop_var_ty,
                    init,
                    &resolved_bindings,
                    kind,
                    body,
                    ty,
                )
            }

            TermKind::Lambda(..) => {
                panic!(
                    "Unexpected lambda in TLC to SSA conversion. \
                     All lambdas should have been lifted to top-level.",
                )
            }

            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                panic!(
                    "Unexpected bare operator in TLC to SSA conversion. \
                     Operators should always be applied to arguments.",
                )
            }

            TermKind::Extern(linkage_name) => self
                .builder
                .push_inst(InstKind::Extern(linkage_name.clone()), ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string())),

            TermKind::Soac(ref soac) => self.convert_soac(soac, ty),

            TermKind::ArrayExpr(ref ae) => self.convert_array_expr(ae, ty),

            TermKind::Force(ref inner) => self.convert_term(inner),

            TermKind::Pack { .. } | TermKind::Unpack { .. } => {
                unreachable!("Pack/Unpack nodes not yet lowered to SSA")
            }
        }
    }

    /// Convert an if-then-else to SSA blocks.
    fn convert_if(
        &mut self,
        cond: &Term,
        then_branch: &Term,
        else_branch: &Term,
        result_ty: Type<TypeName>,
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

    /// Convert an array literal by building it incrementally.
    ///
    /// Instead of converting all elements first and emitting one ArrayLit,
    /// we start with an uninit array and insert each element one at a time.
    /// This is necessary because element conversion may introduce control flow
    /// (if/else) that changes the current block. Each insert is emitted in
    /// whatever block is current after converting that element, so all value
    /// references stay within their defining block's dominance frontier.
    fn convert_array_lit_incremental(
        &mut self,
        elements: &[&Term],
        arr_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        // If no elements could introduce control flow, use the fast path
        if elements.iter().all(|e| !Self::term_may_branch(e)) {
            let values: Vec<ValueId> =
                elements.iter().map(|t| self.convert_term(t)).collect::<Result<_, _>>()?;
            return self
                .builder
                .push_inst(
                    InstKind::ArrayLit {
                        elements: values.into_iter().map(ValueRef::from).collect(),
                    },
                    arr_ty,
                )
                .map_err(|e| ConvertError::BuilderError(e.to_string()));
        }

        // Slow path: build incrementally with uninit + array_with chain
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

        let mut arr = self
            .builder
            .push_call("_w_intrinsic_uninit", vec![], arr_ty.clone())
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        for (i, elem) in elements.iter().enumerate() {
            let val = self.convert_term(elem)?;
            let idx = self
                .builder
                .push_inst(InstKind::Int(i.to_string()), i32_ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            arr = self
                .builder
                .push_call("_w_intrinsic_array_with", vec![arr, idx, val], arr_ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        }

        Ok(arr)
    }

    /// Check whether a term might introduce control flow (if/else, loops, let with CFG body).
    fn term_may_branch(term: &Term) -> bool {
        match &term.kind {
            TermKind::If { .. } | TermKind::Loop { .. } => true,
            TermKind::Let { rhs, body, .. } => Self::term_may_branch(rhs) || Self::term_may_branch(body),
            TermKind::App { func, args } => {
                Self::term_may_branch(func) || args.iter().any(|a| Self::term_may_branch(a))
            }
            TermKind::ArrayExpr(ae) => match ae {
                ArrayExpr::Literal(terms) => terms.iter().any(|t| Self::term_may_branch(t)),
                ArrayExpr::Ref(t) => Self::term_may_branch(t),
                _ => true, // conservative for SOACs etc.
            },
            _ => false,
        }
    }

    /// Convert an application.
    fn convert_app(
        &mut self,
        func: &Term,
        args: &[Term],
        ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let args: Vec<&Term> = args.iter().collect();

        match &func.kind {
            TermKind::BinOp(op) => {
                assert!(args.len() == 2, "BinOp requires exactly 2 arguments");
                let lhs = self.convert_term(args[0])?;
                let rhs = self.convert_term(args[1])?;
                self.builder
                    .push_binop(&op.op, lhs, rhs, ty)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))
            }

            TermKind::UnOp(op) => {
                assert!(args.len() == 1, "UnOp requires exactly 1 argument");
                let operand = self.convert_term(args[0])?;
                self.builder
                    .push_unary(&op.op, operand, ty)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))
            }

            TermKind::Var(sym) => {
                let name = self.symbols.get(*sym).expect("BUG: symbol not in table");
                self.convert_var_application(*sym, name, &args, ty)
            }

            _ => {
                // This shouldn't happen after defunctionalization
                Err(ConvertError::BuilderError(format!(
                    "Computed function application not supported: {:?}",
                    func.kind
                )))
            }
        }
    }

    /// Convert a variable application (function call or intrinsic).
    fn convert_var_application(
        &mut self,
        sym: SymbolId,
        name: &str,
        args: &[&Term],
        ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        // Handle _w_array_lit before converting args, since element conversion
        // may introduce control flow (if/else) that changes the current block.
        // We build the array incrementally so each element insert happens in
        // whatever block is current after that element is converted.
        if name == "_w_array_lit" {
            return self.convert_array_lit_incremental(args, ty);
        }

        let arg_values: Vec<ValueId> =
            args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;

        // Handle special intrinsics
        match name {
            "_w_vec_lit" => {
                return self
                    .builder
                    .push_inst(
                        InstKind::Vector(arg_values.into_iter().map(ValueRef::from).collect()),
                        ty,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_tuple" => {
                return self
                    .builder
                    .push_tuple(arg_values, ty)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_range" if arg_values.len() == 3 => {
                // _w_range(start, len, kind) -> ArrayRange
                return self
                    .builder
                    .push_inst(
                        InstKind::ArrayRange {
                            start: ValueRef::from(arg_values[0]),
                            len: ValueRef::from(arg_values[1]),
                            step: None,
                        },
                        ty,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_range_step" if arg_values.len() == 4 => {
                // _w_range_step(start, step, len, kind) -> ArrayRange
                return self
                    .builder
                    .push_inst(
                        InstKind::ArrayRange {
                            start: ValueRef::from(arg_values[0]),
                            len: ValueRef::from(arg_values[2]),
                            step: Some(ValueRef::from(arg_values[1])),
                        },
                        ty,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            "_w_index" if arg_values.len() == 2 => {
                // SoA-aware: if the array is a tuple-of-arrays, distribute the index
                let arr_ty = &args[0].ty;
                return self.soa_index(arg_values[0], arg_values[1], arr_ty, &ty);
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
                    .push_project(base, index, ty)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()));
            }
            _ => {}
        }

        // Check if it's a known function
        if let Some(def) = self.top_level.get(&sym) {
            let arity = def.arity;
            if arg_values.len() == arity {
                return self
                    .builder
                    .push_call(name, arg_values, ty)
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

        // _w_intrinsic_storage_index(set_const, binding_const, index) → element value
        // Emitted by the buffer_specialize pass for functions that index into buffers.
        if name == "_w_intrinsic_storage_index" && args.len() == 3 {
            let set = match &args[0].kind {
                TermKind::IntLit(s) => s.parse::<u32>().map_err(|_| {
                    ConvertError::InvalidIntrinsic("_w_intrinsic_storage_index: set is not u32".into())
                })?,
                _ => {
                    return Err(ConvertError::InvalidIntrinsic(
                        "_w_intrinsic_storage_index: set must be int literal".into(),
                    ));
                }
            };
            let binding = match &args[1].kind {
                TermKind::IntLit(s) => s.parse::<u32>().map_err(|_| {
                    ConvertError::InvalidIntrinsic("_w_intrinsic_storage_index: binding is not u32".into())
                })?,
                _ => {
                    return Err(ConvertError::InvalidIntrinsic(
                        "_w_intrinsic_storage_index: binding must be int literal".into(),
                    ));
                }
            };
            // Build a storage view for this buffer, then index into it
            let view_ty = ty.clone(); // approximate — just need something array-like
            let view = self
                .builder
                .emit_storage_view(set, binding, view_ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            let index = arg_values[2]; // already converted
            // StorageViewIndex produces a pointer (SPIR-V OpAccessChain),
            // then we load the value from it.
            let ptr = self
                .builder
                .push_inst(
                    InstKind::StorageViewIndex {
                        view: ValueRef::from(view),
                        index: ValueRef::from(index),
                    },
                    ty.clone(),
                )
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            // Load the element from the pointer
            let effect_in = self.builder.entry_effect();
            return self
                .builder
                .push_load(ptr, ty, effect_in)
                .map_err(|e| ConvertError::BuilderError(e.to_string()));
        }

        // Builtins and internal intrinsics → InstKind::Intrinsic
        if name.starts_with("_w_intrinsic_") {
            self.builder
                .push_intrinsic(name, arg_values, ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
        } else {
            // Method-dispatched builtins (f32.sin, etc.) go through Call → impl_source
            self.builder
                .push_call(name, arg_values, ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
        }
    }

    /// Convert a loop to SSA.
    fn convert_loop(
        &mut self,
        loop_var: &str,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(String, Type<TypeName>, &Term)],
        kind: &TlcLoopKind,
        body: &Term,
        _result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        match kind {
            TlcLoopKind::While { cond } => {
                self.convert_while_loop(loop_var, loop_var_ty, init, init_bindings, cond, body)
            }
            TlcLoopKind::ForRange { var, var_ty, bound } => {
                let index_var = self.symbols.get(*var).expect("BUG: symbol not in table");
                self.convert_for_range_loop(
                    loop_var,
                    loop_var_ty,
                    init,
                    init_bindings,
                    index_var,
                    var_ty,
                    bound,
                    body,
                )
            }
            TlcLoopKind::For { var, var_ty, iter } => {
                let elem_var = self.symbols.get(*var).expect("BUG: symbol not in table");
                self.convert_for_in_loop(
                    loop_var,
                    loop_var_ty,
                    init,
                    init_bindings,
                    elem_var,
                    var_ty,
                    iter,
                    body,
                )
            }
        }
    }

    fn convert_while_loop(
        &mut self,
        loop_var: &str,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(String, Type<TypeName>, &Term)],
        cond: &Term,
        body: &Term,
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
        init_bindings: &[(String, Type<TypeName>, &Term)],
        index_var: &str,
        _index_var_ty: &Type<TypeName>,
        bound: &Term,
        body: &Term,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        let loop_blocks = self.builder.create_for_range_loop(acc_ty);

        // Convert initial value and bound
        let init_value = self.convert_term(init)?;
        let bound_value = self.convert_term(bound)?;

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone())
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
            .push_binop("<", loop_blocks.index, bound_value, bool_ty)
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
            .push_int("1", i32_ty.clone())
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop("+", loop_blocks.index, one, i32_ty)
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
        init_bindings: &[(String, Type<TypeName>, &Term)],
        elem_var: &str,
        elem_ty: &Type<TypeName>,
        iter: &Term,
        body: &Term,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        let loop_blocks = self.builder.create_for_range_loop(acc_ty);

        // Convert initial value and iterator
        let init_value = self.convert_term(init)?;
        let iter_ty = iter.ty.clone();
        let iter_value = self.convert_term(iter)?;

        // Get length (SoA-aware)
        let len = self.soa_length(iter_value, &iter_ty)?;

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone())
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

        // Get element at index (SoA-aware)
        let elem = self.soa_index(iter_value, loop_blocks.index, &iter_ty, elem_ty)?;
        self.locals.insert(elem_var.to_string(), elem);

        // Process init_bindings
        for (name, _ty, expr) in init_bindings {
            let value = self.convert_term(expr)?;
            self.locals.insert(name.clone(), value);
        }

        // Check i < len
        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, len, bool_ty)
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
            .push_int("1", i32_ty.clone())
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop("+", loop_blocks.index, one, i32_ty)
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

    // =========================================================================
    // SoA-aware array operation helpers
    // =========================================================================
    // These methods abstract over plain arrays vs SoA tuple-of-arrays,
    // so the rest of the lowering code doesn't need to know about SoA.

    /// Emit length of an array-like value (plain array or SoA tuple-of-arrays).
    fn soa_length(&mut self, arr: ValueId, arr_ty: &Type<TypeName>) -> Result<ValueId, ConvertError> {
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        if let Some(components) = is_soa_tuple(arr_ty) {
            // Recurse: first component might itself be an SoA tuple
            let first = self
                .builder
                .push_project(arr, 0, components[0].clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
            self.soa_length(first, &components[0])
        } else {
            self.builder
                .push_intrinsic("_w_intrinsic_length", vec![arr], i32_ty)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
        }
    }

    /// Emit index into an array-like value.
    /// For SoA tuple-of-arrays: project each component, index each, repack as element tuple.
    fn soa_index(
        &mut self,
        arr: ValueId,
        index: ValueId,
        arr_ty: &Type<TypeName>,
        elem_ty: &Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        if let Some(components) = is_soa_tuple(arr_ty) {
            let mut elem_values = Vec::with_capacity(components.len());
            for (i, comp_ty) in components.iter().enumerate() {
                let comp_arr = self
                    .builder
                    .push_project(arr, i as u32, comp_ty.clone())
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                let comp_elem_ty = match comp_ty {
                    ty if ty.is_array() => ty.elem_type().expect("Array has elem").clone(),
                    ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
                    _ => comp_ty.clone(),
                };
                // Recurse: component might itself be an SoA tuple (nested array-of-tuples)
                let elem = self.soa_index(comp_arr, index, comp_ty, &comp_elem_ty)?;
                elem_values.push(elem);
            }
            self.builder
                .push_tuple(elem_values, elem_ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
        } else {
            self.builder
                .push_index(arr, index, elem_ty.clone())
                .map_err(|e| ConvertError::BuilderError(e.to_string()))
        }
    }

    // =========================================================================
    // SOAC / ArrayExpr lowering
    // =========================================================================

    /// Dispatch SOAC node to the appropriate lowering.
    fn convert_soac(&mut self, soac: &SoacOp, ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
        match soac {
            SoacOp::Map { lam, inputs } => self.convert_soac_map(lam, inputs, ty),
            SoacOp::Reduce { op, ne, input, .. } => self.convert_soac_reduce(op, ne, input, ty),
            SoacOp::Redomap { op, ne, inputs, .. } => self.convert_soac_redomap(op, ne, inputs, ty),
            SoacOp::Scan { op, ne, input } => self.convert_soac_scan(op, ne, input, ty),
            SoacOp::Filter { pred, input } => self.convert_soac_filter(pred, input, ty),
            SoacOp::Scatter { .. } => todo!("SOAC scatter lowering"),
            SoacOp::ReduceByIndex { .. } => todo!("SOAC reduce_by_index lowering"),
        }
    }

    /// Convert an ArrayExpr to SSA.
    fn convert_array_expr(&mut self, ae: &ArrayExpr, ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
        match ae {
            ArrayExpr::Ref(term) => self.convert_term(term),
            ArrayExpr::Zip(_) => {
                panic!("ArrayExpr::Zip should have been eliminated by soa::normalize");
            }
            ArrayExpr::Soac(op) => self.convert_soac(op, ty),
            ArrayExpr::Generate { .. } => todo!("ArrayExpr::Generate lowering"),
            ArrayExpr::Literal(terms) => {
                let term_refs: Vec<&Term> = terms.iter().collect();
                self.convert_array_lit_incremental(&term_refs, ty)
            }
            ArrayExpr::Range { start, len } => {
                let start_val = self.convert_term(start)?;
                let len_val = self.convert_term(len)?;
                self.builder
                    .push_inst(
                        InstKind::ArrayRange {
                            start: ValueRef::from(start_val),
                            len: ValueRef::from(len_val),
                            step: None,
                        },
                        ty,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))
            }
            ArrayExpr::StorageBuffer {
                set,
                binding,
                offset,
                len,
                elem_ty,
            } => {
                let offset_val = self.convert_term(offset)?;
                let len_val = self.convert_term(len)?;
                let array_ty = Type::Constructed(
                    TypeName::Array,
                    vec![
                        elem_ty.clone(),
                        Type::Constructed(TypeName::SizePlaceholder, vec![]),
                        Type::Constructed(TypeName::ArrayVariantView, vec![]),
                    ],
                );
                self.builder
                    .push_inst(
                        InstKind::StorageView {
                            source: ViewSource::Storage {
                                set: *set,
                                binding: *binding,
                            },
                            offset: ValueRef::from(offset_val),
                            len: ValueRef::from(len_val),
                        },
                        array_ty,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))
            }
        }
    }

    /// Get the function name from a Lambda's body (post-defunctionalization).
    /// After defunc, the lambda body is either:
    /// - A Var referencing the lifted function, or
    /// - An App chain of the lifted function applied to args
    fn lambda_fn_name(&self, lam: &Lambda) -> Result<String, ConvertError> {
        match &lam.body.kind {
            TermKind::Var(sym) => Ok(self.symbols.get(*sym).expect("BUG: symbol not in table").clone()),
            _ => Err(ConvertError::BuilderError(
                "SOAC lambda body should be a function reference post-defunc".to_string(),
            )),
        }
    }

    /// Emit a first-class `Soac(Map { ... })` instruction.
    /// The actual loop expansion is deferred to the `ssa_soac_lower` pass.
    fn convert_soac_map(
        &mut self,
        lam: &Lambda,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let f_name = self.lambda_fn_name(lam)?;

        // Convert captures to SSA values
        let capture_values: Vec<ValueId> =
            lam.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        // Collect input array types before converting (needed for SoA-aware ops)
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_type(ae)).collect();

        // Convert all input arrays to SSA values
        let input_values: Vec<ValueId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;

        // Extract element types from input array types (SoA-aware)
        let input_elem_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_elem_type(ae)).collect();

        // Get output element type (SoA-aware)
        let output_elem_ty = match &result_ty {
            ty if ty.is_array() => ty.elem_type().expect("Array has elem").clone(),
            ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
            _ => {
                if !input_elem_types.is_empty() {
                    input_elem_types[0].clone()
                } else {
                    return Err(ConvertError::BuilderError(
                        "map: cannot determine output element type".to_string(),
                    ));
                }
            }
        };

        self.builder
            .push_inst(
                InstKind::Soac(Soac::Map {
                    func: f_name,
                    inputs: input_values,
                    captures: capture_values,
                    input_array_types: input_arr_types,
                    input_elem_types,
                    output_elem_type: output_elem_ty,
                }),
                result_ty,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    /// Emit a first-class `Soac(Reduce { ... })` instruction.
    /// The actual loop expansion is deferred to the `ssa_soac_lower` pass.
    fn convert_soac_reduce(
        &mut self,
        op: &Lambda,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;

        // Convert captures
        let capture_values: Vec<ValueId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        // Get element type and array type (SoA-aware)
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);

        // Convert array and init
        let arr_value = self.convert_array_expr_value(input)?;
        let init_value = self.convert_term(ne)?;

        self.builder
            .push_inst(
                InstKind::Soac(Soac::Reduce {
                    func: op_name,
                    input: arr_value,
                    init: init_value,
                    captures: capture_values,
                    input_array_type: arr_ty,
                    input_elem_type: elem_ty,
                }),
                result_ty,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    /// Emit a first-class `Soac(Redomap { ... })` instruction.
    /// The actual loop expansion is deferred to the `ssa_soac_lower` pass.
    fn convert_soac_redomap(
        &mut self,
        op: &Lambda,
        ne: &Term,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;

        // Convert captures
        let capture_values: Vec<ValueId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        // Collect input array types before converting (needed for SoA-aware ops)
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_type(ae)).collect();

        // Convert all input arrays to SSA values
        let input_values: Vec<ValueId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;

        // Extract element types from input array types (SoA-aware)
        let input_elem_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_elem_type(ae)).collect();

        // Convert init value
        let init_value = self.convert_term(ne)?;

        self.builder
            .push_inst(
                InstKind::Soac(Soac::Redomap {
                    func: op_name,
                    inputs: input_values,
                    init: init_value,
                    captures: capture_values,
                    input_array_types: input_arr_types,
                    input_elem_types,
                }),
                result_ty,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    fn convert_soac_scan(
        &mut self,
        op: &Lambda,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;

        let capture_values: Vec<ValueId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);

        let arr_value = self.convert_array_expr_value(input)?;
        let init_value = self.convert_term(ne)?;

        self.builder
            .push_inst(
                InstKind::Soac(Soac::Scan {
                    func: op_name,
                    input: arr_value,
                    init: init_value,
                    captures: capture_values,
                    input_array_type: arr_ty,
                    input_elem_type: elem_ty,
                }),
                result_ty,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    /// Lower `Soac(Filter { pred, input })` to a call to `_w_intrinsic_filter`.
    /// Filter produces a dynamically-sized output so we emit it as an opaque intrinsic call
    /// rather than an explicit loop.
    fn convert_soac_filter(
        &mut self,
        pred: &Lambda,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let pred_name = self.lambda_fn_name(pred)?;

        // Convert captures to SSA values
        let capture_values: Vec<ValueId> =
            pred.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        // Convert input array
        let arr_value = self.convert_array_expr_value(input)?;

        // Build args: pred function as a global ref, array, then captures
        let fn_ty = Type::Constructed(TypeName::Unit, vec![]); // type doesn't matter for global refs
        let pred_ref = self
            .builder
            .push_global(&pred_name, fn_ty)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let mut args = vec![pred_ref, arr_value];
        args.extend(capture_values);

        self.builder
            .push_intrinsic("_w_intrinsic_filter", args, result_ty)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    // =========================================================================
    // ArrayExpr helpers
    // =========================================================================

    /// Convert an ArrayExpr to its SSA value (just the array, not wrapping in a term).
    fn convert_array_expr_value(&mut self, ae: &ArrayExpr) -> Result<ValueId, ConvertError> {
        match ae {
            ArrayExpr::Ref(term) => self.convert_term(term),
            _ => {
                // For non-Ref ArrayExprs, use convert_array_expr with a dummy type
                let ty = self.array_expr_type(ae);
                self.convert_array_expr(ae, ty)
            }
        }
    }

    /// Get the type of an ArrayExpr.
    fn array_expr_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            ArrayExpr::Ref(t) => t.ty.clone(),
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated by soa::normalize"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]), // placeholder
            ArrayExpr::Generate { elem_ty, .. } => elem_ty.clone(),
            ArrayExpr::Literal(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Range { start, .. } => start.ty.clone(),
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

    /// Extract the element type from an ArrayExpr.
    fn array_expr_elem_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            ArrayExpr::Ref(t) => match &t.ty {
                Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
                ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
                _ => t.ty.clone(),
            },
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated by soa::normalize"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Generate { elem_ty, .. } => elem_ty.clone(),
            ArrayExpr::Literal(terms) => {
                if let Some(first) = terms.first() {
                    first.ty.clone()
                } else {
                    Type::Constructed(TypeName::Unit, vec![])
                }
            }
            ArrayExpr::Range { start, .. } => start.ty.clone(),
            ArrayExpr::StorageBuffer { elem_ty, .. } => elem_ty.clone(),
        }
    }
}
