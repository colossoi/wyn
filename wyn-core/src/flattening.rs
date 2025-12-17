//! Flattening pass: AST -> MIR
//!
//! This pass performs:
//! - Defunctionalization: lambdas become top-level functions with closure records
//! - Pattern flattening: complex patterns become simple let bindings
//! - Lambda lifting: all functions become top-level Def entries

use crate::ast::{self, ExprKind, Expression, NodeCounter, NodeId, PatternKind, Span, Type, TypeName};
use crate::defun_analysis::DefunAnalysis;
use crate::error::Result;
use crate::mir::{self, Body, ExprId, LambdaId, LambdaInfo, LocalDecl, LocalId, LocalKind};
use crate::scope::ScopeStack;
use crate::types;
use crate::{IdArena, bail_flatten, err_flatten, err_type};
use polytype::TypeScheme;
use std::collections::{HashMap, HashSet};

/// Shape classification for desugaring decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgShape {
    Matrix, // mat<n,m,a>
    Vector, // Vec<n,a>
    Other,
}

// Re-export StaticValue from defun_analysis for use in flatten_expr return type
use crate::defun_analysis::StaticValue;

/// Flattener converts AST to MIR with defunctionalization.
pub struct Flattener {
    /// Counter for generating unique names
    next_id: usize,
    /// Counter for generating unique MIR node IDs
    node_counter: NodeCounter,
    /// Generated lambda functions (collected during flattening)
    generated_functions: Vec<mir::Def>,
    /// Stack of enclosing declaration names for lambda naming
    enclosing_decl_stack: Vec<String>,
    /// Pre-computed defunctionalization analysis
    defun_analysis: DefunAnalysis,
    /// Lambda registry: all lambdas (source and synthesized)
    lambda_registry: IdArena<LambdaId, LambdaInfo>,
    /// Type table from type checking - maps NodeId to TypeScheme
    type_table: HashMap<NodeId, TypeScheme<TypeName>>,
    /// Set of builtin names to exclude from free variable capture
    builtins: HashSet<String>,
    /// Set of LocalIds that need backing stores (materialization)
    needs_backing_store: HashSet<LocalId>,
    /// Current body being built (set per-function/entry-point)
    current_body: Body,
    /// Mapping from variable names to LocalIds in the current scope
    name_to_local: ScopeStack<LocalId>,
}

impl Flattener {
    pub fn new(
        type_table: HashMap<NodeId, TypeScheme<TypeName>>,
        builtins: HashSet<String>,
        defun_analysis: DefunAnalysis,
    ) -> Self {
        Flattener {
            next_id: 0,
            node_counter: NodeCounter::new(),
            generated_functions: Vec::new(),
            enclosing_decl_stack: Vec::new(),
            defun_analysis,
            lambda_registry: IdArena::new(),
            type_table,
            builtins,
            needs_backing_store: HashSet::new(),
            current_body: Body::new(),
            name_to_local: ScopeStack::new(),
        }
    }

    /// Start a new body for a function/entry-point. Returns the old body.
    fn begin_body(&mut self) -> Body {
        std::mem::replace(&mut self.current_body, Body::new())
    }

    /// Finish the current body, returning it and restoring the old body.
    fn end_body(&mut self, old_body: Body) -> Body {
        std::mem::replace(&mut self.current_body, old_body)
    }

    /// Allocate a new local variable in the current body.
    fn alloc_local(&mut self, name: String, ty: Type, kind: LocalKind, span: Span) -> LocalId {
        let decl = LocalDecl {
            name: name.clone(),
            span,
            ty,
            kind,
        };
        let local_id = self.current_body.alloc_local(decl);
        self.name_to_local.insert(name, local_id);
        local_id
    }

    /// Allocate a new expression in the current body.
    fn alloc_expr(&mut self, expr: mir::Expr, ty: Type, span: Span) -> ExprId {
        let node_id = self.node_counter.next();
        self.current_body.alloc_expr(expr, ty, span, node_id)
    }

    /// Look up a variable by name, returning its LocalId if it's a local.
    fn lookup_local(&self, name: &str) -> Option<LocalId> {
        self.name_to_local.lookup(name).copied()
    }

    /// Get the NodeCounter for use after flattening
    pub fn into_node_counter(self) -> NodeCounter {
        self.node_counter
    }

    /// Get a fresh NodeId
    fn next_node_id(&mut self) -> NodeId {
        self.node_counter.next()
    }

    /// Get the backing store variable name for a LocalId
    fn backing_store_name(local_id: LocalId) -> String {
        format!("_w_ptr_{}", local_id.0)
    }

    /// Strip Typed wrappers from a pattern, returning the innermost pattern
    fn unwrap_typed_pattern(pattern: &ast::Pattern) -> &ast::Pattern {
        match &pattern.kind {
            PatternKind::Typed(inner, _) => Self::unwrap_typed_pattern(inner),
            _ => pattern,
        }
    }

    /// Register a lambda function and return its ID.
    fn add_lambda(&mut self, name: String, arity: usize) -> LambdaId {
        self.lambda_registry.alloc(LambdaInfo { name, arity })
    }

    /// Query DefunAnalysis for an expression's classification.
    /// Returns Dyn for nodes not in DefunAnalysis (builtins, top-level function names).
    fn get_classification(&self, node_id: NodeId) -> StaticValue {
        self.defun_analysis.get(node_id).cloned().unwrap_or(StaticValue::Dyn)
    }

    /// Extract the monotype from a TypeScheme
    fn get_monotype<'a>(&self, scheme: &'a TypeScheme<TypeName>) -> Option<&'a Type> {
        match scheme {
            TypeScheme::Monotype(ty) => Some(ty),
            TypeScheme::Polytype { body, .. } => self.get_monotype(body),
        }
    }

    /// Get the type of an AST expression from the type table
    fn get_expr_type(&self, expr: &Expression) -> Type {
        self.type_table
            .get(&expr.h.id)
            .and_then(|scheme| self.get_monotype(scheme))
            .cloned()
            .unwrap_or_else(|| {
                eprintln!("BUG: Expression (id={:?}) has no type in type table", expr.h.id);
                eprintln!("Expression kind: {:?}", expr.kind);
                eprintln!("Expression span: {:?}", expr.h.span);
                panic!("BUG: Expression (id={:?}) has no type in type table during flattening. Type checking should ensure all expressions have types.", expr.h.id)
            })
    }

    /// Desugar overloaded function names based on argument types
    /// - mul -> mul_mat_mat, mul_mat_vec, mul_vec_mat
    /// - matav -> matav_n_m
    /// - abs/sign -> abs_f32, abs_i32, abs_u32, sign_f32, sign_i32
    /// - min/max -> min_f32, min_i32, min_u32, max_f32, max_i32, max_u32
    /// - clamp -> clamp_f32, clamp_i32, clamp_u32
    fn desugar_function_name(&self, name: &str, args: &[Expression]) -> Result<String> {
        match name {
            "mul" => self.desugar_mul(args),
            "matav" => self.desugar_matav(args),
            // Type-dispatched math functions (different GLSL opcodes for float/signed/unsigned)
            "abs" | "sign" | "min" | "max" | "clamp" => self.desugar_numeric_op(name, args),
            _ => Ok(name.to_string()),
        }
    }

    /// Desugar mul based on argument shapes
    fn desugar_mul(&self, args: &[Expression]) -> Result<String> {
        if args.len() != 2 {
            return Ok("mul".to_string()); // Let type checker handle the error
        }

        let arg1_ty = self.get_expr_type(&args[0]);
        let arg2_ty = self.get_expr_type(&args[1]);

        let shape1 = Self::classify_shape(&arg1_ty);
        let shape2 = Self::classify_shape(&arg2_ty);

        let variant = match (shape1, shape2) {
            (ArgShape::Matrix, ArgShape::Matrix) => "mul_mat_mat",
            (ArgShape::Matrix, ArgShape::Vector) => "mul_mat_vec",
            (ArgShape::Vector, ArgShape::Matrix) => "mul_vec_mat",
            _ => "mul", // Fall back to original name
        };

        Ok(variant.to_string())
    }

    /// Desugar matav based on array and vector dimensions
    fn desugar_matav(&self, args: &[Expression]) -> Result<String> {
        if args.len() != 1 {
            return Ok("matav".to_string()); // Let type checker handle the error
        }

        let arg_ty = self.get_expr_type(&args[0]);

        // Extract array size n, vector size m, and element type a from [n]vec<m,a>
        if let Type::Constructed(TypeName::Array, array_args) = &arg_ty {
            if array_args.len() >= 2 {
                if let Type::Constructed(TypeName::Vec, vec_args) = &array_args[1] {
                    if vec_args.len() >= 2 {
                        if let (Some(n), Some(m)) = (
                            Self::extract_size(&array_args[0]),
                            Self::extract_size(&vec_args[0]),
                        ) {
                            // Extract element type
                            let elem_type_str = Self::primitive_type_to_string(&vec_args[1])?;
                            return Ok(format!("matav_{}_{}_{}", n, m, elem_type_str));
                        }
                    }
                }
            }
        }

        Ok("matav".to_string()) // Fall back to original name
    }

    /// Desugar numeric operations (abs, sign, min, max, clamp) based on element type
    /// Transforms: abs x → f32.abs x (or i32.abs, etc. based on type)
    fn desugar_numeric_op(&self, name: &str, args: &[Expression]) -> Result<String> {
        if args.is_empty() {
            return Ok(name.to_string());
        }

        // Get the type of the first argument
        let arg_ty = self.get_expr_type(&args[0]);

        // Extract the element type (scalar or vector element)
        let elem_ty = Self::extract_element_type(&arg_ty);

        // Get the type prefix (f32, i32, u32, etc.)
        let type_prefix = Self::primitive_type_to_string(&elem_ty)?;

        // Produce qualified name: f32.abs, i32.min, etc.
        Ok(format!("{}.{}", type_prefix, name))
    }

    /// Extract the element type from a scalar or vector type
    fn extract_element_type(ty: &Type) -> Type {
        match ty {
            Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
                // vec<n, elem> -> elem
                args[1].clone()
            }
            _ => ty.clone(), // Scalar type, return as-is
        }
    }

    /// Classify argument shape for desugaring
    fn classify_shape(ty: &Type) -> ArgShape {
        match ty {
            Type::Constructed(TypeName::Mat, _) => ArgShape::Matrix,
            Type::Constructed(TypeName::Vec, _) => ArgShape::Vector,
            _ => ArgShape::Other,
        }
    }

    /// Extract concrete size from a type
    fn extract_size(ty: &Type) -> Option<usize> {
        match ty {
            Type::Constructed(TypeName::Size(n), _) => Some(*n),
            _ => None,
        }
    }

    /// Check if an expression is the integer literal 0
    fn is_zero(&self, expr: &ast::Expression) -> bool {
        matches!(expr.kind, ast::ExprKind::IntLiteral(0))
    }

    /// Convert a primitive numeric type to a string for name mangling.
    fn primitive_type_to_string(ty: &Type) -> Result<String> {
        match ty {
            Type::Constructed(TypeName::Float(bits), _) => Ok(format!("f{}", bits)),
            Type::Constructed(TypeName::Int(bits), _) => Ok(format!("i{}", bits)),
            Type::Constructed(TypeName::UInt(bits), _) => Ok(format!("u{}", bits)),
            Type::Constructed(TypeName::Str(s), _) if *s == "bool" => Ok("bool".to_string()),
            _ => Err(err_type!(
                "Invalid element type for matrix/vector: {:?}. \
                Only f16/f32/f64, i8/i16/i32/i64, u8/u16/u32/u64, and bool are supported.",
                ty
            )),
        }
    }

    /// Get the type of an AST pattern from the type table
    fn get_pattern_type(&self, pat: &ast::Pattern) -> Type {
        self.type_table
            .get(&pat.h.id)
            .and_then(|scheme| self.get_monotype(scheme))
            .cloned()
            .unwrap_or_else(|| {
                panic!("BUG: Pattern (id={:?}) has no type in type table during flattening. Type checking should ensure all patterns have types.", pat.h.id)
            })
    }

    /// Generate a unique ID
    fn fresh_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Generate a unique variable name
    fn fresh_name(&mut self, prefix: &str) -> String {
        format!("_w_{}{}", prefix, self.fresh_id())
    }

    // TODO(mir-refactor): Re-enable let hoisting as a separate pass
    // Hoist inner Let expressions out of a Let's value.
    // Transforms: let x = (let y = A in B) in C  =>  let y = A in let x = B in C
    // This ensures materialized pointers are at the same scope level as their referents.

    /// Resolve a field name to a numeric index using type information
    fn resolve_field_index(&self, obj: &Expression, field: &str) -> Result<usize> {
        // First try numeric index (for tuple access like .0, .1)
        if let Ok(idx) = field.parse::<usize>() {
            return Ok(idx);
        }

        // Look up the type of the object
        let scheme = self
            .type_table
            .get(&obj.h.id)
            .ok_or_else(|| err_flatten!("No type information for field access target"))?;

        let obj_type = self
            .get_monotype(scheme)
            .ok_or_else(|| err_flatten!("Could not extract monotype from scheme"))?;

        // Resolve based on type
        match obj_type {
            // Vector types: x=0, y=1, z=2, w=3
            Type::Constructed(TypeName::Vec, _) => match field {
                "x" => Ok(0),
                "y" => Ok(1),
                "z" => Ok(2),
                "w" => Ok(3),
                _ => Err(err_flatten!("Unknown vector field: {}", field)),
            },
            // Record types: look up field by name
            Type::Constructed(TypeName::Record(fields), _) => fields
                .iter()
                .enumerate()
                .find(|(_, name)| name.as_str() == field)
                .map(|(idx, _)| idx)
                .ok_or_else(|| err_flatten!("Unknown record field: {}", field)),
            // Tuple types: should use numeric access
            Type::Constructed(TypeName::Tuple(_), _) => Err(err_flatten!(
                "Tuple access must use numeric index, not '{}'",
                field
            )),
            _ => Err(err_flatten!(
                "Cannot access field '{}' on type {:?}",
                field,
                obj_type
            )),
        }
    }

    /// Helper to flatten a single Decl
    fn flatten_single_decl(&mut self, d: &ast::Decl, defs: &mut Vec<mir::Def>) -> Result<()> {
        self.enclosing_decl_stack.push(d.name.clone());
        let span = d.body.h.span;

        let def = if d.params.is_empty() {
            // Constant - create a new body
            let old_body = self.begin_body();
            self.name_to_local.push_scope();

            let (root_id, _) = self.flatten_expr(&d.body)?;
            self.current_body.set_root(root_id);

            self.name_to_local.pop_scope();
            let body = self.end_body(old_body);

            let ty = self.get_expr_type(&d.body);
            mir::Def::Constant {
                id: self.next_node_id(),
                name: d.name.clone(),
                ty,
                attributes: self.convert_attributes(&d.attributes),
                body,
                span,
            }
        } else {
            // Function - create a new body with params as locals
            let old_body = self.begin_body();
            self.name_to_local.push_scope();

            // Allocate params as locals
            let mut param_local_ids = Vec::new();
            for param in &d.params {
                let name = self.extract_param_name(param)?;
                let ty = self.get_pattern_type(param);
                let local_id = self.alloc_local(name, ty, LocalKind::Param, span);
                param_local_ids.push(local_id);
            }

            let (root_id, _) = self.flatten_expr(&d.body)?;
            self.current_body.set_root(root_id);

            self.name_to_local.pop_scope();
            let body = self.end_body(old_body);

            let ret_type = self.get_expr_type(&d.body);
            mir::Def::Function {
                id: self.next_node_id(),
                name: d.name.clone(),
                params: param_local_ids,
                ret_type,
                attributes: self.convert_attributes(&d.attributes),
                body,
                span,
            }
        };

        // Collect generated lambdas before the definition
        defs.append(&mut self.generated_functions);
        defs.push(def);

        self.enclosing_decl_stack.pop();
        Ok(())
    }

    /// Flatten a module declaration with a qualified name (e.g., "rand.init")
    pub fn flatten_module_decl(&mut self, d: &ast::Decl, qualified_name: &str) -> Result<Vec<mir::Def>> {
        self.enclosing_decl_stack.push(qualified_name.to_string());
        let span = d.body.h.span;

        let def = if d.params.is_empty() {
            // Constant - create a new body
            let old_body = self.begin_body();
            self.name_to_local.push_scope();

            let (root_id, _) = self.flatten_expr(&d.body)?;
            self.current_body.set_root(root_id);

            self.name_to_local.pop_scope();
            let body = self.end_body(old_body);

            let ty = self.get_expr_type(&d.body);
            mir::Def::Constant {
                id: self.next_node_id(),
                name: qualified_name.to_string(),
                ty,
                attributes: self.convert_attributes(&d.attributes),
                body,
                span,
            }
        } else {
            // Function - create a new body with params as locals
            let old_body = self.begin_body();
            self.name_to_local.push_scope();

            // Allocate params as locals
            let mut param_local_ids = Vec::new();
            for param in &d.params {
                let name = self.extract_param_name(param)?;
                let ty = self.get_pattern_type(param);
                let local_id = self.alloc_local(name, ty, LocalKind::Param, span);
                param_local_ids.push(local_id);
            }

            let (root_id, _) = self.flatten_expr(&d.body)?;
            self.current_body.set_root(root_id);

            self.name_to_local.pop_scope();
            let body = self.end_body(old_body);

            let ret_type = self.get_expr_type(&d.body);
            mir::Def::Function {
                id: self.next_node_id(),
                name: qualified_name.to_string(),
                params: param_local_ids,
                ret_type,
                attributes: self.convert_attributes(&d.attributes),
                body,
                span,
            }
        };

        // Collect generated lambdas before the definition
        let mut defs = Vec::new();
        defs.append(&mut self.generated_functions);
        defs.push(def);

        self.enclosing_decl_stack.pop();
        Ok(defs)
    }

    /// Flatten an entire program
    pub fn flatten_program(&mut self, program: &ast::Program) -> Result<mir::Program> {
        let mut defs = Vec::new();

        // Flatten user declarations
        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    self.builtins.insert(d.name.clone());
                    self.flatten_single_decl(d, &mut defs)?;
                }
                ast::Declaration::Entry(e) => {
                    self.enclosing_decl_stack.push(e.name.clone());
                    let span = e.body.h.span;

                    // Create a new body for the entry point
                    let old_body = self.begin_body();
                    self.name_to_local.push_scope();

                    // Allocate params as locals and build EntryInputs
                    let mut inputs = Vec::new();
                    for param in &e.params {
                        let name = self.extract_param_name(param).unwrap_or_default();
                        let ty = self.get_pattern_type(param);
                        let decoration = self.extract_io_decoration(param);
                        let local_id = self.alloc_local(name.clone(), ty.clone(), LocalKind::Param, span);
                        inputs.push(mir::EntryInput {
                            local: local_id,
                            name,
                            ty,
                            decoration,
                        });
                    }

                    let (root_id, _) = self.flatten_expr(&e.body)?;
                    self.current_body.set_root(root_id);

                    self.name_to_local.pop_scope();
                    let body = self.end_body(old_body);

                    // Convert entry type to ExecutionModel
                    let execution_model = match &e.entry_type {
                        ast::Attribute::Vertex => mir::ExecutionModel::Vertex,
                        ast::Attribute::Fragment => mir::ExecutionModel::Fragment,
                        ast::Attribute::Compute { local_size } => mir::ExecutionModel::Compute {
                            local_size: *local_size,
                        },
                        _ => panic!("Invalid entry type attribute: {:?}", e.entry_type),
                    };

                    // Convert AST EntryOutput to MIR EntryOutput with IoDecoration
                    let ret_type = self.get_expr_type(&e.body);
                    let outputs: Vec<mir::EntryOutput> =
                        if e.outputs.iter().all(|o| o.attribute.is_none()) && e.outputs.len() == 1 {
                            // Single output without explicit decoration
                            if !matches!(ret_type, polytype::Type::Constructed(ast::TypeName::Unit, _)) {
                                vec![mir::EntryOutput {
                                    ty: ret_type,
                                    decoration: None,
                                }]
                            } else {
                                vec![]
                            }
                        } else {
                            // Multiple outputs with decorations (tuple return)
                            if let polytype::Type::Constructed(ast::TypeName::Tuple(_), component_types) =
                                &ret_type
                            {
                                e.outputs
                                    .iter()
                                    .zip(component_types.iter())
                                    .map(|(output, ty)| mir::EntryOutput {
                                        ty: ty.clone(),
                                        decoration: output
                                            .attribute
                                            .as_ref()
                                            .and_then(|a| self.convert_to_io_decoration(a)),
                                    })
                                    .collect()
                            } else {
                                // Single output with decoration
                                vec![mir::EntryOutput {
                                    ty: ret_type,
                                    decoration: e
                                        .outputs
                                        .first()
                                        .and_then(|o| o.attribute.as_ref())
                                        .and_then(|a| self.convert_to_io_decoration(a)),
                                }]
                            }
                        };

                    let def = mir::Def::EntryPoint {
                        id: self.next_node_id(),
                        name: e.name.clone(),
                        execution_model,
                        inputs,
                        outputs,
                        body,
                        span,
                    };

                    defs.append(&mut self.generated_functions);
                    defs.push(def);

                    self.enclosing_decl_stack.pop();
                }
                ast::Declaration::Uniform(uniform_decl) => {
                    // Uniforms use the declared type directly (already a Type<TypeName>)
                    defs.push(mir::Def::Uniform {
                        id: self.next_node_id(),
                        name: uniform_decl.name.clone(),
                        ty: uniform_decl.ty.clone(),
                        set: uniform_decl.set,
                        binding: uniform_decl.binding,
                    });
                }
                ast::Declaration::Storage(storage_decl) => {
                    // Storage buffers use the declared type directly
                    defs.push(mir::Def::Storage {
                        id: self.next_node_id(),
                        name: storage_decl.name.clone(),
                        ty: storage_decl.ty.clone(),
                        set: storage_decl.set,
                        binding: storage_decl.binding,
                        layout: storage_decl.layout,
                        access: storage_decl.access,
                    });
                }
                ast::Declaration::Sig(_)
                | ast::Declaration::TypeBind(_)
                | ast::Declaration::ModuleBind(_)
                | ast::Declaration::ModuleTypeBind(_)
                | ast::Declaration::Open(_)
                | ast::Declaration::Import(_) => {
                    // Skip declarations that don't produce MIR defs
                }
            }
        }

        Ok(mir::Program {
            defs,
            lambda_registry: self.lambda_registry.clone(),
        })
    }

    /// Convert AST attributes to MIR attributes
    fn convert_attributes(&self, attrs: &[ast::Attribute]) -> Vec<mir::Attribute> {
        attrs.iter().map(|a| self.convert_attribute(a)).collect()
    }

    /// Convert a single AST attribute to MIR attribute
    fn convert_attribute(&self, attr: &ast::Attribute) -> mir::Attribute {
        match attr {
            ast::Attribute::BuiltIn(builtin) => mir::Attribute::BuiltIn(*builtin),
            ast::Attribute::Location(loc) => mir::Attribute::Location(*loc),
            ast::Attribute::Vertex => mir::Attribute::Vertex,
            ast::Attribute::Fragment => mir::Attribute::Fragment,
            ast::Attribute::Compute { local_size } => mir::Attribute::Compute {
                local_size: *local_size,
            },
            // The binding is stored in Def::Uniform, not the Attribute
            ast::Attribute::Uniform { .. } => mir::Attribute::Uniform,
            // The binding is stored in Def::Storage, not the Attribute
            ast::Attribute::Storage { .. } => mir::Attribute::Storage,
        }
    }

    /// Convert an AST attribute to IoDecoration (only Location and BuiltIn are valid)
    fn convert_to_io_decoration(&self, attr: &ast::Attribute) -> Option<mir::IoDecoration> {
        match attr {
            ast::Attribute::BuiltIn(builtin) => Some(mir::IoDecoration::BuiltIn(*builtin)),
            ast::Attribute::Location(loc) => Some(mir::IoDecoration::Location(*loc)),
            _ => None,
        }
    }

    /// Extract IoDecoration from a pattern (for entry point parameters)
    fn extract_io_decoration(&self, pattern: &ast::Pattern) -> Option<mir::IoDecoration> {
        match &pattern.kind {
            PatternKind::Attributed(attrs, inner) => {
                // Look for Location or BuiltIn in attributes
                for attr in attrs {
                    if let Some(dec) = self.convert_to_io_decoration(attr) {
                        return Some(dec);
                    }
                }
                // Recurse into inner pattern
                self.extract_io_decoration(inner)
            }
            PatternKind::Typed(inner, _) => self.extract_io_decoration(inner),
            _ => None,
        }
    }

    // TODO(mir-refactor): Old backing store functions removed - backing stores are now
    // handled inline in flatten_let_in when needed

    /// Extract parameter name from pattern
    fn extract_param_name(&self, pattern: &ast::Pattern) -> Result<String> {
        match &pattern.kind {
            PatternKind::Name(name) => Ok(name.clone()),
            PatternKind::Typed(inner, _) => self.extract_param_name(inner),
            PatternKind::Attributed(_, inner) => self.extract_param_name(inner),
            _ => Err(err_flatten!("Complex parameter patterns not yet supported")),
        }
    }

    /// Flatten an expression, returning the MIR ExprId and its static value
    fn flatten_expr(&mut self, expr: &Expression) -> Result<(ExprId, StaticValue)> {
        let span = expr.h.span;
        let ty = self.get_expr_type(expr);

        let (mir_expr, sv) = match &expr.kind {
            ExprKind::IntLiteral(n) => (mir::Expr::Int(n.to_string()), StaticValue::Dyn),
            ExprKind::FloatLiteral(f) => (mir::Expr::Float(f.to_string()), StaticValue::Dyn),
            ExprKind::BoolLiteral(b) => (mir::Expr::Bool(*b), StaticValue::Dyn),
            ExprKind::StringLiteral(s) => (mir::Expr::String(s.clone()), StaticValue::Dyn),
            ExprKind::Unit => (mir::Expr::Unit, StaticValue::Dyn),

            ExprKind::Identifier(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };

                // Get classification from DefunAnalysis
                let sv = self.get_classification(expr.h.id);

                // Check if it's a local variable or a global reference
                let mir_expr = if let Some(local_id) = self.lookup_local(&full_name) {
                    mir::Expr::Local(local_id)
                } else {
                    mir::Expr::Global(full_name)
                };
                (mir_expr, sv)
            }

            ExprKind::BinaryOp(op, lhs, rhs) => {
                let (lhs_id, _) = self.flatten_expr(lhs)?;
                let (rhs_id, _) = self.flatten_expr(rhs)?;
                (
                    mir::Expr::BinOp {
                        op: op.op.clone(),
                        lhs: lhs_id,
                        rhs: rhs_id,
                    },
                    StaticValue::Dyn,
                )
            }

            ExprKind::UnaryOp(op, operand) => {
                let (operand_id, _) = self.flatten_expr(operand)?;
                (
                    mir::Expr::UnaryOp {
                        op: op.op.clone(),
                        operand: operand_id,
                    },
                    StaticValue::Dyn,
                )
            }

            ExprKind::If(if_expr) => {
                let (cond_id, _) = self.flatten_expr(&if_expr.condition)?;
                let (then_id, _) = self.flatten_expr(&if_expr.then_branch)?;
                let (else_id, _) = self.flatten_expr(&if_expr.else_branch)?;
                (
                    mir::Expr::If {
                        cond: cond_id,
                        then_: then_id,
                        else_: else_id,
                    },
                    StaticValue::Dyn,
                )
            }

            ExprKind::LetIn(let_in) => return self.flatten_let_in(let_in, span, &ty),
            ExprKind::Lambda(lambda) => return self.flatten_lambda(lambda, expr.h.id, span),
            ExprKind::Application(func, args) => return self.flatten_application(func, args, &ty, span),

            ExprKind::Tuple(elems) => {
                let elem_ids: Result<Vec<_>> =
                    elems.iter().map(|e| self.flatten_expr(e).map(|(id, _)| id)).collect();
                (mir::Expr::Tuple(elem_ids?), StaticValue::Dyn)
            }
            ExprKind::ArrayLiteral(elems) => {
                let elem_ids: Result<Vec<_>> =
                    elems.iter().map(|e| self.flatten_expr(e).map(|(id, _)| id)).collect();
                (mir::Expr::Array(elem_ids?), StaticValue::Dyn)
            }
            ExprKind::VecMatLiteral(elems) => {
                // Check if first element is an array literal (matrix) or scalar (vector)
                if elems.is_empty() {
                    bail_flatten!("Empty vector/matrix literal");
                }

                let is_matrix = matches!(&elems[0].kind, ExprKind::ArrayLiteral(_));

                if is_matrix {
                    // Matrix: extract rows
                    let mut rows = Vec::new();
                    for elem in elems {
                        if let ExprKind::ArrayLiteral(row_elems) = &elem.kind {
                            let row: Result<Vec<_>> =
                                row_elems.iter().map(|e| self.flatten_expr(e).map(|(id, _)| id)).collect();
                            rows.push(row?);
                        } else {
                            bail_flatten!("Matrix rows must be array literals");
                        }
                    }
                    (mir::Expr::Matrix(rows), StaticValue::Dyn)
                } else {
                    // Vector
                    let elem_ids: Result<Vec<_>> =
                        elems.iter().map(|e| self.flatten_expr(e).map(|(id, _)| id)).collect();
                    (mir::Expr::Vector(elem_ids?), StaticValue::Dyn)
                }
            }
            ExprKind::RecordLiteral(fields) => {
                // Records become tuples with fields in source order
                let elem_ids: Result<Vec<_>> =
                    fields.iter().map(|(_, expr)| Ok(self.flatten_expr(expr)?.0)).collect();
                (mir::Expr::Tuple(elem_ids?), StaticValue::Dyn)
            }
            ExprKind::ArrayIndex(arr_expr, idx_expr) => {
                let (arr_id, _) = self.flatten_expr(arr_expr)?;
                let (idx_id, _) = self.flatten_expr(idx_expr)?;

                // Check if index is a constant - if so, use tuple_access (OpCompositeExtract)
                // which doesn't need a backing store
                if let mir::Expr::Int(_) = self.current_body.get_expr(idx_id) {
                    // Constant index: use tuple_access directly on value
                    let intrinsic = mir::Expr::Intrinsic {
                        name: "tuple_access".to_string(),
                        args: vec![arr_id, idx_id],
                    };
                    let id = self.alloc_expr(intrinsic, ty, span);
                    return Ok((id, StaticValue::Dyn));
                }

                // Dynamic index: need backing store for OpAccessChain
                if let mir::Expr::Local(local_id) = self.current_body.get_expr(arr_id) {
                    let local_id = *local_id;
                    // Mark this binding as needing a backing store
                    self.needs_backing_store.insert(local_id);
                    // Use the backing store variable name
                    let ptr_name = Self::backing_store_name(local_id);
                    // Look up or create the backing store local
                    if let Some(ptr_local_id) = self.lookup_local(&ptr_name) {
                        let intrinsic = mir::Expr::Intrinsic {
                            name: "index".to_string(),
                            args: vec![
                                self.alloc_expr(
                                    mir::Expr::Local(ptr_local_id),
                                    types::pointer(self.current_body.get_type(arr_id).clone()),
                                    span,
                                ),
                                idx_id,
                            ],
                        };
                        let id = self.alloc_expr(intrinsic, ty, span);
                        return Ok((id, StaticValue::Dyn));
                    }
                }

                // Fallback for dynamic index on complex expression: wrap in Materialize
                let arr_ty = self.current_body.get_type(arr_id).clone();
                let materialized = mir::Expr::Materialize(arr_id);
                let materialized_id = self.alloc_expr(materialized, types::pointer(arr_ty), span);
                (
                    mir::Expr::Intrinsic {
                        name: "index".to_string(),
                        args: vec![materialized_id, idx_id],
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::ArrayWith { array, index, value } => {
                // Flatten array with syntax to a call to _w_array_with intrinsic
                // _w_array_with : [n]a -> i32 -> a -> [n]a
                let (arr_id, _) = self.flatten_expr(array)?;
                let (idx_id, _) = self.flatten_expr(index)?;
                let (val_id, _) = self.flatten_expr(value)?;

                // Generate a call to _w_array_with(arr, idx, val)
                (
                    mir::Expr::Call {
                        func: "_w_array_with".to_string(),
                        args: vec![arr_id, idx_id, val_id],
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::FieldAccess(obj_expr, field) => {
                let (obj_id, _) = self.flatten_expr(obj_expr)?;

                // Resolve field name to index using type information
                let idx = self.resolve_field_index(obj_expr, field)?;

                // Create i32 type for the index literal
                let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
                let idx_expr = mir::Expr::Int(idx.to_string());
                let idx_id = self.alloc_expr(idx_expr, i32_type, span);

                // Pass value directly to tuple_access - no Materialize/backing store needed
                // Lowering handles both pointer and value inputs correctly
                (
                    mir::Expr::Intrinsic {
                        name: "tuple_access".to_string(),
                        args: vec![obj_id, idx_id],
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::Loop(loop_expr) => return self.flatten_loop(loop_expr, &ty, span),
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                // Type annotations don't affect runtime, just flatten inner
                return self.flatten_expr(inner);
            }
            ExprKind::Assert(cond, body) => {
                let (cond_id, _) = self.flatten_expr(cond)?;
                let (body_id, _) = self.flatten_expr(body)?;
                (
                    mir::Expr::Intrinsic {
                        name: "assert".to_string(),
                        args: vec![cond_id, body_id],
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::TypeHole => {
                bail_flatten!("Type holes should be resolved before flattening");
            }
            ExprKind::Match(_) => {
                bail_flatten!("Match expressions not yet supported");
            }
            ExprKind::Range(range) => {
                let (start_id, _) = self.flatten_expr(&range.start)?;
                let step_id = match &range.step {
                    Some(s) => {
                        let (step_id, _) = self.flatten_expr(s)?;
                        Some(step_id)
                    }
                    None => None,
                };
                let (end_id, _) = self.flatten_expr(&range.end)?;
                let kind = match range.kind {
                    ast::RangeKind::Inclusive => mir::RangeKind::Inclusive,
                    ast::RangeKind::Exclusive => mir::RangeKind::Exclusive,
                    ast::RangeKind::ExclusiveLt => mir::RangeKind::ExclusiveLt,
                    ast::RangeKind::ExclusiveGt => mir::RangeKind::ExclusiveGt,
                };
                (
                    mir::Expr::Range {
                        start: start_id,
                        step: step_id,
                        end: end_id,
                        kind,
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::Slice(_) => {
                bail_flatten!("Slice expressions should be desugared before flattening");
            }
        };

        let id = self.alloc_expr(mir_expr, ty, span);
        Ok((id, sv))
    }

    /// Flatten a let-in expression, handling pattern destructuring
    fn flatten_let_in(
        &mut self,
        let_in: &ast::LetInExpr,
        span: Span,
        result_ty: &Type,
    ) -> Result<(ExprId, StaticValue)> {
        // Strip Typed wrappers from pattern first (before flattening value)
        // This prevents double-flattening when recursing through Typed patterns
        let pattern = Self::unwrap_typed_pattern(&let_in.pattern);

        let (value_id, _) = self.flatten_expr(&let_in.value)?;
        let value_ty = self.current_body.get_type(value_id).clone();

        // Handle the unwrapped pattern
        match &pattern.kind {
            PatternKind::Name(name) => {
                // Allocate local for this binding
                self.name_to_local.push_scope();
                let local_id = self.alloc_local(name.clone(), value_ty.clone(), LocalKind::Let, span);

                let (body_id, body_sv) = self.flatten_expr(&let_in.body)?;

                self.name_to_local.pop_scope();

                // Check if this binding needs a backing store
                let body_id = if self.needs_backing_store.contains(&local_id) {
                    // Wrap body with backing store materialization:
                    // let _w_ptr_{id} = materialize(name) in body
                    let ptr_name = Self::backing_store_name(local_id);
                    let var_id = self.alloc_expr(mir::Expr::Local(local_id), value_ty.clone(), span);
                    let materialize_id = self.alloc_expr(
                        mir::Expr::Materialize(var_id),
                        types::pointer(value_ty.clone()),
                        span,
                    );
                    let ptr_local_id =
                        self.alloc_local(ptr_name, types::pointer(value_ty), LocalKind::Let, span);
                    let body_ty = self.current_body.get_type(body_id).clone();
                    self.alloc_expr(
                        mir::Expr::Let {
                            local: ptr_local_id,
                            rhs: materialize_id,
                            body: body_id,
                        },
                        body_ty,
                        span,
                    )
                } else {
                    body_id
                };

                // Create the let expression
                let let_expr = mir::Expr::Let {
                    local: local_id,
                    rhs: value_id,
                    body: body_id,
                };
                let id = self.alloc_expr(let_expr, result_ty.clone(), span);
                Ok((id, body_sv))
            }
            PatternKind::Typed(_, _) => {
                unreachable!("Typed patterns should be stripped before reaching here")
            }
            PatternKind::Wildcard => {
                // Bind to ignored variable, just for side effects
                let ignored_name = self.fresh_name("ignored");
                let local_id = self.alloc_local(ignored_name, value_ty, LocalKind::Let, span);
                let (body_id, body_sv) = self.flatten_expr(&let_in.body)?;
                let let_expr = mir::Expr::Let {
                    local: local_id,
                    rhs: value_id,
                    body: body_id,
                };
                let id = self.alloc_expr(let_expr, result_ty.clone(), span);
                Ok((id, body_sv))
            }
            PatternKind::Tuple(patterns) => {
                // Generate a temp name for the tuple value
                let tmp = self.fresh_name("tup");

                // Get the tuple type and element types
                let tuple_ty = self.get_pattern_type(&let_in.pattern);
                let elem_types: Vec<Type> = match &tuple_ty {
                    Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
                    _ => {
                        // Fallback: use unknown types
                        patterns.iter().map(|p| self.get_pattern_type(p)).collect()
                    }
                };

                // Allocate local for the tuple
                let tuple_local_id = self.alloc_local(tmp, tuple_ty.clone(), LocalKind::Let, span);

                // Push scope for pattern bindings
                self.name_to_local.push_scope();

                // FIRST: Collect pattern info and allocate locals for each element
                // This ensures pattern-bound variables are in scope when we flatten the body
                let mut elem_info = Vec::new();
                for (i, pat) in patterns.iter().enumerate() {
                    let name = match &pat.kind {
                        PatternKind::Name(n) => Some(n.clone()),
                        PatternKind::Typed(inner, _) => match &inner.kind {
                            PatternKind::Name(n) => Some(n.clone()),
                            _ => {
                                bail_flatten!("Nested complex patterns not supported");
                            }
                        },
                        PatternKind::Wildcard => None, // Skip wildcards
                        _ => {
                            bail_flatten!("Complex nested patterns not supported");
                        }
                    };

                    let elem_ty = elem_types
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| {
                            panic!("BUG: Tuple pattern element {} has no type. Type checking should ensure all tuple elements have types.", i)
                        });

                    // Allocate local for this element (adds to name_to_local)
                    let elem_local_id = name
                        .as_ref()
                        .map(|n| self.alloc_local(n.clone(), elem_ty.clone(), LocalKind::Let, span));

                    elem_info.push((i, name, elem_ty, elem_local_id));
                }

                // NOW flatten the body - pattern-bound variables are now in scope
                let (mut body_id, body_sv) = self.flatten_expr(&let_in.body)?;

                // Build nested lets from inside out (reverse order)
                let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
                for (i, name, elem_ty, elem_local_id) in elem_info.into_iter().rev() {
                    // Skip wildcards
                    let (Some(_name), Some(elem_local_id)) = (name, elem_local_id) else {
                        continue;
                    };

                    // Create tuple_access intrinsic
                    let tuple_var_id =
                        self.alloc_expr(mir::Expr::Local(tuple_local_id), tuple_ty.clone(), span);
                    let idx_id = self.alloc_expr(mir::Expr::Int(i.to_string()), i32_type.clone(), span);
                    let extract_id = self.alloc_expr(
                        mir::Expr::Intrinsic {
                            name: "tuple_access".to_string(),
                            args: vec![tuple_var_id, idx_id],
                        },
                        elem_ty,
                        span,
                    );

                    // Wrap body in let
                    let body_ty = self.current_body.get_type(body_id).clone();
                    body_id = self.alloc_expr(
                        mir::Expr::Let {
                            local: elem_local_id,
                            rhs: extract_id,
                            body: body_id,
                        },
                        body_ty,
                        span,
                    );
                }

                self.name_to_local.pop_scope();

                // Wrap with the tuple binding
                let let_expr = mir::Expr::Let {
                    local: tuple_local_id,
                    rhs: value_id,
                    body: body_id,
                };
                let id = self.alloc_expr(let_expr, result_ty.clone(), span);
                Ok((id, body_sv))
            }
            _ => Err(err_flatten!(
                "Pattern kind {:?} not yet supported in let",
                let_in.pattern.kind
            )),
        }
    }

    /// Flatten a lambda expression (defunctionalization)
    fn flatten_lambda(
        &mut self,
        lambda: &ast::LambdaExpr,
        node_id: NodeId,
        span: Span,
    ) -> Result<(ExprId, StaticValue)> {
        // Look up pre-computed analysis for this lambda
        let analysis = self.defun_analysis.get_or_panic(node_id);
        let (func_name, free_vars) = match analysis {
            crate::defun_analysis::StaticValue::Closure { lam_name, free_vars } => {
                (lam_name.clone(), free_vars.clone())
            }
            crate::defun_analysis::StaticValue::Dyn => {
                panic!("BUG: Lambda expression classified as Dyn in defun analysis")
            }
        };

        // Register lambda in the runtime registry
        let arity = lambda.params.len();
        self.add_lambda(func_name.clone(), arity);

        // Build closure captures - look up each free var and get its ExprId
        let mut capture_ids = vec![];
        let mut capture_types = vec![];
        for (var_name, var_type) in &free_vars {
            let capture_id = if let Some(local_id) = self.lookup_local(var_name) {
                self.alloc_expr(mir::Expr::Local(local_id), var_type.clone(), span)
            } else {
                self.alloc_expr(mir::Expr::Global(var_name.clone()), var_type.clone(), span)
            };
            capture_ids.push(capture_id);
            capture_types.push(var_type.clone());
        }

        // The closure type is the captures tuple
        let closure_type = types::tuple(capture_types.clone());

        // Create a new body for the generated function
        let outer_body = self.begin_body();
        self.name_to_local.push_scope();

        // Allocate closure param as local
        let closure_local = self.alloc_local(
            "_w_closure".to_string(),
            closure_type.clone(),
            LocalKind::Param,
            span,
        );
        let mut param_local_ids = vec![closure_local];

        // Allocate lambda params as locals
        for param in &lambda.params {
            let name = param
                .simple_name()
                .ok_or_else(|| err_flatten!("Complex lambda parameter patterns not supported"))?
                .to_string();
            let ty = self.get_pattern_type(param);
            let local_id = self.alloc_local(name, ty, LocalKind::Param, span);
            param_local_ids.push(local_id);
        }

        // Allocate locals for captured variables BEFORE flattening body
        // This ensures body references resolve to the lambda's locals, not the parent's
        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
        let mut capture_locals = vec![];
        for (var_name, var_type) in &free_vars {
            let capture_local = self.alloc_local(var_name.clone(), var_type.clone(), LocalKind::Let, span);
            capture_locals.push(capture_local);
        }

        // Now flatten the lambda body - captured vars will resolve to our new locals
        let (mut body_root, _) = self.flatten_expr(&lambda.body)?;

        // Wrap body with let bindings for captured vars (in reverse order)
        for (idx, ((_var_name, var_type), capture_local)) in free_vars.iter().zip(capture_locals.iter()).enumerate().rev() {
            let closure_ref = self.alloc_expr(mir::Expr::Local(closure_local), closure_type.clone(), span);
            let idx_expr = self.alloc_expr(mir::Expr::Int(idx.to_string()), i32_type.clone(), span);
            let extract = self.alloc_expr(
                mir::Expr::Intrinsic {
                    name: "tuple_access".to_string(),
                    args: vec![closure_ref, idx_expr],
                },
                var_type.clone(),
                span,
            );
            let body_ty = self.current_body.get_type(body_root).clone();
            body_root = self.alloc_expr(
                mir::Expr::Let {
                    local: *capture_local,
                    rhs: extract,
                    body: body_root,
                },
                body_ty,
                span,
            );
        }

        self.current_body.set_root(body_root);
        self.name_to_local.pop_scope();
        let func_body = self.end_body(outer_body);

        let ret_type = func_body.get_type(func_body.root).clone();

        // Create the generated function
        let func = mir::Def::Function {
            id: self.next_node_id(),
            name: func_name.clone(),
            params: param_local_ids,
            ret_type,
            attributes: vec![],
            body: func_body,
            span,
        };
        self.generated_functions.push(func);

        // Return the closure
        let sv = StaticValue::Closure {
            lam_name: func_name.clone(),
            free_vars: free_vars.clone(),
        };

        // Build closure tuple
        let closure_tuple =
            if capture_ids.is_empty() { mir::Expr::Unit } else { mir::Expr::Tuple(capture_ids.clone()) };
        let closure_tuple_id = self.alloc_expr(closure_tuple, closure_type.clone(), span);

        let closure_expr = mir::Expr::Closure {
            lambda_name: func_name,
            captures: vec![closure_tuple_id],
        };
        let closure_result_type = closure_type; // Closure is represented as its captures tuple
        let id = self.alloc_expr(closure_expr, closure_result_type, span);
        Ok((id, sv))
    }

    /// Check if a type is an Arrow type and return the (param, result) if so
    fn as_arrow_type(ty: &Type) -> Option<(&Type, &Type)> {
        types::as_arrow(ty)
    }

    // TODO(mir-refactor): wrap_body_with_closure_bindings integrated into flatten_lambda

    // TODO(mir-refactor): synthesize_partial_application removed - partial application is rejected by type checker

    /// Flatten an application expression
    fn flatten_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
        result_type: &Type,
        span: Span,
    ) -> Result<(ExprId, StaticValue)> {
        let (func_id, func_sv) = self.flatten_expr(func)?;

        // Flatten arguments while keeping static values for closure detection
        let args_with_sv: Result<Vec<_>> = args.iter().map(|a| self.flatten_expr(a)).collect();
        let args_with_sv = args_with_sv?;
        let arg_ids: Vec<_> = args_with_sv.iter().map(|(id, _)| *id).collect();

        // Check if this is applying a known function name
        let call_expr = match &func.kind {
            ExprKind::Identifier(quals, name) => {
                // Check if the identifier is bound to a known closure
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    // Direct call to the lambda function with closure as first argument
                    let mut all_args = vec![func_id];
                    all_args.extend(arg_ids);
                    mir::Expr::Call {
                        func: lam_name,
                        args: all_args,
                    }
                } else {
                    let full_name = if quals.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", quals.join("."), name)
                    };

                    // Desugar overloaded functions based on argument types
                    let desugared_name = self.desugar_function_name(&full_name, args)?;

                    // Check if this is a partial application (result type is Arrow)
                    if Self::as_arrow_type(result_type).is_some() {
                        // Partial application should be rejected by type checker
                        panic!(
                            "partial application should be rejected by type checker: function '{}' with result type {:?}, args count: {}, func expr id: {:?}",
                            full_name,
                            result_type,
                            args.len(),
                            func.h.id
                        );
                    } else {
                        // Direct function call (not a closure)
                        mir::Expr::Call {
                            func: desugared_name,
                            args: arg_ids,
                        }
                    }
                }
            }
            // FieldAccess in application position - must be a closure
            ExprKind::FieldAccess(_, _) => {
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    let mut all_args = vec![func_id];
                    all_args.extend(arg_ids);
                    mir::Expr::Call {
                        func: lam_name,
                        args: all_args,
                    }
                } else {
                    return Err(err_flatten!(
                        "Cannot call closure with unknown static value (field access). \
                         Function expression: {:?}",
                        func.kind
                    ));
                }
            }
            _ => {
                // Closure call: check if we know the static value
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    // Direct call to the lambda function with closure as first argument
                    let mut all_args = vec![func_id];
                    all_args.extend(arg_ids);
                    mir::Expr::Call {
                        func: lam_name,
                        args: all_args,
                    }
                } else {
                    // Unknown closure - this should not happen with proper function value restrictions
                    return Err(err_flatten!(
                        "Cannot call closure with unknown static value. \
                         Function expression: {:?}",
                        func.kind
                    ));
                }
            }
        };

        let id = self.alloc_expr(call_expr, result_type.clone(), span);
        Ok((id, StaticValue::Dyn))
    }

    /// Flatten a loop expression
    fn flatten_loop(
        &mut self,
        loop_expr: &ast::LoopExpr,
        result_ty: &Type,
        span: Span,
    ) -> Result<(ExprId, StaticValue)> {
        // Extract loop_var LocalId, init ExprId, and bindings from pattern
        let (loop_var_local, init_id, init_bindings) =
            self.extract_loop_bindings(&loop_expr.pattern, loop_expr.init.as_deref(), span)?;

        // Flatten loop kind - now uses LocalId for loop vars and ExprId for expressions
        let kind = match &loop_expr.form {
            ast::LoopForm::While(cond) => {
                let (cond_id, _) = self.flatten_expr(cond)?;
                mir::LoopKind::While { cond: cond_id }
            }
            ast::LoopForm::For(var_name, bound) => {
                let (bound_id, _) = self.flatten_expr(bound)?;
                let bound_ty = self.current_body.get_type(bound_id).clone();
                // Get the element type for the loop variable (i32 for range bounds)
                let var_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let var_local = self.alloc_local(var_name.clone(), var_ty, LocalKind::LoopVar, span);
                mir::LoopKind::ForRange {
                    var: var_local,
                    bound: bound_id,
                }
            }
            ast::LoopForm::ForIn(pat, iter) => {
                let var_name = match &pat.kind {
                    PatternKind::Name(n) => n.clone(),
                    _ => {
                        bail_flatten!("Complex for-in patterns not supported");
                    }
                };
                let (iter_id, _) = self.flatten_expr(iter)?;
                // Get element type from array type
                let iter_ty = self.current_body.get_type(iter_id).clone();
                let elem_ty = match &iter_ty {
                    Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
                    _ => iter_ty.clone(), // Fallback
                };
                let var_local = self.alloc_local(var_name, elem_ty, LocalKind::LoopVar, span);
                mir::LoopKind::For {
                    var: var_local,
                    iter: iter_id,
                }
            }
        };

        let (body_id, _) = self.flatten_expr(&loop_expr.body)?;

        let loop_expr = mir::Expr::Loop {
            loop_var: loop_var_local,
            init: init_id,
            init_bindings,
            kind,
            body: body_id,
        };
        let id = self.alloc_expr(loop_expr, result_ty.clone(), span);
        Ok((id, StaticValue::Dyn))
    }

    /// Extract loop_var LocalId, init ExprId, and bindings from pattern and init expression.
    /// Returns (loop_var_local, init_id, bindings) where bindings extract from loop_var.
    fn extract_loop_bindings(
        &mut self,
        pattern: &ast::Pattern,
        init: Option<&Expression>,
        span: Span,
    ) -> Result<(LocalId, ExprId, Vec<(LocalId, ExprId)>)> {
        let init_expr = init.ok_or_else(|| err_flatten!("Loop must have init expression"))?;

        let (init_id, _) = self.flatten_expr(init_expr)?;
        let init_ty = self.current_body.get_type(init_id).clone();
        let loop_var_name = self.fresh_name("loop_var");
        let loop_var_local =
            self.alloc_local(loop_var_name.clone(), init_ty.clone(), LocalKind::LoopVar, span);

        let bindings = match &pattern.kind {
            PatternKind::Name(name) => {
                // Single variable: binding is just identity (Local(loop_var))
                let binding_local = self.alloc_local(name.clone(), init_ty.clone(), LocalKind::Let, span);
                let binding_expr = self.alloc_expr(mir::Expr::Local(loop_var_local), init_ty, span);
                vec![(binding_local, binding_expr)]
            }
            PatternKind::Typed(inner, _) => {
                // Unwrap type annotation and recurse
                self.extract_bindings_from_pattern(inner, loop_var_local, &init_ty, span)?
            }
            PatternKind::Tuple(patterns) => {
                self.extract_tuple_bindings(patterns, loop_var_local, &init_ty, span)?
            }
            PatternKind::Wildcard => {
                // Wildcard pattern: no bindings needed, value is discarded
                vec![]
            }
            _ => {
                bail_flatten!("Loop pattern {:?} not supported", pattern.kind);
            }
        };

        Ok((loop_var_local, init_id, bindings))
    }

    /// Helper to extract bindings from pattern given loop_var LocalId and init_ty
    fn extract_bindings_from_pattern(
        &mut self,
        pattern: &ast::Pattern,
        loop_var_local: LocalId,
        init_ty: &Type,
        span: Span,
    ) -> Result<Vec<(LocalId, ExprId)>> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                let binding_local = self.alloc_local(name.clone(), init_ty.clone(), LocalKind::Let, span);
                let binding_expr = self.alloc_expr(mir::Expr::Local(loop_var_local), init_ty.clone(), span);
                Ok(vec![(binding_local, binding_expr)])
            }
            PatternKind::Typed(inner, _) => {
                self.extract_bindings_from_pattern(inner, loop_var_local, init_ty, span)
            }
            PatternKind::Tuple(patterns) => {
                self.extract_tuple_bindings(patterns, loop_var_local, init_ty, span)
            }
            PatternKind::Wildcard => Ok(vec![]), // Wildcard: no bindings
            _ => Err(err_flatten!("Loop pattern {:?} not supported", pattern.kind)),
        }
    }

    /// Extract bindings for tuple pattern
    fn extract_tuple_bindings(
        &mut self,
        patterns: &[ast::Pattern],
        loop_var_local: LocalId,
        tuple_ty: &Type,
        span: Span,
    ) -> Result<Vec<(LocalId, ExprId)>> {
        // Get element types from tuple type
        let elem_types: Vec<Type> = match tuple_ty {
            Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
            _ => {
                bail_flatten!("Expected tuple type for tuple pattern, got {:?}", tuple_ty);
            }
        };

        let mut bindings = Vec::new();
        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);

        for (i, pat) in patterns.iter().enumerate() {
            let name = match &pat.kind {
                PatternKind::Name(n) => n.clone(),
                PatternKind::Typed(inner, _) => match &inner.kind {
                    PatternKind::Name(n) => n.clone(),
                    PatternKind::Wildcard => continue, // Skip wildcards in typed patterns
                    _ => {
                        bail_flatten!("Complex loop patterns not supported");
                    }
                },
                PatternKind::Wildcard => continue, // Skip wildcards - no binding needed
                _ => {
                    bail_flatten!("Complex loop patterns not supported");
                }
            };

            let elem_ty = elem_types
                .get(i)
                .cloned()
                .ok_or_else(|| err_flatten!("Tuple pattern element {} has no corresponding type", i))?;

            // Create tuple_access intrinsic to extract element
            let loop_var_expr = self.alloc_expr(mir::Expr::Local(loop_var_local), tuple_ty.clone(), span);
            let idx_expr = self.alloc_expr(mir::Expr::Int(i.to_string()), i32_type.clone(), span);
            let extract_id = self.alloc_expr(
                mir::Expr::Intrinsic {
                    name: "tuple_access".to_string(),
                    args: vec![loop_var_expr, idx_expr],
                },
                elem_ty.clone(),
                span,
            );

            let binding_local = self.alloc_local(name, elem_ty, LocalKind::Let, span);
            bindings.push((binding_local, extract_id));
        }

        Ok(bindings)
    }
}
