//! MIR (Mid-level Intermediate Representation) for the Wyn compiler.
//!
//! This representation uses an arena-based approach where:
//! - Expressions are stored in a flat arena indexed by `ExprId`
//! - Local variables are tracked in a separate table indexed by `LocalId`
//! - Types, spans, and NodeIds are stored in parallel vectors
//!
//! Assumptions:
//! - Type checking has already occurred; concrete types are stored with expressions
//! - Patterns have been flattened to simple let bindings
//! - Lambdas have been lifted to top-level functions
//! - Imports and namespacing have been resolved
//! - Range expressions have been desugared

use crate::IdArena;
use crate::ast::{NodeId, Span, TypeName};
use crate::types::TypeScheme;
use polytype::Type;

pub mod parallelism;

#[cfg(test)]
mod tests;

// =============================================================================
// ID Types
// =============================================================================

/// Index into the expression arena within a Body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

impl ExprId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for ExprId {
    fn from(id: u32) -> Self {
        ExprId(id)
    }
}

/// Index into the locals table within a Body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

impl LocalId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for LocalId {
    fn from(id: u32) -> Self {
        LocalId(id)
    }
}

/// Unique identifier for a lambda/closure in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LambdaId(pub u32);

impl From<u32> for LambdaId {
    fn from(id: u32) -> Self {
        LambdaId(id)
    }
}

// =============================================================================
// Local Variable Tracking
// =============================================================================

/// The kind of local variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalKind {
    /// Function parameter
    Param,
    /// Let binding (includes compiler-generated temps)
    Let,
    /// Loop accumulator variable
    LoopVar,
}

/// Declaration of a local variable.
#[derive(Debug, Clone)]
pub struct LocalDecl {
    /// Variable name (for debugging/display).
    pub name: String,
    /// Source location.
    pub span: Span,
    /// Type of this local.
    pub ty: Type<TypeName>,
    /// What kind of local this is.
    pub kind: LocalKind,
}

// =============================================================================
// Expression Arena
// =============================================================================

/// A flat expression in the MIR arena.
/// All nested expressions are referenced by ExprId.
#[derive(Debug, Clone)]
pub enum Expr {
    // --- Atoms ---
    /// Reference to a local variable.
    Local(LocalId),
    /// Reference to a global (top-level def).
    Global(String),
    /// Integer literal.
    Int(String),
    /// Float literal.
    Float(String),
    /// Boolean literal.
    Bool(bool),
    /// Unit value ().
    Unit,
    /// String literal.
    String(String),

    // --- Aggregates ---
    /// Tuple literal.
    Tuple(Vec<ExprId>),
    /// Array literal.
    Array(Vec<ExprId>),
    /// Vector literal (@[1.0, 2.0, 3.0]).
    Vector(Vec<ExprId>),
    /// Matrix literal (@[[1,2], [3,4]]).
    Matrix(Vec<Vec<ExprId>>),

    // --- Operations ---
    /// Binary operation.
    BinOp {
        op: String,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// Unary operation.
    UnaryOp {
        op: String,
        operand: ExprId,
    },

    // --- Binding & Control ---
    /// Let binding: allocates a local and evaluates body with it in scope.
    Let {
        local: LocalId,
        rhs: ExprId,
        body: ExprId,
    },
    /// Conditional expression.
    If {
        cond: ExprId,
        then_: ExprId,
        else_: ExprId,
    },
    /// Unified loop construct.
    Loop {
        /// The loop accumulator variable.
        loop_var: LocalId,
        /// Initial value for the accumulator.
        init: ExprId,
        /// Bindings that extract from loop_var.
        init_bindings: Vec<(LocalId, ExprId)>,
        /// The kind of loop.
        kind: LoopKind,
        /// Loop body expression.
        body: ExprId,
    },

    // --- Calls ---
    /// Regular function call.
    Call {
        func: String,
        args: Vec<ExprId>,
    },
    /// Compiler intrinsic call.
    Intrinsic {
        name: String,
        args: Vec<ExprId>,
    },

    // --- Closures ---
    /// A closure value (defunctionalized lambda).
    /// The captures field is a single ExprId pointing to a Tuple or Unit expression
    /// containing all captured values.
    Closure {
        lambda_name: String,
        captures: ExprId,
    },

    // --- Ranges ---
    /// Range expression: start..end or start..step..end.
    Range {
        start: ExprId,
        step: Option<ExprId>,
        end: ExprId,
        kind: RangeKind,
    },

    // --- Special ---
    /// Materialize a value into a variable for indexing.
    Materialize(ExprId),
    /// Expression with attributes attached.
    Attributed {
        attributes: Vec<Attribute>,
        expr: ExprId,
    },

    // --- Slices ---
    /// Owned slice with capacity-sized buffer and dynamic length.
    /// Produced by filter, array literals with dynamic content.
    /// Type: Slice(cap, elem) where cap is the buffer capacity.
    OwnedSlice {
        /// The backing buffer: [cap]elem
        data: ExprId,
        /// Dynamic length: i32 (0 <= len <= cap)
        len: ExprId,
    },

    /// Borrowed slice referencing part of another array (zero-copy view).
    /// Produced by arr[i:j], take, drop.
    /// Type: Slice(cap, elem) where cap is the source array size.
    BorrowedSlice {
        /// The source array being sliced
        base: ExprId,
        /// Start offset: i32
        offset: ExprId,
        /// Dynamic length: i32
        len: ExprId,
    },
}

// =============================================================================
// Function Body
// =============================================================================

/// A function body containing the expression arena and locals table.
#[derive(Debug, Clone)]
pub struct Body {
    /// ID source for locals.
    local_ids: crate::IdSource<LocalId>,
    /// ID source for expressions.
    expr_ids: crate::IdSource<ExprId>,
    /// All local variables in this body.
    pub locals: Vec<LocalDecl>,
    /// Expression arena.
    pub exprs: Vec<Expr>,
    /// Type per expression (parallel to exprs).
    pub types: Vec<Type<TypeName>>,
    /// Span per expression (parallel to exprs).
    pub spans: Vec<Span>,
    /// NodeId per expression (parallel to exprs, for diagnostics).
    pub node_ids: Vec<NodeId>,
    /// Root expression of the body.
    pub root: ExprId,
}

impl Body {
    /// Create a new empty body.
    pub fn new() -> Self {
        Body {
            local_ids: crate::IdSource::new(),
            expr_ids: crate::IdSource::new(),
            locals: Vec::new(),
            exprs: Vec::new(),
            types: Vec::new(),
            spans: Vec::new(),
            node_ids: Vec::new(),
            root: ExprId(0), // Will be set when root is allocated
        }
    }

    /// Allocate a new local variable.
    pub fn alloc_local(&mut self, decl: LocalDecl) -> LocalId {
        let id = self.local_ids.next_id();
        self.locals.push(decl);
        id
    }

    /// Allocate a new expression with its metadata.
    pub fn alloc_expr(&mut self, expr: Expr, ty: Type<TypeName>, span: Span, node_id: NodeId) -> ExprId {
        let id = self.expr_ids.next_id();
        self.exprs.push(expr);
        self.types.push(ty);
        self.spans.push(span);
        self.node_ids.push(node_id);
        id
    }

    /// Get an expression by ID.
    pub fn get_expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id.index()]
    }

    /// Get a mutable reference to an expression by ID.
    pub fn get_expr_mut(&mut self, id: ExprId) -> &mut Expr {
        &mut self.exprs[id.index()]
    }

    /// Get the type of an expression.
    pub fn get_type(&self, id: ExprId) -> &Type<TypeName> {
        &self.types[id.index()]
    }

    /// Get the span of an expression.
    pub fn get_span(&self, id: ExprId) -> Span {
        self.spans[id.index()]
    }

    /// Get the NodeId of an expression.
    pub fn get_node_id(&self, id: ExprId) -> NodeId {
        self.node_ids[id.index()]
    }

    /// Get a local declaration by ID.
    pub fn get_local(&self, id: LocalId) -> &LocalDecl {
        &self.locals[id.index()]
    }

    /// Get a mutable reference to a local declaration by ID.
    pub fn get_local_mut(&mut self, id: LocalId) -> &mut LocalDecl {
        &mut self.locals[id.index()]
    }

    /// Set the root expression.
    pub fn set_root(&mut self, root: ExprId) {
        self.root = root;
    }

    /// Number of locals in this body.
    pub fn num_locals(&self) -> usize {
        self.locals.len()
    }

    /// Number of expressions in this body.
    pub fn num_exprs(&self) -> usize {
        self.exprs.len()
    }

    /// Iterate over all expressions in the body.
    pub fn iter_exprs(&self) -> impl Iterator<Item = &Expr> {
        self.exprs.iter()
    }

    /// Iterate over all locals in the body.
    pub fn iter_locals(&self) -> impl Iterator<Item = &LocalDecl> {
        self.locals.iter()
    }

    // =========================================================================
    // Hoist Helpers
    // =========================================================================

    /// Replace an expression with a Local reference, returning the original expression.
    ///
    /// This is a low-level helper for hoisting transformations. It:
    /// 1. Creates a new local variable
    /// 2. Replaces the expression at `expr_id` with `Expr::Local(local_id)`
    /// 3. Returns the local ID and the original expression
    ///
    /// The caller is responsible for creating a Let binding with the returned expression.
    pub fn extract_to_local(&mut self, expr_id: ExprId, name: Option<String>) -> (LocalId, Expr) {
        let ty = self.get_type(expr_id).clone();
        let span = self.get_span(expr_id);

        // Allocate local ID first so we can use it in the name
        let local_id = self.local_ids.next_id();
        let name = name.unwrap_or_else(|| format!("_w_hoist_{}", local_id.0));

        self.locals.push(LocalDecl {
            name,
            span,
            ty,
            kind: LocalKind::Let,
        });

        let original = std::mem::replace(self.get_expr_mut(expr_id), Expr::Local(local_id));

        (local_id, original)
    }

    /// Hoist an expression before a target expression by wrapping target in a Let binding.
    ///
    /// This replaces `to_hoist` with a Local reference and wraps `target` in a Let:
    /// ```text
    /// Before: ... target containing to_hoist ...
    /// After:  let _hoist_N = <original to_hoist> in ... target containing Local(_hoist_N) ...
    /// ```
    ///
    /// Returns the LocalId of the new binding.
    pub fn hoist_before(&mut self, to_hoist: ExprId, target: ExprId, name: Option<String>) -> LocalId {
        let to_hoist_ty = self.get_type(to_hoist).clone();
        let to_hoist_span = self.get_span(to_hoist);
        let to_hoist_node_id = self.get_node_id(to_hoist);

        // Extract the expression to hoist, replacing it with a Local reference
        let (local_id, original_expr) = self.extract_to_local(to_hoist, name);

        // Allocate the original expression as rhs for the Let
        let rhs_id = self.alloc_expr(original_expr, to_hoist_ty, to_hoist_span, to_hoist_node_id);

        // Clone the target expression and allocate as body of the Let
        let target_ty = self.get_type(target).clone();
        let target_span = self.get_span(target);
        let target_node_id = self.get_node_id(target);
        let target_expr = self.get_expr(target).clone();
        let body_id = self.alloc_expr(target_expr, target_ty.clone(), target_span, target_node_id);

        // Replace target with the Let expression
        *self.get_expr_mut(target) = Expr::Let {
            local: local_id,
            rhs: rhs_id,
            body: body_id,
        };

        local_id
    }
}

impl Default for Body {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Program and Definitions
// =============================================================================

/// Information about a registered lambda.
#[derive(Debug, Clone)]
pub struct LambdaInfo {
    /// Name of the generated function.
    pub name: String,
    /// Number of parameters (excluding closure parameter).
    pub arity: usize,
}

/// A complete MIR program.
#[derive(Debug, Clone)]
pub struct Program {
    /// All top-level definitions in the program.
    pub defs: Vec<Def>,
    /// Lambda registry: maps LambdaId -> LambdaInfo.
    pub lambda_registry: IdArena<LambdaId, LambdaInfo>,
}

/// A top-level definition.
#[derive(Debug, Clone)]
pub enum Def {
    /// A function definition with parameters.
    Function {
        /// Unique node identifier.
        id: NodeId,
        /// Function name.
        name: String,
        /// Parameter indices into the body's locals table.
        params: Vec<LocalId>,
        /// Return type.
        ret_type: Type<TypeName>,
        /// Original polymorphic type scheme (for monomorphization).
        /// This ensures parameter and return type variables are consistent.
        scheme: Option<TypeScheme>,
        /// Attributes.
        attributes: Vec<Attribute>,
        /// Function body.
        body: Body,
        /// Source location.
        span: Span,
    },
    /// A constant definition (no parameters).
    Constant {
        /// Unique node identifier.
        id: NodeId,
        /// Constant name.
        name: String,
        /// Type of the constant.
        ty: Type<TypeName>,
        /// Attributes.
        attributes: Vec<Attribute>,
        /// Constant body.
        body: Body,
        /// Source location.
        span: Span,
    },
    /// A uniform declaration (external input from host).
    Uniform {
        /// Unique node identifier.
        id: NodeId,
        /// Uniform name.
        name: String,
        /// Type of this uniform.
        ty: Type<TypeName>,
        /// Descriptor set number.
        set: u32,
        /// Binding number.
        binding: u32,
    },
    /// A storage buffer declaration.
    Storage {
        /// Unique node identifier.
        id: NodeId,
        /// Storage buffer name.
        name: String,
        /// Type of this storage buffer.
        ty: Type<TypeName>,
        /// Descriptor set number.
        set: u32,
        /// Binding number.
        binding: u32,
        /// Memory layout.
        layout: crate::ast::StorageLayout,
        /// Access mode.
        access: crate::ast::StorageAccess,
    },
    /// A shader entry point.
    EntryPoint {
        /// Unique node identifier.
        id: NodeId,
        /// Entry point name.
        name: String,
        /// Execution model.
        execution_model: ExecutionModel,
        /// Input parameters (indices into body's locals).
        inputs: Vec<EntryInput>,
        /// Output decorations.
        outputs: Vec<EntryOutput>,
        /// Entry point body.
        body: Body,
        /// Source location.
        span: Span,
    },
}

/// A function parameter (for Def-level representation).
/// Note: Parameters are stored as LocalId in the new MIR,
/// this struct is kept for compatibility during transition.
#[derive(Debug, Clone)]
pub struct Param {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub ty: Type<TypeName>,
}

// =============================================================================
// Supporting Types
// =============================================================================

/// An attribute that can be attached to functions or expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
    Vertex,
    Fragment,
    Compute {
        local_size: (u32, u32, u32),
    },
    Uniform,
    Storage,
    /// Hint for expected size of a dynamic array (in elements).
    SizeHint(u32),
}

/// Execution model for a shader entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionModel {
    Vertex,
    Fragment,
    Compute {
        local_size: (u32, u32, u32),
    },
}

/// Decoration for shader I/O.
#[derive(Debug, Clone, PartialEq)]
pub enum IoDecoration {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
}

/// An input parameter to a shader entry point.
#[derive(Debug, Clone)]
pub struct EntryInput {
    /// LocalId of the input parameter in the body.
    pub local: LocalId,
    /// Parameter name (for debugging).
    pub name: String,
    /// Parameter type.
    pub ty: Type<TypeName>,
    /// I/O decoration.
    pub decoration: Option<IoDecoration>,
}

/// An output from a shader entry point.
#[derive(Debug, Clone)]
pub struct EntryOutput {
    /// Output type.
    pub ty: Type<TypeName>,
    /// I/O decoration.
    pub decoration: Option<IoDecoration>,
}

/// The kind of range expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeKind {
    /// `...` - inclusive on both ends
    Inclusive,
    /// `..` - exclusive on end
    Exclusive,
    /// `..<` - exclusive on end (explicit)
    ExclusiveLt,
    /// `..>` - exclusive on end, stepping downward
    ExclusiveGt,
}

/// The kind of loop construct.
#[derive(Debug, Clone)]
pub enum LoopKind {
    /// For loop over an array: `for x in arr`.
    For {
        /// LocalId of the loop variable.
        var: LocalId,
        /// Array to iterate over.
        iter: ExprId,
    },
    /// For loop with range bound: `for i < n`.
    ForRange {
        /// LocalId of the loop variable.
        var: LocalId,
        /// Upper bound.
        bound: ExprId,
    },
    /// While loop: `while cond`.
    While {
        /// Loop condition.
        cond: ExprId,
    },
}
