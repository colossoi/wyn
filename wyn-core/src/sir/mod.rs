//! SIR (SOAC Intermediate Representation)
//!
//! A typed dataflow IR for Futhark-style parallelization transforms.
//! The surface syntax is a procedural sequence of let-bindings (ANF),
//! but the semantic shape is a typed dataflow DAG where the interesting
//! nodes are SOAC operators (map, reduce, scan, seg*).
//!
//! Key properties:
//! - VarIds are globally unique (SSA-like)
//! - SOACs contain nested lambda bodies
//! - Lambdas have explicit captures (pre-defunctionalization)
//! - Types reuse polytype::Type<TypeName> from the existing type system

pub mod builder;
pub mod types;

use std::collections::HashMap;

use crate::ast::{NodeId, Span, TypeName};
use polytype::Type;

pub use types::{AssocInfo, ScalarTy, Size, SizeVar};

// =============================================================================
// ID Types
// =============================================================================

/// Unique identifier for a variable in SIR.
/// Variables are SSA-like: each VarId is defined exactly once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

impl From<u32> for VarId {
    fn from(id: u32) -> Self {
        VarId(id)
    }
}

impl std::fmt::Display for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Unique identifier for a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StmId(pub u32);

impl From<u32> for StmId {
    fn from(id: u32) -> Self {
        StmId(id)
    }
}

/// Unique identifier for a lambda.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LambdaId(pub u32);

impl From<u32> for LambdaId {
    fn from(id: u32) -> Self {
        LambdaId(id)
    }
}

// =============================================================================
// Type Alias
// =============================================================================

/// SIR reuses the existing polytype infrastructure.
pub type SirType = Type<TypeName>;

// =============================================================================
// Program Structure
// =============================================================================

/// A complete SIR program.
#[derive(Debug, Clone)]
pub struct Program {
    /// All top-level definitions.
    pub defs: Vec<Def>,
    /// Lambda registry: maps LambdaId to Lambda for defunctionalization.
    pub lambdas: std::collections::HashMap<LambdaId, Lambda>,
}

/// A top-level definition.
#[derive(Debug, Clone)]
pub enum Def {
    /// A function definition.
    Function {
        id: NodeId,
        name: String,
        params: Vec<Param>,
        ret_ty: SirType,
        body: Body,
        span: Span,
    },
    /// A shader entry point.
    EntryPoint {
        id: NodeId,
        name: String,
        execution_model: ExecutionModel,
        inputs: Vec<EntryInput>,
        outputs: Vec<EntryOutput>,
        body: Body,
        span: Span,
    },
    /// A constant definition.
    Constant {
        id: NodeId,
        name: String,
        ty: SirType,
        body: Body,
        span: Span,
    },
    /// A uniform declaration.
    Uniform {
        id: NodeId,
        name: String,
        ty: SirType,
        set: u32,
        binding: u32,
    },
    /// A storage buffer declaration.
    Storage {
        id: NodeId,
        name: String,
        ty: SirType,
        set: u32,
        binding: u32,
    },
}

/// Execution model for shader entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionModel {
    Vertex,
    Fragment,
    Compute { local_size: (u32, u32, u32) },
}

/// An input to a shader entry point.
#[derive(Debug, Clone)]
pub struct EntryInput {
    pub var: VarId,
    pub name: String,
    pub ty: SirType,
    pub decoration: Option<IoDecoration>,
}

/// An output from a shader entry point.
#[derive(Debug, Clone)]
pub struct EntryOutput {
    pub ty: SirType,
    pub decoration: Option<IoDecoration>,
}

/// I/O decoration for shader inputs/outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum IoDecoration {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
}

// =============================================================================
// Body and Statements
// =============================================================================

/// A function body: a sequence of statements producing result values.
#[derive(Debug, Clone)]
pub struct Body {
    /// Statements in execution order.
    pub stms: Vec<Stm>,
    /// Result values (multiple for tuple returns).
    pub result: Vec<VarId>,
}

impl Body {
    /// Create an empty body with no statements and no results.
    pub fn empty() -> Self {
        Body {
            stms: Vec::new(),
            result: Vec::new(),
        }
    }

    /// Create a body with a single result and no statements.
    pub fn just(var: VarId) -> Self {
        Body {
            stms: Vec::new(),
            result: vec![var],
        }
    }
}

/// A statement: binds pattern to expression result.
#[derive(Debug, Clone)]
pub struct Stm {
    /// Unique statement identifier.
    pub id: StmId,
    /// Pattern being bound (defines variables).
    pub pat: Pat,
    /// Expression being evaluated.
    pub exp: Exp,
    /// Type of the pattern (tuple type if multiple binds).
    pub ty: SirType,
    /// Source location.
    pub span: Span,
}

/// A pattern for binding statement results.
#[derive(Debug, Clone)]
pub struct Pat {
    /// Individual bindings in the pattern.
    pub binds: Vec<PatElem>,
}

impl Pat {
    /// Create a single-variable pattern.
    pub fn single(var: VarId, ty: SirType, name: String) -> Self {
        Pat {
            binds: vec![PatElem {
                var,
                ty,
                name_hint: name,
            }],
        }
    }

    /// Get the single variable if this is a single-bind pattern.
    pub fn single_var(&self) -> Option<VarId> {
        if self.binds.len() == 1 {
            Some(self.binds[0].var)
        } else {
            None
        }
    }
}

/// A single element in a pattern.
#[derive(Debug, Clone)]
pub struct PatElem {
    /// Variable being bound.
    pub var: VarId,
    /// Type of this binding.
    pub ty: SirType,
    /// Name hint for debugging/display.
    pub name_hint: String,
}

/// A function/lambda parameter.
#[derive(Debug, Clone)]
pub struct Param {
    /// Name hint for debugging.
    pub name_hint: String,
    /// Variable for this parameter.
    pub var: VarId,
    /// Type of the parameter.
    pub ty: SirType,
    /// Source location.
    pub span: Span,
}

// =============================================================================
// Expressions
// =============================================================================

/// An expression in SIR.
#[derive(Debug, Clone)]
pub enum Exp {
    /// Primitive scalar operation.
    Prim(Prim),

    /// Variable reference.
    Var(VarId),

    /// Conditional expression.
    If {
        cond: VarId,
        then_body: Body,
        else_body: Body,
    },

    /// Loop construct.
    Loop {
        /// Loop-carried parameters.
        params: Vec<Param>,
        /// Initial values for loop parameters.
        init: Vec<VarId>,
        /// Loop body.
        body: Body,
    },

    /// SOAC or kernel launch.
    Op(Op),

    /// Function application.
    Apply {
        /// Function name (global).
        func: String,
        /// Arguments.
        args: Vec<VarId>,
    },

    /// Tuple literal.
    Tuple(Vec<VarId>),

    /// Tuple projection.
    TupleProj { tuple: VarId, index: usize },
}

/// Primitive (scalar) operations.
#[derive(Debug, Clone)]
pub enum Prim {
    // Constants
    ConstBool(bool),
    ConstI32(i32),
    ConstI64(i64),
    ConstU32(u32),
    ConstU64(u64),
    ConstF32(f32),
    ConstF64(f64),

    // Binary arithmetic
    Add(VarId, VarId),
    Sub(VarId, VarId),
    Mul(VarId, VarId),
    Div(VarId, VarId),
    Mod(VarId, VarId),

    // Binary comparison
    Eq(VarId, VarId),
    Ne(VarId, VarId),
    Lt(VarId, VarId),
    Le(VarId, VarId),
    Gt(VarId, VarId),
    Ge(VarId, VarId),

    // Binary logical
    And(VarId, VarId),
    Or(VarId, VarId),

    // Unary
    Neg(VarId),
    Not(VarId),

    // Array indexing
    Index { arr: VarId, idx: VarId },

    // Intrinsic call (for ops not worth special-casing)
    Intrinsic { name: String, args: Vec<VarId> },
}

// =============================================================================
// SOAC Operations
// =============================================================================

/// Parallel operation: SOAC or explicit kernel launch.
#[derive(Debug, Clone)]
pub enum Op {
    /// Second-order array combinator.
    Soac(Soac),
    /// Explicit kernel launch (post-kernelization).
    Launch(Launch),
}

/// Second-Order Array Combinator.
#[derive(Debug, Clone)]
pub enum Soac {
    /// Parallel map: apply function to each element.
    Map(Map),
    /// Parallel reduction: combine elements with associative operator.
    Reduce(Reduce),
    /// Parallel prefix scan.
    Scan(Scan),

    // Segmented variants (for nested parallelism)
    SegMap(SegMap),
    SegReduce(SegReduce),
    SegScan(SegScan),

    // Array constructors that often fuse away
    /// Generate [0, 1, 2, ..., n-1].
    Iota { n: Size, elem_ty: ScalarTy },
    /// Generate [v, v, v, ...] of length n.
    Replicate { n: Size, value: VarId },
    /// Reshape array to new dimensions.
    Reshape { new_shape: Vec<Size>, arr: VarId },
}

/// Parallel map operation.
#[derive(Debug, Clone)]
pub struct Map {
    /// Outer width (number of parallel iterations).
    pub w: Size,
    /// Function to apply.
    pub f: Lambda,
    /// Input arrays (zipped together).
    pub arrs: Vec<VarId>,
}

/// Parallel reduction operation.
#[derive(Debug, Clone)]
pub struct Reduce {
    /// Width of input array.
    pub w: Size,
    /// Reduction function: (acc, x) -> acc.
    pub f: Lambda,
    /// Neutral element (identity for the operator).
    pub neutral: VarId,
    /// Input array.
    pub arr: VarId,
    /// Associativity/commutativity info.
    pub assoc: AssocInfo,
}

/// Parallel prefix scan operation.
#[derive(Debug, Clone)]
pub struct Scan {
    /// Width of input array.
    pub w: Size,
    /// Scan function: (acc, x) -> acc.
    pub f: Lambda,
    /// Neutral element.
    pub neutral: VarId,
    /// Input array.
    pub arr: VarId,
    /// Associativity/commutativity info.
    pub assoc: AssocInfo,
}

/// Segmented map: map over irregular segments.
#[derive(Debug, Clone)]
pub struct SegMap {
    /// Segment descriptor.
    pub segs: VarId,
    /// Function to apply within each segment.
    pub f: Lambda,
    /// Input arrays.
    pub arrs: Vec<VarId>,
}

/// Segmented reduction.
#[derive(Debug, Clone)]
pub struct SegReduce {
    /// Segment descriptor.
    pub segs: VarId,
    /// Reduction function.
    pub f: Lambda,
    /// Neutral element.
    pub neutral: VarId,
    /// Input array.
    pub arr: VarId,
    /// Associativity/commutativity info.
    pub assoc: AssocInfo,
}

/// Segmented scan.
#[derive(Debug, Clone)]
pub struct SegScan {
    /// Segment descriptor.
    pub segs: VarId,
    /// Scan function.
    pub f: Lambda,
    /// Neutral element.
    pub neutral: VarId,
    /// Input array.
    pub arr: VarId,
    /// Associativity/commutativity info.
    pub assoc: AssocInfo,
}

// =============================================================================
// Lambda
// =============================================================================

/// A lambda (nested function) with explicit captures.
#[derive(Debug, Clone)]
pub struct Lambda {
    /// Unique identifier for caching/rewriting.
    pub id: LambdaId,
    /// Parameters (bound in body).
    pub params: Vec<Param>,
    /// Captured variables from enclosing scope.
    pub captures: Vec<VarId>,
    /// Lambda body.
    pub body: Body,
    /// Return types.
    pub ret_tys: Vec<SirType>,
    /// Source location.
    pub span: Span,
}

// =============================================================================
// Kernel Launch (Post-Kernelization)
// =============================================================================

/// Explicit kernel launch.
#[derive(Debug, Clone)]
pub struct Launch {
    /// Kind of kernel.
    pub kind: LaunchKind,
    /// Input arrays/values.
    pub inputs: Vec<VarId>,
    /// Output pattern.
    pub outputs: Pat,
    /// Kernel body.
    pub body: KernelBody,
}

/// Kind of kernel being launched.
#[derive(Debug, Clone)]
pub enum LaunchKind {
    MapKernel,
    ReduceKernel,
    ScanKernel,
    SegMapKernel,
    SegReduceKernel,
    SegScanKernel,
}

/// Body of a kernel (restricted compared to general Body).
#[derive(Debug, Clone)]
pub struct KernelBody {
    pub body: Body,
}

// =============================================================================
// Type Environment
// =============================================================================

/// Maps variables to their types.
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    pub var_tys: HashMap<VarId, SirType>,
}

impl TypeEnv {
    pub fn new() -> Self {
        TypeEnv {
            var_tys: HashMap::new(),
        }
    }

    /// Get the type of a variable.
    pub fn ty(&self, v: VarId) -> &SirType {
        self.var_tys.get(&v).expect("missing VarId type")
    }

    /// Set the type of a variable.
    pub fn set_ty(&mut self, v: VarId, ty: SirType) {
        self.var_tys.insert(v, ty);
    }

    /// Check if a variable has a type.
    pub fn has(&self, v: VarId) -> bool {
        self.var_tys.contains_key(&v)
    }
}
