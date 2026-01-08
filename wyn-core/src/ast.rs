pub use spirv;

// Re-export type system types from the types module
use crate::IdSource;
pub use crate::types::{RecordFields, Type, TypeName, TypeScheme};

/// Qualified name representing a path through modules to a name
/// E.g., M.N.x is represented as QualName { qualifiers: ["M", "N"], name: "x" }
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualName {
    pub qualifiers: Vec<String>,
    pub name: String,
}

impl QualName {
    /// Create a new qualified name
    pub fn new(qualifiers: Vec<String>, name: String) -> Self {
        QualName { qualifiers, name }
    }

    /// Create an unqualified name (no qualifiers)
    pub fn unqualified(name: String) -> Self {
        QualName {
            qualifiers: vec![],
            name,
        }
    }

    /// Get the dotted notation (for display/debugging)
    /// E.g., "M.N.x"
    pub fn to_dotted(&self) -> String {
        if self.qualifiers.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.qualifiers.join("."), self.name)
        }
    }

    /// Check if this is an unqualified name
    pub fn is_unqualified(&self) -> bool {
        self.qualifiers.is_empty()
    }
}

/// Source location span tracking (line, column) start and end positions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl Span {
    pub fn new(start_line: usize, start_col: usize, end_line: usize, end_col: usize) -> Self {
        Span {
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    /// Create a dummy/generated span (all zeros) for test code
    #[cfg(test)]
    pub fn dummy() -> Self {
        Span {
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        }
    }

    /// Check if this is a generated/dummy span (all zeros)
    pub fn is_generated(&self) -> bool {
        self.start_line == 0 && self.start_col == 0 && self.end_line == 0 && self.end_col == 0
    }

    /// Merge two spans to create a span covering both
    pub fn merge(&self, other: &Span) -> Span {
        let (start_line, start_col) = if self.start_line < other.start_line
            || (self.start_line == other.start_line && self.start_col <= other.start_col)
        {
            (self.start_line, self.start_col)
        } else {
            (other.start_line, other.start_col)
        };

        let (end_line, end_col) = if self.end_line > other.end_line
            || (self.end_line == other.end_line && self.end_col >= other.end_col)
        {
            (self.end_line, self.end_col)
        } else {
            (other.end_line, other.end_col)
        };

        Span {
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    /// Check if this span contains a position (1-based line/col)
    pub fn contains(&self, line: usize, col: usize) -> bool {
        if line < self.start_line || line > self.end_line {
            return false;
        }
        if line == self.start_line && col < self.start_col {
            return false;
        }
        if line == self.end_line && col > self.end_col {
            return false;
        }
        true
    }

    /// Calculate the "size" of a span for comparison (smaller = more specific)
    pub fn size(&self) -> usize {
        if self.end_line == self.start_line {
            self.end_col.saturating_sub(self.start_col)
        } else {
            // Rough estimate: 100 chars per line
            (self.end_line - self.start_line) * 100 + self.end_col
        }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.start_line == self.end_line {
            write!(f, "{}:{}..{}", self.start_line, self.start_col, self.end_col)
        } else {
            write!(
                f,
                "{}:{}..{}:{}",
                self.start_line, self.start_col, self.end_line, self.end_col
            )
        }
    }
}

/// Unique identifier for AST nodes (expressions)
/// Used to look up inferred types in the type table
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn new(id: u32) -> Self {
        NodeId(id)
    }
}

impl From<u32> for NodeId {
    fn from(value: u32) -> Self {
        NodeId(value)
    }
}

/// Counter for generating unique node IDs across compilation phases
pub type NodeCounter = IdSource<NodeId>;

/// Extension trait for NodeCounter to provide AST node creation helpers
pub trait NodeCounterExt {
    fn mk_node<T>(&mut self, kind: T, span: Span) -> Node<T>;
}

impl NodeCounterExt for NodeCounter {
    fn mk_node<T>(&mut self, kind: T, span: Span) -> Node<T> {
        Node {
            h: Header {
                id: self.next_id(),
                span,
            },
            kind,
        }
    }
}

#[cfg(test)]
pub trait NodeCounterTestExt {
    /// Create a node with a dummy span (for testing only)
    fn mk_node_dummy<T>(&mut self, kind: T) -> Node<T>;
}

#[cfg(test)]
impl NodeCounterTestExt for NodeCounter {
    fn mk_node_dummy<T>(&mut self, kind: T) -> Node<T> {
        self.mk_node(kind, Span::dummy())
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub id: NodeId,
    pub span: Span,
    // hygiene, source file id, etc.
}

#[derive(Clone, Debug)]
pub struct Node<T> {
    pub h: Header,
    pub kind: T,
}

impl<T> PartialEq for Node<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
pub type Expression = Node<ExprKind>;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Decl(Decl),           // Unified let/def declarations
    Entry(EntryDecl),     // Entry point declarations (vertex/fragment shaders)
    Uniform(UniformDecl), // Uniform declarations (no initializer)
    Storage(StorageDecl), // Storage buffer declarations
    Sig(SigDecl),
    TypeBind(TypeBind),             // Type declarations
    Module(ModuleDecl),             // Module and functor declarations
    ModuleTypeBind(ModuleTypeBind), // Module type declarations
    Open(ModuleExpression),         // open mod_exp
    Import(String),                 // import "path"
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
    Vertex,
    Fragment,
    /// Compute shader entry point. Workgroup size is determined by the compiler.
    Compute,
    Uniform {
        set: u32,
        binding: u32,
    },
    Storage {
        set: u32,
        binding: u32,
        layout: StorageLayout,
        access: StorageAccess,
    },
    /// Hint for the expected size of a dynamic array (in elements).
    /// Used for parallelization decisions. Ignored on non-arrays or statically sized arrays.
    SizeHint(u32),
}

impl Attribute {
    pub fn is_vertex(&self) -> bool {
        matches!(self, Attribute::Vertex)
    }
    pub fn is_fragment(&self) -> bool {
        matches!(self, Attribute::Fragment)
    }
    pub fn is_compute(&self) -> bool {
        matches!(self, Attribute::Compute)
    }
}

pub trait AttrExt {
    fn has<F: Fn(&Attribute) -> bool>(&self, pred: F) -> bool;
    fn first_builtin(&self) -> Option<spirv::BuiltIn>;
    fn first_location(&self) -> Option<u32>;
}

impl AttrExt for [Attribute] {
    fn has<F: Fn(&Attribute) -> bool>(&self, pred: F) -> bool {
        self.iter().any(pred)
    }
    fn first_builtin(&self) -> Option<spirv::BuiltIn> {
        self.iter().find_map(|a| if let Attribute::BuiltIn(b) = a { Some(*b) } else { None })
    }
    fn first_location(&self) -> Option<u32> {
        self.iter().find_map(|a| if let Attribute::Location(l) = a { Some(*l) } else { None })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttributedType {
    pub attributes: Vec<Attribute>,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decl {
    pub keyword: &'static str, // Either "let" or "def"
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub params: Vec<Pattern>,     // Parameters as patterns (name, name:type, tuples, etc.)
    pub ty: Option<Type>,         // Return type for functions or type annotation for variables
    pub body: Expression,         // The value/expression for let/def declarations
}

/// Output field for entry point declarations
#[derive(Debug, Clone, PartialEq)]
pub struct EntryOutput {
    pub ty: Type,
    pub attribute: Option<Attribute>,
}

/// Entry point declaration (vertex/fragment shader)
#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl {
    pub entry_type: Attribute, // Attribute::Vertex or Attribute::Fragment
    pub name: String,
    pub params: Vec<Pattern>,      // Input parameters as patterns
    pub outputs: Vec<EntryOutput>, // Output fields with optional attributes
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SigDecl {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub ty: Type,                 // The function type signature
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniformDecl {
    pub name: String,
    pub ty: Type,     // Uniforms always have an explicit type
    pub set: u32,     // Descriptor set number (default 0)
    pub binding: u32, // Explicit binding number (required)
}

#[derive(Debug, Clone, PartialEq)]
pub struct StorageDecl {
    pub name: String,
    pub ty: Type,     // Storage buffers have an explicit type (usually runtime-sized array)
    pub set: u32,     // Descriptor set number
    pub binding: u32, // Binding number within the set
    pub layout: StorageLayout, // Memory layout (std430, std140)
    pub access: StorageAccess, // Access mode (read, write, readwrite)
}

/// Memory layout for storage buffers
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StorageLayout {
    #[default]
    Std430, // Default for SSBOs, tightly packed
    Std140, // More relaxed alignment, compatible with UBOs
}

/// Access mode for storage buffers
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StorageAccess {
    ReadOnly,
    WriteOnly,
    #[default]
    ReadWrite,
}

// Module system types
#[derive(Debug, Clone, PartialEq)]
pub struct TypeBind {
    pub kind: TypeBindKind, // type, type^, or type~
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub definition: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeBindKind {
    Normal, // type
    Lifted, // type^
    Size,   // type~
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeParam {
    Size(String),       // [n]
    Type(String),       // 'a
    SizeType(String),   // '~a
    LiftedType(String), // '^a
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleDecl {
    Module {
        name: String,
        signature: Option<ModuleTypeExpression>,
        body: ModuleExpression,
    },
    Functor {
        name: String,
        params: Vec<ModuleParam>,
        body: ModuleExpression,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleParam {
    pub name: String,
    pub signature: ModuleTypeExpression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleTypeBind {
    pub name: String,
    pub definition: ModuleTypeExpression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleExpression {
    Name(String),                                            // qualname
    Ascription(Box<ModuleExpression>, ModuleTypeExpression), // mod_exp : mod_type_exp
    Lambda(
        Vec<ModuleParam>,
        Option<ModuleTypeExpression>,
        Box<ModuleExpression>,
    ), // \ (params) [: sig] -> body
    Application(Box<ModuleExpression>, Box<ModuleExpression>), // mod_exp mod_exp
    Struct(Vec<Declaration>),                                // { dec* }
    Import(String),                                          // import "path"
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleTypeExpression {
    Name(String),                                                        // qualname
    Signature(Vec<Spec>),                                                // { spec* }
    With(Box<ModuleTypeExpression>, String, Vec<TypeParam>, Type), // mod_type with qualname type_params = type
    Arrow(String, Box<ModuleTypeExpression>, Box<ModuleTypeExpression>), // (name : mod_type) -> mod_type
    FunctorType(Box<ModuleTypeExpression>, Box<ModuleTypeExpression>), // mod_type -> mod_type
}

#[derive(Debug, Clone, PartialEq)]
pub enum Spec {
    Sig(String, Vec<TypeParam>, Type), // sig name type_params : type
    SigOp(String, Type),               // sig (symbol) : type or sig symbol : type
    Type(TypeBindKind, String, Vec<TypeParam>, Option<Type>), // type declarations with optional definition
    Module(String, ModuleTypeExpression), // module name : mod_type_exp
    Include(ModuleTypeExpression),     // include mod_type_exp
}

// We now use polytype::Type instead of our own Type enum

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    IntLiteral(i32),
    FloatLiteral(f32),
    BoolLiteral(bool),
    StringLiteral(String),
    Unit,
    Identifier(Vec<String>, String), // (qualifiers, name) - e.g., ([], "x") or (["f32"], "sin")
    ArrayLiteral(Vec<Expression>),
    VecMatLiteral(Vec<Expression>), // @[...] - vector or matrix literal (type inferred from context)
    ArrayIndex(Box<Expression>, Box<Expression>),
    /// In-place array update: `a with [i] = v` produces a copy of `a` with element `i` set to `v`
    ArrayWith {
        array: Box<Expression>,
        index: Box<Expression>,
        value: Box<Expression>,
    },
    BinaryOp(BinaryOp, Box<Expression>, Box<Expression>),
    UnaryOp(UnaryOp, Box<Expression>), // Unary operations: -, !
    Tuple(Vec<Expression>),
    RecordLiteral(Vec<(String, Expression)>), // e.g. {x: 1, y: 2}
    Lambda(LambdaExpr),
    Application(Box<Expression>, Vec<Expression>), // Function application
    LetIn(LetInExpr),
    FieldAccess(Box<Expression>, String),  // e.g. v.x, v.y
    If(IfExpr),                            // if-then-else expression
    Loop(LoopExpr),                        // loop expression
    Match(MatchExpr),                      // match expression
    Range(RangeExpr),                      // range expressions: a..b, a..<b, a..>b, a...b
    Slice(SliceExpr),                      // array slicing: a[i:j:s]
    TypeAscription(Box<Expression>, Type), // exp : type
    TypeCoercion(Box<Expression>, Type),   // exp :> type
    TypeHole,                              // ??? - placeholder for any expression
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambdaExpr {
    pub params: Vec<Pattern>,
    pub return_type: Option<Type>,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetInExpr {
    pub pattern: Pattern, // Can be Name, Tuple, etc.
    pub ty: Option<Type>, // Optional type annotation
    pub value: Box<Expression>,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOp {
    pub op: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryOp {
    pub op: String, // "-" or "!"
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr {
    pub condition: Box<Expression>,
    pub then_branch: Box<Expression>,
    pub else_branch: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopExpr {
    pub pattern: Pattern,              // loop variable pattern
    pub init: Option<Box<Expression>>, // initial value (optional)
    pub form: LoopForm,                // for/while condition
    pub body: Box<Expression>,         // loop body
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopForm {
    For(String, Box<Expression>),    // for name < exp
    ForIn(Pattern, Box<Expression>), // for pat in exp
    While(Box<Expression>),          // while exp
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpr {
    pub scrutinee: Box<Expression>, // expression being matched
    pub cases: Vec<MatchCase>,      // case branches
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeExpr {
    pub start: Box<Expression>,
    pub step: Option<Box<Expression>>, // Optional middle expression in start..step..end
    pub end: Box<Expression>,
    pub kind: RangeKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RangeKind {
    Inclusive,   // ... (three dots)
    Exclusive,   // .. (two dots)
    ExclusiveLt, // ..<
    ExclusiveGt, // ..>
}

/// Array slice expression: a[start:end]
/// Both start and end are optional: a[:], a[i:], a[:j], a[i:j]
/// TODO: Step support (arr[i:j:s]) deferred to future work
#[derive(Debug, Clone, PartialEq)]
pub struct SliceExpr {
    pub array: Box<Expression>,
    pub start: Option<Box<Expression>>, // None = 0
    pub end: Option<Box<Expression>>,   // None = len
}

// Pattern types for match expressions and let bindings
#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Name(String),                             // Simple name binding
    Wildcard,                                 // _ wildcard
    Literal(PatternLiteral),                  // Literal patterns
    Unit,                                     // () unit pattern
    Tuple(Vec<Pattern>),                      // (pat1, pat2, ...)
    Record(Vec<RecordPatternField>),          // { field1, field2 = pat, ... }
    Constructor(String, Vec<Pattern>),        // Constructor application
    Typed(Box<Pattern>, Type),                // pat : type
    Attributed(Vec<Attribute>, Box<Pattern>), // #[attr] pat
}

pub type Pattern = Node<PatternKind>;

impl Pattern {
    /// Extract the simple name from a pattern if possible
    /// For Name("x") returns Some("x")
    /// For Typed(Name("x"), _) returns Some("x")
    /// For Attributed(_, Name("x")) returns Some("x")
    /// Returns None for complex patterns like tuples, records, etc.
    pub fn simple_name(&self) -> Option<&str> {
        match &self.kind {
            PatternKind::Name(name) => Some(name),
            PatternKind::Typed(inner, _) => inner.simple_name(),
            PatternKind::Attributed(_, inner) => inner.simple_name(),
            _ => None,
        }
    }

    /// Collect all names bound by this pattern (recursively for tuple patterns)
    pub fn bound_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.collect_bound_names(&mut names);
        names
    }

    fn collect_bound_names(&self, names: &mut Vec<String>) {
        match &self.kind {
            PatternKind::Name(name) => names.push(name.clone()),
            PatternKind::Typed(inner, _) => inner.collect_bound_names(names),
            PatternKind::Attributed(_, inner) => inner.collect_bound_names(names),
            PatternKind::Tuple(patterns) => {
                for pat in patterns {
                    pat.collect_bound_names(names);
                }
            }
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Unit => {}
            PatternKind::Constructor(_, patterns) => {
                for pat in patterns {
                    pat.collect_bound_names(names);
                }
            }
            PatternKind::Record(fields) => {
                for field in fields {
                    if let Some(pat) = &field.pattern {
                        pat.collect_bound_names(names);
                    } else {
                        // Shorthand: field name is the bound name
                        names.push(field.field.clone());
                    }
                }
            }
        }
    }

    /// Extract the type from a typed pattern
    pub fn pattern_type(&self) -> Option<&Type> {
        match &self.kind {
            PatternKind::Typed(_, ty) => Some(ty),
            PatternKind::Attributed(_, inner) => inner.pattern_type(),
            _ => None,
        }
    }

    /// Collect all names bound by this pattern
    /// For Name("x") returns vec!["x"]
    /// For Tuple([Name("x"), Name("y")]) returns vec!["x", "y"]
    /// For nested patterns, recursively collects all names
    pub fn collect_names(&self) -> Vec<String> {
        match &self.kind {
            PatternKind::Name(name) => vec![name.clone()],
            PatternKind::Tuple(patterns) => patterns.iter().flat_map(|p| p.collect_names()).collect(),
            PatternKind::Typed(inner, _) => inner.collect_names(),
            PatternKind::Attributed(_, inner) => inner.collect_names(),
            PatternKind::Record(fields) => fields
                .iter()
                .flat_map(|f| {
                    if let Some(ref pat) = f.pattern { pat.collect_names() } else { vec![f.field.clone()] }
                })
                .collect(),
            PatternKind::Constructor(_, patterns) => {
                patterns.iter().flat_map(|p| p.collect_names()).collect()
            }
            _ => vec![], // Wildcard, Literal, Unit bind no names
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternLiteral {
    Int(i32),
    Float(f32),
    Char(char),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordPatternField {
    pub field: String,
    pub pattern: Option<Pattern>, // None means shorthand (just field name)
}
