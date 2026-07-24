pub use spirv;
pub(crate) mod rebuild;

// Re-export type system types from the types module
use crate::builtins::BuiltinId;
use crate::interface::{Attribute, ComputeDispatchGrid, EntryKind, EntryOutputDecl, FeedbackPair};
pub use crate::types::{Diet, RecordFields, Type, TypeName, TypeScheme};
use crate::IdSource;

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
/// Looks up inferred types in the type table
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

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub id: NodeId,
    pub span: Span,
    // hygiene, source file id, etc.
}

/// Header carried by every type-checked expression and pattern.
///
/// Inferred types used to live in a `NodeId -> TypeScheme` side table.
/// They are intrinsic node-owned data, so the typed AST stores them beside
/// the stable source identity and span.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedHeader {
    pub id: NodeId,
    pub span: Span,
    pub ty: TypeScheme,
}

#[derive(Clone, Debug)]
pub struct Node<T, H = Header> {
    pub h: H,
    pub kind: T,
}

impl<T, H> PartialEq for Node<T, H>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

/// The concrete recursive node shapes stored by an AST family.
///
/// Each associated type is an actual tree element. Once an expression edge
/// reaches a pattern, the generic is peeled to the header type that patterns
/// contain. Entry parameters separately expose their attribute type because
/// resource resolution changes only that localized part of the tree.
pub trait TreeFamily: Clone + std::fmt::Debug + PartialEq {
    type Header: Clone + std::fmt::Debug + PartialEq;
    type Identifier: Clone + std::fmt::Debug + PartialEq;
    type TypeHole: Clone + std::fmt::Debug + PartialEq;
}

/// Source spelling of an identifier after syntactic qualification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier {
    pub qualifiers: Vec<String>,
    pub name: String,
}

/// A type-checked identifier's semantic meaning.
///
/// `Ordinary` is a real resolution result for locals and user/module
/// definitions. Builtin overloads are always concrete here; the temporary
/// unresolved-overload state remains private to type-checking.
#[derive(Debug, Clone, PartialEq)]
pub enum IdentifierResolution {
    Ordinary,
    Builtin {
        id: BuiltinId,
        overload_idx: usize,
    },
    VecConstructor {
        target_name: String,
        arity: usize,
        target_elem: String,
    },
    Soac(SoacKind),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedIdentifier {
    pub source: Identifier,
    pub resolution: IdentifierResolution,
}

/// Which second-order array combinator a resolved identifier denotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoacKind {
    Map,
    Reduce,
    Scan,
    Filter,
    Zip,
    ReduceByIndex,
    Scatter,
}

/// The source-level `???` expression.
///
/// It is a concrete syntax node in source and typed-with-holes trees. The
/// holes-resolved tree substitutes `Infallible` for this edge, so a hole
/// cannot be represented there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TypeHole;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SourceTree;

impl TreeFamily for SourceTree {
    type Header = Header;
    type Identifier = Identifier;
    type TypeHole = TypeHole;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TypedTree;

impl TreeFamily for TypedTree {
    type Header = TypedHeader;
    type Identifier = TypedIdentifier;
    type TypeHole = TypeHole;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HolesResolvedTree;

impl TreeFamily for HolesResolvedTree {
    type Header = TypedHeader;
    type Identifier = TypedIdentifier;
    type TypeHole = std::convert::Infallible;
}

/// The phase-varying top-level structures stored in an AST.
///
/// This family is peeled immediately by [`Declaration`]. Recursive expression
/// nodes carry only [`TreeFamily`]; entry declarations additionally receive
/// the concrete attribute type stored on their parameters.
pub trait Family: Clone + std::fmt::Debug + PartialEq {
    type Tree: TreeFamily;
    type DefinitionData: Clone + std::fmt::Debug + PartialEq;
    type EntryData: Clone + std::fmt::Debug + PartialEq;
    type EntryParameterAttribute: Clone + std::fmt::Debug + PartialEq;
    type ExternData: Clone + std::fmt::Debug + PartialEq;
    type FrontendDeclaration: Clone + std::fmt::Debug + PartialEq;
}

pub trait Stage: std::fmt::Debug {
    type Family: Family;
    type GlobalContext: std::fmt::Debug;
}

#[derive(Debug)]
pub struct Program<S: Stage = crate::parser::Parsed> {
    pub declarations: Vec<Declaration<S::Family>>,
    /// The sole allocator for nodes added while this AST is rebuilt.
    pub(crate) node_ids: NodeCounter,
    pub global_context: S::GlobalContext,
}

impl<S: Stage> Program<S> {
    /// Change only the top-level checkpoint when two stages store the same
    /// tree family and global state.
    pub fn into_stage<T>(self) -> Program<T>
    where
        T: Stage<Family = S::Family, GlobalContext = S::GlobalContext>,
    {
        Program {
            declarations: self.declarations,
            node_ids: self.node_ids,
            global_context: self.global_context,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration<F: Family = crate::parser::ParsedFamily> {
    Decl(Decl<F::DefinitionData, F::Tree>),
    Entry(EntryDecl<F::EntryData, F::Tree, F::EntryParameterAttribute>),
    Extern(ExternDecl<F::ExternData>),
    Frontend(F::FrontendDeclaration),
}

/// Source-owned fields of a `let` or `def` declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct DefinitionSyntax {
    pub keyword: &'static str,
    pub attributes: Vec<Attribute>,
}

/// Typed definition data. The definition's polymorphic scheme used to be
/// recovered from a name-keyed side table during AST-to-TLC lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedDefinition {
    pub syntax: DefinitionSyntax,
    pub scheme: TypeScheme,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decl<D = DefinitionSyntax, T: TreeFamily = SourceTree> {
    pub data: D,
    pub name: String,
    pub name_span: Span,
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub params: Vec<Pattern<T::Header>>,
    pub ty: Option<Type>, // Return type for functions or type annotation for variables
    pub body: Expression<T>,
    pub param_diets: Vec<Diet>,
    pub return_diet: Diet,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter<A = Attribute> {
    pub attributes: Vec<A>,
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
    pub param_diets: Vec<Diet>,
    pub return_diet: Diet,
}

/// Source-owned fields of an external linked SPIR-V function.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternSyntax {
    pub linkage_name: String,     // SPIR-V linkage name (from #[linked("...")])
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub ty: Type,                 // Function type signature
    pub span: Span,               // Source location
    pub param_diets: Vec<Diet>,
    pub return_diet: Diet,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedExtern {
    pub syntax: ExternSyntax,
    pub scheme: TypeScheme,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternDecl<D = ExternSyntax> {
    pub name: String,
    pub data: D,
}

/// Source-owned interface fields of a shader entry declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct EntrySyntax<A = Attribute> {
    pub entry_kind: EntryKind,
    pub compute_dispatch: Option<ComputeDispatchGrid>,
    pub outputs: Vec<EntryOutputDecl<A>>,
    pub param_diets: Vec<Diet>,
    pub return_diet: Diet,
}

impl<A> EntrySyntax<A> {
    pub fn try_map_attributes<B, E>(
        self,
        mut map: impl FnMut(A) -> Result<B, E>,
    ) -> Result<EntrySyntax<B>, E> {
        Ok(EntrySyntax {
            entry_kind: self.entry_kind,
            compute_dispatch: self.compute_dispatch,
            outputs: self
                .outputs
                .into_iter()
                .map(|output| {
                    Ok(EntryOutputDecl {
                        ty: output.ty,
                        attribute: output.attribute.map(&mut map).transpose()?,
                    })
                })
                .collect::<Result<_, E>>()?,
            param_diets: self.param_diets,
            return_diet: self.return_diet,
        })
    }
}

/// Entry data after named resources and `#[view]` attributes are resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedEntry {
    pub syntax: EntrySyntax<crate::interface::ResolvedAttribute>,
    pub feedback: Vec<FeedbackPair>,
}

/// Typed entry metadata, including the entry's inferred function scheme.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedEntry {
    pub source: ResolvedEntry,
    pub scheme: TypeScheme,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl<D = EntrySyntax, T: TreeFamily = SourceTree, A = Attribute> {
    pub data: D,
    pub name: String,
    pub name_span: Span,
    pub size_params: Vec<String>,
    pub type_params: Vec<String>,
    pub params: Vec<Pattern<T::Header, A>>,
    pub body: Expression<T>,
}

/// One elaborated module or prelude definition that participates in
/// type-checking and lowering but is not a user-file top-level declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportDefinition<D, T: TreeFamily> {
    /// `Some(module)` qualifies module members; `None` denotes an
    /// automatically imported prelude definition.
    pub namespace: Option<String>,
    pub definition: Decl<D, T>,
}

/// Program-wide typed frontend state that is not intrinsically owned by one
/// user AST node.
///
/// Elaborated module/prelude bodies remain trees here, with the same in-tree
/// typing and identifier data as user definitions. The symbol namespace is
/// derived by walking these trees together with the user declarations and
/// builtin catalog; it is not stored as a second representation.
#[derive(Debug, Clone)]
pub struct TypedGlobal<D, T: TreeFamily> {
    pub support_definitions: Vec<SupportDefinition<D, T>>,
    pub warnings: Vec<crate::types::checker::TypeWarning>,
    pub builtin_names: Vec<String>,
}

// Module system types
#[derive(Debug, Clone, PartialEq)]
pub struct TypeBind {
    pub name: String,
    /// Lifted-type marker on the declaration: `type~` → `SizeLifted`,
    /// `type^` → `FullyLifted`, plain `type` → `None`. Per the spec
    /// (sections "Lifted Types" / "Higher-order Functions"), the marker
    /// controls whether the type may contain existential sizes or
    /// function types, and downstream what shapes can use it (no arrays
    /// of lifted; no fully-lifted out of if/loop). This field records the
    /// syntax for the type-checking boundary that enforces those rules.
    pub lifting: Option<TypeLifting>,
    pub type_params: Vec<TypeParam>,
    pub definition: Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeLifting {
    /// `type~` — RHS may contain existential sizes (`?[n]`).
    SizeLifted,
    /// `type^` — RHS may contain function types (transitively size-lifted too).
    FullyLifted,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeParam {
    Size(String),       // [n]
    Type(String),       // 'a
    SizeType(String),   // '~a
    LiftedType(String), // '^a
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleDecl<D = NestedDeclaration> {
    Module {
        name: String,
        signature: Option<ModuleTypeExpression>,
        body: ModuleExpression<D>,
    },
    Functor {
        name: String,
        params: Vec<ModuleParam>,
        body: ModuleExpression<D>,
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
pub enum ModuleExpression<D = NestedDeclaration> {
    Name(String),                                               // qualname
    Ascription(Box<ModuleExpression<D>>, ModuleTypeExpression), // mod_exp : mod_type_exp
    Lambda(
        Vec<ModuleParam>,
        Option<ModuleTypeExpression>,
        Box<ModuleExpression<D>>,
    ), // \ (params) [: sig] -> body
    Application(Box<ModuleExpression<D>>, Box<ModuleExpression<D>>), // mod_exp mod_exp
    Struct(Vec<D>),                                             // { dec* }
    Import(String),                                             // import "path"
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
    Sig(String, Vec<TypeParam>, Type),          // sig name type_params : type
    SigOp(String, Type),                        // sig (symbol) : type or sig symbol : type
    Type(String, Vec<TypeParam>, Option<Type>), // type declarations with optional definition
    Module(String, ModuleTypeExpression),       // module name : mod_type_exp
    Include(ModuleTypeExpression),              // include mod_type_exp
}

/// Declarations nested inside source module expressions.
///
/// Module elaboration consumes this whole syntax tree before ordinary AST
/// passes begin, so it has its own recursive enum instead of recursively
/// embedding the top-level phase family.
#[derive(Debug, Clone, PartialEq)]
pub enum NestedDeclaration {
    Decl(Decl),
    Entry(EntryDecl),
    Sig(SigDecl),
    Extern(ExternDecl),
    TypeBind(TypeBind),
    Module(ModuleDecl),
    ModuleTypeBind(ModuleTypeBind),
    Open(ModuleExpression),
    Import(String),
    Resource(crate::interface::ResourceDecl),
}

/// Top-level forms accepted directly from the parser.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedFrontend<D> {
    Sig(SigDecl),
    TypeBind(TypeBind),
    Module(ModuleDecl<D>),
    ModuleTypeBind(ModuleTypeBind),
    Open(ModuleExpression<D>),
    Import(String),
    Resource(crate::interface::ResourceDecl),
}

/// Frontend forms after file imports have been expanded.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportsResolvedFrontend<D> {
    Sig(SigDecl),
    TypeBind(TypeBind),
    Module(ModuleDecl<D>),
    ModuleTypeBind(ModuleTypeBind),
    Open(ModuleExpression<D>),
    Resource(crate::interface::ResourceDecl),
}

/// Frontend-only declarations that remain after modules have been elaborated.
#[derive(Debug, Clone, PartialEq)]
pub enum ModulesElaboratedFrontend<D> {
    Sig(SigDecl),
    TypeBind(TypeBind),
    Open(ModuleExpression<D>),
    Resource(crate::interface::ResourceDecl),
}

/// Type-checking declarations after named resources have been consumed.
#[derive(Debug, Clone, PartialEq)]
pub enum ResourcesResolvedFrontend<D> {
    Sig(SigDecl),
    TypeBind(TypeBind),
    Open(ModuleExpression<D>),
}

/// Type-system declarations that remain after `open` directives have been
/// consumed. Type checking consumes these into its environments, so they are
/// absent from the typed declaration family.
#[derive(Debug, Clone, PartialEq)]
pub enum OpensResolvedFrontend {
    Sig(SigDecl),
    TypeBind(TypeBind),
}

pub type Expression<T = SourceTree> = Node<ExprKind<T>, <T as TreeFamily>::Header>;

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind<T: TreeFamily = SourceTree> {
    IntLiteral(crate::lexer::IntString),
    FloatLiteral(f32),
    BoolLiteral(bool),
    Unit,
    Identifier(T::Identifier),
    ArrayLiteral(Vec<Expression<T>>),
    VecMatLiteral(Vec<Expression<T>>), // @[...] - vector or matrix literal (type inferred from context)
    ArrayIndex(Box<Expression<T>>, Box<Expression<T>>),
    /// Array update: `a with [i] = v`. At AST level this is always
    /// the functional form (returns a fresh array). The TLC ownership
    /// pass (`tlc::ownership::apply_ownership`) decides post-lowering
    /// whether the call should become the in-place intrinsic.
    ArrayWith {
        array: Box<Expression<T>>,
        index: Box<Expression<T>>,
        value: Box<Expression<T>>,
    },
    /// Vec swizzle update: `v with .yz = e` produces a copy of `v`
    /// with positions y and z replaced by e.x and e.y. Compound
    /// forms `*= += -= /=` desugar to `target.swizzle <op> rhs` at
    /// AST→TLC time, with the target evaluated once.
    VecWith {
        target: Box<Expression<T>>,
        /// Slot indices in source order: x→0, y→1, z→2, w→3
        /// (rgba aliased). Distinctness is enforced at parse time.
        components: Vec<u8>,
        /// `None` for plain `=`. `Some(op)` for compound `op=`,
        /// where `op` is one of `"*"`, `"+"`, `"-"`, `"/"`.
        op: Option<String>,
        value: Box<Expression<T>>,
    },
    /// Record field update: `r with x = e` or `r with a.x = e`.
    /// Lowers in TLC to a fresh record built from the original's
    /// fields with the path target replaced; nested paths recurse.
    RecordWith {
        record: Box<Expression<T>>,
        /// Field names from outer to inner. Length ≥ 1.
        path: Vec<String>,
        value: Box<Expression<T>>,
    },
    BinaryOp(BinaryOp, Box<Expression<T>>, Box<Expression<T>>),
    UnaryOp(UnaryOp, Box<Expression<T>>), // Unary operations: -, !
    Tuple(Vec<Expression<T>>),
    RecordLiteral(Vec<(String, Expression<T>)>), // e.g. {x: 1, y: 2}
    Lambda(LambdaExpr<T>),
    Application(Box<Expression<T>>, Vec<Expression<T>>), // Function application
    LetIn(LetInExpr<T>),
    FieldAccess(Box<Expression<T>>, String), // e.g. v.x, v.y
    If(IfExpr<T>),                           // if-then-else expression
    Loop(LoopExpr<T>),                       // loop expression
    Match(MatchExpr<T>),                     // match expression
    /// Sum-type constructor application: `#name arg1 arg2 ...`. With
    /// no args (`#none`), `args` is empty.
    Constructor(String, Vec<Expression<T>>),
    Range(RangeExpr<T>), // range expressions: a..b, a..<b, a..>b, a...b
    Slice(SliceExpr<T>), // array slicing: a[i:j:s]
    TypeAscription(Box<Expression<T>>, Type), // exp : type
    TypeCoercion(Box<Expression<T>>, Type), // exp :> type
    TypeHole(T::TypeHole),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambdaExpr<T: TreeFamily = SourceTree> {
    pub params: Vec<Pattern<T::Header>>,
    pub body: Box<Expression<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetInExpr<T: TreeFamily = SourceTree> {
    pub pattern: Pattern<T::Header>, // Can be Name, Tuple, etc.
    pub ty: Option<Type>,            // Optional type annotation
    pub value: Box<Expression<T>>,
    pub body: Box<Expression<T>>,
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
pub struct IfExpr<T: TreeFamily = SourceTree> {
    pub condition: Box<Expression<T>>,
    pub then_branch: Box<Expression<T>>,
    pub else_branch: Box<Expression<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopExpr<T: TreeFamily = SourceTree> {
    pub pattern: Pattern<T::Header>,      // loop variable pattern
    pub init: Option<Box<Expression<T>>>, // initial value (optional)
    pub form: LoopForm<T>,                // for/while condition
    pub body: Box<Expression<T>>,         // loop body
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopForm<T: TreeFamily = SourceTree> {
    For(String, Box<Expression<T>>),               // for name < exp
    ForIn(Pattern<T::Header>, Box<Expression<T>>), // for pat in exp
    While(Box<Expression<T>>),                     // while exp
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpr<T: TreeFamily = SourceTree> {
    pub scrutinee: Box<Expression<T>>, // expression being matched
    pub cases: Vec<MatchCase<T>>,      // case branches
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchCase<T: TreeFamily = SourceTree> {
    pub pattern: Pattern<T::Header>,
    pub body: Box<Expression<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeExpr<T: TreeFamily = SourceTree> {
    pub start: Box<Expression<T>>,
    pub step: Option<Box<Expression<T>>>, // Optional middle expression in start..step..end
    pub end: Box<Expression<T>>,
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
pub struct SliceExpr<T: TreeFamily = SourceTree> {
    pub array: Box<Expression<T>>,
    pub start: Option<Box<Expression<T>>>, // None = 0
    pub end: Option<Box<Expression<T>>>,   // None = len
}

// Pattern types for match expressions and let bindings
#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind<H = Header, A = Attribute> {
    Name(String),              // Simple name binding
    Wildcard,                  // _ wildcard
    Literal(PatternLiteral),   // Literal patterns
    Unit,                      // () unit pattern
    Tuple(Vec<Pattern<H, A>>), // (pat1, pat2, ...)
    Vec(Vec<Pattern<H, A>>),   // @[pat1, pat2, ...] — vec destructure

    Record(Vec<RecordPatternField<H, A>>), // { field1, field2 = pat, ... }
    Constructor(String, Vec<Pattern<H, A>>), // Constructor application
    Typed(Box<Pattern<H, A>>, Type),       // pat : type
    Attributed(Vec<A>, Box<Pattern<H, A>>), // #[attr] pat
}

pub type Pattern<H = Header, A = Attribute> = Node<PatternKind<H, A>, H>;

impl<H, A> Node<PatternKind<H, A>, H> {
    /// Rebuild a pattern while changing only the attribute payload stored on
    /// its `Attributed` nodes. Entry resource resolution uses this localized
    /// operation without imposing an attribute axis on the expression tree.
    pub fn try_map_attributes<B, E>(
        self,
        map: &mut impl FnMut(A) -> Result<B, E>,
    ) -> Result<Pattern<H, B>, E> {
        let Node { h, kind } = self;
        let kind = match kind {
            PatternKind::Name(name) => PatternKind::Name(name),
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Literal(value) => PatternKind::Literal(value),
            PatternKind::Unit => PatternKind::Unit,
            PatternKind::Tuple(patterns) => PatternKind::Tuple(
                patterns
                    .into_iter()
                    .map(|pattern| pattern.try_map_attributes(map))
                    .collect::<Result<_, _>>()?,
            ),
            PatternKind::Vec(patterns) => PatternKind::Vec(
                patterns
                    .into_iter()
                    .map(|pattern| pattern.try_map_attributes(map))
                    .collect::<Result<_, _>>()?,
            ),
            PatternKind::Record(fields) => PatternKind::Record(
                fields
                    .into_iter()
                    .map(|field| {
                        Ok(RecordPatternField {
                            field: field.field,
                            pattern: field
                                .pattern
                                .map(|pattern| pattern.try_map_attributes(map))
                                .transpose()?,
                        })
                    })
                    .collect::<Result<_, E>>()?,
            ),
            PatternKind::Constructor(name, patterns) => PatternKind::Constructor(
                name,
                patterns
                    .into_iter()
                    .map(|pattern| pattern.try_map_attributes(map))
                    .collect::<Result<_, _>>()?,
            ),
            PatternKind::Typed(pattern, ty) => {
                PatternKind::Typed(Box::new(pattern.try_map_attributes(map)?), ty)
            }
            PatternKind::Attributed(attributes, pattern) => PatternKind::Attributed(
                attributes.into_iter().map(&mut *map).collect::<Result<_, _>>()?,
                Box::new(pattern.try_map_attributes(map)?),
            ),
        };
        Ok(Node { h, kind })
    }

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
            PatternKind::Tuple(patterns) | PatternKind::Vec(patterns) => {
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

    /// Attributes on the outer attributed layer, through any type wrapper.
    pub fn attributes(&self) -> &[A] {
        match &self.kind {
            PatternKind::Attributed(attributes, _) => attributes,
            PatternKind::Typed(inner, _) => inner.attributes(),
            _ => &[],
        }
    }

    pub fn attributes_mut(&mut self) -> Option<&mut Vec<A>> {
        match &mut self.kind {
            PatternKind::Attributed(attributes, _) => Some(attributes),
            PatternKind::Typed(inner, _) => inner.attributes_mut(),
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
            PatternKind::Tuple(patterns) | PatternKind::Vec(patterns) => {
                patterns.iter().flat_map(|p| p.collect_names()).collect()
            }
            PatternKind::Typed(inner, _) => inner.collect_names(),
            PatternKind::Attributed(_, inner) => inner.collect_names(),
            PatternKind::Record(fields) => fields
                .iter()
                .flat_map(|f| {
                    if let Some(ref pat) = f.pattern {
                        pat.collect_names()
                    } else {
                        vec![f.field.clone()]
                    }
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
    Int(crate::lexer::IntString),
    Float(f32),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordPatternField<H = Header, A = Attribute> {
    pub field: String,
    pub pattern: Option<Pattern<H, A>>, // None means shorthand (just field name)
}
