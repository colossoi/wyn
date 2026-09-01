use crate::interface;
use crate::lexer;
use crate::name_resolution;
use crate::op;
use crate::types;
use crate::SymbolTable;
pub use spirv;
pub(crate) mod rebuild;

// Re-export type system types from the types module
use crate::builtins::BuiltinId;
use crate::interface::{Attribute, ComputeDispatchGrid, EntryKind, EntryOutputDecl};
pub use crate::types::{Diet, RecordFields, Type, TypeName, TypeScheme};
use crate::SymbolId;
use wyn_base::IdSource;
pub use wyn_module_graph::Span;

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

/// Source AST family produced by the Wyn parser.
pub type ParsedFamily = AstFamily<
    SourceTree,
    DefinitionSyntax,
    EntrySyntax,
    Attribute,
    ExternSyntax,
    ParsedFrontend<NestedDeclaration>,
>;

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
        self.mk_node(kind, Span::generated())
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
pub trait BindingName: Clone + std::fmt::Debug + PartialEq {
    fn source_name(&self) -> &str;
    fn symbol(&self) -> Option<SymbolId>;
}

impl BindingName for String {
    fn source_name(&self) -> &str {
        self
    }
    fn symbol(&self) -> Option<SymbolId> {
        None
    }
}

pub trait TreeFamily: Clone + std::fmt::Debug + PartialEq {
    type Header: Clone + std::fmt::Debug + PartialEq;
    type Identifier: Clone + std::fmt::Debug + PartialEq;
    /// The representation of a lexically bound name in this phase.
    type Binding: BindingName;
    type TypeHole: Clone + std::fmt::Debug + PartialEq;
}

/// Source spelling of an identifier after syntactic qualification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier {
    pub qualifiers: Vec<String>,
    pub name: String,
}

/// A source spelling paired with the binding identity assigned by name
/// resolution. The spelling survives only for diagnostics and pretty-printing;
/// every semantic edge uses `symbol`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResolvedBinding {
    pub symbol: SymbolId,
    pub source: String,
}

impl BindingName for ResolvedBinding {
    fn source_name(&self) -> &str {
        &self.source
    }
    fn symbol(&self) -> Option<SymbolId> {
        Some(self.symbol)
    }
}

/// An identifier whose semantic target was fixed by name resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedIdentifier {
    pub source: Identifier,
    pub resolution: name_resolution::ResolvedValueRef,
}

/// A type-checked identifier's semantic meaning.
///
/// `Ordinary` is a real resolution result for locals and user/module
/// definitions. Builtin overloads are always concrete here; the temporary
/// unresolved-overload state remains private to type-checking.
#[derive(Debug, Clone, PartialEq)]
pub enum IdentifierResolution {
    Symbol(SymbolId),
    Builtin {
        id: BuiltinId,
        overload_idx: usize,
    },
    VecConstructor {
        arity: usize,
        component_conversion: BuiltinId,
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
    Replicate,
    Map,
    Reduce,
    Scan,
    Filter,
    Zip,
    ReduceByIndex,
    Scatter,
    BucketScatter(u8),
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
    type Binding = String;
    type TypeHole = TypeHole;
}

/// AST shape produced by semantic name resolution and consumed by type
/// checking. Ordinary identifiers and every binder carry `SymbolId`s in-tree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolvedTree;

impl TreeFamily for ResolvedTree {
    type Header = Header;
    type Identifier = ResolvedIdentifier;
    type Binding = ResolvedBinding;
    type TypeHole = TypeHole;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TypedTree;

impl TreeFamily for TypedTree {
    type Header = TypedHeader;
    type Identifier = TypedIdentifier;
    type Binding = ResolvedBinding;
    type TypeHole = TypeHole;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HolesResolvedTree;

impl TreeFamily for HolesResolvedTree {
    type Header = TypedHeader;
    type Identifier = TypedIdentifier;
    type Binding = ResolvedBinding;
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

/// A transparent description of the concrete node payloads stored by one AST
/// representation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AstFamily<
    Tree,
    DefinitionData,
    EntryData,
    EntryParameterAttribute,
    ExternData,
    FrontendDeclaration,
>(
    std::marker::PhantomData<
        fn() -> (
            Tree,
            DefinitionData,
            EntryData,
            EntryParameterAttribute,
            ExternData,
            FrontendDeclaration,
        ),
    >,
);

impl<Tree, DefinitionData, EntryData, EntryParameterAttribute, ExternData, FrontendDeclaration> Family
    for AstFamily<Tree, DefinitionData, EntryData, EntryParameterAttribute, ExternData, FrontendDeclaration>
where
    Tree: TreeFamily,
    DefinitionData: Clone + std::fmt::Debug + PartialEq,
    EntryData: Clone + std::fmt::Debug + PartialEq,
    EntryParameterAttribute: Clone + std::fmt::Debug + PartialEq,
    ExternData: Clone + std::fmt::Debug + PartialEq,
    FrontendDeclaration: Clone + std::fmt::Debug + PartialEq,
{
    type Tree = Tree;
    type DefinitionData = DefinitionData;
    type EntryData = EntryData;
    type EntryParameterAttribute = EntryParameterAttribute;
    type ExternData = ExternData;
    type FrontendDeclaration = FrontendDeclaration;
}

#[derive(Debug)]
pub struct Program<Tag, F: Family, GlobalContext> {
    pub declarations: Vec<Declaration<F>>,
    /// The sole allocator for nodes added while this AST is rebuilt.
    pub(crate) node_ids: NodeCounter,
    pub(crate) source_graph: wyn_module_graph::SourceGraph,
    pub global_context: GlobalContext,
    pub(crate) state: std::marker::PhantomData<fn() -> Tag>,
}

impl<Tag, F: Family, GlobalContext> Program<Tag, F, GlobalContext> {
    /// Physical source, package, and import provenance for this compilation.
    pub const fn source_graph(&self) -> &wyn_module_graph::SourceGraph {
        &self.source_graph
    }

    /// Change only the program's nominal state while retaining its exact tree
    /// representation and global context.
    pub fn retag<NewTag>(self) -> Program<NewTag, F, GlobalContext> {
        self.map_global_context(std::convert::identity)
    }

    /// Change the program-wide context while preserving declarations and the
    /// node allocator.
    pub fn map_global_context<NewTag, NewGlobalContext>(
        self,
        map: impl FnOnce(GlobalContext) -> NewGlobalContext,
    ) -> Program<NewTag, F, NewGlobalContext> {
        Program {
            declarations: self.declarations,
            node_ids: self.node_ids,
            source_graph: self.source_graph,
            global_context: map(self.global_context),
            state: std::marker::PhantomData,
        }
    }

    /// Rebuild a whole AST checkpoint while retaining its node allocator.
    ///
    /// The callback owns declaration cardinality and ordering, so the same
    /// primitive supports ordinary maps, filtered frontend declarations, and
    /// import expansion.
    pub fn try_rebuild<NewTag, NewF, NewGlobalContext, E>(
        self,
        rebuild: impl FnOnce(
            Vec<Declaration<F>>,
            GlobalContext,
            &mut NodeCounter,
        ) -> Result<(Vec<Declaration<NewF>>, NewGlobalContext), E>,
    ) -> Result<Program<NewTag, NewF, NewGlobalContext>, E>
    where
        NewF: Family,
    {
        let Program {
            declarations,
            mut node_ids,
            source_graph,
            global_context,
            state: _,
        } = self;
        let (declarations, global_context) = rebuild(declarations, global_context, &mut node_ids)?;
        Ok(Program {
            declarations,
            node_ids,
            source_graph,
            global_context,
            state: std::marker::PhantomData,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration<F: Family = ParsedFamily> {
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

/// Definition data immediately after semantic name resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct NameResolvedDefinition {
    pub syntax: DefinitionSyntax,
    pub symbol: SymbolId,
}

/// Typed definition data. Identity is inherited from name resolution rather
/// than allocated or recovered during AST-to-TLC lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedDefinition {
    pub source: NameResolvedDefinition,
    pub scheme: TypeScheme,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decl<D = DefinitionSyntax, T: TreeFamily = SourceTree> {
    pub data: D,
    pub name: String,
    pub name_span: Span,
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub params: Vec<Pattern<T>>,
    pub ty: Option<Type>, // Return type for functions or type annotation for variables
    pub body: Expression<T>,
    pub param_diets: Vec<Diet>,
    pub return_diet: Diet,
}

impl<D, T: TreeFamily> Decl<D, T> {
    /// Rebuild declaration-owned data and recursive trees while carrying all
    /// source-level signature fields through unchanged.
    pub fn try_rebuild<NewD, NewT: TreeFamily, E>(
        self,
        rebuild_data: impl FnOnce(D, &str, Span) -> Result<NewD, E>,
        rebuild_trees: impl FnOnce(
            Vec<Pattern<T>>,
            Expression<T>,
        ) -> Result<(Vec<Pattern<NewT>>, Expression<NewT>), E>,
    ) -> Result<Decl<NewD, NewT>, E> {
        let Decl {
            data,
            name,
            name_span,
            size_params,
            type_params,
            params,
            ty,
            body,
            param_diets,
            return_diet,
        } = self;
        let data = rebuild_data(data, &name, name_span)?;
        let (params, body) = rebuild_trees(params, body)?;
        Ok(Decl {
            data,
            name,
            name_span,
            size_params,
            type_params,
            params,
            ty,
            body,
            param_diets,
            return_diet,
        })
    }
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
    pub source: NameResolvedExtern,
    pub scheme: TypeScheme,
}

/// Internal identity plus the explicitly textual external ABI contract.
#[derive(Debug, Clone, PartialEq)]
pub struct NameResolvedExtern {
    pub syntax: ExternSyntax,
    pub symbol: SymbolId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternDecl<D = ExternSyntax> {
    pub name: String,
    pub data: D,
}

impl<D> ExternDecl<D> {
    /// Rebuild extern-owned data while carrying its source name through.
    pub fn try_map_data<NewD, E>(
        self,
        map: impl FnOnce(D, &str) -> Result<NewD, E>,
    ) -> Result<ExternDecl<NewD>, E> {
        Ok(ExternDecl {
            data: map(self.data, &self.name)?,
            name: self.name,
        })
    }
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
    pub syntax: EntrySyntax<interface::ResolvedAttribute>,
}

/// Entry data immediately after semantic name resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct NameResolvedEntry {
    pub source: ResolvedEntry,
    pub symbol: SymbolId,
}

/// Typed entry metadata, including the entry's inferred function scheme and
/// the identity inherited from name resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedEntry {
    pub source: NameResolvedEntry,
    pub scheme: TypeScheme,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl<D = EntrySyntax, T: TreeFamily = SourceTree, A = Attribute> {
    pub data: D,
    pub name: String,
    pub name_span: Span,
    pub size_params: Vec<String>,
    pub type_params: Vec<String>,
    pub params: Vec<Pattern<T, A>>,
    pub body: Expression<T>,
}

impl<D, T: TreeFamily, A> EntryDecl<D, T, A> {
    /// Rebuild entry-owned data and recursive trees while carrying its source
    /// signature through unchanged.
    pub fn try_rebuild<NewD, NewT: TreeFamily, NewA, E>(
        self,
        rebuild_data: impl FnOnce(D, &str, Span) -> Result<NewD, E>,
        rebuild_trees: impl FnOnce(
            Vec<Pattern<T, A>>,
            Expression<T>,
        ) -> Result<(Vec<Pattern<NewT, NewA>>, Expression<NewT>), E>,
    ) -> Result<EntryDecl<NewD, NewT, NewA>, E> {
        let EntryDecl {
            data,
            name,
            name_span,
            size_params,
            type_params,
            params,
            body,
        } = self;
        let data = rebuild_data(data, &name, name_span)?;
        let (params, body) = rebuild_trees(params, body)?;
        Ok(EntryDecl {
            data,
            name,
            name_span,
            size_params,
            type_params,
            params,
            body,
        })
    }
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

impl<D, T: TreeFamily> SupportDefinition<D, T> {
    pub fn try_map_definition<NewD, NewT: TreeFamily, E>(
        self,
        map: impl FnOnce(Decl<D, T>) -> Result<Decl<NewD, NewT>, E>,
    ) -> Result<SupportDefinition<NewD, NewT>, E> {
        Ok(SupportDefinition {
            namespace: self.namespace,
            definition: map(self.definition)?,
        })
    }
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
    /// Sole source-binding identity arena, carried into TLC without reallocation.
    pub symbols: SymbolTable,
    pub warnings: Vec<types::checker::TypeWarning>,
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

/// A physical source import before its target has been resolved.
///
/// Its identity is local to the containing source file. The parser keeps the
/// source spelling so the module-graph frontend can interpret package and
/// relative paths without teaching the parser those rules.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceImport {
    pub(crate) site: wyn_module_graph::ImportSiteId,
    pub(crate) path: String,
    pub(crate) span: Span,
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
    Import(SourceImport),                                       // import "path"
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
    Import(SourceImport),
    Resource(interface::ResourceDecl),
}

/// Top-level forms accepted directly from the parser.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedFrontend<D> {
    Sig(SigDecl),
    TypeBind(TypeBind),
    Module(ModuleDecl<D>),
    ModuleTypeBind(ModuleTypeBind),
    Open(ModuleExpression<D>),
    Import(SourceImport),
    Resource(interface::ResourceDecl),
}

/// Frontend forms after file imports have been expanded.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportsResolvedFrontend<D> {
    Sig(SigDecl),
    TypeBind(TypeBind),
    Module(ModuleDecl<D>),
    ModuleTypeBind(ModuleTypeBind),
    Open(ModuleExpression<D>),
    Resource(interface::ResourceDecl),
}

/// Frontend-only declarations that remain after modules have been elaborated.
#[derive(Debug, Clone, PartialEq)]
pub enum ModulesElaboratedFrontend<D> {
    Sig(SigDecl),
    TypeBind(TypeBind),
    Open(ModuleExpression<D>),
    Resource(interface::ResourceDecl),
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
    IntLiteral(lexer::IntString),
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
        op: Option<op::BinaryOperator>,
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
    pub params: Vec<Pattern<T>>,
    pub body: Box<Expression<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetInExpr<T: TreeFamily = SourceTree> {
    pub pattern: Pattern<T>, // Can be Name, Tuple, etc.
    pub ty: Option<Type>,    // Optional type annotation
    pub value: Box<Expression<T>>,
    pub body: Box<Expression<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOp {
    pub op: op::BinaryOperator,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryOp {
    pub op: op::UnaryOperator,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr<T: TreeFamily = SourceTree> {
    pub condition: Box<Expression<T>>,
    pub then_branch: Box<Expression<T>>,
    pub else_branch: Box<Expression<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopExpr<T: TreeFamily = SourceTree> {
    pub pattern: Pattern<T>,              // loop variable pattern
    pub init: Option<Box<Expression<T>>>, // initial value (optional)
    pub form: LoopForm<T>,                // for/while condition
    pub body: Box<Expression<T>>,         // loop body
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopForm<T: TreeFamily = SourceTree> {
    For(Pattern<T>, Box<Expression<T>>),   // for name < exp
    ForIn(Pattern<T>, Box<Expression<T>>), // for pat in exp
    While(Box<Expression<T>>),             // while exp
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpr<T: TreeFamily = SourceTree> {
    pub scrutinee: Box<Expression<T>>, // expression being matched
    pub cases: Vec<MatchCase<T>>,      // case branches
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchCase<T: TreeFamily = SourceTree> {
    pub pattern: Pattern<T>,
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
pub enum PatternKind<T: TreeFamily = SourceTree, A = Attribute> {
    Name(T::Binding),
    Wildcard,
    Literal(PatternLiteral),
    Unit,
    Tuple(Vec<Pattern<T, A>>),
    Vec(Vec<Pattern<T, A>>),
    Record(Vec<RecordPatternField<T, A>>),
    Constructor(String, Vec<Pattern<T, A>>),
    Typed(Box<Pattern<T, A>>, Type),
    Attributed(Vec<A>, Box<Pattern<T, A>>),
}

pub type Pattern<T = SourceTree, A = Attribute> = Node<PatternKind<T, A>, <T as TreeFamily>::Header>;

impl<T: TreeFamily, A> Node<PatternKind<T, A>, T::Header> {
    /// Rebuild a pattern while changing only the attribute payload stored on
    /// its `Attributed` nodes. Entry resource resolution uses this localized
    /// operation without imposing an attribute axis on the expression tree.
    pub fn try_map_attributes<B, E>(
        self,
        map: &mut impl FnMut(A) -> Result<B, E>,
    ) -> Result<Pattern<T, B>, E> {
        self::rebuild::pattern_with(
            self,
            &mut |header| Ok(header),
            &mut |_header, binding| Ok(binding),
            map,
            &mut |ty| Ok(ty),
        )
    }

    /// Extract the simple name from a pattern if possible
    /// For Name("x") returns Some("x")
    /// For Typed(Name("x"), _) returns Some("x")
    /// For Attributed(_, Name("x")) returns Some("x")
    /// Returns None for complex patterns like tuples, records, etc.
    pub fn simple_name(&self) -> Option<&str> {
        match &self.kind {
            PatternKind::Name(name) => Some(name.source_name()),
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
            PatternKind::Name(name) => names.push(name.source_name().to_owned()),
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
                    match &field.target {
                        RecordPatternTarget::Shorthand(binding) => {
                            names.push(binding.source_name().to_owned())
                        }
                        RecordPatternTarget::Pattern(pattern) => pattern.collect_bound_names(names),
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
            PatternKind::Name(name) => vec![name.source_name().to_owned()],
            PatternKind::Tuple(patterns) | PatternKind::Vec(patterns) => {
                patterns.iter().flat_map(|p| p.collect_names()).collect()
            }
            PatternKind::Typed(inner, _) => inner.collect_names(),
            PatternKind::Attributed(_, inner) => inner.collect_names(),
            PatternKind::Record(fields) => fields
                .iter()
                .flat_map(|field| match &field.target {
                    RecordPatternTarget::Shorthand(binding) => vec![binding.source_name().to_owned()],
                    RecordPatternTarget::Pattern(pattern) => pattern.collect_names(),
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
    Int(lexer::IntString),
    Float(f32),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecordPatternTarget<T: TreeFamily = SourceTree, A = Attribute> {
    /// `{x}` binds the field value. The binding is resolved in-tree.
    Shorthand(T::Binding),
    /// `{x = pattern}` uses an explicit nested pattern.
    Pattern(Pattern<T, A>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordPatternField<T: TreeFamily = SourceTree, A = Attribute> {
    pub field: String,
    pub target: RecordPatternTarget<T, A>,
}
