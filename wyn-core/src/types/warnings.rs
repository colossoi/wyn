//! Frontend warnings and source-level unused-code analysis.

use crate::ast::{self, BindingName, ExprKind, IdentifierResolution, LoopForm, PatternKind};
use crate::ast_const_fold::{ConstantBinding, FoldedConstantUse};
use crate::interface;
use crate::name_resolution::NameResolution;
use crate::{LookupMap, LookupSet, SymbolId};
use wyn_module_graph::{PackageId, Span};

use super::run::TypeChecked;
use super::Type;

/// A non-fatal diagnostic produced by the semantic frontend.
#[derive(Debug, Clone, PartialEq)]
pub enum FrontendWarning {
    /// A type hole was filled with an inferred type.
    TypeHoleFilled {
        inferred_type: Type,
        span: Span,
    },
    /// A source binding is never referenced in its enclosing callable.
    UnusedBinding {
        name: String,
        kind: UnusedBindingKind,
        span: Span,
    },
    /// A source declaration cannot be reached from any entry point.
    UnusedDeclaration {
        name: String,
        kind: UnusedDeclarationKind,
        span: Span,
    },
}

impl FrontendWarning {
    /// Source range to highlight for this warning.
    pub const fn span(&self) -> &Span {
        match self {
            Self::TypeHoleFilled { span, .. }
            | Self::UnusedBinding { span, .. }
            | Self::UnusedDeclaration { span, .. } => span,
        }
    }

    /// Format the warning's human-readable message.
    pub fn message(&self, formatter: &dyn Fn(&Type) -> String) -> String {
        match self {
            Self::TypeHoleFilled { inferred_type, .. } => {
                format!("Hole of type {}", formatter(inferred_type))
            }
            Self::UnusedBinding { name, kind, .. } => format!(
                "unused {} `{name}`; prefix its name with `_` to silence this warning",
                kind.label()
            ),
            Self::UnusedDeclaration { name, kind, .. } => {
                format!(
                    "unused {} `{name}` is not reachable from any entry point",
                    kind.label()
                )
            }
        }
    }
}

/// Source construct that introduced an unused local value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnusedBindingKind {
    Parameter,
    LetBinding,
    LoopVariable,
    MatchBinding,
}

impl UnusedBindingKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::LetBinding => "binding",
            Self::LoopVariable => "loop variable",
            Self::MatchBinding => "pattern binding",
        }
    }
}

/// Source construct that introduced an unreachable top-level value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnusedDeclarationKind {
    Definition,
    External,
}

impl UnusedDeclarationKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Definition => "definition",
            Self::External => "external declaration",
        }
    }
}

#[derive(Clone, Copy)]
enum Callable<'a> {
    Definition(&'a ast::Decl<ast::TypedDefinition, ast::TypedTree>),
    Entry(&'a ast::EntryDecl<ast::TypedEntry, ast::TypedTree, interface::ResolvedAttribute>),
    External(&'a ast::ExternDecl<ast::TypedExtern>),
}

impl Callable<'_> {
    fn symbol(self) -> SymbolId {
        match self {
            Self::Definition(definition) => definition.data.source.symbol,
            Self::Entry(entry) => entry.data.source.symbol,
            Self::External(external) => external.data.source.symbol,
        }
    }

    fn package(self) -> Option<PackageId> {
        match self {
            Self::Definition(definition) => definition.data.source.package,
            Self::Entry(entry) => entry.data.source.package,
            Self::External(external) => external.data.source.package,
        }
    }

    fn is_entry(self) -> bool {
        matches!(self, Self::Entry(_))
    }

    fn collect_references(self, references: &mut LookupSet<SymbolId>) {
        match self {
            Self::Definition(definition) => collect_expression_references(&definition.body, references),
            Self::Entry(entry) => collect_expression_references(&entry.body, references),
            Self::External(_) => {}
        }
    }

    fn unused_declaration(self) -> Option<FrontendWarning> {
        match self {
            Self::Definition(definition) => Some(FrontendWarning::UnusedDeclaration {
                name: definition.name.clone(),
                kind: UnusedDeclarationKind::Definition,
                span: definition.name_span,
            }),
            Self::External(external) => Some(FrontendWarning::UnusedDeclaration {
                name: external.name.clone(),
                kind: UnusedDeclarationKind::External,
                span: external.data.source.syntax.span,
            }),
            Self::Entry(_) => None,
        }
    }

    fn collect_local_warnings(
        self,
        folded_uses: Option<&LookupSet<SymbolId>>,
        warnings: &mut Vec<FrontendWarning>,
    ) {
        let mut candidates = Vec::new();
        let mut used = folded_uses.cloned().unwrap_or_default();
        match self {
            Self::Definition(definition) => {
                collect_patterns(&definition.params, UnusedBindingKind::Parameter, &mut candidates);
                collect_expression(&definition.body, &mut candidates, &mut used);
            }
            Self::Entry(entry) => {
                collect_patterns(&entry.params, UnusedBindingKind::Parameter, &mut candidates);
                collect_expression(&entry.body, &mut candidates, &mut used);
            }
            Self::External(_) => return,
        }
        warnings.extend(candidates.into_iter().filter_map(|candidate| {
            (!used.contains(&candidate.symbol)).then_some(FrontendWarning::UnusedBinding {
                name: candidate.name,
                kind: candidate.kind,
                span: candidate.span,
            })
        }));
    }
}

#[derive(Debug)]
struct BindingCandidate {
    symbol: SymbolId,
    name: String,
    kind: UnusedBindingKind,
    span: Span,
}

pub(super) fn collect_unused(program: &TypeChecked) -> Vec<FrontendWarning> {
    let callables = collect_callables(program);
    let by_symbol: LookupMap<_, _> =
        callables.iter().enumerate().map(|(index, callable)| (callable.symbol(), index)).collect();
    let roots: Vec<_> = callables.iter().copied().filter(|callable| callable.is_entry()).collect();
    let executable = !roots.is_empty();
    let reachable = reachable_callables(
        &callables,
        &by_symbol,
        roots,
        &program.global_context.folded_constant_uses,
    );
    let root_package = program.source_graph().package_graph().root().package();

    let mut warnings = Vec::new();
    let mut seen = LookupSet::new();
    for callable in callables {
        if callable.package() != Some(root_package) {
            continue;
        }
        let is_reachable = !executable || reachable.contains(&callable.symbol());
        if !is_reachable {
            if let Some(warning) = callable.unused_declaration() {
                push_unique(&mut warnings, &mut seen, warning);
            }
            continue;
        }

        let mut local = Vec::new();
        callable.collect_local_warnings(
            program.global_context.folded_constant_uses.get(&callable.symbol()),
            &mut local,
        );
        for warning in local {
            push_unique(&mut warnings, &mut seen, warning);
        }
    }
    warnings
}

fn collect_callables(program: &TypeChecked) -> Vec<Callable<'_>> {
    let mut callables = Vec::new();
    for support in &program.global_context.support_definitions {
        callables.push(Callable::Definition(&support.definition));
    }
    for declaration in &program.declarations {
        callables.push(match declaration {
            ast::Declaration::Decl(definition) => Callable::Definition(definition),
            ast::Declaration::Entry(entry) => Callable::Entry(entry),
            ast::Declaration::Extern(external) => Callable::External(external),
            ast::Declaration::Frontend(never) => match *never {},
        });
    }
    callables
}

fn reachable_callables(
    callables: &[Callable<'_>],
    by_symbol: &LookupMap<SymbolId, usize>,
    roots: Vec<Callable<'_>>,
    folded_uses: &LookupMap<SymbolId, LookupSet<SymbolId>>,
) -> LookupSet<SymbolId> {
    let mut reachable = LookupSet::new();
    let mut pending: Vec<_> = roots.into_iter().map(Callable::symbol).collect();
    while let Some(symbol) = pending.pop() {
        if !reachable.insert(symbol) {
            continue;
        }
        let Some(index) = by_symbol.get(&symbol) else {
            continue;
        };
        let mut references = LookupSet::new();
        callables[*index].collect_references(&mut references);
        if let Some(folded) = folded_uses.get(&symbol) {
            references.extend(folded.iter().copied());
        }
        pending.extend(references.into_iter().filter(|reference| by_symbol.contains_key(reference)));
    }
    reachable
}

pub(super) fn resolve_folded_constant_uses(
    name_resolution: &NameResolution,
    uses: &LookupSet<FoldedConstantUse>,
) -> LookupMap<SymbolId, LookupSet<SymbolId>> {
    let mut resolved: LookupMap<SymbolId, LookupSet<SymbolId>> = LookupMap::new();
    for usage in uses {
        let Some(caller) = name_resolution.declaration_symbol(&usage.caller.name, usage.caller.span) else {
            continue;
        };
        let target = match &usage.target {
            ConstantBinding::Declaration(declaration) => {
                name_resolution.declaration_symbol(&declaration.name, declaration.span)
            }
            ConstantBinding::Pattern { node, name } => name_resolution.binding_symbol(*node, name),
        };
        if let Some(target) = target {
            resolved.entry(caller).or_default().insert(target);
        }
    }
    resolved
}

fn push_unique(
    warnings: &mut Vec<FrontendWarning>,
    seen: &mut LookupSet<(Span, String, &'static str)>,
    warning: FrontendWarning,
) {
    let (span, name, category) = match &warning {
        FrontendWarning::UnusedBinding { name, kind, span } => (*span, name.clone(), kind.label()),
        FrontendWarning::UnusedDeclaration { name, kind, span } => (*span, name.clone(), kind.label()),
        FrontendWarning::TypeHoleFilled { .. } => return,
    };
    if seen.insert((span, name, category)) {
        warnings.push(warning);
    }
}

fn collect_patterns<A>(
    patterns: &[ast::Pattern<ast::TypedTree, A>],
    kind: UnusedBindingKind,
    candidates: &mut Vec<BindingCandidate>,
) where
    A: Clone + std::fmt::Debug + PartialEq,
{
    for pattern in patterns {
        collect_pattern(pattern, kind, candidates);
    }
}

fn collect_pattern<A>(
    pattern: &ast::Pattern<ast::TypedTree, A>,
    kind: UnusedBindingKind,
    candidates: &mut Vec<BindingCandidate>,
) where
    A: Clone + std::fmt::Debug + PartialEq,
{
    match &pattern.kind {
        PatternKind::Name(binding) => push_binding(binding, pattern.h.span, kind, candidates),
        PatternKind::Tuple(patterns)
        | PatternKind::Vec(patterns)
        | PatternKind::Constructor(_, patterns) => collect_patterns(patterns, kind, candidates),
        PatternKind::Record(fields) => {
            for field in fields {
                match &field.target {
                    ast::RecordPatternTarget::Shorthand(binding) => {
                        push_binding(binding, pattern.h.span, kind, candidates);
                    }
                    ast::RecordPatternTarget::Pattern(pattern) => {
                        collect_pattern(pattern, kind, candidates);
                    }
                }
            }
        }
        PatternKind::Typed(pattern, _) | PatternKind::Attributed(_, pattern) => {
            collect_pattern(pattern, kind, candidates);
        }
        PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Unit => {}
    }
}

fn push_binding(
    binding: &ast::ResolvedBinding,
    span: Span,
    kind: UnusedBindingKind,
    candidates: &mut Vec<BindingCandidate>,
) {
    if binding.source_name().starts_with('_') {
        return;
    }
    candidates.push(BindingCandidate {
        symbol: binding.symbol,
        name: binding.source.clone(),
        kind,
        span,
    });
}

fn collect_expression_references(
    expression: &ast::Expression<ast::TypedTree>,
    references: &mut LookupSet<SymbolId>,
) {
    let mut ignored = Vec::new();
    collect_expression(expression, &mut ignored, references);
}

fn collect_expression(
    expression: &ast::Expression<ast::TypedTree>,
    candidates: &mut Vec<BindingCandidate>,
    references: &mut LookupSet<SymbolId>,
) {
    match &expression.kind {
        ExprKind::Identifier(identifier) => {
            if let IdentifierResolution::Symbol(symbol) = &identifier.resolution {
                references.insert(*symbol);
            }
        }
        ExprKind::Application(function, arguments) => {
            collect_expression(function, candidates, references);
            collect_expressions(arguments, candidates, references);
        }
        ExprKind::Lambda(lambda) => {
            collect_patterns(&lambda.params, UnusedBindingKind::Parameter, candidates);
            collect_expression(&lambda.body, candidates, references);
        }
        ExprKind::LetIn(let_in) => {
            collect_expression(&let_in.value, candidates, references);
            collect_pattern(&let_in.pattern, UnusedBindingKind::LetBinding, candidates);
            collect_expression(&let_in.body, candidates, references);
        }
        ExprKind::If(if_expression) => {
            collect_expression(&if_expression.condition, candidates, references);
            collect_expression(&if_expression.then_branch, candidates, references);
            collect_expression(&if_expression.else_branch, candidates, references);
        }
        ExprKind::FieldAccess(value, _)
        | ExprKind::UnaryOp(_, value)
        | ExprKind::Spread(value)
        | ExprKind::TypeAscription(value, _)
        | ExprKind::TypeCoercion(value, _) => collect_expression(value, candidates, references),
        ExprKind::BinaryOp(_, left, right) | ExprKind::ArrayIndex(left, right) => {
            collect_expression(left, candidates, references);
            collect_expression(right, candidates, references);
        }
        ExprKind::Tuple(values)
        | ExprKind::ArrayLiteral(values)
        | ExprKind::VecMatLiteral(values)
        | ExprKind::Constructor(_, values) => collect_expressions(values, candidates, references),
        ExprKind::ArrayWith { array, index, value } => {
            collect_expression(array, candidates, references);
            collect_expression(index, candidates, references);
            collect_expression(value, candidates, references);
        }
        ExprKind::VecWith { target, value, .. } => {
            collect_expression(target, candidates, references);
            collect_expression(value, candidates, references);
        }
        ExprKind::RecordWith { record, value, .. } => {
            collect_expression(record, candidates, references);
            collect_expression(value, candidates, references);
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, value) in fields {
                collect_expression(value, candidates, references);
            }
        }
        ExprKind::Loop(loop_expression) => {
            if let Some(init) = &loop_expression.init {
                collect_expression(init, candidates, references);
            }
            collect_pattern(
                &loop_expression.pattern,
                UnusedBindingKind::LoopVariable,
                candidates,
            );
            match &loop_expression.form {
                LoopForm::For(pattern, bound) | LoopForm::ForIn(pattern, bound) => {
                    collect_expression(bound, candidates, references);
                    collect_pattern(pattern, UnusedBindingKind::LoopVariable, candidates);
                }
                LoopForm::While(condition) => collect_expression(condition, candidates, references),
            }
            collect_expression(&loop_expression.body, candidates, references);
        }
        ExprKind::Match(match_expression) => {
            collect_expression(&match_expression.scrutinee, candidates, references);
            for case in &match_expression.cases {
                collect_pattern(&case.pattern, UnusedBindingKind::MatchBinding, candidates);
                collect_expression(&case.body, candidates, references);
            }
        }
        ExprKind::Range(range) => {
            collect_expression(&range.start, candidates, references);
            if let Some(step) = &range.step {
                collect_expression(step, candidates, references);
            }
            collect_expression(&range.end, candidates, references);
        }
        ExprKind::Slice(slice) => {
            collect_expression(&slice.array, candidates, references);
            if let Some(start) = &slice.start {
                collect_expression(start, candidates, references);
            }
            if let Some(end) = &slice.end {
                collect_expression(end, candidates, references);
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole(_) => {}
    }
}

fn collect_expressions(
    expressions: &[ast::Expression<ast::TypedTree>],
    candidates: &mut Vec<BindingCandidate>,
    references: &mut LookupSet<SymbolId>,
) {
    for expression in expressions {
        collect_expression(expression, candidates, references);
    }
}

#[cfg(test)]
#[path = "warnings_tests.rs"]
mod tests;
