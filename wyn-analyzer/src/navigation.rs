//! Source symbol index shared by editor navigation and callable information.
use std::collections::HashMap;
use wyn_core::types::run::TypeChecked;
use wyn_core::{ast, SymbolId};
use wyn_module_graph::{ModuleId, Span};

pub struct Occurrence {
    pub symbol: SymbolId,
    pub span: Span,
    pub declaration: bool,
}

#[derive(Default)]
pub struct Navigation {
    pub occurrences: Vec<Occurrence>,
    definitions: HashMap<SymbolId, Span>,
}

impl Navigation {
    pub fn new(program: &TypeChecked) -> Self {
        let mut index = Self::default();
        for declaration in &program.declarations {
            match declaration {
                ast::Declaration::Decl(def) => index.function(def),
                ast::Declaration::Entry(entry) => {
                    index.add(entry.data.source.symbol, entry.name_span, true);
                    for param in &entry.params {
                        collect_pattern(param, &mut index);
                    }
                    collect_expr(&entry.body, &mut index);
                }
                ast::Declaration::Extern(external) => {
                    index.add(
                        external.data.source.symbol,
                        external.data.source.syntax.span,
                        true,
                    );
                }
                ast::Declaration::Frontend(never) => match *never {},
            }
        }
        for support in &program.global_context.support_definitions {
            index.function(&support.definition);
        }
        for &(symbol, span) in &program.global_context.folded_constant_references {
            index.add(symbol, span, false);
        }
        // Some binding forms (record shorthand and externs) own a wider
        // syntax span. Narrow those declarations to their source name token.
        let mut tokens = HashMap::new();
        for occurrence in &mut index.occurrences {
            if !occurrence.declaration {
                continue;
            }
            let Some(module) = occurrence.span.module() else {
                continue;
            };
            let Some(name) = program.global_context.symbols.get(occurrence.symbol) else {
                continue;
            };
            let source_name = name.rsplit('.').next().unwrap_or(name);
            let module_tokens = tokens.entry(module).or_insert_with(|| {
                program
                    .source_graph()
                    .source(module)
                    .and_then(|source| wyn_core::lexer::tokenize(module, source).ok())
                    .unwrap_or_default()
            });
            if let Some(token) = module_tokens.iter().find(|token| {
                occurrence.span.contains(token.span.range().start()) && token.span.range().end() <= occurrence.span.range().end()
                    && matches!(&token.token, wyn_core::lexer::Token::Identifier(name) if name == source_name)
            }) {
                occurrence.span = token.span;
                index.definitions.insert(occurrence.symbol, token.span);
            }
        }
        index
    }

    fn function(&mut self, def: &ast::Decl<ast::TypedDefinition, ast::TypedTree>) {
        self.add(def.data.source.symbol, def.name_span, true);
        for param in &def.params {
            collect_pattern(param, self);
        }
        collect_expr(&def.body, self);
    }

    fn add(&mut self, symbol: SymbolId, span: Span, declaration: bool) {
        if span.is_generated() {
            return;
        }
        if declaration {
            self.definitions.insert(symbol, span);
        }
        self.occurrences.push(Occurrence {
            symbol,
            span,
            declaration,
        });
    }

    pub fn symbol_at(&self, module: ModuleId, offset: u32) -> Option<SymbolId> {
        self.occurrences
            .iter()
            .filter(|entry| entry.span.module() == Some(module) && entry.span.contains(offset))
            .min_by_key(|entry| entry.span.size())
            .map(|entry| entry.symbol)
    }

    pub fn definition(&self, symbol: SymbolId) -> Option<Span> {
        self.definitions.get(&symbol).copied()
    }
}

fn collect_pattern<A>(pattern: &ast::Pattern<ast::TypedTree, A>, index: &mut Navigation) {
    use ast::PatternKind::*;
    match &pattern.kind {
        Name(name) => index.add(name.symbol, pattern.h.span, true),
        Tuple(patterns) | Vec(patterns) | Constructor(_, patterns) => {
            for pattern in patterns {
                collect_pattern(pattern, index);
            }
        }
        Record(fields) => {
            for field in fields {
                match &field.target {
                    ast::RecordPatternTarget::Pattern(pattern) => collect_pattern(pattern, index),
                    ast::RecordPatternTarget::Shorthand(name) => {
                        index.add(name.symbol, pattern.h.span, true)
                    }
                }
            }
        }
        Typed(pattern, _) | Attributed(_, pattern) => collect_pattern(pattern, index),
        Wildcard | Literal(_) | Unit => {}
    }
}

fn collect_expr(expr: &ast::Expression<ast::TypedTree>, index: &mut Navigation) {
    use ast::ExprKind::*;
    match &expr.kind {
        Identifier(identifier) => {
            if let ast::IdentifierResolution::Symbol(symbol) = identifier.resolution {
                index.add(symbol, expr.h.span, false);
            }
        }
        Application(func, args) => {
            collect_expr(func, index);
            for arg in args {
                collect_expr(arg, index);
            }
        }
        Lambda(lambda) => {
            for param in &lambda.params {
                collect_pattern(param, index);
            }
            collect_expr(&lambda.body, index);
        }
        LetIn(let_in) => {
            collect_pattern(&let_in.pattern, index);
            collect_expr(&let_in.value, index);
            collect_expr(&let_in.body, index);
        }
        If(if_expr) => {
            collect_expr(&if_expr.condition, index);
            collect_expr(&if_expr.then_branch, index);
            collect_expr(&if_expr.else_branch, index);
        }
        BinaryOp(_, lhs, rhs) => {
            collect_expr(lhs, index);
            collect_expr(rhs, index);
        }
        UnaryOp(_, operand) | Spread(operand) => {
            collect_expr(operand, index);
        }
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                collect_expr(elem, index);
            }
        }
        Constructor(_, args) => {
            for arg in args {
                collect_expr(arg, index);
            }
        }
        ArrayIndex(arr, idx) => {
            collect_expr(arr, index);
            collect_expr(idx, index);
        }
        ArrayWith {
            array,
            index: subscript,
            value,
            ..
        } => {
            collect_expr(array, index);
            collect_expr(subscript, index);
            collect_expr(value, index);
        }
        VecWith {
            target: tgt, value, ..
        } => {
            collect_expr(tgt, index);
            collect_expr(value, index);
        }
        RecordWith { record, value, .. } => {
            collect_expr(record, index);
            collect_expr(value, index);
        }
        FieldAccess(base, _) => {
            collect_expr(base, index);
        }
        Loop(loop_expr) => {
            collect_pattern(&loop_expr.pattern, index);
            if let Some(init) = &loop_expr.init {
                collect_expr(init, index);
            }
            match &loop_expr.form {
                ast::LoopForm::For(pattern, value) | ast::LoopForm::ForIn(pattern, value) => {
                    collect_pattern(pattern, index);
                    collect_expr(value, index);
                }
                ast::LoopForm::While(condition) => collect_expr(condition, index),
            }
            collect_expr(&loop_expr.body, index);
        }
        RecordLiteral(fields) => {
            for (_, value) in fields {
                collect_expr(value, index);
            }
        }
        Match(match_expr) => {
            collect_expr(&match_expr.scrutinee, index);
            for case in &match_expr.cases {
                collect_pattern(&case.pattern, index);
                collect_expr(&case.body, index);
            }
        }
        TypeCoercion(inner, _) | TypeAscription(inner, _) => {
            collect_expr(inner, index);
        }
        Range(range_expr) => {
            collect_expr(&range_expr.start, index);
            if let Some(step) = &range_expr.step {
                collect_expr(step, index);
            }
            collect_expr(&range_expr.end, index);
        }
        Slice(slice_expr) => {
            collect_expr(&slice_expr.array, index);
            if let Some(start) = &slice_expr.start {
                collect_expr(start, index);
            }
            if let Some(end) = &slice_expr.end {
                collect_expr(end, index);
            }
        }
        IntLiteral(_) | FloatLiteral(_) | BoolLiteral(_) | Unit | TypeHole(_) => {}
    }
}
