//! Validate the invocation context of unified-pipeline operations.
//!
//! Ordinary definitions are context-polymorphic: their body is checked in the
//! context of each call site. Stage invocation callbacks change that context
//! for exactly their callback argument.

use crate::ast::{self, Declaration, ExprKind, IdentifierResolution, LoopForm, PatternKind, TypeScheme};
use crate::builtins;
use crate::builtins::BuiltinId;
use crate::err_type_at;
use crate::error::Result;
use crate::{LookupMap, LookupSet, SymbolId};

use super::run::TypeChecked;

type Expression = ast::Expression<ast::TypedTree>;
type Definition = ast::Decl<ast::TypedDefinition, ast::TypedTree>;
type CallableBindings<'a> = LookupMap<SymbolId, &'a Expression>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InvocationContext {
    Orchestration,
    Vertex,
    Fragment,
}

impl InvocationContext {
    fn description(self) -> &'static str {
        match self {
            InvocationContext::Orchestration => "orchestration context",
            InvocationContext::Vertex => "vertex callback",
            InvocationContext::Fragment => "fragment callback",
        }
    }
}

struct InvocationBuiltins {
    vertex_output: BuiltinId,
    rasterizers: LookupSet<BuiltinId>,
    shaders: LookupSet<BuiltinId>,
}

impl InvocationBuiltins {
    fn get() -> Self {
        let catalog = builtins::catalog();
        let id = |name: &str| {
            catalog
                .lookup_by_surface_name(name)
                .unwrap_or_else(|| panic!("unified invocation builtin {name} is missing"))
                .id
        };
        Self {
            vertex_output: id("vertex_output"),
            rasterizers: [
                "rasterize_triangles",
                "rasterize_triangle_strip",
                "rasterize_lines",
                "rasterize_line_strip",
                "rasterize_points",
                "rasterize_triangles_with",
                "rasterize_triangle_strip_with",
                "rasterize_lines_with",
                "rasterize_line_strip_with",
                "rasterize_points_with",
            ]
            .into_iter()
            .map(id)
            .collect(),
            shaders: ["shade", "shade_with"].into_iter().map(id).collect(),
        }
    }
}

struct Validator<'a> {
    definitions: LookupMap<SymbolId, &'a Definition>,
    active_definitions: LookupSet<(SymbolId, InvocationContext)>,
    builtins: InvocationBuiltins,
}

pub(super) fn validate(program: &TypeChecked) -> Result<()> {
    let mut definitions = LookupMap::new();
    for declaration in &program.declarations {
        if let Declaration::Decl(definition) = declaration {
            definitions.insert(definition.data.source.symbol, definition);
        }
    }
    for support in &program.global_context.support_definitions {
        definitions.insert(support.definition.data.source.symbol, &support.definition);
    }

    let mut validator = Validator {
        definitions,
        active_definitions: LookupSet::new(),
        builtins: InvocationBuiltins::get(),
    };
    for declaration in &program.declarations {
        if let Declaration::Entry(entry) = declaration {
            validator.validate_expression(
                &entry.body,
                InvocationContext::Orchestration,
                &mut CallableBindings::new(),
            )?;
        }
    }
    Ok(())
}

impl<'a> Validator<'a> {
    fn validate_expression(
        &mut self,
        expression: &'a Expression,
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        match &expression.kind {
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole(_) => Ok(()),
            ExprKind::Identifier(identifier) => {
                let IdentifierResolution::Symbol(symbol) = identifier.resolution else {
                    return Ok(());
                };
                let Some(definition) = self.definitions.get(&symbol).copied() else {
                    return Ok(());
                };
                if definition.params.is_empty() && !Self::is_function_expression(expression) {
                    self.validate_definition(symbol, &[], context, bindings)
                } else {
                    Ok(())
                }
            }
            ExprKind::ArrayLiteral(values) | ExprKind::VecMatLiteral(values) | ExprKind::Tuple(values) => {
                for value in values {
                    self.validate_expression(value, context, bindings)?;
                }
                Ok(())
            }
            ExprKind::ArrayIndex(array, index) | ExprKind::BinaryOp(_, array, index) => {
                self.validate_expression(array, context, bindings)?;
                self.validate_expression(index, context, bindings)
            }
            ExprKind::ArrayWith { array, index, value } => {
                self.validate_expression(array, context, bindings)?;
                self.validate_expression(index, context, bindings)?;
                self.validate_expression(value, context, bindings)
            }
            ExprKind::VecWith { target, value, .. } => {
                self.validate_expression(target, context, bindings)?;
                self.validate_expression(value, context, bindings)
            }
            ExprKind::RecordWith { record, value, .. } => {
                self.validate_expression(record, context, bindings)?;
                self.validate_expression(value, context, bindings)
            }
            ExprKind::UnaryOp(_, operand)
            | ExprKind::FieldAccess(operand, _)
            | ExprKind::TypeAscription(operand, _)
            | ExprKind::TypeCoercion(operand, _) => self.validate_expression(operand, context, bindings),
            ExprKind::RecordLiteral(fields) => {
                for (_, value) in fields {
                    self.validate_expression(value, context, bindings)?;
                }
                Ok(())
            }
            // Creating a function value does not evaluate its body. Its body is
            // validated when an ordinary call or stage operation invokes it.
            ExprKind::Lambda(_) => Ok(()),
            ExprKind::Application(function, arguments) => {
                self.validate_application(function, arguments, expression, context, bindings)
            }
            ExprKind::LetIn(let_in) => {
                let deferred = Self::is_function_expression(&let_in.value);
                if !deferred {
                    self.validate_expression(&let_in.value, context, bindings)?;
                }
                let symbol = Self::simple_pattern_symbol(&let_in.pattern);
                let previous = if deferred {
                    symbol.map(|symbol| (symbol, bindings.insert(symbol, &let_in.value)))
                } else {
                    None
                };
                let result = self.validate_expression(&let_in.body, context, bindings);
                if let Some((symbol, previous)) = previous {
                    if let Some(previous) = previous {
                        bindings.insert(symbol, previous);
                    } else {
                        bindings.remove(&symbol);
                    }
                }
                result
            }
            ExprKind::If(if_expression) => {
                self.validate_expression(&if_expression.condition, context, bindings)?;
                self.validate_expression(&if_expression.then_branch, context, bindings)?;
                self.validate_expression(&if_expression.else_branch, context, bindings)
            }
            ExprKind::Loop(loop_expression) => {
                if let Some(init) = &loop_expression.init {
                    self.validate_expression(init, context, bindings)?;
                }
                match &loop_expression.form {
                    LoopForm::For(_, bound) | LoopForm::ForIn(_, bound) => {
                        self.validate_expression(bound, context, bindings)?
                    }
                    LoopForm::While(condition) => self.validate_expression(condition, context, bindings)?,
                }
                self.validate_expression(&loop_expression.body, context, bindings)
            }
            ExprKind::Match(match_expression) => {
                self.validate_expression(&match_expression.scrutinee, context, bindings)?;
                for case in &match_expression.cases {
                    self.validate_expression(&case.body, context, bindings)?;
                }
                Ok(())
            }
            ExprKind::Constructor(_, arguments) => {
                for argument in arguments {
                    self.validate_expression(argument, context, bindings)?;
                }
                Ok(())
            }
            ExprKind::Range(range) => {
                self.validate_expression(&range.start, context, bindings)?;
                if let Some(step) = &range.step {
                    self.validate_expression(step, context, bindings)?;
                }
                self.validate_expression(&range.end, context, bindings)
            }
            ExprKind::Slice(slice) => {
                self.validate_expression(&slice.array, context, bindings)?;
                if let Some(start) = &slice.start {
                    self.validate_expression(start, context, bindings)?;
                }
                if let Some(end) = &slice.end {
                    self.validate_expression(end, context, bindings)?;
                }
                Ok(())
            }
        }
    }

    fn validate_application(
        &mut self,
        function: &'a Expression,
        arguments: &'a [Expression],
        application: &Expression,
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        if let Some(builtin) = Self::builtin_id(function) {
            if let Some(result) = self.validate_builtin_application(
                function,
                builtin,
                arguments,
                application.h.span,
                context,
                bindings,
            ) {
                return result;
            }
        }

        if let Some(symbol) = Self::symbol(function) {
            if let Some(target) = bindings.get(&symbol).copied() {
                return self.validate_call_target(target, arguments, application.h.span, context, bindings);
            }
            if self.definitions.contains_key(&symbol) {
                return self.validate_symbol_call(symbol, arguments, application.h.span, context, bindings);
            }
        }
        if let ExprKind::Lambda(lambda) = &function.kind {
            return self.validate_lambda_call(lambda, arguments, context, bindings);
        }

        self.validate_expression(function, context, bindings)?;
        self.validate_ordinary_arguments(arguments, context, bindings)
    }

    fn validate_builtin_application(
        &mut self,
        function: &Expression,
        builtin: BuiltinId,
        arguments: &'a [Expression],
        application_span: ast::Span,
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Option<Result<()>> {
        if self.builtins.rasterizers.contains(&builtin) {
            return Some(self.validate_stage_invocation(
                function,
                arguments,
                application_span,
                context,
                InvocationContext::Vertex,
                bindings,
            ));
        }
        if self.builtins.shaders.contains(&builtin) {
            return Some(self.validate_stage_invocation(
                function,
                arguments,
                application_span,
                context,
                InvocationContext::Fragment,
                bindings,
            ));
        }
        if builtin == self.builtins.vertex_output {
            if context != InvocationContext::Vertex {
                return Some(Err(err_type_at!(
                    application_span,
                    "vertex_output may only be evaluated in a vertex callback, not {}",
                    context.description()
                )));
            }
            return Some(self.validate_ordinary_arguments(arguments, context, bindings));
        }
        None
    }
    fn validate_stage_invocation(
        &mut self,
        function: &Expression,
        arguments: &'a [Expression],
        application_span: ast::Span,
        context: InvocationContext,
        callback_context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        if context != InvocationContext::Orchestration {
            let name = match &function.kind {
                ExprKind::Identifier(identifier) => identifier.source.name.as_str(),
                _ => "stage invocation",
            };
            return Err(err_type_at!(
                application_span,
                "stage invocation '{}' cannot be evaluated in a {}",
                name,
                context.description()
            ));
        }
        let Some((callback, ordinary_arguments)) = arguments.split_last() else {
            return Ok(());
        };
        self.validate_ordinary_arguments(ordinary_arguments, context, bindings)?;
        self.validate_callable(callback, callback_context, bindings)
    }

    fn validate_ordinary_arguments(
        &mut self,
        arguments: &'a [Expression],
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        for argument in arguments {
            if Self::is_function_expression(argument) {
                self.validate_callable(argument, context, bindings)?;
            } else {
                self.validate_expression(argument, context, bindings)?;
            }
        }
        Ok(())
    }

    fn validate_call_target(
        &mut self,
        target: &'a Expression,
        arguments: &'a [Expression],
        application_span: ast::Span,
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        if let Some(builtin) = Self::builtin_id(target) {
            if let Some(result) = self.validate_builtin_application(
                target,
                builtin,
                arguments,
                application_span,
                context,
                bindings,
            ) {
                return result;
            }
        }
        match &target.kind {
            ExprKind::Identifier(identifier) => {
                let IdentifierResolution::Symbol(symbol) = identifier.resolution else {
                    return self.validate_ordinary_arguments(arguments, context, bindings);
                };
                if let Some(next) = bindings.get(&symbol).copied() {
                    self.validate_call_target(next, arguments, application_span, context, bindings)
                } else if self.definitions.contains_key(&symbol) {
                    self.validate_symbol_call(symbol, arguments, application_span, context, bindings)
                } else {
                    self.validate_ordinary_arguments(arguments, context, bindings)
                }
            }
            ExprKind::Lambda(lambda) => self.validate_lambda_call(lambda, arguments, context, bindings),
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.validate_call_target(inner, arguments, application_span, context, bindings)
            }
            _ => {
                self.validate_expression(target, context, bindings)?;
                self.validate_ordinary_arguments(arguments, context, bindings)
            }
        }
    }

    fn validate_callable(
        &mut self,
        callable: &'a Expression,
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        match &callable.kind {
            ExprKind::Identifier(identifier) => {
                let IdentifierResolution::Symbol(symbol) = identifier.resolution else {
                    return Ok(());
                };
                if let Some(target) = bindings.get(&symbol).copied() {
                    self.validate_callable(target, context, bindings)
                } else if self.definitions.contains_key(&symbol) {
                    self.validate_symbol_call(symbol, &[], callable.h.span, context, bindings)
                } else {
                    Ok(())
                }
            }
            ExprKind::Lambda(lambda) => self.validate_lambda_call(lambda, &[], context, bindings),
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.validate_callable(inner, context, bindings)
            }
            _ => self.validate_expression(callable, context, bindings),
        }
    }

    fn validate_lambda_call(
        &mut self,
        lambda: &'a ast::LambdaExpr<ast::TypedTree>,
        arguments: &'a [Expression],
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        for argument in arguments {
            if !Self::is_function_expression(argument) {
                self.validate_expression(argument, context, bindings)?;
            }
        }
        let previous = Self::bind_parameters(&lambda.params, arguments, bindings);
        let result = self.validate_expression(&lambda.body, context, bindings);
        Self::restore_bindings(previous, bindings);
        result
    }

    fn validate_symbol_call(
        &mut self,
        symbol: SymbolId,
        arguments: &'a [Expression],
        application_span: ast::Span,
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        let Some(definition) = self.definitions.get(&symbol).copied() else {
            return Ok(());
        };
        if definition.params.is_empty() && Self::is_function_expression(&definition.body) {
            if !self.active_definitions.insert((symbol, context)) {
                return Ok(());
            }
            let result =
                self.validate_call_target(&definition.body, arguments, application_span, context, bindings);
            self.active_definitions.remove(&(symbol, context));
            result
        } else {
            self.validate_definition(symbol, arguments, context, bindings)
        }
    }
    fn validate_definition(
        &mut self,
        symbol: SymbolId,
        arguments: &'a [Expression],
        context: InvocationContext,
        bindings: &mut CallableBindings<'a>,
    ) -> Result<()> {
        let Some(definition) = self.definitions.get(&symbol).copied() else {
            return Ok(());
        };
        for argument in arguments {
            if !Self::is_function_expression(argument) {
                self.validate_expression(argument, context, bindings)?;
            }
        }
        if !self.active_definitions.insert((symbol, context)) {
            return Ok(());
        }
        let previous = Self::bind_parameters(&definition.params, arguments, bindings);
        let result = self.validate_expression(&definition.body, context, bindings);
        Self::restore_bindings(previous, bindings);
        self.active_definitions.remove(&(symbol, context));
        result
    }

    fn bind_parameters(
        parameters: &[ast::Pattern<ast::TypedTree>],
        arguments: &'a [Expression],
        bindings: &mut CallableBindings<'a>,
    ) -> Vec<(SymbolId, Option<&'a Expression>)> {
        parameters
            .iter()
            .zip(arguments)
            .filter_map(|(parameter, argument)| {
                Self::simple_pattern_symbol(parameter)
                    .map(|symbol| (symbol, bindings.insert(symbol, argument)))
            })
            .collect()
    }

    fn restore_bindings(
        previous: Vec<(SymbolId, Option<&'a Expression>)>,
        bindings: &mut CallableBindings<'a>,
    ) {
        for (symbol, value) in previous.into_iter().rev() {
            if let Some(value) = value {
                bindings.insert(symbol, value);
            } else {
                bindings.remove(&symbol);
            }
        }
    }

    fn simple_pattern_symbol(pattern: &ast::Pattern<ast::TypedTree>) -> Option<SymbolId> {
        match &pattern.kind {
            PatternKind::Name(binding) => Some(binding.symbol),
            PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
                Self::simple_pattern_symbol(inner)
            }
            _ => None,
        }
    }

    fn is_function_expression(expression: &Expression) -> bool {
        let mut scheme = &expression.h.ty;
        while let TypeScheme::Polytype { body, .. } = scheme {
            scheme = body;
        }
        matches!(scheme, TypeScheme::Monotype(ty) if super::as_arrow(ty).is_some())
    }

    fn builtin_id(expression: &Expression) -> Option<BuiltinId> {
        match &expression.kind {
            ExprKind::Identifier(identifier) => match identifier.resolution {
                IdentifierResolution::Builtin { id, .. } => Some(id),
                _ => None,
            },
            _ => None,
        }
    }

    fn symbol(expression: &Expression) -> Option<SymbolId> {
        match &expression.kind {
            ExprKind::Identifier(identifier) => match identifier.resolution {
                IdentifierResolution::Symbol(symbol) => Some(symbol),
                _ => None,
            },
            _ => None,
        }
    }
}
