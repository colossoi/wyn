use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use wyn_core::FrontEnd;
use wyn_core::TypeTable;
use wyn_core::ast::{self, NodeCounter, NodeId, Span};
use wyn_core::module_manager::{ModuleManager, PreElaboratedPrelude};
use wyn_core::types::{Type, TypeName, TypeScheme, format_scheme};

/// Cached prelude data AND the node counter state after parsing it
static PRELUDE_CACHE: OnceLock<(PreElaboratedPrelude, NodeCounter)> = OnceLock::new();

fn get_prelude() -> (&'static PreElaboratedPrelude, NodeCounter) {
    let (prelude, counter) = PRELUDE_CACHE.get_or_init(|| {
        let mut nc = NodeCounter::new();
        let prelude = ModuleManager::create_prelude(&mut nc).expect("Failed to create prelude cache");
        (prelude, nc)
    });
    (prelude, counter.clone())
}

fn get_frontend() -> FrontEnd {
    let (prelude, node_counter) = get_prelude();
    FrontEnd::new_from_prelude(prelude.clone(), node_counter)
}

/// Cached document state after successful type checking
struct DocumentState {
    ast: ast::Program,
    type_table: TypeTable,
    /// Top-level function type schemes (name -> scheme)
    schemes: HashMap<String, TypeScheme>,
}

struct Backend {
    client: Client,
    documents: Arc<RwLock<HashMap<Url, DocumentState>>>,
}

impl Backend {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "wyn-analyzer".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: None,
                    work_done_progress_options: Default::default(),
                }),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client.log_message(MessageType::INFO, "wyn-analyzer initialized").await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.on_change(TextDocumentItem {
            uri: params.text_document.uri,
            language_id: params.text_document.language_id,
            version: params.text_document.version,
            text: params.text_document.text,
        })
        .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().next() {
            self.on_change(TextDocumentItem {
                uri: params.text_document.uri,
                language_id: "wyn".to_string(),
                version: params.text_document.version,
                text: change.text,
            })
            .await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        // Remove document from cache when closed
        if let Ok(mut docs) = self.documents.write() {
            docs.remove(&params.text_document.uri);
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        // Convert 0-based LSP position to 1-based internal
        let line = pos.line as usize + 1;
        let col = pos.character as usize + 1;

        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            // First check if cursor is on a declaration name
            if let Some((name, kind)) = find_declaration_name_at(&doc.ast, line, col) {
                if let Some(scheme) = doc.schemes.get(&name) {
                    let type_str = format_scheme(scheme);
                    return Ok(Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: format!("```wyn\n{}: {}\n```", kind, type_str),
                        }),
                        range: None,
                    }));
                }
            }

            // Fall back to expression type lookup
            if let Some((node_id, span)) = find_node_at_position(&doc.ast, line, col) {
                if let Some(scheme) = doc.type_table.get(&node_id) {
                    let type_str = format_scheme(scheme);
                    return Ok(Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: format!("```wyn\n{}\n```", type_str),
                        }),
                        range: Some(Range {
                            start: Position {
                                line: span.start_line.saturating_sub(1) as u32,
                                character: span.start_col.saturating_sub(1) as u32,
                            },
                            end: Position {
                                line: span.end_line.saturating_sub(1) as u32,
                                character: span.end_col.saturating_sub(1) as u32,
                            },
                        }),
                    }));
                }
            }
        }

        Ok(None)
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;

        // Check if triggered by '.'
        let is_dot_trigger = params
            .context
            .as_ref()
            .and_then(|ctx| ctx.trigger_character.as_ref())
            .map(|c| c == ".")
            .unwrap_or(false);

        if is_dot_trigger {
            // Field access completion
            let line = pos.line as usize + 1;
            let col = pos.character.saturating_sub(1) as usize + 1;

            let docs = self.documents.read().ok();
            let doc = docs.as_ref().and_then(|d| d.get(uri));

            if let Some(doc) = doc {
                if let Some((node_id, _span)) = find_node_at_position(&doc.ast, line, col) {
                    if let Some(scheme) = doc.type_table.get(&node_id) {
                        let items = get_field_completions(scheme);
                        if !items.is_empty() {
                            return Ok(Some(CompletionResponse::Array(items)));
                        }
                    }
                }
            }
        }

        // Default: prelude function completions
        let (prelude, _) = get_prelude();
        let items: Vec<CompletionItem> = prelude
            .prelude_functions
            .iter()
            .map(|(name, _decl)| CompletionItem {
                label: name.clone(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("prelude function".to_string()),
                ..Default::default()
            })
            .collect();

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let line = pos.line as usize + 1;
        let col = pos.character as usize + 1;

        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            if let Some(def_span) = find_definition(&doc.ast, line, col) {
                let range = Range {
                    start: Position {
                        line: def_span.start_line.saturating_sub(1) as u32,
                        character: def_span.start_col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: def_span.end_line.saturating_sub(1) as u32,
                        character: def_span.end_col.saturating_sub(1) as u32,
                    },
                };
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range,
                })));
            }
        }

        Ok(None)
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            let symbols: Vec<DocumentSymbol> =
                doc.ast.declarations.iter().filter_map(declaration_to_symbol).collect();
            return Ok(Some(DocumentSymbolResponse::Nested(symbols)));
        }

        Ok(None)
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let line = pos.line as usize + 1;
        let col = pos.character as usize + 1;

        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            if let Some((func_name, arg_index)) = find_application_context(&doc.ast, line, col) {
                if let Some(scheme) = doc.schemes.get(&func_name) {
                    let label = format!("{}: {}", func_name, format_scheme(scheme));
                    return Ok(Some(SignatureHelp {
                        signatures: vec![SignatureInformation {
                            label,
                            documentation: None,
                            parameters: None,
                            active_parameter: Some(arg_index as u32),
                        }],
                        active_signature: Some(0),
                        active_parameter: Some(arg_index as u32),
                    }));
                }

                let (prelude, _) = get_prelude();
                if prelude.prelude_functions.contains_key(&func_name) {
                    return Ok(Some(SignatureHelp {
                        signatures: vec![SignatureInformation {
                            label: format!("{} (prelude function)", func_name),
                            documentation: None,
                            parameters: None,
                            active_parameter: Some(arg_index as u32),
                        }],
                        active_signature: Some(0),
                        active_parameter: Some(arg_index as u32),
                    }));
                }
            }
        }

        Ok(None)
    }
}

impl Backend {
    async fn on_change(&self, doc: TextDocumentItem) {
        let (diagnostics, state) = self.check_document(&doc.text);

        if let Some(state) = state {
            if let Ok(mut docs) = self.documents.write() {
                docs.insert(doc.uri.clone(), state);
            }
        }

        self.client.publish_diagnostics(doc.uri, diagnostics, Some(doc.version)).await;
    }

    fn check_document(&self, text: &str) -> (Vec<Diagnostic>, Option<DocumentState>) {
        let mut diagnostics = Vec::new();

        let mut frontend = get_frontend();
        let result = wyn_core::Compiler::parse(text, &mut frontend.node_counter).and_then(|parsed| {
            parsed
                .desugar(&mut frontend.node_counter)?
                .resolve(&frontend.module_manager)?
                .fold_ast_constants()
                .type_check(&frontend.module_manager, &mut frontend.schemes)
        });

        match result {
            Ok(type_checked) => {
                let state = DocumentState {
                    ast: type_checked.ast,
                    type_table: type_checked.type_table,
                    schemes: frontend.schemes,
                };
                (diagnostics, Some(state))
            }
            Err(e) => {
                let range = if let Some(span) = e.span() {
                    Range {
                        start: Position {
                            line: span.start_line.saturating_sub(1) as u32,
                            character: span.start_col.saturating_sub(1) as u32,
                        },
                        end: Position {
                            line: span.end_line.saturating_sub(1) as u32,
                            character: span.end_col.saturating_sub(1) as u32,
                        },
                    }
                } else {
                    Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 1,
                        },
                    }
                };

                diagnostics.push(Diagnostic {
                    range,
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("wyn-analyzer".to_string()),
                    message: e.to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                });

                (diagnostics, None)
            }
        }
    }
}

/// Find the smallest AST node containing the given position
fn find_node_at_position(ast: &ast::Program, line: usize, col: usize) -> Option<(NodeId, Span)> {
    let mut best: Option<(NodeId, Span)> = None;

    for decl in &ast.declarations {
        find_in_declaration(decl, line, col, &mut best);
    }

    best
}

fn find_in_declaration(
    decl: &ast::Declaration,
    line: usize,
    col: usize,
    best: &mut Option<(NodeId, Span)>,
) {
    match decl {
        ast::Declaration::Decl(def) => {
            find_in_expr(&def.body, line, col, best);
        }
        ast::Declaration::Entry(entry) => {
            find_in_expr(&entry.body, line, col, best);
        }
        _ => {}
    }
}

fn find_in_expr(expr: &ast::Expression, line: usize, col: usize, best: &mut Option<(NodeId, Span)>) {
    let span = expr.h.span;

    if !span.contains(line, col) {
        return;
    }

    let dominated = best.as_ref().is_none_or(|(_, best_span)| span.size() < best_span.size());
    if dominated {
        *best = Some((expr.h.id, span));
    }

    use ast::ExprKind::*;
    match &expr.kind {
        IntLiteral(_) | FloatLiteral(_) | BoolLiteral(_) | StringLiteral(_) | Unit => {}
        Identifier(_, _) | TypeHole => {}
        Application(func, args) => {
            find_in_expr(func, line, col, best);
            for arg in args {
                find_in_expr(arg, line, col, best);
            }
        }
        Lambda(lambda) => {
            find_in_expr(&lambda.body, line, col, best);
        }
        LetIn(let_in) => {
            find_in_expr(&let_in.value, line, col, best);
            find_in_expr(&let_in.body, line, col, best);
        }
        If(if_expr) => {
            find_in_expr(&if_expr.condition, line, col, best);
            find_in_expr(&if_expr.then_branch, line, col, best);
            find_in_expr(&if_expr.else_branch, line, col, best);
        }
        BinaryOp(_, lhs, rhs) => {
            find_in_expr(lhs, line, col, best);
            find_in_expr(rhs, line, col, best);
        }
        UnaryOp(_, operand) => {
            find_in_expr(operand, line, col, best);
        }
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                find_in_expr(elem, line, col, best);
            }
        }
        ArrayIndex(arr, idx) => {
            find_in_expr(arr, line, col, best);
            find_in_expr(idx, line, col, best);
        }
        ArrayWith { array, index, value } => {
            find_in_expr(array, line, col, best);
            find_in_expr(index, line, col, best);
            find_in_expr(value, line, col, best);
        }
        FieldAccess(base, _) => {
            find_in_expr(base, line, col, best);
        }
        Loop(loop_expr) => {
            if let Some(init) = &loop_expr.init {
                find_in_expr(init, line, col, best);
            }
            find_in_expr(&loop_expr.body, line, col, best);
        }
        RecordLiteral(fields) => {
            for (_, value) in fields {
                find_in_expr(value, line, col, best);
            }
        }
        Match(match_expr) => {
            find_in_expr(&match_expr.scrutinee, line, col, best);
            for case in &match_expr.cases {
                find_in_expr(&case.body, line, col, best);
            }
        }
        TypeCoercion(inner, _) | TypeAscription(inner, _) => {
            find_in_expr(inner, line, col, best);
        }
        Range(range_expr) => {
            find_in_expr(&range_expr.start, line, col, best);
            if let Some(step) = &range_expr.step {
                find_in_expr(step, line, col, best);
            }
            find_in_expr(&range_expr.end, line, col, best);
        }
        Slice(slice_expr) => {
            find_in_expr(&slice_expr.array, line, col, best);
            if let Some(start) = &slice_expr.start {
                find_in_expr(start, line, col, best);
            }
            if let Some(end) = &slice_expr.end {
                find_in_expr(end, line, col, best);
            }
        }
    }
}

/// Find the application context at cursor position
fn find_application_context(ast: &ast::Program, line: usize, col: usize) -> Option<(String, usize)> {
    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(def) => {
                if let Some(result) = find_application_in_expr(&def.body, line, col) {
                    return Some(result);
                }
            }
            ast::Declaration::Entry(entry) => {
                if let Some(result) = find_application_in_expr(&entry.body, line, col) {
                    return Some(result);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_application_in_expr(expr: &ast::Expression, line: usize, col: usize) -> Option<(String, usize)> {
    let span = expr.h.span;
    if !span.contains(line, col) {
        return None;
    }

    use ast::ExprKind::*;
    match &expr.kind {
        Application(func, args) => {
            for (i, arg) in args.iter().enumerate() {
                if arg.h.span.contains(line, col) {
                    if let Some(result) = find_application_in_expr(arg, line, col) {
                        return Some(result);
                    }
                    if let Identifier(_, name) = &func.kind {
                        return Some((name.clone(), i));
                    }
                }
            }
            if let Identifier(_, name) = &func.kind {
                return Some((name.clone(), args.len()));
            }
        }
        Lambda(lambda) => {
            return find_application_in_expr(&lambda.body, line, col);
        }
        LetIn(let_in) => {
            if let Some(r) = find_application_in_expr(&let_in.value, line, col) {
                return Some(r);
            }
            return find_application_in_expr(&let_in.body, line, col);
        }
        If(if_expr) => {
            if let Some(r) = find_application_in_expr(&if_expr.condition, line, col) {
                return Some(r);
            }
            if let Some(r) = find_application_in_expr(&if_expr.then_branch, line, col) {
                return Some(r);
            }
            return find_application_in_expr(&if_expr.else_branch, line, col);
        }
        BinaryOp(_, lhs, rhs) => {
            if let Some(r) = find_application_in_expr(lhs, line, col) {
                return Some(r);
            }
            return find_application_in_expr(rhs, line, col);
        }
        UnaryOp(_, operand) => {
            return find_application_in_expr(operand, line, col);
        }
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                if let Some(r) = find_application_in_expr(elem, line, col) {
                    return Some(r);
                }
            }
        }
        ArrayIndex(arr, idx) => {
            if let Some(r) = find_application_in_expr(arr, line, col) {
                return Some(r);
            }
            return find_application_in_expr(idx, line, col);
        }
        _ => {}
    }
    None
}

/// Get field completion items based on a type scheme
fn get_field_completions(scheme: &TypeScheme) -> Vec<CompletionItem> {
    fn unwrap_scheme(scheme: &TypeScheme) -> Option<&Type> {
        match scheme {
            TypeScheme::Monotype(t) => Some(t),
            TypeScheme::Polytype { body, .. } => unwrap_scheme(body.as_ref()),
        }
    }

    let Some(ty) = unwrap_scheme(scheme) else {
        return vec![];
    };

    let mut items = Vec::new();

    if let Type::Constructed(name, args) = ty {
        match name {
            TypeName::Record(fields) => {
                for (i, field_name) in fields.iter().enumerate() {
                    let field_ty = args.get(i).map(|t| format_scheme(&TypeScheme::Monotype(t.clone())));
                    items.push(CompletionItem {
                        label: field_name.clone(),
                        kind: Some(CompletionItemKind::FIELD),
                        detail: field_ty,
                        ..Default::default()
                    });
                }
            }
            TypeName::Vec => {
                for c in ["x", "y", "z", "w"] {
                    items.push(CompletionItem {
                        label: c.to_string(),
                        kind: Some(CompletionItemKind::PROPERTY),
                        detail: Some("component".to_string()),
                        ..Default::default()
                    });
                }
                for c in ["r", "g", "b", "a"] {
                    items.push(CompletionItem {
                        label: c.to_string(),
                        kind: Some(CompletionItemKind::PROPERTY),
                        detail: Some("color component".to_string()),
                        ..Default::default()
                    });
                }
            }
            _ => {}
        }
    }

    items
}

/// Find if cursor is on a declaration name
fn find_declaration_name_at(ast: &ast::Program, line: usize, col: usize) -> Option<(String, &'static str)> {
    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(def) => {
                let body_span = def.body.h.span;
                if line == body_span.start_line && col < body_span.start_col {
                    return Some((def.name.clone(), def.keyword));
                }
                if line < body_span.start_line && line >= body_span.start_line.saturating_sub(5) {
                    return Some((def.name.clone(), def.keyword));
                }
            }
            ast::Declaration::Entry(entry) => {
                let body_span = entry.body.h.span;
                if line == body_span.start_line && col < body_span.start_col {
                    let kind = match &entry.entry_type {
                        ast::Attribute::Vertex => "vertex",
                        ast::Attribute::Fragment => "fragment",
                        ast::Attribute::Compute(_) => "compute",
                        _ => "entry",
                    };
                    return Some((entry.name.clone(), kind));
                }
            }
            _ => {}
        }
    }
    None
}

/// Find the definition site of an identifier at the given position
fn find_definition(ast: &ast::Program, line: usize, col: usize) -> Option<Span> {
    let bindings: Vec<(String, Span)> = Vec::new();

    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(def) => {
                let param_bindings: Vec<_> = def
                    .params
                    .iter()
                    .flat_map(|p| p.collect_names().into_iter().map(|n| (n, p.h.span)))
                    .collect();

                if let Some(span) = find_definition_in_expr(
                    &def.body,
                    line,
                    col,
                    &mut bindings.iter().chain(param_bindings.iter()).cloned().collect(),
                ) {
                    return Some(span);
                }
            }
            ast::Declaration::Entry(entry) => {
                let param_bindings: Vec<_> = entry
                    .params
                    .iter()
                    .flat_map(|p| p.collect_names().into_iter().map(|n| (n, p.h.span)))
                    .collect();

                if let Some(span) = find_definition_in_expr(
                    &entry.body,
                    line,
                    col,
                    &mut bindings.iter().chain(param_bindings.iter()).cloned().collect(),
                ) {
                    return Some(span);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_definition_in_expr(
    expr: &ast::Expression,
    line: usize,
    col: usize,
    bindings: &mut Vec<(String, Span)>,
) -> Option<Span> {
    let span = expr.h.span;
    if !span.contains(line, col) {
        return None;
    }

    use ast::ExprKind::*;
    match &expr.kind {
        Identifier(_qualifiers, name) => {
            if span.contains(line, col) && span.size() < 100 {
                for (bound_name, bound_span) in bindings.iter().rev() {
                    if bound_name == name {
                        return Some(*bound_span);
                    }
                }
            }
            None
        }
        Lambda(lambda) => {
            let saved_len = bindings.len();
            for param in &lambda.params {
                for name in param.collect_names() {
                    bindings.push((name, param.h.span));
                }
            }
            let result = find_definition_in_expr(&lambda.body, line, col, bindings);
            bindings.truncate(saved_len);
            result
        }
        LetIn(let_in) => {
            if let Some(span) = find_definition_in_expr(&let_in.value, line, col, bindings) {
                return Some(span);
            }

            let saved_len = bindings.len();
            for name in let_in.pattern.collect_names() {
                bindings.push((name, let_in.pattern.h.span));
            }
            let result = find_definition_in_expr(&let_in.body, line, col, bindings);
            bindings.truncate(saved_len);
            result
        }
        Application(func, args) => {
            if let Some(s) = find_definition_in_expr(func, line, col, bindings) {
                return Some(s);
            }
            for arg in args {
                if let Some(s) = find_definition_in_expr(arg, line, col, bindings) {
                    return Some(s);
                }
            }
            None
        }
        If(if_expr) => find_definition_in_expr(&if_expr.condition, line, col, bindings)
            .or_else(|| find_definition_in_expr(&if_expr.then_branch, line, col, bindings))
            .or_else(|| find_definition_in_expr(&if_expr.else_branch, line, col, bindings)),
        BinaryOp(_, lhs, rhs) => find_definition_in_expr(lhs, line, col, bindings)
            .or_else(|| find_definition_in_expr(rhs, line, col, bindings)),
        UnaryOp(_, operand) => find_definition_in_expr(operand, line, col, bindings),
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                if let Some(s) = find_definition_in_expr(elem, line, col, bindings) {
                    return Some(s);
                }
            }
            None
        }
        ArrayIndex(arr, idx) => find_definition_in_expr(arr, line, col, bindings)
            .or_else(|| find_definition_in_expr(idx, line, col, bindings)),
        ArrayWith { array, index, value } => find_definition_in_expr(array, line, col, bindings)
            .or_else(|| find_definition_in_expr(index, line, col, bindings))
            .or_else(|| find_definition_in_expr(value, line, col, bindings)),
        FieldAccess(base, _) => find_definition_in_expr(base, line, col, bindings),
        Loop(loop_expr) => {
            let saved_len = bindings.len();
            for name in loop_expr.pattern.collect_names() {
                bindings.push((name, loop_expr.pattern.h.span));
            }
            if let Some(init) = &loop_expr.init {
                if let Some(s) = find_definition_in_expr(init, line, col, bindings) {
                    bindings.truncate(saved_len);
                    return Some(s);
                }
            }
            let result = find_definition_in_expr(&loop_expr.body, line, col, bindings);
            bindings.truncate(saved_len);
            result
        }
        RecordLiteral(fields) => {
            for (_, value) in fields {
                if let Some(s) = find_definition_in_expr(value, line, col, bindings) {
                    return Some(s);
                }
            }
            None
        }
        Match(match_expr) => {
            if let Some(s) = find_definition_in_expr(&match_expr.scrutinee, line, col, bindings) {
                return Some(s);
            }
            for case in &match_expr.cases {
                let saved_len = bindings.len();
                for name in case.pattern.collect_names() {
                    bindings.push((name, case.pattern.h.span));
                }
                if let Some(s) = find_definition_in_expr(&case.body, line, col, bindings) {
                    bindings.truncate(saved_len);
                    return Some(s);
                }
                bindings.truncate(saved_len);
            }
            None
        }
        TypeCoercion(inner, _) | TypeAscription(inner, _) => {
            find_definition_in_expr(inner, line, col, bindings)
        }
        Range(range_expr) => find_definition_in_expr(&range_expr.start, line, col, bindings)
            .or_else(|| {
                range_expr.step.as_ref().and_then(|s| find_definition_in_expr(s, line, col, bindings))
            })
            .or_else(|| find_definition_in_expr(&range_expr.end, line, col, bindings)),
        Slice(slice_expr) => find_definition_in_expr(&slice_expr.array, line, col, bindings)
            .or_else(|| {
                slice_expr.start.as_ref().and_then(|s| find_definition_in_expr(s, line, col, bindings))
            })
            .or_else(|| {
                slice_expr.end.as_ref().and_then(|e| find_definition_in_expr(e, line, col, bindings))
            }),
        _ => None,
    }
}

/// Convert an AST declaration to a DocumentSymbol
#[allow(deprecated)]
fn declaration_to_symbol(decl: &ast::Declaration) -> Option<DocumentSymbol> {
    fn span_to_range(span: Span) -> Range {
        Range {
            start: Position {
                line: span.start_line.saturating_sub(1) as u32,
                character: span.start_col.saturating_sub(1) as u32,
            },
            end: Position {
                line: span.end_line.saturating_sub(1) as u32,
                character: span.end_col.saturating_sub(1) as u32,
            },
        }
    }

    match decl {
        ast::Declaration::Decl(def) => {
            let span = def.body.h.span;
            let range = span_to_range(span);
            Some(DocumentSymbol {
                name: def.name.clone(),
                detail: Some(if def.keyword == "def" { "function" } else { "value" }.to_string()),
                kind: if def.params.is_empty() { SymbolKind::VARIABLE } else { SymbolKind::FUNCTION },
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        }
        ast::Declaration::Entry(entry) => {
            let span = entry.body.h.span;
            let range = span_to_range(span);
            let kind_str = match &entry.entry_type {
                ast::Attribute::Vertex => "vertex",
                ast::Attribute::Fragment => "fragment",
                ast::Attribute::Compute(_) => "compute",
                _ => "entry",
            };
            Some(DocumentSymbol {
                name: entry.name.clone(),
                detail: Some(format!("{} shader", kind_str)),
                kind: SymbolKind::FUNCTION,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        }
        ast::Declaration::Uniform(uniform) => {
            let range = Range::default();
            Some(DocumentSymbol {
                name: uniform.name.clone(),
                detail: Some(format!(
                    "uniform (set={}, binding={})",
                    uniform.set, uniform.binding
                )),
                kind: SymbolKind::VARIABLE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        }
        ast::Declaration::Storage(storage) => {
            let range = Range::default();
            Some(DocumentSymbol {
                name: storage.name.clone(),
                detail: Some(format!(
                    "storage (set={}, binding={})",
                    storage.set, storage.binding
                )),
                kind: SymbolKind::VARIABLE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        }
        ast::Declaration::Sig(sig) => {
            let range = Range::default();
            Some(DocumentSymbol {
                name: sig.name.clone(),
                detail: Some("signature".to_string()),
                kind: SymbolKind::INTERFACE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        }
        ast::Declaration::TypeBind(tb) => {
            let range = Range::default();
            Some(DocumentSymbol {
                name: tb.name.clone(),
                detail: Some("type".to_string()),
                kind: SymbolKind::TYPE_PARAMETER,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        }
        ast::Declaration::Module(module) => {
            let range = Range::default();
            let (name, detail) = match module {
                ast::ModuleDecl::Module { name, .. } => (name.clone(), "module"),
                ast::ModuleDecl::Functor { name, .. } => (name.clone(), "functor"),
            };
            Some(DocumentSymbol {
                name,
                detail: Some(detail.to_string()),
                kind: SymbolKind::MODULE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        }
        _ => None,
    }
}

#[tokio::main]
async fn main() {
    // Pre-initialize the prelude cache before starting the server
    // so any errors are caught early
    let _ = get_prelude();

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
