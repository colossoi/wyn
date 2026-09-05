#![deny(clippy::let_underscore_must_use)]

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use wyn_core::ast::{self, BindingName, Span};
use wyn_core::interface;
use wyn_core::lexer;
use wyn_core::types::{format_scheme, Type, TypeName, TypeScheme};
use wyn_core::{initialize_frontend, CompilerOptions, ParsedModules};
use wyn_module_graph::{ModuleId, ModulePath, PackageIdentity, PackagePlan};
use wyn_package_manager::{find_build_input, prepare_package, prepare_standalone, BuildInput};

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Stable package-relative identity for an editor buffer without a file path.
const VIRTUAL_ROOT_MODULE: &str = "editor-buffer.wyn";

/// Verbose diagnostics are auxiliary to the language-server protocol. If
/// stderr closes, stop attempting them rather than terminating the server.
fn write_verbose(arguments: std::fmt::Arguments<'_>) {
    if !VERBOSE.load(Ordering::Relaxed) {
        return;
    }
    if writeln!(std::io::stderr().lock(), "{arguments}").is_err() {
        VERBOSE.store(false, Ordering::Relaxed);
    }
}

macro_rules! verbose {
    ($($arg:tt)*) => {
        write_verbose(format_args!($($arg)*));
    };
}

/// Cached document state after successful type checking
struct DocumentState {
    ast: wyn_core::types::run::TypeChecked,
    text: String,
}

fn load_source_graph(file_path: Option<&Path>, text: &str) -> std::result::Result<ParsedModules, String> {
    let plan = match file_path {
        Some(file_path) => {
            let input = find_build_input(file_path).map_err(|error| error.to_string())?;
            let plan = match input {
                BuildInput::Package { root, root_module } => {
                    prepare_package(root, root_module).map_err(|error| error.to_string())?
                }
                BuildInput::Standalone(source) => {
                    prepare_standalone(source).map_err(|error| error.to_string())?
                }
            };
            plan.with_root_source(text).map_err(|error| error.to_string())?
        }
        None => {
            let root = ModulePath::new(VIRTUAL_ROOT_MODULE).map_err(|error| error.to_string())?;
            let identity =
                PackageIdentity::new("analyzer/root", "v0.0.0").map_err(|error| error.to_string())?;
            PackagePlan::single_source(identity, root, text)
        }
    };
    ParsedModules::load(plan, CompilerOptions { graphics: true }).map_err(|error| error.to_string())
}

fn position_to_offset(text: &str, position: Position) -> Option<u32> {
    let mut line_start = 0usize;
    for _ in 0..position.line {
        line_start += text[line_start..].find('\n')? + 1;
    }

    let line_end = text[line_start..].find('\n').map_or(text.len(), |end| line_start + end);
    let line = &text[line_start..line_end];
    let mut utf16_column = 0u32;
    for (byte_offset, character) in line.char_indices() {
        if utf16_column == position.character {
            return u32::try_from(line_start + byte_offset).ok();
        }
        utf16_column += u32::try_from(character.len_utf16()).ok()?;
        if utf16_column > position.character {
            return None;
        }
    }

    (utf16_column == position.character).then(|| u32::try_from(line_end).ok()).flatten()
}

fn offset_to_position(text: &str, offset: u32) -> Option<Position> {
    let offset = usize::try_from(offset).ok()?;
    if offset > text.len() || !text.is_char_boundary(offset) {
        return None;
    }

    let prefix = &text[..offset];
    let line = u32::try_from(prefix.bytes().filter(|byte| *byte == b'\n').count()).ok()?;
    let current_line = prefix.rsplit_once('\n').map_or(prefix, |(_, current_line)| current_line);
    let character = u32::try_from(current_line.encode_utf16().count()).ok()?;
    Some(Position { line, character })
}

fn span_to_range(text: &str, span: Span) -> Option<Range> {
    span.module()?;
    let range = span.range();
    Some(Range {
        start: offset_to_position(text, range.start())?,
        end: offset_to_position(text, range.end())?,
    })
}

fn default_diagnostic_range() -> Range {
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
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        verbose!("[wyn-analyzer] initialize request from {:?}", params.root_uri);
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
                semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
                    SemanticTokensOptions {
                        legend: SemanticTokensLegend {
                            token_types: TOKEN_TYPES.to_vec(),
                            token_modifiers: vec![],
                        },
                        full: Some(SemanticTokensFullOptions::Bool(true)),
                        range: None,
                        ..Default::default()
                    },
                )),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
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
        verbose!("[wyn-analyzer] initialized");
        self.client.log_message(MessageType::INFO, "wyn-analyzer initialized").await;
    }

    async fn shutdown(&self) -> Result<()> {
        verbose!("[wyn-analyzer] shutdown");
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        verbose!("[wyn-analyzer] didOpen {}", params.text_document.uri);
        self.on_change(TextDocumentItem {
            uri: params.text_document.uri,
            language_id: params.text_document.language_id,
            version: params.text_document.version,
            text: params.text_document.text,
        })
        .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        verbose!("[wyn-analyzer] didChange {}", params.text_document.uri);
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
        verbose!("[wyn-analyzer] didClose {}", params.text_document.uri);
        if let Ok(mut docs) = self.documents.write() {
            docs.remove(&params.text_document.uri);
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        verbose!("[wyn-analyzer] hover {}:{}", pos.line, pos.character);

        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            let Some(offset) = position_to_offset(&doc.text, pos) else {
                return Ok(None);
            };
            // First check if cursor is on a declaration name
            if let Some((name, kind)) = find_declaration_name_at(&doc.ast, offset) {
                if let Some(scheme) = definition_scheme(&doc.ast, &name) {
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
            if let Some((scheme, span)) = find_node_at_position(&doc.ast, offset) {
                let type_str = format_scheme(scheme);
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: format!("```wyn\n{}\n```", type_str),
                    }),
                    range: span_to_range(&doc.text, span),
                }));
            }
        }

        Ok(None)
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        verbose!("[wyn-analyzer] completion {}:{}", pos.line, pos.character);

        // Check if triggered by '.'
        let is_dot_trigger = params
            .context
            .as_ref()
            .and_then(|ctx| ctx.trigger_character.as_ref())
            .map(|c| c == ".")
            .unwrap_or(false);

        if is_dot_trigger {
            let docs = self.documents.read().ok();
            let doc = docs.as_ref().and_then(|d| d.get(uri));

            if let Some(doc) = doc {
                let preceding = Position {
                    line: pos.line,
                    character: pos.character.saturating_sub(1),
                };
                let Some(offset) = position_to_offset(&doc.text, preceding) else {
                    return Ok(None);
                };
                if let Some((scheme, _span)) = find_node_at_position(&doc.ast, offset) {
                    let items = get_field_completions(scheme);
                    if !items.is_empty() {
                        return Ok(Some(CompletionResponse::Array(items)));
                    }
                }
            }
        }

        // Default: prelude function completions from the checked document.
        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|documents| documents.get(uri));
        let items = doc
            .map(|document| {
                document
                    .ast
                    .global_context
                    .support_definitions
                    .iter()
                    .filter(|definition| definition.namespace.is_none())
                    .map(|definition| CompletionItem {
                        label: definition.definition.name.clone(),
                        kind: Some(CompletionItemKind::FUNCTION),
                        detail: Some("prelude function".to_string()),
                        ..Default::default()
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        verbose!("[wyn-analyzer] gotoDefinition {}:{}", pos.line, pos.character);

        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            let Some(offset) = position_to_offset(&doc.text, pos) else {
                return Ok(None);
            };
            if let Some(def_span) = find_definition(&doc.ast, offset) {
                let Some(range) = span_to_range(&doc.text, def_span) else {
                    return Ok(None);
                };
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range,
                })));
            }
        }

        Ok(None)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        verbose!("[wyn-analyzer] references {}:{}", pos.line, pos.character);

        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            let Some(offset) = position_to_offset(&doc.text, pos) else {
                return Ok(None);
            };
            // Find the name at cursor position
            if let Some(name) = find_name_at_position(&doc.ast, offset) {
                let include_declaration = params.context.include_declaration;
                let refs = find_all_references(&doc.ast, &name, include_declaration);
                let locations: Vec<Location> = refs
                    .into_iter()
                    .filter_map(|span| {
                        Some(Location {
                            uri: uri.clone(),
                            range: span_to_range(&doc.text, span)?,
                        })
                    })
                    .collect();
                if !locations.is_empty() {
                    return Ok(Some(locations));
                }
            }
        }

        Ok(None)
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        verbose!("[wyn-analyzer] documentSymbol {}", uri);
        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            let symbols: Vec<DocumentSymbol> = doc
                .ast
                .declarations
                .iter()
                .filter_map(|declaration| declaration_to_symbol(&doc.text, declaration))
                .collect();
            return Ok(Some(DocumentSymbolResponse::Nested(symbols)));
        }

        Ok(None)
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        verbose!("[wyn-analyzer] signatureHelp {}:{}", pos.line, pos.character);

        let docs = self.documents.read().ok();
        let doc = docs.as_ref().and_then(|d| d.get(uri));

        if let Some(doc) = doc {
            let Some(offset) = position_to_offset(&doc.text, pos) else {
                return Ok(None);
            };
            if let Some((func_name, arg_index)) = find_application_context(&doc.ast, offset) {
                if let Some(scheme) = definition_scheme(&doc.ast, &func_name) {
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
            }
        }

        Ok(None)
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;
        verbose!("[wyn-analyzer] semanticTokens/full {}", uri);

        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };

        let tokens = compute_semantic_tokens(&text);
        verbose!(
            "[wyn-analyzer] semanticTokens/full {} -> {} tokens",
            uri,
            tokens.len()
        );

        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: tokens,
        })))
    }
}

impl Backend {
    async fn on_change(&self, doc: TextDocumentItem) {
        let (diagnostics, state) = self.check_document(&doc.uri, &doc.text);
        verbose!("{}", format_check_result(&doc.uri, &diagnostics, state.is_some()));

        if let Some(state) = state {
            if let Ok(mut docs) = self.documents.write() {
                docs.insert(doc.uri.clone(), state);
            }
        }

        self.client.publish_diagnostics(doc.uri, diagnostics, Some(doc.version)).await;
    }

    fn check_document(&self, uri: &Url, text: &str) -> (Vec<Diagnostic>, Option<DocumentState>) {
        let mut diagnostics = Vec::new();
        let file_path = uri.to_file_path().ok();

        let modules = match load_source_graph(file_path.as_deref(), text) {
            Ok(modules) => modules,
            Err(message) => {
                diagnostics.push(Diagnostic {
                    range: default_diagnostic_range(),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("wyn-analyzer".to_string()),
                    message,
                    related_information: None,
                    tags: None,
                    data: None,
                });
                return (diagnostics, None);
            }
        };
        let result = modules.type_check();

        match result {
            Ok(type_checked) => {
                let state = DocumentState {
                    ast: type_checked,
                    text: text.to_owned(),
                };
                (diagnostics, Some(state))
            }
            Err(failure) => {
                let range = failure
                    .error()
                    .span()
                    .filter(|span| span.module() == Some(failure.source_graph().root()))
                    .and_then(|span| span_to_range(text, span))
                    .unwrap_or_else(default_diagnostic_range);

                diagnostics.push(Diagnostic {
                    range,
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("wyn-analyzer".to_string()),
                    message: failure.to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                });

                (diagnostics, None)
            }
        }
    }
}

fn definition_scheme<'a>(
    program: &'a wyn_core::types::run::TypeChecked,
    name: &str,
) -> Option<&'a TypeScheme> {
    program
        .declarations
        .iter()
        .find_map(|declaration| match declaration {
            ast::Declaration::Decl(definition) if definition.name == name => Some(&definition.data.scheme),
            ast::Declaration::Entry(entry) if entry.name == name => Some(&entry.data.scheme),
            ast::Declaration::Extern(external) if external.name == name => Some(&external.data.scheme),
            ast::Declaration::Decl(_) | ast::Declaration::Entry(_) | ast::Declaration::Extern(_) => None,
            ast::Declaration::Frontend(never) => match *never {},
        })
        .or_else(|| {
            program.global_context.support_definitions.iter().find_map(|definition| {
                (definition.namespace.is_none() && definition.definition.name == name)
                    .then_some(&definition.definition.data.scheme)
            })
        })
}

/// Find the smallest AST node containing the given position
fn find_node_at_position(
    ast: &wyn_core::types::run::TypeChecked,
    offset: u32,
) -> Option<(&TypeScheme, Span)> {
    let mut best: Option<(&TypeScheme, Span)> = None;

    for decl in &ast.declarations {
        find_in_declaration(decl, offset, &mut best);
    }

    best
}

fn find_in_declaration<'a>(
    decl: &'a ast::Declaration<wyn_core::types::run::TypeCheckedFamily>,
    offset: u32,
    best: &mut Option<(&'a TypeScheme, Span)>,
) {
    match decl {
        ast::Declaration::Decl(def) => {
            find_in_expr(&def.body, offset, best);
        }
        ast::Declaration::Entry(entry) => {
            find_in_expr(&entry.body, offset, best);
        }
        _ => {}
    }
}

fn find_in_expr<'a>(
    expr: &'a ast::Expression<ast::TypedTree>,
    offset: u32,
    best: &mut Option<(&'a TypeScheme, Span)>,
) {
    let span = expr.h.span;

    if !span.contains(offset) {
        return;
    }

    let dominated = best.as_ref().is_none_or(|(_, best_span)| span.size() < best_span.size());
    if dominated {
        *best = Some((&expr.h.ty, span));
    }

    use ast::ExprKind::*;
    match &expr.kind {
        IntLiteral(_) | FloatLiteral(_) | BoolLiteral(_) | Unit => {}
        Identifier(_) | TypeHole(_) => {}
        Application(func, args) => {
            find_in_expr(func, offset, best);
            for arg in args {
                find_in_expr(arg, offset, best);
            }
        }
        Lambda(lambda) => {
            find_in_expr(&lambda.body, offset, best);
        }
        LetIn(let_in) => {
            find_in_expr(&let_in.value, offset, best);
            find_in_expr(&let_in.body, offset, best);
        }
        If(if_expr) => {
            find_in_expr(&if_expr.condition, offset, best);
            find_in_expr(&if_expr.then_branch, offset, best);
            find_in_expr(&if_expr.else_branch, offset, best);
        }
        BinaryOp(_, lhs, rhs) => {
            find_in_expr(lhs, offset, best);
            find_in_expr(rhs, offset, best);
        }
        UnaryOp(_, operand) => {
            find_in_expr(operand, offset, best);
        }
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                find_in_expr(elem, offset, best);
            }
        }
        Constructor(_, args) => {
            for arg in args {
                find_in_expr(arg, offset, best);
            }
        }
        ArrayIndex(arr, idx) => {
            find_in_expr(arr, offset, best);
            find_in_expr(idx, offset, best);
        }
        ArrayWith {
            array, index, value, ..
        } => {
            find_in_expr(array, offset, best);
            find_in_expr(index, offset, best);
            find_in_expr(value, offset, best);
        }
        VecWith { target, value, .. } => {
            find_in_expr(target, offset, best);
            find_in_expr(value, offset, best);
        }
        RecordWith { record, value, .. } => {
            find_in_expr(record, offset, best);
            find_in_expr(value, offset, best);
        }
        FieldAccess(base, _) => {
            find_in_expr(base, offset, best);
        }
        Loop(loop_expr) => {
            if let Some(init) = &loop_expr.init {
                find_in_expr(init, offset, best);
            }
            find_in_expr(&loop_expr.body, offset, best);
        }
        RecordLiteral(fields) => {
            for (_, value) in fields {
                find_in_expr(value, offset, best);
            }
        }
        Match(match_expr) => {
            find_in_expr(&match_expr.scrutinee, offset, best);
            for case in &match_expr.cases {
                find_in_expr(&case.body, offset, best);
            }
        }
        TypeCoercion(inner, _) | TypeAscription(inner, _) => {
            find_in_expr(inner, offset, best);
        }
        Range(range_expr) => {
            find_in_expr(&range_expr.start, offset, best);
            if let Some(step) = &range_expr.step {
                find_in_expr(step, offset, best);
            }
            find_in_expr(&range_expr.end, offset, best);
        }
        Slice(slice_expr) => {
            find_in_expr(&slice_expr.array, offset, best);
            if let Some(start) = &slice_expr.start {
                find_in_expr(start, offset, best);
            }
            if let Some(end) = &slice_expr.end {
                find_in_expr(end, offset, best);
            }
        }
    }
}

/// Find the application context at cursor position
fn find_application_context(
    ast: &wyn_core::types::run::TypeChecked,
    offset: u32,
) -> Option<(String, usize)> {
    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(def) => {
                if let Some(result) = find_application_in_expr(&def.body, offset) {
                    return Some(result);
                }
            }
            ast::Declaration::Entry(entry) => {
                if let Some(result) = find_application_in_expr(&entry.body, offset) {
                    return Some(result);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_application_in_expr(
    expr: &ast::Expression<ast::TypedTree>,
    offset: u32,
) -> Option<(String, usize)> {
    let span = expr.h.span;
    if !span.contains(offset) {
        return None;
    }

    use ast::ExprKind::*;
    match &expr.kind {
        Application(func, args) => {
            for (i, arg) in args.iter().enumerate() {
                if arg.h.span.contains(offset) {
                    if let Some(result) = find_application_in_expr(arg, offset) {
                        return Some(result);
                    }
                    if let Identifier(identifier) = &func.kind {
                        return Some((identifier.source.name.clone(), i));
                    }
                }
            }
            if let Identifier(identifier) = &func.kind {
                return Some((identifier.source.name.clone(), args.len()));
            }
        }
        Lambda(lambda) => {
            return find_application_in_expr(&lambda.body, offset);
        }
        LetIn(let_in) => {
            if let Some(r) = find_application_in_expr(&let_in.value, offset) {
                return Some(r);
            }
            return find_application_in_expr(&let_in.body, offset);
        }
        If(if_expr) => {
            if let Some(r) = find_application_in_expr(&if_expr.condition, offset) {
                return Some(r);
            }
            if let Some(r) = find_application_in_expr(&if_expr.then_branch, offset) {
                return Some(r);
            }
            return find_application_in_expr(&if_expr.else_branch, offset);
        }
        BinaryOp(_, lhs, rhs) => {
            if let Some(r) = find_application_in_expr(lhs, offset) {
                return Some(r);
            }
            return find_application_in_expr(rhs, offset);
        }
        UnaryOp(_, operand) => {
            return find_application_in_expr(operand, offset);
        }
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                if let Some(r) = find_application_in_expr(elem, offset) {
                    return Some(r);
                }
            }
        }
        Constructor(_, args) => {
            for arg in args {
                if let Some(r) = find_application_in_expr(arg, offset) {
                    return Some(r);
                }
            }
        }
        ArrayIndex(arr, idx) => {
            if let Some(r) = find_application_in_expr(arr, offset) {
                return Some(r);
            }
            return find_application_in_expr(idx, offset);
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
                // Resolve the vec size if known; default to 4 so that
                // completion on a type-unresolved vec offers everything.
                let size = args
                    .get(1)
                    .and_then(|t| match t {
                        Type::Constructed(TypeName::Size(n), _) => Some(*n),
                        _ => None,
                    })
                    .unwrap_or(4)
                    .min(4);
                const XYZW: &[&str] = &["x", "y", "z", "w"];
                const RGBA: &[&str] = &["r", "g", "b", "a"];
                // Singletons (scalar component access).
                for &c in XYZW.iter().take(size) {
                    items.push(CompletionItem {
                        label: c.to_string(),
                        kind: Some(CompletionItemKind::PROPERTY),
                        detail: Some("component".to_string()),
                        ..Default::default()
                    });
                }
                for &c in RGBA.iter().take(size) {
                    items.push(CompletionItem {
                        label: c.to_string(),
                        kind: Some(CompletionItemKind::PROPERTY),
                        detail: Some("color component".to_string()),
                        ..Default::default()
                    });
                }
                // Identity multi-letter swizzles up to the vec's size —
                // `.xy`, `.xyz`, `.xyzw` plus the `rgba` equivalents.
                // Other 2-4-letter combinations are valid too (Wyn
                // supports the full WGSL swizzle surface with per-letter
                // repetition and reordering), but enumerating every
                // permutation would make the completion list unreadable;
                // the user can type the extras directly and they'll
                // type-check.
                for len in 2..=size {
                    let xyzw_swz: String = XYZW[..len].concat();
                    items.push(CompletionItem {
                        label: xyzw_swz,
                        kind: Some(CompletionItemKind::PROPERTY),
                        detail: Some(format!("vec{} swizzle", len)),
                        ..Default::default()
                    });
                    let rgba_swz: String = RGBA[..len].concat();
                    items.push(CompletionItem {
                        label: rgba_swz,
                        kind: Some(CompletionItemKind::PROPERTY),
                        detail: Some(format!("vec{} color swizzle", len)),
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
fn find_declaration_name_at(
    ast: &wyn_core::types::run::TypeChecked,
    offset: u32,
) -> Option<(String, &'static str)> {
    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(def) => {
                if def.name_span.contains(offset) {
                    return Some((def.name.clone(), def.data.source.syntax.keyword));
                }
            }
            ast::Declaration::Entry(entry) => {
                if entry.name_span.contains(offset) {
                    let kind = match entry.data.source.source.syntax.entry_kind {
                        interface::EntryKind::Vertex => "vertex",
                        interface::EntryKind::Root => "entry",
                        interface::EntryKind::Fragment => "fragment",
                        interface::EntryKind::Compute => "compute",
                    };
                    return Some((entry.name.clone(), kind));
                }
            }
            _ => {}
        }
    }
    None
}

/// Find the name at cursor position (identifier or declaration name)
fn find_name_at_position(ast: &wyn_core::types::run::TypeChecked, offset: u32) -> Option<String> {
    // Check if on a declaration name first
    if let Some((name, _)) = find_declaration_name_at(ast, offset) {
        return Some(name);
    }

    // Check if on an identifier in an expression
    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(def) => {
                // Check parameters
                for param in &def.params {
                    if let Some(name) = find_name_in_pattern(param, offset) {
                        return Some(name);
                    }
                }
                if let Some(name) = find_name_in_expr(&def.body, offset) {
                    return Some(name);
                }
            }
            ast::Declaration::Entry(entry) => {
                for param in &entry.params {
                    if let Some(name) = find_name_in_pattern(param, offset) {
                        return Some(name);
                    }
                }
                if let Some(name) = find_name_in_expr(&entry.body, offset) {
                    return Some(name);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_name_in_pattern<A>(pat: &ast::Pattern<ast::TypedTree, A>, offset: u32) -> Option<String> {
    if !pat.h.span.contains(offset) {
        return None;
    }
    match &pat.kind {
        ast::PatternKind::Name(name) => Some(name.source_name().to_owned()),
        ast::PatternKind::Tuple(pats) => {
            for p in pats {
                if let Some(name) = find_name_in_pattern(p, offset) {
                    return Some(name);
                }
            }
            None
        }
        ast::PatternKind::Constructor(_, pats) => {
            for p in pats {
                if let Some(name) = find_name_in_pattern(p, offset) {
                    return Some(name);
                }
            }
            None
        }
        _ => None,
    }
}

fn find_name_in_expr(expr: &ast::Expression<ast::TypedTree>, offset: u32) -> Option<String> {
    if !expr.h.span.contains(offset) {
        return None;
    }

    use ast::ExprKind::*;
    match &expr.kind {
        Identifier(identifier) => {
            if expr.h.span.contains(offset) {
                return Some(identifier.source.name.clone());
            }
        }
        Application(func, args) => {
            if let Some(name) = find_name_in_expr(func, offset) {
                return Some(name);
            }
            for arg in args {
                if let Some(name) = find_name_in_expr(arg, offset) {
                    return Some(name);
                }
            }
        }
        Lambda(lambda) => {
            for param in &lambda.params {
                if let Some(name) = find_name_in_pattern(param, offset) {
                    return Some(name);
                }
            }
            return find_name_in_expr(&lambda.body, offset);
        }
        LetIn(let_in) => {
            if let Some(name) = find_name_in_pattern(&let_in.pattern, offset) {
                return Some(name);
            }
            if let Some(name) = find_name_in_expr(&let_in.value, offset) {
                return Some(name);
            }
            return find_name_in_expr(&let_in.body, offset);
        }
        If(if_expr) => {
            if let Some(name) = find_name_in_expr(&if_expr.condition, offset) {
                return Some(name);
            }
            if let Some(name) = find_name_in_expr(&if_expr.then_branch, offset) {
                return Some(name);
            }
            return find_name_in_expr(&if_expr.else_branch, offset);
        }
        BinaryOp(_, lhs, rhs) => {
            if let Some(name) = find_name_in_expr(lhs, offset) {
                return Some(name);
            }
            return find_name_in_expr(rhs, offset);
        }
        UnaryOp(_, operand) => {
            return find_name_in_expr(operand, offset);
        }
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                if let Some(name) = find_name_in_expr(elem, offset) {
                    return Some(name);
                }
            }
        }
        Constructor(_, args) => {
            for arg in args {
                if let Some(name) = find_name_in_expr(arg, offset) {
                    return Some(name);
                }
            }
        }
        ArrayIndex(arr, idx) => {
            if let Some(name) = find_name_in_expr(arr, offset) {
                return Some(name);
            }
            return find_name_in_expr(idx, offset);
        }
        ArrayWith {
            array, index, value, ..
        } => {
            if let Some(name) = find_name_in_expr(array, offset) {
                return Some(name);
            }
            if let Some(name) = find_name_in_expr(index, offset) {
                return Some(name);
            }
            return find_name_in_expr(value, offset);
        }
        VecWith { target, value, .. } => {
            if let Some(name) = find_name_in_expr(target, offset) {
                return Some(name);
            }
            return find_name_in_expr(value, offset);
        }
        FieldAccess(base, _) => {
            return find_name_in_expr(base, offset);
        }
        Loop(loop_expr) => {
            if let Some(name) = find_name_in_pattern(&loop_expr.pattern, offset) {
                return Some(name);
            }
            if let Some(init) = &loop_expr.init {
                if let Some(name) = find_name_in_expr(init, offset) {
                    return Some(name);
                }
            }
            return find_name_in_expr(&loop_expr.body, offset);
        }
        Match(match_expr) => {
            if let Some(name) = find_name_in_expr(&match_expr.scrutinee, offset) {
                return Some(name);
            }
            for case in &match_expr.cases {
                if let Some(name) = find_name_in_pattern(&case.pattern, offset) {
                    return Some(name);
                }
                if let Some(name) = find_name_in_expr(&case.body, offset) {
                    return Some(name);
                }
            }
        }
        TypeCoercion(inner, _) | TypeAscription(inner, _) => {
            return find_name_in_expr(inner, offset);
        }
        _ => {}
    }
    None
}

/// Find all references to a name in the AST
fn find_all_references(
    ast: &wyn_core::types::run::TypeChecked,
    target_name: &str,
    include_declaration: bool,
) -> Vec<Span> {
    let mut refs = Vec::new();

    for decl in &ast.declarations {
        match decl {
            ast::Declaration::Decl(def) => {
                if def.name == target_name && include_declaration {
                    refs.push(def.name_span);
                }
                // Check parameters
                for param in &def.params {
                    collect_refs_in_pattern(param, target_name, &mut refs);
                }
                collect_refs_in_expr(&def.body, target_name, &mut refs);
            }
            ast::Declaration::Entry(entry) => {
                if entry.name == target_name && include_declaration {
                    refs.push(entry.name_span);
                }
                for param in &entry.params {
                    collect_refs_in_pattern(param, target_name, &mut refs);
                }
                collect_refs_in_expr(&entry.body, target_name, &mut refs);
            }
            _ => {}
        }
    }

    refs
}

fn collect_refs_in_pattern<A>(pat: &ast::Pattern<ast::TypedTree, A>, target: &str, refs: &mut Vec<Span>) {
    match &pat.kind {
        ast::PatternKind::Name(name) if name.source_name() == target => {
            refs.push(pat.h.span);
        }
        ast::PatternKind::Tuple(pats) | ast::PatternKind::Constructor(_, pats) => {
            for p in pats {
                collect_refs_in_pattern(p, target, refs);
            }
        }
        _ => {}
    }
}

fn collect_refs_in_expr(expr: &ast::Expression<ast::TypedTree>, target: &str, refs: &mut Vec<Span>) {
    use ast::ExprKind::*;
    match &expr.kind {
        Identifier(identifier) if identifier.source.name == target => {
            refs.push(expr.h.span);
        }
        Application(func, args) => {
            collect_refs_in_expr(func, target, refs);
            for arg in args {
                collect_refs_in_expr(arg, target, refs);
            }
        }
        Lambda(lambda) => {
            for param in &lambda.params {
                collect_refs_in_pattern(param, target, refs);
            }
            collect_refs_in_expr(&lambda.body, target, refs);
        }
        LetIn(let_in) => {
            collect_refs_in_pattern(&let_in.pattern, target, refs);
            collect_refs_in_expr(&let_in.value, target, refs);
            collect_refs_in_expr(&let_in.body, target, refs);
        }
        If(if_expr) => {
            collect_refs_in_expr(&if_expr.condition, target, refs);
            collect_refs_in_expr(&if_expr.then_branch, target, refs);
            collect_refs_in_expr(&if_expr.else_branch, target, refs);
        }
        BinaryOp(_, lhs, rhs) => {
            collect_refs_in_expr(lhs, target, refs);
            collect_refs_in_expr(rhs, target, refs);
        }
        UnaryOp(_, operand) => {
            collect_refs_in_expr(operand, target, refs);
        }
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                collect_refs_in_expr(elem, target, refs);
            }
        }
        Constructor(_, args) => {
            for arg in args {
                collect_refs_in_expr(arg, target, refs);
            }
        }
        ArrayIndex(arr, idx) => {
            collect_refs_in_expr(arr, target, refs);
            collect_refs_in_expr(idx, target, refs);
        }
        ArrayWith {
            array, index, value, ..
        } => {
            collect_refs_in_expr(array, target, refs);
            collect_refs_in_expr(index, target, refs);
            collect_refs_in_expr(value, target, refs);
        }
        VecWith {
            target: tgt, value, ..
        } => {
            collect_refs_in_expr(tgt, target, refs);
            collect_refs_in_expr(value, target, refs);
        }
        FieldAccess(base, _) => {
            collect_refs_in_expr(base, target, refs);
        }
        Loop(loop_expr) => {
            collect_refs_in_pattern(&loop_expr.pattern, target, refs);
            if let Some(init) = &loop_expr.init {
                collect_refs_in_expr(init, target, refs);
            }
            collect_refs_in_expr(&loop_expr.body, target, refs);
        }
        RecordLiteral(fields) => {
            for (_, value) in fields {
                collect_refs_in_expr(value, target, refs);
            }
        }
        Match(match_expr) => {
            collect_refs_in_expr(&match_expr.scrutinee, target, refs);
            for case in &match_expr.cases {
                collect_refs_in_pattern(&case.pattern, target, refs);
                collect_refs_in_expr(&case.body, target, refs);
            }
        }
        TypeCoercion(inner, _) | TypeAscription(inner, _) => {
            collect_refs_in_expr(inner, target, refs);
        }
        Range(range_expr) => {
            collect_refs_in_expr(&range_expr.start, target, refs);
            if let Some(step) = &range_expr.step {
                collect_refs_in_expr(step, target, refs);
            }
            collect_refs_in_expr(&range_expr.end, target, refs);
        }
        Slice(slice_expr) => {
            collect_refs_in_expr(&slice_expr.array, target, refs);
            if let Some(start) = &slice_expr.start {
                collect_refs_in_expr(start, target, refs);
            }
            if let Some(end) = &slice_expr.end {
                collect_refs_in_expr(end, target, refs);
            }
        }
        _ => {}
    }
}

/// Find the definition site of an identifier at the given position
fn find_definition(ast: &wyn_core::types::run::TypeChecked, offset: u32) -> Option<Span> {
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
                    offset,
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
                    offset,
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
    expr: &ast::Expression<ast::TypedTree>,
    offset: u32,
    bindings: &mut Vec<(String, Span)>,
) -> Option<Span> {
    let span = expr.h.span;
    if !span.contains(offset) {
        return None;
    }

    use ast::ExprKind::*;
    match &expr.kind {
        Identifier(identifier) => {
            if span.contains(offset) && span.size() < 100 {
                for (bound_name, bound_span) in bindings.iter().rev() {
                    if bound_name == &identifier.source.name {
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
            let result = find_definition_in_expr(&lambda.body, offset, bindings);
            bindings.truncate(saved_len);
            result
        }
        LetIn(let_in) => {
            if let Some(span) = find_definition_in_expr(&let_in.value, offset, bindings) {
                return Some(span);
            }

            let saved_len = bindings.len();
            for name in let_in.pattern.collect_names() {
                bindings.push((name, let_in.pattern.h.span));
            }
            let result = find_definition_in_expr(&let_in.body, offset, bindings);
            bindings.truncate(saved_len);
            result
        }
        Application(func, args) => {
            if let Some(s) = find_definition_in_expr(func, offset, bindings) {
                return Some(s);
            }
            for arg in args {
                if let Some(s) = find_definition_in_expr(arg, offset, bindings) {
                    return Some(s);
                }
            }
            None
        }
        If(if_expr) => find_definition_in_expr(&if_expr.condition, offset, bindings)
            .or_else(|| find_definition_in_expr(&if_expr.then_branch, offset, bindings))
            .or_else(|| find_definition_in_expr(&if_expr.else_branch, offset, bindings)),
        BinaryOp(_, lhs, rhs) => find_definition_in_expr(lhs, offset, bindings)
            .or_else(|| find_definition_in_expr(rhs, offset, bindings)),
        UnaryOp(_, operand) => find_definition_in_expr(operand, offset, bindings),
        Tuple(elems) | ArrayLiteral(elems) | VecMatLiteral(elems) => {
            for elem in elems {
                if let Some(s) = find_definition_in_expr(elem, offset, bindings) {
                    return Some(s);
                }
            }
            None
        }
        Constructor(_, args) => {
            for arg in args {
                if let Some(s) = find_definition_in_expr(arg, offset, bindings) {
                    return Some(s);
                }
            }
            None
        }
        ArrayIndex(arr, idx) => find_definition_in_expr(arr, offset, bindings)
            .or_else(|| find_definition_in_expr(idx, offset, bindings)),
        ArrayWith {
            array, index, value, ..
        } => find_definition_in_expr(array, offset, bindings)
            .or_else(|| find_definition_in_expr(index, offset, bindings))
            .or_else(|| find_definition_in_expr(value, offset, bindings)),
        VecWith { target, value, .. } => find_definition_in_expr(target, offset, bindings)
            .or_else(|| find_definition_in_expr(value, offset, bindings)),
        FieldAccess(base, _) => find_definition_in_expr(base, offset, bindings),
        Loop(loop_expr) => {
            let saved_len = bindings.len();
            for name in loop_expr.pattern.collect_names() {
                bindings.push((name, loop_expr.pattern.h.span));
            }
            if let Some(init) = &loop_expr.init {
                if let Some(s) = find_definition_in_expr(init, offset, bindings) {
                    bindings.truncate(saved_len);
                    return Some(s);
                }
            }
            let result = find_definition_in_expr(&loop_expr.body, offset, bindings);
            bindings.truncate(saved_len);
            result
        }
        RecordLiteral(fields) => {
            for (_, value) in fields {
                if let Some(s) = find_definition_in_expr(value, offset, bindings) {
                    return Some(s);
                }
            }
            None
        }
        Match(match_expr) => {
            if let Some(s) = find_definition_in_expr(&match_expr.scrutinee, offset, bindings) {
                return Some(s);
            }
            for case in &match_expr.cases {
                let saved_len = bindings.len();
                for name in case.pattern.collect_names() {
                    bindings.push((name, case.pattern.h.span));
                }
                if let Some(s) = find_definition_in_expr(&case.body, offset, bindings) {
                    bindings.truncate(saved_len);
                    return Some(s);
                }
                bindings.truncate(saved_len);
            }
            None
        }
        TypeCoercion(inner, _) | TypeAscription(inner, _) => {
            find_definition_in_expr(inner, offset, bindings)
        }
        Range(range_expr) => find_definition_in_expr(&range_expr.start, offset, bindings)
            .or_else(|| range_expr.step.as_ref().and_then(|s| find_definition_in_expr(s, offset, bindings)))
            .or_else(|| find_definition_in_expr(&range_expr.end, offset, bindings)),
        Slice(slice_expr) => find_definition_in_expr(&slice_expr.array, offset, bindings)
            .or_else(|| {
                slice_expr.start.as_ref().and_then(|s| find_definition_in_expr(s, offset, bindings))
            })
            .or_else(|| slice_expr.end.as_ref().and_then(|e| find_definition_in_expr(e, offset, bindings))),
        _ => None,
    }
}

/// Convert an AST declaration to a DocumentSymbol
#[allow(deprecated)]
fn declaration_to_symbol(
    text: &str,
    decl: &ast::Declaration<wyn_core::types::run::TypeCheckedFamily>,
) -> Option<DocumentSymbol> {
    match decl {
        ast::Declaration::Decl(def) => {
            let span = def.body.h.span;
            let range = span_to_range(text, span)?;
            let selection_range = span_to_range(text, def.name_span)?;
            Some(DocumentSymbol {
                name: def.name.clone(),
                detail: Some(
                    if def.data.source.syntax.keyword == "def" { "function" } else { "value" }.to_string(),
                ),
                kind: if def.params.is_empty() { SymbolKind::VARIABLE } else { SymbolKind::FUNCTION },
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        ast::Declaration::Entry(entry) => {
            let span = entry.body.h.span;
            let range = span_to_range(text, span)?;
            let selection_range = span_to_range(text, entry.name_span)?;
            let kind_str = match entry.data.source.source.syntax.entry_kind {
                interface::EntryKind::Vertex => "vertex",
                interface::EntryKind::Root => "pipeline",
                interface::EntryKind::Fragment => "fragment",
                interface::EntryKind::Compute => "compute",
            };
            Some(DocumentSymbol {
                name: entry.name.clone(),
                detail: Some(format!("{} entry", kind_str)),
                kind: SymbolKind::FUNCTION,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        ast::Declaration::Extern(_) => None,
        ast::Declaration::Frontend(never) => match *never {},
    }
}

fn format_check_result(uri: &Url, diagnostics: &[Diagnostic], ok: bool) -> String {
    let status = if ok { "ok" } else { "failed" };
    let count = diagnostics.len();
    match diagnostics.first() {
        Some(d) => format!(
            "[wyn-analyzer] checked {} -> {} diagnostics ({}:{}: {}), {}",
            uri,
            count,
            d.range.start.line + 1,
            d.range.start.character + 1,
            d.message,
            status,
        ),
        None => format!("[wyn-analyzer] checked {} -> 0 diagnostics, {}", uri, status),
    }
}

/// Semantic token type legend — order defines index values sent to the client.
const TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::KEYWORD,   // 0
    SemanticTokenType::NUMBER,    // 1
    SemanticTokenType::STRING,    // 2
    SemanticTokenType::COMMENT,   // 3
    SemanticTokenType::OPERATOR,  // 4
    SemanticTokenType::VARIABLE,  // 5
    SemanticTokenType::DECORATOR, // 6
];

/// Returns the index into TOKEN_TYPES for a given lexer token.
/// The LSP protocol encodes semantic token types as u32 indices into the legend
/// sent during initialization, so we return the index directly.
fn token_type_index(token: &lexer::Token) -> Option<u32> {
    use lexer::Token::*;
    match token {
        Let | Def | Entry | Sig | In | If | Then | Else | Loop | For | While | Do | Match | Case
        | Module | Functor | Open | Import | Type | Include | With | Extern | True | False => Some(0), // KEYWORD

        IntLiteral(_) | FloatLiteral(_) | SuffixedLiteral(_, _) => Some(1), // NUMBER

        StringLiteral(_) => Some(2), // STRING

        Comment(_) => Some(3), // COMMENT

        BinOp(_) | Arrow | Assign | Pipe | PipeOp | Dot | DotDot | DotDotLt | DotDotGt | Ellipsis
        | Star | Minus | Bang | TypeCoercion | Backslash => Some(4), // OPERATOR

        Identifier(_) => Some(5), // VARIABLE

        AttributeStart | Constructor(_) => Some(6), // DECORATOR

        _ => None,
    }
}

fn compute_semantic_tokens(text: &str) -> Vec<SemanticToken> {
    let tokens = match lexer::tokenize(ModuleId::from(0), text) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();
    let mut prev_line: u32 = 0;
    let mut prev_start: u32 = 0;

    for lt in &tokens {
        let Some(token_type) = token_type_index(&lt.token) else {
            continue;
        };

        let Some(range) = span_to_range(text, lt.span) else {
            continue;
        };
        if range.start.line != range.end.line {
            continue;
        }
        let line = range.start.line;
        let start = range.start.character;
        let length = range.end.character.saturating_sub(start);

        let delta_line = line - prev_line;
        let delta_start = if delta_line == 0 { start - prev_start } else { start };

        result.push(SemanticToken {
            delta_line,
            delta_start,
            length,
            token_type,
            token_modifiers_bitset: 0,
        });

        prev_line = line;
        prev_start = start;
    }

    result
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod main_tests;

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "--verbose" || a == "-v") {
        VERBOSE.store(true, Ordering::Relaxed);
        verbose!("[wyn-analyzer] verbose mode enabled");
    }

    // Pre-initialize the compiler's prelude cache before serving requests.
    if let Err(error) = initialize_frontend() {
        eprintln!("failed to initialize Wyn compiler: {error}");
        return;
    }
    verbose!("[wyn-analyzer] prelude cached");

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
