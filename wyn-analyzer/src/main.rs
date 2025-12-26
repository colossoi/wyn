use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use wyn_core::FrontEnd;
use wyn_core::TypeTable;
use wyn_core::ast::{self, NodeCounter, NodeId, Span};
use wyn_core::module_manager::{ModuleManager, PreElaboratedPrelude};
use wyn_core::types::format_scheme;

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
    FrontEnd::new_from_prelude(prelude, node_counter)
}

/// Cached document state after successful type checking
struct DocumentState {
    ast: ast::Program,
    type_table: TypeTable,
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
            // Find the node at the cursor position
            if let Some((node_id, span)) = find_node_at_position(&doc.ast, line, col) {
                // Look up the type
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

    async fn completion(&self, _params: CompletionParams) -> Result<Option<CompletionResponse>> {
        // TODO: Implement completion
        Ok(None)
    }

    async fn goto_definition(
        &self,
        _params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        // TODO: Implement go-to-definition
        Ok(None)
    }
}

impl Backend {
    async fn on_change(&self, doc: TextDocumentItem) {
        let (diagnostics, state) = self.check_document(&doc.text);

        // Cache the document state if type checking succeeded
        if let Some(state) = state {
            if let Ok(mut docs) = self.documents.write() {
                docs.insert(doc.uri.clone(), state);
            }
        }

        self.client.publish_diagnostics(doc.uri, diagnostics, Some(doc.version)).await;
    }

    fn check_document(&self, text: &str) -> (Vec<Diagnostic>, Option<DocumentState>) {
        let mut diagnostics = Vec::new();

        // Try to parse and type-check the document
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
                // Success - return the cached state
                let state = DocumentState {
                    ast: type_checked.ast,
                    type_table: type_checked.type_table,
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
/// Returns the NodeId and Span of the innermost expression at that position
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

    // Check if this expression contains the position
    if !span.contains(line, col) {
        return;
    }

    // This expression contains the position - check if it's better than current best
    let dominated = best.as_ref().is_none_or(|(_, best_span)| span.size() < best_span.size());
    if dominated {
        *best = Some((expr.h.id, span));
    }

    // Recurse into children
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
