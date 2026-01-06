; locals.scm - Local scope definitions for Wyn

; ============================================
; Scopes
; ============================================

(source_file) @local.scope

(def_declaration) @local.scope

(entry_declaration) @local.scope

(let_expression) @local.scope

(lambda_expression) @local.scope

(loop_expression) @local.scope

(match_expression) @local.scope

(case_clause) @local.scope

(module_body) @local.scope

; ============================================
; Definitions
; ============================================

(def_declaration
  name: (identifier) @local.definition)

(entry_declaration
  name: (identifier) @local.definition)

(type_declaration
  name: (identifier) @local.definition)

(param
  name: (pattern) @local.definition)

(let_expression
  pattern: (identifier) @local.definition)

(for_loop
  var: (identifier) @local.definition)

(for_in_loop
  pattern: (identifier) @local.definition)

(lambda_param
  pattern: (identifier) @local.definition)

(case_clause
  pattern: (identifier) @local.definition)

; ============================================
; References
; ============================================

(identifier) @local.reference

(qualified_name) @local.reference
