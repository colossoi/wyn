; highlights.scm - Syntax highlighting for Wyn (Helix editor compatible)

; ============================================
; Keywords
; ============================================

[
  "def"
  "entry"
  "sig"
  "let"
  "in"
  "if"
  "then"
  "else"
  "loop"
  "for"
  "while"
  "do"
  "match"
  "case"
  "type"
  "module"
  "functor"
  "open"
  "import"
] @keyword

; ============================================
; Operators
; ============================================

[
  "+"
  "-"
  "*"
  "/"
  "%"
  "**"
  "//"
  "%%"
  "=="
  "!="
  "<"
  "<="
  ">"
  ">="
  "&&"
  "||"
  "!"
  "&"
  "|"
  "^"
  "<<"
  ">>"
  ">>>"
  "|>"
  ".."
  "..<"
  "..>"
  "..."
  "->"
  ":>"
  "="
] @operator

; ============================================
; Punctuation
; ============================================

[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
  "@["
  "#["
] @punctuation.bracket

[
  ","
  "."
  ":"
] @punctuation.delimiter

; ============================================
; Literals
; ============================================

(integer_literal) @constant.numeric.integer
(float_literal) @constant.numeric.float
(char_literal) @constant.character
(string_literal) @string

(boolean_literal) @constant.builtin.boolean

; ============================================
; Types
; ============================================

(primitive_type) @type.builtin

(vec_type) @type.builtin

(mat_type) @type.builtin

(type_variable) @type

(type_declaration
  name: (identifier) @type.definition)

(array_type) @type

(record_field_type
  name: (identifier) @variable.other.member)

; ============================================
; Functions
; ============================================

(def_declaration
  name: (identifier) @function)

(entry_declaration
  name: (identifier) @function)

(sig_declaration
  name: (identifier) @function)

(sig_declaration
  name: (operator_name) @function.operator)

(call_expression
  function: (identifier) @function.call)

(call_expression
  function: (qualified_name) @function.call)

; ============================================
; Variables and Parameters
; ============================================

(param
  name: (identifier) @variable.parameter)

(let_expression
  pattern: (identifier) @variable)

(for_loop
  var: (identifier) @variable)

(for_in_loop
  pattern: (identifier) @variable)

; ============================================
; Identifiers
; ============================================

(identifier) @variable

(qualified_name) @variable

(wildcard) @variable.builtin

; ============================================
; Constructors
; ============================================

(constructor) @constructor

; ============================================
; Attributes
; ============================================

(attribute) @attribute

(attribute_item) @attribute

; ============================================
; Comments
; ============================================

(comment) @comment.line

; ============================================
; Modules
; ============================================

(module_declaration
  name: (identifier) @module)

(functor_declaration
  name: (identifier) @module)

(open_declaration
  module: (qualified_name) @module)

; ============================================
; Special
; ============================================

(type_hole) @constant.builtin

"$" @operator

; ============================================
; Record fields
; ============================================

(record_field
  name: (identifier) @variable.other.member)

(record_field_pattern
  name: (identifier) @variable.other.member)

(field_expression
  field: (identifier) @variable.other.member)
