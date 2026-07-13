; highlights.scm - Syntax highlighting for Wyn in Helix-compatible editors

[
  "def"
  "entry"
  "sig"
  "extern"
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
  "with"
  "include"
] @keyword

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
  "?"
  "="
] @operator

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

; Literals
(integer_literal) @constant.numeric.integer
(float_literal) @constant.numeric.float
(string_literal) @string
(boolean_literal) @constant.builtin.boolean

; General identifiers first; contextual captures below refine them.
(identifier) @variable
(qualified_name) @variable
(wildcard) @variable.builtin

; Types
[
  (primitive_type)
  (vec_type)
  (mat_type)
] @type.builtin

(type_declaration
  name: (identifier) @type.definition)

(spec_type
  name: (identifier) @type.definition)

(module_type_declaration
  name: (identifier) @type.definition)

(type_variable) @type.parameter

(size_param
  (identifier) @type.parameter)

[
  (param type: (identifier) @type)
  (param type: (qualified_name) @type)
  (extern_param type: (identifier) @type)
  (extern_param type: (qualified_name) @type)
  (def_declaration return_type: (identifier) @type)
  (def_declaration return_type: (qualified_name) @type)
  (entry_declaration return_type: (identifier) @type)
  (entry_declaration return_type: (qualified_name) @type)
  (sig_declaration type: (identifier) @type)
  (sig_declaration type: (qualified_name) @type)
  (spec_sig type: (identifier) @type)
  (spec_sig type: (qualified_name) @type)
  (type_declaration definition: (identifier) @type)
  (type_declaration definition: (qualified_name) @type)
  (spec_type definition: (identifier) @type)
  (spec_type definition: (qualified_name) @type)
  (type_ascription type: (identifier) @type)
  (type_ascription type: (qualified_name) @type)
  (type_coercion type: (identifier) @type)
  (type_coercion type: (qualified_name) @type)
  (record_field_type type: (identifier) @type)
  (record_field_type type: (qualified_name) @type)
  (array_type element: (identifier) @type)
  (array_type element: (qualified_name) @type)
  (unique_type inner: (identifier) @type)
  (unique_type inner: (qualified_name) @type)
  (function_type param: (identifier) @type)
  (function_type param: (qualified_name) @type)
  (function_type return: (identifier) @type)
  (function_type return: (qualified_name) @type)
  (typed_pattern type: (identifier) @type)
  (typed_pattern type: (qualified_name) @type)
  (module_type_with type: (identifier) @type)
  (module_type_with type: (qualified_name) @type)
  (let_expression type: (identifier) @type)
  (let_expression type: (qualified_name) @type)
  (binding_declaration type: (identifier) @type)
  (binding_declaration type: (qualified_name) @type)
]

(record_field_type
  name: (identifier) @variable.other.member)

; Functions
(def_declaration
  name: (identifier) @function)

(extern_declaration
  name: (identifier) @function)

(entry_declaration
  name: (identifier) @function)

(sig_declaration
  name: (identifier) @function)

(sig_declaration
  name: (operator_name) @function.operator)

(spec_sig
  name: (identifier) @function)

(spec_sig
  name: (operator_name) @function.operator)

(call_expression
  function: (identifier) @function.call)

(call_expression
  function: (qualified_name) @function.call)

; Variables and Parameters
(binding_declaration
  name: (identifier) @variable)

(param
  name: (identifier) @variable.parameter)

(extern_param
  name: (identifier) @variable.parameter)

(functor_param
  name: (identifier) @variable.parameter)

(let_expression
  pattern: (identifier) @variable)

(for_loop
  var: (identifier) @variable)

(for_in_loop
  pattern: (identifier) @variable)

; Constructors
(constructor_name) @constructor

; Attributes
(attribute_item
  (identifier) @attribute)

(attribute_arg
  key: (identifier) @variable.other.member)

; Comments
(comment) @comment.line

; Modules
(module_declaration
  name: (identifier) @namespace)

(functor_declaration
  name: (identifier) @namespace)

(spec_module
  name: (identifier) @namespace)

(open_declaration
  module: (identifier) @namespace)

(open_declaration
  module: (qualified_name) @namespace)

[
  (module_type_with name: (identifier) @type)
  (module_type_with name: (qualified_name) @type)
  (spec_include source: (identifier) @namespace)
  (spec_include source: (qualified_name) @namespace)
]

; Special
(type_hole) @constant.builtin
"$" @operator

; Record fields
(record_field
  name: (identifier) @variable.other.member)

(record_field_pattern
  name: (identifier) @variable.other.member)

(field_expression
  field: (identifier) @variable.other.member)
