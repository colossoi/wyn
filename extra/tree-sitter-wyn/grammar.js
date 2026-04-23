/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

/**
 * Tree-sitter grammar for the Wyn shader language.
 *
 * Tracks the lexer in `wyn-core/src/lexer/mod.rs` and the parser in
 * `wyn-core/src/parser.rs` (+ `wyn-core/src/parser/module.rs` for the
 * module system). Regenerate `src/parser.c` / `src/grammar.json` /
 * `src/node-types.json` via `tree-sitter generate` after editing.
 *
 * Not yet covered (features present in the parser but not in general
 * use — add when first demanded by a real testfile):
 *   * `a with [i] = v` array-update expressions.
 *   * Range step form `start..step..end`.
 *   * Existential types `?[n]. T`.
 *   * Uniqueness marker `*T`.
 *   * Constructor patterns and unit/typed/attributed patterns.
 *   * Type params `'~a` (size-lifted) and `'^a` (lifted).
 */

const PREC = {
  ASSIGN: 1,
  TYPE_ASCRIPTION: 2,
  TYPE_COERCION: 3,
  OR: 4,
  AND: 5,
  COMPARE: 6,
  BITWISE: 7,
  SHIFT: 8,
  ADD: 9,
  MUL: 10,
  PIPE: 11,
  POWER: 12,
  UNARY: 13,
  POSTFIX: 14,
  CALL: 15,
};

module.exports = grammar({
  name: 'wyn',

  extras: $ => [
    /\s/,
    $.comment,
  ],

  word: $ => $.identifier,

  conflicts: $ => [
    [$.identifier, $.qualified_name],
    [$.call_expression, $.curry_expression],
  ],

  rules: {
    source_file: $ => repeat($._declaration),

    // ============================================
    // Declarations
    // ============================================

    _declaration: $ => choice(
      $.def_declaration,
      $.binding_declaration,
      $.extern_declaration,
      $.entry_declaration,
      $.sig_declaration,
      $.type_declaration,
      $.module_type_declaration,
      $.module_declaration,
      $.functor_declaration,
      $.open_declaration,
      $.import_declaration,
    ),

    // def can be:
    // - def name = expr                    (no type)
    // - def name: type = expr              (constant with type annotation)
    // - def name(params) = expr            (function, inferred return)
    // - def name(params) type = expr       (function with return type, no colon)
    def_declaration: $ => seq(
      optional($.attribute),
      'def',
      field('name', $.identifier),
      choice(
        // Function form: params followed by optional return type (no colon)
        seq($.params, optional(field('return_type', $._type))),
        // Constant form: optional colon-type annotation
        optional(seq(':', field('return_type', $._type))),
      ),
      '=',
      field('body', $._expression),
    ),

    // Attribute-bound `def` with no body — uniform or storage buffer
    // binding declared by attribute:
    //   #[uniform(set=0, binding=0)] def name: type
    //   #[storage(set=0, binding=0)] def name: type
    binding_declaration: $ => seq(
      $.attribute,
      'def',
      field('name', $.identifier),
      ':',
      field('type', $._type),
    ),

    // `extern` FFI declarations — a linked SPIR-V function. The
    // `#[linked("symbol")]` attribute is required.
    //   #[linked("foo")] extern name<[n], 'a>(p: T, ...) ReturnType
    extern_declaration: $ => seq(
      $.attribute,
      'extern',
      field('name', $.identifier),
      optional($.generic_params),
      field('params', $.extern_params),
      field('return_type', $._type),
    ),

    extern_params: $ => seq(
      '(',
      commaSep($.extern_param),
      ')',
    ),

    extern_param: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type),
    ),

    // `<[n], [m], 'a, 'b>` generic params for sig/extern/def forms
    // with explicit quantification.
    generic_params: $ => seq(
      '<',
      commaSep1(choice($.size_param, $.type_variable)),
      '>',
    ),

    // Entry requires parentheses and explicit return type (see SPECIFICATION.md)
    entry_declaration: $ => seq(
      optional($.attribute),
      'entry',
      field('name', $.identifier),
      $.params,
      optional($.attribute),  // Return type attribute
      field('return_type', $._type),  // Required
      '=',
      field('body', $._expression),
    ),

    sig_declaration: $ => seq(
      'sig',
      field('name', choice($.identifier, $.operator_name)),
      ':',
      field('type', $._type),
    ),

    type_declaration: $ => seq(
      'type',
      field('name', $.identifier),
      optional($.type_params),
      '=',
      field('definition', $._type),
    ),

    // `module NAME [: SIG] = BODY` — signature ascription is optional.
    module_declaration: $ => seq(
      'module',
      field('name', $.identifier),
      optional(seq(':', field('signature', $._module_type_expression))),
      '=',
      field('body', $.module_body),
    ),

    // `module type NAME = MTE` — a named module-signature binding.
    module_type_declaration: $ => seq(
      'module',
      'type',
      field('name', $.identifier),
      '=',
      field('definition', $._module_type_expression),
    ),

    // `functor NAME (params) [: SIG] = BODY` — signature ascription
    // is optional (applied to the body).
    functor_declaration: $ => seq(
      'functor',
      field('name', $.identifier),
      $.functor_params,
      optional(seq(':', field('signature', $._module_type_expression))),
      '=',
      field('body', $.module_body),
    ),

    functor_params: $ => seq(
      '(',
      commaSep1($.functor_param),
      ')',
    ),

    functor_param: $ => seq(
      field('name', $.identifier),
      ':',
      field('signature', $._module_type_expression),
    ),

    module_body: $ => seq(
      '{',
      repeat($._declaration),
      '}',
    ),

    // Module-type expressions: named signatures, inline `{ spec* }`
    // signatures, refinement via `with type t = T`, and arrow/functor
    // signature types.
    _module_type_expression: $ => choice(
      $.signature_body,
      $.module_type_with,
      $.module_type_arrow,
      $.qualified_name,
      $.identifier,
      $.parenthesized_module_type,
    ),

    parenthesized_module_type: $ => seq('(', $._module_type_expression, ')'),

    signature_body: $ => seq(
      '{',
      repeat($._spec),
      '}',
    ),

    // `MTE with qualname type_params = type`
    module_type_with: $ => prec.left(seq(
      field('base', $._module_type_expression),
      'with',
      field('name', choice($.identifier, $.qualified_name)),
      optional($.type_params),
      '=',
      field('type', $._type),
    )),

    // `(name : MTE) -> MTE`  — dependent functor arrow, or
    // `MTE -> MTE` — plain functor arrow.
    module_type_arrow: $ => prec.right(seq(
      choice(
        seq('(', field('param_name', $.identifier), ':', field('param_sig', $._module_type_expression), ')'),
        field('param_sig', $._module_type_expression),
      ),
      '->',
      field('result', $._module_type_expression),
    )),

    // Specs inside a `{ ... }` signature body.
    _spec: $ => choice(
      $.spec_sig,
      $.spec_type,
      $.spec_module,
      $.spec_include,
    ),

    spec_sig: $ => seq(
      'sig',
      field('name', choice($.identifier, $.operator_name)),
      optional($.type_params),
      ':',
      field('type', $._type),
    ),

    // `type NAME [type_params] [= TYPE]` — the `= TYPE` half is
    // optional (abstract type vs. concrete alias).
    spec_type: $ => seq(
      'type',
      field('name', $.identifier),
      optional($.type_params),
      optional(seq('=', field('definition', $._type))),
    ),

    spec_module: $ => seq(
      'module',
      field('name', $.identifier),
      ':',
      field('signature', $._module_type_expression),
    ),

    spec_include: $ => seq(
      'include',
      field('source', $._module_type_expression),
    ),

    // `open` takes a module expression, which can be a qualified name
    // or a module-expression application.
    open_declaration: $ => seq(
      'open',
      field('module', $.qualified_name),
    ),

    import_declaration: $ => seq(
      'import',
      field('path', $.string_literal),
    ),

    // ============================================
    // Parameters
    // ============================================

    params: $ => seq(
      '(',
      commaSep($.param),
      ')',
    ),

    // Entry params: identifier with required type annotation (see SPECIFICATION.md)
    // Note: def params may have optional type annotation
    param: $ => seq(
      optional($.attribute),
      field('name', $.identifier),
      optional(seq(':', field('type', $._type))),
    ),

    type_params: $ => repeat1($._type_param),

    _type_param: $ => choice(
      $.size_param,
      $.type_variable,
    ),

    size_param: $ => seq('[', $.identifier, ']'),

    // ============================================
    // Types
    // ============================================

    _type: $ => choice(
      $.primitive_type,
      $.array_type,
      $.vec_type,
      $.mat_type,
      $.tuple_type,
      $.record_type,
      $.function_type,
      $.type_variable,
      $.identifier,
      $.qualified_name,
      $.parenthesized_type,
    ),

    parenthesized_type: $ => seq('(', $._type, ')'),

    primitive_type: $ => choice(
      'i8', 'i16', 'i32', 'i64',
      'u8', 'u16', 'u32', 'u64',
      'f16', 'f32', 'f64',
      'bool',
    ),

    // Array type binds tighter than function type
    // []i32 -> i32 means ([]i32) -> i32
    array_type: $ => prec(2, seq(
      '[',
      optional(field('size', $._expression)),
      ']',
      field('element', $._type),
    )),

    // Vector types: vec2f32, vec3i32, etc.
    // Use token.immediate to ensure these win over identifier
    vec_type: $ => token(prec(2, /vec[234](i32|u32|f16|f32|f64)/)),

    // Matrix types: mat2f32, mat3x4f32, etc.
    mat_type: $ => token(prec(2, /mat[234](x[234])?(i32|u32|f16|f32|f64)/)),

    // Tuple types must have 0 (unit) or 2+ elements
    // Single element (type) is parsed as parenthesized_type
    tuple_type: $ => choice(
      seq('(', ')'),  // Unit type
      seq('(', $._type, ',', commaSep($._type), ')'),  // 2+ elements
    ),

    record_type: $ => seq(
      '{',
      commaSep($.record_field_type),
      '}',
    ),

    record_field_type: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type),
    ),

    function_type: $ => prec.right(seq(
      field('param', $._type),
      '->',
      field('return', $._type),
    )),

    type_variable: $ => seq("'", $.identifier),

    // ============================================
    // Expressions
    // ============================================

    _expression: $ => choice(
      $.let_expression,
      $.if_expression,
      $.loop_expression,
      $.match_expression,
      $.lambda_expression,
      $._binary_expression,
      $.unary_expression,
      $.field_expression,
      $.index_expression,
      $.call_expression,
      $.type_ascription,
      $.type_coercion,
      $._primary_expression,
    ),

    let_expression: $ => prec.right(seq(
      'let',
      field('pattern', $._pattern),
      optional(seq(':', field('type', $._type))),
      '=',
      field('value', $._expression),
      'in',
      field('body', $._expression),
    )),

    if_expression: $ => prec.right(seq(
      'if',
      field('condition', $._expression),
      'then',
      field('then', $._expression),
      'else',
      field('else', $._expression),
    )),

    loop_expression: $ => prec.right(seq(
      'loop',
      field('pattern', $._pattern),
      optional(seq('=', field('init', $._expression))),
      field('form', $._loop_form),
      'do',
      field('body', $._expression),
    )),

    _loop_form: $ => choice(
      $.for_loop,
      $.for_in_loop,
      $.while_loop,
    ),

    for_loop: $ => seq(
      'for',
      field('var', $.identifier),
      '<',
      field('bound', $._expression),
    ),

    for_in_loop: $ => seq(
      'for',
      field('pattern', $._pattern),
      'in',
      field('iterable', $._expression),
    ),

    while_loop: $ => seq(
      'while',
      field('condition', $._expression),
    ),

    match_expression: $ => prec.right(seq(
      'match',
      field('value', $._expression),
      repeat1($.case_clause),
    )),

    case_clause: $ => seq(
      'case',
      field('pattern', $._pattern),
      '->',
      field('body', $._expression),
    ),

    // Lambda: |params| body
    // Params are patterns (typically just identifiers)
    lambda_expression: $ => prec.right(seq(
      choice(
        seq('|', commaSep($._pattern), '|'),  // Regular: |x, y|
        '||',  // Empty params: ||
      ),
      field('body', $._expression),
    )),

    _binary_expression: $ => choice(
      // Logical OR (lowest precedence)
      prec.left(PREC.OR, seq(
        field('left', $._expression),
        field('operator', '||'),
        field('right', $._expression),
      )),
      // Logical AND
      prec.left(PREC.AND, seq(
        field('left', $._expression),
        field('operator', '&&'),
        field('right', $._expression),
      )),
      // Comparison
      prec.left(PREC.COMPARE, seq(
        field('left', $._expression),
        field('operator', choice('==', '!=', '<', '<=', '>', '>=')),
        field('right', $._expression),
      )),
      // Bitwise
      prec.left(PREC.BITWISE, seq(
        field('left', $._expression),
        field('operator', choice('&', '^')),
        field('right', $._expression),
      )),
      // Shift
      prec.left(PREC.SHIFT, seq(
        field('left', $._expression),
        field('operator', choice('<<', '>>', '>>>')),
        field('right', $._expression),
      )),
      // Additive
      prec.left(PREC.ADD, seq(
        field('left', $._expression),
        field('operator', choice('+', '-')),
        field('right', $._expression),
      )),
      // Multiplicative
      prec.left(PREC.MUL, seq(
        field('left', $._expression),
        field('operator', choice('*', '/', '%', '//', '%%')),
        field('right', $._expression),
      )),
      // Pipe
      prec.left(PREC.PIPE, seq(
        field('left', $._expression),
        field('operator', '|>'),
        field('right', $._expression),
      )),
      // Power (right associative)
      prec.right(PREC.POWER, seq(
        field('left', $._expression),
        field('operator', '**'),
        field('right', $._expression),
      )),
      // Range operators
      prec.left(PREC.COMPARE, seq(
        field('left', $._expression),
        field('operator', choice('..', '..<', '..>', '...')),
        field('right', $._expression),
      )),
    ),

    unary_expression: $ => prec(PREC.UNARY, choice(
      seq('-', field('operand', $._expression)),
      seq('!', field('operand', $._expression)),
    )),

    field_expression: $ => prec.left(PREC.POSTFIX, seq(
      field('object', $._expression),
      '.',
      field('field', choice($.identifier, $.integer_literal)),
    )),

    // Index or slice expression
    // Index: arr[i], arr[i,j]
    // Slice: arr[i..j], arr[..j], arr[i..], arr[..]
    index_expression: $ => prec.left(PREC.POSTFIX, seq(
      field('object', $._expression),
      '[',
      choice(
        // Slice syntax using .. (no conflict with type ascription which uses :)
        seq(
          optional(field('start', $._expression)),
          '..',
          optional(field('end', $._expression)),
        ),
        // Regular index
        commaSep1($._expression),
      ),
      ']',
    )),

    call_expression: $ => prec.left(PREC.CALL, seq(
      field('function', $._expression),
      '(',
      commaSep($._expression),
      ')',
    )),

    type_ascription: $ => prec.left(PREC.TYPE_ASCRIPTION, seq(
      field('expression', $._expression),
      ':',
      field('type', $._type),
    )),

    type_coercion: $ => prec.left(PREC.TYPE_COERCION, seq(
      field('expression', $._expression),
      ':>',
      field('type', $._type),
    )),

    _primary_expression: $ => choice(
      $.identifier,
      $.qualified_name,
      $._literal,
      $.array_literal,
      $.vec_literal,
      $.tuple_expression,
      $.record_expression,
      $.type_hole,
      $.curry_expression,
      $.parenthesized_expression,
    ),

    parenthesized_expression: $ => seq('(', $._expression, ')'),

    array_literal: $ => seq(
      '[',
      commaSep($._expression),
      ']',
    ),

    vec_literal: $ => seq(
      '@[',
      commaSep($._expression),
      ']',
    ),

    tuple_expression: $ => seq(
      '(',
      $._expression,
      ',',
      commaSep1($._expression),
      ')',
    ),

    record_expression: $ => seq(
      '{',
      commaSep($.record_field),
      '}',
    ),

    record_field: $ => choice(
      seq(
        field('name', $.identifier),
        '=',
        field('value', $._expression),
      ),
      // Shorthand: just the identifier
      $.identifier,
    ),

    curry_expression: $ => seq(
      '$',
      field('function', $._expression),
      '(',
      commaSep(choice($._expression, '_')),
      ')',
    ),

    type_hole: $ => '???',

    // ============================================
    // Patterns
    // ============================================

    _pattern: $ => choice(
      $.identifier,
      $.wildcard,
      $._literal,
      $.tuple_pattern,
      $.record_pattern,
      $.parenthesized_pattern,
    ),

    parenthesized_pattern: $ => seq('(', $._pattern, ')'),

    wildcard: $ => '_',

    tuple_pattern: $ => seq(
      '(',
      $._pattern,
      ',',
      commaSep1($._pattern),
      ')',
    ),

    record_pattern: $ => seq(
      '{',
      commaSep($.record_field_pattern),
      '}',
    ),

    record_field_pattern: $ => seq(
      field('name', $.identifier),
      optional(seq('=', field('pattern', $._pattern))),
    ),

    // ============================================
    // Attributes
    // ============================================

    attribute: $ => seq(
      '#[',
      commaSep1($.attribute_item),
      ']',
    ),

    attribute_item: $ => seq(
      $.identifier,
      optional(seq('(', commaSep($.attribute_arg), ')')),
    ),

    attribute_arg: $ => choice(
      // key=value: set=0, binding=1
      seq(field('key', $.identifier), '=', field('value', choice($.integer_literal, $.identifier))),
      // positional: compute(1, 1, 1), builtin(position)
      $.integer_literal,
      $.identifier,
    ),

    // ============================================
    // Literals
    // ============================================

    _literal: $ => choice(
      $.integer_literal,
      $.float_literal,
      $.string_literal,
      $.boolean_literal,
    ),

    integer_literal: $ => token(seq(
      choice(
        /[0-9][0-9_]*/,           // Decimal
        /0[xX][0-9a-fA-F_]+/,     // Hexadecimal
        /0[bB][01_]+/,            // Binary
      ),
      optional(/[iu](8|16|32|64)/), // Type suffix
    )),

    float_literal: $ => token(seq(
      choice(
        /[0-9][0-9_]*\.[0-9][0-9_]*/,                    // 3.14
        /[0-9][0-9_]*[eE][+-]?[0-9]+/,                   // 1e10
        /[0-9][0-9_]*\.[0-9][0-9_]*[eE][+-]?[0-9]+/,     // 1.5e-10
      ),
      optional(/f(16|32|64)/), // Type suffix
    )),

    string_literal: $ => /"[^"]*"/,

    boolean_literal: $ => choice('true', 'false'),

    // ============================================
    // Names
    // ============================================

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_']*/,

    qualified_name: $ => prec.left(1, seq(
      $.identifier,
      repeat1(seq('.', $.identifier)),
    )),

    operator_name: $ => seq('(', $._operator, ')'),

    _operator: $ => choice(
      '+', '-', '*', '/', '%', '**', '//', '%%',
      '==', '!=', '<', '<=', '>', '>=',
      '&&', '||', '!',
      '&', '^', '<<', '>>', '>>>',
      '|>',
    ),

    // ============================================
    // Comments
    // ============================================

    comment: $ => token(seq('--', /.*/)),
  },
});

/**
 * Creates a comma-separated list (zero or more)
 */
function commaSep(rule) {
  return optional(commaSep1(rule));
}

/**
 * Creates a comma-separated list (one or more)
 */
function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)), optional(','));
}
