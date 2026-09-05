/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

/**
 * Tree-sitter grammar for the Wyn shader language.
 *
 * Tracks the lexer in `wyn-core/src/lexer/mod.rs` and the parser in
 * `wyn-core/src/parser.rs` (+ `wyn-core/src/parser/module.rs` and
 * `wyn-core/src/parser/pattern.rs`). Regenerate `src/parser.c` /
 * `src/grammar.json` / `src/node-types.json` via `tree-sitter
 * generate` after editing.
 *
 * This file intentionally follows the surface syntax accepted by the
 * hand-written parser.  Keep the generated files in `src/` in sync by
 * running `tree-sitter generate` after editing it.
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
    [$.existential_type, $.function_type],
    [$.size_argument, $._literal],
    [$.size_argument, $._primary_expression],
    [$.binary_expression],
  ],

  rules: {
    source_file: $ => repeat($._declaration),

    // ============================================
    // Declarations
    // ============================================

    _declaration: $ => choice(
      $.let_declaration,
      $.def_declaration,
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
      repeat($.attribute),
      'def',
      field('name', choice($.identifier, $.operator_name)),
      optional($.generic_params),
      choice(
        // Function form: params followed by optional return type (no colon)
        seq($.params, optional(field('return_type', $._type))),
        // Constant form: optional colon-type annotation
        optional(seq(':', field('return_type', $._type))),
      ),
      '=',
      field('body', $._expression),
    ),

    // Top-level `let` is a monomorphic value declaration. Unlike local
    // let-expressions it does not introduce a trailing `in` body.
    let_declaration: $ => seq(
      repeat($.attribute),
      'let',
      field('name', choice($.identifier, $.operator_name)),
      optional(seq(':', field('type', $._type))),
      '=',
      field('value', $._expression),
    ),

    // `extern` FFI declarations — a linked SPIR-V function. The
    // `#[linked("symbol")]` attribute is required.
    //   #[linked("foo")] extern name<[n], A>(p: T, ...) ReturnType
    extern_declaration: $ => seq(
      repeat1($.attribute),
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

    // `<[n], [m], A, B>` generic parameters. Size parameters retain
    // brackets; ordinary type parameters begin with an uppercase letter.
    generic_params: $ => seq(
      '<',
      commaSep1(choice($.size_param, $.type_parameter)),
      '>',
    ),

    // Entry requires parentheses and explicit return type (see SPECIFICATION.md)
    entry_declaration: $ => seq(
      'entry',
      field('name', $.identifier),
      optional($.generic_params),
      $.entry_params,
      field('return_type', $._type),  // Required
      '=',
      field('body', $._expression),
    ),

    sig_declaration: $ => seq(
      repeat($.attribute),
      'sig',
      field('name', choice($.identifier, $.operator_name)),
      optional($.generic_params),
      choice(
        seq(field('params', $.signature_params), field('return_type', $._type)),
        seq(':', field('type', $._type)),
      ),
    ),

    type_declaration: $ => seq(
      field('lifting', choice('type', 'type~', 'type^')),
      field('name', $.identifier),
      optional($.generic_params),
      '=',
      field('definition', $._type),
    ),

    // `module NAME [: SIG] = BODY` — signature ascription is optional.
    module_declaration: $ => seq(
      'module',
      field('name', $.identifier),
      choice(
        seq(
          ':',
          field('signature', $._module_type_expression),
          optional(seq('=', field('body', $._module_expression))),
        ),
        seq('=', field('body', $._module_expression)),
      ),
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
      repeat1($.functor_param),
      '=',
      field('body', $._module_expression),
    ),

    functor_param: $ => seq(
      '(',
      field('name', $.identifier),
      ':',
      field('signature', $._module_type_expression),
      ')',
    ),

    module_body: $ => seq(
      '{',
      repeat($._declaration),
      '}',
    ),

    _module_expression: $ => choice(
      $.module_body,
      $.module_import,
      $.module_application,
      $.module_ascription,
      $.qualified_name,
      $.identifier,
      $.parenthesized_module_expression,
    ),

    parenthesized_module_expression: $ => seq('(', $._module_expression, ')'),

    module_import: $ => seq('import', field('path', $.string_literal)),

    module_application: $ => prec.left(2, seq(
      field('function', $._module_expression),
      field('argument', choice(
        $.module_body,
        $.qualified_name,
        $.identifier,
        $.parenthesized_module_expression,
      )),
    )),

    module_ascription: $ => prec.left(1, seq(
      field('module', $._module_expression),
      ':',
      field('signature', $._module_type_expression),
    )),

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

    // `MTE with qualname generic_params = type`
    module_type_with: $ => prec.left(seq(
      field('base', $._module_type_expression),
      'with',
      field('name', choice($.identifier, $.qualified_name)),
      optional($.generic_params),
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
      optional($.generic_params),
      choice(
        seq(field('params', $.signature_params), field('return_type', $._type)),
        seq(':', field('type', $._type)),
      ),
    ),

    // `type NAME [generic_params] [= TYPE]` — the `= TYPE` half is
    // optional (abstract type vs. concrete alias).
    spec_type: $ => seq(
      'type',
      field('name', $.identifier),
      optional($.generic_params),
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
      field('module', $._module_expression),
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

    // Ordinary function parameters are patterns; unlike entry parameters,
    // they may destructure tuples, records, vectors, and constructors.
    param: $ => $._pattern,

    entry_params: $ => seq(
      '(',
      commaSep($.entry_param),
      ')',
    ),

    entry_param: $ => seq(
      repeat($.attribute),
      field('name', $.identifier),
      ':',
      field('type', $._type),
    ),

    signature_params: $ => seq(
      '(',
      commaSep1($.extern_param),
      ')',
    ),

    size_param: $ => seq('[', $.identifier, ']'),

    type_parameter: $ => /[A-Z][a-zA-Z0-9_']*/,

    // ============================================
    // Types
    // ============================================

    _type: $ => choice(
      $.array_type,
      $.tuple_type,
      $.record_type,
      $.sum_type,
      $.function_type,
      $.type_application,
      $.unique_type,
      $.existential_type,
      $.builtin_type,
      $.identifier,
      $.qualified_name,
      $.parenthesized_type,
    ),

    parenthesized_type: $ => seq('(', $._type, ')'),

    // `*T` — uniqueness-consuming marker, Futhark-style.
    unique_type: $ => prec.right(seq('*', field('inner', $._type))),

    // `?k l m. T` — existential size quantifier, valid in return
    // position. Binds one or more bare identifiers before the dot.
    existential_type: $ => seq(
      '?',
      repeat1(field('size_var', $.identifier)),
      '.',
      field('inner', $._type),
    ),

    // First-order, fully saturated applications such as `pair<i32, bool>`,
    // `vector<[4], f32>`, and `render_target<vec4f32>`.
    type_application: $ => prec(3, seq(
      field('constructor', choice($.generic_builtin_type, $.qualified_name, $.identifier)),
      '<',
      commaSep1($.type_argument),
      '>',
    )),

    type_argument: $ => choice(
      $.size_argument,
      $._type,
    ),

    // A bracketed argument is a size argument only when it ends at the comma
    // or closing angle. `[n]A` remains an ordinary array-type argument.
    size_argument: $ => seq(
      '[',
      optional(choice($.integer_literal, $.identifier)),
      ']',
    ),

    builtin_type: $ => choice(
      $.primitive_type,
      $.vec_type,
      $.mat_type,
      $.opaque_type,
      $.generic_builtin_type,
      $.graphics_state_type,
    ),

    primitive_type: $ => choice(
      'i8', 'i16', 'i32', 'i64',
      'u8', 'u16', 'u32', 'u64',
      'f16', 'f32', 'f64',
      'bool',
    ),

    // Opaque resources and invocation values with no visible type argument.
    opaque_type: $ => choice(
      'texture2d',
      'sampler',
      'storage_image',
      'vertex_invocation',
      'draw',
    ),

    // Spellable one-argument pipeline/resource type constructors.
    generic_builtin_type: $ => choice(
      'vertex',
      'raster',
      'fragment_invocation',
      'fragment_output',
      'render_target',
    ),

    // Predeclared graphics-state aliases available in graphics mode.
    graphics_state_type: $ => choice(
      'viewport',
      'scissor',
      'raster_state',
      'depth_test',
      'blend_mode',
      'fragment_state',
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
    vec_type: $ => token(prec(2, /vec[234](i8|i16|i32|i64|u8|u16|u32|u64|f16|f32|f64|bool)/)),

    // Matrix types: mat2f32, mat3x4f32, etc.
    mat_type: $ => token(prec(2, /mat[234](x[234])?(i8|i16|i32|i64|u8|u16|u32|u64|f16|f32|f64|bool)/)),

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
      field('name', choice($.identifier, $.integer_literal)),
      ':',
      field('type', $._type),
    ),

    sum_type: $ => prec.left(seq(
      $.sum_variant,
      repeat(seq('|', $.sum_variant)),
    )),

    sum_variant: $ => prec.right(seq(
      field('constructor', $.constructor),
      optional(seq('(', commaSep($._type), ')')),
    )),

    function_type: $ => prec.right(seq(
      field('param', choice($._type, $.named_parameter_type)),
      '->',
      field('return', $._type),
    )),

    named_parameter_type: $ => seq(
      '(',
      field('name', $.identifier),
      ':',
      field('type', $._type),
      ')',
    ),

    // ============================================
    // Expressions
    // ============================================

    _expression: $ => choice(
      $.let_expression,
      $.if_expression,
      $.loop_expression,
      $.match_expression,
      $.lambda_expression,
      $.array_with,
      $.vec_with,
      $.record_with,
      $.binary_expression,
      $.unary_expression,
      $.field_expression,
      $.index_expression,
      $.call_expression,
      $.type_ascription,
      $.type_coercion,
      $._primary_expression,
    ),

    // `arr with [i] = v` — produces a copy of `arr` with element `i`
    // set to `v`. Left-associative chains: `a with [i]=x with [j]=y`
    // parses as `(a with [i]=x) with [j]=y`. Precedence sits below
    // binary operators so `a with [i] = b + c` reads as
    // `a with [i] = (b + c)`.
    array_with: $ => prec.left(PREC.ASSIGN, seq(
      field('array', $._expression),
      'with',
      '[',
      field('index', $._expression),
      ']',
      '=',
      field('value', $._expression),
    )),

    // Vector swizzle update, including the compound forms accepted by the
    // hand-written parser: `v with .xy = rhs` and `v with .xy *= rhs`.
    vec_with: $ => prec.left(PREC.ASSIGN, seq(
      field('vector', $._expression),
      'with',
      '.',
      field('swizzle', $.identifier),
      optional(field('operator', choice('*', '+', '-', '/'))),
      '=',
      field('value', $._expression),
    )),

    // Record updates omit the leading dot and may select a nested field:
    // `record with outer.inner = value`.
    record_with: $ => prec.left(PREC.ASSIGN, seq(
      field('record', $._expression),
      'with',
      field('field', $.identifier),
      repeat(seq('.', field('field', $.identifier))),
      '=',
      field('value', $._expression),
    )),

    let_expression: $ => prec.right(seq(
      'let',
      choice(
        seq(
          field('pattern', $._pattern),
          optional(seq(':', field('type', $._type))),
        ),
        // Local function sugar: `let f(x: T) = value in body`.
        seq(
          field('name', $.identifier),
          field('params', $.params),
        ),
      ),
      '=',
      field('value', $._expression),
      choice(
        seq('in', field('body', $._expression)),
        // `in` may be omitted only when the body is another let.
        field('body', $.let_expression),
      ),
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
    lambda_params: $ => seq('|', commaSep($._pattern), '|'),

    lambda_expression: $ => prec.right(seq(
      choice(
        field('params', $.lambda_params),
        '||',
      ),
      field('body', $._expression),
    )),

    binary_expression: $ => choice(
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
      // Range operators: `start .. end`, `start ..< end`, `start ..> end`,
      // `start ... end`, and the three-part `start .. step ..<end` form.
      prec.left(PREC.COMPARE, seq(
        field('start', $._expression),
        '..',
        field('step', $._expression),
        field('end_op', choice('..<', '..>', '...')),
        field('end', $._expression),
      )),
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
      commaSep(choice($._expression, $.call_placeholder)),
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
      $.constructor_expression,
      $._literal,
      $.array_literal,
      $.vec_literal,
      $.unit_expression,
      $.tuple_expression,
      $.record_expression,
      $.type_hole,
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

    unit_expression: $ => prec(1, seq('(', ')')),

    tuple_expression: $ => seq(
      '(',
      $._expression,
      ',',
      commaSep1($._expression),
      ')',
    ),

    // Prefer consuming a following parenthesized payload here instead of
    // treating a bare constructor as the callee of an ordinary call.
    constructor_expression: $ => prec.right(PREC.CALL + 1, seq(
      field('constructor', $.constructor),
      optional(seq('(', commaSep($._expression), ')')),
    )),

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

    call_placeholder: $ => '_',

    type_hole: $ => '???',

    // ============================================
    // Patterns
    // ============================================

    _pattern: $ => choice(
      $.attributed_pattern,
      $.typed_pattern,
      $._primary_pattern,
    ),

    // `#[attr] pat` — entry-point param attributes and similar.
    attributed_pattern: $ => seq(
      $.attribute,
      field('pattern', $._primary_pattern),
    ),

    // `pat : type` — type-annotated pattern.
    typed_pattern: $ => prec.left(1, seq(
      field('pattern', $._primary_pattern),
      ':',
      field('type', $._type),
    )),

    _primary_pattern: $ => choice(
      $.identifier,
      $.constructor_pattern,
      $.wildcard,
      $.negative_literal_pattern,
      $._literal,
      $.unit_pattern,
      $.tuple_pattern,
      $.vec_pattern,
      $.record_pattern,
      $.parenthesized_pattern,
    ),

    parenthesized_pattern: $ => seq('(', $._pattern, ')'),

    wildcard: $ => '_',

    // Unit pattern `()`.
    unit_pattern: $ => prec(1, seq('(', ')')),

    tuple_pattern: $ => seq(
      '(',
      $._pattern,
      ',',
      commaSep1($._pattern),
      ')',
    ),

    vec_pattern: $ => seq(
      '@[',
      commaSep($._pattern),
      ']',
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

    // Anonymous-sum constructors are `#`-prefixed and take an optional
    // parenthesized payload in expressions, patterns, and types.
    constructor_pattern: $ => prec.right(seq(
      field('constructor', $.constructor),
      optional(seq('(', commaSep($._pattern), ')')),
    )),

    negative_literal_pattern: $ => seq('-', choice($.integer_literal, $.float_literal)),

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
      $.string_literal,
    ),

    // ============================================
    // Literals
    // ============================================

    _literal: $ => choice(
      $.integer_literal,
      $.float_literal,
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

    constructor: $ => token(seq('#', /[a-zA-Z_][a-zA-Z0-9_']*/)),

    qualified_name: $ => prec.left(1, seq(
      $.identifier,
      repeat1(seq('.', $.identifier)),
    )),

    operator_name: $ => seq('(', $._operator, ')'),

    _operator: $ => token(/[+\-*\/%=!><&^|]+/),

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
