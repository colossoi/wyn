use super::*;
use crate::ast::{Span, TypeName};
use crate::tlc::{self, data, VarRef};
use crate::{test_pipeline, types, SymbolTable};
use egglog_engine::EGraph;
use wyn_base::IdSource;

type Term = tlc::Term<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type TermKind = tlc::TermKind<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type ArrayExpr = tlc::ArrayExpr<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type SoacOp = tlc::SoacOp<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type SoacBody = tlc::SoacBody<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type Lambda = tlc::Lambda<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;

fn text(converted: &Converted) -> String {
    converted.program.iter().map(|command| format!("{command}\n")).collect()
}

fn run(converted: &Converted) -> EGraph {
    let mut graph = EGraph::default();
    graph
        .run_program(converted.program.clone())
        .expect("egglog must typecheck and execute the imported AST");
    // Exercise the displayed form too: downstream callers can persist the AST
    // with egglog's Display implementation without a Wyn-specific serializer.
    EGraph::default()
        .parse_and_run_program(None, &text(converted))
        .expect("displayed AST must round-trip");
    graph
}

fn source(source: &str) -> tlc::stage::InputSliceBoundsInferred {
    tlc::infer_input_slice_bounds(test_pipeline::compile_to_reachable(source))
}

fn verify_sources(program: &tlc::stage::InputSliceBoundsInferred, converted: &Converted) {
    run(converted);
    // Every imported shape must also support dependency/effect analysis and a
    // valid scoped schedule, including nested loops and all SOAC constructors.
    super::optimize::analyze(&converted.data).unwrap();
    super::snapshot::analyze(&converted.data).schedules(&converted.data).unwrap();
    assert_eq!(program.defs.len(), converted.data.definitions.len());
    assert_eq!(program.symbols.len(), converted.data.symbols.len());
    let mut expressions = crate::LookupSet::new();
    for expression in converted.data.expressions.values() {
        assert!(expressions.insert(expression.clone()), "pure values are interned");
        assert!(converted.data.types.get(expression.ty).is_some());
    }
    let mut types = crate::LookupSet::new();
    for ty in converted.data.types.values() {
        assert!(types.insert(ty.clone()), "types are interned");
    }
    for origin in converted.data.origins.values() {
        assert!(converted.data.expressions.get(origin.expression).is_some());
        assert!(converted.data.definitions.get(origin.definition).is_some());
    }
    let mut operations = crate::LookupSet::new();
    for (id, region) in &converted.data.regions {
        assert!(converted.data.definitions.get(region.definition).is_some());
        for operation in &region.members {
            assert!(
                operations.insert(*operation),
                "an execution has exactly one region"
            );
            assert_eq!(converted.data.operations[*operation].region, *id);
        }
        for param in &region.parameters {
            assert_eq!(converted.data.parameters[*param].region, *id);
        }
    }
    assert_eq!(operations.len(), converted.data.operations.len());
    let emitted = text(converted);
    for removed in [
        "TermId",
        "ExprId",
        "ParameterId",
        "TypeId",
        "Definition",
        "Origin",
        "FloatBits",
        "BinOp",
        "ApplyBody",
        "Typed",
        "NoExprs",
        "ProgramId",
        "SourcePosition",
        "RegionResult",
        "(Let ",
    ] {
        assert!(!emitted.contains(removed), "unexpected {removed}");
    }
    for (def, source) in converted.data.definitions.values().zip(&program.defs) {
        assert_eq!(converted.data.symbols[def.symbol].source, source.name);
        assert_eq!(converted.data.types[def.ty].ty, source.ty);
        assert_eq!(def.arity, source.arity);
        assert_eq!(def.package, source.package);
        assert_eq!(def.param_diets, source.param_diets);
        assert_eq!(def.return_diet, source.return_diet);
    }
}

#[test]
fn imports_real_tlc_and_preserves_roots_and_metadata() {
    for source_text in [
        "entry scalar(x: i32) i32 = if x > 0 then x + 1 else -x",
        "entry mapped(xs: []i32) []i32 = map(|x: i32| x + 1, xs)",
        "entry summed(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)",
        "entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)",
        "entry evens(xs: []i32) []i32 = filter(|x: i32| x % 2 == 0, xs)",
    ] {
        let program = source(source_text);
        let before = format!("{program:?}");
        let converted = convert_program(&program).expect("import TLC");
        verify_sources(&program, &converted);
        assert_eq!(before, format!("{program:?}"), "conversion must not mutate TLC");
        assert_eq!(text(&converted), text(&convert_program(&program).unwrap()));
        assert_eq!(
            format!("{:?}", converted.data),
            format!("{:?}", convert_program(&program).unwrap().data)
        );
        for (entry_id, entry) in &converted.data.entries {
            let source = program
                .defs
                .iter()
                .find(|def| {
                    def.name
                        == converted.data.symbols[converted.data.definitions[entry.definition].symbol]
                            .source
                })
                .unwrap();
            let tlc::DefMeta::EntryPoint(source_entry) = &source.meta else {
                panic!("entry source")
            };
            assert_eq!(entry.declaration, *source_entry.declaration);
            let params: Vec<_> =
                converted.data.entry_params.values().filter(|param| param.entry == *entry_id).collect();
            assert_eq!(params.len(), source_entry.data.param_bindings.len());
            for (position, param) in params.iter().enumerate() {
                assert_eq!(param.position, position);
                assert_eq!(param.binding, source_entry.data.param_bindings[position]);
            }
        }
    }
}

#[test]
fn empty_program_is_executable() {
    let program = Fixture::new().program(Vec::new());
    let converted = convert_program(&program).unwrap();
    assert!(converted.data.definitions.is_empty());
    assert!(converted.data.expressions.is_empty());
    assert_eq!(converted.data.programs.len(), 1);
    run(&converted);
    assert_eq!(text(&converted), text(&convert_program(&program).unwrap()));
}

#[test]
fn fusion_graph_does_not_grow_with_scalar_body_structure() {
    let small = convert_program(&source(
        "entry mapped(xs: [4]i32) [4]i32 = map(|x: i32| x + 1, xs)",
    ))
    .unwrap();
    let mut expression = "x".to_owned();
    for n in 1..5 {
        expression = format!("({expression} * x + {n})");
    }
    let large = convert_program(&source(&format!(
        "entry mapped(xs: [4]i32) [4]i32 = map(|x: i32| {expression}, xs)"
    )))
    .unwrap();
    assert!(large.data.expressions.len() > small.data.expressions.len());
    let small_graph = run(&small);
    let large_graph = run(&large);
    for relation in [
        "Operation",
        "Screma",
        "InputFrom",
        "Use",
        "DependsOn",
        "EffectBefore",
        "Safe",
        "Movable",
    ] {
        let (small_rows, _, _) = small_graph.function_to_dag(relation, usize::MAX, false).unwrap();
        let (large_rows, _, _) = large_graph.function_to_dag(relation, usize::MAX, false).unwrap();
        assert_eq!(small_rows.len(), large_rows.len(), "{relation}");
    }
    assert_eq!(
        small.program.len(),
        large.program.len(),
        "no scalar facts are emitted"
    );
}

#[test]
fn scalar_only_effects_stay_in_the_sidecar_without_fusion_facts() {
    let mut f = Fixture::new();
    let callee = f.term(TermKind::Extern("effect".into()));
    let call = f.term(TermKind::App {
        func: Box::new(callee),
        args: vec![],
    });
    let program = f.program(vec![call]);
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    let graph = run(&converted);
    let (rows, _, _) = graph.function_to_dag("Operation", usize::MAX, false).unwrap();
    assert!(rows.is_empty(), "no SOACs to optimize");
    assert_eq!(converted.data.operations.len(), 1);
    let schedules = super::snapshot::analyze(&converted.data).schedules(&converted.data).unwrap();
    assert_eq!(
        schedules.values().map(Vec::len).sum::<usize>(),
        1,
        "readout still retains the effect"
    );
}

#[test]
fn input_bounds_are_arena_records_with_deterministic_ids() {
    let mut program = source(
        "entry sliced(xs: []i32, ys: []i32) i32 =
          let a = xs[0..4] in
          let b = ys[0..8] in
          reduce(|x: i32, y: i32| x + y, 0, a) +
          reduce(|x: i32, y: i32| x + y, 0, b)",
    );
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    assert_eq!(converted.data.input_bounds.len(), 2);
    assert_eq!(
        converted.data.programs.values().next().unwrap().next_auto_storage_binding,
        program.global_context.auto_storage_binding_ids.peek_id(),
    );
    for bound in converted.data.input_bounds.values() {
        let entry = &converted.data.entries[bound.entry];
        let definition = &converted.data.definitions[entry.definition];
        let source_symbol = converted.data.symbols[definition.symbol].source;
        let def = program.defs.iter().find(|def| def.name == source_symbol).unwrap();
        let tlc::DefMeta::EntryPoint(source_entry) = &def.meta else {
            panic!("entry source")
        };
        assert_eq!(
            source_entry.data.by_symbol[&converted.data.symbols[bound.symbol].source],
            bound.length
        );
    }
    for def in &mut program.defs {
        if let tlc::DefMeta::EntryPoint(entry) = &mut def.meta {
            let mut bounds: Vec<_> = entry.data.by_symbol.drain().collect();
            bounds.sort_by_key(|(symbol, _)| std::cmp::Reverse(symbol.0));
            entry.data.by_symbol.extend(bounds);
        }
    }
    let reordered = convert_program(&program).unwrap();
    assert_eq!(text(&converted), text(&reordered));
    assert_eq!(format!("{:?}", converted.data), format!("{:?}", reordered.data));
}

struct Fixture {
    symbols: SymbolTable,
    ids: tlc::TermIdSource,
    def: crate::SymbolId,
    x: crate::SymbolId,
    y: crate::SymbolId,
}

impl Fixture {
    fn new() -> Self {
        let mut symbols = SymbolTable::new();
        let def = symbols.alloc("source name with \"quotes\"\n(and parentheses)".into());
        let x = symbols.alloc("shadowed".into());
        let y = symbols.alloc("shadowed".into());
        Self {
            symbols,
            ids: tlc::TermIdSource::new(),
            def,
            x,
            y,
        }
    }

    fn term(&mut self, kind: TermKind) -> Term {
        Term::fresh(&mut self.ids, i32_ty(), Span::generated(), kind)
    }

    fn int(&mut self, n: &str) -> Term {
        self.term(TermKind::IntLit(n.into()))
    }

    fn program(self, bodies: Vec<Term>) -> tlc::stage::InputSliceBoundsInferred {
        let defs = bodies
            .into_iter()
            .map(|body| tlc::Def {
                data: (),
                name: self.def,
                package: None,
                ty: body.ty.clone(),
                body,
                meta: tlc::DefMeta::Function,
                arity: 0,
                param_diets: vec![],
                return_diet: types::Diet::default(),
            })
            .collect();
        tlc::ProgramParts { defs }.with_symbols(
            self.symbols,
            self.ids,
            tlc::context::BackendGlobal {
                auto_storage_binding_ids: IdSource::new(),
            },
        )
    }

    fn lambda(&mut self) -> Lambda {
        let body = self.term(TermKind::Var(VarRef::Symbol(self.x)));
        Lambda {
            params: vec![(self.x, i32_ty())],
            body: Box::new(body),
            ret_ty: i32_ty(),
        }
    }

    fn soac_body(&mut self) -> SoacBody {
        let lam = self.lambda();
        let capture = self.int("37");
        SoacBody {
            lam,
            data: data::ExplicitCaptures {
                captures: vec![(self.y, i32_ty(), capture)],
            },
        }
    }

    fn array(&self) -> ArrayExpr {
        ArrayExpr::Var(VarRef::Symbol(self.x), types::sized_array(4, i32_ty()))
    }
}

fn i32_ty() -> types::Type {
    types::Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
fn literals_and_names_stay_in_the_sidecar() {
    let mut f = Fixture::new();
    let values = vec![
        f.int("18446744073709551615"),
        f.term(TermKind::FloatLit(-0.0)),
        f.term(TermKind::FloatLit(f32::from_bits(0x7fc00042))),
        f.term(TermKind::BoolLit(true)),
        f.term(TermKind::UnitLit),
        f.term(TermKind::Extern("external\"name\n(with syntax)".into())),
        f.term(TermKind::Var(VarRef::Symbol(f.x))),
        f.term(TermKind::Var(VarRef::Symbol(f.y))),
    ];
    let tuple = f.term(TermKind::Tuple(values));
    let program = f.program(vec![tuple]);
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    let emitted = text(&converted);
    assert!(!emitted.contains("18446744073709551615"));
    for literal in [
        ExprKind::Int("18446744073709551615".into()),
        ExprKind::FloatBits(2147483648),
        ExprKind::FloatBits(2143289410),
    ] {
        assert!(converted.data.expressions.values().any(|expr| expr.kind == literal));
    }
    assert!(!emitted.contains("shadowed"));
    assert!(!emitted.contains("external"));
    assert!(!emitted.contains("Wyn"));
    assert_eq!(
        converted.data.externs.values().next().unwrap().linkage_name,
        "external\"name\n(with syntax)"
    );
    let shadowed: Vec<_> =
        converted.data.symbols.iter().filter(|(_, symbol)| symbol.name == "shadowed").collect();
    assert_ne!(shadowed[0].0, shadowed[1].0);
}

#[test]
fn repeated_effectful_calls_keep_distinct_occurrences() {
    let mut f = Fixture::new();
    let callee = f.term(TermKind::Extern("side_effect".into()));
    let call = f.term(TermKind::App {
        func: Box::new(callee),
        args: vec![],
    });
    let tuple = f.term(TermKind::Tuple(vec![call.clone(), call]));
    let program = f.program(vec![tuple]);
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    let root = converted.data.definitions.values().next().unwrap().body;
    let region = &converted.data.regions[root];
    assert_eq!(region.members.len(), 2);
    assert_ne!(*region.members.first().unwrap(), *region.members.last().unwrap());
    let ExprKind::Tuple(ids) = &converted.data.expressions[region.results[0]].kind else {
        panic!("tuple");
    };
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    for (value, operation) in ids.iter().zip(&region.members) {
        assert_eq!(
            converted.data.expressions[*value].kind,
            ExprKind::OperationResult(*operation)
        );
    }
}
#[test]
fn equal_constants_share_a_sidecar_record_without_losing_provenance() {
    let mut f = Fixture::new();
    let left = f.int("42");
    let right = f.int("42");
    let tuple = f.term(TermKind::Tuple(vec![left, right]));
    let converted = convert_program(&f.program(vec![tuple])).unwrap();
    let constants: Vec<_> = converted
        .data
        .expressions
        .iter()
        .filter(|(_, expr)| expr.kind == ExprKind::Int("42".into()))
        .collect();
    assert_eq!(constants.len(), 1);
    let (id, _) = constants[0];
    let root = converted.data.regions.values().next().unwrap().results[0];
    assert_eq!(
        converted.data.expressions[root].kind,
        ExprKind::Tuple(vec![*id, *id])
    );
    assert!(converted.data.origins.values().any(|origin| origin.expression == *id));
    run(&converted);
}
#[test]
fn imports_all_loop_array_and_closure_shapes() {
    let mut f = Fixture::new();
    let mut terms = Vec::new();
    let literal = f.int("1");
    terms.push(f.term(TermKind::ArrayExpr(ArrayExpr::Literal(vec![literal]))));
    terms.push(f.term(TermKind::ArrayExpr(ArrayExpr::Zip(vec![f.array(), f.array()]))));
    for has_step in [false, true] {
        let start = Box::new(f.int("0"));
        let len = Box::new(f.int("8"));
        let step = has_step.then(|| Box::new(f.int("2")));
        terms.push(f.term(TermKind::ArrayExpr(ArrayExpr::Range { start, len, step })));
    }
    let kinds = [
        tlc::LoopKind::For {
            var: f.x,
            var_ty: i32_ty(),
            iter: Box::new(f.int("3")),
        },
        tlc::LoopKind::ForRange {
            var: f.x,
            var_ty: i32_ty(),
            bound: Box::new(f.int("4")),
        },
        tlc::LoopKind::While {
            cond: Box::new(f.term(TermKind::BoolLit(false))),
        },
    ];
    for kind in kinds {
        let init = Box::new(f.int("0"));
        let binding = f.int("5");
        let body = Box::new(f.term(TermKind::Var(VarRef::Symbol(f.y))));
        terms.push(f.term(TermKind::Loop {
            loop_var: f.y,
            loop_var_ty: i32_ty(),
            init,
            init_bindings: vec![(f.x, i32_ty(), binding)],
            kind,
            body,
        }));
    }
    let capture = f.int("6");
    terms.push(f.term(TermKind::Closure(data::ExplicitClosure {
        code: f.def,
        captures: vec![capture],
        param_count: 1,
    })));
    let lambda = f.lambda();
    terms.push(f.term(TermKind::Lambda(lambda)));
    let vector = f.term(TermKind::VecLit(vec![]));
    terms.push(f.term(TermKind::TupleProj {
        tuple: Box::new(vector),
        idx: 0,
    }));
    let inner = f.int("7");
    terms.push(f.term(TermKind::Coerce {
        inner: Box::new(inner),
        target_ty: types::Type::Constructed(TypeName::Float(32), vec![]),
    }));
    let array = f.term(TermKind::ArrayExpr(f.array()));
    let index = f.int("0");
    terms.push(f.term(TermKind::Index {
        array: Box::new(array),
        index: Box::new(index),
    }));
    let tuple = f.term(TermKind::Tuple(terms));
    let program = f.program(vec![tuple]);
    verify_sources(&program, &convert_program(&program).unwrap());
}

#[test]
fn imports_every_soac_and_its_capture_and_destination_data() {
    let mut f = Fixture::new();
    let dest = tlc::Place {
        id: f.y,
        elem_ty: i32_ty(),
    };
    let ops = vec![
        SoacOp::Map {
            lam: f.soac_body(),
            inputs: vec![f.array()],
            destination: types::SoacOwnership::Fresh,
        },
        SoacOp::Reduce {
            op: f.soac_body(),
            ne: Box::new(f.int("0")),
            input: f.array(),
        },
        SoacOp::Scan {
            op: f.soac_body(),
            ne: Box::new(f.int("0")),
            input: f.array(),
            destination: types::SoacOwnership::UniqueInput,
        },
        SoacOp::Filter {
            pred: f.soac_body(),
            input: f.array(),
            destination: types::SoacOwnership::Fresh,
        },
        SoacOp::Scatter {
            dest: dest.clone(),
            lam: f.soac_body(),
            inputs: vec![f.array()],
        },
        SoacOp::BucketScatter {
            dest: dest.clone(),
            lam: f.soac_body(),
            inputs: vec![f.array(), f.array()],
            input_dimensions: vec![vec![0], vec![1, 0]],
            domain_rank: 2,
        },
        SoacOp::ReduceByIndex {
            dest,
            op: f.soac_body(),
            ne: Box::new(f.int("0")),
            indices: f.array(),
            values: f.array(),
        },
    ];
    let terms: Vec<_> = ops.into_iter().map(|op| f.term(TermKind::Soac(op))).collect();
    let tuple = f.term(TermKind::Tuple(terms));
    let program = f.program(vec![tuple]);
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    assert_eq!(converted.data.bucket_shapes.len(), 1);
    let shape = converted.data.bucket_shapes.values().next().unwrap();
    assert_eq!(shape.input_dimensions, vec![vec![0], vec![1, 0]]);
    assert_eq!(shape.domain_rank, 2);
    assert_eq!(
        converted
            .data
            .operations
            .values()
            .filter(|op| matches!(op.kind, OperationKind::Screma { .. }))
            .count(),
        3
    );
    let mut kinds = [false; 4];
    for op in converted.data.operations.values() {
        match op.kind {
            OperationKind::Filter { .. } => kinds[0] = true,
            OperationKind::Scatter { .. } => kinds[1] = true,
            OperationKind::BucketScatter { .. } => kinds[2] = true,
            OperationKind::ReduceByIndex { .. } => kinds[3] = true,
            _ => {}
        }
    }
    assert!(kinds.into_iter().all(|present| present));
}
#[test]
fn missing_symbols_and_duplicate_definitions_return_errors() {
    let mut f = Fixture::new();
    let missing = crate::SymbolId(u32::MAX);
    let term = f.term(TermKind::Var(VarRef::Symbol(missing)));
    assert!(
        matches!(convert_program(&f.program(vec![term])), Err(ConvertError::MissingSymbol(id)) if id == missing)
    );
    let mut f = Fixture::new();
    let term = f.int("0");
    assert!(matches!(
        convert_program(&f.program(vec![term.clone(), term])),
        Err(ConvertError::DuplicateDefinition(_))
    ));
}

#[test]
fn lets_and_repeated_arithmetic_resolve_to_shared_values() {
    let program = source(
        "entry shared(x: i32) (i32, i32, i32) =
      let a = x + 17 in let b = x + 17 in (a, b, x + 17)",
    );
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    let entry = converted.data.entries.values().next().unwrap();
    let region = &converted.data.regions[converted.data.definitions[entry.definition].body];
    assert!(region.members.is_empty());
    let ExprKind::Tuple(values) = &converted.data.expressions[region.results[0]].kind else {
        panic!("tuple");
    };
    assert_eq!(values.len(), 3);
    assert!(values.iter().all(|id| *id == values[0]));
    assert!(matches!(
        converted.data.expressions[values[0]].kind,
        ExprKind::PureApp { .. }
    ));
    let origins: Vec<_> =
        converted.data.origins.values().filter(|origin| origin.expression == values[0]).collect();
    assert!(
        origins.len() > 1,
        "distinct source spans remain associated with one expression"
    );
}

#[test]
fn type_and_exact_float_bits_participate_in_interning() {
    let mut f = Fixture::new();
    let signed = f.int("42");
    let mut unsigned = f.int("42");
    unsigned.ty = types::Type::Constructed(TypeName::UInt(32), vec![]);
    let plus = f.term(TermKind::FloatLit(0.0));
    let minus = f.term(TermKind::FloatLit(-0.0));
    let nan = f.term(TermKind::FloatLit(f32::from_bits(0x7fc00042)));
    let tuple = f.term(TermKind::Tuple(vec![
        signed.clone(),
        signed,
        unsigned,
        plus,
        minus,
        nan.clone(),
        nan,
    ]));
    let converted = convert_program(&f.program(vec![tuple])).unwrap();
    let root = converted.data.regions.values().next().unwrap().results[0];
    let ExprKind::Tuple(values) = &converted.data.expressions[root].kind else {
        panic!("tuple");
    };
    assert_eq!(values[0], values[1]);
    assert_ne!(values[0], values[2]);
    assert_ne!(values[3], values[4]);
    assert_eq!(values[5], values[6]);
    run(&converted);
}

#[test]
fn parameters_are_distinct_across_regions_even_with_the_same_source_symbol() {
    let mut f = Fixture::new();
    let other_name = f.y;
    let lambda = f.lambda();
    let body = f.term(TermKind::Lambda(lambda));
    let mut program = f.program(vec![body.clone(), body]);
    program.defs[1].name = other_name;
    let converted = convert_program(&program).unwrap();
    let params: Vec<_> = converted.data.parameters.iter().collect();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].1.symbol, params[1].1.symbol);
    assert_ne!(params[0].0, params[1].0);
    let results: Vec<_> = converted
        .data
        .definitions
        .values()
        .map(|def| converted.data.regions[def.body].results[0])
        .collect();
    assert_ne!(results[0], results[1]);
    run(&converted);
}

#[test]
fn unused_call_results_remain_ordered_and_branches_keep_their_own_effects() {
    let mut f = Fixture::new();
    let function = f.term(TermKind::Extern("effect".into()));
    let call = f.term(TermKind::App {
        func: Box::new(function),
        args: vec![],
    });
    let cond = f.term(TermKind::BoolLit(true));
    let branch = f.term(TermKind::If {
        cond: Box::new(cond),
        then_branch: Box::new(call.clone()),
        else_branch: Box::new(call.clone()),
    });
    let result = f.int("0");
    let tail = f.term(TermKind::Let {
        name: f.y,
        name_ty: i32_ty(),
        rhs: Box::new(call),
        body: Box::new(result),
    });
    let body = f.term(TermKind::Let {
        name: f.x,
        name_ty: i32_ty(),
        rhs: Box::new(branch),
        body: Box::new(tail),
    });
    let program = f.program(vec![body]);
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    let root = converted.data.definitions.values().next().unwrap().body;
    let region = &converted.data.regions[root];
    assert_eq!(region.members.len(), 2);
    let OperationKind::If {
        then_region,
        else_region,
        ..
    } = converted.data.operations[*region.members.first().unwrap()].kind
    else {
        panic!("branch");
    };
    for child in [then_region, else_region] {
        let ops = &converted.data.regions[child].members;
        assert_eq!(ops.len(), 1);
        assert!(matches!(
            converted.data.operations[*ops.first().unwrap()].kind,
            OperationKind::Call { .. }
        ));
    }
    assert!(matches!(
        converted.data.operations[*region.members.last().unwrap()].kind,
        OperationKind::Call { .. }
    ));
    assert_eq!(
        converted.data.expressions[region.results[0]].kind,
        ExprKind::Int("0".into())
    );
}

#[test]
fn canonical_scremas_preserve_tuple_components_callable_captures_and_ownership() {
    let mut saw_capture = false;
    let mut saw_tuple = false;
    let mut saw_unique = false;
    for input in [
        "entry mapped(xs: []i32, offset: i32) []i32 = map(|x: i32| x + offset, xs)",
        "entry prefix(xs: *[]i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)",
        "entry pairs(xs: [8](i32, i32)) [8](i32, i32) = scan(|a: (i32, i32), b: (i32, i32)| (a.0 + b.0, a.1 + b.1), (0, 0), xs)",
        "entry summed(xs: []i32) (i32, i32) = reduce(|a: (i32, i32), b: (i32, i32)| (a.0 + b.0, a.1 + b.1), (0, 0), map(|x: i32| (x, x + 1), xs))",
    ] {
        let program = source(input);
        let converted = convert_program(&program).unwrap();
        verify_sources(&program, &converted);
        for op in converted.data.operations.values() {
            let OperationKind::Screma { form, ownership, .. } = &op.kind else { continue; };
            assert_eq!(ownership.len(), 1);
            saw_unique |= ownership[0] == types::SoacOwnership::UniqueInput;
            let bodies = std::iter::once(&form.pre).chain(std::iter::once(&form.post))
                .chain(form.scans.iter().map(|scan| &scan.operator))
                .chain(form.reductions.iter().map(|reduction| &reduction.operator));
            for body in bodies {
                if let super::SoacBody::Apply { region, parameters, results, captures } = body {
                    saw_capture |= !captures.is_empty();
                    assert_eq!(results.len(), 1);
                    saw_tuple |= matches!(converted.data.types[results[0]].ty, types::Type::Constructed(TypeName::Tuple(_), _));
                    assert_eq!(converted.data.regions[*region].parameters.len(), parameters.len() + captures.len());
                }
            }
            for scan in &form.scans { assert_eq!(scan.neutral.len(), 1); }
            for reduction in &form.reductions {
                assert_eq!(reduction.neutral.len(), 1);
                assert!(!reduction.commutative);
                assert!(matches!(&form.post, super::SoacBody::Identity(types) if types.is_empty()));
            }
        }
    }
    assert!(saw_capture && saw_tuple && saw_unique);
}

#[test]
fn anonymous_body_captures_are_trailing_region_parameters() {
    let mut f = Fixture::new();
    let mut body = f.soac_body();
    body.lam.body = Box::new(f.term(TermKind::Var(VarRef::Symbol(f.y))));
    let input = f.array();
    let map = f.term(TermKind::Soac(SoacOp::Map {
        lam: body,
        inputs: vec![input],
        destination: types::SoacOwnership::Fresh,
    }));
    let program = f.program(vec![map]);
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    let operation = converted.data.operations.values().next().unwrap();
    let OperationKind::Screma { form, .. } = &operation.kind else {
        panic!("map")
    };
    let super::SoacBody::Apply {
        region,
        parameters,
        captures,
        ..
    } = &form.pre
    else {
        panic!("body application")
    };
    let body = &converted.data.regions[*region];
    assert_eq!(parameters.len(), 1);
    assert_eq!(captures.len(), 1);
    assert_eq!(body.parameters.len(), 2);
    assert_eq!(
        converted.data.expressions[body.results[0]].kind,
        ExprKind::Parameter(body.parameters[1])
    );
    assert_eq!(
        converted.data.expressions[captures[0]].kind,
        ExprKind::Int("37".into())
    );
    assert!(converted.data.definitions.values().all(|def| def.body != *region));
}

#[test]
fn named_body_regions_resolve_forward_references() {
    let mut program = source("entry mapped(xs: [4]i32, offset: i32) [4]i32 = map(|x: i32| x + offset, xs)");
    // Import the caller before its lifted body, regardless of pipeline ordering.
    program.defs.sort_by_key(|def| !matches!(def.meta, tlc::DefMeta::EntryPoint(_)));
    assert!(matches!(program.defs[0].meta, tlc::DefMeta::EntryPoint(_)));
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    let entry = converted.data.entries.values().next().unwrap();
    let caller = converted.data.definitions[entry.definition].body;
    let op = *converted.data.regions[caller].members.first().unwrap();
    let OperationKind::Screma { form, .. } = &converted.data.operations[op].kind else {
        panic!("map")
    };
    let super::SoacBody::Apply {
        region,
        parameters,
        captures,
        ..
    } = &form.pre
    else {
        panic!("body application")
    };
    assert_ne!(*region, caller);
    assert_eq!(captures.len(), 1);
    assert_eq!(
        converted.data.regions[*region].parameters.len(),
        parameters.len() + captures.len()
    );
    assert!(converted.data.definitions.values().any(|def| def.body == *region));
}
