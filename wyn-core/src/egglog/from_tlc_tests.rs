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

fn check(graph: &mut EGraph, command: &str) {
    graph.parse_and_run_program(None, command).unwrap_or_else(|error| panic!("{command}\n{error}"));
}

fn source(source: &str) -> tlc::stage::InputSliceBoundsInferred {
    tlc::infer_input_slice_bounds(test_pipeline::compile_to_reachable(source))
}

fn verify_sources(program: &tlc::stage::InputSliceBoundsInferred, converted: &Converted) {
    let mut graph = run(converted);
    assert_eq!(program.defs.len(), converted.data.definitions.len());
    assert_eq!(program.symbols.len(), converted.data.symbols.len());
    let mut count = 0;
    for def in &program.defs {
        def.body.walk(&mut |term: &Term| {
            // The TLC walker exposes ArrayExpr::Var through synthetic Term
            // adapters. Those atoms have no stored source term or provenance.
            if term.id != tlc::TermId::SYNTHETIC {
                count += 1;
            }
            tlc::WalkDecision::Recurse
        });
    }
    assert_eq!(
        count,
        converted.data.terms.len(),
        "every child, including capture expressions, must be imported"
    );
    for (id, term) in &converted.data.terms {
        assert!(converted.data.types.get(term.ty).is_some());
        assert!(converted.data.definitions.get(term.definition).is_some());
        check(
            &mut graph,
            &format!("(check (SourceTerm {} {}))", id.egglog(), id.binding_name()),
        );
    }
    for ((id, def), source) in converted.data.definitions.iter().zip(&program.defs) {
        assert_eq!(converted.data.symbols[def.symbol].source, source.name);
        assert_eq!(converted.data.types[def.ty].ty, source.ty);
        assert_eq!(def.arity, source.arity);
        assert_eq!(def.package, source.package);
        assert_eq!(def.param_diets, source.param_diets);
        assert_eq!(def.return_diet, source.return_diet);
        check(
            &mut graph,
            &format!(
                "(check (Definition {} {} {} {}))",
                id.egglog(),
                def.symbol.egglog(),
                def.ty.egglog(),
                def.body.binding_name(),
            ),
        );
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
    assert!(converted.data.terms.is_empty());
    assert_eq!(converted.data.programs.len(), 1);
    let mut graph = run(&converted);
    check(&mut graph, "(check (Program (ProgramId 0)))");
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
    let mut graph = run(&converted);
    for (id, bound) in &converted.data.input_bounds {
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
        check(
            &mut graph,
            &format!(
                "(check (InputBound {} {} {}))",
                id.egglog(),
                bound.entry.egglog(),
                bound.symbol.egglog()
            ),
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
        ArrayExpr::Var(VarRef::Symbol(self.x), i32_ty())
    }
}

fn i32_ty() -> types::Type {
    types::Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
fn literals_stay_in_the_program_and_names_stay_in_arenas() {
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
    assert!(emitted.contains("(Int \"18446744073709551615\")"));
    assert!(emitted.contains("(FloatBits 2147483648)"));
    assert!(emitted.contains("(FloatBits 2143289410)"));
    assert!(!emitted.contains("shadowed"));
    assert!(!emitted.contains("external"));
    assert!(!emitted.contains("Wyn"));
    assert_eq!(
        converted.data.externs.values().next().unwrap().linkage_name,
        "external\"name\n(with syntax)"
    );
    let shadowed: Vec<_> =
        converted.data.symbols.iter().filter(|(_, symbol)| symbol.name == "shadowed").collect();
    let mut graph = run(&converted);
    check(
        &mut graph,
        &format!(
            "(fail (check (= (Var {}) (Var {}))))",
            shadowed[0].0.egglog(),
            shadowed[1].0.egglog()
        ),
    );
}

#[test]
fn repeated_effectful_calls_keep_distinct_occurrences() {
    let mut f = Fixture::new();
    let callee = f.term(TermKind::Extern("side_effect".into()));
    let call = f.term(TermKind::App {
        func: Box::new(callee),
        args: vec![],
    });
    let source_id = call.id;
    let tuple = f.term(TermKind::Tuple(vec![call.clone(), call]));
    let program = f.program(vec![tuple]);
    let converted = convert_program(&program).unwrap();
    let ids: Vec<_> = converted
        .data
        .terms
        .iter()
        .filter(|(_, term)| term.source == source_id)
        .map(|(id, _)| *id)
        .collect();
    assert_eq!(ids.len(), 2);
    let mut graph = run(&converted);
    check(
        &mut graph,
        &format!(
            "(fail (check (= {} {})))",
            ids[0].binding_name(),
            ids[1].binding_name()
        ),
    );
}

#[test]
fn equal_constants_share_an_eclass_without_losing_provenance() {
    let mut f = Fixture::new();
    let left = f.int("42");
    let right = f.int("42");
    let source_ids = [left.id, right.id];
    let tuple = f.term(TermKind::Tuple(vec![left, right]));
    let converted = convert_program(&f.program(vec![tuple])).unwrap();
    let ids: Vec<_> = converted
        .data
        .terms
        .iter()
        .filter(|(_, term)| source_ids.contains(&term.source))
        .map(|(id, _)| *id)
        .collect();
    assert_eq!(ids.len(), 2);
    let mut graph = run(&converted);
    check(
        &mut graph,
        &format!("(check (= {} {}))", ids[0].binding_name(), ids[1].binding_name()),
    );
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
    let terms = ops.into_iter().map(|op| f.term(TermKind::Soac(op))).collect();
    let tuple = f.term(TermKind::Tuple(terms));
    let program = f.program(vec![tuple]);
    let converted = convert_program(&program).unwrap();
    verify_sources(&program, &converted);
    assert_eq!(converted.data.bucket_shapes.len(), 1);
    let shape = converted.data.bucket_shapes.values().next().unwrap();
    assert_eq!(shape.input_dimensions, vec![vec![0], vec![1, 0]]);
    assert_eq!(shape.domain_rank, 2);
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
