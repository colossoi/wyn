use super::data::{intern_expr, intern_type, BuiltinData, Evaluation};
use super::*;
use crate::builtins::catalog;
use crate::types;

pub(super) fn eager_program(source: &str) -> Program<Fused> {
    let tlc = crate::tlc::infer_input_slice_bounds(crate::compile_thru_tlc(source).unwrap());
    let mut program = fuse(from_tlc(&tlc).unwrap()).unwrap();
    make_eager(&mut program.ir);
    program
}

pub(super) fn make_eager(data: &mut Ir) {
    let choices = data
        .expressions
        .iter()
        .filter_map(|(&id, _)| {
            data.conditional_value(id).filter(|(_, mode)| *mode == Evaluation::Lazy).map(|(s, _)| (id, s))
        })
        .collect::<Vec<_>>();
    assert!(!choices.is_empty());
    let builtin = data.builtins.alloc(BuiltinData {
        builtin: catalog().known().select,
        overload_idx: 0,
    });
    for (id, choice) in choices {
        let ty = data.types[data.expressions[id].ty].ty.clone();
        let signature = intern_type(
            data,
            types::function(
                ty.clone(),
                types::function(ty.clone(), types::function(types::bool_type(), ty)),
            ),
        );
        let function = intern_expr(data, signature, ExprKind::Builtin(builtin));
        data.expressions[id].kind = ExprKind::PureApp {
            function,
            args: vec![choice.no, choice.yes, choice.condition],
        };
        assert_eq!(data.conditional_value(id), Some((choice, Evaluation::Eager)));
    }
}

#[test]
fn select_survives_egglog_extraction_placement_and_both_backends() {
    for algebra in [false, true] {
        let program = eager_program("entry choose(c:bool,x:i32,y:i32) i32=if c then x+1 else y+2");
        let program = simplify(insert_expressions(program).unwrap(), algebra).unwrap();
        assert!(program
            .ir
            .expressions
            .iter()
            .any(|(&id, _)| matches!(program.ir.conditional_value(id), Some((_, Evaluation::Eager)))));
        let program = schedule(
            place(program).unwrap(),
            crate::PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        let ssa = to_ssa(&program, crate::CodegenTarget::Portable).unwrap();
        let wgsl = crate::lower_ssa_to_wgsl(ssa.clone()).unwrap();
        assert!(wgsl.contains("select("), "{wgsl}");
        let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let words = crate::lower_ssa_to_spirv(ssa).unwrap().spirv;
        let module = wspirv::dr::load_words(&words).unwrap();
        assert!(module
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instructions)
            .any(|i| i.class.opcode == spirv::Op::Select));
    }
}
