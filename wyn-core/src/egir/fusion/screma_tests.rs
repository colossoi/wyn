use super::*;
use crate::compile_thru_tlc;
use crate::egir;
use crate::egir::types::{SideEffectKind, Soac, SoacEffect};
use crate::tlc;
use crate::to_egraph;

fn reified(source: &str) -> Segmented {
    let program = compile_thru_tlc(source).expect("compile through TLC");
    let program = tlc::infer_input_slice_bounds(program);
    let program = to_egraph(program).expect("convert to raw EGIR");
    egir::reify_soacs(program)
}

#[test]
fn conditional_lambda_results_project_through_structured_cfg() {
    let program = reified(
        r#"
entry conditional_map<[n]>(xs: [n]i32) [n]i32 =
  map(|x: i32| if x < 0 then 0 - x else x, xs)
"#,
    );
    let lambda = program
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            Some(op.form.pre.clone())
        })
        .expect("conditional map Screma");
    assert!(lambda_results_projectable(&program, &lambda, 0..1));

    let mut wrapper = EGraph::new();
    let params = lambda_ops::named_parameters(&lambda.parameter_types, "argument");
    let arguments = lambda_ops::function_parameters(&mut wrapper, &params)
        .into_iter()
        .map(|argument| argument.value())
        .collect::<Vec<_>>();
    let mut identities = program.data.identities.clone();
    let outer_types = LookupMap::new();
    let mut context = Context {
        program: &program,
        identities: &mut identities,
        scope: "conditional_projection_test",
        span: Span::generated(),
        outer_types: &outer_types,
    };
    let (results, projected) = emit_projected_lambda_results(
        &mut wrapper,
        &mut context,
        "selected_result",
        &lambda,
        &arguments,
        0..1,
    )
    .expect("project conditional result");
    let projected = projected.into_iter().next().expect("CFG projection should synthesize a helper region");
    assert_eq!(results.len(), 1);
    assert_eq!(projected.params.len(), 1);
    assert!(projected.graph.skeleton.blocks.len() > 1);
    projected.graph.skeleton.verify_branch_arities().expect("projected helper CFG");
}
