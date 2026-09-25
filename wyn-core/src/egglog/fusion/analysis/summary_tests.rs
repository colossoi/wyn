use super::*;
use crate::egglog::fusion::analysis::Egglog;
use crate::egglog::{select_tests, OperationId, SCHEMA};
use egglog_engine::EGraph;

#[test]
fn select_keeps_choice_dependencies_instead_of_generic_application_dependencies() {
    let program = select_tests::eager_program("entry choose(c:bool,x:i32,y:i32) i32=if c then x else y");
    let expression = program
        .ir
        .expressions
        .iter()
        .find_map(|(&id, _)| program.ir.conditional_value(id).map(|_| id))
        .unwrap();
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, SCHEMA).unwrap();
    graph
        .update(|state| {
            let mut sink = Egglog::new(state)?;
            let parameters = program
                .ir
                .parameters
                .iter()
                .enumerate()
                .map(|(index, (&id, _))| Ok((id, sink.scan(OperationId::from(index as u32))?)))
                .collect::<Result<BTreeMap<_, _>, Error>>()?;
            let value = scalar(
                &program.ir,
                expression,
                &parameters,
                &mut BTreeMap::new(),
                &mut sink,
            )?;
            sink.collective_dependency(OperationId::from(0), value.unwrap())
        })
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            "(check (CollectiveDependency (OperationId 0) (ChoiceDependency c yes no)))",
        )
        .unwrap();
}
