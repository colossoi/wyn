//! The keys of occurrence properties are bounded by the actual operands and results.
use super::ir::Code;
use crate::egglog::graph::State;
use egglog_engine::{Error, Read, Value, Write};

#[derive(Clone, Copy)]
enum Keys {
    Operation,
    Slot(i64),
    OutputInput(i64, i64),
}

pub(super) struct Property {
    pub table: &'static str,
    keys: Keys,
    relation: bool,
}

pub(super) enum Contents {
    Present,
    Value(Value),
}

impl Property {
    pub fn read(&self, sink: &State<'_, '_, '_>, operation: Value) -> Result<Option<Contents>, Error> {
        let table = self.table;
        if self.relation {
            let present = match self.keys {
                Keys::Operation => sink.contains(table, operation)?,
                Keys::Slot(slot) => sink.contains(table, (operation, slot))?,
                Keys::OutputInput(out, input) => sink.contains(table, (operation, out, input))?,
            };
            return Ok(present.then_some(Contents::Present));
        }
        let value = match self.keys {
            Keys::Operation => sink.lookup(table, operation)?,
            Keys::Slot(slot) => sink.lookup(table, (operation, slot))?,
            Keys::OutputInput(output, input) => sink.lookup(table, (operation, output, input))?,
        };
        Ok(value.map(Contents::Value))
    }

    pub fn write(
        &self,
        sink: &mut State<'_, '_, '_>,
        operation: Value,
        value: Contents,
    ) -> Result<(), Error> {
        let table = self.table;
        let Contents::Value(value) = value else {
            match self.keys {
                Keys::Operation => sink.add(table, operation)?,
                Keys::Slot(slot) => sink.add(table, (operation, slot))?,
                Keys::OutputInput(output, input) => sink.add(table, (operation, output, input))?,
            };
            return Ok(());
        };
        match self.keys {
            Keys::Operation => sink.set(table, operation, value),
            Keys::Slot(slot) => sink.set(table, (operation, slot), value),
            Keys::OutputInput(output, input) => sink.set(table, (operation, output, input), value),
        }
    }

    fn remove(&self, sink: &mut State<'_, '_, '_>, operation: Value) -> Result<(), Error> {
        match self.keys {
            Keys::Operation => sink.remove(self.table, operation),
            Keys::Slot(slot) => sink.remove(self.table, (operation, slot)),
            Keys::OutputInput(output, input) => sink.remove(self.table, (operation, output, input)),
        }
    }
}

pub(super) fn visit(
    code: &Code,
    inputs: i64,
    outputs: i64,
    mut each: impl FnMut(Property) -> Result<(), Error>,
) -> Result<(), Error> {
    for table in [
        "InstructionWork",
        "LiteralBits",
        "OperationWidth",
        "OperationExecution",
        "ScalarPayload",
    ] {
        each(Property {
            table,
            keys: Keys::Operation,
            relation: false,
        })?;
    }
    for table in [
        "DiscardableOperation",
        "RepeatableOperation",
        "OrderedOperation",
        "LaneDependentOperation",
        "ProvenTotalOperation",
    ] {
        each(Property {
            table,
            keys: Keys::Operation,
            relation: true,
        })?;
    }
    let (table, keys) = match code {
        Code::Constant { .. } => ("ConstantOperation", Keys::Operation),
        Code::Product => ("ProductOperation", Keys::Operation),
        Code::Project { index } => ("ProjectionOperation", Keys::Slot(*index as i64)),
        Code::Length { axis } => ("LengthOperation", Keys::Slot(*axis as i64)),
        Code::Index => ("IndexOperation", Keys::Operation),
        Code::IndexValue => ("IndexCalculationOperation", Keys::Operation),
        Code::Scratch | Code::Uninitialized => ("ScratchAllocation", Keys::Operation),
        Code::Call => ("CallOperation", Keys::Operation),
        Code::Branch => ("BranchOperation", Keys::Operation),
        Code::CountedLoop => ("CountedLoopOperation", Keys::Operation),
        Code::WhileLoop => ("WhileLoopOperation", Keys::Operation),
        Code::Parallel => ("ParallelOperation", Keys::Operation),
        Code::WithAcc => ("AccumulatorOperation", Keys::Operation),
        Code::Copy => ("MaterializedCopyOperation", Keys::Operation),
        Code::View => ("ViewOperation", Keys::Operation),
        Code::ArrayLiteral => ("ArrayLiteralOperation", Keys::Operation),
        Code::Range => ("RangeOperation", Keys::Operation),
        _ => ("", Keys::Operation),
    };
    if !table.is_empty() {
        each(Property {
            table,
            keys,
            relation: true,
        })?;
    }
    for slot in 0..inputs {
        let keys = Keys::Slot(slot);
        each(Property {
            table: "InputDiet",
            keys,
            relation: false,
        })?;
        for table in ["ReadsInput", "WritesInput", "ConsumesInput"] {
            each(Property {
                table,
                keys,
                relation: true,
            })?;
        }
    }
    for slot in 0..outputs {
        let keys = Keys::Slot(slot);
        if *code == Code::Call {
            each(Property {
                table: "CallReturnType",
                keys,
                relation: false,
            })?;
        }
        each(Property {
            table: "FreshOutput",
            keys,
            relation: true,
        })?;
        for input in 0..inputs {
            let keys = Keys::OutputInput(slot, input);
            for table in ["AliasesInput", "PreservesInputShape"] {
                each(Property {
                    table,
                    keys,
                    relation: true,
                })?;
            }
        }
    }
    Ok(())
}

pub(super) fn remove(
    sink: &mut State<'_, '_, '_>,
    operation: Value,
    code: &Code,
    inputs: i64,
    outputs: i64,
) -> Result<(), Error> {
    visit(code, inputs, outputs, |entry| entry.remove(sink, operation))
}
