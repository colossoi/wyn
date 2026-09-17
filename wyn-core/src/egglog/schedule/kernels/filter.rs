use super::super::length;
use super::{chunks, error, singleton, OptimizeError, Planner, Value, WIDTH};
use crate::egglog::data::{Array, BlockId, OperationId, OperationKind, SoacBody};

impl Planner<'_> {
    pub(super) fn parallel_filter(
        &mut self,
        op: OperationId,
        host: BlockId,
    ) -> Result<Value, OptimizeError> {
        let OperationKind::Filter {
            map, body, inputs, ..
        } = self.data.operations[op].kind.clone()
        else {
            return Err(error("invalid compaction recipe"));
        };
        let n = length(&inputs);
        let chunks = chunks(n.clone());
        let output = self.slot(op, "output", 0, false);
        let count = self.slot(op, "length", 0, false);
        let flags = self.slot(op, "flags", 0, false);
        let local_offsets = self.slot(op, "local_offsets", 0, false);
        let totals = self.slot(op, "totals", 0, false);
        let offsets = self.slot(op, "offsets", 0, false);

        let flags_kernel = self.kernel(op, "flags");
        let invocation = self.invocations(flags_kernel, n.clone());
        let (active, i) = (invocation.body, invocation.index.clone());
        let elements = self.read_inputs(active, &inputs, i.clone());
        let elements = self.invoke_body(active, &map, elements, "map")?;
        let result = self.invoke_body(active, &body, elements, "predicate")?;
        let [predicate] = result.as_slice() else {
            return Err(error("filter predicate must return one value"));
        };
        self.store(active, &flags, i, Value::op("bool_to_u32", [predicate.clone()]));
        self.finish_loop(&invocation, active, vec![]);
        self.dispatch(op, host, flags_kernel);

        let prefix_kernel = self.kernel(op, "local_offsets");
        let invocation = self.invocations(prefix_kernel, chunks.clone());
        let (active, chunk) = (invocation.body, invocation.index.clone());
        let start = Value::op("mul", [chunk.clone(), Value::Int(WIDTH)]);
        let bound = Value::op(
            "min",
            [n.clone(), Value::op("add", [start.clone(), Value::Int(WIDTH)])],
        );
        let loop_ = self.start_loop(active, start, bound, vec![Value::Int(0)]);
        self.store(
            loop_.body,
            &local_offsets,
            loop_.index.clone(),
            loop_.state[0].clone(),
        );
        let flag = self.load(loop_.body, flags.clone(), loop_.index.clone(), "flag");
        self.finish_loop(
            &loop_,
            loop_.body,
            vec![Value::op("add", [loop_.state[0].clone(), flag])],
        );
        self.store(loop_.done, &totals, chunk, loop_.state[0].clone());
        self.finish_loop(&invocation, loop_.done, vec![]);
        self.dispatch(op, host, prefix_kernel);

        let combine = self.kernel(op, "offsets");
        let loop_ = self.start_loop(combine, Value::Int(0), chunks, vec![Value::Int(0)]);
        self.store(loop_.body, &offsets, loop_.index.clone(), loop_.state[0].clone());
        let total = self.load(loop_.body, totals.clone(), loop_.index.clone(), "total");
        self.finish_loop(
            &loop_,
            loop_.body,
            vec![Value::op("add", [loop_.state[0].clone(), total])],
        );
        self.store(loop_.done, &count, Value::Int(0), loop_.state[0].clone());
        self.returns(loop_.done, vec![]);
        self.dispatch(op, host, combine);

        let write = self.kernel(op, "compact");
        let invocation = self.invocations(write, n.clone());
        let (active, i) = (invocation.body, invocation.index.clone());
        let selected = self.block(write, vec![]);
        let done = self.block(write, vec![]);
        let flag = self.load(active, flags.clone(), i.clone(), "flag");
        self.branch(
            active,
            Value::op("ne", [flag, Value::Int(0)]),
            selected,
            done,
            Some(done),
        );
        let local = self.load(selected, local_offsets.clone(), i.clone(), "local");
        let offset = self.load(
            selected,
            offsets.clone(),
            Value::op("div", [i.clone(), Value::Int(WIDTH)]),
            "offset",
        );
        let elements = self.read_inputs(selected, &inputs, i);
        let values = self.invoke_body(selected, &map, elements, "map")?;
        let [value] = values.as_slice() else {
            return Err(error("filter map result arity"));
        };
        self.store(
            selected,
            &output,
            Value::op("add", [offset, local]),
            value.clone(),
        );
        self.jump(selected, done, vec![]);
        self.finish_loop(&invocation, done, vec![]);
        self.dispatch(op, host, write);
        Ok(Value::op("slice", [output, singleton(count)]))
    }

    pub(super) fn serial_filter(
        &mut self,
        op: OperationId,
        entry: BlockId,
        local: bool,
        map: &SoacBody,
        body: &SoacBody,
        inputs: &[Array],
    ) -> Result<(BlockId, Value), OptimizeError> {
        let n = length(inputs);
        let output = self.slot(op, "output", 0, local);
        let count = self.slot(op, "length", 0, local);
        let loop_ = self.start_loop(entry, Value::Int(0), n, vec![Value::Int(0)]);
        let elements = self.read_inputs(loop_.body, inputs, loop_.index.clone());
        let mapped = self.invoke_body(loop_.body, map, elements, "map")?;
        let [value] = mapped.as_slice() else {
            return Err(error("filter map result arity"));
        };
        let value = value.clone();
        let values = self.invoke_body(loop_.body, body, vec![value.clone()], "predicate")?;
        let [predicate] = values.as_slice() else {
            return Err(error("filter predicate must return one value"));
        };
        let owner = self.data.state.blocks[entry].function;
        let selected = self.block(owner, vec![]);
        let skipped = self.block(owner, vec![]);
        let count_name = format!("filter_count{}", loop_.body.as_u32());
        let merge = self.block(owner, vec![count_name.clone()]);
        self.branch(loop_.body, predicate.clone(), selected, skipped, Some(merge));
        self.store(selected, &output, loop_.state[0].clone(), value);
        self.jump(
            selected,
            merge,
            vec![Value::op("add", [loop_.state[0].clone(), Value::Int(1)])],
        );
        self.jump(skipped, merge, loop_.state.clone());
        self.finish_loop(&loop_, merge, vec![Value::Local(count_name)]);
        self.store(loop_.done, &count, Value::Int(0), loop_.state[0].clone());
        Ok((loop_.done, Value::op("slice", [output, singleton(count)])))
    }
}
