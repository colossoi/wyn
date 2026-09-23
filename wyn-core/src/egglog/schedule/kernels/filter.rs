use super::super::length;
use super::{chunks, error, singleton, Instruction, OptimizeError, Planner, Value, WIDTH};
use crate::egglog::data::intern_type;
use crate::egglog::data::{Array, BlockId, OperationId, OperationKind, SoacBody};
use crate::types::{Type, TypeName};

impl Planner<'_> {
    pub(super) fn parallel_filter(
        &mut self,
        op: OperationId,
        host: BlockId,
    ) -> Result<Value, OptimizeError> {
        let OperationKind::Filter {
            map,
            body,
            post,
            inputs,
            ..
        } = self.data.operations[op].kind.clone()
        else {
            return Err(error("invalid compaction recipe"));
        };
        let n = length(&inputs);
        let output = self.slot(op, "output", 0, false);
        let count = self.slot(op, "length", 0, false);

        let kernel = self.kernel(op, "compact");
        let element = intern_type(&mut self.data.ir, Type::Constructed(TypeName::UInt(32), vec![]));
        let shared = [0, 1].map(|id| Value::Workgroup {
            id,
            count: WIDTH,
            element,
        });
        let lane = Value::op("local_id", []);
        // Uniform trip counts keep every lane participating in every barrier,
        // including the last partial tile. Each lane carries the same running
        // count; processing tiles in order makes the compaction stable.
        let tile_loop = self.start_loop(kernel, Value::Int(0), chunks(n.clone()), vec![Value::Int(0)]);
        let index = Value::op(
            "add",
            [
                Value::op("mul", [tile_loop.index.clone(), Value::Int(WIDTH)]),
                lane.clone(),
            ],
        );
        let valid = Value::op("lt", [index.clone(), n.clone()]);
        let read = self.block(kernel, vec![]);
        let pad = self.block(kernel, vec![]);
        let loaded = self.block(kernel, vec!["flag".into()]);
        self.branch(tile_loop.body, valid, read, pad, Some(loaded));
        // Evaluate each live predicate in the same invocation that scans its
        // tile. Padding lanes skip the predicate but participate in barriers.
        let elements = self.read_inputs(read, &inputs, index.clone());
        let elements = self.invoke_body(read, &map, elements, "map")?;
        let result = self.invoke_body(read, &body, elements, "predicate")?;
        let [predicate] = result.as_slice() else {
            return Err(error("filter predicate must return one value"));
        };
        let flag = Value::op("bool_to_u32", [predicate.clone()]);
        self.jump(read, loaded, vec![flag]);
        self.jump(pad, loaded, vec![Value::Int(0)]);
        let flag = Value::local("flag");
        let mut prefix = flag.clone();
        self.store(loaded, &shared[0], lane.clone(), prefix.clone());
        self.emit(loaded, Instruction::Barrier);
        let mut block = loaded;
        let mut bank = 0;
        for step in 0..WIDTH.trailing_zeros() {
            let distance = Value::Int(1 << step);
            let read = self.block(kernel, vec![]);
            let skip = self.block(kernel, vec![]);
            let name = format!("prefix{step}");
            let merge = self.block(kernel, vec![name.clone()]);
            self.branch(
                block,
                Value::op("ge", [lane.clone(), distance.clone()]),
                read,
                skip,
                Some(merge),
            );
            let peer = self.load(
                read,
                shared[bank].clone(),
                Value::op("sub", [lane.clone(), distance]),
                "peer",
            );
            self.jump(read, merge, vec![Value::op("add", [prefix.clone(), peer])]);
            self.jump(skip, merge, vec![prefix]);
            prefix = Value::Local(name);
            bank = 1 - bank;
            self.store(merge, &shared[bank], lane.clone(), prefix.clone());
            self.emit(merge, Instruction::Barrier);
            block = merge;
        }
        let total = self.load(block, shared[bank].clone(), Value::Int(WIDTH - 1), "total");
        // Finish all shared reads before any lane starts the next tile.
        self.emit(block, Instruction::Barrier);
        let selected = self.block(kernel, vec![]);
        let done = self.block(kernel, vec![]);
        self.branch(
            block,
            Value::op("ne", [flag, Value::Int(0)]),
            selected,
            done,
            Some(done),
        );
        let elements = self.read_inputs(selected, &inputs, index);
        let values = self.invoke_body(selected, &map, elements, "map")?;
        let values = self.invoke_body(selected, &post, values, "post")?;
        let [value] = values.as_slice() else {
            return Err(error("filter map result arity"));
        };
        self.store(
            selected,
            &output,
            Value::op(
                "add",
                [
                    tile_loop.state[0].clone(),
                    Value::op("sub", [prefix, Value::Int(1)]),
                ],
            ),
            value.clone(),
        );
        self.jump(selected, done, vec![]);
        self.finish_loop(
            &tile_loop,
            done,
            vec![Value::op("add", [tile_loop.state[0].clone(), total])],
        );
        let write = self.block(kernel, vec![]);
        let done = self.block(kernel, vec![]);
        self.branch(
            tile_loop.done,
            Value::op("eq", [lane, Value::Int(0)]),
            write,
            done,
            Some(done),
        );
        // Lane zero also publishes count-derived ABI outputs in the epilogue.
        // The loop's initial state supplies zero even when the input is empty.
        self.store(write, &count, Value::Int(0), tile_loop.state[0].clone());
        self.jump(write, done, vec![]);
        self.returns(done, vec![]);
        self.dispatch(op, host, kernel);
        Ok(Value::op("slice", [output, singleton(count)]))
    }

    pub(super) fn serial_filter(
        &mut self,
        op: OperationId,
        entry: BlockId,
        local: bool,
        map: &SoacBody,
        body: &SoacBody,
        post: &SoacBody,
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
        let values = self.invoke_body(selected, post, vec![value], "post")?;
        let [value] = values.as_slice() else {
            return Err(error("filter post-map result arity"));
        };
        self.store(selected, &output, loop_.state[0].clone(), value.clone());
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
