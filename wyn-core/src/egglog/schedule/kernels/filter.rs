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
        let element = intern_type(&mut self.data.ir, Type::Constructed(TypeName::UInt(32), vec![]));
        let shared = [0, 1].map(|id| Value::Workgroup {
            id,
            count: WIDTH,
            element,
        });
        let lane = Value::op("local_id", []);
        let group = Value::op(
            "div",
            [Value::op("global_id", [Value::Int(0)]), Value::Int(WIDTH)],
        );
        let groups = Value::op(
            "div",
            [Value::op("global_size", [Value::Int(0)]), Value::Int(WIDTH)],
        );
        let per_group = Value::op("ceil_div", [chunks.clone(), groups]);
        // Uniform trip counts keep every lane participating in every barrier,
        // including partial tiles and a grid smaller than the logical domain.
        let tile_loop = self.start_loop(prefix_kernel, Value::Int(0), per_group.clone(), vec![]);
        let tile = Value::op(
            "add",
            [Value::op("mul", [group, per_group]), tile_loop.index.clone()],
        );
        let index = Value::op(
            "add",
            [Value::op("mul", [tile.clone(), Value::Int(WIDTH)]), lane.clone()],
        );
        let valid = Value::op("lt", [index.clone(), n.clone()]);
        let read = self.block(prefix_kernel, vec![]);
        let pad = self.block(prefix_kernel, vec![]);
        let loaded = self.block(prefix_kernel, vec!["flag".into()]);
        self.branch(tile_loop.body, valid.clone(), read, pad, Some(loaded));
        let flag = self.load(read, flags.clone(), index.clone(), "flag");
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
            let read = self.block(prefix_kernel, vec![]);
            let skip = self.block(prefix_kernel, vec![]);
            let name = format!("prefix{step}");
            let merge = self.block(prefix_kernel, vec![name.clone()]);
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
        self.emit(block, Instruction::Barrier);
        let write = self.block(prefix_kernel, vec![]);
        let written = self.block(prefix_kernel, vec![]);
        self.branch(block, valid, write, written, Some(written));
        self.store(write, &local_offsets, index, Value::op("sub", [prefix, flag]));
        self.jump(write, written, vec![]);
        let write = self.block(prefix_kernel, vec![]);
        let done = self.block(prefix_kernel, vec![]);
        self.branch(
            written,
            Value::op(
                "and",
                [
                    Value::op("eq", [lane, Value::Int(0)]),
                    Value::op("lt", [tile.clone(), chunks.clone()]),
                ],
            ),
            write,
            done,
            Some(done),
        );
        self.store(write, &totals, tile, total);
        self.jump(write, done, vec![]);
        self.finish_loop(&tile_loop, done, vec![]);
        self.returns(tile_loop.done, vec![]);
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
