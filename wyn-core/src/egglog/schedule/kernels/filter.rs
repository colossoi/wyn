use super::super::{length, OperationId, OperationKind};
use super::{
    array_value, chunks, error, groups, singleton, Array, BlockId, BufferId, OptimizeError, Planner,
    SoacBody, Storage, Value, WIDTH,
};
use crate::{ast::TypeName, types};

impl Planner<'_> {
    pub(super) fn parallel_filter(
        &mut self,
        op: OperationId,
        host: BlockId,
    ) -> Result<(Value, Vec<BufferId>), OptimizeError> {
        let OperationKind::Filter { body, input, .. } = self.data.operations[op].kind.clone() else {
            return Err(error("invalid compaction recipe"));
        };
        let captures = self.captures(op);
        let n = length(std::slice::from_ref(&input));
        let chunks = chunks(n.clone());
        let element = self.filter_element(&body)?;
        let output = self.allocate(host, "output", n.clone(), element, Storage::Device);
        let count = self.allocate(host, "length", Value::Int(1), uint(), Storage::Device);
        let flags = self.allocate(host, "flags", n.clone(), uint(), Storage::Device);
        let local_offsets = self.allocate(host, "local_offsets", n.clone(), uint(), Storage::Device);
        let totals = self.allocate(host, "totals", chunks.clone(), uint(), Storage::Device);
        let offsets = self.allocate(host, "offsets", chunks.clone(), uint(), Storage::Device);

        let flags_kernel = self.kernel("flags", &captures, WIDTH);
        let invocation = self.invocations(flags_kernel, n.clone());
        let (active, i) = (invocation.body, invocation.index.clone());
        let element = self.load(active, array_value(&input), i.clone(), "element");
        let result = self.invoke_body(active, &body, vec![element], "predicate")?;
        let [predicate] = result.as_slice() else {
            return Err(error("filter predicate must return one value"));
        };
        self.store(active, flags, i, Value::op("bool_to_u32", [predicate.clone()]));
        self.finish_loop(&invocation, active, vec![]);
        self.dispatch(
            op,
            host,
            flags_kernel,
            groups(n.clone()),
            captures.clone(),
            Default::default(),
            [flags].into(),
        );

        let prefix_kernel = self.kernel("local_offsets", &captures, WIDTH);
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
            local_offsets,
            loop_.index.clone(),
            loop_.state[0].clone(),
        );
        let flag = self.load(loop_.body, Value::Buffer(flags), loop_.index.clone(), "flag");
        self.finish_loop(
            &loop_,
            loop_.body,
            vec![Value::op("add", [loop_.state[0].clone(), flag])],
        );
        self.store(loop_.done, totals, chunk, loop_.state[0].clone());
        self.finish_loop(&invocation, loop_.done, vec![]);
        self.dispatch(
            op,
            host,
            prefix_kernel,
            groups(chunks.clone()),
            captures.clone(),
            [flags].into(),
            [local_offsets, totals].into(),
        );

        let combine = self.kernel("offsets", &captures, 1);
        let loop_ = self.start_loop(combine, Value::Int(0), chunks, vec![Value::Int(0)]);
        self.store(loop_.body, offsets, loop_.index.clone(), loop_.state[0].clone());
        let total = self.load(loop_.body, Value::Buffer(totals), loop_.index.clone(), "total");
        self.finish_loop(
            &loop_,
            loop_.body,
            vec![Value::op("add", [loop_.state[0].clone(), total])],
        );
        self.store(loop_.done, count, Value::Int(0), loop_.state[0].clone());
        self.returns(loop_.done, vec![]);
        self.dispatch(
            op,
            host,
            combine,
            Value::Int(1),
            captures.clone(),
            [totals].into(),
            [offsets, count].into(),
        );

        let write = self.kernel("compact", &captures, WIDTH);
        let invocation = self.invocations(write, n.clone());
        let (active, i) = (invocation.body, invocation.index.clone());
        let selected = self.block(write, vec![]);
        let done = self.block(write, vec![]);
        let flag = self.load(active, Value::Buffer(flags), i.clone(), "flag");
        self.branch(active, Value::op("ne", [flag, Value::Int(0)]), selected, done);
        let local = self.load(selected, Value::Buffer(local_offsets), i.clone(), "local");
        let offset = self.load(
            selected,
            Value::Buffer(offsets),
            Value::op("div", [i.clone(), Value::Int(WIDTH)]),
            "offset",
        );
        let value = self.load(selected, array_value(&input), i, "element");
        self.store(selected, output, Value::op("add", [offset, local]), value);
        self.jump(selected, done, vec![]);
        self.finish_loop(&invocation, done, vec![]);
        self.dispatch(
            op,
            host,
            write,
            groups(n),
            captures,
            [flags, local_offsets, offsets].into(),
            [output].into(),
        );
        Ok((
            Value::op("slice", [Value::Buffer(output), singleton(count)]),
            vec![output, count],
        ))
    }

    pub(super) fn serial_filter(
        &mut self,
        entry: BlockId,
        allocate: BlockId,
        storage: Storage,
        body: &SoacBody,
        input: &Array,
    ) -> Result<(BlockId, Value, Vec<BufferId>), OptimizeError> {
        let n = length(std::slice::from_ref(input));
        let element = self.filter_element(body)?;
        let output = self.allocate(allocate, "output", n.clone(), element, storage);
        let count = self.allocate(allocate, "length", Value::Int(1), uint(), storage);
        let loop_ = self.start_loop(entry, Value::Int(0), n, vec![Value::Int(0)]);
        let value = self.load(loop_.body, array_value(input), loop_.index.clone(), "element");
        let values = self.invoke_body(loop_.body, body, vec![value.clone()], "predicate")?;
        let [predicate] = values.as_slice() else {
            return Err(error("filter predicate must return one value"));
        };
        let owner = self.data.blocks[entry].function;
        let selected = self.block(owner, vec![]);
        let skipped = self.block(owner, vec![]);
        self.branch(loop_.body, predicate.clone(), selected, skipped);
        self.store(selected, output, loop_.state[0].clone(), value);
        self.finish_loop(
            &loop_,
            selected,
            vec![Value::op("add", [loop_.state[0].clone(), Value::Int(1)])],
        );
        self.finish_loop(&loop_, skipped, loop_.state.clone());
        self.store(loop_.done, count, Value::Int(0), loop_.state[0].clone());
        Ok((
            loop_.done,
            Value::op("slice", [Value::Buffer(output), singleton(count)]),
            vec![output, count],
        ))
    }

    fn filter_element(&self, body: &SoacBody) -> Result<types::Type, OptimizeError> {
        let parameters = match body {
            SoacBody::Apply { parameters, .. } | SoacBody::Route { parameters, .. } => parameters,
            SoacBody::Identity(types) => types,
            SoacBody::Compose { first, .. } => return self.filter_element(first),
            SoacBody::Parallel { left, .. } => return self.filter_element(left),
        };
        let [element] = parameters.as_slice() else {
            return Err(error("filter body must take one logical element"));
        };
        Ok(self.data.types[*element].ty.clone())
    }
}

pub(super) fn uint() -> types::Type {
    types::Type::Constructed(TypeName::UInt(32), vec![])
}
