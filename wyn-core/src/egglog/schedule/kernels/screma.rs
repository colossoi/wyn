use super::super::super::data::{body_signature, ExprId};
use super::super::{error, length};
use super::{chunks, singleton, OptimizeError, Planner, Value, WIDTH};
use crate::egglog::data::{Array, BlockId, OperationId, OperationKind, ScremaForm};

impl Planner<'_> {
    pub(super) fn parallel_screma(
        &mut self,
        op: OperationId,
        host: BlockId,
    ) -> Result<Value, OptimizeError> {
        let OperationKind::Screma { form, inputs, .. } = self.data.operations[op].kind.clone() else {
            return Err(error("invalid collective recipe"));
        };
        let n = length(&inputs);
        let (reductions, arrays) = self.screma_outputs(op, &form, false);
        let ns: usize = form.scans.iter().map(|s| s.neutral.len()).sum();
        let nr: usize = form.reductions.iter().map(|r| r.neutral.len()).sum();
        if ns + nr == 0 {
            let kernel = self.kernel(op, "elements");
            let invocation = self.invocations(kernel, n.clone());
            let (active, i) = (invocation.body, invocation.index.clone());
            let args = self.read_inputs(active, &inputs, i.clone());
            let pre = self.invoke_body(active, &form.pre, args, "pre")?;
            let values = self.invoke_body(active, &form.post, pre, "post")?;
            self.write_values(active, &arrays, i, values)?;
            self.finish_loop(&invocation, active, vec![]);
            self.dispatch(op, host, kernel);
            return Ok(result(&reductions, &arrays));
        }

        // One invocation owns a contiguous chunk. This preserves operand order
        // for associative operators that are not commutative. The neutral is
        // the operator's identity, including for the single empty chunk.
        let chunks = chunks(n.clone());
        let neutral = neutrals(&form);
        let partials: Vec<_> =
            neutral.iter().enumerate().map(|(i, _)| self.slot(op, "partial", i, false)).collect();
        let prefixes: Vec<_> =
            neutral.iter().take(ns).enumerate().map(|(i, _)| self.slot(op, "prefix", i, false)).collect();
        let offsets: Vec<_> =
            neutral.iter().take(ns).enumerate().map(|(i, _)| self.slot(op, "offset", i, false)).collect();
        let mapped: Vec<_> = if ns == 0 {
            vec![]
        } else {
            body_signature(&form.pre)
                .1
                .into_iter()
                .skip(ns + nr)
                .enumerate()
                .map(|(i, _)| self.slot(op, "mapped", i, false))
                .collect()
        };
        let kernel = self.kernel(op, "chunks");
        let invocation = self.invocations(kernel, chunks.clone());
        let (active, chunk) = (invocation.body, invocation.index.clone());
        let start = Value::op("mul", [chunk.clone(), Value::Int(WIDTH)]);
        let bound = Value::op(
            "min",
            [Value::op("add", [start.clone(), Value::Int(WIDTH)]), n.clone()],
        );
        let loop_ = self.start_loop(
            active,
            start,
            bound,
            neutral.iter().copied().map(Value::Source).collect(),
        );
        let args = self.read_inputs(loop_.body, &inputs, loop_.index.clone());
        let pre = self.invoke_body(loop_.body, &form.pre, args, "pre")?;
        if pre.len() < ns + nr {
            return Err(error("collective input arity mismatch"));
        }
        let next = self.accumulate(
            loop_.body,
            &form,
            loop_.state.clone(),
            pre[..ns + nr].to_vec(),
            "step",
        )?;
        if ns == 0 {
            let values = self.invoke_body(loop_.body, &form.post, pre[ns + nr..].to_vec(), "post")?;
            self.write_values(loop_.body, &arrays, loop_.index.clone(), values)?;
        } else {
            self.write_values(loop_.body, &prefixes, loop_.index.clone(), next[..ns].to_vec())?;
            self.write_values(loop_.body, &mapped, loop_.index.clone(), pre[ns + nr..].to_vec())?;
        }
        self.finish_loop(&loop_, loop_.body, next);
        self.write_values(loop_.done, &partials, chunk, loop_.state.clone())?;
        self.finish_loop(&invocation, loop_.done, vec![]);
        self.dispatch(op, host, kernel);

        // Dispatch boundaries supply device-wide visibility. No workgroup
        // barrier is used as a substitute for global synchronization.
        let combine = self.kernel(op, "combine");
        let loop_ = self.start_loop(
            combine,
            Value::Int(0),
            chunks.clone(),
            neutral.iter().copied().map(Value::Source).collect(),
        );
        self.write_values(
            loop_.body,
            &offsets,
            loop_.index.clone(),
            loop_.state[..ns].to_vec(),
        )?;
        let incoming = self.read_buffers(loop_.body, &partials, loop_.index.clone());
        let next = self.accumulate(loop_.body, &form, loop_.state.clone(), incoming, "combine")?;
        self.finish_loop(&loop_, loop_.body, next);
        self.write_values(loop_.done, &reductions, Value::Int(0), loop_.state[ns..].to_vec())?;
        self.returns(loop_.done, vec![]);
        self.dispatch(op, host, combine);

        if ns > 0 {
            let finish = self.kernel(op, "offsets");
            let invocation = self.invocations(finish, n.clone());
            let (active, i) = (invocation.body, invocation.index.clone());
            let chunk = Value::op("div", [i.clone(), Value::Int(WIDTH)]);
            let local = self.read_buffers(active, &prefixes, i.clone());
            let offset_values = self.read_buffers(active, &offsets, chunk);
            let mut scan_values = vec![];
            let mut position = 0;
            for scan in &form.scans {
                let end = position + scan.neutral.len();
                let args =
                    offset_values[position..end].iter().chain(&local[position..end]).cloned().collect();
                scan_values.extend(self.invoke_body(
                    active,
                    &scan.operator,
                    args,
                    &format!("offset{position}"),
                )?);
                position = end;
            }
            scan_values.extend(self.read_buffers(active, &mapped, i.clone()));
            let post = self.invoke_body(active, &form.post, scan_values, "post")?;
            self.write_values(active, &arrays, i, post)?;
            self.finish_loop(&invocation, active, vec![]);
            self.dispatch(op, host, finish);
        }
        Ok(result(&reductions, &arrays))
    }

    pub(super) fn serial_screma(
        &mut self,
        op: OperationId,
        entry: BlockId,
        local: bool,
        form: &ScremaForm,
        inputs: &[Array],
    ) -> Result<(BlockId, Value), OptimizeError> {
        let n = length(inputs);
        let (reductions, arrays) = self.screma_outputs(op, form, local);
        let neutral = neutrals(form);
        let ns: usize = form.scans.iter().map(|s| s.neutral.len()).sum();
        let count = neutral.len();
        let loop_ = self.start_loop(
            entry,
            Value::Int(0),
            n,
            neutral.into_iter().map(Value::Source).collect(),
        );
        let args = self.read_inputs(loop_.body, inputs, loop_.index.clone());
        let pre = self.invoke_body(loop_.body, &form.pre, args, "pre")?;
        if pre.len() < count {
            return Err(error("collective input arity mismatch"));
        }
        let next = self.accumulate(
            loop_.body,
            form,
            loop_.state.clone(),
            pre[..count].to_vec(),
            "step",
        )?;
        let post_args = next[..ns].iter().chain(&pre[count..]).cloned().collect();
        let post = self.invoke_body(loop_.body, &form.post, post_args, "post")?;
        self.write_values(loop_.body, &arrays, loop_.index.clone(), post)?;
        self.finish_loop(&loop_, loop_.body, next);
        self.write_values(loop_.done, &reductions, Value::Int(0), loop_.state[ns..].to_vec())?;
        Ok((loop_.done, result(&reductions, &arrays)))
    }

    fn screma_outputs(&self, op: OperationId, form: &ScremaForm, local: bool) -> (Vec<Value>, Vec<Value>) {
        let reductions = form
            .reductions
            .iter()
            .flat_map(|r| r.neutral.iter())
            .enumerate()
            .map(|(i, _)| self.slot(op, "total", i, local))
            .collect();
        let arrays = body_signature(&form.post)
            .1
            .into_iter()
            .enumerate()
            .map(|(i, _)| self.slot(op, "output", i, local))
            .collect();
        (reductions, arrays)
    }

    fn accumulate(
        &mut self,
        block: BlockId,
        form: &ScremaForm,
        accumulators: Vec<Value>,
        incoming: Vec<Value>,
        prefix: &str,
    ) -> Result<Vec<Value>, OptimizeError> {
        let mut position = 0;
        let mut result = vec![];
        for (operator, count) in form
            .scans
            .iter()
            .map(|s| (&s.operator, s.neutral.len()))
            .chain(form.reductions.iter().map(|r| (&r.operator, r.neutral.len())))
        {
            let end = position + count;
            if end > accumulators.len() || end > incoming.len() {
                return Err(error("collective accumulator arity mismatch"));
            }
            let args =
                accumulators[position..end].iter().chain(&incoming[position..end]).cloned().collect();
            let values = self.invoke_body(block, operator, args, &format!("{prefix}{position}"))?;
            if values.len() != count {
                return Err(error("collective operator result arity mismatch"));
            }
            result.extend(values);
            position = end;
        }
        Ok(result)
    }

    fn write_values(
        &mut self,
        block: BlockId,
        buffers: &[Value],
        index: Value,
        values: Vec<Value>,
    ) -> Result<(), OptimizeError> {
        if buffers.len() != values.len() {
            return Err(error("buffer result arity mismatch"));
        }
        for (buffer, value) in buffers.iter().zip(values) {
            self.store(block, buffer, index.clone(), value);
        }
        Ok(())
    }

    fn read_buffers(&mut self, block: BlockId, buffers: &[Value], index: Value) -> Vec<Value> {
        buffers.iter().map(|buffer| self.load(block, buffer.clone(), index.clone(), "read")).collect()
    }
}

fn result(reductions: &[Value], arrays: &[Value]) -> Value {
    Value::Tuple(reductions.iter().cloned().map(singleton).chain(arrays.iter().cloned()).collect())
}

fn neutrals(form: &ScremaForm) -> Vec<ExprId> {
    form.scans
        .iter()
        .flat_map(|s| s.neutral.iter())
        .chain(form.reductions.iter().flat_map(|r| r.neutral.iter()))
        .copied()
        .collect()
}
