//! Lower each logical row into its shared arena and its input-indexed descriptor.
use super::{Callback, Clause, Code, Import, Imported, Name, OptimizeError, Scope, SourceArray, Term};
use crate::egglog::graph::Formula;
use crate::egglog::{output_error, payload::Payload};
use crate::op::BinaryOperator;
use crate::types::{self, strip_existentials, Type, TypeName};
use egglog_engine::Write;

impl<'source> Import<'_, '_, '_, '_, 'source> {
    pub(super) fn segmented_flatmap(
        &mut self,
        dest: &'source Term,
        callback: &'source Callback,
        sources: &'source [SourceArray],
        ty: &Type,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let [source] = sources else {
            return Err(output_error("flatmap needs one input array"));
        };
        let arena = Self::leaf(self.visit(dest, scope)?)?;
        let arena = self.scatter_destination(arena, scope)?;
        let Type::Constructed(TypeName::Tuple(2), result_types) = strip_existentials(ty) else {
            return Err(output_error("flatmap must return data and row descriptors"));
        };
        let source = Self::leaf(self.array(source, scope)?)?;
        let length = self.operation(Code::Length { axis: 0 }, &[Name::Value(source.name)])?;
        let lanes = Self::leaf(self.emit(length, &types::i32(), scope)?)?;
        let extent = self.node("IndexValue", lanes.name)?;
        let (domain, axes) = self.domain(&[extent])?;
        let axis = &axes[0];
        let shape = self.vector(vec![extent]);
        let scratch = self.operation(Code::Scratch, &[])?;
        self.sink.set_formula(scratch.0, Formula::Extents, shape)?;
        let records = Self::leaf(self.emit(scratch, &result_types[1], scope)?)?;
        let length = self.operation(Code::Length { axis: 0 }, &[Name::Value(arena.name)])?;
        let capacity = Self::leaf(self.emit(length, &types::i32(), scope)?)?;
        let cursor = self.operation(Code::ReservationCursor, &[])?;
        let cursor_type = types::sized_array(1, Type::Constructed(TypeName::UInt(32), vec![]));
        let cursor = Self::leaf(self.emit(cursor, &cursor_type, scope)?)?;

        let policy = self.unit("DisjointWrites")?;
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        let (data_token, data_info) = self.accumulator(&arena, policy, None, &mut inputs, &mut params)?;
        let (meta_token, meta_info) = self.accumulator(&records, policy, None, &mut inputs, &mut params)?;
        let mut lane = self.scope()?;
        let element = self.traversal_input(source.name, axis.name)?;
        let element = self.emit(element, &Self::row_type(&source.ty)?, &mut lane)?;
        let item = Self::leaf(self.apply_callback(callback, vec![element], &mut lane)?)?;
        let Type::Constructed(TypeName::Tuple(2), fields) = strip_existentials(&item.ty) else {
            return Err(output_error(
                "flatmap callback must produce a row and a side result",
            ));
        };
        let row = self.project(&item, 0, &fields[0], &mut lane)?;
        let length = self.operation(Code::Length { axis: 0 }, &[Name::Value(row.name)])?;
        let count = Self::leaf(self.emit(length, &types::i32(), &mut lane)?)?;
        let side = self.project(&item, 1, &fields[1], &mut lane)?;
        // The destination span is supplied once to the writer. Its size depends only on
        // the row length, so the element producer can fuse with the traversal below.
        let mut writer = self.scope()?;
        let start = self.scalar(&types::i32())?;
        writer.parameters.push(super::Parameter {
            name: self.binding(&start),
            info: start.info,
            diet: self.unit("Observe")?,
        });
        let zero = Self::leaf(self.literal(&types::i32(), 0, &mut lane)?)?;
        let extent = self.node("IndexValue", count.name)?;
        let (row_domain, row_axes) = self.domain(&[extent])?;
        let j = &row_axes[0];
        let mut inner = self.scope()?;
        let valid = Self::leaf(self.primitive(
            Payload::Binary(BinaryOperator::GreaterEqual),
            vec![start.name, zero.name],
            &types::bool_type(),
            true,
            &mut inner,
        )?)?;
        let read = self.traversal_input(row.name, j.name)?;
        let value = Self::leaf(self.emit(read, &Self::row_type(&row.ty)?, &mut inner)?)?;
        let index = Self::leaf(self.primitive(
            Payload::Binary(BinaryOperator::Add),
            vec![start.name, j.name],
            &types::i32(),
            true,
            &mut inner,
        )?)?;
        let invalid = Self::leaf(self.literal(&types::i32(), -1, &mut inner)?)?;
        let index = Self::leaf(self.primitive(
            Payload::Select,
            vec![invalid.name, index.name, valid.name],
            &types::i32(),
            true,
            &mut inner,
        )?)?;
        let (next, _) = self.accumulate(data_token, data_info, index.name, value.name, &mut inner)?;
        let write = self.parallel_operation(
            row_domain,
            inner,
            vec![Clause::Contribute(vec![Name::Accumulator(next)])],
        )?;
        let stored = self.fresh("AccumulatorId")?;
        self.sink.set("AccumulatorType", stored, data_info)?;
        self.statement(vec![Name::Accumulator(stored)], write, &mut writer)?;
        let token_result = self.node("ReturnAccumulator", data_info)?;
        let index_result = self.return_info(&types::i32())?;
        let writer = self.finish_region(
            writer,
            &[Name::Accumulator(stored), self.binding(&start)],
            &[token_result, index_result],
        )?;
        let reserve = self.operation(
            Code::WithReservedSpan,
            &[
                Name::Accumulator(data_token),
                Name::Value(cursor.name),
                Name::Value(capacity.name),
                Name::Value(source.name),
                Name::Value(count.name),
            ],
        )?;
        self.sink.set_control(reserve.0, 0, writer.0);
        let stored = self.fresh("AccumulatorId")?;
        self.sink.set("AccumulatorType", stored, data_info)?;
        let start = self.scalar(&types::i32())?;
        self.statement(
            vec![Name::Accumulator(stored), self.binding(&start)],
            reserve,
            &mut lane,
        )?;
        let valid = Self::leaf(self.primitive(
            Payload::Binary(BinaryOperator::GreaterEqual),
            vec![start.name, zero.name],
            &types::bool_type(),
            true,
            &mut lane,
        )?)?;
        let used = Self::leaf(self.primitive(
            Payload::Select,
            vec![zero.name, count.name, valid.name],
            &types::i32(),
            true,
            &mut lane,
        )?)?;
        let descriptor =
            self.operation(Code::Product, &[Name::Value(start.name), Name::Value(used.name)])?;
        let handle = types::tuple(vec![types::i32(), types::i32()]);
        let descriptor = Self::leaf(self.emit(descriptor, &handle, &mut lane)?)?;
        let record = self.operation(
            Code::Product,
            &[Name::Value(descriptor.name), Name::Value(side.name)],
        )?;
        let record = Self::leaf(self.emit(record, &Self::row_type(&records.ty)?, &mut lane)?)?;
        let meta = self.accumulate(meta_token, meta_info, axis.name, record.name, &mut lane)?;
        self.contribute(
            vec![(stored, data_info), meta],
            domain,
            lane,
            inputs,
            params,
            ty,
            scope,
        )
    }
}
