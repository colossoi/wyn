use super::{
    output_error, types, Clause, Code, Import, Imported, Leaf, Name, OptimizeError, Scope, Type, TypeExt,
    TypeName,
};
use crate::egglog::graph::Formula;
use egglog_engine::{Value, Write};

impl<'source> Import<'_, '_, '_, '_, 'source> {
    pub(super) fn array_layout(
        &mut self,
        source: Leaf,
        ty: &Type,
        transpose: bool,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        let Some(row) = source.ty.elem_type() else {
            return Err(output_error(
                "array layout transform requires a rectangular array",
            ));
        };
        let Some(element) = row.elem_type() else {
            return Err(output_error("array layout transform has no element type"));
        };
        let columns = match row.array_size() {
            Some(Type::Constructed(TypeName::Size(columns), _)) => Some(*columns),
            _ => None,
        };
        let width = if let Some(columns) = columns {
            let columns = i64::try_from(columns).map_err(|_| output_error("column count exceeds i64"))?;
            self.node("IndexConstant", columns)?
        } else {
            self.layout_extent(&source, 1, scope)?
        };
        let height = self.layout_extent(&source, 0, scope)?;
        let mut inner = self.scope()?;
        let (domain, coordinates, layout) = if transpose {
            let (domain, axes) = self.domain(&[width, height])?;
            (domain, [axes[1].name, axes[0].name], "TransposeAxes")
        } else {
            let size = self.node("IndexMul", (height, width))?;
            let (domain, axes) = self.domain(&[size])?;
            let index = self.node("IndexValue", axes[0].name)?;
            let one = self.node("IndexConstant", 1_i64)?;
            let divisor = self.node("IndexMax", (width, one))?;
            let mut coordinates = Vec::new();
            for code in ["IndexDiv", "IndexMod"] {
                let expression = self.node(code, (index, divisor))?;
                let operation = self.operation(Code::IndexValue, &[])?;
                self.sink.set_formula(operation.0, Formula::Index, expression)?;
                coordinates.push(Self::leaf(self.emit(operation, &types::i32(), &mut inner)?)?.name);
            }
            (domain, [coordinates[0], coordinates[1]], "FlattenAxes")
        };
        let read = self.operation(
            Code::Index,
            &[
                Name::Value(source.name),
                Name::Value(coordinates[0]),
                Name::Value(coordinates[1]),
            ],
        )?;
        self.sink.add("ProvenTotalOperation", read.0)?;
        let value = Self::leaf(self.emit(read, element, &mut inner)?)?;
        let operation = self.parallel_operation(
            domain,
            inner,
            vec![Clause::Collect(vec![Name::Value(value.name)])],
        )?;
        let layout = self.unit(layout)?;
        self.sink.set_formula(operation.0, Formula::Layout, layout)?;
        self.emit(operation, ty, scope)
    }

    fn layout_extent(
        &mut self,
        array: &Leaf,
        axis: usize,
        scope: &mut Scope,
    ) -> Result<Value, OptimizeError> {
        let operation = self.operation(Code::Length { axis }, &[Name::Value(array.name)])?;
        let length = Self::leaf(self.emit(operation, &types::i32(), scope)?)?;
        self.node("IndexValue", length.name)
    }
}
