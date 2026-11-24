use super::{
    output_error, types, Callback, Clause, Code, Import, Imported, Leaf, Name, OperationId, OptimizeError,
    Scope, Term, TermKind, Type, TypeExt, TypeName, VarRef,
};
use crate::builtins::catalog;
use crate::egglog::graph::Formula;
use crate::tlc::{extract_lambda_params_ref, stage::OwnershipApplied};
use crate::{LookupSet, SymbolId};
use egglog_engine::{Core, Value, Write};

#[derive(Clone, Copy)]
struct Axis {
    input: usize,
    extent: Value,
}

#[derive(Clone)]
pub(super) struct Layout {
    base: Leaf,
    groups: Vec<Vec<Axis>>,
    operation: Option<OperationId>,
}

pub(super) fn transpose_functions(source: &OwnershipApplied) -> LookupSet<SymbolId> {
    source
        .defs
        .iter()
        .filter_map(|def| {
            let (body, parameters) = extract_lambda_params_ref(&def.body);
            let [(parameter, _)] = parameters.as_slice() else {
                return None;
            };
            transposes_parameter(body, *parameter, &LookupSet::new()).then_some(def.name)
        })
        .collect()
}

pub(super) fn transposes_input(callback: &Callback, transpose_functions: &LookupSet<SymbolId>) -> bool {
    let [(parameter, _)] = callback.lam.params.as_slice() else {
        return false;
    };
    if !callback.data.captures.is_empty() {
        return false;
    }
    let body = &callback.lam.body;
    match &body.kind {
        TermKind::Var(VarRef::Symbol(name)) => transpose_functions.contains(name),
        _ => transposes_parameter(body, *parameter, transpose_functions),
    }
}

fn transposes_parameter(
    mut body: &Term,
    parameter: SymbolId,
    transpose_functions: &LookupSet<SymbolId>,
) -> bool {
    if let TermKind::Let {
        name,
        rhs,
        body: tail,
        ..
    } = &body.kind
    {
        if matches!(&tail.kind, TermKind::Var(VarRef::Symbol(result)) if result == name) {
            body = rhs;
        }
    }
    let TermKind::App { func, args } = &body.kind else {
        return false;
    };
    let is_transpose = match &func.kind {
        TermKind::Var(VarRef::Builtin { id, .. }) => *id == catalog().known().transpose,
        TermKind::Var(VarRef::Symbol(name)) => transpose_functions.contains(name),
        _ => false,
    };
    is_transpose
        && matches!(args.as_slice(), [arg] if matches!(&arg.kind, TermKind::Var(VarRef::Symbol(name)) if *name == parameter))
}

impl<'source> Import<'_, '_, '_, '_, 'source> {
    pub(super) fn array_layout(
        &mut self,
        source: Leaf,
        ty: &Type,
        transpose_axis: Option<usize>,
        scope: &mut Scope,
    ) -> Result<Imported<'source>, OptimizeError> {
        // Adjacent transforms can share their source without crossing a mutation.
        let mut layout = self
            .layouts
            .get(&source.name)
            .filter(|layout| layout.operation.as_ref() == scope.statements.last())
            .cloned()
            .unwrap_or(Layout {
                base: source,
                groups: Vec::new(),
                operation: None,
            });
        let group_count = transpose_axis.map_or(2, |axis| axis + 2);
        let mut input = layout.groups.iter().map(Vec::len).sum::<usize>();
        let base = layout.base.clone();
        let mut element = &base.ty;
        for _ in 0..input {
            let Some(child) = element.elem_type() else {
                return Err(output_error("layout axes exceed the source rank"));
            };
            element = child;
        }
        while layout.groups.len() < group_count {
            let extent = self.layout_extent(&base, element, input, scope)?;
            layout.groups.push(vec![Axis { input, extent }]);
            let Some(child) = element.elem_type() else {
                return Err(output_error(
                    "array layout transform requires a rectangular array",
                ));
            };
            element = child;
            input += 1;
        }

        if let Some(axis) = transpose_axis {
            layout.groups.swap(axis, axis + 1);
        } else {
            let inner = layout.groups.remove(1);
            layout.groups[0].extend(inner);
        }
        if base.ty == *ty
            && layout
                .groups
                .iter()
                .enumerate()
                .all(|(axis, group)| matches!(group.as_slice(), [dimension] if dimension.input == axis))
        {
            return Ok(Imported::Leaf(base));
        }
        let mut extents = Vec::new();
        for group in &layout.groups {
            let Some((first, rest)) = group.split_first() else {
                return Err(output_error("layout axis group is empty"));
            };
            let mut extent = first.extent;
            for axis in rest {
                extent = self.node("IndexMul", (extent, axis.extent))?;
            }
            extents.push(extent);
        }
        let (domain, axes) = self.domain(&extents)?;
        let mut inner = self.scope()?;
        let mut indices = vec![Name::Value(base.name); 1 + input];
        for (group, axis) in layout.groups.iter().zip(axes) {
            let Some((first, rest)) = group.split_first() else {
                return Err(output_error("layout axis group is empty"));
            };
            if rest.is_empty() {
                indices[1 + first.input] = Name::Value(axis.name);
                continue;
            }
            let one = self.node("IndexConstant", 1_i64)?;
            let mut index = self.node("IndexValue", axis.name)?;
            for axis in rest.iter().rev() {
                let divisor = self.node("IndexMax", (axis.extent, one))?;
                let offset = self.node("IndexMod", (index, divisor))?;
                indices[1 + axis.input] = self.layout_index(offset, &mut inner)?;
                index = self.node("IndexDiv", (index, divisor))?;
            }
            indices[1 + first.input] = self.layout_index(index, &mut inner)?;
        }

        let read = self.operation(Code::Index, &indices)?;
        self.sink.add("ProvenTotalOperation", read.0)?;
        let value = Self::leaf(self.emit(read, element, &mut inner)?)?;
        let operation = self.parallel_operation(
            domain,
            inner,
            vec![Clause::Collect(vec![Name::Value(value.name)])],
        )?;
        let rank = input as i64;
        let order =
            layout.groups.iter().flatten().map(|axis| self.sink.base_to_value(axis.input as i64)).collect();
        let order = self.vector(order);
        let transform = self.node("AxisLayout", (rank, order))?;
        self.sink.set_formula(operation.0, Formula::Layout, transform)?;
        let result = Self::leaf(self.emit(operation, ty, scope)?)?;
        layout.operation = Some(operation);
        self.layouts.insert(result.name, layout);
        Ok(Imported::Leaf(result))
    }

    fn layout_index(&mut self, index: Value, scope: &mut Scope) -> Result<Name, OptimizeError> {
        let operation = self.operation(Code::IndexValue, &[])?;
        self.sink.set_formula(operation.0, Formula::Index, index)?;
        Ok(Name::Value(
            Self::leaf(self.emit(operation, &types::i32(), scope)?)?.name,
        ))
    }

    fn layout_extent(
        &mut self,
        array: &Leaf,
        ty: &Type,
        axis: usize,
        scope: &mut Scope,
    ) -> Result<Value, OptimizeError> {
        if let Some(Type::Constructed(TypeName::Size(size), _)) = ty.array_size() {
            let size = i64::try_from(*size).map_err(|_| output_error("layout extent exceeds i64"))?;
            return self.node("IndexConstant", size);
        }
        let operation = self.operation(Code::Length { axis }, &[Name::Value(array.name)])?;
        let length = Self::leaf(self.emit(operation, &types::i32(), scope)?)?;
        self.node("IndexValue", length.name)
    }
}
