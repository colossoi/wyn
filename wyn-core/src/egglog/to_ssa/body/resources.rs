//! Final storage views and launch extents decoded from the selected plan.
use super::{error, Body, OptimizeError, Typed, Value};
use crate::egglog::to_ssa::interface::{storage_type, view_type};
use crate::egglog::to_ssa::sizes;
use crate::host::SizeExpr;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::tlc::ArrayExpr;
use crate::tlc::SoacOp;
use crate::tlc::TermKind;
use crate::types::{self, buffer_tag};

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn source_length(
        &mut self,
        scope: Value,
        source: Value,
    ) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.values.get(&source).cloned() {
            return self.length(value);
        }
        if let Some(actual) = self.compiler.facts.alias(source) {
            return self.source_length(scope, actual);
        }
        if let Some((_, start, end)) = self.compiler.facts.slice(source) {
            let a = self.value(scope, start)?;
            let b = self.value(scope, end)?;
            return self.binary(BinaryOperator::Subtract, b, a);
        }
        if let Some(extent) = self.compiler.plan.view_length(source) {
            return self.extent(scope, extent);
        }
        if let Some(part) = self.compiler.facts.array_part(source) {
            return self.source_length(scope, part);
        }
        if let Some(resource) = self.compiler.plan.value_ref(source) {
            if let Some(extent) = self.compiler.plan.live_length(resource) {
                return self.extent(scope, extent);
            }
        }
        if let Some(&(array, owner)) = self.compiler.program.identities.arrays.get(&source) {
            match array {
                ArrayExpr::Literal(values) => {
                    return self.literal(&values.len().to_string(), &types::i32())
                }
                ArrayExpr::Range { len, .. } => return self.source(owner, len),
                _ => {}
            }
        }
        if let Some((parent, _)) = self.compiler.facts.projection(source) {
            if self.compiler.facts.operation(parent).is_some() {
                return self.source_length(scope, parent);
            }
        }
        if let Some(operation) = self.compiler.facts.operation(source) {
            if let Some(plan) = self.compiler.plan.group(operation) {
                let Some(owner) = self.compiler.plan.owner(plan) else {
                    return Err(error("missing owner"));
                };
                if let Some(resource) = self.compiler.plan.slot(owner, "length", 0) {
                    let view = self.resource(scope, resource, 1)?;
                    let zero = self.literal("0", &types::i32())?;
                    return self.index(view, zero);
                }
            }
            if let Some(&(term, owner)) = self.compiler.program.identities.origins.get(&source) {
                match &term.kind {
                    TermKind::Soac(SoacOp::Map { .. } | SoacOp::Scan { .. }) => {
                        let Some(input) = self.compiler.facts.input(operation, 0) else {
                            return Err(error("array domain missing"));
                        };
                        return self.source_length(owner, input);
                    }
                    _ => {}
                }
            }
        }
        let value = self.value(scope, source)?;
        self.length(value)
    }

    pub(in crate::egglog::to_ssa) fn resource(
        &mut self,
        scope: Value,
        resource: Value,
        access: i64,
    ) -> Result<Typed, OptimizeError> {
        let backing = self.compiler.plan.backing(resource).unwrap_or(resource);
        *self.resource_uses.entry(backing).or_default() |= access;
        if !self.compiler.plan.buffers.contains_key(&backing) {
            if let Some(source) = self.compiler.plan.external(backing) {
                return self.value(scope, source);
            }
        }
        let Some(buffer) = self.compiler.plan.buffers.get(&backing).cloned() else {
            return Err(error("selected resource has no allocation"));
        };
        let extent = if access == 2 {
            buffer.extent
        } else {
            self.compiler.plan.live_length(resource).unwrap_or(buffer.extent)
        };
        let length = self.extent(scope, extent)?;
        let zero = self.literal("0", &types::i32())?;
        let element = storage_type(&buffer.element)?;
        // Static capacity refines the view type. Failure to prove one leaves a dynamic view.
        let constant = if self.compiler.plan.extent(extent).is_some_and(|(name, _)| name != "Stored") {
            sizes::extent(self.compiler, extent).ok()
        } else {
            None
        };
        let ty = if let Some(SizeExpr::Integer(n)) = constant {
            types::view_array_with_size(
                &element,
                types::Type::Constructed(types::TypeName::Size(n.max(0) as usize), vec![]),
                buffer_tag(buffer.binding),
            )
        } else {
            view_type(&element, buffer_tag(buffer.binding))
        };
        self.op(
            OpTag::StorageView(PureViewSource::Storage(buffer.binding)),
            vec![zero, length],
            ty,
        )
    }

    pub(in crate::egglog::to_ssa) fn slot(
        &mut self,
        scope: Value,
        operation: Value,
        role: &str,
        index: i64,
        access: i64,
    ) -> Result<Option<Typed>, OptimizeError> {
        let Some(resource) = self.compiler.plan.slot(operation, role, index) else {
            return Err(error(format!("missing {role} slot {index}")));
        };
        if !self.compiler.plan.backing(resource).is_some() {
            return Ok(None);
        }
        self.resource(scope, resource, access).map(Some)
    }

    pub(in crate::egglog::to_ssa) fn extent(
        &mut self,
        scope: Value,
        extent: Value,
    ) -> Result<Typed, OptimizeError> {
        let Some((name, children)) = self.compiler.plan.extent(extent) else {
            return Err(error("unknown extent"));
        };
        match name {
            "Fixed" => {
                let n = self.compiler.program.graph.value_to_base::<i64>(children[0]);
                self.literal(&n.to_string(), &types::i32())
            }
            "Length" | "Scalar" => {
                let Some(source) = self.compiler.plan.expr(children[0]) else {
                    return Err(error("extent source is missing"));
                };
                if name == "Length" {
                    self.source_length(scope, source)
                } else {
                    self.value(scope, source)
                }
            }
            "Stored" => {
                let view = self.resource(scope, children[0], 1)?;
                let zero = self.literal("0", &types::i32())?;
                self.index(view, zero)
            }
            "ChunkCount" => {
                let value = self.extent(scope, children[0])?;
                let width = self.compiler.program.graph.value_to_base::<i64>(children[1]);
                let extra = self.literal(&(width - 1).to_string(), &types::i32())?;
                let sum = self.binary(BinaryOperator::Add, value, extra)?;
                let width = self.literal(&width.to_string(), &types::i32())?;
                self.binary(BinaryOperator::Divide, sum, width)
            }
            "Product" => {
                let a = self.extent(scope, children[0])?;
                let b = self.extent(scope, children[1])?;
                self.binary(BinaryOperator::Multiply, a, b)
            }
            _ => Err(error("invalid extent constructor")),
        }
    }
}
