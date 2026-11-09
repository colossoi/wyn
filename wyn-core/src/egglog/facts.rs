//! Borrowed queries into egglog; this view owns no source or planning facts.
use super::{output_error as error, query::Query, OptimizeError, Optimized, Program};
use crate::types::{self, Type};
use crate::BindingRef;
use crate::SymbolId;
use egglog_engine::sort::{SetContainer, VecContainer};
use egglog_engine::{IntoValues, Value};

mod plan;
pub(super) use plan::OutputAction;

pub(super) struct Facts<'a, 'source> {
    pub program: &'a Program<'source, Optimized>,
}

pub(super) struct LoopCarry {
    pub slot: i64,
    pub initial: Value,
    pub state: Value,
    pub next: Value,
    pub array: bool,
}
impl<'a, 'source> Facts<'a, 'source> {
    // Access declarations are loaded with this compiler module. Wrong names or
    // arities here are compiler programming errors, never missing user facts.
    pub fn lookup(&self, table: &str, keys: impl IntoValues) -> Option<Value> {
        match Query(&self.program.graph).lookup(table, keys) {
            Ok(value) => value,
            Err(error) => panic!("invalid native fact accessor {table}: {error}"),
        }
    }
    pub fn contains(&self, table: &str, keys: impl IntoValues) -> bool {
        match Query(&self.program.graph).contains(table, keys) {
            Ok(value) => value,
            Err(error) => panic!("invalid native fact predicate {table}: {error}"),
        }
    }
    pub fn constructor(&self, table: &str, keys: impl IntoValues) -> Option<Value> {
        match Query(&self.program.graph).constructor(table, keys) {
            Ok(value) => value,
            Err(error) => panic!("invalid native constructor accessor {table}: {error}"),
        }
    }
    pub fn enode(&self, name: &str, value: Value) -> Option<Vec<Value>> {
        Query(&self.program.graph)
            .enode(name, value)
            .unwrap_or_else(|error| panic!("invalid native enode accessor {name}: {error}"))
    }
    pub fn record(&self, table: &str, constructor: &str, keys: impl IntoValues) -> Option<Vec<Value>> {
        Query(&self.program.graph)
            .record(table, constructor, keys)
            .unwrap_or_else(|error| panic!("invalid native record accessor {table}: {error}"))
    }
    pub fn branch_condition(&self, value: Value) -> Option<Value> {
        Some(self.record("SourceBranch", "BranchInfo", (value,))?[0])
    }
    pub fn integer(&self, value: Value) -> i64 {
        self.program.graph.value_to_base::<i64>(value)
    }
    pub fn unsigned(&self, value: Value, field: &str) -> Result<u32, OptimizeError> {
        u32::try_from(self.integer(value))
            .map_err(|_| error(format!("{field} must fit an unsigned 32-bit integer")))
    }
    pub fn positive(&self, value: Value, field: &str) -> Result<u32, OptimizeError> {
        let n = self.unsigned(value, field)?;
        if n == 0 {
            return Err(error(format!("{field} must be positive")));
        }
        Ok(n)
    }
    pub fn grid(&self, value: Value) -> Result<(u32, u32, u32), OptimizeError> {
        let Some(fields) = self.enode("FixedGrid", value) else {
            return Err(error("selected dimensions are not a fixed grid"));
        };
        Ok((
            self.positive(fields[0], "grid x")?,
            self.positive(fields[1], "grid y")?,
            self.positive(fields[2], "grid z")?,
        ))
    }
    pub fn set(&self, table: &str, keys: impl IntoValues) -> Vec<Value> {
        // These tables are relational collections: no row means no members.
        let Some(value) = self.lookup(table, keys) else {
            return Vec::new();
        };
        let Some(set) = self.program.graph.value_to_container::<SetContainer>(value) else {
            panic!("invalid native set container in {table}: {value:?}");
        };
        set.data.iter().copied().collect()
    }
    pub fn alias(&self, value: Value) -> Option<Value> {
        self.lookup("SourceAlias", (value,))
    }
    pub fn context(&self, value: Value) -> Option<Value> {
        self.lookup("ExprContextForScope", (value,))
    }
    pub fn result(&self, value: Value) -> Option<Value> {
        Some(self.record("RegionDefinition", "RegionBody", (value,))?[2])
    }
    pub fn loop_state(&self, value: Value) -> Option<Value> {
        Some(self.record("LoopHeader", "LoopInfo", (value,))?[3])
    }
    pub fn loop_carries(&self, operation: Value) -> Result<Vec<LoopCarry>, OptimizeError> {
        let mut rows = Vec::new();
        Query(&self.program.graph).for_each("LoopCarry", |row| {
            if row[0] == operation {
                rows.push(LoopCarry {
                    slot: self.integer(row[1]),
                    initial: row[2],
                    state: row[3],
                    next: row[4],
                    array: self.program.graph.value_to_base::<bool>(row[6]),
                });
            }
            Ok(())
        })?;
        rows.sort_by_key(|carry| carry.slot);
        Ok(rows)
    }
    pub fn source_field(&self, parent: Value, index: usize) -> Result<Option<Value>, OptimizeError> {
        Ok(Query(&self.program.graph)
            .row("SourceField", |row| {
                row[0] == parent && self.integer(row[1]) == index as i64
            })?
            .map(|row| row[2]))
    }
    pub fn iteration(&self, value: Value) -> Option<Value> {
        self.lookup("SourceIterationValue", (value,))
    }
    pub fn operation(&self, value: Value) -> Option<Value> {
        self.lookup("SourceValueOperation", (value,))
    }
    pub fn loop_initial(&self, region: Value) -> Result<Value, OptimizeError> {
        let Some(info) = self.record("LoopHeader", "LoopInfo", (region,)) else {
            return Err(error("loop region has no initial state"));
        };
        Ok(info[4])
    }
    pub fn loop_form(&self, source: Value) -> Option<Value> {
        Some(self.record("SourceLoop", "LoopInfo", (source,))?[2])
    }
    pub fn iteration_dimensions(&self, operation: Value) -> Result<Vec<Value>, OptimizeError> {
        let query = Query(&self.program.graph);
        let shape = query.required("IterationShape", (operation,))?;
        self.vector(query.required("ShapeDimensions", (shape,))?)
    }
    pub fn domain_shape(&self, operation: Value) -> Result<(Vec<Value>, Vec<Vec<usize>>), OptimizeError> {
        let dimensions = self.iteration_dimensions(operation)?;
        let rank = dimensions.len();
        let mut inputs = Vec::new();
        for (input, _) in self.inputs(operation)? {
            let mapping = Query(&self.program.graph).required("SourceInputMap", (operation, input))?;
            let mut axes = Vec::new();
            for axis in self.vector(mapping)? {
                let axis = usize::try_from(self.integer(axis)).map_err(|_| error("invalid domain axis"))?;
                if axis >= rank {
                    return Err(error("input axis exceeds its domain rank"));
                }
                axes.push(axis);
            }
            inputs.push(axes);
        }
        Ok((dimensions, inputs))
    }
    pub fn definition_name(&self, scope: Value) -> Option<SymbolId> {
        let token = self.lookup("SourceDefinitionSymbol", (scope,))?;
        Some(*self.program.identities.symbols.resolve(self.program.graph.value_to_base::<i64>(token)))
    }
    pub fn definition(&self, symbol: SymbolId) -> Option<Value> {
        self.lookup(
            "SourceDefinition",
            (self.program.identities.symbols.get(&symbol)?,),
        )
    }
    pub fn entry_region(&self, scope: Value) -> bool {
        self.lookup("SourceDefinitionSymbol", (scope,)).is_some_and(|symbol| {
            self.contains("SourceOriginalEntry", (symbol, scope, false))
                || self.contains("SourceOriginalEntry", (symbol, scope, true))
        })
    }
    pub fn entry_root(&self, owner: SymbolId, stage: Option<Value>) -> Result<Value, OptimizeError> {
        let root = if let Some(stage) = stage {
            self.constructor("KernelRoot", (stage,))
        } else {
            let token = self
                .program
                .identities
                .symbols
                .get(&owner)
                .ok_or_else(|| error("entry identity missing"))?;
            self.constructor("EntryRoot", (token,))
        };
        root.ok_or_else(|| error("entry resource root missing"))
    }
    pub fn binding(&self, value: Value) -> Result<BindingRef, OptimizeError> {
        let fields = self
            .enode("InputBinding", value)
            .ok_or_else(|| error("selected descriptor binding missing"))?;
        Ok(BindingRef::new(
            self.unsigned(fields[0], "descriptor set")?,
            self.unsigned(fields[1], "descriptor binding")?,
        ))
    }
    pub fn storage_access(
        &self,
        flags: Value,
    ) -> Result<Option<crate::interface::StorageAccess>, OptimizeError> {
        use crate::interface::StorageAccess;
        match self.integer(flags) {
            0 => Ok(None),
            1 => Ok(Some(StorageAccess::ReadOnly)),
            2 => Ok(Some(StorageAccess::WriteOnly)),
            3 => Ok(Some(StorageAccess::ReadWrite)),
            _ => Err(error("invalid selected storage access")),
        }
    }
    pub fn input_storage(&self, source: Value) -> Result<Option<(BindingRef, u32)>, OptimizeError> {
        let Some(storage) = self.lookup("AbiStorage", (source,)) else {
            return Ok(None);
        };
        let Some(fields) = self.enode("StorageInput", storage) else {
            return Err(error("ABI storage has no input descriptor"));
        };
        Ok(Some((
            self.binding(fields[0])?,
            self.positive(fields[1], "input element stride")?,
        )))
    }
    pub fn ty(&self, value: Value) -> Option<&'a Type> {
        Some(self.program.identities.types.resolve(self.integer(self.enode("TypeId", value)?[0])))
    }
    pub fn physical_type(&self, ty: &Type, storage: bool) -> Result<Type, OptimizeError> {
        let Some(token) = self.program.identities.types.get(ty) else {
            return Err(error(format!("physical type has no semantic identity: {ty:?}")));
        };
        let Some(key) = self.constructor("TypeId", (token,)) else {
            return Err(error("physical type identity missing"));
        };
        let layout = self.layout(key)?;
        let ty = self.layout_type(layout)?;
        Ok(if storage { crate::ssa::layout::storage_value_type(&ty) } else { ty })
    }
    pub fn layout_type(&self, layout: Value) -> Result<Type, OptimizeError> {
        let facts = self;
        if let Some(fields) = facts.enode("ViewLayout", layout) {
            let Some(element) = facts.ty(fields[1]) else {
                return Err(error("view layout has no element type"));
            };
            return Ok(types::view_array_with_size(
                element,
                Type::Constructed(types::TypeName::SizePlaceholder, vec![]),
                types::no_buffer(),
            ));
        }
        if let Some(fields) = facts.enode("ValueLayout", layout) {
            let Some(ty) = facts.ty(fields[0]) else {
                return Err(error("boundary layout has no value type"));
            };
            return Ok(types::strip_existentials(ty).clone());
        }
        if let Some(fields) = facts.enode("ArrayLayout", layout) {
            let Some(capacity) = facts.lookup("PhysicalArrayCapacity", (layout,)) else {
                return Err(error("array layout has no physical capacity"));
            };
            let n =
                usize::try_from(facts.integer(capacity)).map_err(|_| error("invalid array capacity"))?;
            return Ok(types::sized_array(n, self.layout_type(fields[1])?));
        }
        if let Some(fields) = facts.enode("TupleLayout", layout) {
            let Some(Type::Constructed(name, _)) = facts.ty(fields[0]).map(types::strip_existentials)
            else {
                return Err(error("tuple layout has no aggregate type"));
            };
            let fields = facts
                .vector(fields[1])?
                .into_iter()
                .map(|field| self.layout_type(field))
                .collect::<Result<_, _>>()?;
            return Ok(Type::Constructed(name.clone(), fields));
        }
        Err(error("unknown boundary layout"))
    }

    pub fn vector(&self, value: Value) -> Result<Vec<Value>, OptimizeError> {
        let Some(values) = self.program.graph.value_to_container::<VecContainer>(value) else {
            return Err(error("expected a native fact vector"));
        };
        Ok(values.data.clone())
    }
    pub fn layout(&self, ty: Value) -> Result<Value, OptimizeError> {
        let Some(layout) = self.lookup("BoundaryLayout", (ty,)) else {
            return Err(error(format!(
                "type {:?} has no selected control-boundary layout",
                self.ty(ty)
            )));
        };
        Ok(layout)
    }
    pub fn value_layout(&self, value: Value) -> Result<Value, OptimizeError> {
        let Some(layout) = self.lookup("ValueRepresentation", (value,)) else {
            return Err(error(format!("value {value:?} has no selected boundary layout")));
        };
        Ok(layout)
    }
    pub fn source_type(&self, value: Value) -> Option<&'a Type> {
        self.ty(self.lookup("SourceType", (value,))?)
    }
    pub fn parameter(&self, scope: Value, index: i64) -> Option<Value> {
        let body = self.record("RegionDefinition", "RegionBody", (scope,))?;
        let parameters = self.program.graph.value_to_container::<VecContainer>(body[0])?;
        parameters.data.get(usize::try_from(index).ok()?).copied()
    }
    pub fn branches(&self, value: Value) -> Option<(Value, Value)> {
        let info = self.record("SourceBranch", "BranchInfo", (value,))?;
        Some((info[1], info[2]))
    }
    pub fn loops(&self, value: Value) -> Option<(Value, Value)> {
        let info = self.record("SourceLoop", "LoopInfo", (value,))?;
        Some((info[0], info[1]))
    }
    pub fn projection(&self, value: Value) -> Option<(Value, usize)> {
        let info = self.record("SourceProjection", "ProjectionInfo", (value,))?;
        Some((info[0], self.integer(info[1]) as usize))
    }
    pub fn slice(&self, value: Value) -> Option<(Value, Value, Value)> {
        let info = self.record("SourceSlice", "SliceInfo", (value,))?;
        Some((info[0], info[1], info[2]))
    }
    pub fn inputs(&self, operation: Value) -> Result<Vec<(i64, Value)>, OptimizeError> {
        let inputs = Query(&self.program.graph).required("OperatorInputs", (operation,))?;
        Ok(self.vector(inputs)?.into_iter().enumerate().map(|(i, value)| (i as i64, value)).collect())
    }
    pub fn bind_arguments<T>(
        &self,
        region: Value,
        arguments: Vec<T>,
    ) -> Result<Vec<(Value, T)>, OptimizeError> {
        let Some(body) = self.record("RegionDefinition", "RegionBody", (region,)) else {
            return Err(error("region definition missing"));
        };
        let parameters = self.vector(body[0])?;
        if parameters.len() != arguments.len() {
            return Err(error(format!(
                "region expects {} arguments, got {}",
                parameters.len(),
                arguments.len()
            )));
        }
        Ok(parameters.into_iter().zip(arguments).collect())
    }
    pub fn captures(&self, scope: Value) -> Result<Vec<(Value, Value)>, OptimizeError> {
        let Some(body) = self.record("RegionDefinition", "RegionBody", (scope,)) else {
            return Err(error("region definition missing"));
        };
        self.vector(body[1])?
            .into_iter()
            .map(|binding| {
                let Some(fields) = self.enode("CaptureBinding", binding) else {
                    return Err(error("invalid region capture"));
                };
                Ok((fields[0], fields[1]))
            })
            .collect()
    }
    pub fn dispatch_context(&self, operation: Value) -> Option<Value> {
        self.constructor("ExprContextDispatch", (operation,))
    }
}

pub(super) fn prepare(program: &mut Program<'_, Optimized>) -> Result<(), OptimizeError> {
    program.graph.parse_and_run_program(
        Some("lowering-access.egg".into()),
        include_str!("facts/access.egg"),
    )?;
    Ok(())
}
