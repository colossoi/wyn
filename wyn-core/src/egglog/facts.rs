//! Borrowed queries into egglog; this view owns no source or planning facts.
use super::{output_error as error, query::Query, OptimizeError, Optimized, Program};
use crate::types::{self, Type};
use crate::BindingRef;
use crate::SymbolId;
use egglog_engine::sort::{SetContainer, VecContainer, S};
use egglog_engine::{IntoValues, Value};

mod plan;

pub(super) struct Facts<'a, 'source> {
    pub program: &'a Program<'source, Optimized>,
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
        self.lookup("ScalarScopeContext", (value,))
    }
    pub fn result(&self, value: Value) -> Option<Value> {
        self.lookup("SourceResult", (value,))
    }
    pub fn loop_state(&self, value: Value) -> Option<Value> {
        self.lookup("SsaLoopState", (value,))
    }
    pub fn iteration(&self, value: Value) -> Option<Value> {
        self.lookup("SourceIterationValue", (value,))
    }
    pub fn destination(&self, value: Value) -> Option<Value> {
        self.lookup("SourceDestination", (value,))
    }
    pub fn operation(&self, value: Value) -> Option<Value> {
        self.lookup("SourceValueOperation", (value,))
    }
    pub fn operation_kind(&self, operation: Value) -> Result<String, OptimizeError> {
        let Some(kind) = self.lookup("SsaOperationKind", (operation,)) else {
            return Err(error("operation has no source kind"));
        };
        Ok(self.program.graph.value_to_base::<S>(kind).to_string())
    }
    pub fn operation_scope(&self, operation: Value) -> Result<Value, OptimizeError> {
        let Some(scope) = self.lookup("SsaOperationScope", (operation,)) else {
            return Err(error("operation has no source region"));
        };
        Ok(scope)
    }
    pub fn neutral(&self, operation: Value) -> Result<Value, OptimizeError> {
        let Some(value) = self.lookup("SourceNeutral", (operation,)) else {
            return Err(error("accumulator has no neutral value"));
        };
        Ok(value)
    }
    pub fn loop_initial(&self, region: Value) -> Result<Value, OptimizeError> {
        let Some(initial) = self.lookup("SsaLoopInitial", (region,)) else {
            return Err(error("loop region has no initial state"));
        };
        Ok(initial)
    }
    pub fn bucket_shape(&self, operation: Value) -> Result<(usize, Vec<Vec<usize>>), OptimizeError> {
        let Some(rank) = self.lookup("SourceBucketRank", (operation,)) else {
            return Err(error("bucket operation has no domain rank"));
        };
        let rank = usize::try_from(self.integer(rank)).map_err(|_| error("invalid bucket domain rank"))?;
        let mut inputs = Vec::new();
        for (input, _) in self.inputs(operation)? {
            let Some(dimensions) = self.lookup("SourceBucketInputRank", (operation, input)) else {
                return Err(error("bucket input has no dimension mapping"));
            };
            let mut axes = Vec::new();
            for dimension in 0..self.integer(dimensions) {
                let Some(axis) = self.lookup("SourceBucketInputDimension", (operation, input, dimension))
                else {
                    return Err(error("bucket input dimension is missing"));
                };
                let axis = usize::try_from(self.integer(axis)).map_err(|_| error("invalid bucket axis"))?;
                if axis >= rank {
                    return Err(error("bucket input axis exceeds its domain rank"));
                }
                axes.push(axis);
            }
            inputs.push(axes);
        }
        Ok((rank, inputs))
    }
    pub fn callback(&self, value: Value) -> Option<Value> {
        self.lookup("SourceOperatorBody", (value,))
    }
    pub fn definition_name(&self, scope: Value) -> Option<SymbolId> {
        let token = self.lookup("SsaDefinitionName", (scope,))?;
        Some(*self.program.identities.symbols.resolve(self.program.graph.value_to_base::<i64>(token)))
    }
    pub fn definition(&self, symbol: SymbolId) -> Option<Value> {
        self.lookup("SsaDefinition", (self.program.identities.symbols.get(&symbol)?,))
    }
    pub fn entry_region(&self, scope: Value) -> bool {
        self.lookup("SsaDefinitionName", (scope,))
            .is_some_and(|symbol| self.contains("SourceEntryPoint", (symbol, scope)))
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
        let Some(storage) =
            self.lookup("SourceExprKey", (source,)).and_then(|key| self.lookup("AbiStorage", (key,)))
        else {
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
        self.lookup("SourceParameter", (scope, index))
    }
    pub fn branches(&self, value: Value) -> Option<(Value, Value)> {
        Some((
            self.lookup("SsaBranchYes", (value,))?,
            self.lookup("SsaBranchNo", (value,))?,
        ))
    }
    pub fn loops(&self, value: Value) -> Option<(Value, Value)> {
        Some((
            self.lookup("SsaLoopHeader", (value,))?,
            self.lookup("SsaLoopIteration", (value,))?,
        ))
    }
    pub fn projection(&self, value: Value) -> Option<(Value, usize)> {
        Some((
            self.lookup("SsaProjectionBase", (value,))?,
            self.lookup("SsaProjectionIndex", (value,)).map(|v| self.integer(v))? as usize,
        ))
    }
    pub fn slice(&self, value: Value) -> Option<(Value, Value, Value)> {
        Some((
            self.lookup("SsaSliceBase", (value,))?,
            self.lookup("SsaSliceStart", (value,))?,
            self.lookup("SsaSliceEnd", (value,))?,
        ))
    }
    pub fn input(&self, operation: Value, index: i64) -> Option<Value> {
        self.lookup("SourceInput", (operation, index))
    }
    pub fn inputs(&self, operation: Value) -> Result<Vec<(i64, Value)>, OptimizeError> {
        let Some(count) = self.lookup("SourceInputCount", (operation,)) else {
            return Err(error("operation has no input count"));
        };
        let count = self.integer(count);
        if count < 0 {
            return Err(error("operation input count is negative"));
        }
        (0..count)
            .map(|index| {
                let Some(value) = self.input(operation, index) else {
                    return Err(error("operation input is missing"));
                };
                Ok((index, value))
            })
            .collect()
    }
    pub fn captures(&self, scope: Value) -> Result<Vec<(Value, Value)>, OptimizeError> {
        self.set("SsaCaptures", (scope,))
            .into_iter()
            .map(|formal| {
                let Some(actual) = self.lookup("SsaCaptureAt", (scope, formal)) else {
                    return Err(error("capture has no selected actual value"));
                };
                Ok((formal, actual))
            })
            .collect()
    }
    pub fn dispatch_context(&self, operation: Value) -> Option<Value> {
        self.constructor("ScalarDispatch", (operation,))
    }
}

pub(super) fn prepare(program: &mut Program<'_, Optimized>) -> Result<(), OptimizeError> {
    program.graph.parse_and_run_program(
        Some("lowering-access.egg".into()),
        include_str!("facts/access.egg"),
    )?;
    Ok(())
}
