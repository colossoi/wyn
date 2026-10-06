//! Allocate descriptor slots and retain final entry-output ABI values.
use crate::egglog::facts::Facts;
use crate::egglog::query::Query;
use crate::egglog::{output_error as error, OptimizeError, Optimized, Program};
use crate::interface::{EntryOutput, EntryParamBindingKind};
use crate::tlc::extract_lambda_params_ref;
use crate::tlc::DefMeta;
use crate::types::{strip_existentials, Type, TypeName};
use crate::{BindingRef, LookupMap, SymbolId};
use egglog_engine::Value;
use wyn_base::IdSource;

pub(in crate::egglog) struct Bindings<'a, 'source> {
    query: Facts<'a, 'source>,
    bindings: LookupMap<Value, BindingRef>,
    pub captures: LookupMap<egglog_engine::TermId, BindingRef>,
    pub entry_outputs: LookupMap<SymbolId, Vec<EntryOutput>>,
}
impl<'a, 'source> Bindings<'a, 'source> {
    pub fn new(program: &'a Program<'source, Optimized>) -> Result<Self, OptimizeError> {
        let query = Facts { program };
        let mut reserved = std::collections::BTreeSet::new();
        let mut entry_outputs = LookupMap::default();
        for definition in &program.source.defs {
            let DefMeta::EntryPoint(entry) = &definition.meta else {
                continue;
            };
            let mut output_bindings = IdSource::new();
            let mut reserve_input = |binding: BindingRef| {
                reserved.insert(binding);
                if binding.set == 0 {
                    while output_bindings.peek_id() <= binding.binding {
                        output_bindings.next_id();
                    }
                }
            };
            for parameter in &entry.declaration.params {
                for (set, binding) in parameter.attributes.iter().filter_map(|a| a.binding_slot()) {
                    reserve_input(BindingRef::new(set, binding));
                }
            }
            for parameter in entry.data.param_bindings.iter().flatten() {
                match &parameter.kind {
                    EntryParamBindingKind::Single { binding, .. } => reserve_input(*binding),
                    EntryParamBindingKind::TupleOfViews(fields) => {
                        for field in fields {
                            reserve_input(field.binding);
                        }
                    }
                }
            }
            let outputs = crate::egglog::abi::outputs(program, definition.name, &mut output_bindings)?;
            for output in &outputs {
                if let Some(binding) = output.storage_binding() {
                    reserved.insert(binding);
                }
            }
            entry_outputs.insert(definition.name, outputs);
        }
        let mut allocations = Vec::new();
        program.graph.constructor_enodes("PlannedBuffer", |row| allocations.push(row.children[0]))?;
        allocations.sort();
        allocations.dedup();
        let mut bindings = LookupMap::default();
        for &value in &allocations {
            if let Some(binding) = query.lookup("PhysicalBinding", (value,)) {
                let binding = query.binding(binding)?;
                reserved.insert(binding);
                bindings.insert(value, binding);
            }
        }
        // TODO: Consider assigning generated buffer and capture bindings in
        // Egglog; this ABI choice might fit better alongside PhysicalBinding.
        let mut next = 0;
        for value in allocations {
            if bindings.contains_key(&value) {
                continue;
            }
            while reserved.contains(&BindingRef::new(0, next)) {
                next += 1;
            }
            let binding = BindingRef::new(0, next);
            reserved.insert(binding);
            bindings.insert(value, binding);
        }
        let mut captures = LookupMap::default();
        for &term in program.stage.captures.keys() {
            while reserved.contains(&BindingRef::new(0, next)) {
                next += 1;
            }
            let binding = BindingRef::new(0, next);
            reserved.insert(binding);
            captures.insert(term, binding);
        }
        Ok(Self {
            query,
            bindings,
            captures,
            entry_outputs,
        })
    }

    pub fn buffer(&self, key: Value) -> Result<Option<(BindingRef, &'a Type, Value)>, OptimizeError> {
        let query = Query(&self.query.program.graph);
        let Some(allocation) = query.row("PlannedBuffer", |r| r[0] == key)? else {
            return Ok(None);
        };
        let Some(&binding) = self.bindings.get(&key) else {
            return Err(error("allocation has no physical ABI"));
        };
        Ok(Some((
            binding,
            self.query.program.identities.types.resolve(self.query.integer(allocation[1])),
            allocation[2],
        )))
    }

    pub fn buffer_name(&self, key: Value) -> Result<String, OptimizeError> {
        let query = Query(&self.query.program.graph);
        let owner = query.required("AllocationOwner", (key,))?;
        let symbol = self.query.program.identities.symbols.resolve(self.query.integer(owner));
        let Some(owner) = self.query.program.source.symbols.get(*symbol) else {
            return Err(error("allocation owner name missing"));
        };
        let Some((binding, _, _)) = self.buffer(key)? else {
            return Err(error("allocation missing"));
        };
        if query.lookup("PhysicalBinding", (key,))?.is_some() {
            for (index, output) in self.query.outputs(*symbol)?.into_iter().enumerate() {
                if query.contains(
                    "AbiOutputBinding",
                    (output, i64::from(binding.set), i64::from(binding.binding)),
                )? {
                    return Ok(format!("{owner}_output_{index}"));
                }
            }
            return Err(error("pinned allocation has no output declaration"));
        }
        for (index, output) in self.query.outputs(*symbol)?.into_iter().enumerate() {
            let (_, _, resource) = self.query.output(output)?;
            if self.query.backing(resource) != Some(key) {
                continue;
            }
            let Some(definition) = self.query.program.source.defs.iter().find(|d| d.name == *symbol) else {
                return Err(error("output definition missing"));
            };
            let (body, _) = extract_lambda_params_ref(&definition.body);
            let field = match strip_existentials(&body.ty) {
                Type::Constructed(TypeName::Record(names), _) => names.0[index].clone(),
                Type::Constructed(TypeName::Tuple(_), _) => format!("result_{index}"),
                _ => "output".into(),
            };
            return Ok(format!("{owner}_{field}"));
        }
        Ok(format!("{owner}_scratch_{}_{}", binding.set, binding.binding))
    }
}
