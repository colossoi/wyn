//! Lossless structural import of backend-ready TLC into egglog.

use std::fmt::Write;

use crate::tlc::{self, data, VarRef};
use crate::{types, LookupMap, LookupSet};

use super::*;

type Term = tlc::Term<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type TermKind = tlc::TermKind<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type ArrayExpr = tlc::ArrayExpr<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type Lambda = tlc::Lambda<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type SoacBody = tlc::SoacBody<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type SoacOp = tlc::SoacOp<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;

/// An egglog command AST and its matching metadata arenas.
#[derive(Clone, Debug)]
pub struct Converted {
    pub program: Vec<egglog_engine::ast::Command>,
    pub data: AssociatedData,
}

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("egglog import: symbol {0:?} is missing from the TLC symbol table")]
    MissingSymbol(crate::SymbolId),
    #[error("egglog import: duplicate definition for symbol {0:?}")]
    DuplicateDefinition(crate::SymbolId),
    #[error("egglog import generated an invalid program: {0}")]
    InvalidProgram(String),
}

/// Import the same TLC checkpoint accepted by `egir::from_tlc`, without
/// mutating it or running any EGIR passes. Every current TLC variant is
/// represented structurally; source names never become egglog identifiers.
///
/// The AST can be passed directly to `egglog::EGraph::run_program`.
/// Conversion does not run rules or extraction.
pub fn convert_program(program: &tlc::stage::InputSliceBoundsInferred) -> Result<Converted, ConvertError> {
    let mut converter = Converter::default();
    let program_id = converter.data.programs.alloc(ProgramData {
        next_auto_storage_binding: program.global_context.auto_storage_binding_ids.peek_id(),
    });
    converter.emit(format!("(Program {})", program_id.egglog()));

    for (&source, name) in &program.symbols {
        let id = converter.data.symbols.alloc(SymbolData {
            source,
            name: name.clone(),
        });
        converter.symbols.insert(source, id);
    }

    let mut seen = LookupSet::new();
    for def in &program.defs {
        if !seen.insert(def.name) {
            return Err(ConvertError::DuplicateDefinition(def.name));
        }
        let id = converter.data.definitions.alloc_id();
        let symbol = converter.symbol(def.name)?;
        let ty = converter.ty(&def.ty);
        let kind = match &def.meta {
            tlc::DefMeta::Function => DefinitionKind::Function,
            tlc::DefMeta::LiftedLambda => DefinitionKind::LiftedLambda,
            tlc::DefMeta::EntryPoint(entry) => {
                let entry_id = converter.data.entries.alloc(EntryData {
                    definition: id,
                    declaration: (*entry.declaration).clone(),
                });
                converter.emit(format!("(Entry {} {})", entry_id.egglog(), id.egglog()));
                for (position, binding) in entry.data.param_bindings.iter().enumerate() {
                    if let Some(binding) = binding {
                        converter.symbol(binding.param_sym)?;
                    }
                    let param_id = converter.data.entry_params.alloc(EntryParamData {
                        entry: entry_id,
                        position,
                        binding: binding.clone(),
                    });
                    converter.emit(format!(
                        "(EntryParam {} {} {position})",
                        param_id.egglog(),
                        entry_id.egglog(),
                    ));
                }
                // Entry bounds originate in a HashMap. Sort before allocating
                // IDs so output and sidecar order are reproducible.
                let mut bounds: Vec<_> = entry.data.by_symbol.iter().collect();
                bounds.sort_by_key(|(symbol, _)| symbol.0);
                for (&source, length) in bounds {
                    let symbol = converter.symbol(source)?;
                    let bound_id = converter.data.input_bounds.alloc(InputBoundData {
                        entry: entry_id,
                        symbol,
                        length: length.clone(),
                    });
                    converter.emit(format!(
                        "(InputBound {} {} {})",
                        bound_id.egglog(),
                        entry_id.egglog(),
                        symbol.egglog(),
                    ));
                }
                DefinitionKind::Entry(entry_id)
            }
        };
        let body = converter.term(&def.body, id)?;
        converter.data.definitions.insert(
            id,
            DefinitionData {
                symbol,
                package: def.package,
                ty,
                body,
                kind,
                arity: def.arity,
                param_diets: def.param_diets.clone(),
                return_diet: def.return_diet.clone(),
            },
        );
        converter.emit(format!(
            "(Definition {} {} {} {})",
            id.egglog(),
            symbol.egglog(),
            ty.egglog(),
            body.binding_name(),
        ));
    }

    let program = egglog_engine::ast::Parser::default()
        .get_program_from_string(Some("wyn-from-tlc.egg".into()), &converter.output)
        .map_err(|error| ConvertError::InvalidProgram(error.to_string()))?;
    Ok(Converted {
        program,
        data: converter.data,
    })
}

struct Converter {
    data: AssociatedData,
    output: String,
    next_aux: usize,
    symbols: LookupMap<crate::SymbolId, SymbolId>,
    types: LookupMap<types::Type, TypeId>,
    builtins: LookupMap<(crate::builtins::BuiltinId, usize), BuiltinId>,
    externs: LookupMap<String, ExternId>,
}

impl Default for Converter {
    fn default() -> Self {
        Self {
            data: AssociatedData::default(),
            output: SCHEMA.to_owned(),
            next_aux: 0,
            symbols: LookupMap::new(),
            types: LookupMap::new(),
            builtins: LookupMap::new(),
            externs: LookupMap::new(),
        }
    }
}

impl Converter {
    fn emit(&mut self, command: String) {
        writeln!(self.output, "{command}").unwrap();
    }

    fn bind(&mut self, expression: String) -> String {
        let name = format!("aux-{}", self.next_aux);
        self.next_aux += 1;
        self.emit(format!("(let {name} {expression})"));
        name
    }

    /// Emit lists as flat let-bindings, avoiding quadratic string construction
    /// and unbounded parser nesting for large arrays or argument lists.
    fn list(&mut self, nil: &str, cons: &str, items: Vec<String>) -> String {
        let mut tail = format!("({nil})");
        for item in items.into_iter().rev() {
            tail = self.bind(format!("({cons} {item} {tail})"));
        }
        tail
    }

    fn symbol(&self, source: crate::SymbolId) -> Result<SymbolId, ConvertError> {
        self.symbols.get(&source).copied().ok_or(ConvertError::MissingSymbol(source))
    }

    fn ty(&mut self, ty: &types::Type) -> TypeId {
        if let Some(&id) = self.types.get(ty) {
            return id;
        }
        let id = self.data.types.alloc(TypeData { ty: ty.clone() });
        self.types.insert(ty.clone(), id);
        id
    }

    fn var(&mut self, var: VarRef) -> Result<String, ConvertError> {
        Ok(match var {
            VarRef::Symbol(source) => format!("(Var {})", self.symbol(source)?.egglog()),
            VarRef::Builtin { id, overload_idx } => {
                let builtin = *self.builtins.entry((id, overload_idx)).or_insert_with(|| {
                    self.data.builtins.alloc(BuiltinData {
                        builtin: id,
                        overload_idx,
                    })
                });
                format!("(Builtin {})", builtin.egglog())
            }
        })
    }

    fn param(&mut self, symbol: crate::SymbolId, ty: &types::Type) -> Result<String, ConvertError> {
        Ok(format!(
            "(MkParam {} {})",
            self.symbol(symbol)?.egglog(),
            self.ty(ty).egglog()
        ))
    }

    fn terms(&mut self, terms: &[Term], owner: DefinitionId) -> Result<String, ConvertError> {
        let items = terms
            .iter()
            .map(|term| self.term(term, owner).map(TermId::binding_name))
            .collect::<Result<_, _>>()?;
        Ok(self.list("NoExprs", "ExprsCons", items))
    }

    fn bindings(
        &mut self,
        bindings: &[(crate::SymbolId, types::Type, Term)],
        owner: DefinitionId,
    ) -> Result<String, ConvertError> {
        let mut items = Vec::with_capacity(bindings.len());
        for (symbol, ty, value) in bindings {
            let param = self.param(*symbol, ty)?;
            let value = self.term(value, owner)?.binding_name();
            items.push(self.bind(format!("(MkBinding {param} {value})")));
        }
        Ok(self.list("NoBindings", "BindingsCons", items))
    }

    fn lambda(&mut self, lambda: &Lambda, owner: DefinitionId) -> Result<String, ConvertError> {
        let params =
            lambda.params.iter().map(|(symbol, ty)| self.param(*symbol, ty)).collect::<Result<_, _>>()?;
        let params = self.list("NoParams", "ParamsCons", params);
        let ty = self.ty(&lambda.ret_ty).egglog();
        let body = self.term(&lambda.body, owner)?.binding_name();
        Ok(self.bind(format!("(MkLambda {params} {ty} {body})")))
    }

    fn soac_body(&mut self, body: &SoacBody, owner: DefinitionId) -> Result<String, ConvertError> {
        let lambda = self.lambda(&body.lam, owner)?;
        let captures = self.bindings(&body.data.captures, owner)?;
        Ok(self.bind(format!("(MkSoacBody {lambda} {captures})")))
    }

    fn arrays(&mut self, arrays: &[ArrayExpr], owner: DefinitionId) -> Result<String, ConvertError> {
        let items = arrays.iter().map(|array| self.array(array, owner)).collect::<Result<_, _>>()?;
        Ok(self.list("NoArrays", "ArraysCons", items))
    }

    fn array(&mut self, array: &ArrayExpr, owner: DefinitionId) -> Result<String, ConvertError> {
        let expression = match array {
            ArrayExpr::Var(var, ty) => {
                format!("(ArrayVar {} {})", self.ty(ty).egglog(), self.var(*var)?)
            }
            ArrayExpr::Zip(arrays) => format!("(Zip {})", self.arrays(arrays, owner)?),
            ArrayExpr::Literal(terms) => format!("(ArrayLiteral {})", self.terms(terms, owner)?),
            ArrayExpr::Range { start, len, step } => {
                let start = self.term(start, owner)?.binding_name();
                let len = self.term(len, owner)?.binding_name();
                let step = match step {
                    Some(step) => format!("(SomeExpr {})", self.term(step, owner)?.binding_name()),
                    None => "(NoExpr)".into(),
                };
                format!("(Range {start} {len} {step})")
            }
        };
        Ok(self.bind(expression))
    }

    fn place(&mut self, place: &tlc::Place) -> Result<String, ConvertError> {
        Ok(format!(
            "(MkPlace {} {})",
            self.symbol(place.id)?.egglog(),
            self.ty(&place.elem_ty).egglog()
        ))
    }

    fn soac(&mut self, soac: &SoacOp, site: TermId, owner: DefinitionId) -> Result<String, ConvertError> {
        let site = site.egglog();
        Ok(match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => format!(
                "(Map {site} {} {} {})",
                self.soac_body(lam, owner)?,
                self.arrays(inputs, owner)?,
                ownership(*destination),
            ),
            SoacOp::Reduce { op, ne, input } => format!(
                "(Reduce {site} {} {} {})",
                self.soac_body(op, owner)?,
                self.term(ne, owner)?.binding_name(),
                self.array(input, owner)?,
            ),
            SoacOp::Scan {
                op,
                ne,
                input,
                destination,
            } => format!(
                "(Scan {site} {} {} {} {})",
                self.soac_body(op, owner)?,
                self.term(ne, owner)?.binding_name(),
                self.array(input, owner)?,
                ownership(*destination),
            ),
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => format!(
                "(Filter {site} {} {} {})",
                self.soac_body(pred, owner)?,
                self.array(input, owner)?,
                ownership(*destination),
            ),
            SoacOp::Scatter { dest, lam, inputs } => format!(
                "(Scatter {site} {} {} {})",
                self.place(dest)?,
                self.soac_body(lam, owner)?,
                self.arrays(inputs, owner)?,
            ),
            SoacOp::BucketScatter {
                dest,
                lam,
                inputs,
                input_dimensions,
                domain_rank,
            } => {
                let shape = self.data.bucket_shapes.alloc(BucketShapeData {
                    input_dimensions: input_dimensions.clone(),
                    domain_rank: *domain_rank,
                });
                format!(
                    "(BucketScatter {site} {} {} {} {})",
                    self.place(dest)?,
                    self.soac_body(lam, owner)?,
                    self.arrays(inputs, owner)?,
                    shape.egglog(),
                )
            }
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => format!(
                "(ReduceByIndex {site} {} {} {} {} {})",
                self.place(dest)?,
                self.soac_body(op, owner)?,
                self.term(ne, owner)?.binding_name(),
                self.array(indices, owner)?,
                self.array(values, owner)?,
            ),
        })
    }

    fn term(&mut self, term: &Term, owner: DefinitionId) -> Result<TermId, ConvertError> {
        let ty = self.ty(&term.ty);
        // Allocate per occurrence, even when cloned TLC nodes share a source ID.
        let id = self.data.terms.alloc(TermData {
            source: term.id,
            span: term.span,
            ty,
            definition: owner,
        });
        let expression = match &term.kind {
            TermKind::Var(var) => self.var(*var)?,
            TermKind::BinOp(op) => format!("(BinOp {})", quote(op.op.symbol())),
            TermKind::UnOp(op) => format!("(UnOp {})", quote(op.op.symbol())),
            TermKind::Lambda(lambda) => format!("(LambdaValue {})", self.lambda(lambda, owner)?),
            TermKind::Closure(closure) => format!(
                "(Closure {} {} {})",
                self.symbol(closure.code)?.egglog(),
                closure.param_count,
                self.terms(&closure.captures, owner)?,
            ),
            TermKind::App { func, args } => format!(
                "(App {} {} {})",
                id.egglog(),
                self.term(func, owner)?.binding_name(),
                self.terms(args, owner)?,
            ),
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => format!(
                "(Let {} {} {})",
                self.param(*name, name_ty)?,
                self.term(rhs, owner)?.binding_name(),
                self.term(body, owner)?.binding_name(),
            ),
            TermKind::IntLit(value) => format!("(Int {})", quote(value)),
            TermKind::FloatLit(value) => format!("(FloatBits {})", value.to_bits()),
            TermKind::BoolLit(value) => format!("(Bool {value})"),
            TermKind::UnitLit => "(UnitLit)".into(),
            TermKind::Coerce { inner, target_ty } => format!(
                "(Coerce {} {})",
                self.term(inner, owner)?.binding_name(),
                self.ty(target_ty).egglog(),
            ),
            TermKind::Extern(name) => {
                let extern_id = *self.externs.entry(name.clone()).or_insert_with(|| {
                    self.data.externs.alloc(ExternData {
                        linkage_name: name.clone(),
                    })
                });
                format!("(Extern {})", extern_id.egglog())
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => format!(
                "(If {} {} {})",
                self.term(cond, owner)?.binding_name(),
                self.term(then_branch, owner)?.binding_name(),
                self.term(else_branch, owner)?.binding_name(),
            ),
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let param = self.param(*loop_var, loop_var_ty)?;
                let init = self.term(init, owner)?.binding_name();
                let bindings = self.bindings(init_bindings, owner)?;
                let kind = match kind {
                    tlc::LoopKind::For { var, var_ty, iter } => format!(
                        "(For {} {})",
                        self.param(*var, var_ty)?,
                        self.term(iter, owner)?.binding_name(),
                    ),
                    tlc::LoopKind::ForRange { var, var_ty, bound } => format!(
                        "(ForRange {} {})",
                        self.param(*var, var_ty)?,
                        self.term(bound, owner)?.binding_name(),
                    ),
                    tlc::LoopKind::While { cond } => {
                        format!("(While {})", self.term(cond, owner)?.binding_name())
                    }
                };
                let body = self.term(body, owner)?.binding_name();
                format!("(Loop {} {param} {init} {bindings} {kind} {body})", id.egglog())
            }
            TermKind::Soac(soac) => self.soac(soac, id, owner)?,
            TermKind::ArrayExpr(array) => format!("(ArrayValue {})", self.array(array, owner)?),
            TermKind::Tuple(items) => format!("(Tuple {})", self.terms(items, owner)?),
            TermKind::TupleProj { tuple, idx } => {
                format!("(Project {} {idx})", self.term(tuple, owner)?.binding_name())
            }
            TermKind::Index { array, index } => format!(
                "(Index {} {} {})",
                id.egglog(),
                self.term(array, owner)?.binding_name(),
                self.term(index, owner)?.binding_name(),
            ),
            TermKind::VecLit(items) => format!("(Vector {})", self.terms(items, owner)?),
        };
        let name = id.binding_name();
        self.emit(format!("(let {name} (Typed {} {expression}))", ty.egglog()));
        self.emit(format!("(SourceTerm {} {name})", id.egglog()));
        Ok(id)
    }
}

fn ownership(ownership: types::SoacOwnership) -> &'static str {
    match ownership {
        types::SoacOwnership::Fresh => "(Fresh)",
        types::SoacOwnership::UniqueInput => "(UniqueInput)",
    }
}

fn quote(value: &str) -> String {
    // Only literal values and the fixed operator vocabulary enter the program
    // as strings. Source names and extern linkage strings stay in their arenas.
    serde_json::to_string(value).expect("serializing a string cannot fail")
}
