//! Retain backend-ready TLC structure in arenas and emit its fusion summary.

use super::data::{
    Array, AssociatedData, BucketShapeData, BuiltinData, BuiltinId, DefinitionData, DefinitionId,
    DefinitionKind, EntryData, EntryParamData, ExprData, ExprId, ExprKind, ExternData, ExternId,
    InputBoundData, LoopKind, OperationData, OperationKind, OriginData, OriginId, ParameterData, Place,
    ProgramData, Reduction, RegionData, RegionId, Scan, ScremaForm, SoacBody, SymbolData, SymbolId,
    TypeData, TypeId,
};
use super::emit;
use crate::ast::Span;
use crate::builtins::lowering::PrimOp;
use crate::builtins::BuiltinLowering;
use crate::tlc::{self, data, VarRef};
use crate::{builtins, types, LookupMap};
use wyn_base::Interner;

type Term = tlc::Term<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type TermKind = tlc::TermKind<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type ArrayExpr = tlc::ArrayExpr<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type TlcLambda = tlc::Lambda<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type TlcSoacBody = tlc::SoacBody<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;
type SoacOp = tlc::SoacOp<data::ExplicitClosurePayload, data::ExplicitCapturesPayload>;

#[derive(Clone, Debug)]
pub struct Converted {
    /// Facts from the last completed pass: a fusion summary after import or
    /// optimization, augmented with expressions after insertion; scheduling
    /// replaces the fusion summary with planning and block/dispatch facts.
    pub program: Vec<egglog_engine::ast::Command>,
    /// Full bodies, expressions, arguments, types, and diagnostic metadata.
    pub data: AssociatedData,
    /// Independent post-fusion layer. Scheduling retains these commands when
    /// replacing the fusion summary with block topology.
    pub(super) expression_program: Option<Vec<egglog_engine::ast::Command>>,
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

/// Import the same TLC checkpoint accepted by `egir::from_tlc`. Types and pure
/// values are structurally interned in the sidecar; local lets become references.
/// Operations have their own identities and remain in their execution regions.
/// Map/reduce/scan construct Scremas directly.
/// Egglog receives only the fusion summary. This conversion neither mutates TLC
/// nor runs optimization or extraction.
pub fn convert_program(program: &tlc::stage::InputSliceBoundsInferred) -> Result<Converted, ConvertError> {
    let _timing = super::timing::span("from TLC");
    let import = super::timing::span("import sidecar");
    let mut converter = Converter::default();
    converter.data.programs.alloc(ProgramData {
        next_auto_storage_binding: program.global_context.auto_storage_binding_ids.peek_id(),
    });
    for (&source, name) in &program.symbols {
        let id = converter.data.symbols.alloc(SymbolData {
            source,
            name: name.clone(),
        });
        converter.symbols.insert(source, id);
    }
    for def in &program.defs {
        // Reserve callable identities before importing bodies, including forward references.
        let region = converter.data.regions.alloc_id();
        if converter.globals.insert(def.name, (def.arity, def.ty.clone(), region)).is_some() {
            return Err(ConvertError::DuplicateDefinition(def.name));
        }
    }
    for def in &program.defs {
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
                for (position, binding) in entry.data.param_bindings.iter().enumerate() {
                    if let Some(binding) = binding {
                        converter.symbol(binding.param_sym)?;
                    }
                    converter.data.entry_params.alloc(EntryParamData {
                        entry: entry_id,
                        position,
                        binding: binding.clone(),
                    });
                }
                let mut bounds: Vec<_> = entry.data.by_symbol.iter().collect();
                bounds.sort_by_key(|(symbol, _)| symbol.0);
                for (&source, length) in bounds {
                    let symbol = converter.symbol(source)?;
                    converter.data.input_bounds.alloc(InputBoundData {
                        entry: entry_id,
                        symbol,
                        length: length.clone(),
                    });
                }
                DefinitionKind::Entry(entry_id)
            }
        };
        let region = converter.globals[&def.name].2;
        let mut scope = converter.scope(region, id, None, LookupMap::new());
        let result = if let TermKind::Lambda(lambda) = &def.body.kind {
            converter.parameters(&lambda.params, &mut scope)?;
            converter.term(&lambda.body, &mut scope)?
        } else {
            converter.term(&def.body, &mut scope)?
        };
        let body = converter.finish(scope, vec![result]);
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
    }
    converter.data.types = converter.types.into_arena();
    converter.data.expressions = converter.expressions.into_arena();
    converter.data.origins = converter.origins.into_arena();
    drop(import);
    let output = emit::program(&converter.data);
    let _parse = super::timing::span("parse fusion facts");
    let program = egglog_engine::ast::Parser::default()
        .get_program_from_string(Some("wyn-from-tlc.egg".into()), &output)
        .map_err(|error| ConvertError::InvalidProgram(error.to_string()))?;
    Ok(Converted {
        program,
        data: converter.data,
        expression_program: None,
    })
}

#[derive(Default)]
struct Converter {
    data: AssociatedData,
    symbols: LookupMap<crate::SymbolId, SymbolId>,
    globals: LookupMap<crate::SymbolId, (usize, types::Type, RegionId)>,
    types: Interner<TypeId, TypeData>,
    expressions: Interner<ExprId, ExprData>,
    origins: Interner<OriginId, OriginData>,
    builtins: LookupMap<(builtins::BuiltinId, usize), BuiltinId>,
    externs: LookupMap<String, ExternId>,
}

struct Scope {
    id: RegionId,
    data: RegionData,
    locals: LookupMap<crate::SymbolId, ExprId>,
}

impl Converter {
    fn symbol(&self, source: crate::SymbolId) -> Result<SymbolId, ConvertError> {
        let Some(id) = self.symbols.get(&source) else {
            return Err(ConvertError::MissingSymbol(source));
        };
        Ok(*id)
    }
    fn ty(&mut self, ty: &types::Type) -> TypeId {
        self.types.intern(&TypeData { ty: ty.clone() })
    }
    fn expr(&mut self, ty: TypeId, kind: ExprKind) -> ExprId {
        self.expressions.intern(&ExprData { ty, kind })
    }
    fn retype(&mut self, value: ExprId, ty: TypeId) -> ExprId {
        let data = self.expressions.resolve(value);
        if data.ty == ty {
            value
        } else {
            self.expr(ty, data.kind.clone())
        }
    }
    fn scope(
        &mut self,
        id: RegionId,
        definition: DefinitionId,
        parent: Option<RegionId>,
        locals: LookupMap<crate::SymbolId, ExprId>,
    ) -> Scope {
        Scope {
            id,
            locals,
            data: RegionData {
                definition,
                parent,
                parameters: vec![],
                members: Default::default(),
                results: vec![],
            },
        }
    }
    fn child(&mut self, scope: &Scope) -> Scope {
        let id = self.data.regions.alloc_id();
        self.scope(id, scope.data.definition, Some(scope.id), scope.locals.clone())
    }
    fn finish(&mut self, mut scope: Scope, results: Vec<ExprId>) -> RegionId {
        scope.data.results = results;
        self.data.regions.insert(scope.id, scope.data);
        scope.id
    }
    fn parameters(
        &mut self,
        parameters: &[(crate::SymbolId, types::Type)],
        scope: &mut Scope,
    ) -> Result<(), ConvertError> {
        for (source, ty) in parameters {
            let symbol = self.symbol(*source)?;
            let ty = self.ty(ty);
            let parameter = self.data.parameters.alloc(ParameterData {
                symbol,
                ty,
                region: scope.id,
            });
            scope.data.parameters.push(parameter);
            let value = self.expr(ty, ExprKind::Parameter(parameter));
            scope.locals.insert(*source, value);
        }
        Ok(())
    }
    fn operation(&mut self, kind: OperationKind, ty: TypeId, span: Span, scope: &mut Scope) -> ExprId {
        let id = self.data.operations.alloc(OperationData {
            kind,
            ty,
            span,
            region: scope.id,
            source_position: scope.data.members.len(),
        });
        scope.data.members.insert(id);
        self.expr(ty, ExprKind::OperationResult(id))
    }
    fn var(
        &mut self,
        var: VarRef,
        ty: TypeId,
        span: Span,
        scope: &mut Scope,
    ) -> Result<ExprId, ConvertError> {
        match var {
            VarRef::Symbol(source) => {
                let symbol = self.symbol(source)?;
                if let Some(&value) = scope.locals.get(&source) {
                    return Ok(self.retype(value, ty));
                }
                if self.globals.get(&source).is_some_and(|(arity, _, _)| *arity == 0) {
                    return Ok(self.operation(OperationKind::EvalGlobal(symbol), ty, span, scope));
                }
                Ok(self.expr(ty, ExprKind::Global(symbol)))
            }
            VarRef::Builtin { id, overload_idx } => {
                let builtin = *self.builtins.entry((id, overload_idx)).or_insert_with(|| {
                    self.data.builtins.alloc(BuiltinData {
                        builtin: id,
                        overload_idx,
                    })
                });
                Ok(self.expr(ty, ExprKind::Builtin(builtin)))
            }
        }
    }
    fn terms(&mut self, terms: &[Term], scope: &mut Scope) -> Result<Vec<ExprId>, ConvertError> {
        terms.iter().map(|term| self.term(term, scope)).collect()
    }
    fn lambda(&mut self, lambda: &TlcLambda, scope: &Scope) -> Result<RegionId, ConvertError> {
        let mut body = self.child(scope);
        self.parameters(&lambda.params, &mut body)?;
        let result = self.term(&lambda.body, &mut body)?;
        Ok(self.finish(body, vec![result]))
    }
    fn soac_body(&mut self, body: &TlcSoacBody, scope: &mut Scope) -> Result<SoacBody, ConvertError> {
        let mut captures = Vec::new();
        for (_, _, capture) in &body.data.captures {
            captures.push(self.term(capture, scope)?);
        }
        let results = vec![self.ty(&body.lam.ret_ty)];
        let parameters = body.lam.params.iter().map(|(_, ty)| self.ty(ty)).collect();
        // Closure conversion stores a callable reference here. Its arguments
        // are the SOAC inputs followed by the explicit capture values.
        if let TermKind::Var(VarRef::Symbol(function)) = &body.lam.body.kind {
            if let Some(&(arity, _, region)) = self.globals.get(function) {
                if arity > 0 {
                    return Ok(SoacBody::Apply {
                        region,
                        parameters,
                        results,
                        captures,
                    });
                }
            }
        }
        // A callable has an explicit interface: no inherited local bindings.
        let id = self.data.regions.alloc_id();
        let mut inner = self.scope(id, scope.data.definition, Some(scope.id), LookupMap::new());
        self.parameters(&body.lam.params, &mut inner)?;
        let capture_parameters =
            body.data.captures.iter().map(|(symbol, ty, _)| (*symbol, ty.clone())).collect::<Vec<_>>();
        self.parameters(&capture_parameters, &mut inner)?;
        let result = self.term(&body.lam.body, &mut inner)?;
        let region = self.finish(inner, vec![result]);
        Ok(SoacBody::Apply {
            region,
            parameters,
            results,
            captures,
        })
    }
    fn array(&mut self, array: &ArrayExpr, scope: &mut Scope) -> Result<Array, ConvertError> {
        Ok(match array {
            ArrayExpr::Var(var, ty) => {
                let ty = self.ty(ty);
                Array::Value(self.var(*var, ty, Span::generated(), scope)?)
            }
            ArrayExpr::Zip(arrays) => Array::Zip(self.arrays(arrays, scope)?),
            ArrayExpr::Literal(terms) => Array::Literal(self.terms(terms, scope)?),
            ArrayExpr::Range { start, len, step } => Array::Range {
                start: self.term(start, scope)?,
                len: self.term(len, scope)?,
                step: step.as_ref().map(|step| self.term(step, scope)).transpose()?,
            },
        })
    }
    fn arrays(&mut self, arrays: &[ArrayExpr], scope: &mut Scope) -> Result<Vec<Array>, ConvertError> {
        arrays.iter().map(|array| self.array(array, scope)).collect()
    }
    fn place(&mut self, place: &tlc::Place, scope: &mut Scope) -> Result<Place, ConvertError> {
        let elem_ty = self.ty(&place.elem_ty);
        let value = if let Some(&value) = scope.locals.get(&place.id) {
            value
        } else {
            let ty = self
                .globals
                .get(&place.id)
                .map(|(_, ty, _)| ty.clone())
                .unwrap_or_else(|| place.elem_ty.clone());
            let ty = self.ty(&ty);
            self.var(VarRef::Symbol(place.id), ty, Span::generated(), scope)?
        };
        Ok(Place { value, elem_ty })
    }
    fn soac(&mut self, soac: &SoacOp, term: &Term, scope: &mut Scope) -> Result<ExprId, ConvertError> {
        let kind = match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => {
                let pre = self.soac_body(lam, scope)?;
                let post = SoacBody::Identity(vec![self.ty(&lam.lam.ret_ty)]);
                let inputs = self.arrays(inputs, scope)?;
                OperationKind::Screma {
                    form: ScremaForm {
                        pre,
                        scans: vec![],
                        reductions: vec![],
                        post,
                    },
                    inputs,
                    ownership: vec![*destination],
                }
            }
            SoacOp::Reduce { op, ne, input } => {
                let pre = SoacBody::Identity(vec![self.ty(&op.lam.ret_ty)]);
                let operator = self.soac_body(op, scope)?;
                let neutral = vec![self.term(ne, scope)?];
                let input = self.array(input, scope)?;
                let form = ScremaForm {
                    pre,
                    scans: vec![],
                    reductions: vec![Reduction {
                        operator,
                        neutral,
                        commutative: false,
                    }],
                    post: SoacBody::Identity(vec![]),
                };
                OperationKind::Screma {
                    form,
                    inputs: vec![input],
                    ownership: vec![types::SoacOwnership::Fresh],
                }
            }
            SoacOp::Scan {
                op,
                ne,
                input,
                destination,
            } => {
                let pre = SoacBody::Identity(vec![self.ty(&op.lam.ret_ty)]);
                let operator = self.soac_body(op, scope)?;
                let neutral = vec![self.term(ne, scope)?];
                let input = self.array(input, scope)?;
                let form = ScremaForm {
                    post: pre.clone(),
                    pre,
                    scans: vec![Scan { operator, neutral }],
                    reductions: vec![],
                };
                OperationKind::Screma {
                    form,
                    inputs: vec![input],
                    ownership: vec![*destination],
                }
            }
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => {
                let body = self.soac_body(pred, scope)?;
                OperationKind::Filter {
                    map: SoacBody::Identity(super::fusion::signature(&body).0),
                    body,
                    inputs: vec![self.array(input, scope)?],
                    ownership: *destination,
                }
            }
            SoacOp::Scatter { dest, lam, inputs } => OperationKind::Scatter {
                destination: self.place(dest, scope)?,
                body: self.soac_body(lam, scope)?,
                inputs: self.arrays(inputs, scope)?,
            },
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
                OperationKind::BucketScatter {
                    destination: self.place(dest, scope)?,
                    body: self.soac_body(lam, scope)?,
                    inputs: self.arrays(inputs, scope)?,
                    shape,
                }
            }
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => {
                let mut parameters = Vec::new();
                for input in [indices, values] {
                    let array_ty = types::canonical_storage_buffer_ty(&input.array_type());
                    let element = types::array_elem(&array_ty).ok_or_else(|| {
                        ConvertError::InvalidProgram("reduce-by-index input must be an array".into())
                    })?;
                    parameters.push(self.ty(element));
                }
                OperationKind::ReduceByIndex {
                    destination: self.place(dest, scope)?,
                    map: SoacBody::Identity(parameters),
                    body: self.soac_body(op, scope)?,
                    neutral: self.term(ne, scope)?,
                    inputs: vec![self.array(indices, scope)?, self.array(values, scope)?],
                }
            }
        };
        let ty = self.ty(&term.ty);
        if matches!(&kind, OperationKind::Screma { .. }) {
            let tuple_ty = self.ty(&types::tuple(vec![term.ty.clone()]));
            let tuple = self.operation(kind, tuple_ty, term.span, scope);
            Ok(self.expr(ty, ExprKind::Project { tuple, index: 0 }))
        } else {
            Ok(self.operation(kind, ty, term.span, scope))
        }
    }
    fn loop_value(&mut self, term: &Term, scope: &mut Scope) -> Result<ExprId, ConvertError> {
        let TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } = &term.kind
        else {
            unreachable!("loop_value requires a TLC loop");
        };
        let init = self.term(init, scope)?;
        let mut header = self.child(scope);
        self.parameters(&[(*loop_var, loop_var_ty.clone())], &mut header)?;
        let kind = match kind {
            tlc::LoopKind::For { var, var_ty, iter } => {
                let iter = self.term(iter, scope)?;
                self.parameters(&[(*var, var_ty.clone())], &mut header)?;
                LoopKind::For(iter)
            }
            tlc::LoopKind::ForRange { var, var_ty, bound } => {
                let bound = self.term(bound, scope)?;
                self.parameters(&[(*var, var_ty.clone())], &mut header)?;
                LoopKind::ForRange(bound)
            }
            tlc::LoopKind::While { .. } => LoopKind::While,
        };
        for (name, _, binding) in init_bindings {
            let value = self.term(binding, &mut header)?;
            header.locals.insert(*name, value);
        }
        let condition = if let TermKind::Loop {
            kind: tlc::LoopKind::While { cond },
            ..
        } = &term.kind
        {
            vec![self.term(cond, &mut header)?]
        } else {
            vec![]
        };
        let mut body_scope = self.child(&header);
        let result = self.term(body, &mut body_scope)?;
        let body = self.finish(body_scope, vec![result]);
        let header = self.finish(header, condition);
        let ty = self.ty(&term.ty);
        Ok(self.operation(
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            },
            ty,
            term.span,
            scope,
        ))
    }
    fn pure_callee(&self, function: ExprId, args: &[ExprId], result: TypeId) -> bool {
        match &self.expressions.resolve(function).kind {
            ExprKind::BinOp(_) | ExprKind::UnOp(_) => true,
            ExprKind::Builtin(id) => {
                let record = &self.data.builtins[*id];
                let builtin = record.builtin;
                // Slicing constructs a view; it does not read or write elements.
                if builtin == builtins::catalog().known().slice {
                    return true;
                }
                let definition = builtins::by_id(builtin);
                let Some(overload) = definition.overloads().get(record.overload_idx) else {
                    return false;
                };
                // A movable pure application depends only on its operands.
                // Invocation queries, derivatives and opaque builtin lowering
                // retain an ordered execution even when the catalog calls them pure.
                let movable = match &overload.lowering {
                    BuiltinLowering::PrimOp(PrimOp::DPdx | PrimOp::DPdy | PrimOp::Fwidth) => false,
                    BuiltinLowering::PrimOp(_) | BuiltinLowering::ExtInstSplat { .. } => !args.is_empty(),
                    _ => false,
                };
                builtin != builtins::catalog().known().storage_index
                    && movable
                    && definition.raw.purity == builtins::Purity::Pure
                    && types::is_copy(&self.types.resolve(result).ty)
                    && args
                        .iter()
                        .all(|id| types::is_copy(&self.types.resolve(self.expressions.resolve(*id).ty).ty))
            }
            _ => false,
        }
    }
    fn term(&mut self, term: &Term, scope: &mut Scope) -> Result<ExprId, ConvertError> {
        let ty = self.ty(&term.ty);
        let value = match &term.kind {
            TermKind::Var(var) => self.var(*var, ty, term.span, scope)?,
            TermKind::BinOp(op) => self.expr(ty, ExprKind::BinOp(op.op.symbol().into())),
            TermKind::UnOp(op) => self.expr(ty, ExprKind::UnOp(op.op.symbol().into())),
            TermKind::IntLit(value) => self.expr(ty, ExprKind::Int(value.clone())),
            TermKind::FloatLit(value) => self.expr(ty, ExprKind::FloatBits(value.to_bits())),
            TermKind::BoolLit(value) => self.expr(ty, ExprKind::Bool(*value)),
            TermKind::UnitLit => self.expr(ty, ExprKind::Unit),
            TermKind::Lambda(lambda) => {
                let region = self.lambda(lambda, scope)?;
                self.expr(ty, ExprKind::Lambda(region))
            }
            TermKind::Closure(closure) => {
                let code = self.symbol(closure.code)?;
                let captures = self.terms(&closure.captures, scope)?;
                self.expr(
                    ty,
                    ExprKind::Closure {
                        code,
                        param_count: closure.param_count,
                        captures,
                    },
                )
            }
            TermKind::App { func, args } => {
                let function = self.term(func, scope)?;
                let args = self.terms(args, scope)?;
                if self.pure_callee(function, &args, ty) {
                    self.expr(ty, ExprKind::PureApp { function, args })
                } else {
                    self.operation(OperationKind::Call { function, args }, ty, term.span, scope)
                }
            }
            TermKind::Let { name, rhs, body, .. } => {
                self.symbol(*name)?;
                let rhs = self.term(rhs, scope)?;
                let old = scope.locals.insert(*name, rhs);
                let result = self.term(body, scope)?;
                if let Some(old) = old {
                    scope.locals.insert(*name, old);
                } else {
                    scope.locals.remove(name);
                }
                self.retype(result, ty)
            }
            TermKind::Coerce { inner, .. } => {
                let inner = self.term(inner, scope)?;
                self.expr(ty, ExprKind::Coerce(inner))
            }
            TermKind::Extern(name) => {
                let id = *self.externs.entry(name.clone()).or_insert_with(|| {
                    self.data.externs.alloc(ExternData {
                        linkage_name: name.clone(),
                    })
                });
                self.expr(ty, ExprKind::Extern(id))
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let condition = self.term(cond, scope)?;
                let mut then_scope = self.child(scope);
                let then_value = self.term(then_branch, &mut then_scope)?;
                let mut else_scope = self.child(scope);
                let else_value = self.term(else_branch, &mut else_scope)?;
                let pure = then_scope.data.members.is_empty() && else_scope.data.members.is_empty();
                let then_region = self.finish(then_scope, vec![then_value]);
                let else_region = self.finish(else_scope, vec![else_value]);
                if pure {
                    self.expr(
                        ty,
                        ExprKind::If {
                            condition,
                            then_value,
                            else_value,
                        },
                    )
                } else {
                    self.operation(
                        OperationKind::If {
                            condition,
                            then_region,
                            else_region,
                        },
                        ty,
                        term.span,
                        scope,
                    )
                }
            }
            TermKind::Loop { .. } => self.loop_value(term, scope)?,
            TermKind::Soac(soac) => self.soac(soac, term, scope)?,
            TermKind::ArrayExpr(array) => {
                let array = self.array(array, scope)?;
                if let Array::Value(value) = array {
                    self.retype(value, ty)
                } else {
                    self.expr(ty, ExprKind::Array(array))
                }
            }
            TermKind::Tuple(items) => {
                let items = self.terms(items, scope)?;
                self.expr(ty, ExprKind::Tuple(items))
            }
            TermKind::TupleProj { tuple, idx } => {
                let tuple = self.term(tuple, scope)?;
                self.expr(ty, ExprKind::Project { tuple, index: *idx })
            }
            TermKind::Index { array, index } => {
                let array = self.term(array, scope)?;
                let index = self.term(index, scope)?;
                self.operation(OperationKind::Index { array, index }, ty, term.span, scope)
            }
            TermKind::VecLit(items) => {
                let items = self.terms(items, scope)?;
                self.expr(ty, ExprKind::Vector(items))
            }
        };
        self.origins.intern(&OriginData {
            span: term.span,
            expression: value,
            definition: scope.data.definition,
        });
        Ok(value)
    }
}
