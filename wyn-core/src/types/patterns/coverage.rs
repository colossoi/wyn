//! Pattern coverage analysis: exhaustiveness + redundancy checking
//! using the "usefulness" relation from Maranget's algorithm.
//!
//! References:
//!   - Luc Maranget, "Compiling Pattern Matching to Good Decision Trees",
//!     ML Workshop 2008. Section 3 defines the U(M, p) usefulness
//!     relation used here.
//!   - Alan Hu, "Pattern Matching" — accessible walkthrough with worked
//!     examples:
//!     https://web.archive.org/web/20220410075515/https://alan-j-hu.github.io/writing/pattern-matching.html
//!
//! We use the usefulness relation only for checking. We do NOT compile
//! to decision trees; the TLC lowering generates sequential if-chains
//! and lets the SPIR-V optimizer clean up redundant comparisons.
//!
//! The two checks built on `useful`:
//!   - Exhaustiveness: is the wildcard pattern useful against the match
//!     matrix? If yes, the returned witness names a missing case.
//!   - Redundancy: for each arm i, is its pattern useful against the
//!     matrix of arms 1..i? If not, arm i is dead.

use crate::ast::{self, PatternKind, PatternLiteral, Span};
use crate::types::{Type, TypeName};
use crate::LookupSet;

/// Pattern in coverage form. AST `Typed`/`Attributed` are stripped
/// (delegated to inner). `Name` collapses to `Wild` — name bindings
/// don't constrain values, only refine names.
#[derive(Debug, Clone)]
pub enum CovPat {
    Wild,
    Lit(CovLit),
    UnitP,
    Tuple(Vec<CovPat>),
    Record(Vec<(String, CovPat)>),
    Ctor(String, Vec<CovPat>),
}

/// Literal value as a comparison key. `Float` stores the f32 bit
/// pattern so structurally-distinct NaN payloads compare as distinct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CovLit {
    Int(String),
    Float(u32),
    Bool(bool),
}

/// The set of values inhabiting a type at a pattern position. Drives
/// specialization and "is this column complete" decisions.
#[derive(Debug, Clone)]
pub enum Universe {
    /// Sum type with its variants (constructor name → payload types).
    Sum(Vec<(String, Vec<Type>)>),
    /// `{true, false}`.
    Bool,
    /// Any integer width — infinite, so a wildcard is required for
    /// exhaustive coverage.
    IntLike,
    /// Any float width — infinite.
    FloatLike,
    /// Single inhabitant.
    Unit,
    /// Fixed-arity tuple with known component types.
    Tuple(Vec<Type>),
    /// Record with named fields, source order preserved.
    Record(Vec<(String, Type)>),
    /// Type variable or some other type we can't enumerate — treat as
    /// infinite for coverage purposes.
    Opaque,
}

impl Universe {
    /// Build a Universe from a (post-substitution) type.
    pub fn of(ty: &Type) -> Universe {
        match ty {
            Type::Constructed(TypeName::Sum(variants), _) => Universe::Sum(variants.clone()),
            Type::Constructed(TypeName::Bool, _) => Universe::Bool,
            Type::Constructed(TypeName::Int(_), _) | Type::Constructed(TypeName::UInt(_), _) => {
                Universe::IntLike
            }
            Type::Constructed(TypeName::Float(_), _) => Universe::FloatLike,
            Type::Constructed(TypeName::Unit, _) => Universe::Unit,
            Type::Constructed(TypeName::Tuple(_), args) => Universe::Tuple(args.clone()),
            Type::Constructed(TypeName::Record(field_names), args) => {
                Universe::Record(field_names.iter().cloned().zip(args.iter().cloned()).collect())
            }
            _ => Universe::Opaque,
        }
    }
}

/// Errors from `check_match`.
#[derive(Debug, Clone)]
pub enum CoverageError {
    /// At least one value of `scrutinee_ty` is not matched by any arm.
    /// `missing` is a witness pattern naming an uncovered shape.
    NonExhaustive {
        missing: CovPat,
        match_span: Span,
    },
    /// Arm `arm_index` is unreachable: every value it matches is
    /// already covered by an earlier arm.
    Redundant {
        arm_index: usize,
        arm_span: Span,
    },
}

/// Render a `CovPat` as a user-facing pattern string. Used for
/// non-exhaustive-match error messages.
pub fn format_cov_pat(pat: &CovPat) -> String {
    match pat {
        CovPat::Wild => "_".to_string(),
        CovPat::UnitP => "()".to_string(),
        CovPat::Lit(CovLit::Int(s)) => s.clone(),
        CovPat::Lit(CovLit::Float(bits)) => format!("{}f32", f32::from_bits(*bits)),
        CovPat::Lit(CovLit::Bool(b)) => b.to_string(),
        CovPat::Tuple(sub) => {
            let parts: Vec<String> = sub.iter().map(format_cov_pat).collect();
            format!("({})", parts.join(", "))
        }
        CovPat::Record(fields) => {
            let parts: Vec<String> =
                fields.iter().map(|(n, p)| format!("{} = {}", n, format_cov_pat(p))).collect();
            format!("{{{}}}", parts.join(", "))
        }
        CovPat::Ctor(name, sub) if sub.is_empty() => format!("#{}", name),
        CovPat::Ctor(name, sub) => {
            let parts: Vec<String> = sub.iter().map(format_cov_pat).collect();
            format!("#{}({})", name, parts.join(", "))
        }
    }
}

/// Translate an AST pattern into coverage form. Strips
/// `Typed`/`Attributed` and collapses `Name` to `Wild`.
pub fn lower(pat: &ast::Pattern) -> CovPat {
    match &pat.kind {
        PatternKind::Name(_) | PatternKind::Wildcard => CovPat::Wild,
        PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => lower(inner),
        PatternKind::Unit => CovPat::UnitP,
        PatternKind::Literal(lit) => CovPat::Lit(lower_literal(lit)),
        PatternKind::Tuple(sub) | PatternKind::Vec(sub) => CovPat::Tuple(sub.iter().map(lower).collect()),
        PatternKind::Record(fields) => CovPat::Record(
            fields
                .iter()
                .map(|f| {
                    let sub = f.pattern.as_ref().map(lower).unwrap_or(CovPat::Wild); // shorthand `{name}` binds without constraining
                    (f.field.clone(), sub)
                })
                .collect(),
        ),
        PatternKind::Constructor(name, sub) => CovPat::Ctor(name.clone(), sub.iter().map(lower).collect()),
    }
}

fn lower_literal(lit: &PatternLiteral) -> CovLit {
    match lit {
        PatternLiteral::Int(s) => CovLit::Int(canonicalize_int_literal(&s.0)),
        PatternLiteral::Float(v) => CovLit::Float(v.to_bits()),
        PatternLiteral::Bool(b) => CovLit::Bool(*b),
    }
}

/// Normalize an integer literal's text so two arms whose patterns
/// denote the same numeric value compare equal in coverage. The lexer
/// already maps hex / binary / underscored forms to a plain-decimal
/// string, but doesn't strip leading zeros or normalize `-0` vs `0`,
/// and a future lexer change could widen the surface forms further.
/// Parse-and-reformat as i128 for canonical decimal. On parse failure
/// (out-of-range, surprising form), fall back to the raw text — the
/// type checker will reject the literal at the boundary anyway.
fn canonicalize_int_literal(s: &str) -> String {
    s.parse::<i128>().map(|n| n.to_string()).unwrap_or_else(|_| s.to_string())
}

/// Maranget's U(M, q): is the query row `q` useful against matrix `M`?
/// Returns `Some(witness)` when a value matching `q` but no row of `M`
/// can be constructed; `None` when every value matching `q` is already
/// matched by some row.
///
/// `col_tys[i]` is the static type of column `i` (universe driver).
pub fn useful(matrix: &[Vec<CovPat>], q: &[CovPat], col_tys: &[Type]) -> Option<Vec<CovPat>> {
    // Base cases
    if q.is_empty() {
        // Zero columns: q matches the single value (). Useful iff M is empty.
        return if matrix.is_empty() { Some(vec![]) } else { None };
    }
    if matrix.is_empty() {
        // Any value of the column types is a witness; q itself works.
        return Some(q.to_vec());
    }

    debug_assert_eq!(q.len(), col_tys.len(), "query arity must match column types");

    match &q[0] {
        CovPat::Ctor(name, sub) => specialize_query_ctor(matrix, name, sub, &q[1..], col_tys),
        CovPat::Tuple(sub) => specialize_query_tuple(matrix, sub, &q[1..], col_tys),
        CovPat::Record(fields) => specialize_query_record(matrix, fields, &q[1..], col_tys),
        CovPat::UnitP => specialize_query_unit(matrix, &q[1..], col_tys),
        CovPat::Lit(lit) => specialize_query_lit(matrix, lit, &q[1..], col_tys),
        CovPat::Wild => useful_wildcard(matrix, &q[1..], col_tys),
    }
}

/// Specialization S(c, M): keep rows whose head matches constructor `c`
/// (or is wild), replacing the head column with `c`'s sub-patterns.
fn specialize_ctor(matrix: &[Vec<CovPat>], name: &str, arity: usize) -> Vec<Vec<CovPat>> {
    let mut out = Vec::new();
    for row in matrix {
        match &row[0] {
            CovPat::Ctor(c, sub) if c == name => {
                let mut new_row: Vec<CovPat> = sub.clone();
                new_row.extend(row[1..].iter().cloned());
                out.push(new_row);
            }
            CovPat::Wild => {
                let mut new_row: Vec<CovPat> = (0..arity).map(|_| CovPat::Wild).collect();
                new_row.extend(row[1..].iter().cloned());
                out.push(new_row);
            }
            // Other constructors / literals filter out
            _ => {}
        }
    }
    out
}

/// Specialization for a tuple head (single shape — every value of a
/// tuple type matches the tuple form).
fn specialize_tuple(matrix: &[Vec<CovPat>], arity: usize) -> Vec<Vec<CovPat>> {
    let mut out = Vec::new();
    for row in matrix {
        match &row[0] {
            CovPat::Tuple(sub) => {
                let mut new_row: Vec<CovPat> = sub.clone();
                new_row.extend(row[1..].iter().cloned());
                out.push(new_row);
            }
            CovPat::Wild => {
                let mut new_row: Vec<CovPat> = (0..arity).map(|_| CovPat::Wild).collect();
                new_row.extend(row[1..].iter().cloned());
                out.push(new_row);
            }
            _ => {}
        }
    }
    out
}

/// Specialization for a record head with a known field-name order.
fn specialize_record(matrix: &[Vec<CovPat>], field_names: &[String]) -> Vec<Vec<CovPat>> {
    let arity = field_names.len();
    let mut out = Vec::new();
    for row in matrix {
        match &row[0] {
            CovPat::Record(row_fields) => {
                // Reorder row_fields to match field_names. Missing
                // fields become Wild — shouldn't happen if the
                // checker validated record patterns, but be defensive.
                let mut new_head: Vec<CovPat> = Vec::with_capacity(arity);
                for fname in field_names {
                    let sub = row_fields
                        .iter()
                        .find(|(n, _)| n == fname)
                        .map(|(_, p)| p.clone())
                        .unwrap_or(CovPat::Wild);
                    new_head.push(sub);
                }
                let mut new_row = new_head;
                new_row.extend(row[1..].iter().cloned());
                out.push(new_row);
            }
            CovPat::Wild => {
                let mut new_row: Vec<CovPat> = (0..arity).map(|_| CovPat::Wild).collect();
                new_row.extend(row[1..].iter().cloned());
                out.push(new_row);
            }
            _ => {}
        }
    }
    out
}

/// Specialization for a unit head (single inhabitant).
fn specialize_unit(matrix: &[Vec<CovPat>]) -> Vec<Vec<CovPat>> {
    let mut out = Vec::new();
    for row in matrix {
        match &row[0] {
            CovPat::UnitP | CovPat::Wild => {
                out.push(row[1..].to_vec());
            }
            _ => {}
        }
    }
    out
}

/// Specialization for a literal head: keep rows whose head is the same
/// literal or wild.
fn specialize_lit(matrix: &[Vec<CovPat>], lit: &CovLit) -> Vec<Vec<CovPat>> {
    let mut out = Vec::new();
    for row in matrix {
        match &row[0] {
            CovPat::Lit(l) if l == lit => out.push(row[1..].to_vec()),
            CovPat::Wild => out.push(row[1..].to_vec()),
            _ => {}
        }
    }
    out
}

/// Default matrix D(M): keep rows whose head is wild; drop the head
/// column.
fn default_matrix(matrix: &[Vec<CovPat>]) -> Vec<Vec<CovPat>> {
    matrix
        .iter()
        .filter_map(|row| match &row[0] {
            CovPat::Wild => Some(row[1..].to_vec()),
            _ => None,
        })
        .collect()
}

fn specialize_query_ctor(
    matrix: &[Vec<CovPat>],
    name: &str,
    sub: &[CovPat],
    rest: &[CovPat],
    col_tys: &[Type],
) -> Option<Vec<CovPat>> {
    let arity = sub.len();
    let m = specialize_ctor(matrix, name, arity);
    let mut new_q: Vec<CovPat> = sub.to_vec();
    new_q.extend(rest.iter().cloned());
    let new_col_tys = column_tys_for_ctor(name, &col_tys[0], &col_tys[1..]);
    let witness = useful(&m, &new_q, &new_col_tys)?;
    Some(reassemble_ctor(name, arity, witness))
}

fn specialize_query_tuple(
    matrix: &[Vec<CovPat>],
    sub: &[CovPat],
    rest: &[CovPat],
    col_tys: &[Type],
) -> Option<Vec<CovPat>> {
    let arity = sub.len();
    let m = specialize_tuple(matrix, arity);
    let mut new_q: Vec<CovPat> = sub.to_vec();
    new_q.extend(rest.iter().cloned());
    let new_col_tys = column_tys_for_tuple(&col_tys[0], &col_tys[1..]);
    let witness = useful(&m, &new_q, &new_col_tys)?;
    Some(reassemble_tuple(arity, witness))
}

fn specialize_query_record(
    matrix: &[Vec<CovPat>],
    fields: &[(String, CovPat)],
    rest: &[CovPat],
    col_tys: &[Type],
) -> Option<Vec<CovPat>> {
    let field_names: Vec<String> = fields.iter().map(|(n, _)| n.clone()).collect();
    let m = specialize_record(matrix, &field_names);
    let sub_pats: Vec<CovPat> = fields.iter().map(|(_, p)| p.clone()).collect();
    let mut new_q: Vec<CovPat> = sub_pats;
    new_q.extend(rest.iter().cloned());
    let new_col_tys = column_tys_for_record(&field_names, &col_tys[0], &col_tys[1..]);
    let witness = useful(&m, &new_q, &new_col_tys)?;
    Some(reassemble_record(&field_names, witness))
}

fn specialize_query_unit(matrix: &[Vec<CovPat>], rest: &[CovPat], col_tys: &[Type]) -> Option<Vec<CovPat>> {
    let m = specialize_unit(matrix);
    let witness = useful(&m, rest, &col_tys[1..])?;
    let mut out = vec![CovPat::UnitP];
    out.extend(witness);
    Some(out)
}

fn specialize_query_lit(
    matrix: &[Vec<CovPat>],
    lit: &CovLit,
    rest: &[CovPat],
    col_tys: &[Type],
) -> Option<Vec<CovPat>> {
    let m = specialize_lit(matrix, lit);
    let witness = useful(&m, rest, &col_tys[1..])?;
    let mut out = vec![CovPat::Lit(lit.clone())];
    out.extend(witness);
    Some(out)
}

fn useful_wildcard(matrix: &[Vec<CovPat>], rest: &[CovPat], col_tys: &[Type]) -> Option<Vec<CovPat>> {
    let universe = Universe::of(&col_tys[0]);
    let heads = collect_heads(matrix);

    match &universe {
        Universe::Sum(variants) => {
            let all_covered = variants.iter().all(|(n, _)| heads.has_ctor(n));
            if all_covered {
                // Try each constructor specialization.
                for (name, payload) in variants {
                    let arity = payload.len();
                    let m = specialize_ctor(matrix, name, arity);
                    let mut new_q: Vec<CovPat> = (0..arity).map(|_| CovPat::Wild).collect();
                    new_q.extend(rest.iter().cloned());
                    let mut new_col_tys: Vec<Type> = payload.clone();
                    new_col_tys.extend(col_tys[1..].iter().cloned());
                    if let Some(w) = useful(&m, &new_q, &new_col_tys) {
                        return Some(reassemble_ctor(name, arity, w));
                    }
                }
                None
            } else {
                // A missing constructor is the witness head.
                let witness = useful(&default_matrix(matrix), rest, &col_tys[1..])?;
                let missing_name = variants
                    .iter()
                    .find(|(n, _)| !heads.has_ctor(n))
                    .map(|(n, payload)| (n.clone(), payload.len()))
                    .expect("at least one variant missing");
                let mut out = vec![CovPat::Ctor(
                    missing_name.0,
                    (0..missing_name.1).map(|_| CovPat::Wild).collect(),
                )];
                out.extend(witness);
                Some(out)
            }
        }
        Universe::Bool => {
            let true_covered = heads.has_lit(&CovLit::Bool(true));
            let false_covered = heads.has_lit(&CovLit::Bool(false));
            if true_covered && false_covered {
                for b in [true, false] {
                    let lit = CovLit::Bool(b);
                    let m = specialize_lit(matrix, &lit);
                    if let Some(w) = useful(&m, rest, &col_tys[1..]) {
                        let mut out = vec![CovPat::Lit(lit)];
                        out.extend(w);
                        return Some(out);
                    }
                }
                None
            } else {
                let witness = useful(&default_matrix(matrix), rest, &col_tys[1..])?;
                let missing = if !true_covered { true } else { false };
                let mut out = vec![CovPat::Lit(CovLit::Bool(missing))];
                out.extend(witness);
                Some(out)
            }
        }
        Universe::Unit => {
            // Single inhabitant; if any row's head is UnitP or Wild,
            // it's covered.
            if heads.has_unit_or_wild() {
                let m = specialize_unit(matrix);
                let witness = useful(&m, rest, &col_tys[1..])?;
                let mut out = vec![CovPat::UnitP];
                out.extend(witness);
                Some(out)
            } else {
                let witness = useful(&default_matrix(matrix), rest, &col_tys[1..])?;
                let mut out = vec![CovPat::UnitP];
                out.extend(witness);
                Some(out)
            }
        }
        Universe::Tuple(elem_tys) => {
            let arity = elem_tys.len();
            let m = specialize_tuple(matrix, arity);
            let mut new_q: Vec<CovPat> = (0..arity).map(|_| CovPat::Wild).collect();
            new_q.extend(rest.iter().cloned());
            let mut new_col_tys: Vec<Type> = elem_tys.clone();
            new_col_tys.extend(col_tys[1..].iter().cloned());
            let witness = useful(&m, &new_q, &new_col_tys)?;
            Some(reassemble_tuple(arity, witness))
        }
        Universe::Record(fields) => {
            let field_names: Vec<String> = fields.iter().map(|(n, _)| n.clone()).collect();
            let m = specialize_record(matrix, &field_names);
            let sub_wilds: Vec<CovPat> = (0..field_names.len()).map(|_| CovPat::Wild).collect();
            let mut new_q: Vec<CovPat> = sub_wilds;
            new_q.extend(rest.iter().cloned());
            let mut new_col_tys: Vec<Type> = fields.iter().map(|(_, t)| t.clone()).collect();
            new_col_tys.extend(col_tys[1..].iter().cloned());
            let witness = useful(&m, &new_q, &new_col_tys)?;
            Some(reassemble_record(&field_names, witness))
        }
        Universe::IntLike | Universe::FloatLike | Universe::Opaque => {
            // Infinite universe: useful iff D(M) is useful on the rest.
            // Witness gets Wild (any not-yet-covered value).
            let witness = useful(&default_matrix(matrix), rest, &col_tys[1..])?;
            let mut out = vec![CovPat::Wild];
            out.extend(witness);
            Some(out)
        }
    }
}

fn column_tys_for_ctor(name: &str, head_ty: &Type, rest: &[Type]) -> Vec<Type> {
    if let Type::Constructed(TypeName::Sum(variants), _) = head_ty {
        if let Some((_, payload)) = variants.iter().find(|(n, _)| n == name) {
            let mut out = payload.clone();
            out.extend(rest.iter().cloned());
            return out;
        }
    }
    // Fallback: caller violated invariants. Fill with the head type to
    // avoid panicking — useful() will likely fail downstream.
    let mut out = vec![head_ty.clone()];
    out.extend(rest.iter().cloned());
    out
}

fn column_tys_for_tuple(head_ty: &Type, rest: &[Type]) -> Vec<Type> {
    if let Type::Constructed(TypeName::Tuple(_), args) = head_ty {
        let mut out = args.clone();
        out.extend(rest.iter().cloned());
        return out;
    }
    let mut out = vec![head_ty.clone()];
    out.extend(rest.iter().cloned());
    out
}

fn column_tys_for_record(field_names: &[String], head_ty: &Type, rest: &[Type]) -> Vec<Type> {
    if let Type::Constructed(TypeName::Record(record_names), args) = head_ty {
        let mut out: Vec<Type> = field_names
            .iter()
            .map(|n| {
                record_names
                    .iter()
                    .position(|rn| rn == n)
                    .map(|i| args[i].clone())
                    .unwrap_or_else(|| head_ty.clone())
            })
            .collect();
        out.extend(rest.iter().cloned());
        return out;
    }
    let mut out = vec![head_ty.clone()];
    out.extend(rest.iter().cloned());
    out
}

fn reassemble_ctor(name: &str, arity: usize, witness: Vec<CovPat>) -> Vec<CovPat> {
    let (head_parts, tail) = witness.split_at(arity);
    let mut out = vec![CovPat::Ctor(name.to_string(), head_parts.to_vec())];
    out.extend(tail.iter().cloned());
    out
}

fn reassemble_tuple(arity: usize, witness: Vec<CovPat>) -> Vec<CovPat> {
    let (head_parts, tail) = witness.split_at(arity);
    let mut out = vec![CovPat::Tuple(head_parts.to_vec())];
    out.extend(tail.iter().cloned());
    out
}

fn reassemble_record(field_names: &[String], witness: Vec<CovPat>) -> Vec<CovPat> {
    let arity = field_names.len();
    let (head_parts, tail) = witness.split_at(arity);
    let head = CovPat::Record(field_names.iter().cloned().zip(head_parts.iter().cloned()).collect());
    let mut out = vec![head];
    out.extend(tail.iter().cloned());
    out
}

/// Quick lookup table for which head shapes appear in the first column.
struct Heads {
    ctors: LookupSet<String>,
    lits: LookupSet<CovLit>,
    has_unit: bool,
    has_wild: bool,
}

impl Heads {
    fn has_ctor(&self, name: &str) -> bool {
        self.ctors.contains(name)
    }
    fn has_lit(&self, lit: &CovLit) -> bool {
        self.lits.contains(lit)
    }
    fn has_unit_or_wild(&self) -> bool {
        self.has_unit || self.has_wild
    }
}

fn collect_heads(matrix: &[Vec<CovPat>]) -> Heads {
    let mut h = Heads {
        ctors: Default::default(),
        lits: Default::default(),
        has_unit: false,
        has_wild: false,
    };
    for row in matrix {
        match &row[0] {
            CovPat::Ctor(name, _) => {
                h.ctors.insert(name.clone());
            }
            CovPat::Lit(lit) => {
                h.lits.insert(lit.clone());
            }
            CovPat::UnitP => h.has_unit = true,
            CovPat::Wild => h.has_wild = true,
            // Tuple and Record patterns at the head don't refine
            // coverage at this position — the head universe is "the
            // tuple shape" or "the record shape" and a single
            // structural pattern always specializes.
            CovPat::Tuple(_) | CovPat::Record(_) => h.has_wild = true,
        }
    }
    h
}

/// Check a complete `match` for exhaustiveness and arm reachability.
///
/// `arms` are paired with spans for error reporting.
pub fn check_match(
    scrutinee_ty: &Type,
    arms: &[(ast::Pattern, Span)],
    match_span: Span,
) -> Result<(), CoverageError> {
    // Empty match is special-cased upstream; we don't reach here.
    debug_assert!(!arms.is_empty(), "check_match called with no arms");

    let cov_pats: Vec<CovPat> = arms.iter().map(|(p, _)| lower(p)).collect();
    let col_tys = vec![scrutinee_ty.clone()];

    // Redundancy: walk arm-by-arm. Arm i must be useful against the
    // matrix of arms 0..i.
    let mut matrix: Vec<Vec<CovPat>> = Vec::new();
    for (idx, pat) in cov_pats.iter().enumerate() {
        let query = vec![pat.clone()];
        if useful(&matrix, &query, &col_tys).is_none() {
            return Err(CoverageError::Redundant {
                arm_index: idx,
                arm_span: arms[idx].1,
            });
        }
        matrix.push(query);
    }

    // Exhaustiveness: wildcard query must NOT be useful against the
    // full matrix.
    let wild_query = vec![CovPat::Wild];
    if let Some(witness) = useful(&matrix, &wild_query, &col_tys) {
        return Err(CoverageError::NonExhaustive {
            missing: witness.into_iter().next().unwrap_or(CovPat::Wild),
            match_span,
        });
    }

    Ok(())
}

#[cfg(test)]
#[path = "coverage_tests.rs"]
mod coverage_tests;
