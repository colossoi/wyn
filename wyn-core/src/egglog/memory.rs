//! Structural operations derived from the memory IR declarations.
#[cfg(debug_assertions)]
use super::query::Query;
use super::{output_error, parse_program, OptimizeError};
#[cfg(debug_assertions)]
use crate::{LookupMap, LookupSet};
use egglog_engine::ast::{Command, Subdatatypes, Variant};
use egglog_engine::EGraph;
use std::collections::{BTreeMap, BTreeSet};

#[path = "memory/container_reads.rs"]
mod container_reads;

const SCHEMA: &str = include_str!("memory/schema.egg");
const NAMES: [(&str, &str, &str); 4] = [
    ("ValueName", "ValueId", "ValueNameRef"),
    ("BlockName", "MemoryId", "BlockNameRef"),
    ("AccName", "AccumulatorId", "AccNameRef"),
    ("StatementId", "StatementTag", "StatementRef"),
];

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog schema and rules");
    super::timing::evaluate(graph, Some("memory IR".into()), SCHEMA)?;
    super::timing::evaluate(
        graph,
        Some("analysis slot types".into()),
        include_str!("memory/slot-kinds.egg"),
    )?;
    let (declarations, rules) = structure()?;
    super::timing::evaluate(graph, Some("memory structure types".into()), &declarations)?;
    super::timing::evaluate(
        graph,
        Some("analysis collections".into()),
        include_str!("memory/collections.egg"),
    )?;
    container_reads::register(graph);
    super::timing::evaluate(graph, Some("memory structure".into()), &rules)?;
    super::timing::evaluate(
        graph,
        Some("memory effects".into()),
        include_str!("memory/effects.egg"),
    )?;
    super::timing::evaluate(
        graph,
        Some("memory scopes".into()),
        include_str!("memory/regions.egg"),
    )?;
    #[cfg(debug_assertions)]
    super::timing::evaluate(
        graph,
        Some("memory validation".into()),
        include_str!("memory/validate.egg"),
    )?;
    super::timing::evaluate(
        graph,
        Some("logical analysis".into()),
        concat!(
            include_str!("memory/analysis.egg"),
            "\n",
            include_str!("memory/work.egg")
        ),
    )?;
    super::timing::evaluate(
        graph,
        Some("memory fusion".into()),
        concat!(
            include_str!("memory/rewrites.egg"),
            "\n",
            include_str!("fusion/schema.egg"),
            "\n",
            include_str!("fusion/import.egg"),
            "\n",
            include_str!("fusion/compaction.egg"),
            "\n",
            include_str!("fusion/compact-fold.egg"),
            "\n",
            include_str!("fusion/fusion.egg"),
        ),
    )?;
    for pass in [
        "",
        "inline",
        "simplify",
        "dead-collections",
        "compact-count",
        "hoist",
        "conditional",
        "fusion",
        "distribution",
        "placement",
    ] {
        let (ruleset, key) = if pass.is_empty() {
            ("memory-rewrite".to_owned(), "pass".to_owned())
        } else {
            (format!("memory-rewrite-{pass}"), format!("\"{pass}\""))
        };
        let rules = include_str!("memory/apply-rewrites.egg")
            .replace("REWRITE_RULESET", &ruleset)
            .replace("REWRITE_PASS", &key);
        super::timing::evaluate(graph, Some("rewrite application".into()), &rules)?;
    }
    super::timing::evaluate(
        graph,
        Some("region inlining".into()),
        include_str!("memory/inline.egg"),
    )?;
    super::timing::evaluate(
        graph,
        Some("region simplification".into()),
        concat!(
            include_str!("memory/simplify.egg"),
            "\n",
            include_str!("memory/compact-lengths.egg")
        ),
    )?;
    super::scalar::load(graph)?;
    super::planning::load(graph)?;
    Ok(())
}

/// Inline structured helpers by substituting complete regions and annotations.
pub(super) fn inline(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog inlining");
    validate(graph)?;
    super::timing::run(
        graph,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-origin inline-discover)
            (saturate inline-prepare memory-rename)
            (saturate inline-select)
            (saturate memory-rewrite-inline memory-rename))))",
    )?;
    validate(graph)
}

pub(super) fn simplify(graph: &mut EGraph, hoist: bool) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog simplification");
    validate(graph)?;
    super::timing::run(
        graph,
        &format!("(run-schedule (saturate {}))", simplification_schedule(hoist)),
    )?;
    validate(graph)
}

pub(super) fn simplification_schedule(hoist: bool) -> String {
    let hoists = if hoist {
        "(saturate conditional-analysis distribution-analysis fusion-domains memory-rename)
         (saturate conditional-select)
         (saturate memory-rewrite-conditional memory-rename)
         (saturate simplify-hoists)"
    } else {
        ""
    };
    let apply_hoists = if hoist { "(saturate memory-rewrite-hoist memory-rename)" } else { "" };
    format!(
        "(seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes)
            (saturate schedule-cost memory-origin)
            (saturate memory-effects simplify-analysis scalar-analysis compact-length-analysis)
            (saturate simplify-uses)
            (saturate simplify-result-analysis)
            (saturate simplify-result-select memory-edits)
            (saturate memory-rewrite-dead-collections memory-rename)
            (saturate scalar-copies)
            (saturate simplify-copies simplify-removals scalar-candidates)
            (saturate compact-length-select memory-edits)
            (saturate memory-rewrite-compact-count memory-rename)
            {hoists}
            (saturate simplify-select scalar-select inline-select memory-edits)
            (saturate memory-local-rewrites)
            (saturate memory-edits)
            (saturate memory-local-rewrite-select)
            (saturate memory-rewrite-simplify memory-rename)
            (saturate memory-rewrite-inline memory-rename)
            {apply_hoists})"
    )
}

#[cfg(not(debug_assertions))]
pub(super) fn validate(_graph: &mut EGraph) -> Result<(), OptimizeError> {
    Ok(())
}

/// Audit compiler invariants in debug builds; optimization schedules derive their own facts.
#[cfg(debug_assertions)]
pub(super) fn validate(graph: &mut EGraph) -> Result<(), OptimizeError> {
    super::timing::time("memory scope analysis", || {
        super::timing::evaluate(
            graph,
            None,
            "(run-schedule (saturate fusion-index memory-names) (saturate memory-scopes))",
        )
    })?;
    #[cfg(debug_assertions)]
    validate_scopes(graph)?;
    Ok(())
}

#[cfg(debug_assertions)]
fn validate_scopes(graph: &EGraph) -> Result<(), OptimizeError> {
    let _readout = super::timing::span("memory validation readout");
    let query = Query(graph);
    if query.row("CurrentProgram", |_| true)?.is_none() {
        return Err(output_error("memory program has no current root"));
    }
    let mut programs = LookupSet::default();
    query.for_each("CurrentProgram", |row| {
        programs.insert(row[0]);
        Ok(())
    })?;
    let mut functions = LookupSet::default();
    query.for_each("ProgramFunction", |row| {
        if programs.contains(&row[0]) {
            functions.insert(row[1]);
        }
        Ok(())
    })?;
    query.for_each("ProgramExternal", |row| {
        if programs.contains(&row[0]) {
            let Some(context) = query.constructor("ExternalScope", (row[1],))? else {
                return Err(output_error("external signature was not checked"));
            };
            functions.insert(query.required("ScopeRoot", (context,))?);
        }
        Ok(())
    })?;
    for &function in &functions {
        query.required("NamesFunction", (function,))?;
    }
    let mut roots = LookupMap::default();
    query.for_function("ScopeRoot", |row, root| {
        if functions.contains(&root) {
            roots.insert(row[0], root);
        }
        Ok(())
    })?;
    let mut operands = LookupSet::default();
    query.for_function("OperandSlot", |row, _| {
        if functions.contains(&row[0]) {
            operands.insert((row[0], row[1]));
        }
        Ok(())
    })?;
    query.for_each("InvalidFunction", |invalid| {
        if functions.contains(&invalid[0]) {
            return Err(output_error("memory function contains duplicate binders"));
        }
        Ok(())
    })?;
    query.for_each("CheckSlot", |slot| {
        if let Some(&function) = roots.get(&slot[0]) {
            if !operands.contains(&(function, slot[2])) {
                return Err(output_error("memory operand has no declared type"));
            }
        }
        Ok(())
    })?;
    query.for_each("InvalidScope", |invalid| {
        if roots.contains_key(&invalid[0]) {
            let message = graph.value_to_base::<egglog_engine::sort::S>(invalid[1]);
            return Err(output_error(format!(
                "invalid memory scope: {}",
                message.as_str()
            )));
        }
        Ok(())
    })
}

enum Declaration {
    Variants(Vec<Variant>),
    Vector(String),
}

/// Generate syntax traversals once from declarations, never from program data.
/// Renaming stays in egglog and covers annotations whenever the schema grows.
fn structure() -> Result<(String, String), OptimizeError> {
    let mut declarations = BTreeMap::new();
    for command in parse_program("memory IR", SCHEMA)? {
        match command {
            Command::Datatype { name, variants, .. } => {
                declarations.insert(name, Declaration::Variants(variants));
            }
            Command::Sort {
                name,
                presort_and_args: Some((sort, args)),
                ..
            } => {
                declarations.insert(name, vector(&sort, &args)?);
            }
            Command::Datatypes { datatypes, .. } => {
                for (_, name, datatype) in datatypes {
                    declarations.insert(
                        name,
                        match datatype {
                            Subdatatypes::Variants(variants) => Declaration::Variants(variants),
                            Subdatatypes::NewSort(sort, args) => vector(&sort, &args)?,
                        },
                    );
                }
            }
            _ => return Err(output_error("memory schema must contain only data declarations")),
        }
    }
    // Only schema types containing identifiers can change during substitution.
    let mut renamable: BTreeSet<_> = NAMES.iter().map(|(name, _, _)| *name).collect();
    loop {
        let previous = renamable.len();
        for (sort, declaration) in &declarations {
            let changes = match declaration {
                Declaration::Vector(element) => renamable.contains(element.as_str()),
                Declaration::Variants(variants) => {
                    variants.iter().flat_map(|v| &v.types).any(|child| renamable.contains(child.as_str()))
                }
            };
            if changes {
                renamable.insert(sort.as_str());
            }
        }
        if renamable.len() == previous {
            break;
        }
    }
    let mut rules = String::from(
        "(datatype SyntaxRoot (ProgramRoot Program) (FunctionRoot Function) (ParallelRoot ParallelBody) (LambdaRoot Lambda) (ReturnRoot ReturnInfos))\n\
         (datatype Name (ValueNameRef ValueName) (BlockNameRef BlockName) (AccNameRef AccName))\n\
         (sort Names (Set Name))\n\
         (datatype Access (MemoryRead BlockName) (MemoryWrite BlockName) (SharedRead) (SharedWrite) (IndexedRead ValueName Indices) (OrderedEffect))\n\
         (sort Accesses (Set Access))\n(ruleset memory-effects)\n\
         (function ArrayReads (Function ValueName) Accesses :merge (set-union old new))\n\
         (function ArrayWrites (Function ValueName) Accesses :merge (set-union old new))\n\
         (function TokenWrites (Function AccName) Accesses :merge (set-union old new))\n\
         (function ArrayElement (ValueInfo) ElementType :no-merge)\n\
         (relation DirectArrayAccess (Exp ValueName))\n\
         (function WritesResult (Exp) bool :no-merge)\n\
         (function DirectNamesExp (Exp) Names :merge (set-union old new))\n\
         (function DestinationEffects (Function Pattern) Accesses :merge (set-union old new))\n\
         (function ResultEffects (Function Exp Pattern) Accesses :merge (set-union old new))\n\
         (relation RepeatableOpcode (String))\n\
         (function PrimitiveEffects (String ValueInfo) Accesses :no-merge)\n\
         (function PrimitiveResultType (Function Exp) ValueInfo :no-merge)\n\
         (function CallEffects (Function FunctionName) Accesses :merge (set-union old new))\n\
         (function LoadBoundsEffects (Function Exp) Accesses :merge (set-union old new))\n\
         (function IndexDivisionEffects (Function IndexExp) Accesses :merge (set-union old new))\n\
         (relation EffectContext (Function))\n(ruleset memory-names)\n(ruleset memory-rename)\n",
    );
    for (sort, _, _) in NAMES {
        rules.push_str(&format!("(sort {sort}Renaming (Map {sort} {sort}))\n"));
    }
    rules.push_str("(sort StatementReplacements (Map StatementId Stms))\n\
        (datatype Renaming (Rename ValueNameRenaming BlockNameRenaming AccNameRenaming StatementIdRenaming StatementReplacements))\n");
    for sort in declarations.keys() {
        rules.push_str(&format!(
            "(relation Contains{sort} (SyntaxRoot {sort}))\n\
             (function Effects{sort} (Function {sort}) Accesses :merge (set-union old new))\n\
             (relation Effects{sort}Demand (Function {sort}))\n\
             (function Names{sort} ({sort}) Names :merge (set-union old new))\n\
             (function Binders{sort} ({sort}) Names :merge (set-union old new))\n\
             (relation Rename{sort}Demand (Renaming {sort}))\n\
             (function Rename{sort} (Renaming {sort}) {sort} :no-merge)\n",
        ));
        if matches!(declarations[sort], Declaration::Vector(_)) {
            rules.push_str(&format!("(relation Walk{sort} ({sort} i64))\n"));
            let Declaration::Vector(element) = &declarations[sort] else {
                unreachable!()
            };
            rules.push_str(&format!(
                "(function {sort}At ({sort} i64) {element} :no-merge)\n\
                 (function {sort}RangeMid ({sort} i64 i64) i64 :no-merge)\n"
            ));
        }
    }
    let types = std::mem::take(&mut rules);
    for (sort, declaration) in &declarations {
        if let Some(index) = NAMES.iter().position(|(name, _, _)| name == sort) {
            let names = if index < 3 {
                format!("(set-of ({} node))", NAMES[index].2)
            } else {
                "(set-empty)".into()
            };
            let Declaration::Variants(variants) = declaration else {
                return Err(output_error("name sorts must be datatypes"));
            };
            for variant in variants {
                let arguments =
                    (0..variant.types.len()).map(|i| format!("a{i}")).collect::<Vec<_>>().join(" ");
                let term = format!("({} {arguments})", variant.name);
                rules.push_str(&format!(
                    "(rule ((= node {term})) ((set (Names{sort} node) {names}) (set (Binders{sort} node) (set-empty))) :ruleset memory-names)\n\
                     (rule ((Effects{sort}Demand function node) (= node {term})) ((set (Effects{sort} function node) (set-empty))) :ruleset memory-effects)\n"
                ));
            }
            rules.push_str(&format!(
                "(rule ((Rename{sort}Demand env node) (= env (Rename m0 m1 m2 m3 edits)) (map-excludes m{index} node)) ((set (Rename{sort} env node) node)) :ruleset memory-rename)\n\
                 (rule ((Rename{sort}Demand env node) (= env (Rename m0 m1 m2 m3 edits)) (= next (map-at m{index} node))) ((set (Rename{sort} env node) next)) :ruleset memory-rename)\n"
            ));
            continue;
        }
        match declaration {
            Declaration::Vector(element) => {
                if matches!(declarations.get(element), Some(Declaration::Vector(_))) {
                    rules.push_str(&format!("(rule ((Walk{sort} node i) (< i (vec-size node))) ((Walk{element} (vec-at node i) 0)) :ruleset memory-names)\n"));
                }
                if declarations.contains_key(element) {
                    rules.push_str(&format!("(rule ((Contains{sort} root node) (Walk{sort} node i) (< i (vec-size node))) ((Contains{element} root (vec-at node i))) :ruleset memory-names)\n"));
                }
                vector_rules(&mut rules, sort, element, declarations.contains_key(element));
            }
            Declaration::Variants(variants) => {
                for variant in variants {
                    variant_rules(&mut rules, sort, variant, &declarations, &renamable);
                    effect_rules(&mut rules, sort, variant, &declarations)?;
                }
            }
        }
    }
    Ok((types, rules))
}

fn vector(sort: &str, args: &[egglog_engine::ast::Expr]) -> Result<Declaration, OptimizeError> {
    let [element] = args else {
        return Err(output_error("memory container requires one element sort"));
    };
    if sort != "Vec" {
        return Err(output_error("unsupported memory schema container"));
    }
    Ok(Declaration::Vector(element.to_string()))
}

fn vector_rules(rules: &mut String, sort: &str, element: &str, structured: bool) {
    // Materialize the vector/position key before joining an element's metadata.
    // Primitive reads in rule bodies are evaluated after the relational join.
    for pass in ["memory-names", "memory-rename", "memory-effects"] {
        rules.push_str(&format!(
            "(rule ((Walk{sort} node i) (< i (vec-size node)))\n\
             ((set ({sort}At node i) (vec-at node i)) (Walk{sort} node (+ i 1))) :ruleset {pass})\n"
        ));
    }
    let (effect_premise, effects) = if structured {
        (
            format!("(= effects (Effects{element} function ({sort}At node i)))"),
            "effects",
        )
    } else {
        (String::new(), "(set-empty)")
    };
    rules.push_str(&format!(
        "(relation Effects{sort}RangeDemand (Function {sort} i64 i64))\n\
         (function Effects{sort}Range (Function {sort} i64 i64) Accesses :merge (set-union old new))\n\
         (rule ((Effects{sort}Demand function node)) ((Walk{sort} node 0) (Effects{sort}RangeDemand function node 0 (vec-size node))) :ruleset memory-effects)\n\
         (rule ((Effects{sort}RangeDemand function node i i)) ((set (Effects{sort}Range function node i i) (set-empty))) :ruleset memory-effects)\n\
         (rule ((Effects{sort}RangeDemand function node i j) (> (- j i) 1) (= mid (+ i (/ (- j i) 2))))\n\
             ((set ({sort}RangeMid node i j) mid) (Effects{sort}RangeDemand function node i mid) (Effects{sort}RangeDemand function node mid j)) :ruleset memory-effects)\n\
         (rule ((Effects{sort}RangeDemand function node i j) (> (- j i) 1) (= mid ({sort}RangeMid node i j))\n\
             (= left (Effects{sort}Range function node i mid)) (= right (Effects{sort}Range function node mid j)))\n\
             ((set (Effects{sort}Range function node i j) (set-union left right))) :ruleset memory-effects)\n\
         (rule ((Effects{sort}RangeDemand function node i j) (= j (+ i 1)) {effect_premise}) ((set (Effects{sort}Range function node i j) {effects})) :ruleset memory-effects)\n\
         (rule ((= effects (Effects{sort}Range function node 0 (vec-size node)))) ((set (Effects{sort} function node) effects)) :ruleset memory-effects)\n"
    ));
    if structured {
        rules.push_str(&format!("(rule ((Effects{sort}RangeDemand function node i j) (= j (+ i 1))) ((Effects{element}Demand function (vec-at node i))) :ruleset memory-effects)\n"));
    }
    let (uses, bound, names, demand, rename, child) = if structured {
        (
            "uses",
            "bound",
            format!("(= uses (Names{element} child)) (= bound (Binders{element} child))"),
            format!("(Rename{element}Demand env (vec-at node i))"),
            format!("(= child (Rename{element} env ({sort}At node i)))"),
            "child",
        )
    } else {
        (
            "(set-empty)",
            "(set-empty)",
            String::new(),
            String::new(),
            String::new(),
            "(vec-at node i)",
        )
    };
    let unchanged = if sort == "Stms" {
        "(= env (Rename m0 m1 m2 m3 edits)) (= (StmsAt node i) (Let id pattern certs expression)) (map-excludes edits id)"
    } else {
        ""
    };
    rules.push_str(&format!(
        "(rule ((= node (vec-of))) ((set (Names{sort} node) (set-empty)) (set (Binders{sort} node) (set-empty))) :ruleset memory-names)\n\
         (rule ((Walk{sort} node i) (< i (vec-size node))) ((Walk{sort} node (+ i 1))) :ruleset memory-names)\n\
         (relation Names{sort}RangeDemand ({sort} i64 i64))\n\
         (function Names{sort}Range ({sort} i64 i64) Names :merge (set-union old new))\n\
         (function Binders{sort}Range ({sort} i64 i64) Names :merge (set-union old new))\n\
         (rule ((Walk{sort} node 0)) ((Names{sort}RangeDemand node 0 (vec-size node))) :ruleset memory-names)\n\
         (rule ((Names{sort}RangeDemand node i i)) ((set (Names{sort}Range node i i) (set-empty)) (set (Binders{sort}Range node i i) (set-empty))) :ruleset memory-names)\n\
         (rule ((Names{sort}RangeDemand node i j) (> (- j i) 1) (= mid (+ i (/ (- j i) 2))))\n\
             ((set ({sort}RangeMid node i j) mid) (Names{sort}RangeDemand node i mid) (Names{sort}RangeDemand node mid j)) :ruleset memory-names)\n\
         (rule ((Names{sort}RangeDemand node i j) (> (- j i) 1) (= mid ({sort}RangeMid node i j))\n\
             (= left (Names{sort}Range node i mid)) (= right (Names{sort}Range node mid j))\n\
             (= bound-left (Binders{sort}Range node i mid)) (= bound-right (Binders{sort}Range node mid j)))\n\
             ((set (Names{sort}Range node i j) (set-union left right))\n\
              (set (Binders{sort}Range node i j) (set-union bound-left bound-right))) :ruleset memory-names)\n\
         (rule ((Names{sort}RangeDemand node i j) (= j (+ i 1)) (= child ({sort}At node i)) {names})\n\
             ((set (Names{sort}Range node i j) {uses}) (set (Binders{sort}Range node i j) {bound})) :ruleset memory-names)\n\
         (rule ((= names (Names{sort}Range node 0 (vec-size node))) (= bound (Binders{sort}Range node 0 (vec-size node))))\n\
             ((set (Names{sort} node) names) (set (Binders{sort} node) bound)) :ruleset memory-names)\n\
         (relation Rename{sort}RangeDemand (Renaming {sort} i64 i64))\n\
         (function Rename{sort}Range (Renaming {sort} i64 i64) {sort} :no-merge)\n\
         (rule ((Rename{sort}Demand env node)) ((Walk{sort} node 0) (Rename{sort}RangeDemand env node 0 (vec-size node))) :ruleset memory-rename)\n\
         (rule ((Rename{sort}RangeDemand env node i i)) ((set (Rename{sort}Range env node i i) (vec-of))) :ruleset memory-rename)\n\
         (rule ((Rename{sort}RangeDemand env node i j) (> (- j i) 1) (= mid (+ i (/ (- j i) 2))))\n\
             ((set ({sort}RangeMid node i j) mid) (Rename{sort}RangeDemand env node i mid) (Rename{sort}RangeDemand env node mid j)) :ruleset memory-rename)\n\
         (rule ((Rename{sort}RangeDemand env node i j) (> (- j i) 1) (= mid ({sort}RangeMid node i j))\n\
             (= left (Rename{sort}Range env node i mid)) (= right (Rename{sort}Range env node mid j)))\n\
             ((set (Rename{sort}Range env node i j) (vec-append left right))) :ruleset memory-rename)\n\
         (rule ((Rename{sort}RangeDemand env node i j) (= j (+ i 1)) {unchanged}) ({demand}) :ruleset memory-rename)\n\
         (rule ((Rename{sort}RangeDemand env node i j) (= j (+ i 1)) {unchanged} {rename}) ((set (Rename{sort}Range env node i j) (vec-of {child}))) :ruleset memory-rename)\n\
         (rule ((= result (Rename{sort}Range env node 0 (vec-size node)))) ((set (Rename{sort} env node) result)) :ruleset memory-rename)\n",
    ));
    if sort == "Stms" {
        // Replacements are already constructed in their destination scope. Do not
        // revisit them: a replacement may intentionally retain its statement ID.
        rules.push_str(
            "(rule ((RenameStmsRangeDemand env node i j) (= j (+ i 1))\n\
            (= env (Rename m0 m1 m2 m3 edits)) (= (StmsAt node i) (Let id pattern certs expression))\n\
            (= replacement (map-at edits id)))\n\
            ((set (RenameStmsRange env node i j) replacement)) :ruleset memory-rename)\n",
        );
    }
}

fn variant_rules(
    rules: &mut String,
    sort: &str,
    variant: &Variant,
    declarations: &BTreeMap<String, Declaration>,
    renamable: &BTreeSet<&str>,
) {
    let arguments: Vec<_> = (0..variant.types.len()).map(|i| format!("a{i}")).collect();
    let term = format!("({} {})", variant.name, arguments.join(" "));
    let mut uses = String::from("(set-empty)");
    let mut direct_uses = String::from("(set-empty)");
    let mut bound = String::from("(set-empty)");
    let mut premises = String::new();
    let mut demands = String::new();
    let mut renamed = arguments.clone();
    let mut rename_premises = String::new();
    let mut walk = String::new();
    let mut contained = String::new();
    for (i, child) in variant.types.iter().enumerate() {
        if !declarations.contains_key(child) {
            continue;
        }
        contained.push_str(&format!(" (Contains{child} root a{i})"));
        premises.push_str(&format!(
            " (= u{i} (Names{child} a{i})) (= b{i} (Binders{child} a{i}))"
        ));
        uses = format!("(set-union {uses} u{i})");
        if !matches!(child.as_str(), "Body" | "Lambda" | "ParallelBody") {
            direct_uses = format!("(set-union {direct_uses} u{i})");
        }
        bound = format!("(set-union {bound} b{i})");
        if renamable.contains(child.as_str()) {
            demands.push_str(&format!(" (Rename{child}Demand env a{i})"));
            rename_premises.push_str(&format!(" (= r{i} (Rename{child} env a{i}))"));
            renamed[i] = format!("r{i}");
        }
        if matches!(declarations[child], Declaration::Vector(_)) {
            walk.push_str(&format!(" (Walk{child} a{i} 0)"));
        }
    }
    if !contained.is_empty() {
        rules.push_str(&format!(
            "(rule ((Contains{sort} root node) (= node {term})) ({contained}) :ruleset memory-names)\n"
        ));
    }
    if sort == "Exp" {
        rules.push_str(&format!(
            "(rule ((= node {term}) {premises}) ((set (DirectNamesExp node) {direct_uses})) :ruleset memory-names)\n"
        ));
    }
    if !walk.is_empty() {
        for pass in ["memory-names", "memory-rename", "memory-effects"] {
            rules.push_str(&format!("(rule ((= node {term})) ({walk}) :ruleset {pass})\n"));
        }
    }
    if matches!(
        variant.name.as_str(),
        "BindValue" | "BindAccumulator" | "Counted" | "DenseAxis" | "IterationSpace"
    ) {
        let leaf = &variant.types[0];
        let wrapper = NAMES.iter().find(|(name, _, _)| name == leaf).map(|(_, _, wrapper)| *wrapper);
        if let Some(wrapper) = wrapper {
            bound = format!("(set-insert {bound} ({wrapper} a0))");
        }
    }
    rules.push_str(&format!(
        "(rule ((= node {term}) {premises}) ((set (Names{sort} node) {uses}) (set (Binders{sort} node) {bound})) :ruleset memory-names)\n\
         (rule ((Rename{sort}Demand env node) (= node {term})) ({demands}) :ruleset memory-rename)\n\
         (rule ((Rename{sort}Demand env node) (= node {term}) {rename_premises}) ((set (Rename{sort} env node) ({} {}))) :ruleset memory-rename)\n",
        variant.name, renamed.join(" "),
    ));
}

/// Effects follow the same syntax as renaming. Unknown opcodes stay ordered until
/// the scalar catalogue proves them pure; new expression forms require a decision.
fn effect_rules(
    rules: &mut String,
    sort: &str,
    variant: &Variant,
    declarations: &BTreeMap<String, Declaration>,
) -> Result<(), OptimizeError> {
    let arguments: Vec<_> = (0..variant.types.len()).map(|i| format!("a{i}")).collect();
    let term = format!("({} {})", variant.name, arguments.join(" "));
    let mut premises = String::new();
    let mut demands = String::new();
    let mut effects = String::from("(set-empty)");
    for (i, child) in variant.types.iter().enumerate() {
        if declarations.contains_key(child) {
            demands.push_str(&format!(" (Effects{child}Demand function a{i})"));
            premises.push_str(&format!(" (= e{i} (Effects{child} function a{i}))"));
            effects = format!("(set-union {effects} e{i})");
        }
    }
    if !demands.is_empty() {
        rules.push_str(&format!("(rule ((Effects{sort}Demand function node) (= node {term})) ({demands}) :ruleset memory-effects)\n"));
    }
    if sort == "IndexExp" && matches!(variant.name.as_str(), "IndexDiv" | "IndexMod") {
        premises.push_str(" (= division (IndexDivisionEffects function a1))");
        effects = format!("(set-union {effects} division)");
    }
    if sort == "LoopForm" && variant.name == "While" {
        effects = format!("(set-insert {effects} (OrderedEffect))");
    }
    if sort == "ParallelResult" && variant.name == "Concatenate" {
        premises.push_str(" (= reads (ArrayReads function a1))");
        effects = format!("(set-union {effects} reads)");
    }
    if sort == "Exp" {
        for (i, child) in variant.types.iter().enumerate() {
            if child == "ValueName" {
                rules.push_str(&format!("(rule ((= expression {term})) ((DirectArrayAccess expression a{i})) :ruleset memory-names)\n"));
            }
        }
        let direct = match variant.name.as_str() {
            "Use" | "IndexOp" | "View" | "If" | "Loop" | "Parallel" | "GenerateArray" | "Product"
            | "Project" | "Length" | "Scratch" => "(set-empty)",
            "Index" => {
                premises.push_str(" (= direct (ArrayReads function a0))");
                "(set-insert direct (OrderedEffect))"
            }
            "Copy" => {
                premises.push_str(" (= direct (ArrayReads function a0))");
                "direct"
            }
            "Update" => {
                premises.push_str(" (= direct (ArrayWrites function a1))");
                "direct"
            }
            "PrimOp" => {
                premises.push_str(" (= element (PrimitiveResultType function node)) (= direct (PrimitiveEffects a0 element))");
                "direct"
            }
            "Call" => {
                premises.push_str(" (= direct (CallEffects function a0))");
                "direct"
            }
            "Accumulate" => {
                premises.push_str(" (= writes (TokenWrites function a1))");
                "writes"
            }
            "Reserve" => {
                premises.push_str(" (= writes (TokenWrites function a0))");
                "writes"
            }
            "Assert" | "WithAcc" | "BucketScatter" => "(set-of (OrderedEffect))",
            other => {
                return Err(output_error(format!(
                    "missing memory effect classification for {other}"
                )))
            }
        };
        effects = format!("(set-union {effects} {direct})");
        let writes = matches!(
            variant.name.as_str(),
            "Copy" | "GenerateArray" | "Parallel" | "Alloc" | "Scratch"
        );
        rules.push_str(&format!("(rule ((= expression {term})) ((set (WritesResult expression) {writes})) :ruleset memory-effects)\n"));
    }
    if sort == "Stm" {
        premises.push_str(" (= destinations (ResultEffects function a3 a1))");
        effects = format!("(set-union {effects} destinations)");
    }
    rules.push_str(&format!("(rule ((Effects{sort}Demand function node) (= node {term}) {premises}) ((set (Effects{sort} function node) {effects})) :ruleset memory-effects)\n"));
    Ok(())
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
