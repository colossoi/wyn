//! Structural operations derived from the memory IR declarations.
use super::{output_error, parse_program, query::Query, OptimizeError};
use egglog_engine::ast::{Command, Subdatatypes, Variant};
use egglog_engine::EGraph;
use std::collections::BTreeMap;

const SCHEMA: &str = include_str!("memory/schema.egg");
const NAMES: [(&str, &str, &str); 4] = [
    ("ValueName", "ValueId", "ValueNameRef"),
    ("BlockName", "MemoryId", "BlockNameRef"),
    ("AccName", "AccumulatorId", "AccNameRef"),
    ("StatementId", "StatementTag", "StatementRef"),
];

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("memory IR".into()), SCHEMA)?;
    graph.parse_and_run_program(Some("memory structure".into()), &structure()?)?;
    graph.parse_and_run_program(Some("memory effects".into()), include_str!("memory/effects.egg"))?;
    graph.parse_and_run_program(Some("memory scopes".into()), include_str!("memory/regions.egg"))?;
    graph.parse_and_run_program(
        Some("memory fusion".into()),
        concat!(
            include_str!("memory/rewrites.egg"),
            "\n",
            include_str!("fusion/schema.egg"),
            "\n",
            include_str!("fusion/import.egg"),
            "\n",
            include_str!("fusion/fusion.egg"),
        ),
    )?;
    graph.parse_and_run_program(Some("region inlining".into()), include_str!("memory/inline.egg"))?;
    graph.parse_and_run_program(
        Some("region simplification".into()),
        include_str!("memory/simplify.egg"),
    )?;
    super::scalar::load(graph)?;
    super::planning::load(graph)?;
    let (declarations, rules) = allocation_structure()?;
    graph.parse_and_run_program(Some("allocation traversal declarations".into()), &declarations)?;
    graph.parse_and_run_program(
        Some("allocation elaboration".into()),
        include_str!("planning/allocation.egg"),
    )?;
    graph.parse_and_run_program(Some("allocation traversal".into()), &rules)?;
    graph.parse_and_run_program(
        Some("kernel memory lowering".into()),
        include_str!("planning/kernels.egg"),
    )?;
    graph.parse_and_run_program(
        Some("coordinate fusion".into()),
        include_str!("fusion/reindex.egg"),
    )?;
    Ok(())
}

/// Inline structured helpers by substituting complete regions and annotations.
pub(super) fn inline(graph: &mut EGraph) -> Result<(), OptimizeError> {
    validate(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-origin inline-discover)
            (saturate inline-prepare memory-rename)
            (saturate inline-select)
            (saturate memory-rewrite memory-rename))))",
    )?;
    validate(graph)
}

pub(super) fn simplify(graph: &mut EGraph, hoist: bool) -> Result<(), OptimizeError> {
    // Combine uniform branches before hoisting can materialize only one arm.
    let hoists = if hoist {
        "(saturate fusion-forward dispatch-shapes local-storage-analysis distribution-analysis fusion-domains memory-rename memory-effects)
         (saturate conditional-select) (saturate memory-rewrite memory-rename)
         (saturate simplify-hoists)
         (saturate collection-effects) (saturate memory-effects)
         (saturate collection-analysis) (saturate collection-select)
         (saturate memory-rewrite memory-rename)"
    } else {
        ""
    };
    validate(graph)?;
    graph.parse_and_run_program(
        None,
        &format!(
            "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes)
            (saturate schedule-cost memory-origin)
            (saturate memory-effects simplify-analysis scalar-analysis)
            (saturate scalar-copies)
            (saturate simplify-copies simplify-removals scalar-candidates)
            {hoists}
            (saturate simplify-select scalar-select inline-select memory-edits)
            (saturate memory-rewrite memory-rename))))"
        ),
    )?;
    validate(graph)
}

/// Reject malformed lexical scopes before transformations or final emission.
pub(super) fn validate(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate fusion-index memory-names) (saturate memory-scopes))",
    )?;
    let query = Query(graph);
    if query.row("CurrentProgram", |_| true)?.is_none() {
        return Err(output_error("memory program has no current root"));
    }
    let check = |function| {
        query.required("NamesFunction", (function,))?;
        query.for_each("InvalidFunction", |invalid| {
            if invalid[0] == function {
                return Err(output_error("memory function contains duplicate binders"));
            }
            Ok(())
        })?;
        query.for_each("CheckSlot", |slot| {
            if query.lookup("ScopeRoot", (slot[0],))? == Some(function)
                && query.lookup("OperandSlot", (function, slot[2]))?.is_none()
            {
                return Err(output_error("memory operand has no declared type"));
            }
            Ok(())
        })?;
        query.for_each("InvalidScope", |invalid| {
            if query.lookup("ScopeRoot", (invalid[0],))? == Some(function) {
                let message = graph.value_to_base::<egglog_engine::sort::S>(invalid[1]);
                return Err(output_error(format!(
                    "invalid memory scope: {}",
                    message.as_str()
                )));
            }
            Ok(())
        })
    };
    query.for_each("ProgramFunction", |row| {
        if query.contains("CurrentProgram", (row[0],))? {
            check(row[1])?;
        }
        Ok(())
    })?;
    query.for_each("ProgramExternal", |row| {
        if query.contains("CurrentProgram", (row[0],))? {
            let Some(context) = query.constructor("ExternalScope", (row[1],))? else {
                return Err(output_error("external signature was not checked"));
            };
            check(query.required("ScopeRoot", (context,))?)?;
        }
        Ok(())
    })
}

/// Elaborate logical arrays after fusion. Rust only checks completion; the
/// transformation and root replacement both execute inside egglog.
pub(super) fn allocate(graph: &mut EGraph) -> Result<(), OptimizeError> {
    validate(graph)?;
    graph.parse_and_run_program(None,
        "(run-schedule
            (saturate schedule-cost placement-context) (saturate placement-scopes)
            (saturate (seq (saturate dispatch-shapes) (saturate memory-names) (saturate local-storage-analysis)))
            (saturate schedule-cost dispatch-shapes allocation-bounds)
            (saturate allocation-sizes) memory-allocation-current (saturate schedule-cost memory-origin) (saturate memory-effects)
            (saturate fusion-forward memory-reuse) (saturate memory-reuse-propose)
            (saturate memory-reuse-select) (saturate memory-allocation))
         (run-schedule memory-allocation-demand (saturate memory-elaborate))")?;
    Query(graph).for_each("CurrentProgram", |row| {
        Query(graph).required("AllocateProgram", (row[0],))?;
        Ok(())
    })?;
    graph.parse_and_run_program(None, "(run-schedule memory-allocation-commit)")?;
    validate(graph)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate schedule-cost))")?;
    let query = Query(graph);
    query.for_each("CurrentProgram", |row| {
        let Some(root) = query.constructor("ProgramRoot", (row[0],))? else {
            return Err(output_error("allocated program was not indexed"));
        };
        for (index, logical) in [
            ("ContainsValueInfo", "ArrayValue"),
            ("ContainsReturnInfo", "ReturnArrayValue"),
        ] {
            query.for_each(index, |value| {
                if value[0] == root && query.enode(logical, value[1])?.is_some() {
                    return Err(output_error("allocation left a logical array in the program"));
                }
                Ok(())
            })?;
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
fn structure() -> Result<String, OptimizeError> {
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
         (function DestinationEffects (Function Pattern) Accesses :merge (set-union old new))\n\
         (function ResultEffects (Function Exp Pattern) Accesses :merge (set-union old new))\n\
         (relation RepeatableOpcode (String))\n\
         (function PrimitiveEffects (String ElementType) Accesses :no-merge)\n\
         (function PrimitiveResultType (Function Exp) ElementType :no-merge)\n\
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
        }
    }
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
                "(rule ((Rename{sort}Demand env node) (= env (Rename m0 m1 m2 m3 edits)) (map-not-contains m{index} node)) ((set (Rename{sort} env node) node)) :ruleset memory-rename)\n\
                 (rule ((Rename{sort}Demand env node) (= env (Rename m0 m1 m2 m3 edits)) (= next (map-get m{index} node))) ((set (Rename{sort} env node) next)) :ruleset memory-rename)\n"
            ));
            continue;
        }
        match declaration {
            Declaration::Vector(element) => {
                if declarations.contains_key(element) {
                    rules.push_str(&format!("(rule ((Contains{sort} root node) (Walk{sort} node i) (< i (vec-length node))) ((Contains{element} root (vec-get node i))) :ruleset memory-names)\n"));
                }
                vector_rules(&mut rules, sort, element, declarations.contains_key(element));
            }
            Declaration::Variants(variants) => {
                for variant in variants {
                    variant_rules(&mut rules, sort, variant, &declarations);
                    effect_rules(&mut rules, sort, variant, &declarations)?;
                }
            }
        }
    }
    Ok(rules)
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
    let (effect_premise, effects) = if structured {
        (
            format!("(= effects (Effects{element} function (vec-get node i)))"),
            "effects",
        )
    } else {
        (String::new(), "(set-empty)")
    };
    rules.push_str(&format!(
        "(function Effects{sort}Prefix (Function {sort} i64) Accesses :merge (set-union old new))\n\
         (rule ((Effects{sort}Demand function node)) ((set (Effects{sort}Prefix function node 0) (set-empty))) :ruleset memory-effects)\n\
         (rule ((= prefix (Effects{sort}Prefix function node i)) (< i (vec-length node)) {effect_premise}) ((set (Effects{sort}Prefix function node (+ i 1)) (set-union prefix {effects}))) :ruleset memory-effects)\n\
         (rule ((= effects (Effects{sort}Prefix function node (vec-length node)))) ((set (Effects{sort} function node) effects)) :ruleset memory-effects)\n"
    ));
    if structured {
        rules.push_str(&format!("(rule ((= prefix (Effects{sort}Prefix function node i)) (< i (vec-length node))) ((Effects{element}Demand function (vec-get node i))) :ruleset memory-effects)\n"));
    }
    let (uses, bound, names, demand, rename, child) = if structured {
        (
            "uses",
            "bound",
            format!("(= uses (Names{element} child)) (= bound (Binders{element} child))"),
            format!("(Rename{element}Demand env (vec-get node i))"),
            format!("(= child (Rename{element} env (vec-get node i)))"),
            "child",
        )
    } else {
        (
            "(set-empty)",
            "(set-empty)",
            String::new(),
            String::new(),
            String::new(),
            "(vec-get node i)",
        )
    };
    let unchanged = if sort == "Stms" {
        "(= env (Rename m0 m1 m2 m3 edits)) (= (vec-get node i) (Let id pattern certs expression)) (map-not-contains edits id)"
    } else {
        ""
    };
    rules.push_str(&format!(
        "(rule ((= node (vec-of))) ((set (Names{sort} node) (set-empty)) (set (Binders{sort} node) (set-empty))) :ruleset memory-names)\n\
         (rule ((Walk{sort} node i) (< i (vec-length node)) (= child (vec-get node i)) {names}) ((set (Names{sort} node) {uses}) (set (Binders{sort} node) {bound}) (Walk{sort} node (+ i 1))) :ruleset memory-names)\n\
         (function Rename{sort}Prefix (Renaming {sort} i64) {sort} :no-merge)\n\
         (rule ((Rename{sort}Demand env node)) ((set (Rename{sort}Prefix env node 0) (vec-of))) :ruleset memory-rename)\n\
         (rule ((= prefix (Rename{sort}Prefix env node i)) (< i (vec-length node)) {unchanged}) ({demand}) :ruleset memory-rename)\n\
         (rule ((= prefix (Rename{sort}Prefix env node i)) (< i (vec-length node)) {unchanged} {rename}) ((set (Rename{sort}Prefix env node (+ i 1)) (vec-push prefix {child}))) :ruleset memory-rename)\n\
         (rule ((= result (Rename{sort}Prefix env node (vec-length node)))) ((set (Rename{sort} env node) result)) :ruleset memory-rename)\n",
    ));
    if sort == "Stms" {
        // Replacements are already constructed in their destination scope. Do not
        // revisit them: a replacement may intentionally retain its statement ID.
        rules.push_str("(rule ((= prefix (RenameStmsPrefix env node i)) (< i (vec-length node))\n\
            (= env (Rename m0 m1 m2 m3 edits)) (= (vec-get node i) (Let id pattern certs expression))\n\
            (= replacement (map-get edits id)))\n\
            ((set (RenameStmsPrefix env node (+ i 1)) (vec-append prefix replacement))) :ruleset memory-rename)\n");
    }
}

fn variant_rules(
    rules: &mut String,
    sort: &str,
    variant: &Variant,
    declarations: &BTreeMap<String, Declaration>,
) {
    let arguments: Vec<_> = (0..variant.types.len()).map(|i| format!("a{i}")).collect();
    let term = format!("({} {})", variant.name, arguments.join(" "));
    let mut uses = String::from("(set-empty)");
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
        bound = format!("(set-union {bound} b{i})");
        demands.push_str(&format!(" (Rename{child}Demand env a{i})"));
        rename_premises.push_str(&format!(" (= r{i} (Rename{child} env a{i}))"));
        renamed[i] = format!("r{i}");
        if matches!(declarations[child], Declaration::Vector(_)) {
            walk.push_str(&format!(" (Walk{child} a{i} 0)"));
        }
    }
    if !contained.is_empty() {
        rules.push_str(&format!(
            "(rule ((Contains{sort} root node) (= node {term})) ({contained}) :ruleset memory-names)\n"
        ));
    }
    if !walk.is_empty() {
        rules.push_str(&format!(
            "(rule ((= node {term})) ({walk}) :ruleset memory-names)\n"
        ));
    }
    if matches!(
        variant.name.as_str(),
        "BindValue"
            | "BindBlock"
            | "BindAccumulator"
            | "Counted"
            | "DenseAxis"
            | "RaggedAxis"
            | "IterationSpace"
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
            "Use" | "IndexOp" | "View" | "Reshape" | "Permute" | "If" | "Loop" | "Parallel" | "Launch"
            | "GenerateArray" | "Alloc" => "(set-empty)",
            "Load" => {
                premises.push_str(
                    " (= direct (ArrayReads function a0)) (= bounds (LoadBoundsEffects function node))",
                );
                "(set-union direct bounds)"
            }
            "Copy" => {
                premises.push_str(" (= direct (ArrayReads function a0))");
                "direct"
            }
            "Update" | "Store" => {
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
            "Assert" | "WithAcc" | "Barrier" => "(set-of (OrderedEffect))",
            other => {
                return Err(output_error(format!(
                    "missing memory effect classification for {other}"
                )))
            }
        };
        effects = format!("(set-union {effects} {direct})");
        let writes = matches!(
            variant.name.as_str(),
            "Copy" | "GenerateArray" | "Parallel" | "Alloc"
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

fn allocation_structure() -> Result<(String, String), OptimizeError> {
    let mut declarations = String::new();
    let mut rules = String::new();
    declarations.push_str(
        "(function NormalizedIndex (IndexExp) IndexExp :no-merge)\n\
        (function ConstantIndex (IndexExp) bool :no-merge)\n",
    );
    rules.push_str("(rule ((= x (IndexConstant n))) ((set (NormalizedIndex x) x) (set (ConstantIndex x) true)) :ruleset memory-allocation)\n\
        (rule ((= x (IndexValue name))) ((set (NormalizedIndex x) x) (set (ConstantIndex x) false)) :ruleset memory-allocation)\n");
    for (constructor, operator, guards) in [
        ("IndexAdd", "+", "(!= b (IndexConstant 0)) (!= a (IndexConstant 0))"),
        ("IndexSub", "-", "(!= b (IndexConstant 0))"),
        ("IndexMul", "*", "(!= a (IndexConstant 0)) (!= b (IndexConstant 0)) (!= a (IndexConstant 1)) (!= b (IndexConstant 1))"),
        ("IndexDiv", "/", "(!= b (IndexConstant 1))"),
        ("IndexMod", "%", "(!= b (IndexConstant 1))"),
        ("IndexMin", "min", ""), ("IndexMax", "max", ""),
    ] {
        let nonzero = if matches!(constructor, "IndexDiv" | "IndexMod") { "(!= y 0)" } else { "" };
        rules.push_str(&format!(
            "(rule ((= expression ({constructor} left right))) ((set (ConstantIndex expression) false)) :ruleset memory-allocation)\n\
             (rule ((= expression ({constructor} left right)) (= (NormalizedIndex left) (IndexConstant x)) (= (NormalizedIndex right) (IndexConstant y)) {nonzero}) ((set (NormalizedIndex expression) (IndexConstant ({operator} x y)))) :ruleset memory-allocation)\n\
             (rule ((= expression ({constructor} left right)) (= a (NormalizedIndex left)) (= b (NormalizedIndex right)) (= ca (ConstantIndex a)) (= cb (ConstantIndex b)) (= (and ca cb) false) {guards}) ((set (NormalizedIndex expression) ({constructor} a b))) :ruleset memory-allocation)\n"
        ));
    }
    for pattern in ["(IndexAdd a b)", "(IndexSub a b)"] {
        rules.push_str(&format!("(rule ((= expression {pattern}) (= (NormalizedIndex b) (IndexConstant 0)) (= result (NormalizedIndex a))) ((set (NormalizedIndex expression) result)) :ruleset memory-allocation)\n"));
    }
    for (pattern, side, constant, result) in [
        ("IndexAdd", "a", 0, "y"),
        ("IndexMul", "a", 0, "(IndexConstant 0)"),
        ("IndexMul", "b", 0, "(IndexConstant 0)"),
        ("IndexMul", "a", 1, "y"),
        ("IndexMul", "b", 1, "x"),
        ("IndexDiv", "b", 1, "x"),
        ("IndexMod", "b", 1, "(IndexConstant 0)"),
    ] {
        rules.push_str(&format!("(rule ((= expression ({pattern} a b)) (= x (NormalizedIndex a)) (= y (NormalizedIndex b)) (= (NormalizedIndex {side}) (IndexConstant {constant}))) ((set (NormalizedIndex expression) {result})) :ruleset memory-allocation)\n"));
    }
    for constructor in [
        "IndexAdd", "IndexSub", "IndexMul", "IndexDiv", "IndexMod", "IndexMin", "IndexMax",
    ] {
        rules.push_str(&format!(
            "(rule ((ResolveIndex function ({constructor} a b))) ((ResolveIndex function a) (ResolveIndex function b)) :ruleset memory-allocation)\n\
             (rule ((ResolveIndex function ({constructor} a b)) (= x (ResolvedIndex function a)) (= y (ResolvedIndex function b))) ((set (ResolvedIndex function ({constructor} a b)) ({constructor} x y))) :ruleset memory-allocation)\n"
        ));
    }
    for (vector, element) in [
        ("Pattern", "Binding"),
        ("Params", "Param"),
        ("Merges", "Merge"),
        ("Stms", "Stm"),
    ] {
        declarations.push_str(&format!(
            "(relation Allocate{vector}Demand (ScopeContext {vector}))\n\
             (function Allocate{vector} (ScopeContext {vector}) {vector} :no-merge)\n\
             (function Allocate{vector}Prefix (ScopeContext {vector} i64) {vector} :no-merge)\n"
        ));
        rules.push_str(&format!(
            "(rule ((Allocate{vector}Demand context input)) ((set (Allocate{vector}Prefix context input 0) (vec-of))) :ruleset memory-elaborate)\n\
             (rule ((= prefix (Allocate{vector}Prefix context input i)) (< i (vec-length input))) ((Allocate{element}Demand context (vec-get input i))) :ruleset memory-elaborate)\n\
             (rule ((= prefix (Allocate{vector}Prefix context input i)) (< i (vec-length input)) (= next (Allocate{element} context (vec-get input i)))) ((set (Allocate{vector}Prefix context input (+ i 1)) (vec-append prefix next))) :ruleset memory-elaborate)\n\
             (rule ((= output (Allocate{vector}Prefix context input (vec-length input)))) ((set (Allocate{vector} context input) output)) :ruleset memory-elaborate)\n"
        ));
    }
    for (primitive, bytes) in [
        ("I8", 1),
        ("U8", 1),
        ("I16", 2),
        ("U16", 2),
        ("F16", 2),
        ("I32", 4),
        ("U32", 4),
        ("F32", 4),
        ("Bool", 4),
        ("I64", 8),
        ("U64", 8),
        ("F64", 8),
        ("UnitType", 0),
    ] {
        rules.push_str(&format!("(set (PrimitiveBytes ({primitive})) {bytes})\n"));
    }
    for info in [
        "(ScalarInfo element)",
        "(ArrayInfo element location)",
        "(ResourceInfo resource)",
    ] {
        rules.push_str(&format!(
            "(rule ((AllocateBindingDemand context binding) (= binding (BindValue name {info}))) ((set (AllocateBinding context binding) (vec-of binding))) :ruleset memory-elaborate)\n\
             (rule ((= binding (BindValue name {info}))) ((set (BindingWidth binding) 1)) :ruleset memory-elaborate)\n"
        ));
    }
    for info in ["(ScalarInfo element)", "(ResourceInfo resource)"] {
        rules.push_str(&format!("(rule ((= (LogicalInfo function name) {info})) ((set (PackedOperand function (ValueOperand (Var name))) (vec-of (ValueOperand (Var name))))) :ruleset memory-elaborate)\n"));
    }
    for binding in ["(BindBlock block capacity)", "(BindAccumulator token info)"] {
        rules.push_str(&format!(
            "(rule ((AllocateBindingDemand context binding) (= binding {binding})) ((set (AllocateBinding context binding) (vec-of binding))) :ruleset memory-elaborate)\n\
             (rule ((= binding {binding})) ((set (BindingWidth binding) 1)) :ruleset memory-elaborate)\n\
             (rule ((= prefix (AllocationPrefix context id pattern i)) (< i (vec-length pattern)) (= (vec-get pattern i) {binding})) ((set (AllocationPrefix context id pattern (+ i 1)) prefix)) :ruleset memory-elaborate)\n"
        ));
    }
    for operand in [
        "(ValueOperand (LiteralBits primitive bits))",
        "(IndexOperand index)",
        "(BlockOperand block)",
        "(AccOperand token)",
    ] {
        rules.push_str(&format!(
            "(rule ((EffectContext function) (= operand {operand})) ((set (PackedOperand function operand) (vec-of operand))) :ruleset memory-elaborate)\n"
        ));
    }
    for info in [
        "(ReturnScalar element)",
        "(ReturnAccumulator accumulator)",
        "(ReturnResource resource)",
    ] {
        rules.push_str(&format!(
            "(rule ((= info {info})) ((set (ReturnWidth info) 1)) :ruleset memory-elaborate)\n\
             (rule ((= prefix (AllocateReturnPrefix context returns i)) (< i (vec-length returns)) (= info (vec-get returns i)) (= info {info})) ((set (AllocateReturn context returns i info) (vec-of info))) :ruleset memory-elaborate)\n"
        ));
    }
    for constructor in [
        "ExtAdd", "ExtSub", "ExtMul", "ExtDiv", "ExtMod", "ExtMin", "ExtMax",
    ] {
        rules.push_str(&format!(
            "(rule ((AllocateExtIndexDemand returns ({constructor} a b))) ((AllocateExtIndexDemand returns a) (AllocateExtIndexDemand returns b)) :ruleset memory-elaborate)\n\
             (rule ((AllocateExtIndexDemand returns original) (= original ({constructor} a b)) (= x (AllocateExtIndex returns a)) (= y (AllocateExtIndex returns b))) ((set (AllocateExtIndex returns original) ({constructor} x y))) :ruleset memory-elaborate)\n"
        ));
    }
    for result in ["(Collect values)", "(Contribute tokens)"] {
        rules.push_str(&format!(
            "(rule ((= prefix (AllocateParallelPrefix context results i)) (< i (vec-length results)) (= result (vec-get results i)) (= result {result})) ((set (AllocateParallelPrefix context results (+ i 1)) (vec-push prefix result))) :ruleset memory-elaborate)\n"
        ));
    }
    for policy in ["DisjointWrites", "IdenticalWrites", "FetchAddWrites"] {
        rules.push_str(&format!(
            "(rule ((= prefix (AllocateAccInputPrefix context id inputs i)) (< i (vec-length inputs)) (= (vec-get inputs i) (AccumulatorInput arrays ({policy})))) ((set (AllocateAccInputPrefix context id inputs (+ i 1)) (vec-push prefix (vec-get inputs i)))) :ruleset memory-elaborate)\n"
        ));
    }
    for command in parse_program("memory IR", SCHEMA)? {
        let Command::Datatypes { datatypes, .. } = command else {
            continue;
        };
        for (_, name, definition) in datatypes {
            if name != "Exp" {
                continue;
            }
            let Subdatatypes::Variants(variants) = definition else {
                return Err(output_error("Exp must be a datatype"));
            };
            for variant in variants {
                match variant.name.as_str() {
                    "Call" | "If" | "Loop" | "Parallel" | "Launch" | "WithAcc" | "Reshape" => continue,
                    "Use" | "PrimOp" | "IndexOp" | "Assert" | "Alloc" | "View" | "Permute" | "Load"
                    | "GenerateArray" | "Copy" | "Update" | "Store" | "Accumulate" | "Reserve"
                    | "Barrier" => {}
                    other => {
                        return Err(output_error(format!(
                            "missing allocation elaboration for {other}"
                        )))
                    }
                }
                let arguments =
                    (0..variant.types.len()).map(|i| format!("a{i}")).collect::<Vec<_>>().join(" ");
                rules.push_str(&format!(
                    "(rule ((AllocateExpDemand context id expression) (= expression ({} {arguments}))) ((set (AllocateExp context id expression) expression)) :ruleset memory-elaborate)\n", variant.name
                ));
            }
        }
    }
    Ok((declarations, rules))
}
