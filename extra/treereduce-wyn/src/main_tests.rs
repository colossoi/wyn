use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug)]
struct Contains(&'static [u8]);

impl Check for Contains {
    type State = bool;

    fn start(&self, source: &[u8]) -> io::Result<Self::State> {
        Ok(source.windows(self.0.len()).any(|window| window == self.0))
    }

    fn cancel(&self, _state: Self::State) -> io::Result<()> {
        Ok(())
    }

    fn try_wait(&self, state: &mut Self::State) -> io::Result<Option<bool>> {
        Ok(Some(*state))
    }

    fn wait(&self, state: Self::State) -> io::Result<bool> {
        Ok(state)
    }
}

#[derive(Clone, Debug)]
struct RecordingContains {
    needle: &'static [u8],
    accepted: AcceptedWarnings,
    checks: Arc<AtomicUsize>,
}

impl Check for RecordingContains {
    type State = Vec<u8>;

    fn start(&self, source: &[u8]) -> io::Result<Self::State> {
        self.checks.fetch_add(1, Ordering::Relaxed);
        Ok(source.to_vec())
    }

    fn cancel(&self, _state: Self::State) -> io::Result<()> {
        Ok(())
    }

    fn try_wait(&self, state: &mut Self::State) -> io::Result<Option<bool>> {
        Ok(Some(
            state.windows(self.needle.len()).any(|window| window == self.needle),
        ))
    }

    fn wait(&self, state: Self::State) -> io::Result<bool> {
        let interesting = state.windows(self.needle.len()).any(|window| window == self.needle);
        if interesting {
            self.accepted.record(state, b"")?;
        }
        Ok(interesting)
    }
}

fn language() -> Language {
    tree_sitter_wyn::LANGUAGE.into()
}

fn type_probe() -> TypeProbe {
    TypeProbe {
        wyn: std::env::current_exe().unwrap(),
        check_args: Vec::new(),
        temp_dir: None,
        inferred_type: Regex::new(r"type hole inferred as `([^`]+)`").unwrap(),
    }
}

#[test]
fn compiler_preflight_runs_help_and_requires_an_executable() {
    let mut probe = type_probe();
    probe.wyn = std::env::current_exe().unwrap();
    probe.verify_compiler().unwrap();

    let temp = tempfile::tempdir().unwrap();
    probe.wyn = temp.path().join("missing-wyn-compiler");
    let error = probe.verify_compiler().unwrap_err().to_string();
    assert!(error.contains("failed to run Wyn compiler preflight"));
}

#[test]
fn parses_unused_warning_locations_as_utf8_byte_offsets() {
    let source = "def café = 1\nentry main() i32 = 0\n";
    let diagnostics = b"warning: unused definition `caf\xC3\xA9` is not reachable from any entry point\n  --> candidate.wyn:1:5\n";
    assert_eq!(
        parse_unused_warnings(source.as_bytes(), diagnostics),
        vec![UnusedWarning {
            kind: UnusedWarningKind::Declaration,
            start: 4,
            end: 9,
        }]
    );
}

#[test]
fn warning_locations_create_semantic_cleanup_candidates() {
    let source = b"def dead = 1\nentry main(unused_parameter: i32) i32 = let unused_binding = 2 in ???\n";
    let diagnostics = b"warning: unused definition `dead` is not reachable from any entry point\n  --> candidate.wyn:1:5\nwarning: unused parameter `unused_parameter`; prefix its name with `_` to silence this warning\n  --> candidate.wyn:2:12\nwarning: unused binding `unused_binding`; prefix its name with `_` to silence this warning\n  --> candidate.wyn:2:45\n";
    let warnings = parse_unused_warnings(source, diagnostics);
    assert_eq!(warnings.len(), 3);
    let tree = parse(&language(), source).unwrap();
    let rendered: Vec<_> = collect_unused_warning_candidates(&tree, source, &warnings)
        .iter()
        .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
        .collect();
    assert!(rendered
        .iter()
        .any(|candidate| candidate
            == "entry main(unused_parameter: i32) i32 = let unused_binding = 2 in ???\n"));
    assert!(rendered.iter().any(|candidate| candidate.contains("entry main() i32")));
    assert!(rendered
        .iter()
        .any(|candidate| candidate.contains("entry main(unused_parameter: i32) i32 = ???")));
}

#[test]
fn warning_postprocessing_reuses_the_accepted_check_output() {
    let source = b"def dead = 1\nentry main() i32 = bug()\n".to_vec();
    let diagnostics = b"warning: unused definition `dead` is not reachable from any entry point\n  --> candidate.wyn:1:5\n";
    let accepted = AcceptedWarnings::default();
    accepted.record(source.clone(), diagnostics).unwrap();
    let checks = Arc::new(AtomicUsize::new(0));
    let check = RecordingContains {
        needle: b"bug()",
        accepted: accepted.clone(),
        checks: checks.clone(),
    };

    let (reduced, stats) = postprocess_unused_warnings(&language(), source, &check, &accepted, 0).unwrap();

    assert_eq!(reduced, b"entry main() i32 = bug()\n");
    assert_eq!(stats.attempts, 1);
    assert_eq!(stats.accepted, 1);
    assert_eq!(checks.load(Ordering::Relaxed), 1);
}

#[test]
fn promotes_interesting_branch_out_of_if_expression() {
    let source = b"def main = if true then bug() else other()".to_vec();
    let (reduced, stats) =
        structural_reduce(&language(), source, &Contains(b"bug()"), &type_probe(), 1, 0).unwrap();
    assert_eq!(String::from_utf8(reduced).unwrap(), "def main = bug()");
    assert!(stats.accepted > 0);
}

#[test]
fn parallel_structural_reduction_preserves_candidate_order() {
    let source = b"def main = if true then bug() else other()".to_vec();
    let sequential = structural_reduce(
        &language(),
        source.clone(),
        &Contains(b"bug()"),
        &type_probe(),
        1,
        0,
    )
    .unwrap();
    let parallel =
        structural_reduce(&language(), source, &Contains(b"bug()"), &type_probe(), 4, 0).unwrap();
    assert_eq!(parallel.0, sequential.0);
}

#[test]
fn does_not_promote_elements_out_of_collections() {
    for source in [
        b"def f = [x, y]".as_slice(),
        b"def f = @[x, y]",
        b"def f = (x, y)",
    ] {
        let tree = parse(&language(), source).unwrap();
        let promoted: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::Promotion)
            .map(|candidate| apply_candidate(source, candidate))
            .collect();
        assert!(!promoted.iter().any(|candidate| candidate == b"def f = x"));
        assert!(!promoted.iter().any(|candidate| candidate == b"def f = y"));
    }
}

#[test]
fn avoids_low_probability_promotions() {
    let cases: &[(&[u8], &[&str], &[&str])] = &[
        (
            b"def f = let x = scalar in vector",
            &["def f = vector"],
            &["def f = scalar"],
        ),
        (
            b"def f = if condition then yes else no",
            &["def f = yes", "def f = no"],
            &["def f = condition"],
        ),
        (
            b"def f = normalize(vector)",
            &["def f = vector"],
            &["def f = normalize"],
        ),
    ];
    for (source, expected, forbidden) in cases {
        let tree = parse(&language(), source).unwrap();
        let promoted: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::Promotion)
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        for candidate in *expected {
            assert!(promoted.iter().any(|value| value == candidate));
        }
        for candidate in *forbidden {
            assert!(!promoted.iter().any(|value| value == candidate));
        }
    }
}

#[test]
fn promotes_let_body_only_when_it_does_not_reference_the_binding() {
    let cases: &[(&[u8], &str, bool)] = &[
        (b"def f = let x = value in unrelated", "def f = unrelated", true),
        (b"def f = let x = value in use(x)", "def f = use(x)", false),
        (b"def f = let (x, y) = pair in x + z", "def f = x + z", false),
    ];
    for (source, promoted_body, expected) in cases {
        let tree = parse(&language(), source).unwrap();
        let promoted: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::Promotion)
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        assert_eq!(
            promoted.iter().any(|candidate| candidate == promoted_body),
            *expected
        );
    }
}

#[test]
fn syntax_check_rejects_new_parse_errors() {
    let check = SyntaxCheck {
        inner: Contains(b"bug"),
        language: language(),
        reject_errors: true,
    };
    assert!(check.interesting(b"def f = bug()").unwrap());
    assert!(!check.interesting(b"def f = (bug()").unwrap());
}

#[test]
fn list_deletion_consumes_an_adjacent_comma() {
    let source = b"def f(x, y) = x";
    let tree = parse(&language(), source).unwrap();
    let rendered: Vec<_> = collect_candidates(&tree, source)
        .iter()
        .filter(|candidate| candidate.kind == CandidateKind::ListDeletion)
        .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
        .collect();
    assert!(rendered.iter().any(|candidate| candidate == "def f(y) = x"));
    assert!(rendered.iter().any(|candidate| candidate == "def f(x) = x"));
}

#[test]
fn list_deletion_consumes_a_single_trailing_comma() {
    let source = b"def f(x,) = 0";
    let tree = parse(&language(), source).unwrap();
    let rendered: Vec<_> = collect_candidates(&tree, source)
        .iter()
        .filter(|candidate| candidate.kind == CandidateKind::ListDeletion)
        .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
        .collect();
    assert!(rendered.iter().any(|candidate| candidate == "def f() = 0"));
}

#[test]
fn tuple_deletion_can_collapse_a_trailing_comma_tuple() {
    let source = b"def f = (1, 2,)";
    let tree = parse(&language(), source).unwrap();
    let rendered: Vec<_> = collect_candidates(&tree, source)
        .iter()
        .filter(|candidate| candidate.kind == CandidateKind::ListDeletion)
        .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
        .collect();
    assert!(rendered.iter().any(|candidate| candidate == "def f = (1)"));
}

#[test]
fn concrete_replacements_run_before_holes() {
    let source = b"def f = 12345";
    let tree = parse(&language(), source).unwrap();
    let concrete: Vec<_> = collect_candidates(&tree, source)
        .iter()
        .filter(|candidate| candidate.kind == CandidateKind::Concrete)
        .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
        .collect();
    let holes = hole_replacements();
    assert!(concrete.iter().any(|candidate| candidate == "def f = 0"));
    assert!(concrete.iter().any(|candidate| candidate == "def f = 1"));
    assert_eq!(holes["binary_expression"], ["???"]);
}

#[test]
fn renders_defaults_for_compiler_type_names() {
    assert_eq!(default_literal("i32").as_deref(), Some("0"));
    assert_eq!(default_literal("f32").as_deref(), Some("0.0"));
    assert_eq!(default_literal("vec3f32").as_deref(), Some("@[0.0, 0.0, 0.0]"));
    assert_eq!(
        default_literal("(bool, vec2i32)").as_deref(),
        Some("(false, @[0, 0])")
    );
    assert_eq!(default_literal("[2]f32").as_deref(), Some("[0.0, 0.0]"));
    assert_eq!(default_literal("i32 -> i32"), None);
}

#[test]
fn removes_all_comments_before_reduction() {
    let mut source = b"-- first\ndef f(x:f32) f32 = x -- second\n".to_vec();
    let tree = parse(&language(), &source).unwrap();
    assert_eq!(remove_comments(&tree, &mut source), (2, 17));
    assert_eq!(source, b"\ndef f(x:f32) f32 = x \n");
}

#[test]
fn collapses_whitespace_only_line_runs() {
    let source = b"def f = 0\n   \n\t\n\n-- comment\n\n\n";
    assert_eq!(collapse_blank_lines(source), b"def f = 0\n\n-- comment\n\n");
}

#[test]
fn uninteresting_stdout_requires_stdout_capture() {
    let error =
        Args::try_parse_from(["treereduce-wyn", "--uninteresting-stdout", "BLOCK", "true"]).unwrap_err();
    assert_eq!(error.kind(), clap::error::ErrorKind::MissingRequiredArgument);

    Args::try_parse_from([
        "treereduce-wyn",
        "--interesting-stdout",
        "BUG",
        "--uninteresting-stdout",
        "BLOCK",
        "true",
    ])
    .unwrap();
}

#[test]
fn uninteresting_stderr_requires_stderr_capture() {
    let error =
        Args::try_parse_from(["treereduce-wyn", "--uninteresting-stderr", "BLOCK", "true"]).unwrap_err();
    assert_eq!(error.kind(), clap::error::ErrorKind::MissingRequiredArgument);

    Args::try_parse_from([
        "treereduce-wyn",
        "--interesting-stderr",
        "BUG",
        "--uninteresting-stderr",
        "BLOCK",
        "true",
    ])
    .unwrap();
}

#[test]
fn inherited_output_cannot_also_be_matched() {
    let stdout_error = Args::try_parse_from([
        "treereduce-wyn",
        "--inherit-stdout",
        "--interesting-stdout",
        "BUG",
        "true",
    ])
    .unwrap_err();
    assert_eq!(stdout_error.kind(), clap::error::ErrorKind::ArgumentConflict);

    let stderr_error = Args::try_parse_from([
        "treereduce-wyn",
        "--inherit-stderr",
        "--interesting-stderr",
        "BUG",
        "true",
    ])
    .unwrap_err();
    assert_eq!(stderr_error.kind(), clap::error::ErrorKind::ArgumentConflict);
}
