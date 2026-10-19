use super::*;

#[test]
fn parses_source_level_feedback_and_resolves_relative_seed() {
    let feedback = parse(
        r#"{
                "version": 1,
                "feedback": [{
                    "entry": "pulse",
                    "input": "previous",
                    "result": 0,
                    "initial": { "kind": "file", "path": "seed.bin" }
                }]
            }"#,
        Path::new("examples"),
    )
    .expect("valid sidecar");

    assert_eq!(feedback.len(), 1);
    assert_eq!(feedback[0].entry, "pulse");
    assert_eq!(feedback[0].input, "previous");
    assert_eq!(feedback[0].result, 0);
    assert_eq!(
        feedback[0].initial,
        FeedbackInitial::File {
            path: PathBuf::from("examples").join("seed.bin")
        }
    );
}

#[test]
fn rejects_unknown_versions_and_duplicate_destinations() {
    let bad_version = parse(r#"{"version": 2}"#, Path::new("."));
    assert!(bad_version.unwrap_err().to_string().contains("unsupported viz config version"));

    let duplicate = parse(
        r#"{
                "version": 1,
                "feedback": [
                    {"entry":"pulse","input":"previous","result":0},
                    {"entry":"pulse","input":"previous","result":1}
                ]
            }"#,
        Path::new("."),
    );
    assert!(duplicate.unwrap_err().to_string().contains("duplicate feedback destination"));
}

#[test]
fn derives_sibling_sidecar_name() {
    assert_eq!(
        automatic_path(Path::new("tmp/particles.spv")),
        PathBuf::from("tmp/particles.viz.json")
    );
}
