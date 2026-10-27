mod support;

use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Action {
    Check,
    Build,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expectations {
    command: Action,
    exit: i32,
    #[serde(default)]
    targets: Vec<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    emit_mir: bool,
    source_path: Option<PathBuf>,
    output_path: Option<String>,
    #[serde(default)]
    artifacts: Vec<Artifact>,
    // Omission permits warnings; [] explicitly requires warning-free output.
    warnings: Option<Vec<String>>,
    #[serde(default)]
    errors: Vec<String>,
    #[serde(default)]
    contains: Vec<String>,
    #[serde(default)]
    excludes: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    #[serde(default)]
    targets: Vec<String>,
    #[serde(default)]
    absent: bool,
    starts_with: Option<String>,
    #[serde(default)]
    contains: Vec<String>,
    #[serde(default)]
    excludes: Vec<String>,
}

struct OutputDirectory(PathBuf);
impl Drop for OutputDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn assert_diagnostics(stderr: &str, kind: &str, expected: &[String]) {
    let prefix = format!("{kind}: ");
    let actual: Vec<_> = stderr.lines().filter_map(|line| line.strip_prefix(&prefix)).collect();
    assert_eq!(actual.len(), expected.len(), "unexpected {kind} count:\n{stderr}");
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            actual.starts_with(expected),
            "expected {kind} {expected:?}:\n{stderr}"
        );
    }
}

fn run_fixture(path: &Path) {
    let source = fs::read_to_string(path).expect("read fixture");
    let header = source
        .lines()
        .take_while(|line| line.starts_with("--!"))
        .map(|line| line.strip_prefix("--!").unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!header.is_empty(), "fixture must start with --! expectations");
    let expected: Expectations = toml::from_str(&header).expect("valid fixture expectations");
    let targets = match expected.command {
        Action::Check => {
            assert!(
                expected.targets.is_empty(),
                "check fixtures cannot specify shader targets"
            );
            vec![""]
        }
        Action::Build => {
            assert!(
                !expected.targets.is_empty(),
                "build fixtures must specify targets"
            );
            assert!(expected.targets.iter().all(|target| matches!(target.as_str(), "spirv" | "wgsl")));
            expected.targets.iter().map(String::as_str).collect()
        }
    };
    assert!(
        !expected.emit_mir || matches!(expected.command, Action::Build),
        "MIR output requires a build"
    );
    for artifact in &expected.artifacts {
        assert!(
            artifact.targets.iter().all(|target| expected.targets.contains(target)),
            "artifact target must be one of the fixture targets"
        );
        assert!(
            !artifact.absent
                || (artifact.contains.is_empty()
                    && artifact.excludes.is_empty()
                    && artifact.starts_with.is_none()),
            "absent artifacts cannot have content expectations"
        );
    }
    if matches!(expected.command, Action::Build) && expected.exit == 0 {
        for target in &expected.targets {
            assert!(
                expected.artifacts.iter().any(|artifact| {
                    !artifact.absent
                        && (artifact.targets.is_empty() || artifact.targets.contains(target))
                        && (artifact.contains.iter().any(|text| !text.is_empty())
                            || artifact.starts_with.as_ref().is_some_and(|text| !text.is_empty()))
                }),
                "successful build for {target} needs an affirmative artifact-content check"
            );
        }
    }
    let name = path.file_stem().unwrap().to_str().unwrap();
    let directory =
        OutputDirectory(std::env::temp_dir().join(format!("wyn_fixture_{}_{name}", std::process::id(),)));
    fs::create_dir(&directory.0).expect("create fixture output directory");
    for target in targets {
        let directory = OutputDirectory(directory.0.join(if target.is_empty() { "check" } else { target }));
        fs::create_dir(&directory.0).expect("create target output directory");
        let mut command = Command::new(env!("CARGO_BIN_EXE_wyn"));
        command.arg(match expected.command {
            Action::Check => "check",
            Action::Build => "build",
        });
        command.current_dir(&directory.0);
        if let Some(relative) = &expected.source_path {
            assert!(
                relative.components().all(|part| matches!(part, std::path::Component::Normal(_))),
                "source_path must stay inside the output directory"
            );
            let copied = directory.0.join(relative);
            fs::create_dir_all(copied.parent().unwrap()).expect("create source directory");
            fs::write(copied, &source).expect("copy fixture source");
            command.arg(relative);
        } else {
            command.arg(path);
        }
        if !expected.args.iter().any(|arg| arg == "--max-warnings") {
            command.args(["--max-warnings", "100"]);
        }
        let output_path =
            expected.output_path.as_deref().unwrap_or("output.{target}").replace("{target}", target);
        let output_path = Path::new(&output_path);
        assert!(
            output_path.components().all(|part| matches!(part, std::path::Component::Normal(_))),
            "output_path must stay inside the output directory"
        );
        let artifact = directory.0.join(output_path);
        if matches!(expected.command, Action::Build) {
            fs::create_dir_all(artifact.parent().unwrap()).expect("create artifact directory");
            command.args(["--target", target, "--output"]).arg(output_path);
        }
        if expected.emit_mir {
            command.arg("--output-mir").arg(directory.0.join("output.ssa"));
        }
        let output = command.args(&expected.args).output().expect("run Wyn CLI");
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("fixture {name} {target}: {}", output.status);
        assert_eq!(
            output.status.code(),
            Some(expected.exit),
            "{name} {target}:\n{stderr}\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert_diagnostics(&stderr, "error", &expected.errors);
        if let Some(warnings) = &expected.warnings {
            assert_diagnostics(&stderr, "warning", warnings);
        }
        for text in &expected.contains {
            assert!(stderr.contains(text), "missing {text:?}:\n{stderr}");
        }
        for text in &expected.excludes {
            assert!(!stderr.contains(text), "unexpected {text:?}:\n{stderr}");
        }
        if matches!(expected.command, Action::Build) && expected.exit == 0 {
            support::validate_shader(&artifact, target);
        }
        for artifact in &expected.artifacts {
            if !artifact.targets.is_empty() && !artifact.targets.iter().any(|wanted| wanted == target) {
                continue;
            }
            let path = directory.0.join(artifact.path.replace("{target}", target));
            if artifact.absent {
                assert!(!path.exists(), "unexpected artifact: {}", path.display());
                continue;
            }
            assert!(path.is_file(), "missing artifact: {}", path.display());
            if !artifact.contains.is_empty()
                || !artifact.excludes.is_empty()
                || artifact.starts_with.is_some()
            {
                let text = fs::read_to_string(&path).expect("read text artifact");
                if let Some(prefix) = &artifact.starts_with {
                    assert!(
                        text.starts_with(prefix),
                        "{} must start with {prefix:?}:\n{text}",
                        path.display()
                    );
                }
                for wanted in &artifact.contains {
                    assert!(
                        text.contains(wanted),
                        "{} missing {wanted:?}:\n{text}",
                        path.display()
                    );
                }
                for forbidden in &artifact.excludes {
                    assert!(
                        !text.contains(forbidden),
                        "{} contains {forbidden:?}:\n{text}",
                        path.display()
                    );
                }
            }
        }
    }
}

#[test]
fn compiler_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/compiler/cases");
    let mut paths: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "wyn"))
        .collect();
    let filter = std::env::var("WYN_TEST_FILTER").unwrap_or_default();
    paths.retain(|path| path.file_stem().unwrap().to_string_lossy().contains(&filter));
    paths.sort();
    assert!(!paths.is_empty(), "no compiler fixtures discovered");
    // Bound concurrency: each fixture starts a compiler process.
    let worker_count = std::thread::available_parallelism().map_or(1, usize::from).min(24).min(paths.len());
    let pending = std::sync::Mutex::new(paths.iter());
    let failures = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..worker_count)
            .map(|_| {
                scope.spawn(|| {
                    let mut failures = Vec::new();
                    loop {
                        let path = pending.lock().unwrap().next();
                        let Some(path) = path else { break };
                        if std::panic::catch_unwind(|| run_fixture(path)).is_err() {
                            failures.push(path.display().to_string());
                        }
                    }
                    failures
                })
            })
            .collect();
        workers.into_iter().flat_map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
    });
    assert!(failures.is_empty(), "failed fixtures:\n{}", failures.join("\n"));
}
