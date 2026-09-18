use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static CASE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Case {
    directory: PathBuf,
}

impl Case {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "wyn_compiler_features_{}_{}",
            std::process::id(),
            CASE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&directory).expect("create compiler feature test directory");
        fs::write(
            directory.join("input.wyn"),
            "entry main(xs: []i32) []i32 = map(|x| x + 1, xs)",
        )
        .expect("write test source");
        Self { directory }
    }

    fn compile(&self, target: &str, flags: &[&str]) -> Output {
        let extension = if target == "spirv" { "spv" } else { "wgsl" };
        Command::new(env!("CARGO_BIN_EXE_wyn"))
            .arg("build")
            .arg(self.directory.join("input.wyn"))
            .args(["--target", target, "--verbose", "--output"])
            .arg(self.directory.join(format!("output.{extension}")))
            .arg("--output-mir")
            .arg(self.directory.join("output.ssa"))
            .args(flags)
            .current_dir(&self.directory)
            .output()
            .expect("run compiler")
    }

    fn assert_compiled(&self, target: &str, result: &Output, egglog: bool) {
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mir = fs::read_to_string(self.directory.join("output.ssa")).expect("SSA output");
        assert_eq!(
            mir.contains("egg_kernel_"),
            egglog,
            "unexpected compiler route: {mir}"
        );
        let descriptor: serde_json::Value =
            serde_json::from_slice(&fs::read(self.directory.join("output.json")).expect("descriptor"))
                .expect("valid descriptor JSON");
        assert!(!descriptor["pipelines"].as_array().expect("pipelines").is_empty());
        if target == "spirv" {
            let binary = fs::read(self.directory.join("output.spv")).expect("SPIR-V output");
            assert_eq!(binary.get(..4), Some([0x03, 0x02, 0x23, 0x07].as_slice()));
        } else {
            let wgsl = fs::read_to_string(self.directory.join("output.wgsl")).expect("WGSL output");
            assert!(wgsl.contains("@compute"));
        }
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).expect("remove compiler feature test directory");
    }
}

#[test]
fn default_route_compiles_spirv_and_wgsl() {
    for target in ["spirv", "wgsl"] {
        let case = Case::new();
        let result = case.compile(target, &[]);
        case.assert_compiled(target, &result, true);
    }
}

#[test]
fn explicit_egglog_route_emits_shader() {
    let case = Case::new();
    let result = case.compile("wgsl", &["--egglog"]);
    case.assert_compiled("wgsl", &result, true);
}

#[test]
fn direct_compiles_authored_compute() {
    let case = Case::new();
    fs::write(
        case.directory.join("input.wyn"),
        "entry main(xs: [4]i32) [4]i32 = map(|x| x+1, xs)",
    )
    .unwrap();
    let result = case.compile("wgsl", &["--direct"]);
    case.assert_compiled("wgsl", &result, true);
}
