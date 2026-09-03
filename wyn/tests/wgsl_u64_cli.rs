use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const BLAKE2B_MIX_SOURCE: &str = r#"
def rotr16(x: u64) u64 = (x >> 16u64) | (x << 48u64)
def rotr24(x: u64) u64 = (x >> 24u64) | (x << 40u64)
def rotr32(x: u64) u64 = (x >> 32u64) | (x << 32u64)
def rotr63(x: u64) u64 = (x >> 63u64) | (x << 1u64)

def blake2b_lane(x: u64, y: u64) u64 =
  let a0 = 0x6a09e667f3bcc908u64 + 0xbb67ae8584caa73bu64 + x
  let d0 = rotr32(0xa54ff53a5f1d36f1u64 ^ a0)
  let c0 = 0x3c6ef372fe94f82bu64 + d0
  let b0 = rotr24(0xbb67ae8584caa73bu64 ^ c0)
  let a1 = a0 + b0 + y
  let d1 = rotr16(d0 ^ a1)
  let c1 = c0 + d1 in
  rotr63(b0 ^ c1)

entry blake2b_mix(words: []u32) []u32 =
  map(|word: u32|
    u32.u64(blake2b_lane(u64.u32(word), 0xfedcba9876543210u64)),
    words)
"#;

fn temp_case() -> (PathBuf, PathBuf, PathBuf) {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_nanos();
    let directory = std::env::temp_dir().join(format!("wyn_wgsl_u64_{}_{}", std::process::id(), unique));
    fs::create_dir_all(&directory).expect("create test directory");
    let source = directory.join("blake2b_mix.wyn");
    let output = directory.join("blake2b_mix.wgsl");
    fs::write(&source, BLAKE2B_MIX_SOURCE).expect("write test source");
    (directory, source, output)
}

fn compile(source: &PathBuf, output: &PathBuf, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wyn"));
    command.arg("build").arg(source).arg("--target").arg("wgsl").arg("--output").arg(output).args(extra);
    command.output().expect("run wyn compiler")
}

#[test]
fn wgsl_u64_requires_opt_in_and_compiles_blake2b_mix_when_enabled() {
    let (directory, source, output) = temp_case();

    let rejected = compile(&source, &output, &[]);
    assert!(!rejected.status.success(), "u64 unexpectedly enabled by default");
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("64-bit scalars"),
        "unexpected default diagnostic: {}",
        String::from_utf8_lossy(&rejected.stderr)
    );

    let accepted = compile(&source, &output, &["--wgsl-emulate-u64"]);
    assert!(
        accepted.status.success(),
        "opt-in compilation failed: {}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    let wgsl = fs::read_to_string(&output).expect("generated WGSL");
    assert!(wgsl.contains("fn _wyn_u64_add"));
    assert!(wgsl.contains("vec2<u32>(1985229328u, 4275878552u)"));
    assert!(
        !wgsl.contains("fn _wyn_u64_shl"),
        "fixed rotates should use specialized shifts"
    );
    assert!(
        !wgsl.contains("fn _wyn_u64_shr"),
        "fixed rotates should use specialized shifts"
    );

    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn wgsl_u64_flag_is_rejected_for_spirv() {
    let (directory, source, output) = temp_case();
    let result = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("build")
        .arg(&source)
        .arg("--target")
        .arg("spirv")
        .arg("--output")
        .arg(&output)
        .arg("--wgsl-emulate-u64")
        .output()
        .expect("run wyn compiler");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("--wgsl-emulate-u64 requires --target wgsl"));
    fs::remove_dir_all(directory).expect("remove test directory");
}
