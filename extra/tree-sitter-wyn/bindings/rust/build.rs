fn main() {
    let manifest_dir = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    );
    // Published crates place the generated parser under `src/`; a checkout of
    // this repository keeps it at the Tree-sitter package root.
    let packaged_src = manifest_dir.join("src");
    let repository_src = manifest_dir.join("../../src");
    let src_dir = if packaged_src.join("parser.c").exists() {
        packaged_src
    } else {
        repository_src
    };

    let mut c_config = cc::Build::new();
    c_config.std("c11").include(&src_dir);

    #[cfg(target_env = "msvc")]
    c_config.flag("-utf-8");

    let parser_path = src_dir.join("parser.c");
    c_config.file(&parser_path);
    println!("cargo:rerun-if-changed={}", parser_path.to_str().unwrap());

    // Handle scanner.c if it exists
    let scanner_path = src_dir.join("scanner.c");
    if scanner_path.exists() {
        c_config.file(&scanner_path);
        println!("cargo:rerun-if-changed={}", scanner_path.to_str().unwrap());
    }

    c_config.compile("tree-sitter-wyn");
}
