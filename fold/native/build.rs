fn main() {
    let source = "../generated/constraints.cpp";
    println!("cargo:rerun-if-changed={source}");
    assert!(
        std::path::Path::new(source).exists(),
        "Run python3 fold/build.py first"
    );
    cc::Build::new()
        .cpp(true)
        .compiler(std::env::var("CXX").unwrap_or_else(|_| "clang++".into()))
        .file(source)
        .flag("-std=c++20")
        .flag("-ffp-contract=off")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-missing-field-initializers")
        .compile("fold_cpu");
}
