use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use super::*;
use crate::{ModulePath, PackageGraph, PackageGraphBuilder, PackageIdentity, TextRange};

struct TestPlan {
    plan: PackageGraph,
    root: ModuleKey,
    dependency: ModuleKey,
}

fn test_plan() -> TestPlan {
    let root_path = ModulePath::new("src/main.wyn").expect("valid root path");
    let dependency_path = ModulePath::new("src/lib.wyn").expect("valid dependency path");
    let mut builder = PackageGraphBuilder::new();
    let root_package = builder
        .add_package(
            PackageIdentity::new("example/root", "v0.1.0").expect("valid root identity"),
            root_path.clone(),
        )
        .expect("root package should be unique");
    let dependency_package = builder
        .add_package(
            PackageIdentity::new("example/dependency", "v2.3.4").expect("valid dependency identity"),
            dependency_path.clone(),
        )
        .expect("dependency package should be unique");
    let root = ModuleKey::new(root_package, root_path);
    let dependency = ModuleKey::new(dependency_package, dependency_path);
    builder.set_root(root.clone()).expect("root module should belong to plan");
    TestPlan {
        plan: builder.build().expect("complete package plan"),
        root,
        dependency,
    }
}

fn range(start: u32, end: u32) -> TextRange {
    TextRange::new(start, end).expect("valid source range")
}

#[test]
fn load_failure_names_dependency_release_and_import_chain() {
    let TestPlan {
        plan,
        root,
        dependency,
    } = test_plan();
    let root_id = ModuleId::from(0);
    let dependency_id = ModuleId::from(1);
    let missing_id = ModuleId::from(2);
    let missing = ModuleKey::new(
        dependency.package(),
        ModulePath::new("src/missing.wyn").expect("valid missing path"),
    );
    let root_import = Span::new(root_id, range(0, 10));
    let dependency_import = Span::new(dependency_id, range(6, 13));
    let trace = vec![
        ImportTraceFrame {
            span: root_import,
            requested: dependency.clone(),
        },
        ImportTraceFrame {
            span: dependency_import,
            requested: missing.clone(),
        },
    ];
    let error = BuildError::<io::Error, io::Error>::Load {
        module: missing.clone(),
        trace: trace.into_boxed_slice(),
        source: io::Error::new(io::ErrorKind::NotFound, "missing test source"),
    };
    let module_ids = HashMap::from([
        (root, root_id),
        (dependency, dependency_id),
        (missing, missing_id),
    ]);
    let mut sources = SourceMap::default();
    sources.insert(root_id, Arc::from("import dep")).expect("valid root source");
    sources.insert(dependency_id, Arc::from("first\nimport dep")).expect("valid dependency source");
    let failure = BuildFailure::new(error, plan, module_ids, sources);

    assert_eq!(
        failure.to_string(),
        concat!(
            "failed to load example/dependency@v2.3.4:src/missing.wyn: missing test source\n",
            "  imported from src/main.wyn:1:1\n",
            "  imported from example/dependency@v2.3.4:src/lib.wyn:2:1",
        )
    );
}

#[test]
fn dependency_error_uses_package_relative_location() {
    let TestPlan {
        plan,
        root,
        dependency,
    } = test_plan();
    let root_id = ModuleId::from(0);
    let dependency_id = ModuleId::from(1);
    let root_import = Span::new(root_id, range(0, 10));
    let dependency_import = Span::new(dependency_id, range(6, 17));
    let error = BuildError::<io::Error, io::Error>::UnknownDependency {
        alias: DependencyAlias::new("missing").expect("valid dependency alias"),
        span: dependency_import,
        trace: vec![ImportTraceFrame {
            span: root_import,
            requested: dependency.clone(),
        }]
        .into_boxed_slice(),
    };
    let module_ids = HashMap::from([(root, root_id), (dependency, dependency_id)]);
    let mut sources = SourceMap::default();
    sources.insert(root_id, Arc::from("import dep")).expect("valid root source");
    sources.insert(dependency_id, Arc::from("first\ndep:missing")).expect("valid dependency source");
    let failure = BuildFailure::new(error, plan, module_ids, sources);

    assert_eq!(
        failure.to_string(),
        concat!(
            "example/dependency@v2.3.4:src/lib.wyn:2:1: ",
            "unknown package dependency alias `missing`\n",
            "  imported from src/main.wyn:1:1",
        )
    );
}
