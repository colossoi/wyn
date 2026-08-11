# Corvid WGSL integration punchlist

Each item lands as one focused commit after the full formatting, workspace-test,
SPIR-V-testfile, and WGSL/Naga-testfile gates pass.

- [x] Emit a pipeline descriptor beside WGSL output, not only SPIR-V output.
- [x] Emit transitive dependencies of composite top-level constants.
- [x] Preserve WGSL argument types for large storage arrays passed through named helpers.
- [x] Generate Naga-valid WGSL for ranked literal `bucket_scatter` inputs.
- [x] Report fixed byte lengths for fixed external arrays in pipeline descriptors.
