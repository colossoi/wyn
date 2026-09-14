import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import init, { inspect_pass } from "../src/wasm-pkg/wyn_egir_viz_wasm.js";

// Exercise the same catalog and generated compiler that the browser loads.
async function loadSource(path) {
  const compiled = ts.transpileModule(
    await readFile(new URL(path, import.meta.url), "utf8"),
    { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } },
  );
  return import(`data:text/javascript;base64,${Buffer.from(compiled.outputText).toString("base64")}`);
}
const { passDefinitions } = await loadSource("../src/passes.ts");
const { metadataRecords, metadataLeaves, matchingBodyGroups } = await loadSource("../src/metadata.ts");
await init({
  module_or_path: await readFile(new URL("../src/wasm-pkg/wyn_egir_viz_wasm_bg.wasm", import.meta.url)),
});

for (const [pass, { example }] of Object.entries(passDefinitions)) {
  test(`${pass} inspects its browser example`, () => {
    const result = inspect_pass(example, pass);
    assert.equal(result.success, true, result.error?.message);
    assert.equal(result.pass, pass);
    for (const side of ["before", "after"]) {
      const snapshot = result[side];
      assert.ok(snapshot, `${side} snapshot is missing`);
      const ids = snapshot.nodes.map((node) => node.id);
      assert.equal(new Set(ids).size, ids.length, `${side} contains duplicate node IDs`);
      const metadata = metadataRecords(snapshot);
      const fieldIds = metadata.flatMap((record) => metadataLeaves(record.fields).map((field) => field.id));
      assert.equal(new Set(fieldIds).size, fieldIds.length, `${side} contains duplicate metadata IDs`);
      for (const record of metadata) {
        for (const field of metadataLeaves(record.fields)) {
          assert.equal(typeof JSON.stringify(field.value), "string",
            `${side} metadata ${field.label} cannot be rendered`);
        }
      }
      const bodies = snapshot.groups.filter((group) =>
        ["entry", "stage", "kernel"].includes(group.kind));
      assert.ok(bodies.length > 0, `${side} snapshot has no executable bodies`);
      for (const body of bodies) {
        assert.ok(snapshot.blocks.some((block) => block.group === body.id),
          `${side} body ${body.label} has no control flow`);
      }
    }
  });
}

test("partial inlining example replaces the loop call with the helper body", () => {
  const pass = "egir::partially_inline_calls";
  const { before, after, success, error } = inspect_pass(passDefinitions[pass].example, pass);
  assert.equal(success, true, error?.message);
  const caller = before.groups.find((group) => group.kind === "kernel");
  assert.ok(caller);
  const calls = (snapshot) => snapshot.nodes.filter((node) =>
    node.group === caller.id && node.variant === "call");
  assert.equal(calls(before).length, 1, "the example must reach this pass with its call intact");
  assert.equal(calls(after).length, 0, "the pass must inline the caller's only call");
  const helperMath = (snapshot) => snapshot.nodes.filter((node) =>
    node.group === caller.id && node.label === "BinOp(Multiply)");
  assert.equal(helperMath(before).length, 0);
  assert.equal(helperMath(after).length, 1, "the caller must contain the helper's scale calculation");
  assert.ok(after.blocks.filter((block) => block.group === caller.id).length
    > before.blocks.filter((block) => block.group === caller.id).length,
  "the caller must contain the helper's branching body");
});

test("absent optional metadata has a renderable value", () => {
  const [record] = metadataRecords({ recipes: [{ id: "stage:0/component:0", operation: undefined }] });
  assert.equal(record.fields[0].value, null);
});

test("scratch metadata changes from requirements to resources without changing recipe identity", () => {
  const pass = "egir::allocate_recipe_scratch";
  const { before, after, success, error } = inspect_pass(passDefinitions[pass].example, pass);
  assert.equal(success, true, error?.message);
  assert.deepEqual(after.recipes.map((recipe) => recipe.id), before.recipes.map((recipe) => recipe.id));
  const fields = (snapshot) => new Map(metadataRecords(snapshot)
    .flatMap((record) => metadataLeaves(record.fields).map((field) => [field.id, field])));
  const afterFields = fields(after);
  const requirements = metadataRecords(before).flatMap((record) =>
    metadataLeaves(record.fields.filter((field) => field.label === "scratch")))
    .filter((field) => field.label === "state" && field.value === "required");
  assert.ok(requirements.length > 0);
  for (const required of requirements) {
    assert.equal(afterFields.get(required.id).value, "bound");
  }
  for (const recipe of after.recipes) {
    const original = before.recipes.find((candidate) => candidate.id === recipe.id);
    assert.equal(recipe.operation, original.operation);
    assert.equal(recipe.kind, original.kind);
    for (const slot of recipe.scratch) {
      assert.ok(after.resources.some((resource) => resource.id === slot.resource));
    }
  }
});

test("scheduled phases point to their planned components and preserve IDs through physicalization", () => {
  const source = passDefinitions["egir::build_kernel_schedule"].example;
  const scheduled = inspect_pass(source, "egir::build_kernel_schedule");
  assert.equal(scheduled.success, true, scheduled.error?.message);
  const components = new Set(scheduled.before.recipes.map((recipe) => recipe.id));
  assert.ok(scheduled.after.kernels.length > components.size);
  assert.ok(scheduled.after.kernels.some((kernel) => kernel.dependencies.length > 0));
  for (const kernel of scheduled.after.kernels) {
    assert.ok(components.has(kernel.planned_component));
    assert.ok(scheduled.after.groups.some((group) => group.id === kernel.entry_group));
  }
  for (const recipe of scheduled.before.recipes) {
    assert.deepEqual(matchingBodyGroups(scheduled.before, scheduled.after, recipe.entry_group),
      scheduled.after.kernels.filter((kernel) => kernel.planned_component === recipe.id)
        .map((kernel) => kernel.entry_group));
  }
  const physical = inspect_pass(source, "egir::physicalize_kernel_schedule");
  assert.equal(physical.success, true, physical.error?.message);
  assert.deepEqual(physical.after.kernels.map((kernel) => kernel.id),
    scheduled.after.kernels.map((kernel) => kernel.id));
});

test("removed pass IDs are rejected", () => {
  for (const pass of ["egir::analyze_kernel_recipes", "egir::bind_mapped_output_destinations",
    "egir::finalize_kernel_schedule", "egir::plan_logical_resources", "egir::plan_kernel_recipes", "egir::expand_soacs"]) {
    const result = inspect_pass(passDefinitions["egir::reify_soacs"].example, pass);
    assert.equal(result.success, false);
  }
});

test("repeated resource roles have distinct metadata identities", () => {
  const [record] = metadataRecords({ publications: [{ id: "entry:0", resources: [
    { role: "input", resource: "$r0" }, { role: "input", resource: "$r1" },
  ] }] });
  const fields = metadataLeaves(record.fields);
  assert.equal(new Set(fields.map((field) => field.id)).size, fields.length);
});
