export interface MetadataSnapshot {
  resources: readonly unknown[];
  stages: readonly unknown[];
  flows: readonly unknown[];
  external_inputs: readonly unknown[];
  recipes: readonly unknown[];
  kernels: readonly unknown[];
  publications: readonly unknown[];
}

export interface MetadataField {
  id: string;
  label?: string;
  value: unknown;
  children: MetadataField[];
}

export function metadataLeaves(fields: readonly MetadataField[]): MetadataField[] {
  return fields.flatMap((field) => field.children.length ? metadataLeaves(field.children) : [field]);
}

export function metadataRecords(snapshot: MetadataSnapshot) {
  return Object.entries(snapshot).filter(([kind]) =>
    ["resources", "stages", "flows", "external_inputs", "recipes", "kernels", "publications"].includes(kind),
  ).flatMap(([kind, values]) => (values as unknown[]).map((value, index) => {
    const record = value as Record<string, unknown>;
    const id = `metadata:${kind}:${record.id ?? index}`;
    const visit = (value: unknown, path: string[], label?: string): MetadataField => {
      const entries = value !== null && typeof value === "object" ? Object.entries(value) : [];
      return {
        id: `${id}:${JSON.stringify(path)}`, label, value: value ?? null,
        children: entries.map(([key, child]) => {
          const segment = Array.isArray(value) && child !== null && typeof child === "object"
            ? String((child as Record<string, unknown>).role !== undefined &&
                entries.filter(([, item]) => item !== null && typeof item === "object" &&
                  item.role === (child as Record<string, unknown>).role).length === 1
                ? (child as Record<string, unknown>).role : key)
            : key;
          return visit(child, [...path, segment], Array.isArray(value) ? undefined : key);
        }),
      };
    };
    const fields = Object.entries(record).filter(([key]) => key !== "id")
      .map(([key, child]) => visit(child, [key], key));
    return { id, kind, label: String(record.name ?? record.entry_name ?? record.id ?? index), fields };
  }));
}

interface BodySnapshot {
  groups: { id: string }[];
  stages: { id: string; kernels: string[] }[];
  recipes: { id: string; entry_group: string }[];
  kernels: { entry_group: string; planned_component?: string }[];
}

export function matchingBodyGroups(source: BodySnapshot, target: BodySnapshot, group: string): string[] {
  const matches = new Set<string>();
  if (target.groups.some((candidate) => candidate.id === group)) matches.add(group);
  const stage = source.stages.find((candidate) => candidate.kernels.includes(group));
  for (const body of target.stages.find((candidate) => candidate.id === stage?.id)?.kernels ?? []) {
    matches.add(body);
  }
  const component = source.recipes.find((recipe) => recipe.entry_group === group)?.id
    ?? source.kernels.find((kernel) => kernel.entry_group === group)?.planned_component;
  if (component) {
    for (const recipe of target.recipes.filter((recipe) => recipe.id === component)) matches.add(recipe.entry_group);
    for (const kernel of target.kernels.filter((kernel) => kernel.planned_component === component)) matches.add(kernel.entry_group);
  }
  return [...matches];
}
