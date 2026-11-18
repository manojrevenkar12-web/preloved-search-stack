import type { SearchResponse } from "../../core/types";

const LABEL: Record<string, string> = { brand: "Brand", model: "Model", category: "Category", colour: "Colour",
  material: "Material", style: "Style", size: "Size" };

/** What query understanding read from the words: the slots, and any spelling it corrected. */
export function Understood({ r }: { r: SearchResponse | null | undefined }) {
  if (!r?.understood) return <p className="text-sm text-muted">Design B does not read the query; it matches words in titles.</p>;
  const slots = Object.entries(r.understood);
  return (
    <dl className="flex flex-wrap items-center gap-2 text-sm" aria-label="What the query was read as">
      {slots.length === 0 && <dd className="text-muted">No brand, model or attribute recognised: the learned retriever does the work.</dd>}
      {slots.map(([k, v]) => (
        <div key={k} className="flex items-baseline gap-1.5 rounded-[var(--radius-tag)] border border-rule bg-card px-2 py-1">
          <dt className="text-xs text-muted">{LABEL[k] ?? k}</dt><dd className="font-semibold">{v}</dd>
        </div>
      ))}
      {r.corrected.map(([from, to]) => (
        <div key={from} className="rounded-[var(--radius-tag)] bg-garnet-soft px-2 py-1 text-xs">
          read <s>{from}</s> as <b>{to}</b>
        </div>
      ))}
    </dl>
  );
}
