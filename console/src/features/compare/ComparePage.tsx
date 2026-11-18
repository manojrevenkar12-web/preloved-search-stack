import { useState } from "react";
import { useSearch } from "../../core/api";
import type { Bench } from "../../core/types";
import { DesignColumn } from "./DesignColumn";
import { Understood } from "./Understood";

export function ComparePage({ bench }: { bench: Bench }) {
  const recorded = bench.recorded ? Object.keys(bench.recorded) : [];
  const [query, setQuery] = useState(recorded[0] ?? "sac chanel noir");
  const [a, setA] = useState(bench.report?.control ?? "lexical");
  const [b, setB] = useState(bench.report?.proposed ?? "personalised");
  const qa = useSearch(bench, query, a);
  const qb = useSearch(bench, query, b);
  return (
    <div className="flex flex-col gap-5">
      <form role="search" onSubmit={(e) => e.preventDefault()} className="flex flex-col gap-3">
        <label htmlFor="q" className="text-sm text-muted">A buyer types, in any of five languages, with typos and nicknames</label>
        <input id="q" value={query} onChange={(e) => setQuery(e.target.value)} autoComplete="off"
               className="w-full rounded-[var(--radius-field)] border border-rule bg-card px-4 py-3 font-[family-name:var(--font-tag)] text-2xl" />
        {recorded.length > 0 && (
          <div className="flex flex-wrap gap-2" aria-label="Recorded queries">
            {recorded.map((r) => (
              <button key={r} type="button" onClick={() => setQuery(r)} aria-pressed={r === query}
                      className={`rounded-full border px-3 py-1 text-sm ${r === query ? "border-garnet bg-garnet text-white" : "border-rule bg-card hover:border-garnet"}`}>
                {r}
              </button>
            ))}
          </div>
        )}
      </form>
      <Understood r={qb.data} />
      <div className="grid gap-4 lg:grid-cols-2">
        <DesignColumn bench={bench} query={query} design={a} onDesign={setA} side="A" q={qa} other={qb.data} />
        <DesignColumn bench={bench} query={query} design={b} onDesign={setB} side="B" q={qb} other={qa.data} />
      </div>
    </div>
  );
}
