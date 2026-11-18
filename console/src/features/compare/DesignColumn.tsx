import { AnimatePresence, motion } from "motion/react";
import type { UseQueryResult } from "@tanstack/react-query";
import type { Bench, SearchResponse } from "../../core/types";
import { Empty, Failure, Loading } from "../../shared/States";
import { fade } from "../../theme/motion";
import { ResultRow } from "./ResultRow";

type Props = { bench: Bench; query: string; design: string; onDesign: (d: string) => void; side: "A" | "B";
  q: UseQueryResult<SearchResponse | null>; other: SearchResponse | null | undefined };

/** One design's results page. Each row says how far it moved against the other design's page. */
export function DesignColumn({ bench, query, design, onDesign, side, q, other }: Props) {
  const tone = side === "A" ? "border-slate text-slate" : "border-garnet text-garnet";
  const otherRank = new Map((other?.items ?? []).map((x, i) => [x.item_id, i]));
  return (
    <section aria-label={`Design ${side}`} className="min-w-0 rounded-[var(--radius-field)] bg-card p-4 shadow-[0_1px_0_var(--color-rule)]">
      <header className={`flex items-center justify-between gap-3 border-l-4 pl-3 ${tone}`}>
        <label className="flex items-center gap-2 text-sm font-semibold">
          <span>{side}</span>
          <select value={design} onChange={(e) => onDesign(e.target.value)}
                  className="rounded-[var(--radius-field)] border border-rule bg-tissue px-2 py-1 text-ink">
            {bench.designs.map((d) => <option key={d.name} value={d.name}>{d.name.replace(/_/g, " ")}</option>)}
          </select>
        </label>
        {q.data && <span className="text-xs text-muted">{q.data.candidates} candidates · {q.data.ms} ms</span>}
      </header>
      <div className="mt-2">
        {!query.trim() ? <Empty title="Type a query to see this design's results." />
          : q.isLoading ? <Loading label={`Loading results for design ${side}`} />
          : q.isError ? <Failure message={(q.error as Error).message} retry={() => q.refetch()} />
          : !q.data ? <Empty title="This query was not recorded for the demo.">Pick one of the recorded queries, or run the service (`make serve`) and set VITE_API_URL to search anything.</Empty>
          : q.data.items.length === 0 ? <Empty title="No listing for sale matches.">Try fewer words, or a brand or category on its own.</Empty>
          : (
            <AnimatePresence mode="popLayout">
              <motion.ol key={design} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={fade}>
                {q.data.relaxed.length > 0 && (
                  <li className="mb-2 rounded-[var(--radius-field)] bg-slate-soft px-3 py-2 text-xs">
                    Too few listings matched, so the {q.data.relaxed.join(" and ")} filter was dropped.
                  </li>
                )}
                {q.data.items.map((it, i) => (
                  <ResultRow key={it.item_id} item={it} rank={i + 1}
                             moved={other ? (otherRank.has(it.item_id) ? otherRank.get(it.item_id)! - i : null) : 0} />
                ))}
              </motion.ol>
            </AnimatePresence>
          )}
      </div>
    </section>
  );
}
