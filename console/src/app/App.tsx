import { useState } from "react";
import { mode, useBench } from "../core/api";
import { ComparePage } from "../features/compare/ComparePage";
import { StudyPage } from "../features/study/StudyPage";
import { Failure, Loading } from "../shared/States";

const TABS = [["compare", "Compare designs"], ["study", "Offline vs online"]] as const;

export function App() {
  const bench = useBench();
  const [tab, setTab] = useState<(typeof TABS)[number][0]>("compare");
  return (
    <div className="mx-auto max-w-[var(--width-page)] px-4 pb-16 sm:px-8">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-rule py-6">
        <div>
          <h1 className="font-[family-name:var(--font-tag)] text-2xl leading-none">Search bench</h1>
          <p className="mt-2 text-sm text-muted">
            One-of-a-kind listings, five languages, two designs side by side
            {bench.data?.report && ` · ${bench.data.report.world} world`} · {mode === "demo" ? "recorded demo" : "live service"}
          </p>
        </div>
        <nav aria-label="Views" className="flex gap-1">
          {TABS.map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)} aria-current={tab === k ? "page" : undefined}
                    className={`rounded-[var(--radius-field)] px-3 py-2 text-sm ${tab === k ? "bg-ink text-white" : "text-muted hover:text-ink"}`}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="pt-6">
        {bench.isLoading ? <Loading label="Loading the search bench" />
          : bench.isError ? <Failure message={(bench.error as Error).message} retry={() => bench.refetch()} />
          : tab === "compare" ? <ComparePage bench={bench.data!} /> : <StudyPage report={bench.data!.report} />}
      </main>
    </div>
  );
}
