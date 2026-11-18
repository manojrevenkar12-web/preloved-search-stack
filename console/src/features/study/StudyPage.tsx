import { motion } from "motion/react";
import type { Report, StudyRow } from "../../core/types";
import { Empty } from "../../shared/States";
import { pct, share } from "../../shared/format";

export const METRICS: Record<string, string> = {
  judged_dcg10: "Judged relevance of the top 10 (rater labels)",
  recall_today: "Bought item in top 20, today's tables",
  ndcg_clicks: "Click nDCG on the logged searches",
  ndcg_ips: "Click nDCG, weighted for position",
  recall_pit: "Bought item in top 20, point in time",
};

function TauBar({ name, row }: { name: string; row: StudyRow }) {
  const w = Math.max(0, row.kendall_tau) * 100;
  return (
    <li className="grid grid-cols-[minmax(0,16rem)_1fr_auto] items-center gap-3 py-2">
      <span className="text-sm">{METRICS[name] ?? name}</span>
      <div className="h-3 rounded-full bg-rule/50" aria-hidden>
        <motion.div className="h-3 rounded-full bg-garnet" initial={{ width: 0 }} animate={{ width: `${w}%` }}
                    transition={{ duration: 0.6, ease: "easeOut" }} />
      </div>
      <span className="w-48 text-right text-sm">
        τ {row.kendall_tau.toFixed(2)} · right direction {share(row.sign_agreement)}
      </span>
    </li>
  );
}

export function StudyPage({ report }: { report: Report | null }) {
  if (!report) return <Empty title="No evaluation report yet.">Run <code>make evaluate</code>, then reload.</Empty>;
  const rows = Object.entries(report.study).filter((e): e is [string, StudyRow] => typeof e[1] === "object")
    .sort((x, y) => y[1].kendall_tau - x[1].kendall_tau);
  const names = Object.keys(report.online).filter((n) => n !== report.control)
    .sort((x, y) => report.online[y].purchase_lift - report.online[x].purchase_lift);
  const p = report.online[report.proposed];
  const ab = report.ab_summary;
  return (
    <div className="flex flex-col gap-8">
      <p className="max-w-3xl font-[family-name:var(--font-tag)] text-xl leading-snug">
        Rolled out to every buyer, the {report.proposed} design sold {pct(p.purchase_lift)} more per search than
        today's {report.control} search. Interleaving reached the same verdict on {Math.round(report.efficiency.ratio)}×
        fewer searches than an A/B test.
      </p>

      <section aria-labelledby="tau">
        <h2 id="tau" className="text-lg font-semibold">Which offline metric orders the designs the way buyers did</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">Kendall's τ between each metric's change and the purchase lift of a full rollout, over {names.length} designs and variants.</p>
        <ul className="mt-3">{rows.map(([n, r]) => <TauBar key={n} name={n} row={r} />)}</ul>
      </section>

      <section aria-labelledby="designs">
        <h2 id="designs" className="text-lg font-semibold">Each design against today's search</h2>
        <div className="mt-3 overflow-x-auto rounded-[var(--radius-field)] bg-card">
          <table className="w-full text-sm">
            <thead className="text-left text-muted">
              <tr className="border-b border-rule">
                <th className="px-3 py-2 font-medium">Design</th>
                <th className="px-3 py-2 text-right font-medium">Purchases, rollout</th>
                <th className="px-3 py-2 text-right font-medium">Revenue, rollout</th>
                {rows.map(([n]) => <th key={n} className="px-3 py-2 text-right font-medium">{METRICS[n]?.split(",")[0] ?? n}</th>)}
                <th className="px-3 py-2 text-right font-medium">p95</th>
              </tr>
            </thead>
            <tbody>
              {names.map((n) => (
                <tr key={n} className={`border-b border-rule last:border-0 ${n === report.proposed ? "bg-garnet-soft" : ""}`}>
                  <th scope="row" className="px-3 py-2 text-left font-semibold">{n.replace(/_/g, " ")}</th>
                  <Cell v={report.online[n].purchase_lift} />
                  <Cell v={report.online[n].gmv_lift} />
                  {rows.map(([m]) => <Cell key={m} v={report.offline[n][m] / report.offline[report.control][m] - 1} />)}
                  <td className="px-3 py-2 text-right">{report.online[n].p95_ms.toFixed(1)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="ab" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <h2 id="ab" className="sr-only">Experiments</h2>
        <Fact big={pct(ab.mean_lift)} label={`mean A/B lift over ${ab.replications} buyer splits; the rollout gave ${pct(ab.rollout_lift)}`} />
        <Fact big={`${ab.covered} of ${ab.replications}`} label="A/B intervals that contain the rollout lift" />
        <Fact big={share(report.aa.plain, 1)} label={`A/A tests called significant at 5%, over ${report.aa.splits} splits`} />
        <Fact big={share(ab.cuped_variance_reduction, 1)} label="variance CUPED removed with pre-period purchases" />
      </section>
    </div>
  );
}

function Cell({ v }: { v: number }) {
  return <td className={`px-3 py-2 text-right ${v > 0.005 ? "text-gain" : v < -0.005 ? "text-loss" : ""}`}>{pct(v)}</td>;
}

function Fact({ big, label }: { big: string; label: string }) {
  return (
    <div className="rounded-[var(--radius-field)] bg-card p-4">
      <p className="font-[family-name:var(--font-tag)] text-2xl">{big}</p>
      <p className="mt-1 text-sm text-muted">{label}</p>
    </div>
  );
}
