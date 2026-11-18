"""Write the README dashboard from reports/*.json. Never edit the dashboard by hand.

python3 scripts/readme.py           rewrite the block between the dashboard markers
python3 scripts/readme.py --check   fail if the README is not what the reports say
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- dashboard:start -->", "<!-- dashboard:end -->"
METRICS = {
    "judged_dcg10": "judged relevance of the top 10 (rater labels)",
    "recall_today": "bought item in top 20, against today's tables",
    "ndcg_clicks": "click nDCG on the logged searches",
    "ndcg_ips": "click nDCG, clicks weighted for position",
    "recall_pit": "bought item in top 20, point in time",
}
WHAT = {
    "lexical": "today: BM25 over the seller's title",
    "understood": "+ query understanding, slot filters",
    "hybrid": "+ two-tower vectors, RRF",
    "personalised": "+ personalised ranker, IPS (proposed)",
    "biased_ranker": "ablation: ranker without IPS",
    "id_tower": "ablation: item-ID tower",
    "not_personal": "ablation: ranker without buyer history",
    "no_recency": "variant: no recency boost",
    "no_fuzzy_filters": "variant: slots as words only, no filters",
    "vectors_heavy": "variant: vectors weighted 3x",
    "shallow_rerank": "variant: rerank top 30 only",
    "no_relax": "variant: never relax filters",
}


def pct(v: float, d: int = 1) -> str:
    return f"{v * 100:+.{d}f}%"


def share(v: float, d: int = 0) -> str:
    return f"{v * 100:.{d}f}%"


def dashboard(dev: dict, held: dict | None) -> str:
    c, p = dev["control"], dev["proposed"]
    worlds = [("dev", dev)] + ([("held-out", held)] if held else [])
    L = []
    L.append("### Headline\n")
    L.append("| | " + " | ".join(n for n, _ in worlds) + " |")
    L.append("|---|" + "---:|" * len(worlds))

    def row(label, f):
        L.append(f"| {label} | " + " | ".join(f(r) for _, r in worlds) + " |")

    row(f"purchases per search, `{p}` vs `{c}`, full rollout", lambda r: pct(r["online"][p]["purchase_lift"]))
    row("revenue per search, same comparison", lambda r: pct(r["online"][p]["gmv_lift"]))
    row(
        "relevant listings in the top 10 (of 10)",
        lambda r: f"{r['online'][c]['relevant_in_top10']:.1f} → {r['online'][p]['relevant_in_top10']:.1f}",
    )
    row(
        "searches with no results",
        lambda r: f"{share(r['online'][c]['zero_results'], 1)} → {share(r['online'][p]['zero_results'], 1)}",
    )
    row("p95 latency per search (one CPU core, Python)", lambda r: f"{r['online'][p]['p95_ms']:.1f} ms")
    row(
        "best offline predictor of the online lift (Kendall τ)",
        lambda r: "{} ({:.2f})".format(
            *max(((m, v["kendall_tau"]) for m, v in r["study"].items() if isinstance(v, dict)), key=lambda x: x[1])
        ),
    )
    logged = [m for m in METRICS if m != "judged_dcg10"]
    row(
        "log-based offline metrics that scored every new design below today's search",
        lambda r: f"{sum(1 for m in logged if r['study'][m]['sign_agreement'] == 0)} of {len(logged)}",
    )
    row("interleaving vs A/B: searches needed for the same verdict", lambda r: f"{r['efficiency']['ratio']:.0f}× fewer")
    row("A/A tests called significant at 5%", lambda r: share(r["aa"]["plain"], 1))
    row(
        f"A/B intervals containing the rollout lift ({dev['ab_summary']['replications']} splits)",
        lambda r: f"{r['ab_summary']['covered']} of {r['ab_summary']['replications']}",
    )
    row("ships through the gate", lambda r: "yes" if r["gate"]["ship"] else "no")

    for name, r in worlds:
        L.append(
            f"\n### Every design, rolled out to every buyer ({name} world, days {r['days']['online'][0]}–"
            f"{r['days']['online'][1] - 1}, {r['online'][c]['searches']:,} searches each)\n"
        )
        L.append("| design | what changes | purchases / search | revenue / search | relevant in top 10 | p95 ms |")
        L.append("|---|---|---:|---:|---:|---:|")
        for n in sorted(r["online"], key=lambda n: -r["online"][n]["purchase_lift"]):
            o = r["online"][n]
            name_md = f"**{n}**" if n == p else f"`{n}`"
            lift = "—" if n == c else pct(o["purchase_lift"])
            gl = "—" if n == c else pct(o["gmv_lift"])
            L.append(
                f"| {name_md} | {WHAT.get(n, '')} | {o['purchases_per_search']:.4f} ({lift}) | {gl} | "
                f"{o['relevant_in_top10']:.2f} | {o['p95_ms']:.1f} |"
            )

    for name, r in worlds:
        n_cfg = len(r["online"]) - 1
        L.append(
            f"\n### Which offline metric predicts the online result ({name} world, {n_cfg} designs and "
            f"variants, offline on days {r['days']['offline_test'][0]}–{r['days']['offline_test'][1] - 1})\n"
        )
        L.append("| offline metric | Kendall τ with the rollout lift | p | right direction vs today | picks |")
        L.append("|---|---:|---:|---:|---|")
        for m, v in sorted(
            ((m, v) for m, v in r["study"].items() if isinstance(v, dict)), key=lambda x: -x[1]["kendall_tau"]
        ):
            L.append(
                f"| {METRICS.get(m, m)} | {v['kendall_tau']:.2f} | {v['p']:.3f} | "
                f"{share(v['sign_agreement'])} | `{v['picks']}` |"
            )
        L.append(f"\nThe rollout winner: `{r['study']['online_winner']}`. Offline values per design:\n")
        L.append("| design | " + " | ".join(m for m in METRICS) + " |")
        L.append("|---|" + "---:|" * len(METRICS))
        for n in r["offline"]:
            L.append(f"| `{n}` | " + " | ".join(f"{r['offline'][n][m]:.3f}" for m in METRICS) + " |")

    L.append("\n### Experiments on the proposed design\n")
    L.append("| | " + " | ".join(n for n, _ in worlds) + " |")
    L.append("|---|" + "---:|" * len(worlds))
    row("rollout lift (the truth an A/B test estimates)", lambda r: pct(r["ab_summary"]["rollout_lift"]))
    row(
        "A/B lift, mean over buyer splits (± spread)",
        lambda r: f"{pct(r['ab_summary']['mean_lift'])} ± {r['ab_summary']['spread'] * 100:.1f}",
    )
    row(
        "A/B splits significant at 5% (CUPED)",
        lambda r: f"{r['ab_summary']['significant']} of {r['ab_summary']['replications']}",
    )
    row(
        "variance removed by CUPED (pre-period purchases)",
        lambda r: share(r["ab_summary"]["cuped_variance_reduction"], 1),
    )
    row(
        "interference: largest change in an arm's purchases vs the same searches in a rollout",
        lambda r: share(r["ab_summary"]["max_interference"], 1),
    )
    row(
        "A/A false positives, plain / CUPED",
        lambda r: f"{share(r['aa']['plain'], 1)} / {share(r['aa']['cuped'], 1)} of {r['aa']['splits']}",
    )
    row(
        "interleaving: share of decided searches won by the proposed design",
        lambda r: share(r["efficiency"]["interleaving"]["b_share"], 1),
    )
    row(
        "searches for an 80%-power verdict, interleaving / A/B",
        lambda r: (
            f"{r['efficiency']['interleaving']['searches_needed']:,.0f} / "
            f"{r['efficiency']['ab_cuped']['searches_needed']:,.0f}"
        ),
    )

    world = tomllib.loads((ROOT / "config" / "world.toml").read_text())
    decay = world["behaviour"]["exam_decay"]
    L.append("\n### What the models learned\n")
    for name, _r in worlds:
        t = (
            json.loads((ROOT / "reports" / f"training-{name.replace('held-out', 'heldout')}.json").read_text())
            if (ROOT / "reports" / f"training-{name.replace('held-out', 'heldout')}.json").exists()
            else None
        )
        if t:
            est = t["position_bias"][:5]
            true = [1 / (1 + k) ** decay for k in range(5)]
            L.append(
                f"- Position bias ({name}), positions 1–5, estimated from the 5% shuffled searches: "
                + ", ".join(f"{e:.2f}" for e in est)
                + "; the world's true curve: "
                + ", ".join(f"{x:.2f}" for x in true)
                + "."
            )
    for name, r in worlds:
        if r.get("nicknames"):
            L.append(
                f"- Nicknames no dictionary in the stack contains ({name}): share of the content tower's top 50 "
                "that is the intended brand: "
                + ", ".join(f"“{k}” → {v['brand']} {share(v['top50_share'])}" for k, v in r["nicknames"].items())
                + "."
            )
    L.append("\n### Gate (config/stack.toml)\n")
    L.append("| check | " + " | ".join(n for n, _ in worlds) + " |")
    L.append("|---|" + "---|" * len(worlds))
    for k in dev["gate"]:
        if k != "ship":
            L.append(
                f"| {k.replace('_', ' ')} | "
                + " | ".join("pass" if r["gate"][k] else "**fail**" for _, r in worlds)
                + " |"
            )
    L.append(
        f"\n_Generated by `scripts/readme.py` from `reports/*.json`; evaluation ran in "
        f"{dev['runtime_s'] // 60} min on two cores._"
    )
    return "\n".join(L)


def main() -> None:
    dev = json.loads((ROOT / "reports" / "dev.json").read_text())
    hp = ROOT / "reports" / "heldout.json"
    held = json.loads(hp.read_text()) if hp.exists() else None
    text = README.read_text()
    a, b = text.index(START) + len(START), text.index(END)
    new = text[:a] + "\n" + dashboard(dev, held) + "\n" + text[b:]
    if "--check" in sys.argv:
        if new != text:
            sys.exit("README dashboard is stale: run `python3 scripts/readme.py`")
        print("README matches the reports")
        return
    README.write_text(new)
    print("README dashboard written")


if __name__ == "__main__":
    main()
