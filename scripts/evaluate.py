"""Every design offline and online, the A/B tests, interleaving and the offline-to-online study.

    python3 scripts/evaluate.py [--world dev|heldout]   -> reports/<world>.json

1. Offline: the logged test days (after the training days) replayed through every design and variant.
2. Online: each design rolled out to every buyer for the evaluation days, in a fork of the same world (same buyers,
   same searches, same luck). The lift over today's search is the truth the offline metrics are judged against.
3. A/B: today's search against the proposed design, split by buyer, several splits; plain and with CUPED.
   A/A on the control rollout gives the false-positive rate.
4. Interleaving: the two rankings mixed on one page, credited per click.
5. The study: for each offline metric, how well it orders the designs the way the online lift does (Kendall tau).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preloved.lab.common import WORLDS, data_dir, designs, load_world, models_dir, settings  # noqa: E402
from preloved.lab.run import run_days  # noqa: E402
from preloved.lab.state import load_models, stack_after_history  # noqa: E402
from preloved.lab.truth import intents, judged_dcg  # noqa: E402
from preloved.measure import abtest, offline  # noqa: E402
from preloved.measure.interleave import preference  # noqa: E402
from preloved.search.qu import QueryUnderstanding  # noqa: E402
from preloved.search.stack import Design, Stack  # noqa: E402
from preloved.world.sim import NICKNAMES  # noqa: E402

LISTING_COLS = [
    "item_id",
    "title",
    "price",
    "market",
    "listed_day",
    "brand",
    "category",
    "model",
    "material",
    "colour",
    "size",
    "condition",
]
CONTROL, PROPOSED = "lexical", "personalised"
OFFLINE_METRICS = ("ndcg_clicks", "ndcg_ips", "recall_pit", "recall_today", "judged_dcg10")
_STATE: dict = {}  # shared with forked workers


def _rollout(job: tuple) -> tuple[str, list[dict]]:
    """One fork of the world served by one policy for the evaluation days."""
    key, kind, arg = job
    world, stack, days, cfg = (_STATE[k] for k in ("world", "stack", "days", "cfg"))
    f, st = world.fork(), copy.deepcopy(stack)
    if kind == "design":
        d = cfg[arg]
        return key, run_days(f, st, days, lambda _: (d.name, d))
    if kind == "ab":
        a, b = cfg[CONTROL], cfg[PROPOSED]
        return key, run_days(
            f, st, days, lambda s: ("B", b) if abtest.split(np.array([s.user_id]), arg)[0] else ("A", a)
        )
    return key, run_days(f, st, days, None, interleave=(cfg[CONTROL], cfg[PROPOSED]))


def _per_user(recs: list[dict], users_filter=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.array([r["user_id"] for r in recs])
    y = np.array([r["purchases"] for r in recs], dtype=float)
    keep = np.ones(len(u), dtype=bool) if users_filter is None else users_filter(recs)
    users, inv = np.unique(u[keep], return_inverse=True)
    return users, np.bincount(inv, weights=y[keep]), np.bincount(inv).astype(float)


def summary(recs: list[dict]) -> dict:
    n = len(recs)
    ms = np.array([r["ms"] for r in recs])
    return {
        "searches": n,
        "purchases_per_search": sum(r["purchases"] for r in recs) / n,
        "gmv_per_search": sum(r["gmv"] for r in recs) / n,
        "clicks_per_search": sum(r["clicks"] for r in recs) / n,
        "relevant_in_top10": float(np.mean([r["relevant10"] for r in recs])),
        "zero_results": float(np.mean([r["shown"] == 0 for r in recs])),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=list(WORLDS), default="dev")
    args = ap.parse_args()
    t_start = time.perf_counter()
    s = settings()
    D, V = designs()
    cfg = {n: Design.from_config(n, c) for n, c in {**D, **V}.items()}
    world = load_world(args.world)
    models = load_models(args.world)
    logs = data_dir(args.world) / "logs"
    train_days, hist = s["timeline"]["train_days"], world.cfg["timeline"]["history_days"]
    eval_days = range(hist, hist + world.cfg["timeline"]["eval_days"])
    exam = np.array(json.loads((models_dir(args.world) / "training.json").read_text())["position_bias"])

    # ---- 1. offline ---------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    off = offline.load(logs, range(train_days, hist), LISTING_COLS)
    st = Stack(s, towers=models[0], rankers=models[1])
    st.add_listings([x for x in off.listings if x["listed_day"] < train_days])
    for d in range(train_days):
        st.observe_clicks(off.day_clicks.get(d, []))
    ranked = offline.replay(st, cfg, off)
    today = offline.replay_today(stack_after_history(world, args.world, models), cfg, off, set(off.purchases))
    truth = intents(ROOT / "config" / WORLDS[args.world], range(train_days, hist))
    assert all(truth[sid]["query"] == q for sid, _, _, _, q in off.searches), "the replayed searches match the log"
    off_metrics = {}
    for name in cfg:
        m = offline.metrics(ranked[name], off, exam, today[name])
        m["judged_dcg10"] = judged_dcg(world, ranked[name], truth)
        off_metrics[name] = m
    print(f"offline: {len(off.searches)} searches x {len(cfg)} designs in {time.perf_counter() - t0:.0f} s")

    # ---- 2-4. online: full rollouts, A/B splits, interleaving ---------------------------------------------
    t0 = time.perf_counter()
    _STATE.update(world=world, stack=stack_after_history(world, args.world, models), days=eval_days, cfg=cfg)
    reps = s["experiment"]["ab_replications"]
    jobs = (
        [(n, "design", n) for n in cfg]
        + [(f"ab{r}", "ab", 100 + r) for r in range(reps)]
        + [("interleave", "interleave", None)]
    )
    with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count()), mp_context=mp.get_context("fork")) as ex:
        res = dict(ex.map(_rollout, jobs))
    print(f"online: {len(jobs)} forks x {len(eval_days)} days in {time.perf_counter() - t0:.0f} s")
    runs = data_dir(args.world) / "online"
    runs.mkdir(exist_ok=True)
    for k, recs in res.items():  # every served search, for the tests and for anyone checking the numbers
        pd.DataFrame(recs).to_parquet(runs / f"{k}.parquet", index=False)

    online = {n: summary(res[n]) for n in cfg}
    base = online[CONTROL]["purchases_per_search"]
    for n in cfg:
        online[n]["purchase_lift"] = online[n]["purchases_per_search"] / base - 1
        online[n]["gmv_lift"] = online[n]["gmv_per_search"] / online[CONTROL]["gmv_per_search"] - 1

    pre = duckdb.sql(f"""select s.user_id, count(p.item_id) from read_parquet('{logs / "searches.parquet"}') s
                         left join read_parquet('{logs / "purchases.parquet"}') p using (search_id)
                         group by 1""").fetchall()
    pre_buys = dict(pre)
    alpha = s["experiment"]["alpha"]
    true_lift = online[PROPOSED]["purchase_lift"]
    ab = []
    for r in range(reps):
        recs = res[f"ab{r}"]
        arm = {}
        for k in ("A", "B"):
            users, y, n = _per_user(recs, lambda rr, k=k: np.array([x["arm"] == k for x in rr]))
            arm[k] = (y, n, np.array([pre_buys.get(int(u), 0) for u in users], dtype=float))
        plain = abtest.ratio_test(arm["A"][0], arm["A"][1], arm["B"][0], arm["B"][1], alpha=alpha)
        cuped = abtest.ratio_test(*arm["A"][:2], *arm["B"][:2], arm["A"][2], arm["B"][2], alpha=alpha)
        # interference: the same searches under the full rollouts. Luck is keyed per search, so any difference is
        # one arm's buyers taking the one-of-a-kind items the other arm's buyers would have bought.
        bought = {k: {x["search_id"]: x["purchases"] for x in res[k]} for k in (CONTROL, PROPOSED)}
        a_ids = [x["search_id"] for x in recs if x["arm"] == "A"]
        b_ids = [x["search_id"] for x in recs if x["arm"] == "B"]
        got = {x["search_id"]: x["purchases"] for x in recs}
        interference = {
            "control": sum(got[i] for i in a_ids) / max(sum(bought[CONTROL][i] for i in a_ids), 1) - 1,
            "treatment": sum(got[i] for i in b_ids) / max(sum(bought[PROPOSED][i] for i in b_ids), 1) - 1,
        }
        ab.append(
            {
                "split": 100 + r,
                "searches": len(recs),
                "plain": plain,
                "cuped": cuped,
                "cuped_variance_reduction": 1 - ((cuped["high"] - cuped["low"]) / (plain["high"] - plain["low"])) ** 2,
                "interference": interference,
                "covers_rollout_lift": bool(cuped["low"] <= true_lift <= cuped["high"]),
            }
        )
    users, y, n = _per_user(res[CONTROL])
    aa = abtest.aa_false_positives(
        users,
        y,
        n,
        np.array([pre_buys.get(int(u), 0) for u in users], dtype=float),
        s["experiment"]["aa_splits"],
        alpha,
    )
    il = res["interleave"]
    pref = preference(np.array([r["wins_a"] for r in il]), np.array([r["wins_b"] for r in il]))
    z_il = (pref["b_share"] - 0.5) / np.sqrt(0.25 / max(pref["decided"], 1))
    z_ab = float(np.mean([a["cuped"]["z"] for a in ab]))
    efficiency = {
        "interleaving": {
            "searches": len(il),
            "z": float(z_il),
            **pref,
            "searches_needed": abtest.searches_needed(z_il, len(il)),
        },
        "ab_cuped": {
            "searches": ab[0]["searches"],
            "z": z_ab,
            "searches_needed": abtest.searches_needed(z_ab, ab[0]["searches"]),
        },
    }
    ab_summary = {
        "replications": reps,
        "rollout_lift": true_lift,
        "mean_lift": float(np.mean([a["cuped"]["lift"] for a in ab])),
        "spread": float(np.std([a["cuped"]["lift"] for a in ab], ddof=1)),
        "covered": sum(a["covers_rollout_lift"] for a in ab),
        "significant": sum(a["cuped"]["low"] > 0 for a in ab),
        "max_interference": float(max(max(abs(v) for v in a["interference"].values()) for a in ab)),
        "cuped_variance_reduction": float(np.mean([a["cuped_variance_reduction"] for a in ab])),
    }
    efficiency["ratio"] = efficiency["ab_cuped"]["searches_needed"] / efficiency["interleaving"]["searches_needed"]

    # ---- 5. which offline metric predicts the online lift --------------------------------------------------
    others = [n for n in cfg if n != CONTROL]
    on = np.array([online[n]["purchase_lift"] for n in others])
    study = {}
    for m in OFFLINE_METRICS:
        offv = np.array([off_metrics[n][m] / off_metrics[CONTROL][m] - 1 for n in others])
        tau = stats.kendalltau(offv, on)
        study[m] = {
            "kendall_tau": float(tau.statistic),
            "p": float(tau.pvalue),
            "picks": others[int(np.argmax(offv))],
            "sign_agreement": float(np.mean(np.sign(offv) == np.sign(on))),
        }
    study["online_winner"] = others[int(np.argmax(on))]

    # what the content tower learned from clicks alone: nicknames no dictionary in the stack contains
    tower, items = models[0]["content"], [it.listing() for it in world.items]
    vecs, qu = tower.encode_items(items), QueryUnderstanding()
    nick = {}
    for brand, forms in NICKNAMES.items():
        for form in forms:
            if not qu.parse(form).slots().get("brand"):
                top = np.argsort(-(vecs @ tower.encode_query(form, qu.parse(form))))[:50]
                nick[form] = {
                    "brand": brand,
                    "top50_share": float(np.mean([world.items[i].true["brand"] == brand for i in top])),
                }

    g, p = s["gate"], online[PROPOSED]
    gate = {
        "ab_lower_bound": min(a["cuped"]["low"] for a in ab) > g["min_purchase_lift"],
        "zero_results": p["zero_results"] <= g["max_zero_result_rate"],
        "p95_ms": p["p95_ms"] <= g["max_p95_ms"],
    }
    gate["ship"] = all(gate.values())

    report = {
        "world": args.world,
        "seed": world.seed,
        "control": CONTROL,
        "proposed": PROPOSED,
        "days": {"train": train_days, "offline_test": [train_days, hist], "online": [eval_days.start, eval_days.stop]},
        "designs": list(D),
        "variants": list(V),
        "offline": off_metrics,
        "online": online,
        "ab": ab,
        "ab_summary": ab_summary,
        "aa": aa,
        "efficiency": efficiency,
        "study": study,
        "nicknames": nick,
        "gate": gate,
        "runtime_s": round(time.perf_counter() - t_start),
    }
    out = ROOT / "reports" / f"{args.world}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_round(report), indent=2) + "\n")
    print(
        json.dumps(
            _round(
                {
                    "online": {n: (online[n]["purchase_lift"], online[n]["p95_ms"]) for n in cfg},
                    "study": study,
                    "gate": gate,
                    "eff": efficiency["ratio"],
                    "aa": aa,
                    "ab_summary": ab_summary,
                    "ab": [
                        (
                            a["cuped"]["lift"],
                            a["cuped"]["low"],
                            a["cuped"]["high"],
                            a["plain"]["low"],
                            a["plain"]["high"],
                        )
                        for a in ab
                    ],
                }
            ),
            indent=1,
        )
    )


def _round(x):
    if isinstance(x, float):
        return round(x, 5) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v) for v in x]
    if isinstance(x, np.generic):
        return _round(x.item())
    return x


if __name__ == "__main__":
    main()
