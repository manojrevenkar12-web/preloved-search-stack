# preloved-search-stack

Multistage composable search for a pre-loved luxury marketplace where every item is one of a kind: multilingual query
understanding, hybrid OpenSearch + vector retrieval, personalised PyTorch re-ranking, and an offline-to-online
framework that tests which offline metrics predict A/B results.

Built by [Manoj Kumar Prakash](https://www.linkedin.com/in/manoj-kumar-prakash-52b7a01b6/) as a portfolio project
for the Senior Data Scientist / Applied Scientist role in Discovery at Vestiaire Collective. The marketplace is
simulated (see [Limits](#limits)); the code is written to run against a real one.

## The problem

A resale catalogue breaks the assumptions most search and recommendation stacks are built on:

- **Every item sells once.** An embedding per item ID never gets enough clicks, and a search that ranks a sold bag
  first is a dead end. Retrieval has to work from what the listing says, and only from what is still for sale.
- **Sellers write titles in their own language.** A French seller's "Sac Timeless cuir noir" has no word in common
  with an Italian buyer's "borsa chanel nera". Attributes the seller filled in are missing or wrong some of the time.
- **Buyers type nicknames and typos.** "loubs 38", "bv bag", "birkn togo".
- **Offline metrics come from the old ranking's logs.** Clicks only exist on what today's search showed, at the
  positions it showed them. A team needs to know which offline numbers it can trust before it spends A/B traffic.

## What is here

```mermaid
flowchart LR
  Q[query] --> QU[understand<br/>slots, typos, 5 languages]
  QU --> F[filter<br/>for sale + brand, category<br/>relaxed if too narrow]
  F --> B[BM25<br/>title + canonical slots]
  F --> V[two towers<br/>content, not item IDs]
  B --> R[RRF fusion<br/>+ recency]
  V --> R
  R --> RR[personalised ranker<br/>PyTorch, IPS-trained]
  RR --> P[results page]
  P -. clicks, purchases .-> LOG[(event log<br/>parquet / DuckDB)]
  LOG --> T[train: towers, ranker,<br/>position bias] --> V & RR
  LOG --> O[offline metrics] --> S{{offline-to-online study}}
  P --> AB[A/B, CUPED, A/A,<br/>interleaving] --> S
```

- **Composable stages, a design is config** ([ADR 1](docs/adr/0001-composable-stages.md)). `config/designs.toml`
  holds 7 designs and 5 variants; one `Stack.search` serves all of them, and so does the FastAPI service.
- **Query understanding** (`search/qu.py`): longest-phrase matching of brands, models, categories, colours,
  materials, styles and sizes in English, French, German, Italian and Spanish; bounded edit-distance spelling
  correction; ambiguous model names ("kelly", "love") need their brand.
- **Retrieval**: BM25 over the title plus canonical slot tokens (`brand:chanel`), and a content two-tower model
  ([ADR 2](docs/adr/0002-content-tower-not-item-ids.md)) trained in PyTorch on clicks and purchases, served from
  NumPy. The production equivalent is the OpenSearch mapping and query in `search/index.py`.
- **Re-ranking**: an MLP over 18 features (match, price against the buyer's usual price, condition, age, market,
  the buyer's brand and category history), trained listwise and weighted against position bias estimated from 5%
  randomised traffic ([ADR 3](docs/adr/0003-unbiased-ranker.md)).
- **Measurement** (`measure/`): offline replay metrics with point-in-time availability, A/B tests on a ratio metric
  with the delta method and CUPED, A/A false-positive rates, team-draft interleaving, and the study that scores each
  offline metric by how well it predicts the rollout ([ADR 4](docs/adr/0004-offline-to-online.md)).
- **The simulated marketplace** (`world/`): 14,000 starting listings and 900 new ones a day, 6,000 buyers in seven
  markets with hidden tastes, budgets and sizes, position-biased attention, and one purchase per item. Every random
  draw is keyed by search, item and day, so every design is compared on the same buyers and the same luck.

## Results

The dev world (seed 41) shaped every decision; the held-out world (seed 977, a messier catalogue) ran once at the end
([docs/heldout.md](docs/heldout.md)).

<!-- dashboard:start -->
### Headline

| | dev | held-out |
|---|---:|---:|
| purchases per search, `personalised` vs `lexical`, full rollout | +35.7% | +36.3% |
| revenue per search, same comparison | +17.1% | +29.0% |
| relevant listings in the top 10 (of 10) | 7.9 → 9.9 | 7.8 → 9.8 |
| searches with no results | 0.4% → 0.0% | 0.6% → 0.0% |
| p95 latency per search (one CPU core, Python) | 4.6 ms | 4.5 ms |
| best offline predictor of the online lift (Kendall τ) | judged_dcg10 (0.67) | judged_dcg10 (0.66) |
| log-based offline metrics that scored every new design below today's search | 4 of 4 | 4 of 4 |
| interleaving vs A/B: searches needed for the same verdict | 39× fewer | 42× fewer |
| A/A tests called significant at 5% | 5.5% | 5.3% |
| A/B intervals containing the rollout lift (8 splits) | 7 of 8 | 7 of 8 |
| ships through the gate | yes | yes |

### Every design, rolled out to every buyer (dev world, days 21–25, 8,916 searches each)

| design | what changes | purchases / search | revenue / search | relevant in top 10 | p95 ms |
|---|---|---:|---:|---:|---:|
| **personalised** | + personalised ranker, IPS (proposed) | 0.0721 (+35.7%) | +17.1% | 9.88 | 4.6 |
| `no_relax` | variant: never relax filters | 0.0721 (+35.7%) | +17.1% | 9.88 | 5.1 |
| `biased_ranker` | ablation: ranker without IPS | 0.0708 (+33.1%) | +13.4% | 9.87 | 4.4 |
| `shallow_rerank` | variant: rerank top 30 only | 0.0703 (+32.3%) | +18.4% | 9.85 | 6.0 |
| `not_personal` | ablation: ranker without buyer history | 0.0700 (+31.6%) | +12.1% | 9.88 | 4.2 |
| `understood` | + query understanding, slot filters | 0.0622 (+17.1%) | +10.4% | 9.35 | 2.5 |
| `hybrid` | + two-tower vectors, RRF | 0.0617 (+16.0%) | +9.7% | 9.51 | 3.8 |
| `id_tower` | ablation: item-ID tower | 0.0607 (+14.1%) | +2.4% | 9.51 | 4.4 |
| `vectors_heavy` | variant: vectors weighted 3x | 0.0592 (+11.4%) | +4.4% | 9.54 | 4.1 |
| `no_fuzzy_filters` | variant: slots as words only, no filters | 0.0585 (+10.1%) | +5.1% | 8.71 | 1.3 |
| `no_recency` | variant: no recency boost | 0.0544 (+2.3%) | +3.2% | 7.94 | 0.7 |
| `lexical` | today: BM25 over the seller's title | 0.0532 (—) | — | 7.90 | 0.7 |

### Every design, rolled out to every buyer (held-out world, days 21–25, 8,879 searches each)

| design | what changes | purchases / search | revenue / search | relevant in top 10 | p95 ms |
|---|---|---:|---:|---:|---:|
| `not_personal` | ablation: ranker without buyer history | 0.0737 (+38.0%) | +28.0% | 9.85 | 4.7 |
| **personalised** | + personalised ranker, IPS (proposed) | 0.0728 (+36.3%) | +29.0% | 9.85 | 4.5 |
| `no_relax` | variant: never relax filters | 0.0728 (+36.3%) | +29.0% | 9.85 | 4.3 |
| `biased_ranker` | ablation: ranker without IPS | 0.0716 (+34.2%) | +28.8% | 9.84 | 4.8 |
| `shallow_rerank` | variant: rerank top 30 only | 0.0711 (+33.1%) | +25.2% | 9.84 | 4.0 |
| `hybrid` | + two-tower vectors, RRF | 0.0659 (+23.4%) | +23.3% | 9.49 | 3.9 |
| `id_tower` | ablation: item-ID tower | 0.0651 (+21.9%) | +23.9% | 9.54 | 4.4 |
| `understood` | + query understanding, slot filters | 0.0631 (+18.1%) | +10.7% | 9.30 | 2.5 |
| `vectors_heavy` | variant: vectors weighted 3x | 0.0609 (+14.1%) | +13.5% | 9.55 | 4.0 |
| `no_fuzzy_filters` | variant: slots as words only, no filters | 0.0600 (+12.4%) | +5.8% | 8.70 | 1.4 |
| `no_recency` | variant: no recency boost | 0.0559 (+4.6%) | +11.6% | 7.87 | 0.7 |
| `lexical` | today: BM25 over the seller's title | 0.0534 (—) | — | 7.84 | 0.7 |

### Which offline metric predicts the online result (dev world, 11 designs and variants, offline on days 18–20)

| offline metric | Kendall τ with the rollout lift | p | right direction vs today | picks |
|---|---:|---:|---:|---|
| judged relevance of the top 10 (rater labels) | 0.67 | 0.005 | 100% | `not_personal` |
| bought item in top 20, against today's tables | 0.41 | 0.085 | 0% | `understood` |
| click nDCG on the logged searches | 0.37 | 0.117 | 0% | `understood` |
| click nDCG, clicks weighted for position | 0.37 | 0.117 | 0% | `understood` |
| bought item in top 20, point in time | 0.24 | 0.307 | 0% | `understood` |

The rollout winner: `personalised`. Offline values per design:

| design | judged_dcg10 | recall_today | ndcg_clicks | ndcg_ips | recall_pit |
|---|---:|---:|---:|---:|---:|
| `lexical` | 0.686 | 0.856 | 0.676 | 0.548 | 1.000 |
| `understood` | 0.818 | 0.774 | 0.587 | 0.487 | 0.868 |
| `hybrid` | 0.813 | 0.679 | 0.528 | 0.435 | 0.790 |
| `personalised` | 0.831 | 0.757 | 0.558 | 0.463 | 0.827 |
| `biased_ranker` | 0.827 | 0.753 | 0.571 | 0.472 | 0.831 |
| `id_tower` | 0.707 | 0.247 | 0.192 | 0.172 | 0.259 |
| `not_personal` | 0.836 | 0.733 | 0.555 | 0.460 | 0.831 |
| `no_recency` | 0.699 | 0.588 | 0.440 | 0.371 | 0.716 |
| `no_fuzzy_filters` | 0.779 | 0.770 | 0.581 | 0.481 | 0.860 |
| `vectors_heavy` | 0.798 | 0.663 | 0.506 | 0.416 | 0.749 |
| `shallow_rerank` | 0.834 | 0.720 | 0.558 | 0.461 | 0.794 |
| `no_relax` | 0.831 | 0.757 | 0.558 | 0.463 | 0.827 |

### Which offline metric predicts the online result (held-out world, 11 designs and variants, offline on days 18–20)

| offline metric | Kendall τ with the rollout lift | p | right direction vs today | picks |
|---|---:|---:|---:|---|
| judged relevance of the top 10 (rater labels) | 0.66 | 0.005 | 100% | `biased_ranker` |
| bought item in top 20, against today's tables | 0.56 | 0.019 | 0% | `biased_ranker` |
| bought item in top 20, point in time | 0.31 | 0.183 | 0% | `biased_ranker` |
| click nDCG on the logged searches | 0.29 | 0.212 | 0% | `understood` |
| click nDCG, clicks weighted for position | 0.29 | 0.212 | 0% | `understood` |

The rollout winner: `not_personal`. Offline values per design:

| design | judged_dcg10 | recall_today | ndcg_clicks | ndcg_ips | recall_pit |
|---|---:|---:|---:|---:|---:|
| `lexical` | 0.680 | 0.868 | 0.672 | 0.549 | 0.993 |
| `understood` | 0.815 | 0.750 | 0.570 | 0.476 | 0.844 |
| `hybrid` | 0.811 | 0.667 | 0.517 | 0.430 | 0.792 |
| `personalised` | 0.832 | 0.757 | 0.549 | 0.457 | 0.823 |
| `biased_ranker` | 0.837 | 0.774 | 0.568 | 0.473 | 0.854 |
| `id_tower` | 0.723 | 0.281 | 0.182 | 0.164 | 0.319 |
| `not_personal` | 0.833 | 0.760 | 0.551 | 0.460 | 0.837 |
| `no_recency` | 0.692 | 0.604 | 0.433 | 0.365 | 0.722 |
| `no_fuzzy_filters` | 0.774 | 0.740 | 0.565 | 0.470 | 0.844 |
| `vectors_heavy` | 0.796 | 0.653 | 0.497 | 0.412 | 0.750 |
| `shallow_rerank` | 0.832 | 0.708 | 0.545 | 0.454 | 0.795 |
| `no_relax` | 0.832 | 0.757 | 0.549 | 0.457 | 0.819 |

### Experiments on the proposed design

| | dev | held-out |
|---|---:|---:|
| rollout lift (the truth an A/B test estimates) | +35.7% | +36.3% |
| A/B lift, mean over buyer splits (± spread) | +41.0% ± 10.1 | +35.0% ± 10.4 |
| A/B splits significant at 5% (CUPED) | 8 of 8 | 8 of 8 |
| variance removed by CUPED (pre-period purchases) | 0.0% | -0.1% |
| interference: largest change in an arm's purchases vs the same searches in a rollout | 1.3% | 2.5% |
| A/A false positives, plain / CUPED | 5.5% / 5.5% of 1000 | 5.3% / 5.3% of 1000 |
| interleaving: share of decided searches won by the proposed design | 68.6% | 66.7% |
| searches for an 80%-power verdict, interleaving / A/B | 107 / 4,182 | 128 / 5,384 |

### What the models learned

- Position bias (dev), positions 1–5, estimated from the 5% shuffled searches: 1.00, 0.53, 0.37, 0.28, 0.23; the world's true curve: 1.00, 0.54, 0.37, 0.29, 0.23.
- Position bias (held-out), positions 1–5, estimated from the 5% shuffled searches: 1.00, 0.54, 0.37, 0.29, 0.23; the world's true curve: 1.00, 0.54, 0.37, 0.29, 0.23.
- Nicknames no dictionary in the stack contains (dev): share of the content tower's top 50 that is the intended brand: “bv” → Bottega Veneta 100%, “loubs” → Christian Louboutin 98%, “slp” → Saint Laurent 100%.
- Nicknames no dictionary in the stack contains (held-out): share of the content tower's top 50 that is the intended brand: “bv” → Bottega Veneta 100%, “loubs” → Christian Louboutin 98%, “slp” → Saint Laurent 100%.

### Gate (config/stack.toml)

| check | dev | held-out |
|---|---|---|
| ab lower bound | pass | pass |
| zero results | pass | pass |
| p95 ms | pass | pass |

_Generated by `scripts/readme.py` from `reports/*.json`; evaluation ran in 8 min on two cores._
<!-- dashboard:end -->

## What the numbers say

- **Understanding the query and re-ranking carry the gain, in both worlds.** Slots as filters and canonical tokens
  add 17–18% purchases per search; the ranker takes the total to about 36%. Training the ranker without the position
  correction costs 2–3 points in both worlds.
- **Personalisation is not established.** The buyer's own history added 4 points on dev and lost 1.7 on the held-out
  world, where the ranker without it was the rollout winner. Both are inside the noise of one rollout. The proposed
  design still clears the gate, and the next experiment is the personalised ranker against the anonymous one.
- **Vectors pay off as the catalogue gets messier.** On dev the two towers added no purchases over query
  understanding; on the held-out world, with a fifth of attributes missing and more nicknames, they added 5 points.
  The content tower learned "bv", "loubs" and "slp" from clicks alone. An item-ID tower lost most of the revenue gain
  on dev (+2% against +17%), because most items sell before they collect clicks.
- **Every log-based offline metric preferred today's search, in both worlds.** The logged pages are today's ranking,
  so any metric built from them rewards reproducing it. Among the new designs they still order things somewhat; only
  judged relevance (rater labels, here from the simulation's truth) got the direction right for every design. In
  practice that argues for a small rated query set, and for interleaving before an A/B test.
- **Interleaving reaches a verdict on about 40× fewer searches than an A/B test.** Each buyer compares both rankings
  on the same query, so buyer-to-buyer variance cancels.
- **CUPED did nothing here, and the report says so.** Buyers in this world have no lasting propensity to buy that
  last month's purchases reveal. On a real marketplace with repeat buyers the same code should remove much more
  variance; measure it before sizing an experiment on it.
- **Interference was small.** With about 22,000 items for sale and about 90 purchases a day, one arm of an A/B test
  rarely buys the item the other arm wanted; measured search by search, no arm moved more than 2.5%.
- **A known weak spot.** The tower reads "bv" as Bottega Veneta, but for "bv bag" BM25's matches on "bag" outrank it
  in the fusion, so the page shows other brands (recorded in the console demo). A fix is to let a confident learned
  brand read propose a soft slot; it would have to be tested on dev and a new held-out world.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[train,dev]"
make check        # lint, tests, README matches the reports: seconds
make all          # history -> train -> evaluate -> README, about 12 minutes on two cores
make serve        # FastAPI on :8000   GET /search?q=sac+chanel+noir&design=personalised, /designs, /report
make console      # the React console; reads console/public/demo.json, or the service with VITE_API_URL
```

Airflow (`dags/search_retrain.py`) runs the same scripts nightly and publishes models only when the gate says ship.
`infra/terraform` describes the AWS side (S3 for model artifacts, OpenSearch for listings, ECR for the service image);
CI validates it but never applies it.

![console: the same query under today's search and the proposed design](docs/console.png)

![console: offline metrics against the rollout](docs/study.png)

## Limits

- **The marketplace is simulated.** Buyers, sellers and their behaviour are a model with stated rules
  (`world/sim.py`, `config/world.toml`). The numbers show how the methods compare under those rules, not what they
  would do at Vestiaire Collective.
- **Scale.** About 33,000 listings and 9,000 searches per evaluated design, against a catalogue of millions. BM25 and
  the vectors run in-process on one core; at scale they would be OpenSearch and a vector index, and latency numbers
  would be those systems'.
- **Judged relevance reads the simulation's truth.** It stands in for a rater panel, which costs money and has its own
  noise.
- **The study has twelve configurations and one marketplace.** Kendall's τ over eleven points has wide intervals; the
  p-values are in the table.
- **No images.** Real listings are judged by their photos; the simulation hides an "appeal" per item that no model
  sees.

## Layout

See [AGENTS.md](AGENTS.md) for the layout, the invariants and the held-out rule, and [SKILLS.md](SKILLS.md) for the
repeatable tasks.
