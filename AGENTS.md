# AGENTS.md

How to work in this repository, for people and coding agents alike.

## Fast feedback

`make check` lints, runs the tests and checks the README matches the reports: under ten seconds, no data needed.
The full dev study (`make all`) takes about twelve minutes on two cores; run it when a change can move a number.

## Layout

| Path | What it is | May import |
|---|---|---|
| `src/preloved/world` | the simulated marketplace: sellers, buyers, their hidden tastes, and the truth | numpy, taxonomy |
| `src/preloved/search` | the product: query understanding, BM25, two towers, ranker, the composable stack | search only |
| `src/preloved/measure` | the product's measurement: offline metrics, A/B statistics, interleaving | search, measure |
| `src/preloved/service` | the FastAPI search service | search |
| `src/preloved/lab` | the bridge that runs the world through the stack, and the lab-only judgments | everything |
| `scripts/` | history, train, evaluate, readme, heldout | lab |
| `config/` | `world.toml` (the world), `stack.toml` and `designs.toml` (the product) | |
| `console/` | the React console over the service | |

## Invariants

1. **The truth stays in the world.** Nothing under `search`, `measure` or `service` imports `world` or `lab`. Items'
   true attributes, appeal and buyers' intents never leave the world except through `lab/truth.py`, which only the
   evaluation script uses, and only for the judged-relevance metric (a stand-in for a rater panel).
2. **The world does not read the product's settings.** `world/` never opens `stack.toml` or `designs.toml`; the
   world's own page size and behaviour live in `world.toml`.
3. **Same luck for every design.** Every random draw that decides a search, a click, a purchase, a sale elsewhere or
   an expiry is keyed by what it is about (search, item, day), never by call order. Two forks served different pages
   still get the same searches tomorrow. A test checks.
4. **The model reads and code decides.** Learned models score; code applies the filters, the availability rule, the
   relaxation and the gate.
5. **Settings in config.** No number that a person might tune lives in code.
6. **Generated reports.** `reports/*.json` come from `scripts/evaluate.py`; the README dashboard comes from
   `scripts/readme.py`. Never edit either by hand; `make check` fails if the README is stale.

## The held-out rule

Every design choice is made on the dev world (seed 41). The held-out world (`config/heldout.toml`, seed 977, a
messier catalogue) runs once through `scripts/heldout.py`, which writes a lock. Its numbers are reported as they came
out, including where they disagree with dev. See docs/heldout.md.
