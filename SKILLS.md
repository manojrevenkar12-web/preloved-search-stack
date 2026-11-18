# SKILLS.md

Repeatable tasks, each with the commands and what done looks like.

## Add a design or a variant
1. Add a `[designs.<name>]` or `[variants.<name>]` table to `config/designs.toml`: it configures stages that exist.
2. `make evaluate readme`. Done when the new row is in the README tables and `make check` passes.

## Add a stage to the stack
1. Add the stage to `Stack.search` behind a `Design` field with a default that keeps every existing design unchanged.
2. Add a test in `tests/test_stack.py`, then a design that uses it. Re-run the dev study and check the old rows did
   not move (same luck means they should be identical).

## Add an offline metric
1. Write it in `src/preloved/measure/offline.py` from the log only; add it to `OFFLINE_METRICS` in
   `scripts/evaluate.py`. It joins the Kendall-tau study automatically.
2. A metric that needs the truth belongs in `src/preloved/lab/truth.py` and is labelled as a rater-panel stand-in.

## Retrain the models
`make train` (two towers and three rankers, about two minutes). The model cards in `models/<world>/*/card.json`
record the data, parameters and loss. The Airflow DAG in `dags/` runs the same scripts nightly and publishes only when
the gate says ship.

## Run the held-out world
Once, at the end: `make heldout readme`. Never re-run it to get a better number; see docs/heldout.md.
