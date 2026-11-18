"""Nightly retraining of the search models, as an Airflow DAG.

    land the day's event log -> train the two towers and the rankers -> evaluate offline and gate -> publish

The tasks call the same scripts a developer runs (Makefile). Publishing is the only step that changes production,
and it runs only when the gate in reports/<world>.json says ship.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

HOME = Path(__file__).resolve().parents[1]


def _gate_says_ship(world: str = "dev") -> bool:
    report = json.loads((HOME / "reports" / f"{world}.json").read_text())
    return bool(report["gate"]["ship"])


with DAG(
    dag_id="preloved_search_retrain",
    start_date=datetime(2026, 11, 1),
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["search", "ml"],
) as dag:
    land = BashOperator(task_id="land_event_log", bash_command=f"cd {HOME} && python3 scripts/history.py")
    train = BashOperator(task_id="train_models", bash_command=f"cd {HOME} && python3 scripts/train.py")
    evaluate = BashOperator(task_id="evaluate_and_gate", bash_command=f"cd {HOME} && python3 scripts/evaluate.py")
    gate = ShortCircuitOperator(task_id="gate", python_callable=_gate_says_ship)
    publish = BashOperator(
        task_id="publish_models",
        bash_command=f"cd {HOME} && aws s3 sync models/dev s3://$MODEL_BUCKET/search/{{{{ ds }}}}/ --only-show-errors",
    )
    land >> train >> evaluate >> gate >> publish
