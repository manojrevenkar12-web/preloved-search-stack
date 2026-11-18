PY ?= python3
WORLD ?= dev

.PHONY: all history train evaluate readme test lint check heldout serve console

all: history train evaluate readme          ## the whole dev study, about 12 minutes on two cores

history:
	$(PY) scripts/history.py --world $(WORLD)
train:
	$(PY) scripts/train.py --world $(WORLD)
evaluate:
	$(PY) scripts/evaluate.py --world $(WORLD)
readme:
	$(PY) scripts/readme.py
test:
	$(PY) -m pytest -q
lint:
	ruff check . && ruff format --check src tests scripts dags
check: lint test                            ## fast feedback: under ten seconds
	$(PY) scripts/readme.py --check
heldout:                                    ## once, after the dev study is final (docs/heldout.md)
	$(PY) scripts/heldout.py
serve:
	uvicorn preloved.service.app:app --port 8000
console:
	cd console && npm ci && npm run dev
