PY := .venv/bin/python

.PHONY: setup bdb ingest features train comps tracking report all test lint app

setup:            ## create the virtualenv and install pinned dependencies
	python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

bdb:              ## download Big Data Bowl 2026 Analytics (the 2024 files were taken down from Kaggle in Aug 2025)
	set -a; . ./.env; set +a; .venv/bin/kaggle competitions list -s "big data bowl"
	set -a; . ./.env; set +a; .venv/bin/kaggle competitions download -c nfl-big-data-bowl-2026-analytics -p data/raw/bdb26 && cd data/raw/bdb26 && unzip -o -q '*.zip'

ingest features train comps tracking report:
	$(PY) -m src.cli $@

all:              ## full pipeline from cached/raw data to predictions
	$(PY) -m src.cli all

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

app:
	.venv/bin/streamlit run app/Home.py
