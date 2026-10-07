PY := .venv/bin/python

.PHONY: setup bdb ingest features train comps tracking report all test lint app

setup:            ## create the virtualenv and install pinned dependencies
	python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

bdb:              ## download Big Data Bowl 2024 (accept the rules on kaggle.com first)
	set -a; . ./.env; set +a; .venv/bin/kaggle competitions list -s "big data bowl"
	set -a; . ./.env; set +a; .venv/bin/kaggle competitions download -c nfl-big-data-bowl-2024 -p data/raw/bdb && cd data/raw/bdb && unzip -o -q '*.zip'

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
