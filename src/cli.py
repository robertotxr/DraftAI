"""Pipeline entrypoint: python -m src.cli <step> [<step> ...]   (steps: ingest features train comps tracking report all)"""

from __future__ import annotations

import argparse
import logging
import warnings

STEPS = ["ingest", "features", "train", "comps", "tracking", "report"]


def run_step(step: str) -> None:
    if step == "ingest":
        from src.ingest import etl

        etl.run()
    elif step == "features":
        from src.features import build

        build.run()
    elif step == "train":
        from src.models import train

        train.run()
    elif step == "comps":
        from src.comps import knn

        knn.run()
    elif step == "tracking":
        from src.tracking import run

        run.run()
    elif step == "report":
        from src.models import report

        report.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=UserWarning)
    for step in STEPS if "all" in args.steps else args.steps:
        logging.getLogger("cli").info("== %s ==", step)
        run_step(step)


if __name__ == "__main__":
    main()
