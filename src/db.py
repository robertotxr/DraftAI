"""Warehouse access layer.

Everything downstream talks to `Warehouse` (read/write DataFrames, run SQL), so the
backend can be swapped by changing `paths.warehouse_url`:
  duckdb:///data/warehouse.duckdb      (default, local file)
  postgresql://user:pass@host/db       (requires sqlalchemy + psycopg)
Tables are namespaced as <layer>.<name> with layers raw / staging / marts.
"""

from __future__ import annotations

import pandas as pd

from src.config import ROOT, cfg

LAYERS = ("raw", "staging", "marts")


class Warehouse:
    def __init__(self, url: str | None = None, read_only: bool = False):
        self.url = url or cfg()["paths"]["warehouse_url"]
        if self.url.startswith("duckdb:///"):
            import duckdb

            file = ROOT / self.url.removeprefix("duckdb:///")
            file.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(file), read_only=read_only)
            self.backend = "duckdb"
        elif self.url.startswith("postgresql"):
            from sqlalchemy import create_engine

            self._engine = create_engine(self.url)
            self.backend = "postgres"
        else:
            raise ValueError(f"Unsupported warehouse url: {self.url}")
        if not read_only:
            for layer in LAYERS:
                self.execute(f"CREATE SCHEMA IF NOT EXISTS {layer}")

    def execute(self, sql: str) -> None:
        if self.backend == "duckdb":
            self._con.execute(sql)
        else:
            from sqlalchemy import text

            with self._engine.begin() as c:
                c.execute(text(sql))

    def read(self, sql: str) -> pd.DataFrame:
        if self.backend == "duckdb":
            return self._con.execute(sql).df()
        return pd.read_sql(sql, self._engine)

    def table(self, name: str) -> pd.DataFrame:
        return self.read(f"SELECT * FROM {name}")

    def write(self, name: str, df: pd.DataFrame) -> None:
        """Idempotent full replace of `layer.table`."""
        layer, table = name.split(".")
        assert layer in LAYERS, f"unknown layer {layer}"
        if self.backend == "duckdb":
            self._con.register("_df", df)
            self._con.execute(f"CREATE OR REPLACE TABLE {layer}.{table} AS SELECT * FROM _df")
            self._con.unregister("_df")
        else:
            df.to_sql(table, self._engine, schema=layer, if_exists="replace", index=False)

    def close(self) -> None:
        if self.backend == "duckdb":
            self._con.close()
