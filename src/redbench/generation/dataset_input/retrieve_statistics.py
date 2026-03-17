import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Dict

import duckdb
from tqdm import tqdm

from redbench.utils.log import log


@dataclass
class ColumnStats:
    data_type: str
    quantiles: list


@dataclass
class TableStats:
    total_rows: int
    columns: Dict[str, ColumnStats]


def list_tables(con):
    query = "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE';"
    return [row[0] for row in con.execute(query).fetchall()]


def get_table_columns(con, table_name: str):
    query = f"DESCRIBE {table_name};"
    return [(row[0], row[1]) for row in con.execute(query).fetchall()]


def get_table_row_count(con, table_name: str):
    query = f"SELECT COUNT(*) FROM {table_name};"
    return con.execute(query).fetchone()[0]


def get_column_statistics(con, table_name: str, column_name: str, data_type: str, num_quantiles: int = 1000):
    quantile_fractions = [i / num_quantiles for i in range(0, num_quantiles + 1)]

    # Handle numeric types (e.g., INTEGER, DECIMAL, etc.)
    if any(
        numeric_type in data_type
        for numeric_type in (
            "INTEGER",
            "BIGINT",
            "DOUBLE",
            "REAL",
            "DECIMAL",
            "NUMERIC",
        )
    ):
        fractions_str = ", ".join(str(f) for f in quantile_fractions)
        query = f"""
            SELECT quantile_cont("{column_name}", [{fractions_str}])
            FROM {table_name};
        """
        result = con.execute(query).fetchone()
        if result is None or result[0] is None:
            return None
        quantile_values = list(result[0])

    # Handle string types (e.g., VARCHAR, TEXT, STRING)
    elif any(string_type in data_type for string_type in ("VARCHAR", "TEXT", "STRING")):
        # Use quantile_disc which supports string types in DuckDB
        fractions_str = ", ".join(str(f) for f in quantile_fractions)
        query = f"""
            SELECT quantile_disc("{column_name}", [{fractions_str}])
            FROM {table_name};
        """
        result = con.execute(query).fetchone()
        if result is None or result[0] is None:
            return None
        quantile_values = list(result[0])

        # Fill None values forward
        for i in range(len(quantile_values)):
            if quantile_values[i] is None and i + 1 < len(quantile_values):
                for j in range(i + 1, len(quantile_values)):
                    if quantile_values[j] is not None:
                        quantile_values[i] = quantile_values[j]
                        break

    # Handle date/time types (e.g., DATE, TIMESTAMP, TIME)
    elif any(date_type in data_type for date_type in ("DATE", "TIMESTAMP", "TIME")):
        fractions_str = ", ".join(str(f) for f in quantile_fractions)
        query = f"""
            SELECT quantile_cont("{column_name}", [{fractions_str}])
            FROM {table_name};
        """
        result = con.execute(query).fetchone()
        if result is None or result[0] is None:
            return None
        quantile_values = list(result[0])

    else:
        return None

    return convert_quantiles_to_serializable(quantile_values)


def convert_quantiles_to_serializable(quantile_values) -> Dict[str, any]:
    """Convert a list of quantile values to the serializable format expected by the rest of the codebase."""
    serializable_quantiles = [
        (
            float(value) if isinstance(value, Decimal) else
            value.isoformat() if isinstance(value, (date, datetime)) else
            value if value is not None else None
        )
        for value in quantile_values
    ]
    return {"quantiles": serializable_quantiles}


def convert_to_serializable(stats) -> Dict[str, any]:
    # Extract quantiles as a list instead of a dictionary
    quantiles = [value for key, value in stats.items() if key.startswith("q_")]
    stats_without_quantiles = {
        key: value for key, value in stats.items() if not key.startswith("q_")
    }

    # Convert quantiles to serializable format
    serializable_quantiles = [
        (
            float(value) if isinstance(value, Decimal) else
            value.isoformat() if isinstance(value, (date, datetime)) else
            value if value is not None else None
        )
        for value in quantiles
    ]

    return {
        **{
            key: (
                float(value) if isinstance(value, Decimal) else
                value.isoformat() if isinstance(value, (date, datetime)) else
                value if value is not None else None
            )
            for key, value in stats_without_quantiles.items()
        },
        "quantiles": serializable_quantiles  # Store quantiles as a serializable list
    }


def create_quantiles(db_path: str, output_file: str, force: bool = False):
    if not force and os.path.exists(output_file):
        log(f"Quantile stats file already exists at: {output_file}. Skipping.")
        return

    con = duckdb.connect(db_path, read_only=True)
    tables = list_tables(con)
    all_stats = {}

    for table in tqdm(tables, desc="Computing quantiles"):
        row_count = get_table_row_count(con, table)
        columns = get_table_columns(con, table)
        table_stats = TableStats(total_rows=row_count, columns={})

        import concurrent.futures

        def process_column(args):
            col_name, col_type = args
            dd_con = duckdb.connect(db_path, read_only=True)
            stats = get_column_statistics(dd_con, table, col_name, col_type)
            dd_con.close()
            if stats:
                return col_name, ColumnStats(data_type=col_type, **stats)
            return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_column, columns))

        for result in results:
            if result:
                col_name, col_stats = result
                table_stats.columns[col_name] = col_stats

        if table_stats.columns:
            all_stats[table] = asdict(table_stats)

    con.close()

    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=4)

    log(f"Quantile stats written to: {output_file}")
