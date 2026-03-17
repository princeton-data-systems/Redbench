#!/usr/bin/env python3
"""
Set up TPC-H data for Redbench generation pipeline.
Uses DuckDB's built-in tpch extension to generate data at a given scale factor,
then exports CSV files, schema.json, and postgres.sql in the format Redbench expects.
"""
import argparse
import json
import os
import duckdb


# TPC-H schema definition
TPCH_TABLES = {
    "region": {
        "columns": [
            ("r_regionkey", "integer", True),
            ("r_name", "varchar", False),
            ("r_comment", "varchar", False),
        ],
    },
    "nation": {
        "columns": [
            ("n_nationkey", "integer", True),
            ("n_name", "varchar", False),
            ("n_regionkey", "integer", False),
            ("n_comment", "varchar", False),
        ],
    },
    "supplier": {
        "columns": [
            ("s_suppkey", "integer", True),
            ("s_name", "varchar", False),
            ("s_address", "varchar", False),
            ("s_nationkey", "integer", False),
            ("s_phone", "varchar", False),
            ("s_acctbal", "decimal(15,2)", False),
            ("s_comment", "varchar", False),
        ],
    },
    "customer": {
        "columns": [
            ("c_custkey", "integer", True),
            ("c_name", "varchar", False),
            ("c_address", "varchar", False),
            ("c_nationkey", "integer", False),
            ("c_phone", "varchar", False),
            ("c_acctbal", "decimal(15,2)", False),
            ("c_mktsegment", "varchar", False),
            ("c_comment", "varchar", False),
        ],
    },
    "part": {
        "columns": [
            ("p_partkey", "integer", True),
            ("p_name", "varchar", False),
            ("p_mfgr", "varchar", False),
            ("p_brand", "varchar", False),
            ("p_type", "varchar", False),
            ("p_size", "integer", False),
            ("p_container", "varchar", False),
            ("p_retailprice", "decimal(15,2)", False),
            ("p_comment", "varchar", False),
        ],
    },
    "partsupp": {
        "columns": [
            ("ps_partkey", "integer", True),
            ("ps_suppkey", "integer", True),
            ("ps_availqty", "integer", False),
            ("ps_supplycost", "decimal(15,2)", False),
            ("ps_comment", "varchar", False),
        ],
    },
    "orders": {
        "columns": [
            ("o_orderkey", "integer", True),
            ("o_custkey", "integer", False),
            ("o_orderstatus", "varchar", False),
            ("o_totalprice", "decimal(15,2)", False),
            ("o_orderdate", "date", False),
            ("o_orderpriority", "varchar", False),
            ("o_clerk", "varchar", False),
            ("o_shippriority", "integer", False),
            ("o_comment", "varchar", False),
        ],
    },
    "lineitem": {
        "columns": [
            ("l_orderkey", "integer", True),
            ("l_partkey", "integer", False),
            ("l_suppkey", "integer", False),
            ("l_linenumber", "integer", True),
            ("l_quantity", "decimal(15,2)", False),
            ("l_extendedprice", "decimal(15,2)", False),
            ("l_discount", "decimal(15,2)", False),
            ("l_tax", "decimal(15,2)", False),
            ("l_returnflag", "varchar", False),
            ("l_linestatus", "varchar", False),
            ("l_shipdate", "date", False),
            ("l_commitdate", "date", False),
            ("l_receiptdate", "date", False),
            ("l_shipinstruct", "varchar", False),
            ("l_shipmode", "varchar", False),
            ("l_comment", "varchar", False),
        ],
    },
}

# TPC-H foreign key relationships: [child_table, child_col, parent_table, parent_col]
TPCH_RELATIONSHIPS = [
    ["nation", "n_regionkey", "region", "r_regionkey"],
    ["supplier", "s_nationkey", "nation", "n_nationkey"],
    ["customer", "c_nationkey", "nation", "n_nationkey"],
    ["partsupp", "ps_partkey", "part", "p_partkey"],
    ["partsupp", "ps_suppkey", "supplier", "s_suppkey"],
    ["orders", "o_custkey", "customer", "c_custkey"],
    ["lineitem", "l_orderkey", "orders", "o_orderkey"],
    ["lineitem", "l_partkey", "part", "p_partkey"],
    ["lineitem", "l_suppkey", "supplier", "s_suppkey"],
]


def type_to_redbench(col_type: str) -> str:
    """Map TPC-H types to Redbench schema.json types."""
    t = col_type.lower()
    if t == "integer":
        return "integer"
    if t.startswith("decimal"):
        return t  # keep as decimal(15,2) — Redbench DataType.from_str handles this
    if t == "date":
        return "date"
    if t == "varchar":
        return "varchar"
    return "varchar"


def type_to_postgres(col_name: str, col_type: str) -> str:
    """Map TPC-H types to PostgreSQL DDL types."""
    t = col_type.lower()
    if t == "integer":
        return "integer"
    if t.startswith("decimal"):
        return t.replace("decimal", "numeric")
    if t == "date":
        return "date"
    if t == "varchar":
        return "character varying"
    return "character varying"


def generate_data(sf: int, output_dir: str):
    """Generate TPC-H data using DuckDB CLI and export to CSV."""
    csv_dir = os.path.join(output_dir, "raw_csvs")
    os.makedirs(csv_dir, exist_ok=True)

    # Use DuckDB CLI (which has working tpch extension) instead of Python API
    duckdb_cli = os.path.expanduser("~/.duckdb/cli/latest/duckdb")
    if not os.path.exists(duckdb_cli):
        # Fallback: try Python API
        duckdb_cli = None

    print(f"Generating TPC-H SF={sf} data...")

    if duckdb_cli:
        import subprocess
        # Build SQL commands to generate and export
        sql_cmds = [f"INSTALL tpch;", f"LOAD tpch;", f"CALL dbgen(sf={sf});"]
        for table_name in TPCH_TABLES:
            csv_path = os.path.join(csv_dir, f"{table_name}.csv")
            sql_cmds.append(f"COPY {table_name} TO '{csv_path}' (HEADER, DELIMITER ',');")
            sql_cmds.append(f"SELECT '{table_name}' as tbl, COUNT(*) as cnt FROM {table_name};")
        
        full_sql = "\n".join(sql_cmds)
        result = subprocess.run(
            [duckdb_cli, "-c", full_sql],
            capture_output=True, text=True, timeout=600
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"DuckDB CLI failed with exit code {result.returncode}")
    else:
        # Fallback to Python API
        con = duckdb.connect()
        con.execute("INSTALL tpch; LOAD tpch;")
        con.execute(f"CALL dbgen(sf={sf});")
        for table_name in TPCH_TABLES:
            csv_path = os.path.join(csv_dir, f"{table_name}.csv")
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  Exporting {table_name}: {count:,} rows -> {csv_path}")
            con.execute(f"COPY {table_name} TO '{csv_path}' (HEADER, DELIMITER ',')")
        con.close()

    print(f"Data generation complete. CSVs in {csv_dir}")
    return csv_dir


def create_schema_json(output_dir: str):
    """Create schema.json in Redbench format."""
    table_col_info = {}
    for table_name, table_def in TPCH_TABLES.items():
        cols = {}
        for col_name, col_type, is_pk in table_def["columns"]:
            cols[col_name] = {
                "type": type_to_redbench(col_type),
                "pk": is_pk,
            }
        table_col_info[table_name] = cols

    schema = {
        "name": "tpch",
        "csv_kwargs": {
            "escapechar": "\\",
            "encoding": "utf-8",
            "quotechar": '"',
        },
        "db_load_kwargs": {},
        "relationships": TPCH_RELATIONSHIPS,
        "table_col_info": table_col_info,
    }

    path = os.path.join(output_dir, "schema.json")
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Written: {path}")
    return path


def create_postgres_sql(output_dir: str):
    """Create postgres.sql DDL file."""
    lines = []
    for table_name, table_def in TPCH_TABLES.items():
        lines.append(f'DROP TABLE IF EXISTS "{table_name}";')
        lines.append(f"CREATE TABLE {table_name}")
        lines.append("(")
        col_lines = []
        pk_cols = []
        for col_name, col_type, is_pk in table_def["columns"]:
            pg_type = type_to_postgres(col_name, col_type)
            not_null = " NOT NULL" if is_pk else ""
            col_lines.append(f"    {col_name:20s} {pg_type}{not_null}")
            if is_pk:
                pk_cols.append(col_name)
        if len(pk_cols) == 1:
            # Single PK: add PRIMARY KEY inline
            for i, (col_name, col_type, is_pk) in enumerate(table_def["columns"]):
                if is_pk:
                    pg_type = type_to_postgres(col_name, col_type)
                    col_lines[i] = f"    {col_name:20s} {pg_type} NOT NULL PRIMARY KEY"
                    break
        else:
            # Composite PK: add as constraint
            col_lines.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
        lines.append(",\n".join(col_lines))
        lines.append(");\n")

    path = os.path.join(output_dir, "postgres.sql")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Set up TPC-H data for Redbench")
    parser.add_argument("--sf", type=int, default=10, help="TPC-H scale factor (default: 10)")
    parser.add_argument("--output", type=str, default="work/tpch_data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Generate data
    csv_dir = generate_data(args.sf, args.output)

    # Create schema files
    create_schema_json(args.output)
    create_postgres_sql(args.output)

    print(f"\nTPC-H SF={args.sf} setup complete in {args.output}/")
    print(f"  CSV data: {csv_dir}/")
    print(f"  Schema:   {args.output}/schema.json")
    print(f"  DDL:      {args.output}/postgres.sql")
    print(f"\nTo use with Redbench generation, create a config with:")
    print(f'  "database_name": "tpch"')
    print(f'  "raw_database_tables": "{csv_dir}"')


if __name__ == "__main__":
    main()
