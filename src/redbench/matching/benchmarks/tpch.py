import os
import re
import subprocess

import duckdb
from matching.benchmarks.benchmark import Benchmark
from matching.utils import remove_file
from utils.log import log

QUERIES_DIR_PATH = "queries"
NUM_QUERY_INSTANCES = 1000  # number of random seeds to try per template


class TPCHBenchmark(Benchmark):
    def __init__(self, config, benchmark_config):
        super().__init__(config, benchmark_config)
        self.queries_dir_path = os.path.join(self.output_dir, QUERIES_DIR_PATH)

    def _setup_db(self):
        """Create TPC-H database using DuckDB CLI."""
        sf = getattr(self.benchmark_config, "sf", 1)

        duckdb_cli = os.path.expanduser("~/.duckdb/cli/latest/duckdb")
        if os.path.exists(duckdb_cli):
            sql = "INSTALL tpch; LOAD tpch;\n"
            sql += f"CALL dbgen(sf={sf});\n"
            sql += f"ATTACH '{self.db_filepath}' AS tpch_db;\n"
            for table in [
                "region", "nation", "supplier", "customer",
                "part", "partsupp", "orders", "lineitem",
            ]:
                sql += f"CREATE TABLE tpch_db.{table} AS SELECT * FROM {table};\n"
            sql += "DETACH tpch_db;\n"

            result = subprocess.run(
                [duckdb_cli, "-c", sql],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"DuckDB CLI failed to create TPC-H database: {result.stderr}"
                )
            log(f"TPC-H SF={sf} database created at {self.db_filepath}")
        else:
            conn = duckdb.connect(self.db_filepath)
            conn.execute("INSTALL tpch; LOAD tpch;")
            conn.execute(f"CALL dbgen(sf={sf});")
            conn.close()

    def _extract_template_from_filepath(self, filepath):
        assert self.queries_dir_path in filepath, (
            f"Expected {self.queries_dir_path} in {filepath}"
        )
        return filepath.split(self.queries_dir_path)[1].split("/")[-2]

    def _is_benchmarks_setup(self):
        return os.path.exists(self.queries_dir_path)

    @staticmethod
    def _clean_qgen_output(raw_output):
        """Clean qgen output and return list of (select_sql, view_ddl_or_None) tuples."""
        lines = [
            line for line in raw_output.split("\n")
            if not line.strip().startswith("--")
        ]
        text = "\n".join(lines).strip()
        statements = [s.strip() for s in text.split(";") if s.strip()]

        # Detect CREATE VIEW (template 15)
        create_view_stmt = None
        for stmt in statements:
            if stmt.strip().lower().startswith("create view"):
                # Remove any trailing "limit -1"
                create_view_stmt = re.sub(
                    r"\s+limit\s+-1\s*\Z", "", stmt, flags=re.IGNORECASE
                ).strip()
                break

        _LIMIT_NEG_RE = re.compile(r"\s+limit\s+-1\s*\Z", re.IGNORECASE)

        results = []
        for stmt in statements:
            lower = stmt.lower().strip()
            if lower == "limit -1":
                continue
            if lower.startswith("create view") or lower.startswith("drop view"):
                continue
            if lower.startswith("select") or lower.startswith("(select"):
                stmt = _LIMIT_NEG_RE.sub("", stmt)
                results.append((stmt, create_view_stmt))
        return results

    def _setup(self):
        """Build tpch-kit and generate query instances using qgen with random seeds."""
        sf = getattr(self.benchmark_config, "sf", 1)
        remove_file(self.queries_dir_path)

        kit_dir = os.path.abspath(os.path.join(self.output_dir, "tpch-kit"))

        if not os.path.exists(kit_dir):
            log("Cloning tpch-kit...")
            ret = os.system(
                f"cd {self.output_dir} && git clone https://github.com/gregrahn/tpch-kit.git"
            )
            if ret != 0:
                raise RuntimeError("Failed to clone tpch-kit")

        dbgen_dir = os.path.join(kit_dir, "dbgen")
        qgen_path = os.path.join(dbgen_dir, "qgen")

        if not os.path.exists(qgen_path):
            log("Building tpch-kit (dbgen + qgen)...")
            os.system(f"cd {dbgen_dir} && make MACHINE=LINUX DATABASE=POSTGRESQL")

        if not os.path.exists(qgen_path):
            raise RuntimeError(
                f"Failed to build qgen at {qgen_path}. "
                "Check that gcc and make are installed."
            )

        env = os.environ.copy()
        env["DSS_CONFIG"] = dbgen_dir
        env["DSS_QUERY"] = os.path.join(dbgen_dir, "queries")

        for template_num in range(1, 23):
            template_dir = os.path.join(self.queries_dir_path, str(template_num))
            os.makedirs(template_dir, exist_ok=True)

            queries = {}  # sql -> view_ddl_or_None
            for seed in range(1, NUM_QUERY_INSTANCES + 1):
                result = subprocess.run(
                    [qgen_path, "-c", "-s", str(sf), "-r", str(seed),
                     str(template_num)],
                    capture_output=True, text=True, timeout=30,
                    cwd=dbgen_dir, env=env,
                )
                if result.returncode != 0:
                    if seed == 1:
                        log(
                            f"qgen failed for template {template_num} seed {seed}: "
                            f"{result.stderr.strip()}",
                            log_mode="warning",
                        )
                    continue

                for stmt, view_ddl in self._clean_qgen_output(result.stdout):
                    if stmt not in queries:
                        queries[stmt] = view_ddl

            for i, query in enumerate(sorted(queries.keys())):
                with open(os.path.join(template_dir, f"{i}.sql"), "w") as f:
                    f.write(query)
                view_ddl = queries[query]
                if view_ddl:
                    with open(os.path.join(template_dir, f"{i}.view.sql"), "w") as f:
                        f.write(view_ddl)

            num_queries = len([
                f for f in os.listdir(template_dir)
                if f.endswith(".sql") and not f.endswith(".view.sql")
            ])
            log(
                f"Template {template_num}: {num_queries} distinct query instances "
                f"from {NUM_QUERY_INSTANCES} seeds"
            )

        total = sum(
            len([
                f for f in os.listdir(os.path.join(self.queries_dir_path, str(t)))
                if f.endswith(".sql") and not f.endswith(".view.sql")
            ])
            for t in range(1, 23)
            if os.path.exists(os.path.join(self.queries_dir_path, str(t)))
        )
        log(f"TPC-H query generation complete: {total} total instances across 22 templates")

    def _process_tpch_dir(self, dir_path, query_stats):
        """Like _process_dir but handles TPC-H view-dependent queries (template 15)."""
        import json

        template_to_num_joins = dict()
        template_to_scanset = dict()
        failed = set()

        assert os.path.exists(self.db_filepath), (
            f"Database file {self.db_filepath} does not exist."
        )

        sql_files = [
            f for f in os.listdir(dir_path)
            if f.endswith(".sql") and not f.endswith(".view.sql")
        ]
        if not sql_files:
            return

        for filename in sql_files:
            filepath = os.path.join(dir_path, filename)
            template = self._extract_template_from_filepath(filepath)

            if template in template_to_num_joins:
                query_stats[filepath] = {
                    "num_joins": template_to_num_joins[template],
                    "template": template,
                    "scanset": template_to_scanset[template],
                }
                continue
            if template in failed:
                continue

            with open(filepath, "r") as f:
                query = f.read()

            # Check for companion view DDL
            view_file = filepath.replace(".sql", ".view.sql")
            view_ddl = None
            if os.path.exists(view_file):
                with open(view_file, "r") as f:
                    view_ddl = f.read()

            try:
                # Need read_only=False if we must create a view
                conn = duckdb.connect(self.db_filepath, read_only=(view_ddl is None))

                if view_ddl:
                    conn.execute(view_ddl)

                result = conn.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                profile = result.fetchall()
                assert len(profile) == 1
                assert profile[0][0] in ["physical_plan", "analyzed_plan"]
                profile = profile[0][1]
                assert isinstance(profile, str)

                if view_ddl:
                    # Extract view name and drop it
                    view_match = re.search(
                        r"create\s+view\s+(\w+)", view_ddl, re.IGNORECASE
                    )
                    if view_match:
                        conn.execute(f"DROP VIEW IF EXISTS {view_match.group(1)}")

                conn.close()

                num_joins = profile.count(
                    '"operator_type": "HASH_JOIN"'
                ) - profile.count('"operator_type": "COLUMN_DATA_SCAN"')
                scanset = self._extract_scanset_from_profile(json.loads(profile))

            except Exception:
                log(
                    f'Failed to process query "{filepath}". Skipping.',
                    log_mode="error",
                )
                failed.add(template)
                continue

            template_to_num_joins[template] = num_joins
            template_to_scanset[template] = scanset

            query_stats[filepath] = {
                "num_joins": num_joins,
                "template": template,
                "scanset": scanset,
            }

    def _compute_stats(self):
        self._override_stats_table()

        benchmark_stats = dict()
        for template_num in range(1, 23):
            template_dir = os.path.join(self.queries_dir_path, str(template_num))
            if os.path.exists(template_dir):
                self._process_tpch_dir(template_dir, benchmark_stats)

        for filepath, stats in benchmark_stats.items():
            self._insert_stats(filepath, stats)
