#!/bin/bash
# Screen top provisioned clusters with IMDB generation (limited to 500K rows for speed).
# Note: this uses first-N rows (truncated time range). For full-range analysis,
# use sample_provisioned.py + run_prov_sampled.sh instead.
set -e
export MISE_QUIET=1
source .venv/bin/activate

BASE_DIR=$(pwd)
REDSET="work/data/full_provisioned.parquet"
CONFIG="configs/config_gen_imdb_prov.json"

REF_IMDB="${BASE_DIR}/work/output_select_full/tmp_generation/imdb"

CLUSTERS="158 4 49 109 100 79 103 34"
DB=0

LOGDIR="work/logs_provisioned"
mkdir -p "${LOGDIR}"

SHARED_FILES="schema.json postgres.sql schema_augmented_x2.sql db_augmented_x2.duckdb db_original.duckdb column_statistics.json"
SHARED_DIRS="tables_normalized tables_augmented_x2"

setup_output_dir() {
    local cluster=$1
    local outdir="work/output_prov_c${cluster}_d${DB}"
    mkdir -p "${outdir}/tmp_generation/imdb"
    for dir_name in ${SHARED_DIRS}; do
        if [ ! -e "${outdir}/tmp_generation/imdb/${dir_name}" ]; then
            ln -sf "${REF_IMDB}/${dir_name}" "${outdir}/tmp_generation/imdb/${dir_name}"
        fi
    done
    for fname in ${SHARED_FILES}; do
        if [ ! -e "${outdir}/tmp_generation/imdb/${fname}" ]; then
            ln -sf "${REF_IMDB}/${fname}" "${outdir}/tmp_generation/imdb/${fname}"
        fi
    done
    echo "${outdir}"
}

for cluster in ${CLUSTERS}; do
    outdir=$(setup_output_dir ${cluster})
    logfile="${LOGDIR}/screen_imdb_c${cluster}.log"

    echo "[$(date)] Starting IMDB screening for provisioned cluster=${cluster}"
    python src/redbench/run.py \
        --redset_path "${REDSET}" \
        --output_dir "${outdir}" \
        --generation_strategy generation \
        --config_path_generation "${CONFIG}" \
        --instance_id "${cluster}" \
        --database_id "${DB}" \
        > "${logfile}" 2>&1

    echo "[$(date)] Cluster ${cluster} done (exit=$?)"
done

echo "[$(date)] All provisioned screening runs complete"
