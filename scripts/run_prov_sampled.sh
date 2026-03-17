#!/bin/bash
# Run IMDB generation for provisioned clusters using uniformly sampled parquet files.
# Each sampled file contains 500K rows spanning the full 3-month range.
set -e
export MISE_QUIET=1
source .venv/bin/activate

BASE_DIR=$(pwd)
CONFIG="configs/config_gen_imdb_prov.json"

# Reference directory with shared IMDB setup
REF_IMDB="${BASE_DIR}/work/output_select_full/tmp_generation/imdb"

CLUSTERS="158 4 49 109 100"
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
    REDSET="work/data/provisioned_sampled/prov_c${cluster}_d${DB}.parquet"
    outdir=$(setup_output_dir ${cluster})
    logfile="${LOGDIR}/sampled_imdb_c${cluster}.log"

    if [ ! -f "${REDSET}" ]; then
        echo "[$(date)] Skipping cluster ${cluster} -- no sampled parquet at ${REDSET}"
        continue
    fi

    rm -rf "${outdir}/generated_workloads" "${outdir}/paper_workloads"

    echo "[$(date)] Starting IMDB generation for provisioned cluster=${cluster} (sampled)"
    python src/redbench/run.py \
        --redset_path "${REDSET}" \
        --output_dir "${outdir}" \
        --generation_strategy generation \
        --config_path_generation "${CONFIG}" \
        --instance_id "${cluster}" \
        --database_id "${DB}" \
        > "${logfile}" 2>&1

    echo "[$(date)] Cluster ${cluster} completed (exit=$?)"
done

echo "[$(date)] All provisioned sampled runs complete"
