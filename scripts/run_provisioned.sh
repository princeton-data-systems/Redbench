#!/bin/bash
# Run IMDB generation for top provisioned clusters (full, no row limit).
# Uses symlinks to shared IMDB artifacts to avoid duplicating ~14GB per cluster.
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
    logfile="${LOGDIR}/gen_imdb_c${cluster}.log"

    if ls "${outdir}"/generated_workloads/imdb/provisioned/cluster_*/database_*/generation_*/workload.csv 1>/dev/null 2>&1; then
        echo "[$(date)] Skipping cluster ${cluster} -- output already exists"
        continue
    fi

    echo "[$(date)] Starting IMDB generation for provisioned cluster=${cluster}"
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

echo "[$(date)] All provisioned cluster runs complete"
