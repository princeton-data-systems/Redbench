#!/bin/bash
# Run IMDB generation for multiple serverless clusters.
# All output goes under work/.
set -e
export MISE_QUIET=1
source .venv/bin/activate

BASE_DIR=$(pwd)
REDSET="work/data/full_serverless.parquet"
GEN_CONFIG="configs/config_gen_imdb.json"

# Reference directory with shared IMDB setup artifacts
REF_IMDB="${BASE_DIR}/work/output_select_full/tmp_generation/imdb"

CLUSTERS="0:0 85:0 104:0 134:0 55:0 19:0 128:0 105:0 126:0 129:0"

SHARED_FILES="schema.json postgres.sql schema_augmented_x2.sql db_augmented_x2.duckdb db_original.duckdb column_statistics.json"
SHARED_DIRS="tables_normalized tables_augmented_x2"

setup_output_dir() {
    local cluster=$1
    local db=$2
    local outdir="work/output_c${cluster}_d${db}"

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

run_cluster() {
    local cluster=$1
    local db=$2
    local outdir=$(setup_output_dir $cluster $db)
    local logfile="${outdir}/run.log"

    echo "[$(date)] Starting cluster=${cluster} db=${db} -> ${outdir}" | tee -a "${logfile}"

    python src/redbench/run.py \
        --redset_path "${REDSET}" \
        --output_dir "${outdir}" \
        --generation_strategy generation \
        --config_path_generation "${GEN_CONFIG}" \
        --instance_id ${cluster} \
        --database_id ${db} \
        --overwrite_existing \
        >> "${logfile}" 2>&1

    echo "[$(date)] Done cluster=${cluster} db=${db}" | tee -a "${logfile}"
}

pids=()
for entry in $CLUSTERS; do
    IFS=':' read -r cluster db <<< "$entry"
    run_cluster $cluster $db &
    pids+=($!)
    echo "Started cluster=${cluster} db=${db} PID=${pids[-1]}"
done

echo "Waiting for ${#pids[@]} parallel jobs..."
for pid in "${pids[@]}"; do
    wait $pid
    echo "PID $pid finished (exit=$?)"
done
echo "All IMDB clusters done."
