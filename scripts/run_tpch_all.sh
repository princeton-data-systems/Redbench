#!/bin/bash
# Run TPC-H SF=1 generation for all viable clusters.
# 1. Build reference DB via cluster 134 with force_setup_creation=true
# 2. Symlink artifacts into remaining output dirs
# 3. Run remaining clusters with force_setup_creation=false
# Skipping clusters 80 and 24 (0 usable SELECTs).
set -e
export MISE_QUIET=1
source .venv/bin/activate

BASE_DIR=$(pwd)
REDSET="work/data/full_serverless.parquet"
TPCH_CONFIG_SETUP="configs/config_gen_tpch_setup.json"
TPCH_CONFIG="configs/config_gen_tpch.json"
TPCH_DATA="work/tpch_data_sf1"

REFERENCE_CLUSTER=134
REMAINING_CLUSTERS="19 104 55 85 128 105 0 126"

setup_tpch_dir() {
    local cluster=$1
    local outdir="work/output_tpch_sf1_c${cluster}_d0"
    mkdir -p "${outdir}/tmp_generation/tpch"
    cp -f "${BASE_DIR}/${TPCH_DATA}/schema.json" "${outdir}/tmp_generation/tpch/schema.json"
    cp -f "${BASE_DIR}/${TPCH_DATA}/postgres.sql" "${outdir}/tmp_generation/tpch/postgres.sql"
    echo "${outdir}"
}

# Step 1: Build reference DB
echo "=== Building TPC-H SF=1 reference DB via cluster ${REFERENCE_CLUSTER} ==="
REF_OUTDIR=$(setup_tpch_dir $REFERENCE_CLUSTER)
REF_LOGFILE="${REF_OUTDIR}/run.log"
python src/redbench/run.py \
    --redset_path "${REDSET}" \
    --output_dir "${REF_OUTDIR}" \
    --generation_strategy generation \
    --config_path_generation "${TPCH_CONFIG_SETUP}" \
    --instance_id ${REFERENCE_CLUSTER} \
    --database_id 0 \
    --overwrite_existing \
    > "${REF_LOGFILE}" 2>&1

REF_TPCH_DIR="${BASE_DIR}/${REF_OUTDIR}/tmp_generation/tpch"
echo "Reference artifacts at: ${REF_TPCH_DIR}"

# Step 2: Symlink and run remaining clusters in parallel
echo "=== Starting remaining TPC-H clusters ==="
pids=()
for cluster in $REMAINING_CLUSTERS; do
    outdir=$(setup_tpch_dir $cluster)
    for f in db_augmented_x2.duckdb tables_normalized tables_augmented_x2 column_statistics.json; do
        if [ -e "${REF_TPCH_DIR}/${f}" ] && [ ! -e "${outdir}/tmp_generation/tpch/${f}" ]; then
            ln -sf "${REF_TPCH_DIR}/${f}" "${outdir}/tmp_generation/tpch/${f}"
        fi
    done

    logfile="${outdir}/run.log"
    python src/redbench/run.py \
        --redset_path "${REDSET}" \
        --output_dir "${outdir}" \
        --generation_strategy generation \
        --config_path_generation "${TPCH_CONFIG}" \
        --instance_id ${cluster} \
        --database_id 0 \
        --overwrite_existing \
        > "${logfile}" 2>&1 &
    pids+=($!)
    echo "  Started cluster=${cluster} PID=${pids[-1]}"
done

for pid in "${pids[@]}"; do
    wait $pid
    echo "PID $pid finished (exit=$?)"
done
echo "=== All TPC-H runs complete ==="
