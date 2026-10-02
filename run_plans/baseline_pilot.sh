#!/usr/bin/env bash
# Baseline pilot (2026-10-02): the current model with the fixes of commit ad4f9c7,
# run into fresh variant directories so nothing cached for the paper is overwritten.
#
#   1. Baumgardt survivability, Schechter IMF, logpoly3 radial profile
#   2. GG23 no-BH survivability (eta-dependent birth masses + Jacobian), same IMF/radial
#
# Stage 1 maps and refines the profile likelihood on a grid; stage 2 runs the parallel
# profiled MCMC with an explicit, wide uniform prior box. The purpose is to see whether
# eta_t is bounded by the data before queuing the full campaign.
#
# Usage:  bash run_plans/baseline_pilot.sh            (from the project root)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"
# One BLAS/OpenMP thread per process: the grid and chain workers already use the cores,
# and numpy's default thread pool per worker oversubscribed the machine (load ~150 on 14).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
TAG="baseline_2026-10"
GRID_WORKERS="${GRID_WORKERS:-6}"
CHAINS="${CHAINS:-7}"
STEPS="${STEPS:-2400}"
BURN="${BURN:-400}"

run_job() {
  local name="$1" backend="$2" gg23_model="$3" eta_min="$4" eta_max="$5" eta_n="$6" prior_eta="$7"
  local out="${TAG}_${name}"
  local log_dir="variants/${out}/outputs"
  mkdir -p "$log_dir"
  local gg23_args=()
  if [[ "$backend" == "gg23" ]]; then
    gg23_args=(--gg23-model "$gg23_model")
  fi
  {
    echo "STAGE1 START $(date -u +%FT%TZ) $out"
    "$PYTHON" scripts/run_profile_map_and_exact_mcmc_schechter_powerlaw_a.py \
      --output-root-name "$out" \
      --survivability-backend "$backend" ${gg23_args[@]+"${gg23_args[@]}"} \
      --radial-model logpoly3 \
      --coarse-eta-min "$eta_min" --coarse-eta-max "$eta_max" --coarse-eta-n "$eta_n" \
      --coarse-alpha-min -2.2 --coarse-alpha-max -0.4 --coarse-alpha-n 10 \
      --coarse-logmc-min 5.6 --coarse-logmc-max 7.0 --coarse-logmc-n 8 \
      --refine-delta-logl 3.0 --refine-min-points 10 --refine-padding-steps 1.0 \
      --local-eta-n 9 --local-alpha-n 9 --local-logmc-n 7 \
      --local-max-passes 3 --local-expand-steps 1.0 \
      --anchor-k 18 --grid-workers "$GRID_WORKERS" --skip-mcmc
    echo "STAGE1 DONE $(date -u +%FT%TZ) $out"
    echo "STAGE2 START $(date -u +%FT%TZ) $out"
    "$PYTHON" scripts/run_parallel_exact_mcmc_from_existing_refined_grid.py \
      --source-output-root-name "$out" \
      --radial-model logpoly3 \
      --survivability-backend "$backend" ${gg23_args[@]+"${gg23_args[@]}"} \
      --mcmc-chains "$CHAINS" --mcmc-steps "$STEPS" --mcmc-burn "$BURN" --mcmc-thin 2 \
      --mcmc-adapt-until 300 --mcmc-adapt-every 20 \
      --prior-eta "$prior_eta" --prior-alpha=-2.5,-0.3 --prior-logmc 5.5,7.5
    echo "STAGE2 DONE $(date -u +%FT%TZ) $out"
  } >"${log_dir}/pilot.log" 2>&1
}

run_job baumgardt_logpoly3 baumgardt "" 0.3 3.0 10 0.2,4.0 &
run_job gg23_no_bh_logpoly3 gg23 gg23_no_bh 0.2 6.0 12 0.1,10.0 &
wait
echo "PILOT DONE $(date -u +%FT%TZ)"
