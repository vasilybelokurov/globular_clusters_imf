#!/usr/bin/env bash
# Baseline campaign (2026-10): the current model with the fixes on fix/baseline-rerun,
# written into fresh variant directories (variants/baseline_2026-10_*).
#
# Queues (run one at a time per invocation, each job = stage 1 grid + stage 2 MCMC):
#   gg23       five GG23 laws, logpoly3, eta_t FIXED at 1 (GG23 as published). With the
#              change-of-variables term, eta_t is not bounded by the data under GG23
#              (the likelihood rises monotonically to the no-disruption limit; pilot
#              2026-10-02), so it is not a free parameter here.
#   baumgardt  Baumgardt survivability with free eta_t, radial families step5,
#              powerlaw_a, cored_powerlaw_a (logpoly3 is the pilot run).
#
# Usage:  bash run_plans/baseline_campaign.sh gg23|baumgardt   (from the project root)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
TAG="baseline_2026-10"
WORKERS="${WORKERS:-7}"

run_job() {
  # name backend gg23_model radial eta_min eta_max eta_n local_eta_n prior_eta steps burn
  local name="$1" backend="$2" gg23_model="$3" radial="$4"
  local eta_min="$5" eta_max="$6" eta_n="$7" local_eta_n="$8" prior_eta="$9" steps="${10}" burn="${11}"
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
      --radial-model "$radial" \
      --coarse-eta-min "$eta_min" --coarse-eta-max "$eta_max" --coarse-eta-n "$eta_n" \
      --coarse-alpha-min -2.2 --coarse-alpha-max -0.4 --coarse-alpha-n 10 \
      --coarse-logmc-min 5.6 --coarse-logmc-max 7.0 --coarse-logmc-n 8 \
      --refine-delta-logl 3.0 --refine-min-points 10 --refine-padding-steps 1.0 \
      --local-eta-n "$local_eta_n" --local-alpha-n 9 --local-logmc-n 7 \
      --local-max-passes 3 --local-expand-steps 1.0 \
      --anchor-k 18 --grid-workers "$WORKERS" --skip-mcmc || { echo "STAGE1 FAILED $out"; return 1; }
    echo "STAGE1 DONE $(date -u +%FT%TZ) $out"
    echo "STAGE2 START $(date -u +%FT%TZ) $out"
    "$PYTHON" scripts/run_parallel_exact_mcmc_from_existing_refined_grid.py \
      --source-output-root-name "$out" \
      --radial-model "$radial" \
      --survivability-backend "$backend" ${gg23_args[@]+"${gg23_args[@]}"} \
      --mcmc-chains "$WORKERS" --mcmc-steps "$steps" --mcmc-burn "$burn" --mcmc-thin 2 \
      --mcmc-adapt-until 300 --mcmc-adapt-every 20 \
      --prior-eta "$prior_eta" --prior-alpha=-2.5,-0.3 --prior-logmc 5.5,7.5 || { echo "STAGE2 FAILED $out"; return 1; }
    echo "STAGE2 DONE $(date -u +%FT%TZ) $out"
  } >"${log_dir}/campaign.log" 2>&1 || return 1
}

case "${1:-}" in
  gg23)
    for model in gg23_no_bh gg23_bh gg23_bh_feh_gradient gg23_bh_past_tidal gg23_bh_feh_gradient_past_tidal; do
      run_job "${model}_logpoly3_eta1" gg23 "$model" logpoly3 1.0 1.0 1 1 1.0,1.0 1000 200 || echo "FAILED ${model}"
    done
    ;;
  baumgardt)
    for radial in step5 powerlaw_a cored_powerlaw_a; do
      run_job "baumgardt_${radial}" baumgardt "" "$radial" 0.3 3.0 10 9 0.2,4.0 2400 400 || echo "FAILED ${radial}"
    done
    ;;
  *)
    echo "usage: $0 gg23|baumgardt" >&2
    exit 2
    ;;
esac
echo "QUEUE ${1} DONE $(date -u +%FT%TZ)"
