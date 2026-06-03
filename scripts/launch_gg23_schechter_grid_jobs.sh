#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_JOBS="${MAX_JOBS:-4}"
PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p variants/gg23_grid_logs

jobs=(
  "gg23_schechter_no_bh_logpoly3 gg23_no_bh logpoly3"
  "gg23_schechter_no_bh_cored_powerlaw_a gg23_no_bh cored_powerlaw_a"
  "gg23_schechter_bh_logpoly3 gg23_bh logpoly3"
  "gg23_schechter_bh_cored_powerlaw_a gg23_bh cored_powerlaw_a"
  "gg23_schechter_bh_feh_gradient_logpoly3 gg23_bh_feh_gradient logpoly3"
  "gg23_schechter_bh_feh_gradient_cored_powerlaw_a gg23_bh_feh_gradient cored_powerlaw_a"
  "gg23_schechter_bh_past_tidal_logpoly3 gg23_bh_past_tidal logpoly3"
  "gg23_schechter_bh_past_tidal_cored_powerlaw_a gg23_bh_past_tidal cored_powerlaw_a"
  "gg23_schechter_bh_feh_gradient_past_tidal_logpoly3 gg23_bh_feh_gradient_past_tidal logpoly3"
  "gg23_schechter_bh_feh_gradient_past_tidal_cored_powerlaw_a gg23_bh_feh_gradient_past_tidal cored_powerlaw_a"
)

run_one() {
  local output_root="$1"
  local gg23_model="$2"
  local radial_model="$3"
  local outdir="variants/${output_root}/outputs"
  mkdir -p "$outdir"
  {
    echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ) output_root=${output_root} gg23_model=${gg23_model} radial_model=${radial_model}"
    "$PYTHON" scripts/run_profile_map_and_exact_mcmc_schechter_powerlaw_a.py \
      --output-root-name "$output_root" \
      --survivability-backend gg23 \
      --gg23-model "$gg23_model" \
      --radial-model "$radial_model" \
      --coarse-eta-min 0.4 \
      --coarse-eta-max 3.0 \
      --coarse-eta-n 9 \
      --coarse-alpha-min -1.8 \
      --coarse-alpha-max -0.4 \
      --coarse-alpha-n 8 \
      --coarse-logmc-min 5.8 \
      --coarse-logmc-max 6.9 \
      --coarse-logmc-n 7 \
      --refine-delta-logl 3.0 \
      --refine-min-points 10 \
      --refine-padding-steps 1.0 \
      --local-eta-n 9 \
      --local-alpha-n 9 \
      --local-logmc-n 7 \
      --local-max-passes 3 \
      --local-expand-steps 1.0 \
      --anchor-k 12 \
      --skip-mcmc
    echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) output_root=${output_root}"
  } >"${outdir}/grid.log" 2>&1
}

running=0
for job in "${jobs[@]}"; do
  read -r output_root gg23_model radial_model <<<"$job"
  run_one "$output_root" "$gg23_model" "$radial_model" &
  running=$((running + 1))
  if [[ "$running" -ge "$MAX_JOBS" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All GG23 grid jobs finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
