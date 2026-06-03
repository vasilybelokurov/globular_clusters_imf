from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', str(PROJECT_ROOT / '.cache'))
(PROJECT_ROOT / '.mplconfig').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / '.cache' / 'fontconfig').mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from globular_clusters_imf.joint_model import JointModelSpec
from globular_clusters_imf.model import fit_catalog_models
from run_profile_map_and_exact_mcmc_schechter_powerlaw_a import (
    _compute_rhat,
    _corner_plot,
    _evaluate_theta_multistart,
    _lightweight_entry,
    _round_key,
    _run_exact_mcmc_chain_worker,
    _save_best_payload,
    _select_anchor_start_state,
    _select_diverse_entries,
    _trace_plot,
)


def _load_catalog(prepare_root: Path) -> pd.DataFrame:
    catalog_path = PROJECT_ROOT / 'data' / 'processed' / 'baumgardt_gc_catalog_with_origin_flags.csv'
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / 'data' / 'processed' / 'baumgardt_gc_catalog.csv'
    catalog = pd.read_csv(catalog_path)
    prepared = fit_catalog_models(catalog, prepare_root)['catalog']
    return prepared


def _dummy_entries_from_table(table: pd.DataFrame) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    good = table.loc[np.isfinite(table['log_likelihood'])].copy()
    for _, row in good.iterrows():
        entries.append(
            {
                'theta': np.array([
                    float(row['eta_t']),
                    float(row['input_alpha_dndm']),
                    float(row['input_log10_m_c_msun']),
                ], dtype=float),
                'log_posterior': float(row['log_likelihood']),
                'row': row.to_dict(),
                'result': None,
                'start_state': None,
            }
        )
    return entries


def _build_exact_anchor_entries(
    *,
    prepared_catalog: pd.DataFrame,
    spec: JointModelSpec,
    project_root: Path,
    refined_table: pd.DataFrame,
    refined_bounds: np.ndarray,
    anchor_k: int,
    anchor_pool: int,
    survivability_backend: str,
    gg23_model_name: str | None,
) -> list[dict[str, object]]:
    dummy_entries = _dummy_entries_from_table(refined_table)
    candidate_entries = _select_diverse_entries(
        dummy_entries,
        n_select=anchor_k,
        bounds=refined_bounds,
        candidate_pool=anchor_pool,
    )
    candidate_entries = sorted(candidate_entries, key=lambda entry: float(entry['log_posterior']), reverse=True)
    exact_entries: list[dict[str, object]] = []
    for candidate in candidate_entries:
        theta = np.asarray(candidate['theta'], dtype=float)
        anchor_state = _select_anchor_start_state(theta=theta, anchors=exact_entries, bounds=refined_bounds)
        exact_entry = _evaluate_theta_multistart(
            prepared_catalog=prepared_catalog,
            spec=spec,
            theta=theta,
            stage='parallel_anchor',
            project_root=project_root,
            anchor_start_state=anchor_state,
            survivability_backend=survivability_backend,
            gg23_model_name=gg23_model_name,
        )
        exact_entries.append(exact_entry)
    return exact_entries


def _write_worker_result(output_path: Path, payload: dict[str, object]) -> None:
    with output_path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _run_worker_mode(config_path: Path, output_path: Path) -> None:
    with config_path.open('rb') as handle:
        config = pickle.load(handle)
    result = _run_exact_mcmc_chain_worker(**config)
    _write_worker_result(output_path, result)
    best_row = result['best_row']
    print(
        f"[worker chain={config['chain_id']}] accept={float(result['acceptance']):.3f} "
        f"best logL={float(best_row['log_likelihood']):.3f} eta_t={float(best_row['eta_t']):.3f} "
        f"alpha={float(best_row['input_alpha_dndm']):.3f} logMc={float(best_row['input_log10_m_c_msun']):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-output-root-name', default='profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic_parallel_long')
    parser.add_argument('--output-root-name', default='')
    parser.add_argument('--radial-model', default='', choices=['', 'powerlaw_a', 'cored_powerlaw_a', 'logpoly3', 'step5'])
    parser.add_argument('--survivability-backend', default='', choices=['', 'baumgardt', 'gg23'])
    parser.add_argument(
        '--gg23-model',
        default='',
        choices=[
            '',
            'gg23_no_bh',
            'gg23_bh',
            'gg23_bh_feh_gradient',
            'gg23_bh_past_tidal',
            'gg23_bh_feh_gradient_past_tidal',
        ],
    )
    parser.add_argument('--mcmc-chains', type=int, default=6)
    parser.add_argument('--mcmc-steps', type=int, default=900)
    parser.add_argument('--mcmc-burn', type=int, default=300)
    parser.add_argument('--mcmc-thin', type=int, default=2)
    parser.add_argument('--mcmc-adapt-until', type=int, default=240)
    parser.add_argument('--mcmc-adapt-every', type=int, default=20)
    parser.add_argument('--mcmc-seed', type=int, default=20260527)
    parser.add_argument('--anchor-k', type=int, default=18)
    parser.add_argument('--anchor-pool', type=int, default=36)
    parser.add_argument('--chain-worker-config')
    parser.add_argument('--chain-worker-output')
    args = parser.parse_args()

    if args.chain_worker_config and args.chain_worker_output:
        _run_worker_mode(Path(args.chain_worker_config), Path(args.chain_worker_output))
        return

    output_root_name = args.output_root_name if args.output_root_name else args.source_output_root_name
    output_root = PROJECT_ROOT / 'variants' / output_root_name
    tables_dir = output_root / 'outputs' / 'tables'
    figures_dir = output_root / 'outputs' / 'figures'
    worker_dir = output_root / 'outputs' / 'parallel_exact_mcmc_workers'
    worker_dir.mkdir(parents=True, exist_ok=True)

    refined_table = pd.read_csv(tables_dir / 'refined_grid_results.csv')
    summary_path = tables_dir / 'summary.json'
    radial_model = 'powerlaw_a'
    survivability_backend = 'baumgardt'
    gg23_model_name = ''
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            radial_model = summary.get('model_spec', {}).get('radial_model', radial_model)
            survivability_backend = summary.get('survivability_backend', survivability_backend)
            gg23_model_name = summary.get('gg23_model_name', gg23_model_name)
        except Exception:
            pass
    if args.radial_model:
        radial_model = args.radial_model
    if args.survivability_backend:
        survivability_backend = args.survivability_backend
    if args.gg23_model:
        gg23_model_name = args.gg23_model
    if survivability_backend == 'gg23' and not gg23_model_name:
        raise ValueError('--gg23-model is required when using GG23 survivability.')
    if survivability_backend != 'gg23':
        gg23_model_name = ''
    refined_best_row = refined_table.loc[refined_table['log_likelihood'].idxmax()].to_dict()
    refined_bounds = np.array([
        [float(refined_table['eta_t'].min()), float(refined_table['eta_t'].max())],
        [float(refined_table['input_alpha_dndm'].min()), float(refined_table['input_alpha_dndm'].max())],
        [float(refined_table['input_log10_m_c_msun'].min()), float(refined_table['input_log10_m_c_msun'].max())],
    ], dtype=float)
    widths = refined_bounds[:, 1] - refined_bounds[:, 0]

    prepared_catalog = _load_catalog(output_root / 'outputs' / 'parallel_exact_mcmc_prepare')
    spec = JointModelSpec(imf_family='schechter', radial_model=radial_model)

    exact_anchor_entries = _build_exact_anchor_entries(
        prepared_catalog=prepared_catalog,
        spec=spec,
        project_root=output_root,
        refined_table=refined_table,
        refined_bounds=refined_bounds,
        anchor_k=int(args.anchor_k),
        anchor_pool=int(args.anchor_pool),
        survivability_backend=survivability_backend,
        gg23_model_name=gg23_model_name or None,
    )
    lightweight_anchors = [_lightweight_entry(entry, include_surfaces=True) for entry in exact_anchor_entries]
    chain_start_entries = _select_diverse_entries(
        exact_anchor_entries,
        n_select=int(args.mcmc_chains),
        bounds=refined_bounds,
        candidate_pool=max(int(args.anchor_pool), int(args.mcmc_chains) * 4),
    )
    while len(chain_start_entries) < int(args.mcmc_chains):
        chain_start_entries.append(chain_start_entries[-1])

    procs = []
    for chain_id in range(int(args.mcmc_chains)):
        config = {
            'chain_id': chain_id,
            'n_steps': int(args.mcmc_steps),
            'adapt_until': int(args.mcmc_adapt_until),
            'adapt_every': int(args.mcmc_adapt_every),
            'seed': int(args.mcmc_seed) + chain_id,
            'prepared_catalog': prepared_catalog,
            'spec': spec,
            'project_root': output_root,
            'bounds': refined_bounds,
            'widths': widths,
            'initial_entry': _lightweight_entry(chain_start_entries[chain_id], include_surfaces=True),
            'fixed_anchor_library': lightweight_anchors,
            'survivability_backend': survivability_backend,
            'gg23_model_name': gg23_model_name or None,
            'surface_output_path': str(worker_dir / f'chain_{chain_id}_selection_surfaces.npz'),
            'surface_burn_in': int(args.mcmc_burn),
            'surface_thin': int(args.mcmc_thin),
        }
        config_path = worker_dir / f'chain_{chain_id}_config.pkl'
        result_path = worker_dir / f'chain_{chain_id}_result.pkl'
        log_path = worker_dir / f'chain_{chain_id}.log'
        with config_path.open('wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_handle = log_path.open('w')
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                '--chain-worker-config', str(config_path),
                '--chain-worker-output', str(result_path),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        procs.append((chain_id, proc, log_handle, result_path, log_path))

    chain_results = []
    for chain_id, proc, log_handle, result_path, log_path in procs:
        return_code = proc.wait()
        log_handle.close()
        if return_code != 0:
            log_text = log_path.read_text()
            raise RuntimeError(f'Chain worker {chain_id} failed with code {return_code}\n{log_text}')
        with result_path.open('rb') as handle:
            chain_results.append(pickle.load(handle))
        best_row = chain_results[-1]['best_row']
        print(
            f"[parallel exact mcmc] chain={chain_id} done accept={float(chain_results[-1]['acceptance']):.3f} "
            f"best logL={float(best_row['log_likelihood']):.3f} eta_t={float(best_row['eta_t']):.3f} "
            f"alpha={float(best_row['input_alpha_dndm']):.3f} logMc={float(best_row['input_log10_m_c_msun']):.3f}"
        )

    chain_results.sort(key=lambda item: int(item['chain_id']))
    records = [row for result in chain_results for row in result['records']]
    chain_table = pd.DataFrame(records).sort_values(['chain', 'step']).reset_index(drop=True)
    chain_table.to_csv(tables_dir / 'exact_parallel_mcmc_chain.csv', index=False)

    posterior_parts = []
    for _, frame in chain_table.loc[chain_table['step'] >= int(args.mcmc_burn)].groupby('chain'):
        posterior_parts.append(frame.iloc[:: int(args.mcmc_thin)])
    posterior_table = pd.concat(posterior_parts, ignore_index=True)
    posterior_table.to_csv(tables_dir / 'exact_parallel_mcmc_posterior_samples.csv', index=False)

    summary_candidate_columns = [
        'eta_t',
        'input_alpha_dndm',
        'input_log10_m_c_msun',
        'gamma_linear_a',
        'log10_a_core_kpc',
        'final_total_initial_count_above_log10_4',
        'final_total_initial_stellar_mass_above_log10_4_msun',
        'mean_detectability_above_log10_4',
        'log_likelihood',
    ]
    summary_rows = []
    summary_columns = []
    for column in summary_candidate_columns:
        if column not in posterior_table.columns:
            continue
        values = np.asarray(posterior_table[column], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        q16, q50, q84 = np.quantile(finite, [0.16, 0.50, 0.84])
        summary_rows.append({
            'parameter': column,
            'q16': float(q16),
            'q50': float(q50),
            'q84': float(q84),
            'minus': float(q50 - q16),
            'plus': float(q84 - q50),
        })
        summary_columns.append(column)
    posterior_summary = pd.DataFrame(summary_rows)
    posterior_summary.to_csv(tables_dir / 'exact_parallel_posterior_summary.csv', index=False)

    rhat = {}
    for column in ['eta_t', 'input_alpha_dndm', 'input_log10_m_c_msun', 'gamma_linear_a', 'log10_a_core_kpc']:
        if column not in posterior_table.columns:
            continue
        pivot = (
            posterior_table.pivot_table(index='step', columns='chain', values=column, aggfunc='last')
            .dropna()
            .to_numpy()
            .T
        )
        if pivot.size == 0:
            continue
        rhat[column] = _compute_rhat(pivot)

    acceptance_by_chain = {str(result['chain_id']): float(result['acceptance']) for result in chain_results}
    best_posterior_row = posterior_table.sort_values('log_likelihood', ascending=False).iloc[0].to_dict()

    _corner_plot(posterior_table, refined_best_row, figures_dir / 'exact_parallel_profiled_posterior_corner.png')
    _trace_plot(chain_table, figures_dir / 'exact_parallel_profiled_posterior_traces.png', burn_in=int(args.mcmc_burn))

    best_theta = np.array([
        float(best_posterior_row['eta_t']),
        float(best_posterior_row['input_alpha_dndm']),
        float(best_posterior_row['input_log10_m_c_msun']),
    ], dtype=float)
    best_anchor = _select_anchor_start_state(theta=best_theta, anchors=lightweight_anchors, bounds=refined_bounds)
    best_entry = _evaluate_theta_multistart(
        prepared_catalog=prepared_catalog,
        spec=spec,
        theta=best_theta,
        stage='exact_parallel_mcmc_best',
        project_root=output_root,
        anchor_start_state=best_anchor,
        survivability_backend=survivability_backend,
        gg23_model_name=gg23_model_name or None,
    )
    _save_best_payload(best_entry, tables_dir, prefix='exact_parallel_mcmc')

    summary = {
        'source_output_root_name': args.source_output_root_name,
        'output_root_name': output_root_name,
        'survivability_backend': survivability_backend,
        'gg23_model_name': gg23_model_name,
        'gg23_mini_eta_t_dependent': bool(survivability_backend == 'gg23'),
        'sampler': 'exact_profiled_random_walk_metropolis_subprocess_parallel',
        'n_chains': int(args.mcmc_chains),
        'n_steps': int(args.mcmc_steps),
        'burn_in': int(args.mcmc_burn),
        'thin': int(args.mcmc_thin),
        'acceptance_by_chain': acceptance_by_chain,
        'rhat': rhat,
        'best_posterior_sample': json.loads(pd.Series(best_posterior_row).to_json()),
        'posterior_summary': posterior_summary.to_dict(orient='records'),
        'worker_cache_sizes': {str(result['chain_id']): int(result['cache_size']) for result in chain_results},
        'selection_surface_archive': {
            str(result['chain_id']): {
                'path': result.get('surface_path'),
                'n_surface_records': int(result.get('n_surface_records', 0)),
            }
            for result in chain_results
        },
        'anchor_count': int(len(lightweight_anchors)),
        'refined_bounds': refined_bounds.tolist(),
    }
    (tables_dir / 'exact_parallel_mcmc_summary.json').write_text(json.dumps(summary, indent=2))

    print(figures_dir / 'exact_parallel_profiled_posterior_corner.png')
    print(figures_dir / 'exact_parallel_profiled_posterior_traces.png')
    print(tables_dir / 'exact_parallel_posterior_summary.csv')
    print(tables_dir / 'exact_parallel_mcmc_summary.json')


if __name__ == '__main__':
    main()
