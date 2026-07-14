"""Cross-script contracts for the paper asset pipeline.

Regression suite for two failures that made the paper impossible to rebuild from a
clean tree:

* three scripts wrote `paper_results_summary.json` with `write_text(json.dumps(...))`,
  clobbering each other, so whichever ran last erased the other two schemas;
* the single-component detectability EM was the only EM summary writer that did not
  emit `best_joint_model`, which its readers assume.

Both only ever "worked" because stale artefacts from an older code generation were
sitting on disk.
"""

from __future__ import annotations

import json

import pytest

from globular_clusters_imf.paper_summary import (
    PaperSummaryKeyError,
    merge_paper_results_summary,
    read_paper_results_summary,
    require_paper_summary_keys,
)


class TestMergeSemantics:
    def test_second_writer_does_not_erase_the_first(self, tmp_path):
        """The bug: three writers, each clobbering the file."""
        merge_paper_results_summary(tmp_path, {"single_component_best_model": {"total_initial_count": 1234.0}})
        merge_paper_results_summary(tmp_path, {"single_component_exact_global_logpoly3": {"alpha": -1.2}})
        merge_paper_results_summary(tmp_path, {"split_alpha_two_component": {"alpha_in_situ": -0.9}})

        summary = read_paper_results_summary(tmp_path)
        assert set(summary) == {
            "single_component_best_model",
            "single_component_exact_global_logpoly3",
            "split_alpha_two_component",
        }
        assert summary["single_component_best_model"]["total_initial_count"] == 1234.0

    def test_rewriting_a_key_updates_it(self, tmp_path):
        merge_paper_results_summary(tmp_path, {"a": 1, "b": 2})
        merge_paper_results_summary(tmp_path, {"b": 99})
        assert read_paper_results_summary(tmp_path) == {"a": 1, "b": 99}

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_paper_results_summary(tmp_path) == {}

    def test_merge_creates_the_directory(self, tmp_path):
        target = tmp_path / "nested" / "tables"
        merge_paper_results_summary(target, {"k": 1})
        assert json.loads((target / "paper_results_summary.json").read_text()) == {"k": 1}


class TestKeyGuard:
    def test_missing_key_names_the_script_to_run(self, tmp_path):
        summary = read_paper_results_summary(tmp_path)
        with pytest.raises(PaperSummaryKeyError) as excinfo:
            require_paper_summary_keys(summary, "single_component_best_model")
        message = str(excinfo.value)
        # a bare KeyError told the user nothing; this must point at the producer
        assert "build_paper_assets.py" in message
        assert "single_component_best_model" in message

    def test_present_keys_pass(self, tmp_path):
        merge_paper_results_summary(tmp_path, {"single_component_best_model": {}})
        require_paper_summary_keys(read_paper_results_summary(tmp_path), "single_component_best_model")


class TestEmSummarySchema:
    """Two kinds of EM summary exist, and they must not be confused.

    * A *comparison* writer fits several specs, picks a winner, and reports it under
      `best_joint_model` / `best_model_detectability_summary`.
    * A *single-model* writer fits ONE spec. There is no "best" to report; the fitted
      model is `final_model` and its detectability is `final_mean_detectability`.

    `run_abs_longitude_detectability_inference` read the comparison keys out of the
    single-model file and raised KeyError. The right fix is to read the keys that file
    actually has -- not to duplicate comparison keys into a payload that has no
    comparison in it.
    """

    def test_single_model_payload_reports_final_model_not_a_best(self):
        import inspect

        from globular_clusters_imf.detectability_model import fit_single_component_detectability_em

        source = inspect.getsource(fit_single_component_detectability_em)
        assert '"final_model"' in source
        assert '"final_mean_detectability"' in source
        assert '"best_joint_model"' not in source, (
            "the single-spec EM must not claim to have chosen a 'best' model"
        )

    def test_comparison_writers_do_report_a_best(self):
        import inspect

        from globular_clusters_imf import detectability_longitude_model, two_component_model

        for module in (detectability_longitude_model, two_component_model):
            source = inspect.getsource(module)
            assert '"best_joint_model"' in source, (
                f"{module.__name__} compares several specs but never names a best"
            )

    def test_reader_uses_the_keys_the_single_model_payload_actually_has(self):
        """Guards the exact miswiring that produced the KeyError."""
        from pathlib import Path

        source = Path("scripts/run_abs_longitude_detectability_inference.py").read_text()
        assert 'baseline_detectability_summary["best_joint_model"]' not in source
        assert 'baseline_detectability_summary["best_model_detectability_summary"]' not in source
        assert 'baseline_detectability_summary["final_model"]' in source
