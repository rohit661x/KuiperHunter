"""Tests for the prior report generator in sanity.py."""
import numpy as np
import pytest


class TestReportPriors:
    def test_returns_results_dict(self, tmp_path):
        from src.injector.sanity import report_priors
        results = report_priors(
            sample_type="tno",
            mode="kbo",
            N=500,
            T=5,
            baseline_hours=4.0,
            plate_scale=0.187,
            patch=128,
            out_dir=str(tmp_path),
            seed=0,
            plot=False,
        )
        assert isinstance(results, dict)
        for key in ("R_au", "mu", "phi_offset", "snr", "drift_px", "clipping_rate"):
            assert key in results, f"Missing key: {key}"

    def test_clipping_rate_is_fraction(self, tmp_path):
        from src.injector.sanity import report_priors
        results = report_priors(
            sample_type="tno", mode="kbo", N=500,
            T=5, baseline_hours=4.0, plate_scale=0.187, patch=128,
            out_dir=str(tmp_path), seed=0, plot=False,
        )
        assert 0.0 <= results["clipping_rate"] <= 1.0

    def test_all_kbo_checks_pass(self, tmp_path):
        from src.injector.sanity import report_priors
        results = report_priors(
            sample_type="tno", mode="kbo", N=5000,
            T=5, baseline_hours=4.0, plate_scale=0.187, patch=128,
            out_dir=str(tmp_path), seed=0, plot=False,
        )
        checks = results["checks"]
        failed = [name for name, ok in checks.items() if not ok]
        assert not failed, f"KBO checks failed: {failed}"

    def test_clipping_file_written(self, tmp_path):
        from src.injector.sanity import report_priors
        report_priors(
            sample_type="tno", mode="kbo", N=200,
            T=5, baseline_hours=4.0, plate_scale=0.187, patch=128,
            out_dir=str(tmp_path), seed=0, plot=False,
        )
        assert (tmp_path / "clipping_rate.txt").exists()

    def test_unsupported_sample_type_raises(self):
        from src.injector.sanity import report_priors
        with pytest.raises(ValueError, match="report_priors\\(\\) only supports"):
            report_priors(sample_type="mba", mode="kbo", N=10, seed=0, plot=False)
