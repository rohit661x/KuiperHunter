"""Tests for kbo_prior dataclasses and configuration."""
import math
import pytest
import numpy as np
from src.injector.kbo_prior import KBOConfig, KBOSample


class TestKBOConfig:
    def test_defaults(self):
        c = KBOConfig()
        assert c.mode == "kbo"
        assert c.phi_ecl_sigma_deg == pytest.approx(10.0)
        assert c.motion_scatter == pytest.approx(0.08)
        assert c.nominal_sigma_sky == pytest.approx(10.0)
        assert c.plate_scale == pytest.approx(0.187)
        assert c.baseline_hours == pytest.approx(4.0)
        assert c.T == 5

    def test_dt_hours_property(self):
        c = KBOConfig(baseline_hours=4.0, T=5)
        assert c.dt_hours == pytest.approx(1.0)  # 4/(5-1) = 1

    def test_override(self):
        c = KBOConfig(mode="broad", T=3)
        assert c.mode == "broad"
        assert c.T == 3

    def test_T_less_than_2_raises(self):
        import pytest
        with pytest.raises(ValueError, match="T must be >= 2"):
            KBOConfig(T=1)


class TestKBOSampleFields:
    def test_has_all_fields(self):
        """KBOSample must carry every field specified in the design doc."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(KBOSample)}
        required = {
            "population_class", "R_au", "mu_arcsec_hr",
            "phi_offset_deg", "snr", "dropout_mask",
            "phi_img_rad", "motion_ra", "motion_dec",
            "vx_px_per_frame", "vy_px_per_frame",
            "flux_peak", "mode",
        }
        assert required <= fields


class TestPopulationClass:
    VALID = {"classical_cold", "classical_hot", "plutino", "scattering"}

    def test_returns_valid_class(self):
        from src.injector.kbo_prior import _sample_population_class
        rng = np.random.default_rng(0)
        for _ in range(100):
            cls = _sample_population_class(rng)
            assert cls in self.VALID

    def test_mixture_proportions(self):
        """Over 50 000 draws the mixture should be within 2 pp of target."""
        from src.injector.kbo_prior import _sample_population_class, KBO_MIXTURE
        rng = np.random.default_rng(42)
        N = 50_000
        counts = {k: 0 for k in KBO_MIXTURE}
        for _ in range(N):
            counts[_sample_population_class(rng)] += 1
        for cls, target_frac in KBO_MIXTURE.items():
            actual = counts[cls] / N
            assert abs(actual - target_frac) < 0.02, (
                f"{cls}: expected ~{target_frac:.2f}, got {actual:.3f}"
            )


class TestDistanceSampling:
    def test_classical_cold_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(0)
        for _ in range(1000):
            R = _sample_R("classical_cold", rng)
            assert 40.0 <= R <= 50.0, f"classical_cold R={R} out of [40,50]"

    def test_classical_hot_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(1)
        for _ in range(1000):
            R = _sample_R("classical_hot", rng)
            assert 40.0 <= R <= 50.0

    def test_plutino_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(2)
        for _ in range(1000):
            R = _sample_R("plutino", rng)
            assert 37.0 <= R <= 42.0

    def test_scattering_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(3)
        for _ in range(1000):
            R = _sample_R("scattering", rng)
            assert 30.0 <= R <= 100.0

    def test_classical_cold_mean_near_44(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(99)
        Rs = [_sample_R("classical_cold", rng) for _ in range(10_000)]
        mean = np.mean(Rs)
        assert 43.0 < mean < 45.0, f"classical_cold mean R={mean:.2f}, expected near 44"

    def test_plutino_mean_near_394(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(99)
        Rs = [_sample_R("plutino", rng) for _ in range(10_000)]
        mean = np.mean(Rs)
        assert 39.0 < mean < 40.0, f"plutino mean R={mean:.2f}, expected near 39.4"

    def test_unknown_class_raises(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unknown population_class"):
            _sample_R("centaur", rng)
