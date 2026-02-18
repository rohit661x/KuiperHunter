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


class TestMotionModel:
    def test_mu_at_44au(self):
        """mu(44 AU) should be near 2.92 arcsec/hr (opposition approximation)."""
        from src.injector.kbo_prior import mu_of_R
        mu = mu_of_R(44.0)
        assert 2.8 < mu < 3.1, f"mu(44)={mu:.3f}, expected ~2.92"

    def test_mu_decreases_with_distance(self):
        from src.injector.kbo_prior import mu_of_R
        assert mu_of_R(30.0) > mu_of_R(44.0) > mu_of_R(80.0)

    def test_mu_sample_cap_kbo(self):
        from src.injector.kbo_prior import _sample_mu, KBOConfig
        cfg = KBOConfig(mode="kbo")
        rng = np.random.default_rng(0)
        for _ in range(5000):
            mu = _sample_mu(30.0, cfg, rng)  # R=30 → high mu; cap must hold
            assert mu <= 4.5 + 1e-9, f"mu={mu:.4f} exceeds 4.5 cap"

    def test_mu_positive(self):
        from src.injector.kbo_prior import _sample_mu, KBOConfig
        cfg = KBOConfig()
        rng = np.random.default_rng(0)
        for _ in range(1000):
            mu = _sample_mu(44.0, cfg, rng)
            assert mu > 0

    def test_mu_scatter_mean_near_nominal(self):
        """With scatter=0.08, sample mean should converge to mu_of_R(44)."""
        from src.injector.kbo_prior import _sample_mu, mu_of_R, KBOConfig
        cfg = KBOConfig(motion_scatter=0.08)
        rng = np.random.default_rng(7)
        samples = [_sample_mu(44.0, cfg, rng) for _ in range(10_000)]
        nominal = mu_of_R(44.0)
        assert abs(np.mean(samples) - nominal) < 0.05


class TestDirectionAndSNR:
    def test_phi_offset_is_normal(self):
        """phi_offset ~ Normal(0, sigma); check mean ≈ 0 and std ≈ sigma."""
        from src.injector.kbo_prior import _sample_phi_offset, KBOConfig
        cfg = KBOConfig(phi_ecl_sigma_deg=10.0)
        rng = np.random.default_rng(0)
        samples = [_sample_phi_offset(cfg, rng) for _ in range(20_000)]
        assert abs(np.mean(samples)) < 0.5
        assert abs(np.std(samples) - 10.0) < 0.3

    def test_phi_img_rad_canonical(self):
        """phi_img_rad = pi + radians(phi_offset)."""
        from src.injector.kbo_prior import _phi_img_rad
        assert _phi_img_rad(0.0) == pytest.approx(math.pi)
        assert _phi_img_rad(90.0) == pytest.approx(math.pi + math.pi / 2)

    def test_snr_always_in_range(self):
        from src.injector.kbo_prior import _sample_snr
        rng = np.random.default_rng(0)
        for _ in range(5000):
            snr = _sample_snr(rng)
            assert 3.0 <= snr <= 10.0, f"snr={snr} out of [3, 10]"

    def test_snr_faint_heavy(self):
        """~75% of SNR draws should fall in [3, 6]."""
        from src.injector.kbo_prior import _sample_snr
        rng = np.random.default_rng(42)
        N = 20_000
        faint = sum(1 for _ in range(N) if _sample_snr(rng) <= 6.0)
        frac = faint / N
        assert 0.70 < frac < 0.80, f"faint fraction={frac:.3f}, expected ~0.75"


class TestSampleKBO:
    def _draw(self, seed=0):
        from src.injector.kbo_prior import sample_kbo, KBOConfig
        rng = np.random.default_rng(seed)
        return sample_kbo(rng, KBOConfig())

    def test_returns_kbo_sample(self):
        from src.injector.kbo_prior import KBOSample
        s = self._draw()
        assert isinstance(s, KBOSample)

    def test_motion_ra_dec_components(self):
        """motion_ra/dec must equal mu * cos/sin(phi_img_rad)."""
        s = self._draw()
        assert s.motion_ra == pytest.approx(
            s.mu_arcsec_hr * math.cos(s.phi_img_rad), abs=1e-9
        )
        assert s.motion_dec == pytest.approx(
            s.mu_arcsec_hr * math.sin(s.phi_img_rad), abs=1e-9
        )

    def test_vx_vy_mirror_trajectory_formula(self):
        """vx/vy_px_per_frame must use the same formula as trajectory.py."""
        from src.injector.kbo_prior import KBOConfig
        cfg = KBOConfig()
        s = self._draw()
        expected_vx = (s.motion_ra / cfg.plate_scale) * cfg.dt_hours
        expected_vy = (s.motion_dec / cfg.plate_scale) * cfg.dt_hours
        assert s.vx_px_per_frame == pytest.approx(expected_vx, abs=1e-9)
        assert s.vy_px_per_frame == pytest.approx(expected_vy, abs=1e-9)

    def test_flux_peak_equals_snr(self):
        """flux_peak is dimensionless SNR placeholder in Step 2."""
        s = self._draw()
        assert s.flux_peak == pytest.approx(s.snr)

    def test_dropout_mask_is_none(self):
        s = self._draw()
        assert s.dropout_mask is None

    def test_mode_is_kbo(self):
        s = self._draw()
        assert s.mode == "kbo"

    def test_seed_reproducibility(self):
        from src.injector.kbo_prior import sample_kbo, KBOConfig
        cfg = KBOConfig()
        s1 = sample_kbo(np.random.default_rng(99), cfg)
        s2 = sample_kbo(np.random.default_rng(99), cfg)
        assert s1.R_au == s2.R_au
        assert s1.mu_arcsec_hr == s2.mu_arcsec_hr
        assert s1.snr == s2.snr

    def test_mu_capped_at_4_5(self):
        from src.injector.kbo_prior import sample_kbo, KBOConfig
        cfg = KBOConfig()
        rng = np.random.default_rng(0)
        for _ in range(2000):
            s = sample_kbo(rng, cfg)
            assert s.mu_arcsec_hr <= 4.5 + 1e-9


class TestPriorAdapter:
    def test_tno_returns_prior_sample(self):
        from src.injector.priors import sample, PriorSample
        rng = np.random.default_rng(0)
        s = sample("tno", rng)
        assert isinstance(s, PriorSample)

    def test_tno_motion_ra_is_arcsec_hr(self):
        """motion_ra from adapter must be in arcsec/hr (trajectory.py contract)."""
        from src.injector.priors import sample
        rng = np.random.default_rng(0)
        for _ in range(200):
            s = sample("tno", rng)
            assert abs(s.motion_ra) <= 4.5 + 1e-6, (
                f"motion_ra={s.motion_ra:.3f} exceeds KBO cap — likely px/frame confusion"
            )

    def test_tno_flux_peak_equals_snr_range(self):
        """flux_peak = snr (dimensionless, 3–10)."""
        from src.injector.priors import sample
        rng = np.random.default_rng(0)
        for _ in range(200):
            s = sample("tno", rng)
            assert 3.0 <= s.flux_peak <= 10.0

    def test_other_priors_unchanged(self):
        """mba/nea/static must still work after modifying tno."""
        from src.injector.priors import sample
        rng = np.random.default_rng(0)
        for name in ("mba", "nea", "static"):
            s = sample(name, rng)
            assert s.flux_peak > 0
            assert s.motion_ra is not None
