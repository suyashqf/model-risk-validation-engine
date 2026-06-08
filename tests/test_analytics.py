from __future__ import annotations

import polars as pl

from mrm_os.analytics import MIN_OUTCOME_SAMPLE_SIZE, calculate_auc_gini, calculate_ks, calculate_psi, normalize_schema, run_validation
from mrm_os.models import StressScenario
from mrm_os.rules import AlertLevel, evaluate_ks, evaluate_psi


def sample_frame() -> pl.DataFrame:
    return normalize_schema(
        pl.DataFrame(
            {
                "loan_id": ["a", "b", "c", "d", "e", "f"],
                "exposure_at_default": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "model_pd": [0.01, 0.02, 0.10, 0.20, 0.60, 0.80],
                "actual_outcome": [0, 0, 0, 1, 1, 1],
                "vintage_month": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"],
                "lgd": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
            }
        )
    )


def test_ks_and_gini_are_high_for_ordered_defaults() -> None:
    frame = sample_frame()
    auc, gini = calculate_auc_gini(frame)

    assert calculate_ks(frame) == 1.0
    assert auc == 1.0
    assert gini == 1.0


def test_stress_scenario_increases_ecl_under_rate_shock() -> None:
    metrics, details = run_validation(sample_frame(), StressScenario(interest_rate_delta=2.5))

    assert metrics.stressed_ecl > metrics.base_ecl
    assert metrics.ecl_delta == details["metrics"]["ecl_delta"]
    assert metrics.outcome_metrics_scored is False
    assert "SR 26-2 validation minimum" in (metrics.outcome_exception or "")


def test_threshold_rules_match_risk_appetite() -> None:
    assert evaluate_ks(None, sample_size=24).level == AlertLevel.NOT_SCORED
    assert evaluate_ks(0.29).level == AlertLevel.RED
    assert evaluate_ks(0.35).level == AlertLevel.AMBER
    assert evaluate_ks(0.45).level == AlertLevel.GREEN
    assert evaluate_psi(0.30).level == AlertLevel.RED


def test_validation_scores_outcomes_only_above_sample_floor() -> None:
    frame = normalize_schema(
        pl.DataFrame(
            {
                "loan_id": [f"loan-{idx}" for idx in range(MIN_OUTCOME_SAMPLE_SIZE)],
                "exposure_at_default": [100.0] * MIN_OUTCOME_SAMPLE_SIZE,
                "model_pd": [0.01 + (idx / MIN_OUTCOME_SAMPLE_SIZE) * 0.5 for idx in range(MIN_OUTCOME_SAMPLE_SIZE)],
                "actual_outcome": [1 if idx >= MIN_OUTCOME_SAMPLE_SIZE // 2 else 0 for idx in range(MIN_OUTCOME_SAMPLE_SIZE)],
                "vintage_month": ["2024-01-01"] * MIN_OUTCOME_SAMPLE_SIZE,
            }
        )
    )

    metrics, _ = run_validation(frame)

    assert metrics.outcome_metrics_scored is True
    assert metrics.ks_statistic is not None
    assert metrics.auc is not None
    assert metrics.gini is not None


def test_psi_laplace_smoothing_handles_empty_bins() -> None:
    frame = normalize_schema(
        pl.DataFrame(
            {
                "loan_id": [f"loan-{idx}" for idx in range(20)],
                "exposure_at_default": [100.0] * 20,
                "model_pd": [0.01] * 10 + [0.95] * 10,
                "actual_outcome": [0] * 10 + [1] * 10,
                "vintage_month": ["2024-01-01"] * 10 + ["2025-01-01"] * 10,
            }
        )
    )

    psi = calculate_psi(frame)

    assert psi > 0
    assert psi < 20
