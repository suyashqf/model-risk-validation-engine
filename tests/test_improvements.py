from __future__ import annotations

import tempfile
from pathlib import Path
import polars as pl
import pytest

from mrm_os.config import load_validation_config
from mrm_os.analytics import run_validation, compute_stressed_pds
from mrm_os.models import StressScenario, AlertLevel, RuleResult
from mrm_os.reporting import build_validation_context, render_metric_tables


def test_empty_config_file_raises_value_error() -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as f:
        f.write(b"   \n   ")
        temp_path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="Validation config file is empty"):
            load_validation_config(temp_path)
    finally:
        temp_path.unlink()


def test_compute_stressed_pds_under_shock() -> None:
    df = pl.DataFrame({
        "model_pd": [0.1, 0.2, 0.3],
        "lgd": [0.45, 0.45, 0.45]
    })
    scenario = StressScenario(interest_rate_delta=2.0, gdp_growth_delta=-1.0)
    stressed_pds = compute_stressed_pds(df, scenario)
    
    assert len(stressed_pds) == 3
    # Shocks should shift the PDs upwards
    assert stressed_pds[0] > df["model_pd"][0]
    assert stressed_pds[1] > df["model_pd"][1]
    assert stressed_pds[2] > df["model_pd"][2]


def test_stressed_metrics_recomputation() -> None:
    # A tiny frame with N >= 500 to scoring outcomes
    df = pl.DataFrame({
        "loan_id": [f"L{i}" for i in range(500)],
        "exposure_at_default": [100.0] * 500,
        "model_pd": [0.05 + 0.001 * (i % 100) for i in range(500)],
        "actual_outcome": [1 if i % 5 == 0 else 0 for i in range(500)],
        "vintage_month": ["2024-01-01"] * 500
    })
    
    # Run validation under base scenario (no shock)
    metrics_base, _ = run_validation(df, StressScenario())
    
    # Run validation under gdp growth shock
    metrics_stressed, _ = run_validation(df, StressScenario(gdp_growth_delta=-2.5))
    
    assert metrics_stressed.stressed_calibration_variance is not None
    assert metrics_stressed.stressed_hosmer_lemeshow is not None
    # Stressed metrics (calibration variance and HL) should change from base
    assert metrics_stressed.stressed_calibration_variance != metrics_base.calibration_variance or metrics_stressed.stressed_hosmer_lemeshow != metrics_base.hosmer_lemeshow


def test_hl_statistic_row_format_and_comparison_table() -> None:
    # Set up mock metrics
    from mrm_os.models import ValidationMetrics
    metrics = ValidationMetrics(
        row_count=1000,
        ks_statistic=0.45,
        auc=0.80,
        gini=0.60,
        calibration_variance=0.04,
        hosmer_lemeshow=12.5,
        hosmer_lemeshow_df=8,
        hosmer_lemeshow_pvalue=0.1301,
        psi=0.05,
        base_ecl=1000.0,
        stressed_ecl=1200.0,
        ecl_delta=200.0,
        outcome_metrics_scored=True,
        outcome_exception=None,
        stressed_ks_statistic=0.40,
        stressed_auc=0.75,
        stressed_gini=0.50,
        stressed_calibration_variance=0.08,
        stressed_hosmer_lemeshow=18.2,
        stressed_hosmer_lemeshow_df=8,
        stressed_hosmer_lemeshow_pvalue=0.0198,
        stressed_psi=0.12
    )
    
    rules = [
        RuleResult("KS Statistic", 0.45, AlertLevel.GREEN, "Continue standard monitoring.")
    ]
    
    tables_md = render_metric_tables(metrics, rules)
    
    # Check Hosmer-Lemeshow formatting
    assert "Hosmer-Lemeshow χ²(8)" in tables_md
    assert "12.5000, p = 0.1301 — Pass" in tables_md
    
    # Check Comparison table columns
    assert "| Metric | Base | Stressed | Change |" in tables_md
    assert "| KS Statistic | 0.4500 | 0.4000 | -0.0500 |" in tables_md
    assert "| AUC | 0.8000 | 0.7500 | -0.0500 |" in tables_md
    assert "| Calibration Variance | 0.0400 | 0.0800 | +0.0400 |" in tables_md


def test_breach_narrative_context_trigger() -> None:
    config = {
        "model_name": "Test Model",
        "model_type": "Logistic Regression",
        "features": ["interest_rate"],
        "intended_use": "IFRS 9 ECL",
        "development_data": "Synthetic loan portfolio",
        "validation_data": "Synthetic - 1000 obs",
        "data_type": "synthetic",
        "analyst_name": "Suyash",
        "review_date": "2026-06-08"
    }
    
    from mrm_os.models import ValidationMetrics
    metrics = ValidationMetrics(
        row_count=1000, ks_statistic=0.25, auc=0.8, gini=0.6,
        calibration_variance=0.04, hosmer_lemeshow=12.5,
        hosmer_lemeshow_df=8, hosmer_lemeshow_pvalue=0.13, psi=0.05,
        base_ecl=100.0, stressed_ecl=100.0, ecl_delta=0.0,
        outcome_metrics_scored=True, outcome_exception=None
    )
    
    # Trigger RED breach on KS Statistic (<0.30)
    rules = [
        RuleResult("KS Statistic", 0.25, AlertLevel.RED, "Demand challenger model.")
    ]
    
    context = build_validation_context(config, metrics, rules, AlertLevel.RED, StressScenario())
    
    assert context["first_red_breach"] is not None
    assert context["first_red_breach"]["metric"] == "KS Statistic"
    assert context["first_red_breach"]["value"] == 0.25
    assert context["first_red_breach"]["red_threshold"] == "0.30"
