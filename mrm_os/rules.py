from __future__ import annotations

from .config import REGULATORY_REF
from .models import AlertLevel, RuleResult, ValidationMetrics


def evaluate_ks(ks: float | None, sample_size: int | None = None) -> RuleResult:
    if ks is None:
        action = f"Outcome metric was not scored because the sample size is below the {REGULATORY_REF} validation minimum."
        if sample_size is not None:
            action = (
                f"Exception: Sample size (N={sample_size}) is beneath the {REGULATORY_REF} validation minimum of 500. "
                "Results are statistically invalid."
            )
        return RuleResult("KS Statistic", None, AlertLevel.NOT_SCORED, action)
    if ks > 0.40:
        return RuleResult("KS Statistic", ks, AlertLevel.GREEN, "Continue standard monitoring.")
    if ks >= 0.30:
        return RuleResult("KS Statistic", ks, AlertLevel.AMBER, "Increase monitoring cadence and document challenger evidence.")
    return RuleResult("KS Statistic", ks, AlertLevel.RED, "Flag discriminatory failure; demand challenger model.")


def evaluate_psi(psi: float) -> RuleResult:
    if psi < 0.10:
        return RuleResult("Population Stability Index", psi, AlertLevel.GREEN, "Continue standard monitoring.")
    if psi <= 0.25:
        return RuleResult("Population Stability Index", psi, AlertLevel.AMBER, "Monitor for portfolio drift and refresh stability analysis.")
    return RuleResult("Population Stability Index", psi, AlertLevel.RED, "Trigger severe data drift alert; halt model usage.")


def evaluate_calibration(variance: float) -> RuleResult:
    if variance < 0.05:
        return RuleResult("Expected vs Actual Variance", variance, AlertLevel.GREEN, "Continue standard monitoring.")
    if variance <= 0.10:
        return RuleResult("Expected vs Actual Variance", variance, AlertLevel.AMBER, "Review calibration overlays and provisioning sensitivity.")
    return RuleResult("Expected vs Actual Variance", variance, AlertLevel.RED, "Force provision override and ECL adjustment.")


def evaluate_all(metrics: ValidationMetrics) -> list[RuleResult]:
    return [
        evaluate_ks(metrics.ks_statistic, metrics.row_count),
        evaluate_psi(metrics.psi),
        evaluate_calibration(metrics.calibration_variance),
    ]


def highest_alert(results: list[RuleResult]) -> AlertLevel:
    if any(result.level == AlertLevel.RED for result in results):
        return AlertLevel.RED
    if any(result.level == AlertLevel.AMBER for result in results):
        return AlertLevel.AMBER
    return AlertLevel.GREEN
