from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlertLevel(str, Enum):
    GREEN = "Green"
    AMBER = "Amber"
    RED = "Red"
    NOT_SCORED = "Not Scored"


@dataclass(frozen=True)
class RuleResult:
    metric: str
    value: float | None
    level: AlertLevel
    action: str


@dataclass(frozen=True)
class StressScenario:
    interest_rate_delta: float = 0.0
    gdp_growth_delta: float = 0.0
    unemployment_delta: float = 0.0
    lgd_default: float = 0.45


@dataclass(frozen=True)
class ValidationMetrics:
    row_count: int
    ks_statistic: float | None
    auc: float | None
    gini: float | None
    calibration_variance: float
    hosmer_lemeshow: float
    hosmer_lemeshow_df: int
    hosmer_lemeshow_pvalue: float
    psi: float
    base_ecl: float
    stressed_ecl: float
    ecl_delta: float
    outcome_metrics_scored: bool
    outcome_exception: str | None
    stressed_ks_statistic: float | None = None
    stressed_auc: float | None = None
    stressed_gini: float | None = None
    stressed_calibration_variance: float | None = None
    stressed_hosmer_lemeshow: float | None = None
    stressed_hosmer_lemeshow_df: int | None = None
    stressed_hosmer_lemeshow_pvalue: float | None = None
    stressed_psi: float | None = None

