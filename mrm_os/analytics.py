from __future__ import annotations

import math
from dataclasses import asdict

import polars as pl
from scipy import stats

from .config import REGULATORY_REF
from .models import StressScenario, ValidationMetrics


MIN_OUTCOME_SAMPLE_SIZE = 500
PSI_EPSILON = 1e-4
PD_EPSILON = 1e-6
BETA_RATE = 0.15
BETA_GDP = -0.12
BETA_UNEMPLOYMENT = 0.10

REQUIRED_COLUMNS = {
    "loan_id": pl.Utf8,
    "exposure_at_default": pl.Float64,
    "model_pd": pl.Float64,
    "actual_outcome": pl.Int64,
    "vintage_month": pl.Date,
}


def load_portfolio_csv(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    return normalize_schema(df)


def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    normalized = df.with_columns(
        [
            pl.col("loan_id").cast(pl.Utf8),
            pl.col("exposure_at_default").cast(pl.Float64),
            pl.col("model_pd").cast(pl.Float64),
            pl.col("actual_outcome").cast(pl.Int64),
            pl.col("vintage_month").cast(pl.Date, strict=False),
        ]
    )

    invalid_pd = normalized.filter((pl.col("model_pd") < 0) | (pl.col("model_pd") > 1)).height
    invalid_outcome = normalized.filter(~pl.col("actual_outcome").is_in([0, 1])).height
    if invalid_pd:
        raise ValueError("model_pd must be between 0 and 1.")
    if invalid_outcome:
        raise ValueError("actual_outcome must be binary: 0 or 1.")

    return normalized.drop_nulls(["loan_id", "exposure_at_default", "model_pd", "actual_outcome", "vintage_month"])


def calculate_ks(df: pl.DataFrame) -> float:
    scored = df.sort("model_pd", descending=True).with_columns(
        [
            (pl.col("actual_outcome") == 1).cast(pl.Int64).alias("bad"),
            (pl.col("actual_outcome") == 0).cast(pl.Int64).alias("good"),
        ]
    )
    total_bad = scored["bad"].sum()
    total_good = scored["good"].sum()
    if total_bad == 0 or total_good == 0:
        return 0.0

    cdf = scored.with_columns(
        [
            (pl.col("bad").cum_sum() / total_bad).alias("cdf_bad"),
            (pl.col("good").cum_sum() / total_good).alias("cdf_good"),
        ]
    )
    return float((cdf["cdf_bad"] - cdf["cdf_good"]).abs().max())


def calculate_auc_gini(df: pl.DataFrame) -> tuple[float, float]:
    pairs = df.select(["model_pd", "actual_outcome"]).sort("model_pd").iter_rows()
    rows = list(pairs)
    positives = sum(1 for _, outcome in rows if outcome == 1)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
        return 0.5, 0.0

    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(rows):
        score = rows[index][0]
        start = index
        while index < len(rows) and rows[index][0] == score:
            index += 1
        end = index
        avg_rank = (rank + rank + (end - start) - 1) / 2
        rank_sum += avg_rank * sum(1 for _, outcome in rows[start:end] if outcome == 1)
        rank += end - start

    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc), float(2 * auc - 1)


def calibration_by_decile(df: pl.DataFrame, groups: int = 10) -> tuple[float, float, int, float, list[dict[str, float]]]:
    scored = df.sort("model_pd").with_row_index("row_number").with_columns(
        ((pl.col("row_number") * groups / pl.len()).floor().clip(0, groups - 1).cast(pl.Int64) + 1).alias("decile")
    )
    table = scored.group_by("decile").agg(
        [
            pl.len().alias("n"),
            pl.col("actual_outcome").sum().alias("observed"),
            pl.col("model_pd").sum().alias("expected"),
            pl.col("model_pd").mean().alias("avg_pd"),
        ]
    ).sort("decile")

    observed_total = float(table["observed"].sum())
    expected_total = float(table["expected"].sum())
    calibration_variance = abs(observed_total - expected_total) / max(expected_total, 1e-12)

    hl = 0.0
    for row in table.iter_rows(named=True):
        n = float(row["n"])
        expected = float(row["expected"])
        observed = float(row["observed"])
        denominator = max(expected * (1 - expected / max(n, 1.0)), 1e-12)
        hl += (observed - expected) ** 2 / denominator

    hl_df = max(groups - 2, 1)
    hl_pvalue = float(1 - stats.chi2.cdf(hl, df=hl_df))

    return float(calibration_variance), float(hl), hl_df, hl_pvalue, table.to_dicts()


def calculate_psi(df: pl.DataFrame, buckets: int = 10, epsilon: float = PSI_EPSILON) -> float:
    ordered = df.sort("vintage_month")
    midpoint = max(ordered.height // 2, 1)
    expected = ordered.head(midpoint)
    actual = ordered.tail(ordered.height - midpoint)
    if expected.is_empty() or actual.is_empty():
        return 0.0

    min_score = float(ordered["model_pd"].min())
    max_score = float(ordered["model_pd"].max())
    if math.isclose(min_score, max_score):
        return 0.0

    def proportions(frame: pl.DataFrame) -> list[float]:
        values = frame["model_pd"].to_list()
        counts = [0] * buckets
        for value in values:
            bucket = min(int((float(value) - min_score) / (max_score - min_score) * buckets), buckets - 1)
            counts[bucket] += 1
        total = max(len(values), 1)
        return [count / total for count in counts]

    expected_props = proportions(expected)
    actual_props = proportions(actual)
    return float(sum((a - e) * math.log(a / e) for e, a in zip(
        [value + epsilon for value in expected_props],
        [value + epsilon for value in actual_props],
    )))


def calculate_ecl(df: pl.DataFrame, scenario: StressScenario) -> tuple[float, float, float]:
    lgd_expr = pl.col("lgd").cast(pl.Float64) if "lgd" in df.columns else pl.lit(scenario.lgd_default)
    logit_shift = (
        BETA_RATE * scenario.interest_rate_delta
        + BETA_GDP * scenario.gdp_growth_delta
        + BETA_UNEMPLOYMENT * scenario.unemployment_delta
    )
    ecl_frame = df.with_columns(
        [
            lgd_expr.alias("_lgd"),
            pl.col("model_pd").clip(PD_EPSILON, 1 - PD_EPSILON).alias("_pd_base_bounded"),
        ]
    ).with_columns(
        [
            (pl.col("_pd_base_bounded") / (1.0 - pl.col("_pd_base_bounded"))).log().alias("_log_odds_base"),
        ]
    ).with_columns(
        [
            (pl.col("_log_odds_base") + logit_shift).alias("_log_odds_stressed"),
        ]
    ).with_columns(
        [
            (1.0 / (1.0 + (-pl.col("_log_odds_stressed")).exp())).alias("_stressed_pd"),
        ]
    ).with_columns(
        [
            (pl.col("model_pd") * pl.col("_lgd") * pl.col("exposure_at_default")).alias("_base_ecl"),
            (pl.col("_stressed_pd") * pl.col("_lgd") * pl.col("exposure_at_default")).alias("_stressed_ecl"),
        ]
    )
    base = float(ecl_frame["_base_ecl"].sum())
    stressed = float(ecl_frame["_stressed_ecl"].sum())
    return base, stressed, stressed - base


def compute_stressed_pds(df: pl.DataFrame, scenario: StressScenario) -> pl.Series:
    logit_shift = (
        BETA_RATE * scenario.interest_rate_delta
        + BETA_GDP * scenario.gdp_growth_delta
        + BETA_UNEMPLOYMENT * scenario.unemployment_delta
    )
    # Compute using polars expressions
    temp = df.select(
        (
            1.0 / (
                1.0 + (
                    -(
                        (pl.col("model_pd").clip(PD_EPSILON, 1 - PD_EPSILON) / (1.0 - pl.col("model_pd").clip(PD_EPSILON, 1 - PD_EPSILON))).log()
                        + logit_shift
                    )
                ).exp()
            )
        ).alias("stressed_pd")
    )
    return temp["stressed_pd"]


def run_validation(df: pl.DataFrame, scenario: StressScenario | None = None) -> tuple[ValidationMetrics, dict[str, object]]:
    scenario = scenario or StressScenario()
    outcome_exception = None
    if df.height < MIN_OUTCOME_SAMPLE_SIZE:
        ks = None
        auc = None
        gini = None
        outcome_exception = (
            f"Exception: Sample size (N={df.height}) is beneath the {REGULATORY_REF} validation minimum "
            f"of {MIN_OUTCOME_SAMPLE_SIZE}. Results are statistically invalid."
        )
    else:
        ks = calculate_ks(df)
        auc, gini = calculate_auc_gini(df)
    calibration_variance, hosmer_lemeshow, hl_df, hl_pvalue, deciles = calibration_by_decile(df)
    psi = calculate_psi(df)
    base_ecl, stressed_ecl, ecl_delta = calculate_ecl(df, scenario)

    # Recompute metrics under stressed scenario
    stressed_pds = compute_stressed_pds(df, scenario)
    df_stressed = df.with_columns(stressed_pds.alias("model_pd"))

    if df_stressed.height < MIN_OUTCOME_SAMPLE_SIZE:
        stressed_ks = None
        stressed_auc = None
        stressed_gini = None
    else:
        stressed_ks = calculate_ks(df_stressed)
        stressed_auc, stressed_gini = calculate_auc_gini(df_stressed)
    stressed_cal_var, stressed_hl, stressed_hl_df, stressed_hl_pvalue, _ = calibration_by_decile(df_stressed)
    stressed_psi = calculate_psi(df_stressed)

    metrics = ValidationMetrics(
        row_count=df.height,
        ks_statistic=ks,
        auc=auc,
        gini=gini,
        calibration_variance=calibration_variance,
        hosmer_lemeshow=hosmer_lemeshow,
        hosmer_lemeshow_df=hl_df,
        hosmer_lemeshow_pvalue=hl_pvalue,
        psi=psi,
        base_ecl=base_ecl,
        stressed_ecl=stressed_ecl,
        ecl_delta=ecl_delta,
        outcome_metrics_scored=outcome_exception is None,
        outcome_exception=outcome_exception,
        # Stressed metrics
        stressed_ks_statistic=stressed_ks,
        stressed_auc=stressed_auc,
        stressed_gini=stressed_gini,
        stressed_calibration_variance=stressed_cal_var,
        stressed_hosmer_lemeshow=stressed_hl,
        stressed_hosmer_lemeshow_df=stressed_hl_df,
        stressed_hosmer_lemeshow_pvalue=stressed_hl_pvalue,
        stressed_psi=stressed_psi,
    )
    details = {
        "metrics": asdict(metrics),
        "deciles": deciles,
        "scenario": asdict(scenario),
    }
    return metrics, details
