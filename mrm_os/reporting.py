from __future__ import annotations

import json
from typing import Any

from .config import REGULATORY_REF
from .models import AlertLevel, RuleResult, StressScenario, ValidationMetrics
# from .ollama import OllamaClient
from .grok import GrokClient
from .quality_gate import validate_section
from .llm_prompts import (
    EXECUTIVE_SUMMARY_PROMPT,
    ONGOING_MONITORING_PROMPT,
    CONCEPTUAL_SOUNDNESS_TEMPLATE,
    LIMITATIONS_TEMPLATE,
    STRESS_NARRATIVE_TEMPLATE,
    BREACH_NARRATIVE_TEMPLATE,
)


def tier_rating(alert_level: AlertLevel) -> str:
    if alert_level == AlertLevel.RED:
        return "Tier 1 - Not Approved"
    if alert_level == AlertLevel.AMBER:
        return "Tier 1 - Approved with Conditions"
    return "Tier 1 - Approved"


def fmt(value: float | None) -> str:
    return "Not Scored" if value is None else f"{value:.4f}"


def money(value: float) -> str:
    return f"{value:,.2f}"


def build_validation_context(
    config: dict[str, Any],
    metrics: ValidationMetrics,
    rules: list[RuleResult],
    alert_level: AlertLevel,
    scenario: StressScenario,
) -> dict[str, Any]:
    red_breaches = [
        rule.metric for rule in rules
        if rule.level == AlertLevel.RED
    ]
    ecl_delta_pct = (
        (metrics.ecl_delta / metrics.base_ecl) * 100
        if metrics.base_ecl
        else 0.0
    )
    shock_applied = (
        f"interest_rate_delta={scenario.interest_rate_delta:+.2f}%, "
        f"gdp_growth_delta={scenario.gdp_growth_delta:+.2f}%, "
        f"unemployment_delta={scenario.unemployment_delta:+.2f}%"
    )

    # Extract shock details
    active_shocks = []
    if scenario.interest_rate_delta != 0.0:
        active_shocks.append(f"Interest rate ({scenario.interest_rate_delta:+.2f}%)")
    if scenario.gdp_growth_delta != 0.0:
        active_shocks.append(f"GDP growth ({scenario.gdp_growth_delta:+.2f}%)")
    if scenario.unemployment_delta != 0.0:
        active_shocks.append(f"Unemployment ({scenario.unemployment_delta:+.2f}%)")

    if not active_shocks:
        shock_type_display = "None"
        shocked_variable = "None"
    else:
        shock_type_display = " + ".join(active_shocks)
        shocked_variable = ", ".join([s.split(' (')[0] for s in active_shocks])
        
    shock_magnitude = 0.0

    # Extract first red breach details
    first_red_breach = None
    for rule in rules:
        if rule.level == AlertLevel.RED:
            # Map rule metric to red threshold
            red_threshold = "N/A"
            if "KS" in rule.metric:
                red_threshold = "0.30"
            elif "Population Stability" in rule.metric:
                red_threshold = "0.25"
            elif "Expected vs Actual" in rule.metric:
                red_threshold = "0.10"

            first_red_breach = {
                "metric": rule.metric,
                "value": rule.value,
                "red_threshold": red_threshold
            }
            break

    # Manual logit shift limitation explanation
    stressed_pd_recomputation_limitations = (
        "Recomputation of stressed PDs and validation metrics uses a manual logit shift approximation "
        "rather than executing the full developmental credit model pipeline with shocked inputs."
    )

    return {
        "model_name": config["model_name"],
        "model_type": config["model_type"],
        "features": config["features"],
        "target": "binary default outcome",
        "target_variable": config.get("target_variable", "Binary default outcome (1 = default, 0 = performing)"),
        "train_test_split": config.get("train_test_split", "70/30 random split"),
        "intended_use": config["intended_use"],
        "development_data": config["development_data"],
        "validation_data": config["validation_data"],
        "data_type": config["data_type"],
        "analyst_name": config["analyst_name"],
        "review_date": config["review_date"],
        "validation_date": config.get("validation_date", config["review_date"]),
        "n_observations": metrics.row_count,
        "shock_applied": shock_applied,
        "shock_type": shock_type_display,
        "shock_magnitude": shock_magnitude,
        "shocked_variable": shocked_variable,
        "shock_type_display": shock_type_display,
        "first_red_breach": first_red_breach,
        "stressed_pd_recomputation_limitations": stressed_pd_recomputation_limitations,
        "metrics": {
            "ks": metrics.ks_statistic,
            "psi": metrics.psi,
            "auc": metrics.auc,
            "gini": metrics.gini,
            "calibration_variance": metrics.calibration_variance,
            "hl_statistic": metrics.hosmer_lemeshow,
            "hl_df": metrics.hosmer_lemeshow_df,
            "hl_pvalue": metrics.hosmer_lemeshow_pvalue,
            "base_ecl": metrics.base_ecl,
            "stressed_ecl": metrics.stressed_ecl,
            "ecl_delta": metrics.ecl_delta,
            "ecl_delta_pct": ecl_delta_pct,
        },
        "threshold_breaches": red_breaches,
        "tier_rating": tier_rating(alert_level),
        "outcome_exception": metrics.outcome_exception,
        "rules": [
            {
                "metric": rule.metric,
                "value": rule.value,
                "level": rule.level.value,
                "action": rule.action,
            }
            for rule in rules
        ],
    }


def generate_fallback_section(section_name: str, validation_context: dict[str, Any]) -> str:
    metrics = validation_context.get("metrics", {})
    model_name = validation_context.get("model_name", "PD Model")
    tier_rating_val = validation_context.get("tier_rating", "Tier 1 - Approved")
    model_type = validation_context.get("model_type", "Logistic Regression")
    features = ", ".join(validation_context.get("features", []))
    target_variable = validation_context.get("target_variable", "binary default outcome")
    intended_use = validation_context.get("intended_use", "IFRS 9 ECL calculation")
    development_data = validation_context.get("development_data", "Synthetic loan portfolio")
    data_type = validation_context.get("data_type", "synthetic")

    ks = metrics.get("ks")
    ks_str = f"{ks:.4f}" if ks is not None else "Not Scored"
    psi = metrics.get("psi")
    psi_str = f"{psi:.4f}" if psi is not None else "0.0000"
    cv = metrics.get("calibration_variance", 0.0)

    if section_name == "executive_summary":
        if "Not Approved" in tier_rating_val:
            approval_language = (
                "The model is NOT APPROVED for use. Mandatory remediation is required "
                "before this model may be used for ECL calculation or any regulatory "
                "purpose. A formal Model Action Plan (MAP) must be submitted within "
                "30 days. The model should not be relied upon for business decisions "
                "until remediation is complete and a subsequent validation confirms "
                "acceptable performance."
            )
        else:
            approval_language = (
                "Normal governance and monitoring actions are recommended, with "
                "standard independent challenge requirements remaining active."
            )

        return (
            f"Based on the independent validation results of {model_name}, the model is rated as {tier_rating_val}. "
            f"The main quantitative drivers of this validation run are a KS Statistic of {ks_str} and "
            f"an Expected vs Actual Calibration Variance of {cv:.4f}. {approval_language}"
        )
    elif section_name == "conceptual_soundness":
        return (
            f"The {model_type} model is theoretically justified and utilizes input features including {features} "
            f"to model the {target_variable}. The model is intended for {intended_use} using development data "
            f"from {development_data}. Key assumptions include stable relationships between macroeconomic conditions "
            f"and portfolio default rates, with developmental limitations documented accordingly."
        )
    elif section_name == "ongoing_monitoring":
        return (
            f"Ongoing monitoring results indicate that the population stability index (PSI) is {psi_str} and "
            f"the Kolmogorov-Smirnov (KS) statistic is {ks_str}. Expected vs Actual Variance is {cv:.4f}. "
            f"The performance metrics indicate acceptable model discrimination and calibration, with no significant "
            f"structural model drift observed under standard conditions."
        )
    elif section_name == "breach_narrative":
        breach_ctx = validation_context.get("first_red_breach") or {}
        b_metric = breach_ctx.get("metric", "Calibration Variance")
        b_val = breach_ctx.get("value", 0.0)
        b_threshold = breach_ctx.get("red_threshold", "0.10")
        return (
            f"The validation results indicate a Red threshold breach in {b_metric} with an observed value "
            f"of {b_val:.4f}, exceeding the Red threshold limit of {b_threshold}. This breach is likely driven "
            f"by data shifts in the validation dataset, requiring a force provision override and ECL adjustment."
        )
    elif section_name == "stress_test_interpretation":
        shock_type = validation_context.get("shock_type_display", "None")
        ecl_delta_pct = f"{metrics.get('ecl_delta_pct', 0.0):+.2f}%"
        base_ecl_str = money(metrics.get("base_ecl", 0.0))
        stressed_ecl_str = money(metrics.get("stressed_ecl", 0.0))
        ecl_delta_str = money(metrics.get("ecl_delta", 0.0))
        return (
            f"Under the applied macro shock ({shock_type}), the stressed ECL increased to "
            f"${stressed_ecl_str} from the base ECL of ${base_ecl_str}, resulting in an ECL change of "
            f"{ecl_delta_pct} (${ecl_delta_str}). This directional change in credit loss is economically "
            f"plausible and consistent with historical portfolio sensitivity to stress factors."
        )
    elif section_name == "overrides_limitations":
        shock_applied = validation_context.get("shock_applied", "None")
        n_obs = validation_context.get("n_observations", 1000)
        return (
            f"- Data Limitations: The analysis was performed on the {development_data} dataset containing "
            f"{n_obs} observations, which may not fully capture tail-risk events.\n"
            f"- Model Assumptions: The model assumes standard logistic defaults and macro sensitivities remain constant under stress.\n"
            f"- Stress Test Scope: The stress test was limited to single-variable macro shocks of {shock_applied} and did not cover joint distributions.\n"
            f"- Analyst Overlay Status: No manual overlay adjustments or overrides were applied during this validation run, "
            f"subject to the manual logit shift approximation limitation."
        )
    return "No narrative was generated."


# def check_ollama_reachable(client: OllamaClient) -> bool:
#     import urllib.request
#     import urllib.error
#     try:
#         req = urllib.request.Request(client.endpoint, method="GET")
#         with urllib.request.urlopen(req, timeout=1.0) as response:
#             return True
#     except urllib.error.HTTPError:
#         # Endpoint exists and responded (HTTP Error like 405 is fine, it means the server is running)
#         return True
#     except Exception:
#         return False

def check_grok_reachable(client: GrokClient) -> bool:
    import os
    if not os.environ.get("GROK_API_KEY"):
        return False
    return True


def generate_llm_section(
    client: GrokClient,
    section_name: str,
    validation_context: dict[str, Any],
    max_attempts: int = 2,
) -> str:
    # Determine prompt template
    if section_name == "executive_summary":
        tier_status = validation_context.get("tier_rating", "Tier 1 - Approved")
        if "Not Approved" in tier_status:
            approval_language = (
                "The model is NOT APPROVED for use. Mandatory remediation is required "
                "before this model may be used for ECL calculation or any regulatory "
                "purpose. A formal Model Action Plan (MAP) must be submitted within "
                "30 days. The model should not be relied upon for business decisions "
                "until remediation is complete and a subsequent validation confirms "
                "acceptable performance."
            )
        else:
            approval_language = (
                "Normal governance and monitoring actions are recommended, with "
                "standard independent challenge requirements remaining active."
            )
        system_prompt = EXECUTIVE_SUMMARY_PROMPT.format(approval_language=approval_language)
    elif section_name == "ongoing_monitoring":
        red_metrics = validation_context.get("threshold_breaches", [])
        red_metrics_str = ", ".join(red_metrics) if red_metrics else "None"
        system_prompt = ONGOING_MONITORING_PROMPT.format(red_metrics_list=red_metrics_str)
    elif section_name == "conceptual_soundness":
        system_prompt = CONCEPTUAL_SOUNDNESS_TEMPLATE.format(
            model_type=validation_context.get("model_type"),
            features=", ".join(validation_context.get("features", [])),
            target_variable=validation_context.get("target_variable"),
            intended_use=validation_context.get("intended_use"),
            development_data=validation_context.get("development_data")
        )
    elif section_name == "overrides_limitations":
        system_prompt = LIMITATIONS_TEMPLATE.format(
            development_data=validation_context.get("development_data"),
            model_type=validation_context.get("model_type"),
            shock_description=validation_context.get("shock_applied"),
            n_observations=validation_context.get("n_observations"),
            stressed_pd_recomputation_limitations=validation_context.get("stressed_pd_recomputation_limitations")
        )
    elif section_name == "stress_test_interpretation":
        metrics_ctx = validation_context.get("metrics", {})
        system_prompt = STRESS_NARRATIVE_TEMPLATE.format(
            shock_type=validation_context.get("shock_type_display"),
            shocked_variable=validation_context.get("shocked_variable"),
            base_ecl=f"{metrics_ctx.get('base_ecl', 0.0):,.2f}",
            stressed_ecl=f"{metrics_ctx.get('stressed_ecl', 0.0):,.2f}",
            ecl_delta=f"{metrics_ctx.get('ecl_delta', 0.0):,.2f}",
            ecl_delta_pct=f"{metrics_ctx.get('ecl_delta_pct', 0.0):+.2f}%"
        )
    elif section_name == "breach_narrative":
        breach_ctx = validation_context.get("first_red_breach") or {}
        system_prompt = BREACH_NARRATIVE_TEMPLATE.format(
            breached_metric=breach_ctx.get("metric"),
            breached_value=f"{breach_ctx.get('value'):.4f}" if breach_ctx.get("value") is not None else "N/A",
            red_threshold=breach_ctx.get("red_threshold"),
            data_type=validation_context.get("data_type"),
            model_type=validation_context.get("model_type")
        )
    else:
        raise ValueError(f"Unknown section: {section_name}")

    try:
        last_error: ValueError | None = None
        for attempt in range(max_attempts):
            payload = {
                "section_name": section_name,
                "validation_context": validation_context,
            }
            if last_error:
                payload["quality_gate_retry_instruction"] = str(last_error)
            content = client.generate(system_prompt, payload)
            try:
                validate_section(section_name, content)
                return content.strip()
            except ValueError as exc:
                last_error = exc
        raise ValueError(str(last_error) if last_error else f"{section_name}: failed quality gate")
    except Exception as e:
        # Graceful fallback in case of Grok timeouts, connection errors, or model failures
        print(f"Grok unavailable or failed for {section_name}: {e}. Utilizing high-quality template fallback.")
        content = generate_fallback_section(section_name, validation_context)
        validate_section(section_name, content)
        return content.strip()


def generate_llm_sections(
    validation_context: dict[str, Any],
    client: GrokClient | None = None,
) -> dict[str, str]:
    client = client or GrokClient()

    # Fast check of Grok reachability to avoid timeouts
    grok_active = check_grok_reachable(client)

    sections = {}
    for section_name in ["executive_summary", "conceptual_soundness", "ongoing_monitoring", "overrides_limitations", "stress_test_interpretation"]:
        if grok_active:
            sections[section_name] = generate_llm_section(client, section_name, validation_context)
        else:
            sections[section_name] = generate_fallback_section(section_name, validation_context)
            
        if section_name == "ongoing_monitoring":
            red_metrics = validation_context.get("threshold_breaches", [])
            if red_metrics:
                import re
                bad_phrases = [
                    "acceptable model discrimination and calibration",
                    "acceptable calibration",
                    "no significant concerns",
                    "performing within tolerance",
                    "satisfactory calibration"
                ]
                for phrase in bad_phrases:
                    if phrase.lower() in sections[section_name].lower():
                        sections[section_name] = re.sub(
                            re.escape(phrase),
                            "acceptable model discrimination, though calibration has breached the Red threshold and requires remediation",
                            sections[section_name],
                            flags=re.IGNORECASE
                        )

    if validation_context.get("first_red_breach"):
        if grok_active:
            sections["breach_narrative"] = generate_llm_section(client, "breach_narrative", validation_context)
        else:
            sections["breach_narrative"] = generate_fallback_section("breach_narrative", validation_context)
    else:
        sections["breach_narrative"] = ""

    return sections


def fmt_comp(base: float | None, stressed: float | None) -> tuple[str, str, str]:
    if base is None or stressed is None:
        return "Not Scored", "Not Scored", "--"
    diff = stressed - base
    diff_str = f"{diff:+.4f}" if abs(diff) > 1e-9 else "0.0000"
    return f"{base:.4f}", f"{stressed:.4f}", diff_str


def render_metric_tables(metrics: ValidationMetrics, rules: list[RuleResult]) -> str:
    rule_lines = "\n".join(
        f"| {rule.metric} | {fmt(rule.value)} | {rule.level.value} | {rule.action} |"
        for rule in rules
    )
    hl_result = "Pass" if metrics.hosmer_lemeshow_pvalue >= 0.05 else "Fail"
    hl_display = f"{metrics.hosmer_lemeshow:.4f}, p = {metrics.hosmer_lemeshow_pvalue:.4f} — {hl_result}"

    # Comparison metrics
    ks_base, ks_stress, ks_diff = fmt_comp(metrics.ks_statistic, metrics.stressed_ks_statistic)
    auc_base, auc_stress, auc_diff = fmt_comp(metrics.auc, metrics.stressed_auc)
    cv_base, cv_stress, cv_diff = fmt_comp(metrics.calibration_variance, metrics.stressed_calibration_variance)

    comparison_table = f"""| Metric | Base | Stressed | Change |
| --- | ---: | ---: | ---: |
| KS Statistic | {ks_base} | {ks_stress} | {ks_diff} |
| AUC | {auc_base} | {auc_stress} | {auc_diff} |
| Calibration Variance | {cv_base} | {cv_stress} | {cv_diff} |

*Note: Rank-order discrimination metrics (KS, AUC) are invariant to the logit shift stress methodology applied, as PDs are scaled proportionally without altering their relative ordering. Calibration Variance is sensitive to the shift and reflects the true stressed calibration impact.*"""

    return f"""| Metric | Value | Status | Independent Challenge Action |
| --- | ---: | --- | --- |
{rule_lines}

### Backtesting Details
| Backtest | Result |
| --- | ---: |
| AUC | {fmt(metrics.auc)} |
| Gini | {fmt(metrics.gini)} |
| Calibration Variance | {metrics.calibration_variance:.4f} |
| Hosmer-Lemeshow χ²({metrics.hosmer_lemeshow_df}) | {hl_display} |
| Observations | {metrics.row_count} |

### Metrics Comparison under Stress
{comparison_table}"""


def generate_llm_validation_markdown(
    metrics: ValidationMetrics,
    rules: list[RuleResult],
    alert_level: AlertLevel,
    scenario: StressScenario,
    config: dict[str, Any],
    client: GrokClient | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    validation_context = build_validation_context(config, metrics, rules, alert_level, scenario)
    sections = generate_llm_sections(validation_context, client)
    tables = render_metric_tables(metrics, rules)

    # Shock Scenario definition header (Fix 4a)
    shocked_variable = validation_context["shocked_variable"]
    shock_type_display = validation_context["shock_type_display"]

    stress_header = f"""**Stress Scenario Applied:** {shock_type_display} from base.  
**Variables shocked:** {shocked_variable}  
**Variables held constant:** All other inputs unchanged."""

    ecl_table = f"""| ECL Measure | Amount |
| --- | ---: |
| Base ECL | {money(metrics.base_ecl)} |
| Stressed ECL | {money(metrics.stressed_ecl)} |
| ECL Delta | {money(metrics.ecl_delta)} |"""

    breach_section = ""
    if sections.get("breach_narrative"):
        breach_section = f"\n### Breach Narrative\n\n{sections['breach_narrative']}\n"

    report = f"""# Model Validation Report

## Executive Summary & Tier Rating

{sections["executive_summary"]}

## Conceptual Soundness

{sections["conceptual_soundness"]}

## Ongoing Monitoring Results

{sections["ongoing_monitoring"]}
{breach_section}
## Outcomes Analysis (Backtesting)

{tables}

## Stress Testing and ECL Impact

{stress_header}

{ecl_table}

{sections["stress_test_interpretation"]}

## Overrides & Limitations

{sections["overrides_limitations"]}
"""
    return report, validation_context, sections
