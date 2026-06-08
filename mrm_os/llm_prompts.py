from __future__ import annotations

from .config import REGULATORY_REF

EXECUTIVE_SUMMARY_PROMPT = f"""You are the validation lead writing the Executive Summary for an {REGULATORY_REF} model validation report. Use the provided JSON context only. Write 3 concise sentences covering tier rating, main quantitative drivers, and required governance action. Do not invent facts.

IMPORTANT: Use the following specific language for the required governance action sentence:
{{approval_language}}"""

ONGOING_MONITORING_PROMPT = f"""You are an independent model validator writing the Ongoing Monitoring Results section under {REGULATORY_REF}. Use the metric meanings, thresholds, and rule actions in the JSON context. Write one technical paragraph interpreting PSI, KS scoring status, calibration variance, and any threshold breaches.

CRITICAL INSTRUCTION — YOU MUST FOLLOW THIS:
The following metrics have breached Red thresholds: {{red_metrics_list}}

You MUST NOT describe any Red-breached metric as acceptable, 
satisfactory, within tolerance, or performing normally.
For each Red metric, you MUST explicitly state it has breached 
its threshold and requires remediation action.
Failure to follow this instruction will cause the report to be 
rejected."""

CONCEPTUAL_SOUNDNESS_TEMPLATE = f"""You are a Senior Model Risk analyst writing the Conceptual Soundness 
section of an {REGULATORY_REF} compliant model validation report.

Write exactly 3-4 sentences covering:
1. Model type and methodology
2. Input features used and why they are theoretically justified
3. Intended use and business purpose
4. Key assumptions and their limitations

Be technical and precise. Do not use bullet points. Write in formal report prose.

Model details:
- Model type: {{model_type}}
- Features: {{features}}
- Target variable: {{target_variable}}
- Intended use: {{intended_use}}
- Development data: {{development_data}}"""

LIMITATIONS_TEMPLATE = f"""You are a Senior Model Risk analyst writing the Overrides and Limitations 
section of an {REGULATORY_REF} compliant validation report.

Write exactly 4 bullet points:
1. Data limitations (source, size, synthetic vs real)
2. Model assumption limitations (what the model assumes that may not hold)
3. Stress test scope limitations (what shocks were and were not tested)
4. Analyst overlay status (whether any manual adjustments were made)

Be specific and technical. Each bullet should be 1-2 sentences.

Context:
- Data: {{development_data}}
- Model type: {{model_type}}
- Shock applied this run: {{shock_description}}
- Manual overlays applied: None
- Observations: {{n_observations}}
- Model update/recomputation: {{stressed_pd_recomputation_limitations}}"""

STRESS_NARRATIVE_TEMPLATE = f"""You are writing the stress test interpretation paragraph for an 
{REGULATORY_REF} model validation report. Write exactly 2 sentences.

Sentence 1: State what happened to ECL and by how much in percentage terms.
Sentence 2: State whether this is directionally plausible given the 
shock applied and basic economic logic. Be specific.

Data:
- Shock applied: {{shock_type}}
- Shocked variable: {{shocked_variable}}
- Base ECL: {{base_ecl}}
- Stressed ECL: {{stressed_ecl}}
- ECL change: {{ecl_delta}} ({{ecl_delta_pct}})"""

BREACH_NARRATIVE_TEMPLATE = f"""You are a Senior Model Risk analyst explaining a Red threshold breach 
in an {REGULATORY_REF} validation report.

Write 2 sentences only:
Sentence 1: Which metric breached, what the value was, and what the 
Red threshold is.
Sentence 2: The most likely technical reason for the breach given 
the data context, and what the remediation action means in practice.

Context:
- Breached metric: {{breached_metric}}
- Observed value: {{breached_value}}
- Red threshold: {{red_threshold}}
- Data type: {{data_type}}
- Model type: {{model_type}}"""
