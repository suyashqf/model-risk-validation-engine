from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import mrm_os.app as app_module


def fake_llm_report(metrics, rules, alert_level, scenario, config, client=None):
    return (
        "# Model Validation Report\n\n"
        "## Conceptual Soundness\n\n"
        "The model is supported by structured LLM narrative generation using supplied validation context.",
        {"n_observations": metrics.row_count, "metrics": {"ecl_delta": metrics.ecl_delta}},
        {"conceptual_soundness": "The model is supported by supplied validation context."},
    )


def test_validate_endpoint_accepts_sample_portfolio() -> None:
    app_module.generate_llm_validation_markdown = fake_llm_report
    client = TestClient(app_module.app)
    sample = Path("samples/portfolio_sample.csv")

    with sample.open("rb") as handle:
        response = client.post(
            "/api/validate",
            files={"portfolio": ("portfolio_sample.csv", handle, "text/csv")},
            data={
                "interest_rate_delta": "2.5",
                "gdp_growth_delta": "-1.0",
                "unemployment_delta": "1.5",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    expected_rows = len(sample.read_text().splitlines()) - 1
    assert payload["metrics"]["row_count"] == expected_rows
    assert payload["metrics"]["ecl_delta"] > 0
    assert payload["alert_level"] in {"Green", "Amber", "Red"}
    assert "# Model Validation Report" in payload["report_markdown"]


def test_export_endpoint_returns_report_artifact() -> None:
    client = TestClient(app_module.app)
    response = client.post("/api/export-pdf", data={"markdown_text": "# Report\n\nValidation memo."})

    assert response.status_code == 200
    assert response.headers["content-type"] in {"application/pdf", "text/html; charset=utf-8"}
