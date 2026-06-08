from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrm_os.analytics import load_portfolio_csv, run_validation
from mrm_os.config import load_validation_config
from mrm_os.models import StressScenario
from mrm_os.ollama import OllamaClient
from mrm_os.reporting import generate_llm_validation_markdown
from mrm_os.rules import evaluate_all, highest_alert


def main() -> None:
    frame = load_portfolio_csv(str(ROOT / "samples" / "portfolio_sample.csv"))
    config = load_validation_config(ROOT / "validation_config.yaml")
    scenario = StressScenario(
        interest_rate_delta=2.5,
        gdp_growth_delta=-1.0,
        unemployment_delta=1.5,
    )
    metrics, _ = run_validation(frame, scenario)
    rules = evaluate_all(metrics)
    alert = highest_alert(rules)
    report, context, sections = generate_llm_validation_markdown(
        metrics=metrics,
        rules=rules,
        alert_level=alert,
        scenario=scenario,
        config=config,
        client=OllamaClient(model=str(config.get("ollama_model", "phi3:latest"))),
    )
    print(f"LLM report generated with {len(sections)} sections.")
    print(f"Model: {config.get('ollama_model')}")
    print(f"Rows: {context['n_observations']}")
    print(f"Report chars: {len(report)}")


if __name__ == "__main__":
    main()
