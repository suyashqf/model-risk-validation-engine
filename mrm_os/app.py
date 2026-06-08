from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown import markdown
from starlette.requests import Request

from .analytics import load_portfolio_csv, run_validation
from .config import load_validation_config
from .models import StressScenario
# from .ollama import OllamaClient, OllamaUnavailableError
from .grok import GrokClient, GrokUnavailableError
from .reporting import generate_llm_validation_markdown
from .rules import evaluate_all, highest_alert


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

app = FastAPI(title="MRM OS", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/validate")
async def validate_portfolio(
    portfolio: UploadFile = File(...),
    validation_config: UploadFile | None = File(None),
    interest_rate_delta: float = Form(0.0),
    gdp_growth_delta: float = Form(0.0),
    unemployment_delta: float = Form(0.0),
) -> dict[str, object]:
    if not portfolio.filename or not portfolio.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV portfolio file.")

    try:
        contents = await portfolio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        scenario = StressScenario(
            interest_rate_delta=interest_rate_delta,
            gdp_growth_delta=gdp_growth_delta,
            unemployment_delta=unemployment_delta,
        )
        config_path = PROJECT_DIR / "validation_config.yaml"
        if validation_config and validation_config.filename:
            suffix = Path(validation_config.filename).suffix or ".yaml"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as cfg:
                cfg.write(await validation_config.read())
                config_tmp_path = cfg.name
            config_path = Path(config_tmp_path)

        config = load_validation_config(config_path)
        frame = load_portfolio_csv(tmp_path)
        metrics, details = run_validation(frame, scenario)
        rule_results = evaluate_all(metrics)
        alert = highest_alert(rule_results)
        report, validation_context, llm_sections = generate_llm_validation_markdown(
            metrics=metrics,
            rules=rule_results,
            alert_level=alert,
            scenario=scenario,
            config=config,
            # client=OllamaClient(model=str(config.get("ollama_model", "phi3:latest"))),
            client=GrokClient(model=str(config.get("grok_model", "grok-beta"))),
        )
        return {
            **details,
            "rules": [rule.__dict__ | {"level": rule.level.value} for rule in rule_results],
            "alert_level": alert.value,
            "validation_context": validation_context,
            "llm_sections": llm_sections,
            "report_markdown": report,
        }
    # except OllamaUnavailableError as exc:
    #     raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GrokUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals():
            Path(tmp_path).unlink(missing_ok=True)
        if "config_tmp_path" in locals():
            Path(config_tmp_path).unlink(missing_ok=True)


@app.post("/api/export-pdf")
def export_pdf(markdown_text: str = Form(...)) -> Response:
    html = markdown(markdown_text, extensions=["tables"])
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model Validation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 38px; color: #17202a; line-height: 1.45; }}
    h1, h2 {{ color: #102a43; }}
    table {{ width: 100%; border-collapse: collapse; margin: 18px 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>{html}</body>
</html>"""
    try:
        from weasyprint import HTML

        pdf_bytes = HTML(string=document).write_pdf()
        return Response(
            pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=model-validation-report.pdf"},
        )
    except Exception:
        pass

    return Response(
        document,
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=model-validation-report.html"},
    )
