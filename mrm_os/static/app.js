const form = document.querySelector("#validation-form");
const fileInput = document.querySelector("#portfolio");
const configInput = document.querySelector("#validation-config");
const fileName = document.querySelector("#file-name");
const configName = document.querySelector("#config-name");
const alertPill = document.querySelector("#alert-pill");
const exportButton = document.querySelector("#export-button");
const reportOutput = document.querySelector("#report-output");
const chart = document.querySelector("#decile-chart");
const chartContext = chart.getContext("2d");

const outputs = {
  interest_rate_delta: document.querySelector("#rate-output"),
  gdp_growth_delta: document.querySelector("#gdp-output"),
  unemployment_delta: document.querySelector("#unemployment-output"),
};

for (const [name, output] of Object.entries(outputs)) {
  const input = form.elements[name];
  input.addEventListener("input", () => {
    output.value = `${Number(input.value).toFixed(2)}%`;
  });
}

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name || "No file selected";
});

configInput.addEventListener("change", () => {
  configName.textContent = configInput.files[0]?.name || "Using validation_config.yaml";
});

function formatPct(value) {
  if (value === null || value === undefined) {
    return "N/S";
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatMoney(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(value));
}

function drawDecileChart(deciles) {
  const width = chart.width;
  const height = chart.height;
  chartContext.clearRect(0, 0, width, height);
  chartContext.fillStyle = "transparent";
  chartContext.fillRect(0, 0, width, height);

  const padding = 44;
  const chartHeight = height - padding * 1.5;
  const barWidth = (width - padding * 2) / Math.max(deciles.length, 1);
  const maxValue = Math.max(...deciles.map((row) => Math.max(row.observed, row.expected)), 1);

  chartContext.strokeStyle = "rgba(255, 255, 255, 0.1)";
  chartContext.beginPath();
  chartContext.moveTo(padding, 18);
  chartContext.lineTo(padding, height - padding);
  chartContext.lineTo(width - 16, height - padding);
  chartContext.stroke();

  deciles.forEach((row, index) => {
    const x = padding + index * barWidth + 8;
    const expectedHeight = (row.expected / maxValue) * chartHeight;
    const observedHeight = (row.observed / maxValue) * chartHeight;
    const baseline = height - padding;

    chartContext.fillStyle = "#38bdf8";
    chartContext.fillRect(x, baseline - expectedHeight, barWidth * 0.34, expectedHeight);
    chartContext.fillStyle = "#8b5cf6";
    chartContext.fillRect(x + barWidth * 0.38, baseline - observedHeight, barWidth * 0.34, observedHeight);

    chartContext.fillStyle = "#94a3b8";
    chartContext.font = "12px Inter, Arial";
    chartContext.fillText(String(row.decile), x + barWidth * 0.2, baseline + 18);
  });

  chartContext.fillStyle = "#38bdf8";
  chartContext.fillRect(width - 190, 18, 12, 12);
  chartContext.fillStyle = "#f8fafc";
  chartContext.fillText("Expected", width - 172, 29);
  chartContext.fillStyle = "#8b5cf6";
  chartContext.fillRect(width - 100, 18, 12, 12);
  chartContext.fillStyle = "#f8fafc";
  chartContext.fillText("Actual", width - 82, 29);
}

function renderRules(rules) {
  const container = document.querySelector("#rules-table");
  container.innerHTML = "";
  rules.forEach((rule) => {
    const value = rule.value === null || rule.value === undefined ? "N/S" : Number(rule.value).toFixed(4);
    const row = document.createElement("div");
    row.className = "rule-row";
    row.innerHTML = `
      <span>${rule.metric}</span>
      <strong>${value}</strong>
      <span class="alert ${rule.level}">${rule.level}</span>
      <span>${rule.action}</span>
    `;
    container.appendChild(row);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  document.querySelector("#portfolio-status").textContent = "Running statistical tests";
  exportButton.disabled = true;

  const response = await fetch("/api/validate", {
    method: "POST",
    body: data,
  });

  if (!response.ok) {
    const error = await response.json();
    document.querySelector("#portfolio-status").textContent = error.detail || "Validation failed";
    reportOutput.value = "";
    return;
  }

  const result = await response.json();
  const metrics = result.metrics;
  document.querySelector("#portfolio-status").textContent = `${metrics.row_count.toLocaleString()} observations validated`;
  document.querySelector("#ks-value").textContent = formatPct(metrics.ks_statistic);
  document.querySelector("#gini-value").textContent = formatPct(metrics.gini);
  document.querySelector("#psi-value").textContent = Number(metrics.psi).toFixed(4);
  document.querySelector("#ecl-value").textContent = formatMoney(metrics.ecl_delta);

  alertPill.textContent = result.alert_level;
  alertPill.className = `alert ${result.alert_level}`;
  renderRules(result.rules);
  drawDecileChart(result.deciles);
  reportOutput.value = result.report_markdown;
  exportButton.disabled = false;
});

exportButton.addEventListener("click", async () => {
  const data = new FormData();
  data.append("markdown_text", reportOutput.value);
  const response = await fetch("/api/export-pdf", { method: "POST", body: data });
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "model-validation-report.html";
  link.click();
  URL.revokeObjectURL(link.href);
});
