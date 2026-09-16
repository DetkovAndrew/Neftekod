#!/usr/bin/env python3
"""
Прогоняет Оркестратор по диапазону истории с заданным шагом и собирает
результаты в самодостаточный офлайн-дашборд (ARCHITECTURE.md, "приветствуется:
интерактивная визуализация"). Никакого обучения -- просто много повторных
вызовов уже готового детерминированного цикла принятия решения.

Дашборд -- один HTML-файл с данными, встроенными прямо в него (никаких
fetch()/CORS, никаких внешних CDN -- открывается двойным кликом в
браузере без доступа к интернету, что прямо соответствует требованию
Q&A-сессии "решение должно работать локально без доступа к Интернету").

Результаты сканирования (JSON/HTML) НЕ коммитятся в git -- это
производные от реальных данных хакатона (см. .gitignore, runs/).

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/run_history_scan.py --start 2023-01-01 --end 2023-04-01 --step-hours 12
    # результат: runs/dashboard.html -- открыть в браузере
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_demo_cycle import build_orchestrator  # noqa: E402
from neftekod_mas.data.loaders import data_dir, load_kip, load_lims, load_pak  # noqa: E402


def scan(start: datetime, end: datetime, step: timedelta) -> list[dict]:
    orchestrator = build_orchestrator()
    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    pak = load_pak(data_dir() / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx")

    rows: list[dict] = []
    ts = start
    while ts <= end:
        rec = orchestrator.run_cycle(ts, avt, ht, lims, pak)
        decision = "refuse" if rec.is_refusal else ("recommend" if rec.proposed_actions else "no_action")
        rows.append({
            "t": ts.isoformat(),
            "decision": decision,
            "confidence": rec.confidence.value,
            "problem": rec.problem_or_risk,
            "key_state": rec.key_state,
            "n_warnings": len(rec.confidence_warnings),
        })
        ts += step
    return rows


DASHBOARD_TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Нефтекод МАС -- прогон истории</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 24px; background: #0b0f14; color: #e6edf3; }
  h1 { font-size: 18px; font-weight: 600; }
  .legend span { display: inline-block; margin-right: 16px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  svg { background: #131820; border-radius: 8px; }
  .tick { font-size: 10px; fill: #8b949e; }
  #tooltip { position: fixed; background: #1c2530; border: 1px solid #30363d; padding: 8px 10px;
             border-radius: 6px; font-size: 12px; pointer-events: none; display: none; max-width: 360px; white-space: pre-wrap; }
</style></head>
<body>
<h1>Нефтекод МАС -- прогон истории (офлайн, без внешних зависимостей)</h1>
<div class="legend">
  <span><i class="dot" style="background:#3fb950"></i>нет действия</span>
  <span><i class="dot" style="background:#d29922"></i>рекомендация</span>
  <span><i class="dot" style="background:#f85149"></i>отказ</span>
</div>
<div id="chart"></div>
<div id="tooltip"></div>
<script>
const DATA = __DATA_JSON__;

function build() {
  const W = Math.min(window.innerWidth - 60, 1400), H = 420, padL = 60, padR = 20, padT = 20, padB = 40;
  const times = DATA.map(d => new Date(d.t).getTime());
  const t0 = Math.min(...times), t1 = Math.max(...times);
  const x = t => padL + (W - padL - padR) * (t - t0) / Math.max(1, (t1 - t0));

  const metrics = ["sulfur_mg_kg", "t95_c", "cetane_number", "severity_index"];
  const colors = {sulfur_mg_kg: "#58a6ff", t95_c: "#bc8cff", cetane_number: "#39c5cf", severity_index: "#f0883e"};
  const rowH = (H - padT - padB) / metrics.length;

  let svg = `<svg width="${W}" height="${H}">`;

  metrics.forEach((m, i) => {
    const vals = DATA.map(d => d.key_state[m]).filter(v => v !== undefined && v !== null);
    if (vals.length === 0) return;
    const vmin = Math.min(...vals), vmax = Math.max(...vals);
    const y0 = padT + i * rowH, y1 = padT + (i + 1) * rowH - 8;
    const yScale = v => y1 - (y1 - y0) * (v - vmin) / Math.max(1e-9, (vmax - vmin));

    svg += `<text x="${padL}" y="${y0 + 10}" class="tick" fill="${colors[m]}">${m} [${vmin.toFixed(2)}, ${vmax.toFixed(2)}]</text>`;

    let path = "";
    DATA.forEach((d, idx) => {
      const v = d.key_state[m];
      if (v === undefined || v === null) return;
      const px = x(times[idx]), py = yScale(v);
      path += (path ? "L" : "M") + px.toFixed(1) + "," + py.toFixed(1) + " ";
    });
    svg += `<path d="${path}" fill="none" stroke="${colors[m]}" stroke-width="1.5" opacity="0.85"/>`;
  });

  // маркеры решений внизу
  const decisionColor = {no_action: "#3fb950", recommend: "#d29922", refuse: "#f85149"};
  const markerY = H - 14;
  DATA.forEach((d, idx) => {
    svg += `<circle data-idx="${idx}" cx="${x(times[idx]).toFixed(1)}" cy="${markerY}" r="4" fill="${decisionColor[d.decision]}" />`;
  });

  svg += `<line x1="${padL}" y1="${markerY + 10}" x2="${W - padR}" y2="${markerY + 10}" stroke="#30363d"/>`;
  svg += "</svg>";

  const chart = document.getElementById("chart");
  chart.innerHTML = svg;

  const tooltip = document.getElementById("tooltip");
  chart.querySelectorAll("circle").forEach(c => {
    c.addEventListener("mousemove", e => {
      const d = DATA[+c.dataset.idx];
      tooltip.style.display = "block";
      tooltip.style.left = (e.clientX + 12) + "px";
      tooltip.style.top = (e.clientY + 12) + "px";
      tooltip.textContent = d.t + "\\n" + d.decision + " (" + d.confidence + ")\\n" + d.problem;
    });
    c.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
  });
}
build();
window.addEventListener("resize", build);
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-hours", type=float, default=12.0)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "runs"))
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    step = timedelta(hours=args.step_hours)

    rows = scan(start, end, step)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "scan.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    html = DASHBOARD_TEMPLATE.replace("__DATA_JSON__", json.dumps(rows, ensure_ascii=False))
    html_path = out_dir / "dashboard.html"
    html_path.write_text(html, encoding="utf-8")

    n_no_action = sum(1 for r in rows if r["decision"] == "no_action")
    n_recommend = sum(1 for r in rows if r["decision"] == "recommend")
    n_refuse = sum(1 for r in rows if r["decision"] == "refuse")
    print(f"Прогнано {len(rows)} точек: no_action={n_no_action}, recommend={n_recommend}, refuse={n_refuse}")
    print(f"Дашборд -> {html_path} (открыть в браузере)")


if __name__ == "__main__":
    main()
