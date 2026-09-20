#!/usr/bin/env python3
"""
Офлайн-дашборд прогона по истории (ARCHITECTURE.md §12.2).

Один самодостаточный HTML-файл: данные встроены внутрь, внешних
запросов нет, открывается двойным кликом в закрытом контуре -- это
прямое требование ТЗ (продакшен без интернета).

О визуализации. Каждый показатель -- отдельная мини-панель со своей
шкалой (small multiples). Совмещать серу (мг/кг), T95 (°C) и индекс
тяжести (0..1) на одной оси нельзя: у них разные единицы и разный
масштаб, а две оси Y в одном графике создают произвольное впечатление
о соотношении кривых. В каждой панели одна кривая, поэтому легенда по
цвету не нужна -- панель подписана своим заголовком.

Исход решения кодируется палитрой состояний (good/warning/critical),
и НИКОГДА не только цветом: рядом всегда стоит подпись и форма метки,
иначе карточка нечитаема для оператора с нарушением цветовосприятия и
на чёрно-белой распечатке.
"""

from __future__ import annotations

import json
from datetime import datetime

# Палитра -- эталонный валидированный набор из руководства по визуализации,
# значения не изменены. Состояния (good/warning/critical) намеренно взяты
# из отдельной шкалы статусов, чтобы цвет исхода нельзя было спутать с
# цветом ряда данных.
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "panel": "#ffffff", "text": "#0b0b0b",
        "secondary": "#52514e", "muted": "#898781", "axis": "#c3c2b7",
        "series": "#2a78d6",
        "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "panel": "#222221", "text": "#ffffff",
        "secondary": "#c3c2b7", "muted": "#898781", "axis": "#383835",
        "series": "#3987e5",
        "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b",
    },
}

DECISION_LABEL = {
    "no_action": "без действия",
    "recommend": "рекомендация",
    "refuse": "отказ",
}
DECISION_STATUS = {"no_action": "good", "recommend": "warning", "refuse": "critical"}
# Форма метки дублирует цвет -- требование доступности.
DECISION_GLYPH = {"no_action": "●", "recommend": "▲", "refuse": "■"}
# Уровень уверенности в карточке -- это ConfidenceLevel; в журнале он
# должен читаться словами, а не программным кодом ("refuse" рядом с
# исходом "отказ" выглядит как дубль, хотя означает другое).
CONFIDENCE_LABEL = {"high": "высокая", "medium": "средняя", "low": "низкая", "refuse": "решение не принято"}

# (ключ, заголовок, единица, опорная линия, подпись линии, неотрицателен)
# Индекс тяжести режима ограничен сверху 1.5 (reliability_agent), а
# значение 1.0 -- не "норматив", а порог критического режима; подпись
# опорной линии поэтому задаётся отдельно, а не хардкодится словом
# "норматив" для всех панелей.
METRIC_PANELS = [
    ("sulfur_mg_kg", "Сера в товарном ДТ", "мг/кг", 10.0, "норматив 10", True),
    ("t95_c", "T95 товарного ДТ", "°C", 360.0, "норматив 360", True),
    ("cetane_number", "Цетановое число", "ед.", 51.0, "норматив 51", True),
    ("severity_index", "Индекс тяжести режима", "0..1.5", 1.0, "критический режим", True),
]


def classify_refusal(row: dict) -> str:
    """Категория отказа для разбора: по тексту причины, которую
    сформировал сам Оркестратор."""
    if row.get("decision") != "refuse":
        return ""
    if row.get("transient"):
        return "переходный режим реактора"
    problem = (row.get("problem") or "").lower()
    if "нет снимка кип" in problem:
        return "нет данных КИП на момент"
    if "не увеличивает запас" in problem or "нет расчётной связи" in problem:
        return "нет рычага для нарушенного показателя"
    if "не проходит жёсткие ограничения" in problem:
        return "все варианты нарушают ограничения"
    if "устарел" in problem or "оценку одного или нескольких" in problem:
        return "недостаточно свежих данных"
    if "риска оборудования" in problem:
        return "не оценивается риск оборудования"
    return "прочее"


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    counts = {k: 0 for k in DECISION_LABEL}
    reasons: dict[str, int] = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
        reason = classify_refusal(row)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total": total,
        "counts": counts,
        "pct": {k: (100.0 * v / total if total else 0.0) for k, v in counts.items()},
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def render(rows: list[dict], period: str) -> str:
    for row in rows:
        row["reason_class"] = classify_refusal(row)
    payload = {
        "rows": rows,
        "summary": summarize(rows),
        "period": period,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "palette": PALETTE,
        "decisionLabel": DECISION_LABEL,
        "decisionStatus": DECISION_STATUS,
        "decisionGlyph": DECISION_GLYPH,
        "confidenceLabel": CONFIDENCE_LABEL,
        "panels": [{"key": k, "title": t, "unit": u, "limit": lim, "limitLabel": ll, "nonNegative": nn}
                   for k, t, u, lim, ll, nn in METRIC_PANELS],
    }
    return TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, default=str))


TEMPLATE = r"""<!doctype html>
<html lang="ru" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Нефтекод МАС — прогон по истории</title>
<style>
  :root {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #222221; --text: #ffffff; --secondary: #c3c2b7;
    --muted: #898781; --axis: #383835; --series: #3987e5;
    --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
  }
  :root[data-theme="light"] {
    color-scheme: light;
    --surface: #fcfcfb; --panel: #ffffff; --text: #0b0b0b; --secondary: #52514e;
    --muted: #898781; --axis: #c3c2b7; --series: #2a78d6;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px 16px 48px; background: var(--surface); color: var(--text);
         font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
  .wrap { max-width: 1280px; margin: 0 auto; }
  header { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; justify-content: space-between; }
  h1 { font-size: 20px; font-weight: 600; margin: 0; }
  h2 { font-size: 15px; font-weight: 600; margin: 32px 0 12px; }
  .sub { color: var(--secondary); font-size: 13px; }
  button { font: inherit; color: var(--text); background: var(--panel); border: 1px solid var(--axis);
           border-radius: 6px; padding: 5px 12px; cursor: pointer; }
  button[aria-pressed="true"] { border-color: var(--series); color: var(--series); }
  .tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-top: 20px; }
  .tile { background: var(--panel); border: 1px solid var(--axis); border-radius: 8px; padding: 14px 16px; }
  .tile .label { color: var(--secondary); font-size: 12px; }
  .tile .value { font-size: 26px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .tile .value small { font-size: 14px; font-weight: 400; color: var(--secondary); margin-left: 6px; }
  .panelbox { background: var(--panel); border: 1px solid var(--axis); border-radius: 8px; padding: 12px 8px 4px; }
  .legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0 0; color: var(--secondary); font-size: 13px; }
  .legend b { font-weight: 600; margin-right: 5px; }
  .filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--axis); vertical-align: top; }
  th { color: var(--secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
  td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .tablewrap { max-height: 460px; overflow: auto; border: 1px solid var(--axis); border-radius: 8px; }
  .problem { color: var(--secondary); }
  #tip { position: fixed; z-index: 10; display: none; max-width: 380px; pointer-events: none;
         background: var(--panel); color: var(--text); border: 1px solid var(--axis);
         border-radius: 6px; padding: 9px 11px; font-size: 12px; white-space: pre-wrap;
         box-shadow: 0 6px 20px rgba(0,0,0,.35); }
  svg text { fill: var(--muted); font-size: 11px; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style></head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>Нефтекод МАС — прогон по истории</h1>
    <div class="sub" id="period"></div>
  </div>
  <button id="theme" aria-pressed="false">Светлая тема</button>
</header>

<div class="tiles" id="tiles"></div>

<h2>Решения во времени</h2>
<div class="panelbox"><div id="chart"></div></div>
<div class="legend" id="legend"></div>
<p class="note">У каждого показателя своя шкала: единицы разные, объединять их на одной оси было бы
   неверно. Пунктир — норматив. Наведите курсор на график, чтобы увидеть решение системы в этот момент.</p>

<h2>Из-за чего система отказывалась</h2>
<div class="panelbox"><div id="reasons"></div></div>

<h2>Журнал решений</h2>
<div class="filters" id="filters"></div>
<div class="tablewrap"><table id="table">
  <thead><tr><th>Момент</th><th>Исход</th><th>Уверенность</th><th class="num">Сера</th>
  <th class="num">T95</th><th class="num">Цетан</th><th class="num">Тяжесть</th><th>Проблема / причина</th></tr></thead>
  <tbody></tbody></table></div>
<p class="note" id="tablenote"></p>
</div>
<div id="tip" role="tooltip"></div>

<script>
const D = __PAYLOAD__;
const rows = D.rows.map(r => ({...r, ts: new Date(r.t).getTime()})).sort((a,b) => a.ts - b.ts);
const fmt = (v, n) => (v === undefined || v === null || Number.isNaN(v)) ? "—" : Number(v).toFixed(n);
const statusVar = d => `var(--${D.decisionStatus[d]})`;
let filter = "all";

document.getElementById("period").textContent =
  `Период ${D.period} · ${D.summary.total} точек принятия решения · построено ${D.generated_at}`;

/* ---------- сводка ---------- */
function tiles() {
  const s = D.summary;
  const items = [["Точек проверено", s.total, ""]].concat(
    ["no_action","recommend","refuse"].map(k =>
      [D.decisionLabel[k][0].toUpperCase() + D.decisionLabel[k].slice(1),
       s.counts[k] || 0, s.pct[k].toFixed(0) + "%"]));
  document.getElementById("tiles").innerHTML = items.map(([l, v, sub]) =>
    `<div class="tile"><div class="label">${l}</div>
     <div class="value">${v}${sub ? `<small>${sub}</small>` : ""}</div></div>`).join("");
}

/* ---------- график ---------- */
function chart() {
  const host = document.getElementById("chart");
  const W = Math.max(host.clientWidth || 900, 360);
  const padL = 56, padR = 14, padT = 26, panelH = 86, gap = 26, markerH = 34;
  const H = padT + D.panels.length * (panelH + gap) + markerH;
  const t0 = rows[0].ts, t1 = rows[rows.length - 1].ts;
  const x = t => padL + (W - padL - padR) * (t - t0) / Math.max(1, t1 - t0);

  let svg = `<svg width="${W}" height="${H}" role="img" aria-label="Показатели качества и решения системы во времени">`;
  let overlay = "";   // подписи опорных линий -- поверх кривых

  D.panels.forEach((p, i) => {
    const y0 = padT + i * (panelH + gap), y1 = y0 + panelH;
    const vals = rows.map(r => r.key_state ? r.key_state[p.key] : null)
                     .filter(v => v !== undefined && v !== null);
    if (!vals.length) return;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (p.limit !== null) { lo = Math.min(lo, p.limit); hi = Math.max(hi, p.limit); }
    const pad = (hi - lo) * 0.12 || 1;
    lo -= pad; hi += pad;
    // Показатели, у которых отрицательные значения физически невозможны,
    // не должны получать отрицательную нижнюю границу из-за отступа --
    // это рисовало бы шкалу, которой не бывает.
    if (p.nonNegative) lo = Math.max(lo, 0);
    const y = v => y1 - (y1 - y0) * (v - lo) / Math.max(1e-9, hi - lo);

    svg += `<text x="0" y="${y0 - 9}" style="fill:var(--secondary);font-weight:600">${p.title}, ${p.unit}</text>`;
    svg += `<line x1="${padL}" y1="${y1}" x2="${W - padR}" y2="${y1}" stroke="var(--axis)"/>`;
    svg += `<text x="${padL - 6}" y="${y0 + 8}" text-anchor="end">${hi.toFixed(1)}</text>`;
    svg += `<text x="${padL - 6}" y="${y1 - 2}" text-anchor="end">${lo.toFixed(1)}</text>`;
    if (p.limit !== null && p.limit >= lo && p.limit <= hi) {
      svg += `<line x1="${padL}" y1="${y(p.limit).toFixed(1)}" x2="${W - padR}" y2="${y(p.limit).toFixed(1)}"
               stroke="var(--critical)" stroke-width="1" stroke-dasharray="4 3" opacity="0.85"/>`;
      // Подпись линии кладётся на подложку цвета панели: без неё текст
      // ложится прямо на кривую и становится нечитаемым.
      // Подпись кладётся в НАКЛАДКУ и рисуется после всех кривых: SVG
      // рисует в порядке объявления, и подпись, выведенная здесь же,
      // оказалась бы под линией показателя и стала нечитаемой.
      const ly = y(p.limit), lw = p.limitLabel.length * 6.2 + 10;
      overlay += `<rect x="${(W - padR - lw).toFixed(1)}" y="${(ly - 15).toFixed(1)}" width="${lw.toFixed(1)}"
               height="14" rx="3" fill="var(--panel)" opacity="0.95"/>`;
      overlay += `<text x="${W - padR - 5}" y="${(ly - 4).toFixed(1)}" text-anchor="end"
               style="fill:var(--critical)">${p.limitLabel}</text>`;
    }
    // Разрыв ряда (показатель не оценён) обязан остаться разрывом, а не
    // соединиться прямой через пропуск: иначе график утверждает то, чего
    // в данных не было. Поэтому команда пути зависит от того, была ли
    // валидной ПРЕДЫДУЩАЯ точка. (Ранее здесь проверялся вид уже
    // накопленной строки, и из-за завершающего пробела каждая точка
    // становилась "M" -- линия не рисовалась вообще.)
    let d = "", penDown = false;
    rows.forEach(r => {
      const v = r.key_state ? r.key_state[p.key] : null;
      if (v === undefined || v === null || Number.isNaN(v)) { penDown = false; return; }
      const px = x(r.ts).toFixed(1), py = y(v).toFixed(1);
      d += (penDown ? "L" : "M") + px + "," + py + " ";
      penDown = true;
    });
    svg += `<path d="${d}" fill="none" stroke="var(--series)" stroke-width="2"
             stroke-linejoin="round" stroke-linecap="round"/>`;
  });

  svg += overlay;

  const my = H - 14;
  // Полоса исходов должна читаться и на 200 точках, и на 1300. Кружки с
  // обводкой при высокой плотности сливаются в бледную линию, поэтому
  // при нехватке места маркеры рисуются вплотную стоящими штрихами без
  // обводки: цвет сохраняется, а полоса остаётся сплошной и читаемой.
  const spacing = (W - padL - padR) / Math.max(rows.length - 1, 1);
  const dense = spacing < 6;
  rows.forEach((r, i) => {
    const cx = x(r.ts);
    if (dense) {
      const w = Math.max(spacing * 0.9, 1.2);
      svg += `<rect data-i="${i}" x="${(cx - w / 2).toFixed(2)}" y="${my - 6}" width="${w.toFixed(2)}"
               height="12" fill="${statusVar(r.decision)}"/>`;
    } else {
      svg += `<circle data-i="${i}" cx="${cx.toFixed(1)}" cy="${my}" r="3.5"
               fill="${statusVar(r.decision)}" stroke="var(--panel)" stroke-width="1"/>`;
    }
  });
  svg += `<text x="4" y="${my + 4}" style="fill:var(--secondary)">исход</text>`;
  svg += `<text x="${padL}" y="${H - 1}">${new Date(t0).toLocaleDateString("ru")}</text>`;
  svg += `<text x="${W - padR}" y="${H - 1}" text-anchor="end">${new Date(t1).toLocaleDateString("ru")}</text>`;
  svg += `<line class="cross" x1="0" y1="${padT}" x2="0" y2="${my}" stroke="var(--muted)"
           stroke-dasharray="3 3" opacity="0"/>`;
  svg += "</svg>";
  host.innerHTML = svg;

  const el = host.querySelector("svg"), cross = el.querySelector(".cross"), tip = document.getElementById("tip");
  el.addEventListener("mousemove", e => {
    const box = el.getBoundingClientRect();
    const t = t0 + (t1 - t0) * (e.clientX - box.left - padL) / Math.max(1, W - padL - padR);
    let best = 0, bd = Infinity;
    rows.forEach((r, i) => { const dd = Math.abs(r.ts - t); if (dd < bd) { bd = dd; best = i; } });
    const r = rows[best];
    cross.setAttribute("x1", x(r.ts)); cross.setAttribute("x2", x(r.ts)); cross.setAttribute("opacity", "0.7");
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 400) + "px";
    tip.style.top = (e.clientY + 14) + "px";
    const ks = r.key_state || {};
    tip.textContent = `${r.t}\n${D.decisionGlyph[r.decision]} ${D.decisionLabel[r.decision]} `
      + `(уверенность: ${D.confidenceLabel[r.confidence] || r.confidence})\n`
      + `сера ${fmt(ks.sulfur_mg_kg,2)} мг/кг · T95 ${fmt(ks.t95_c,1)} °C · цетан ${fmt(ks.cetane_number,1)}\n`
      + (r.problem || "");
  });
  el.addEventListener("mouseleave", () => { cross.setAttribute("opacity", "0"); tip.style.display = "none"; });
}

/* ---------- причины отказов ---------- */
function reasons() {
  const host = document.getElementById("reasons");
  const entries = Object.entries(D.summary.reasons);
  if (!entries.length) { host.innerHTML = `<p class="note">Отказов не было.</p>`; return; }
  const W = Math.max(host.clientWidth || 900, 360), barH = 26, padL = 8, padR = 60;
  const labelW = Math.min(300, Math.round(W * 0.34));
  const max = Math.max(...entries.map(e => e[1]));
  const H = entries.length * barH + 8;
  let svg = `<svg width="${W}" height="${H}" role="img" aria-label="Причины отказов">`;
  entries.forEach(([name, n], i) => {
    const y = i * barH + 4, w = (W - labelW - padL - padR) * n / max;
    svg += `<text x="${padL}" y="${y + 16}" style="fill:var(--text)">${name}</text>`;
    svg += `<rect x="${labelW}" y="${y + 4}" width="${Math.max(w, 2).toFixed(1)}" height="14"
             rx="4" fill="var(--critical)" opacity="0.85"/>`;
    svg += `<text x="${labelW + Math.max(w, 2) + 8}" y="${y + 16}" style="fill:var(--secondary)">${n}</text>`;
  });
  host.innerHTML = svg + "</svg>";
}

/* ---------- журнал ---------- */
function filters() {
  const opts = [["all", "все"]].concat(Object.keys(D.decisionLabel).map(k => [k, D.decisionLabel[k]]));
  document.getElementById("filters").innerHTML = opts.map(([k, l]) =>
    `<button data-f="${k}" aria-pressed="${k === filter}">${l}</button>`).join("");
  document.querySelectorAll("#filters button").forEach(b =>
    b.addEventListener("click", () => { filter = b.dataset.f; filters(); table(); }));
}

function table() {
  const body = document.querySelector("#table tbody");
  const shown = rows.filter(r => filter === "all" || r.decision === filter);
  body.innerHTML = shown.map(r => {
    const ks = r.key_state || {};
    return `<tr><td class="num">${r.t}</td>
      <td style="color:${statusVar(r.decision)}">${D.decisionGlyph[r.decision]} ${D.decisionLabel[r.decision]}</td>
      <td>${D.confidenceLabel[r.confidence] || r.confidence}</td>
      <td class="num">${fmt(ks.sulfur_mg_kg,2)}</td><td class="num">${fmt(ks.t95_c,1)}</td>
      <td class="num">${fmt(ks.cetane_number,1)}</td><td class="num">${fmt(ks.severity_index,2)}</td>
      <td class="problem">${r.reason_class ? "<b>" + r.reason_class + "</b> — " : ""}${r.problem || ""}</td></tr>`;
  }).join("");
  document.getElementById("tablenote").textContent =
    `Показано ${shown.length} из ${rows.length}. Таблица дублирует график: исход читается текстом, а не только цветом.`;
}

function legend() {
  document.getElementById("legend").innerHTML =
    Object.keys(D.decisionLabel).map(k =>
      `<span><b style="color:${statusVar(k)}">${D.decisionGlyph[k]}</b>${D.decisionLabel[k]}</span>`).join("")
    + `<span><b style="color:var(--series)">—</b>значение показателя</span>`
    + `<span><b style="color:var(--critical)">- -</b>норматив / порог</span>`;
}

const themeBtn = document.getElementById("theme");
themeBtn.addEventListener("click", () => {
  const light = document.documentElement.dataset.theme !== "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  themeBtn.textContent = light ? "Тёмная тема" : "Светлая тема";
  themeBtn.setAttribute("aria-pressed", String(light));
  chart(); reasons();
});

tiles(); legend(); chart(); reasons(); filters(); table();
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { chart(); reasons(); }, 120);
});
</script>
</body></html>
"""
