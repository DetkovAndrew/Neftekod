#!/usr/bin/env python3
"""
Снимает скриншоты офлайн-дашборда в headless-браузере
(ARCHITECTURE.md §12.2).

Зачем отдельный скрипт: дашборд -- это HTML с расчётами в JavaScript, и
"файл сгенерировался" не означает "страница отрисовалась". Здесь она
реально открывается в Chromium, скрипт ЖДЁТ появления отрисованных
элементов, ПРОВЕРЯЕТ, что на странице нет ошибок JavaScript и что
ключевые блоки непустые, и только потом сохраняет изображение. Если
проверка не прошла -- скрипт падает с ненулевым кодом, а не молча
кладёт картинку пустой страницы.

Playwright нужен только для этой проверки и в рантайме системы не
участвует:
    pip install playwright && playwright install chromium

Запуск:
    python scripts/capture_dashboard.py
    python scripts/capture_dashboard.py --dashboard runs/dashboard.html --out-dir docs/screenshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VIEWPORT = {"width": 1440, "height": 1000}
# Блоки, без которых дашборд бессмысленен: если любой пуст -- это
# провал отрисовки, а не косметика.
REQUIRED_NODES = {
    "#tiles .tile": "плитки сводки",
    "#chart svg": "график показателей",
    "#reasons svg, #reasons .note": "разбор причин отказов",
    "#table tbody tr": "журнал решений",
    "#legend span": "легенда",
}


def capture(dashboard: Path, out_dir: Path) -> list[Path]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover -- зависимость только для проверки
        raise SystemExit(
            "Нужен playwright: pip install playwright && playwright install chromium"
        ) from None

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.on("pageerror", lambda e: errors.append(f"JS-ошибка: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}")
                if m.type == "error" else None)

        page.goto(dashboard.resolve().as_uri())
        page.wait_for_load_state("networkidle")

        for selector, human in REQUIRED_NODES.items():
            try:
                page.wait_for_selector(selector, timeout=15000, state="attached")
            except Exception:
                errors.append(f"не отрисован блок: {human} ({selector})")

        if errors:
            browser.close()
            raise SystemExit("Дашборд не прошёл проверку отрисовки:\n  " + "\n  ".join(errors))

        rows = page.eval_on_selector_all("#table tbody tr", "els => els.length")
        markers = page.eval_on_selector_all("#chart svg [data-i]", "els => els.length")
        print(f"Отрисовано: строк журнала {rows}, маркеров решений {markers}")
        if rows == 0 or markers == 0:
            browser.close()
            raise SystemExit("Дашборд отрисован пустым: нет строк журнала или маркеров решений")

        # (имя, тема, фильтр журнала, снимать всю страницу, к чему прокрутить)
        shots = [
            ("01_обзор_тёмная", "dark", "all", False, None),
            ("02_обзор_светлая", "light", "all", False, None),
            ("03_журнал_отказов", "dark", "refuse", False, "#filters"),
            ("04_страница_целиком", "dark", "all", True, None),
        ]
        for name, theme, flt, full, scroll_to in shots:
            page.evaluate("t => document.documentElement.dataset.theme = t", theme)
            page.evaluate(
                "f => { const b = document.querySelector(`#filters button[data-f='${f}']`); if (b) b.click(); }",
                flt,
            )
            if scroll_to:
                page.eval_on_selector(scroll_to, "el => el.scrollIntoView({block: 'start'})")
            else:
                page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(400)
            path = out_dir / f"{name}.png"
            page.screenshot(path=str(path), full_page=full)
            saved.append(path)
            print(f"  сохранено: {path.relative_to(REPO_ROOT)}")

        browser.close()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", default=str(REPO_ROOT / "runs" / "dashboard.html"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "screenshots"))
    args = parser.parse_args()

    dashboard = Path(args.dashboard)
    if not dashboard.exists():
        raise SystemExit(
            f"{dashboard} не найден. Сначала: python scripts/run_history_scan.py --start ... --end ..."
        )
    saved = capture(dashboard, Path(args.out_dir))
    print(f"\nГотово: {len(saved)} скриншотов проверенного дашборда.")


if __name__ == "__main__":
    sys.exit(main())
