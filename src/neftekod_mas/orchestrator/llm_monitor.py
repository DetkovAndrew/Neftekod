"""
LLM Monitor (ARCHITECTURE.md §9) -- лёгкая локальная LLM пишет
человекочитаемый комментарий к УЖЕ ГОТОВОЙ и УЖЕ ПРОВЕРЕННОЙ Guard'ом
карточке рекомендации. Роль -- "Monitor" в терминах AAE (Schall 2026):
вне критического пути, ничего не решает и ничего не проверяет.

Гарантии:
1. Вызывается после формирования Recommendation; ни одно поле, кроме
   llm_commentary/llm_status, не меняется. Любая ошибка/таймаут -> карточка
   уходит оператору без комментария (шаблонное объяснение explain.py
   остаётся основным).
2. Grounding-проверка: каждое число и каждый технологический тег в ответе
   модели обязаны присутствовать в фактах карточки (с точностью до
   округления, которое использовала модель). Иначе комментарий
   отбрасывается целиком (llm_status="rejected: ...") -- частично
   правдивый текст оператору не показывается.
3. Модель получает только факты карточки (без сырых рядов КИП/ЛИМС).

Сервер -- любой OpenAI-совместимый /v1/chat/completions (Ollama, vLLM,
llama.cpp server). Зависимостей нет: urllib из стандартной библиотеки.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from neftekod_mas.schemas import Recommendation

SYSTEM_PROMPT = (
    "Ты -- помощник оператора установки гидроочистки дизельного топлива. "
    "Тебе дают карточку рекомендации, уже рассчитанную и проверенную системой. "
    "Перескажи её оператору простым русским языком в 3-5 предложениях: что происходит, "
    "что предлагается и почему, на что обратить внимание. Правила: "
    "используй ТОЛЬКО числа и обозначения тегов из карточки, не придумывай новых; "
    "не предлагай других действий, кроме указанных в карточке; "
    "если карточка -- отказ, объясни причину отказа и не советуй изменений режима; "
    "не утверждай ничего, чего нет в карточке; не переписывай карточку построчно; "
    "норматив -- это предел спецификации, а порог действия и порог наблюдения -- "
    "внутренние пороги системы, которые срабатывают раньше норматива: не называй их пределом; "
    "не расшифровывай теги и коды сверх того, что написано в карточке; "
    "не используй markdown и списки."
)

# Человекочитаемые названия вместо программных ключей -- модель пересказывает
# их, а не копирует "sulfur_mg_kg=9.97".
METRIC_LABELS = {
    "sulfur_mg_kg": "сера, мг/кг",
    "t95_c": "температура выкипания 95% (T95), °C",
    "cetane_number": "цетановое число",
    "cfpp_c": "предельная температура фильтруемости (CFPP), °C",
    "density_kg_m3": "плотность при 15 °C, кг/м3",
    "severity_index": "индекс тяжести режима оборудования (0..1)",
    "equipment_risk_severity": "индекс тяжести режима оборудования (0..1)",
    "energy_cost_proxy": "величина изменения уставки (прокси затрат)",
}
MIN_COMMENTARY_CHARS = 80
# Коды флагов Data & Sync по-русски -- иначе модель переводит их сама
# ("stuck_sensor" -> "стучавшиеся датчики").
FLAG_WORDS = {
    "missing_kip_snapshot": "нет снимка КИП на момент решения",
    "missing_kip_tag": "нет текущего значения тега",
    "out_of_range": "значение вне правдоподобного диапазона",
    "stuck_sensor": "датчик выдаёт одно и то же значение (возможно, завис)",
    "stale_lims": "устаревший анализ ЛИМС",
    "stale_pak": "устаревшее значение ПАК",
    "stale_other": "прочие устаревшие лабораторные точки",
    "transient_regime": "переходный режим реактора",
}

_NUMBER_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,]\d+)?")
# Технологические теги: qualified ("242000:T5", "avt:F30") или голые ("T11", "Q21", "P13").
_TAG_RE = re.compile(r"\b(?:(?:avt|242000):)?[TPFLQD]\d{1,3}\b")
_QUALIFIED_TAG_RE = re.compile(r"\b(?:avt|242000):[A-Z]+\d{1,3}\b")
# Не числа-факты: номера шагов, "2 ч", ГОСТ-номера и т.п. маленькие целые допустимы.
_FREE_SMALL_INTS = {0, 1, 2, 3}
# Обозначения показателей качества, совпадающие по форме с тегами КИП.
_QUALITY_PARAM_NAMES = {"T95", "T50", "T10", "D15"}


class ChatClient(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


@dataclass
class OpenAICompatClient:
    base_url: str  # напр. http://localhost:11434/v1
    model: str  # напр. qwen2.5:3b
    timeout_s: float = 120.0  # первый запрос включает загрузку модели в GPU
    temperature: float = 0.2
    max_tokens: int = 400

    def __call__(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 -- URL задаёт оператор
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"].strip()


def card_facts(rec: Recommendation, tag_descriptions: dict[str, str] | None = None) -> str:
    """Факты карточки в компактном текстовом виде -- ровно то, что видит
    модель, и ровно то, против чего проверяется её ответ. tag_descriptions
    ("avt:D10" -> "Плотность нефти на подаче", справочник КИП) -- смысл
    упомянутых тегов, чтобы модель не угадывала его по букве имени (ТЗ п.2)."""
    lines = [f"Момент решения: {rec.decision_at:%Y-%m-%d %H:%M}"]
    lines.append("Тип: ОТКАЗ от рекомендации" if rec.is_refusal else "Тип: рекомендация")
    if rec.key_state:
        lines.append("Текущее состояние: " + "; ".join(f"{_label(k)} = {_fmt(v)}" for k, v in rec.key_state.items()))
    lines.append(f"Проблема/риск: {_humanize(rec.problem_or_risk)}")
    for a in rec.proposed_actions:
        lines.append(f"Действие: {a.variable_name} [{a.tag}] {_fmt(a.current_value)} -> {_fmt(a.recommended_value)} {a.unit}")
    if rec.expected_effect:
        lines.append("Ожидаемый эффект: " + "; ".join(_effect(k, v, rec.key_state) for k, v in rec.expected_effect.items()))
    for c in rec.constraints_checked:
        lines.append(f"Проверка {c.check_name}: {c.verdict.value} ({c.detail})")
    lines.append(f"Уверенность: {rec.confidence.value}")
    for w in rec.confidence_warnings[:8]:
        lines.append(f"Предупреждение: {_humanize_warning(w, tag_descriptions or {})}")
    lines.append(f"Объяснение системы: {_humanize(rec.explanation)}")
    return "\n".join(lines)


def _effect(key: str, after: float, before: dict[str, float]) -> str:
    base_key = "severity_index" if key == "equipment_risk_severity" else key
    if base_key in before:
        b = before[base_key]
        change = "без изменений" if abs(after - b) < 1e-3 * max(abs(b), 1.0) else f"{_fmt(b)} -> {_fmt(after)}"
        return f"{_label(key)}: {change}"
    return f"{_label(key)} = {_fmt(after)}"


def _humanize_warning(w: str, tag_descriptions: dict[str, str]) -> str:
    code, sep, rest = w.partition(": ")
    if sep and code in FLAG_WORDS:
        w = f"{FLAG_WORDS[code]}: {rest}"
    for tag in dict.fromkeys(_QUALIFIED_TAG_RE.findall(w)):
        if tag in tag_descriptions:
            w = w.replace(tag, f"{tag} ({tag_descriptions[tag]})", 1)
    return w


def _label(key: str) -> str:
    return METRIC_LABELS.get(key, key)


def _humanize(text: str) -> str:
    for key, label in METRIC_LABELS.items():
        text = text.replace(key, label.split(",")[0])
    return text


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def _to_float(token: str) -> float:
    return float(token.replace("−", "-").replace(",", "."))


def _decimals(token: str) -> int:
    t = token.replace(",", ".")
    return len(t.split(".", 1)[1]) if "." in t else 0


def ungrounded_items(text: str, facts: str) -> list[str]:
    """Числа/теги из text, которых нет в facts. Число считается подтверждённым,
    если оно совпадает с каким-либо числом карточки после округления до того
    числа знаков, которое использовала модель (9.8 для 9.7834 -- допустимо)."""
    fact_numbers = [_to_float(t) for t in _NUMBER_RE.findall(facts)]
    bad: list[str] = []
    for token in _NUMBER_RE.findall(text):
        x = _to_float(token)
        if x.is_integer() and abs(x) in _FREE_SMALL_INTS:
            continue
        tol = 0.5 * 10 ** (-_decimals(token)) + 1e-9
        # Положительное число может быть модулем отрицательного факта (знак передан
        # словами: "превышение на 9.37"), но отрицательное число обязано совпасть со
        # знаком: "-19.37" при факте 19.37 -- искажение, а не округление.
        if x < 0:
            ok = any(abs(x - y) <= tol for y in fact_numbers)
        else:
            ok = any(abs(x - abs(y)) <= tol for y in fact_numbers)
        if not ok:
            bad.append(token)

    fact_tags = {t.split(":")[-1] for t in _TAG_RE.findall(facts)}
    for tag in _TAG_RE.findall(text):
        bare = tag.split(":")[-1]
        if bare not in fact_tags and bare not in _QUALITY_PARAM_NAMES:
            bad.append(tag)
    return bad


# Qwen при длинных карточках иногда переключается на китайский посреди
# ответа -- такие куски не содержат чисел и grounding их не ловит.
_FOREIGN_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]")


class LLMMonitor:
    def __init__(
        self,
        client: ChatClient | Callable[[str, str], str],
        max_attempts: int = 2,
        tag_descriptions: dict[str, str] | None = None,
    ):
        self.client = client
        self.max_attempts = max_attempts
        self.tag_descriptions = tag_descriptions or {}

    def annotate(self, rec: Recommendation) -> Recommendation:
        facts = card_facts(rec, self.tag_descriptions)
        last_reason = ""
        for _ in range(self.max_attempts):
            try:
                text = self.client(SYSTEM_PROMPT, "Карточка:\n" + facts)
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
                return rec.model_copy(update={"llm_commentary": None, "llm_status": f"unavailable: {type(exc).__name__}: {exc}"[:300]})
            text = (text or "").strip()
            if len(text) < MIN_COMMENTARY_CHARS:
                last_reason = f"слишком короткий ответ ({len(text)} символов)"
                continue
            if _FOREIGN_SCRIPT_RE.search(text):
                last_reason = "ответ содержит текст не на русском языке"
                continue
            bad = ungrounded_items(text, facts)
            if not bad:
                return rec.model_copy(update={"llm_commentary": text, "llm_status": "ok"})
            last_reason = "не подтверждены карточкой: " + ", ".join(dict.fromkeys(bad))
        return rec.model_copy(update={"llm_commentary": None, "llm_status": f"rejected: {last_reason}"[:300]})
