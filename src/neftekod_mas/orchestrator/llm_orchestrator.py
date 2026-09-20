"""
LLM-оркестратор на локальной Qwen (ARCHITECTURE.md §9.1).

В отличие от LLM Monitor (§9), который только пересказывает готовую
карточку, здесь модель ВЕДЁТ ЦИКЛ: сама решает, каких агентов опросить,
в каком порядке, какие кандидаты разобрать подробнее и что вынести
оператору. Это и есть "оркестратор" в смысле ТЗ п.3 -- роль, которая
"запрашивает оценки агентов, разрешает конфликт целей, проверяет
согласованность и формирует итог".

Почему это безопасно
--------------------
Модель не вычисляет НИЧЕГО. Она только выбирает, какой детерминированный
инструмент вызвать. Каждый инструмент -- это вызов уже существующего,
покрытого тестами агента; числа приходят из него, а не из модели.
Финальный шаг `finalize` принимает ТОЛЬКО идентификатор кандидата,
который до этого сгенерировал и признал допустимым детерминированный
Агент оптимизации: выдумать свои значения уставок модель физически не
может -- такого канала у неё нет. После выбора кандидат всё равно
проходит независимый Guard, у которого право вето (§6.5). Если Guard
блокирует -- выбор модели отменяется, и цикл продолжается без неё.

Иначе говоря, LLM здесь -- планировщик и диспетчер, а не источник
числовых решений. Это прямая реализация разделения Monitor /
Orchestrator / Verification / Execution из Schall (2026), где право
действия остаётся у детерминированных слоёв.

Деградация
----------
Модель недоступна, зациклилась, вернула мусор, выбрала несуществующий
кандидат или её выбор заблокировал Guard -- во всех случаях
отрабатывает обычный детерминированный `Orchestrator`, и его результат
и уходит оператору. Поэтому подключение LLM не может ухудшить решение:
худшее, что она может сделать -- не помочь.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from neftekod_mas.schemas import (
    AgentMessage,
    GuardVerdict,
    Recommendation,
    metric_name,
)

MAX_TOOL_CALLS = 10
DEFAULT_TIMEOUT_S = 300.0

SYSTEM_PROMPT = """Ты -- оркестратор мультиагентной системы поддержки решений оператора
установки гидроочистки дизельного топлива (цепочка АВТ -> гидроочистка -> блендинг).

Твоя работа -- ПРОВЕСТИ ЦИКЛ ПРИНЯТИЯ РЕШЕНИЯ, вызывая инструменты. Сам ты ничего не
вычисляешь и не придумываешь чисел: все значения приходят только из инструментов.

Порядок работы:
1. get_process_state -- состояние процесса и качество данных.
2. assess_quality -- показатели качества и нарушения норматива.
3. assess_reliability -- тяжесть режима и риск для оборудования.
4. Если есть нарушение по T95 или нужно понять состав сырья -- assess_blending.
5. list_candidates -- допустимые варианты действий (их придумал детерминированный
   агент оптимизации, не ты).
6. inspect_candidate по одному-двум самым перспективным.
7. finalize -- ровно один раз, в самом конце.

Правила, которые нельзя нарушать:
- Качество и жёсткие ограничения важнее экономики. Недопустимый режим нельзя
  оправдать выпуском или экономией.
- В finalize можно передать ТОЛЬКО candidate_id из list_candidates. Свои значения
  уставок предлагать запрещено, такого канала у тебя нет.
- Если допустимых кандидатов нет, или ни один не устраняет нарушение, или данные
  ненадёжны -- вызови finalize с action="refuse" и объясни причину. Отказ -- это
  нормальный и правильный исход, а не неудача.
- Не вызывай один и тот же инструмент с теми же аргументами дважды.
- rationale пиши по-русски, 1-3 предложения, без markdown.
- finalize ОБЯЗАТЕЛЬНО вызывай как инструмент. Не пиши JSON в текст ответа --
  такой ответ система не примет.
- Поле идентификатора называется ровно candidate_id."""


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result_summary: str


@dataclass
class LLMOrchestratorOutcome:
    """Результат работы LLM-слоя. `recommendation` заполняется только
    если модель довела цикл до конца И её выбор прошёл Guard."""

    recommendation: Recommendation | None
    status: str  # ok | refused | fallback: <причина>
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    rationale: str | None = None


@dataclass
class OpenAIToolClient:
    """Минимальный клиент /v1/chat/completions с поддержкой tools.
    Зависимостей нет -- urllib стандартной библиотеки, как и в §9."""

    base_url: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    temperature: float = 0.1
    max_tokens: int = 700

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        opener = _local_opener(self.base_url)
        with opener.open(req, timeout=self.timeout_s) as resp:  # noqa: S310 -- URL задаёт оператор
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]



def _local_opener(base_url: str):
    """Открыватель HTTP, который НЕ ходит через системный прокси, если
    сервер модели локальный.

    Практическая причина: в среде с заданным `http_proxy` (корпоративный
    периметр, WSL, докер-хост) urllib отправляет даже запрос к
    127.0.0.1 в прокси, и локальная модель отвечает "502 Bad Gateway".
    Маска вида `127.*` в `no_proxy` при этом не помогает: Python
    сопоставляет записи `no_proxy` как суффиксы имени хоста, а не как
    шаблоны. Для локального адреса прокси не нужен по определению,
    поэтому он отключается явно -- это также ровно тот случай, который
    описан в ТЗ: продакшен работает в закрытом контуре без интернета.
    """
    host = urllib.parse.urlsplit(base_url).hostname or ""
    is_local = host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")
    if is_local:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()

def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


FINALIZE_TOOL = _tool(
    "finalize",
    "Завершить цикл. action='recommend' с candidate_id из list_candidates -- "
    "предложить вариант оператору; action='refuse' -- мотивированно отказаться.",
    {
        "action": {"type": "string", "enum": ["recommend", "refuse"]},
        "candidate_id": {"type": "string", "description": "Обязателен при action='recommend'."},
        "rationale": {"type": "string", "description": "Почему именно так. По-русски, 1-3 предложения."},
    },
    ["action", "rationale"],
)

TOOLS = [
    _tool("get_process_state",
          "Снимок состояния процесса на момент решения: ключевые теги КИП, свежесть "
          "лабораторных анализов, флаги качества данных, стационарность режима реактора."),
    _tool("assess_quality",
          "Оценка Агента качества: значение каждого нормируемого показателя (сера, T95, "
          "цетановое число, CFPP, плотность), источник оценки, её типичная ошибка и "
          "список показателей, нарушающих норматив или подошедших к порогу действия."),
    _tool("assess_reliability",
          "Оценка Агента надёжности: индекс тяжести режима, факторы риска, признак "
          "hard_stop (режим уже критичен, ухудшать его запрещено)."),
    _tool("assess_blending",
          "Состояние дизельного пула АВТ: компоненты, их расходы и доли (сумма = 100 %), "
          "качество смеси и то, какие показатели модель смешения умеет прогнозировать."),
    _tool("list_candidates",
          "Список допустимых вариантов действий от детерминированного Агента оптимизации: "
          "candidate_id, что меняется, прогноз по нарушенным показателям, экономика. "
          "Только эти candidate_id разрешено передавать в finalize."),
    _tool("explain_rejections",
          "Почему отклонены недопустимые варианты -- помогает понять, есть ли вообще "
          "рычаг у проблемы.",
          {"metric": {"type": "string",
                      "description": "Ключ показателя, напр. sulfur_mg_kg или t95_c. "
                                     "Пусто -- по всем."}}),
    _tool("inspect_candidate",
          "Подробности одного варианта: все прогнозируемые показатели, риск оборудования, "
          "экономический эффект в тоннах, Гкал, кВт и рублях, оговорки.",
          {"candidate_id": {"type": "string"}}, ["candidate_id"]),
    FINALIZE_TOOL,
]


class LLMOrchestrator:
    """Обёртка вокруг детерминированного оркестратора.

    Сначала детерминированный цикл считает ВСЁ (состояние, оценки агентов,
    кандидатов, Guard) -- это происходит независимо от модели и является
    источником истины. Затем LLM получает доступ к уже посчитанному через
    инструменты и выбирает, что вынести оператору. Такой порядок выбран
    намеренно: он гарантирует, что детерминированный результат существует
    всегда, ещё до первого обращения к модели.
    """

    def __init__(self, deterministic, client: Callable[[list[dict], list[dict]], dict],
                 max_tool_calls: int = MAX_TOOL_CALLS):
        self.deterministic = deterministic
        self.client = client
        self.max_tool_calls = max_tool_calls

    def run_cycle(self, *args, **kwargs) -> Recommendation:
        baseline = self.deterministic.run_cycle(*args, **kwargs)
        context = getattr(self.deterministic, "last_cycle_context", None)
        if context is None:
            baseline.llm_status = "fallback: детерминированный цикл не отдал контекст для инструментов"
            return baseline

        outcome = self._drive(context, baseline)
        for i, call in enumerate(outcome.tool_calls, start=1):
            baseline.trace.append(AgentMessage(
                seq=len(baseline.trace) + 1,
                sender="llm_orchestrator", recipient=call.name,
                topic="ToolCall", summary=f"#{i} {call.name}({_short_args(call.arguments)}) -> {call.result_summary}",
            ))

        if outcome.status == "refused" and baseline.is_refusal:
            baseline.llm_status = outcome.status
            baseline.llm_commentary = outcome.rationale
            baseline.orchestration_mode = "llm"
            return baseline

        if outcome.recommendation is not None:
            outcome.recommendation.trace = baseline.trace
            outcome.recommendation.llm_status = outcome.status
            outcome.recommendation.llm_commentary = outcome.rationale
            outcome.recommendation.orchestration_mode = "llm"
            return outcome.recommendation

        # Любой сбой модели -- это возврат к детерминированной карточке,
        # которая была посчитана ДО обращения к ней. Комментарий модели
        # при откате не подставляется: он относился к другому решению.
        baseline.llm_status = outcome.status
        baseline.orchestration_mode = "deterministic"
        return baseline

    # -- цикл tool-calling ----------------------------------------------

    def _drive(self, context: dict, baseline: Recommendation) -> LLMOrchestratorOutcome:
        tools_impl = _ToolImpl(context, self.deterministic)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Момент принятия решения: {context['decision_at']:%Y-%m-%d %H:%M}. "
                "Проведи цикл и заверши его вызовом finalize."},
        ]
        calls: list[ToolCallRecord] = []
        seen: set[str] = set()
        repeats = 0
        finalize_only = False

        for step in range(self.max_tool_calls):
            # Локальные модели 7B и меньше склонны "залипать": собрав все
            # факты, они продолжают опрашивать те же инструменты вместо
            # finalize. Когда данных для решения заведомо достаточно
            # (повторы или конец бюджета шагов), список инструментов
            # сужается до одного finalize -- выбрать что-то другое модель
            # тогда просто не может. Это ограничение диспетчера, а не
            # подсказка: на содержание решения оно не влияет.
            if not finalize_only and (repeats >= 2 or step >= self.max_tool_calls - 2):
                finalize_only = True
                messages.append({
                    "role": "user",
                    "content": "Данных достаточно. Заверши цикл: вызови finalize "
                               "(action='recommend' с candidate_id из list_candidates, "
                               "либо action='refuse' с причиной).",
                })

            active_tools = [FINALIZE_TOOL] if finalize_only else TOOLS
            try:
                message = self.client(messages, active_tools)
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as exc:
                return LLMOrchestratorOutcome(None, f"fallback: модель недоступна ({type(exc).__name__}: {exc})", calls)

            tool_calls = message.get("tool_calls") or []
            content = (message.get("content") or "").strip()
            if not tool_calls:
                # Модели 7B и меньше регулярно печатают вызов finalize как
                # JSON в тексте вместо настоящего tool-call. Разбираем и
                # такой ответ: это не ослабляет защиту -- finalize всё
                # равно умеет принять только уже проверенный candidate_id,
                # а решение всё равно проходит Guard.
                inline = _finalize_from_content(content)
                if inline is not None:
                    return self._finalize(tools_impl, inline, calls, baseline)
                return LLMOrchestratorOutcome(
                    None, "fallback: модель завершила ход без вызова finalize", calls,
                    rationale=content or None,
                )

            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                arguments = _parse_arguments(fn.get("arguments"))

                if name == "finalize":
                    return self._finalize(tools_impl, arguments, calls, baseline)

                signature = f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
                if signature in seen:
                    repeats += 1
                    result = {"error": "Этот инструмент уже вызывался с теми же аргументами. "
                                       "Переходи к следующему шагу или вызови finalize."}
                else:
                    seen.add(signature)
                    result = tools_impl.call(name, arguments)

                calls.append(ToolCallRecord(name, arguments, _summarize(result)))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", name),
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        return LLMOrchestratorOutcome(
            None, f"fallback: модель не вызвала finalize за {self.max_tool_calls} шагов", calls
        )

    def _finalize(self, tools_impl, arguments: dict, calls: list[ToolCallRecord],
                  baseline: Recommendation) -> LLMOrchestratorOutcome:
        action = str(arguments.get("action", "")).lower()
        rationale = (arguments.get("rationale") or "").strip() or None
        calls.append(ToolCallRecord("finalize", arguments, action or "(пусто)"))

        if action == "refuse":
            # Отказ модели принимается, только если детерминированный слой
            # тоже не нашёл решения. Если решение есть, а модель хочет
            # отказаться -- это её ошибка, а не новая информация: все
            # проверки уже сделаны детерминированно.
            if baseline.is_refusal:
                return LLMOrchestratorOutcome(None, "refused", calls, rationale=rationale)
            return LLMOrchestratorOutcome(
                None, "fallback: модель предложила отказ, хотя проверенное решение существует", calls
            )

        candidate_id = next(
            (arguments[k] for k in ("candidate_id", "selected_candidate_id", "id", "candidate")
             if arguments.get(k)),
            None,
        )
        if not candidate_id:
            return LLMOrchestratorOutcome(None, "fallback: finalize без candidate_id", calls)

        built = tools_impl.build_recommendation(str(candidate_id))
        if built is None:
            return LLMOrchestratorOutcome(
                None, f"fallback: кандидат '{candidate_id}' не из числа проверенных допустимых", calls
            )
        recommendation, guard_verdict = built
        if guard_verdict == GuardVerdict.BLOCK:
            return LLMOrchestratorOutcome(
                None, f"fallback: выбор модели '{candidate_id}' заблокирован Guard", calls
            )
        return LLMOrchestratorOutcome(recommendation, "ok", calls, rationale=rationale)


class _ToolImpl:
    """Реализация инструментов. Каждый -- чтение уже посчитанного
    детерминированного результата; ни один не запускает новых вычислений
    от имени модели и ничего не меняет в состоянии."""

    def __init__(self, context: dict, deterministic):
        self.ctx = context
        self.deterministic = deterministic

    def call(self, name: str, arguments: dict) -> dict:
        handler = {
            "get_process_state": self.get_process_state,
            "assess_quality": self.assess_quality,
            "assess_reliability": self.assess_reliability,
            "assess_blending": self.assess_blending,
            "list_candidates": self.list_candidates,
            "explain_rejections": self.explain_rejections,
            "inspect_candidate": self.inspect_candidate,
        }.get(name)
        if handler is None:
            return {"error": f"Нет такого инструмента: {name}"}
        try:
            return handler(**arguments) if arguments else handler()
        except TypeError as exc:
            return {"error": f"Неверные аргументы: {exc}"}

    # -- инструменты ----------------------------------------------------

    def get_process_state(self) -> dict:
        state = self.ctx["state"]
        return {
            "decision_at": f"{self.ctx['decision_at']:%Y-%m-%d %H:%M}",
            "steady_regime": state.steady_regime,
            "key_state": {k: round(v, 4) for k, v in self.ctx["key_state"].items()},
            "data_quality_flags": [
                {"code": f.code, "tag_or_point": f.tag_or_point, "severity": f.severity.value}
                for f in state.quality_report.flags[:10]
            ],
            "flags_total": len(state.quality_report.flags),
        }

    def assess_quality(self) -> dict:
        quality = self.ctx["quality"]
        return {
            "metrics": [
                {"metric": metric_name(e.metric), "key": e.metric, "value": round(e.value, 4),
                 "unit": e.unit, "source": e.source.value,
                 "age_minutes": None if e.age_minutes is None else round(e.age_minutes),
                 "typical_error": e.typical_error, "confidence": e.confidence.value}
                for e in quality.current
            ],
            "violations": [
                {"metric": metric_name(v.metric), "key": v.metric, "limit": v.limit,
                 "op": v.op, "margin": round(v.margin, 4), "risk_class": v.risk_class.value}
                for v in quality.violations
            ],
            "violated_metric_keys": sorted(self.ctx["violated_metrics"]),
            "overall_confidence": quality.overall_confidence.value,
        }

    def assess_reliability(self) -> dict:
        risk = self.ctx["risk"]
        return {
            "severity_index": round(risk.severity_index, 4),
            "risk_class": risk.risk_class.value,
            "hard_stop": risk.hard_stop,
            "factors": [
                {"tag": f.tag_id, "description": f.description,
                 "contribution": round(f.contribution, 4), "is_assumption": f.is_assumption}
                for f in risk.factors[:6]
            ],
            "missing_inputs": risk.missing_inputs[:6],
        }

    def assess_blending(self) -> dict:
        blending = self.ctx.get("blending")
        if blending is None:
            return {"available": False,
                    "reason": "Агент блендинга не подключён (нет config/blend_model.yaml)"}
        if not blending.available:
            return {"available": False, "reason": blending.unavailable_reason}
        return {
            "available": True,
            "total_flow_t_h": blending.total_flow_t_h,
            "share_sum_pct": blending.share_sum_pct,
            "components": [
                {"key": c.key, "description": c.description, "flow_t_h": c.flow_t_h,
                 "mass_share_pct": c.mass_share_pct, "sulfur_pct_mass": c.sulfur_pct_mass}
                for c in blending.components
            ],
            "blend_quality": blending.blend_quality,
            "metrics_model_can_predict": blending.supported_metrics,
        }

    def list_candidates(self) -> dict:
        result = self.ctx["optimization"]
        if result.no_feasible_solution:
            return {"feasible": [], "no_feasible_solution": True,
                    "reason": result.no_feasible_reason}
        ranked = sorted(result.feasible_candidates,
                        key=lambda c: (c.score if c.score is not None else -1e9), reverse=True)
        return {
            "no_feasible_solution": False,
            "recommended_by_optimizer": result.recommended_candidate_id,
            "pareto_front_ids": result.pareto_front_ids,
            "feasible": [self._brief(c) for c in ranked[:10]],
            "feasible_total": len(result.feasible_candidates),
        }

    def explain_rejections(self, metric: str = "") -> dict:
        rejected = [c for c in self.ctx.get("all_candidates", []) if not c.feasible]
        rows = []
        for c in rejected[:12]:
            rows.append({
                "candidate_id": c.candidate_id,
                "actions": [f"{a.variable_name} {a.current_value:.4g} -> {a.recommended_value:.4g} {a.unit}"
                            for a in c.actions],
                "rejection_reason": c.rejection_reason,
            })
        return {"rejected_total": len(rejected), "rejected": rows,
                "note": "Эти варианты отбракованы детерминированными проверками и недоступны для finalize."}

    def inspect_candidate(self, candidate_id: str) -> dict:
        c = self._by_id().get(candidate_id)
        if c is None:
            return {"error": f"Нет допустимого кандидата с id '{candidate_id}'",
                    "available_ids": sorted(self._by_id())}
        out = self._brief(c)
        out["predicted_quality"] = [
            {"metric": metric_name(e.metric), "key": e.metric, "value": round(e.value, 4),
             "unit": e.unit, "typical_error": e.typical_error, "confidence": e.confidence.value}
            for e in c.predicted_quality
        ]
        out["equipment_risk"] = {"severity_index": round(c.predicted_risk.severity_index, 4),
                                 "risk_class": c.predicted_risk.risk_class.value,
                                 "hard_stop": c.predicted_risk.hard_stop}
        out["caveats"] = c.caveats
        out["economics"] = c.economics
        return out

    # -- сборка итоговой карточки по выбору модели ----------------------

    def build_recommendation(self, candidate_id: str):
        candidate = self._by_id().get(candidate_id)
        if candidate is None:
            return None
        return self.deterministic.recommendation_for(self.ctx, candidate)

    def _by_id(self) -> dict:
        result = self.ctx["optimization"]
        return {c.candidate_id: c for c in result.feasible_candidates}

    def _brief(self, c) -> dict:
        violated = self.ctx["violated_metrics"]
        return {
            "candidate_id": c.candidate_id,
            "actions": [{"variable": a.variable_name, "tag": a.tag,
                         "from": round(a.current_value, 4), "to": round(a.recommended_value, 4),
                         "unit": a.unit} for a in c.actions],
            "effect_on_violated": {
                metric_name(e.metric): round(e.value, 4)
                for e in c.predicted_quality if e.metric in violated
            },
            "equipment_severity_index": round(c.predicted_risk.severity_index, 4),
            "net_rub_per_day": (c.economics or {}).get("net_rub_per_day"),
            "score": None if c.score is None else round(c.score, 4),
            "caveats_count": len(c.caveats),
        }


_FINALIZE_ID_KEYS = ("candidate_id", "selected_candidate_id", "id", "candidate")


def _finalize_from_content(content: str) -> dict | None:
    """Вытаскивает вызов finalize из текста ответа модели.

    Возвращает словарь аргументов или None, если это не finalize.
    Намеренно не пытается угадывать намерение по свободному тексту:
    распознаётся только явный JSON-объект, в котором упомянут finalize
    или есть пара action/candidate_id.
    """
    if not content or "finaliz" not in content.lower():
        return None
    for chunk in re.findall(r"\{.*?\}", content, flags=re.DOTALL):
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else parsed
        candidate_id = next((args[k] for k in _FINALIZE_ID_KEYS if args.get(k)), None)
        action = str(args.get("action", "")).lower()
        if action in ("recommend", "refuse"):
            pass
        elif candidate_id:
            action = "recommend"
        else:
            continue
        rationale = args.get("rationale") or _strip_code_blocks(content)
        return {"action": action, "candidate_id": candidate_id, "rationale": rationale}
    return None


def _strip_code_blocks(text: str) -> str:
    """Убирает JSON-блоки из текста: в комментарий оператору не должен
    попадать служебный вызов инструмента."""
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"\{.*?\}", " ", cleaned, flags=re.DOTALL)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _short_args(arguments: dict) -> str:
    if not arguments:
        return ""
    return ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:2])


def _summarize(result: dict) -> str:
    if "error" in result:
        return f"ошибка: {result['error']}"[:120]
    for key in ("violated_metric_keys", "feasible_total", "severity_index", "available", "flags_total"):
        if key in result:
            return f"{key}={result[key]}"
    return "ok"
