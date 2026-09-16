"""
Цифровой P&ID -- граф тегов, используемый Guard-слоем Оркестратора для
структурной проверки предложений (ARCHITECTURE.md §3, §6.5), по образцу
P&ID-grounded validation из Schall (2026), §3.2.5.

Важно: `avt_tags.csv` и `242000_tags.csv` -- это два НЕЗАВИСИМЫХ файла
со своими короткими кодами колонок (T1, F9, T6...), которые совпадают
между установками чисто текстуально, но обозначают разные физические
теги (напр. "T6" в avt_tags.csv -- температура низа К-1, а "T6" в
242000_tags.csv -- температура ГСС на входе Р-202). Поэтому узлы графа
идентифицируются составным ключом `installation:tag_id`
(см. `qualify()`), а не голым кодом тега -- ранняя версия этого модуля
без составного ключа молча затирала 8 тегов АВТ одноимёнными тегами
24-2000 при построении графа.

Топология рёбер здесь -- намеренно грубая (на уровне стадий: К1 -> К2 ->
К10 -> [гидроочистка 24-2000] -> [блендинг]), а не полная схема
трубопроводов "тег-к-тегу": детальной пообвязочной связности в выданных
материалах нет (только 4 частичных скана P&ID для АВТ). Это явное
ограничение, а не скрытая неточность -- см. ARCHITECTURE.md §13.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


class NodeKind:
    SENSOR = "sensor"
    CONTROLLER = "controller"
    LAB_ANALYZER = "lab_analyzer"
    PAK_ANALYZER = "pak_analyzer"
    VIRTUAL_ANALYZER = "virtual_analyzer"


class Confidence:
    CONFIRMED_FROM_PID = "confirmed_from_pid"
    ASSUMPTION = "assumption"


def qualify(installation: str, tag_id: str) -> str:
    """Составной идентификатор узла графа -- см. docstring модуля."""
    return f"{installation}:{tag_id}"


@dataclass
class TagNode:
    tag_id: str  # код колонки, как в исходном CSV (не уникален глобально!)
    installation: str  # "avt" | "242000"
    stage: str  # "K1" | "K2" | "K10" | "reactor" | "stabilization" | "unknown"
    description: str
    unit: str
    physical_quantity: str
    node_kind: str
    actuatable: bool
    confidence: str
    controller_type: str | None = None  # напр. "FC", "TC", None для индикаторов

    @property
    def qualified_id(self) -> str:
        return qualify(self.installation, self.tag_id)


# Грубый порядок стадий по потоку вещества (ARCHITECTURE.md §3).
_STAGE_ORDER: dict[str, list[str]] = {
    "avt": ["K1", "K2", "K10"],
    "242000": ["reactor", "stabilization", "unit_general"],
}
_INSTALLATION_ORDER = ["avt", "242000", "blend"]


@dataclass
class TagGraph:
    # ключ -- qualified_id ("avt:T6", "242000:T6", ...), см. docstring модуля
    nodes: dict[str, TagNode] = field(default_factory=dict)

    # -- построение --------------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> "TagGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = {qid: TagNode(**attrs) for qid, attrs in data["nodes"].items()}
        return cls(nodes=nodes)

    def to_json(self, path: Path) -> None:
        payload = {"nodes": {qid: vars(n) for qid, n in self.nodes.items()}}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, node: TagNode) -> None:
        self.nodes[node.qualified_id] = node

    # -- проверки для Guard-слоя (ARCHITECTURE.md §6.5) ----------------
    # Везде ниже tag_ref -- составной qualified_id ("avt:T55"), а не
    # голый код тега, во избежание коллизий между установками.

    def tag_exists(self, tag_ref: str) -> bool:
        return tag_ref in self.nodes

    def is_actuatable(self, tag_ref: str) -> bool:
        node = self.nodes.get(tag_ref)
        return bool(node and node.actuatable)

    def downstream_impact(self, tag_ref: str) -> list[str]:
        """BFS по грубой топологии стадий: какие другие теги того же и
        последующих узлов потенциально затронуты изменением tag_ref.
        Аналог "downstream impact" проверки в Schall (2026), §3.2.5,
        но на уровне стадий процесса, а не индивидуальных трубопроводов
        -- см. ограничение в docstring модуля."""
        node = self.nodes.get(tag_ref)
        if node is None:
            return []

        order = _INSTALLATION_ORDER
        if node.installation in order:
            idx = order.index(node.installation)
            affected_installations = order[idx:]
        else:
            affected_installations = [node.installation]

        stage_seq = _STAGE_ORDER.get(node.installation, [])
        if node.stage in stage_seq:
            idx = stage_seq.index(node.stage)
            affected_stages = set(stage_seq[idx:])
        else:
            affected_stages = {node.stage}

        downstream: list[str] = []
        for qid, other in self.nodes.items():
            if qid == tag_ref:
                continue
            if other.installation == node.installation and other.stage in affected_stages:
                downstream.append(qid)
            elif other.installation in affected_installations[1:]:
                downstream.append(qid)
        return downstream

    def bfs_reachable(self, start_tags: list[str]) -> set[str]:
        """Общий BFS-примитив поверх downstream_impact -- для случаев,
        когда нужно объединить влияние нескольких одновременно
        меняющихся тегов (несколько компонентов одного кандидата)."""
        visited: set[str] = set(start_tags)
        queue = deque(start_tags)
        while queue:
            qid = queue.popleft()
            for nxt in self.downstream_impact(qid):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited
