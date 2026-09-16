from pathlib import Path

from neftekod_mas.tags.pid_graph import TagGraph, TagNode, qualify

REPO_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = REPO_ROOT / "config" / "tag_ontology.json"


def test_qualify_avoids_cross_installation_collision():
    """T6 существует в обоих CSV (avt_tags.csv и 242000_tags.csv) и
    обозначает РАЗНЫЕ физические теги -- граф не должен их путать."""
    g = TagGraph()
    g.add(TagNode(
        tag_id="T6", installation="avt", stage="K1", description="Температура низа К-1",
        unit="°C", physical_quantity="temperature", node_kind="controller",
        actuatable=True, confidence="confirmed_from_pid", controller_type="TC",
    ))
    g.add(TagNode(
        tag_id="T6", installation="242000", stage="reactor",
        description="Температура ГСС на входе Р-202", unit="°C",
        physical_quantity="temperature", node_kind="sensor",
        actuatable=False, confidence="assumption",
    ))
    assert len(g.nodes) == 2
    assert g.is_actuatable(qualify("avt", "T6")) is True
    assert g.is_actuatable(qualify("242000", "T6")) is False


def test_downstream_impact_within_avt_stage_order():
    g = TagGraph()
    g.add(TagNode(tag_id="P2", installation="avt", stage="K1", description="", unit="",
                   physical_quantity="pressure", node_kind="controller", actuatable=True,
                   confidence="confirmed_from_pid", controller_type="PIC"))
    g.add(TagNode(tag_id="P22", installation="avt", stage="K2", description="", unit="",
                   physical_quantity="pressure", node_kind="controller", actuatable=True,
                   confidence="confirmed_from_pid", controller_type="PIC"))
    g.add(TagNode(tag_id="T55", installation="avt", stage="K10", description="", unit="",
                   physical_quantity="temperature", node_kind="controller", actuatable=True,
                   confidence="confirmed_from_pid", controller_type="TC"))
    downstream = g.downstream_impact(qualify("avt", "P2"))
    assert qualify("avt", "P22") in downstream
    assert qualify("avt", "T55") in downstream
    # K10 ничего не должно затрагивать выше по потоку (K1/K2)
    downstream_from_k10 = g.downstream_impact(qualify("avt", "T55"))
    assert qualify("avt", "P2") not in downstream_from_k10
    assert qualify("avt", "P22") not in downstream_from_k10


def test_generated_ontology_if_present():
    """Если config/tag_ontology.json уже сгенерирован локально
    (scripts/build_tag_ontology.py), сверяем базовые инварианты."""
    if not ONTOLOGY_PATH.exists():
        return
    g = TagGraph.from_json(ONTOLOGY_PATH)
    assert len(g.nodes) >= 90
    assert g.tag_exists(qualify("avt", "T55"))
    assert g.is_actuatable(qualify("avt", "T55"))
    assert not g.is_actuatable(qualify("avt", "T1"))
    assert g.tag_exists(qualify("242000", "F15"))
