from types import SimpleNamespace

from src.detection.train import metrics_dict, write_metrics_report


def test_metrics_dict_is_serialisable_and_stable():
    results = SimpleNamespace(
        box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.75, map=0.6,
                            maps=[0.5, 0.6, 0.7]),
        names={0: "traffic_cone", 1: "barrier", 2: "stop_sign"},
    )
    assert metrics_dict(results) == {
        "precision": 0.8,
        "recall": 0.7,
        "map50": 0.75,
        "map50_95": 0.6,
        "per_class_map50_95": {
            "traffic_cone": 0.5,
            "barrier": 0.6,
            "stop_sign": 0.7,
        },
    }


def test_write_metrics_report_writes_json_and_markdown(tmp_path):
    output = tmp_path / "metrics.json"
    write_metrics_report({"precision": 0.8, "recall": 0.7,
                         "map50": 0.75, "map50_95": 0.6}, output)
    assert '"precision": 0.8' in output.read_text()
    assert "mAP50-95" in output.with_suffix(".md").read_text()
