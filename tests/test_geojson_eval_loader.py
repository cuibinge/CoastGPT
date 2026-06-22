import json


def _feature(coords, label="sea"):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"DLMC": label},
    }


def test_load_dataset_entries_accepts_data_key(tmp_path):
    from Tools.eval_geojson import load_dataset_entries

    dataset = tmp_path / "geojson_train.json"
    dataset.write_text(
        json.dumps({"data": [{"name": "a.tif"}, {"name": "b.tif"}]}),
        encoding="utf-8",
    )

    entries = load_dataset_entries(dataset)

    assert [entry["name"] for entry in entries] == ["a.tif", "b.tif"]


def test_evaluate_entry_requires_iou_even_when_properties_match():
    from Tools.eval_geojson import evaluate_entry

    gt = _feature([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], label="sea")
    pred = _feature([[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]], label="sea")
    entry = {"features": [gt]}
    pred_text = json.dumps({"type": "FeatureCollection", "features": [pred]})

    result = evaluate_entry(
        entry,
        pred_text,
        iou_threshold=0.5,
        property_keys={"DLMC"},
    )

    assert result["matches"] == []
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_evaluate_entry_matches_when_iou_and_properties_match():
    from Tools.eval_geojson import evaluate_entry

    gt = _feature([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], label="sea")
    pred = _feature([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], label="sea")
    entry = {"features": [gt]}
    pred_text = json.dumps({"type": "FeatureCollection", "features": [pred]})

    result = evaluate_entry(
        entry,
        pred_text,
        iou_threshold=0.5,
        property_keys={"DLMC"},
    )

    assert len(result["matches"]) == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
