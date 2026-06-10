import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def load_dataset_entries(dataset_path: Path) -> List[Dict[str, Any]]:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if 'features' in data:
            return data['features']
        return [data]
    return []


def load_prediction_map(predictions_jsonl: Path) -> Dict[str, str]:
    pred_map: Dict[str, str] = {}
    if not predictions_jsonl.exists():
        return pred_map
    for raw_line in predictions_jsonl.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get('sample_id', '')).strip()
        pred_text = str(record.get('prediction', '')).strip()
        if sample_id and pred_text:
            pred_map[sample_id] = pred_text
    return pred_map


def parse_property_keys(property_keys_str: str) -> Optional[Set[str]]:
    if not property_keys_str or property_keys_str.strip() == '':
        return None
    keys = {k.strip() for k in property_keys_str.split(',') if k.strip()}
    return keys if keys else None


def _candidate_ids_from_dataset_entry(entry: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ('sample_id', 'id', 'name', 'filename'):
        value = entry.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                ids.append(text)
    return ids


def _extract_geojson_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    for start_marker in ('{"type"', '{ "type"'):
        idx = text.find(start_marker)
        if idx >= 0:
            text = text[idx:]
            break
    brace_count = 0
    for i, ch in enumerate(text):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
        if brace_count == 0 and i > 0:
            text = text[:i + 1]
            break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        text = _repair_truncated_geojson(text)
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _repair_truncated_geojson(text: str) -> str:
    if not text.startswith('{'):
        text = '{' + text
    open_brackets = text.count('[') - text.count(']')
    open_braces = text.count('{') - text.count('}')
    suffix = ']' * open_brackets + '}' * open_braces
    text = text.rstrip(', \t\n\r')
    if not text.endswith(suffix):
        text += suffix
    return text


def _compute_polygon_iou(poly_a, poly_b):
    try:
        from shapely.geometry import Polygon
        p_a = Polygon(poly_a)
        p_b = Polygon(poly_b)
        if not p_a.is_valid:
            p_a = p_a.buffer(0)
        if not p_b.is_valid:
            p_b = p_b.buffer(0)
        intersection = p_a.intersection(p_b).area
        union = p_a.union(p_b).area
        if union == 0:
            return 0.0
        return intersection / union
    except ImportError:
        return _simple_bbox_iou(poly_a, poly_b)


def _simple_bbox_iou(poly_a, poly_b):
    def bbox(coords):
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return min(xs), min(ys), max(xs), max(ys)

    ax1, ay1, ax2, ay2 = bbox(poly_a)
    bx1, by1, bx2, by2 = bbox(poly_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    if union == 0:
        return 0.0
    return intersection / union


def _extract_polygons_from_feature(feature):
    polygons = []
    geom = feature.get('geometry') or {}
    geom_type = str(geom.get('type', '')).lower()
    coords = geom.get('coordinates') or []

    if geom_type == 'polygon':
        if isinstance(coords, list) and len(coords) > 0:
            if isinstance(coords[0], list) and len(coords[0]) > 0:
                polygons.append(coords[0])
    elif geom_type == 'multipolygon':
        for poly in coords:
            if isinstance(poly, list) and len(poly) > 0:
                if isinstance(poly[0], list) and len(poly[0]) > 0:
                    polygons.append(poly[0])
    return polygons


def _extract_feature_properties(feature, property_keys):
    props = feature.get('properties') or {}
    if property_keys is None:
        return dict(props)
    return {k: v for k, v in props.items() if k in property_keys}


def _match_features_by_property(pred_features, gt_features, property_keys, iou_threshold):
    matches = []
    matched_pred = set()
    matched_gt = set()

    scored = []
    for pi, pf in enumerate(pred_features):
        pred_polys = _extract_polygons_from_feature(pf)
        pred_props = _extract_feature_properties(pf, property_keys)
        for gi, gf in enumerate(gt_features):
            gt_polys = _extract_polygons_from_feature(gf)
            gt_props = _extract_feature_properties(gf, property_keys)
            prop_match = _props_match(pred_props, gt_props, property_keys)
            best_iou = 0.0
            for pp in pred_polys:
                for gp in gt_polys:
                    iou = _compute_polygon_iou(pp, gp)
                    if iou > best_iou:
                        best_iou = iou
            scored.append((pi, gi, best_iou, prop_match))

    scored.sort(key=lambda x: (x[2], x[3]), reverse=True)

    for pi, gi, iou, prop_match in scored:
        if pi in matched_pred or gi in matched_gt:
            continue
        if iou >= iou_threshold or prop_match:
            matches.append((pi, gi, iou))
            matched_pred.add(pi)
            matched_gt.add(gi)

    unmatched_pred = set(range(len(pred_features))) - matched_pred
    unmatched_gt = set(range(len(gt_features))) - matched_gt
    return matches, unmatched_pred, unmatched_gt


def _props_match(pred_props, gt_props, property_keys):
    if not pred_props or not gt_props:
        return False
    keys_to_check = property_keys if property_keys else (set(pred_props.keys()) & set(gt_props.keys()))
    if not keys_to_check:
        return False
    for k in keys_to_check:
        pv = str(pred_props.get(k, '')).strip().lower()
        gv = str(gt_props.get(k, '')).strip().lower()
        if pv and gv and pv == gv:
            return True
    return False


def evaluate_entry(entry, pred_text, iou_threshold=0.5, property_keys=None, allow_geometry_collection=False):
    gt_features = []
    if 'features' in entry:
        gt_features = entry['features']
    elif isinstance(entry.get('conv'), list):
        for turn in entry['conv']:
            answer = turn.get('Answer', '')
            gt_obj = _extract_geojson_object(answer)
            if gt_obj and 'features' in gt_obj:
                gt_features = gt_obj['features']
                break

    if not isinstance(gt_features, list):
        gt_features = []

    gt_feature_count = len(gt_features)

    if pred_text is None:
        return {
            'gt_feature_count': gt_feature_count,
            'pred_feature_count': 0,
            'matches': [],
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'arcgis_ready': False,
            'complete_match': False,
            'error': 'no_prediction',
        }

    pred_obj = _extract_geojson_object(pred_text)
    if pred_obj is None:
        return {
            'gt_feature_count': gt_feature_count,
            'pred_feature_count': 0,
            'matches': [],
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'arcgis_ready': False,
            'complete_match': False,
            'error': 'geojson_parse_failed',
        }

    pred_features = pred_obj.get('features') or []
    if not isinstance(pred_features, list):
        pred_features = []

    pred_feature_count = len(pred_features)

    arcgis_ready = (
        isinstance(pred_obj.get('type'), str)
        and pred_obj['type'] == 'FeatureCollection'
        and all(
            isinstance(f, dict) and isinstance(f.get('geometry'), dict)
            for f in pred_features
        )
    )

    matches, unmatched_pred, unmatched_gt = _match_features_by_property(
        pred_features, gt_features, property_keys, iou_threshold
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    complete_match = (gt_feature_count > 0 and tp == gt_feature_count and fp == 0 and fn == 0)

    return {
        'gt_feature_count': gt_feature_count,
        'pred_feature_count': pred_feature_count,
        'matches': [{'pred_idx': pi, 'gt_idx': gi, 'iou': iou} for pi, gi, iou in matches],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'arcgis_ready': arcgis_ready,
        'complete_match': complete_match,
        'error': None,
    }


def summarize_results(results):
    total = len(results)
    if total == 0:
        return {'total': 0}

    parse_failures = sum(1 for r in results if r.get('error') == 'geojson_parse_failed')
    no_predictions = sum(1 for r in results if r.get('error') == 'no_prediction')
    arcgis_ready = sum(1 for r in results if r.get('arcgis_ready'))
    complete_matches = sum(1 for r in results if r.get('complete_match'))

    valid_results = [r for r in results if r.get('error') is None]
    avg_precision = sum(r['precision'] for r in valid_results) / len(valid_results) if valid_results else 0.0
    avg_recall = sum(r['recall'] for r in valid_results) / len(valid_results) if valid_results else 0.0
    avg_f1 = sum(r['f1'] for r in valid_results) / len(valid_results) if valid_results else 0.0

    f1_values = sorted([r['f1'] for r in valid_results])

    def percentile(values, p):
        if not values:
            return 0.0
        idx = int(len(values) * p / 100.0)
        return values[min(idx, len(values) - 1)]

    return {
        'total': total,
        'valid_count': len(valid_results),
        'parse_failure_count': parse_failures,
        'no_prediction_count': no_predictions,
        'arcgis_ready_count': arcgis_ready,
        'arcgis_ready_rate': arcgis_ready / total if total > 0 else 0.0,
        'complete_match_count': complete_matches,
        'sample_complete_match_rate': complete_matches / total if total > 0 else 0.0,
        'avg_f1': avg_f1,
        'f1_iou': avg_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'f1_p50': percentile(f1_values, 50),
        'f1_p75': percentile(f1_values, 75),
        'f1_p90': percentile(f1_values, 90),
        'f1_min': min(f1_values) if f1_values else 0.0,
        'f1_max': max(f1_values) if f1_values else 0.0,
    }
