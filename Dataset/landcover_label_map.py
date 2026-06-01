"""
Land cover label map: DLMC class name → canonical_id → train_id (1-24).

PoC-2: 24 active DLMC classes + background (0) + ignore (255).
Spec-only classes with no training data are marked active_in_poc2=false.
"""

from typing import Dict, List, Optional

IGNORE_INDEX = 255
BACKGROUND_ID = 0

# 24 active DLMC classes in training data, sorted by pinyin.
_DLMC_NAMES: List[str] = [
    "城镇村道路用地",
    "港口码头用地",
    "工业用地",
    "公路用地",
    "公园与绿地",
    "沟渠",
    "旱地",
    "河流水面",
    "坑塘水面",
    "裸岩石砾地",
    "内陆滩涂",
    "农村道路",
    "农村宅基地",
    "其他草地",
    "其他林地",
    "其他园地",
    "设施农用地",
    "水工建筑用地",
    "水浇地",
    "水田",
    "铁路用地",
    "沿海滩涂",
    "盐田",
    "养殖坑塘",
]

# Spec-defined classes NOT in current training data (reserved for future).
_SPEC_ONLY_NAMES: List[str] = [
    "果园",
    "茶园",
    "乔木林地",
    "灌木林地",
    "竹林地",
    "城镇住宅用地",
    "采矿用地",
]


def build_label_map() -> Dict[str, dict]:
    """Build the full label map keyed by DLMC class name.

    Returns a dict mapping class_name → {
        canonical_id, train_id, active_in_poc2, dir_name
    }.
    """
    label_map: Dict[str, dict] = {}

    # Active classes: train_id = 1..24
    for idx, name in enumerate(_DLMC_NAMES, start=1):
        label_map[name] = {
            "canonical_id": idx,
            "train_id": idx,
            "active_in_poc2": True,
        }

    # Spec-only classes: train_id = None, active_in_poc2 = False
    for idx, name in enumerate(_SPEC_ONLY_NAMES, start=len(_DLMC_NAMES) + 1):
        label_map[name] = {
            "canonical_id": idx,
            "train_id": None,
            "active_in_poc2": False,
        }

    return label_map


def get_active_train_ids(label_map: Dict[str, dict]) -> List[int]:
    """Return sorted list of active train_ids (1-24)."""
    return sorted(
        v["train_id"]
        for v in label_map.values()
        if v.get("train_id") is not None
    )


def dlmc_to_train_id(dlmc: str, label_map: Optional[Dict[str, dict]] = None) -> int:
    """Convert a DLMC class name to its PoC-2 train_id (1-24).

    Args:
        dlmc: DLMC class name string (e.g. "水田", "公路用地").
        label_map: Optional pre-built label_map. If None, builds a fresh copy.

    Returns:
        train_id in [1, 24].

    Raises:
        KeyError: if dlmc is not in the active class set.
    """
    if label_map is None:
        label_map = build_label_map()
    if dlmc not in label_map:
        raise KeyError(
            f"Unknown DLMC class '{dlmc}'. Known: {sorted(label_map.keys())}"
        )
    entry = label_map[dlmc]
    if not entry["active_in_poc2"]:
        raise KeyError(
            f"DLMC class '{dlmc}' is spec-only (active_in_poc2=false), "
            f"no train_id assigned"
        )
    return entry["train_id"]


def train_id_to_dlmc(train_id: int, label_map: Optional[Dict[str, dict]] = None) -> str:
    """Reverse lookup: train_id → DLMC class name."""
    if label_map is None:
        label_map = build_label_map()
    for name, entry in label_map.items():
        if entry.get("train_id") == train_id:
            return name
    raise KeyError(f"No DLMC class with train_id={train_id}")


_DEFAULT_LABEL_MAP = None


def get_label_map() -> Dict[str, dict]:
    """Return the shared label_map singleton."""
    global _DEFAULT_LABEL_MAP
    if _DEFAULT_LABEL_MAP is None:
        _DEFAULT_LABEL_MAP = build_label_map()
    return _DEFAULT_LABEL_MAP


def num_classes() -> int:
    """Return number of semantic head output channels (active classes + background)."""
    return len(_DLMC_NAMES) + 1  # 25


def active_class_names() -> List[str]:
    """Return the list of 24 active DLMC class names."""
    return list(_DLMC_NAMES)


# ---- Directory name → DLMC mapping (for tile grouping) ----
_DIR_TO_DLMC: Dict[str, str] = {
    "养殖池塘": "养殖坑塘",
    "农村、城市建筑用地": "农村宅基地",
    "工矿仓储用地": "工业用地",
    "裸地": "裸岩石砾地",
    "林地": "其他林地",
    "园地": "其他园地",
}


def dir_name_to_dlmc(dir_name: str) -> str:
    """Map a data directory name to the canonical DLMC class name."""
    return _DIR_TO_DLMC.get(dir_name, dir_name)


def dir_name_to_train_id(dir_name: str) -> int:
    """Map a data directory name directly to train_id."""
    return dlmc_to_train_id(dir_name_to_dlmc(dir_name))


if __name__ == "__main__":
    lm = build_label_map()
    active = get_active_train_ids(lm)
    print(f"Active classes: {len(active)} (train_ids {active[0]}-{active[-1]})")
    print(f"Spec-only classes: {len(_SPEC_ONLY_NAMES)}")
    print(f"Semantic head channels: {num_classes()} (24 + bg)")

    # Verify continuity
    assert active == list(range(1, 25)), f"train_ids not 1-24: {active}"
    assert dlmc_to_train_id("水田") > 0
    assert dlmc_to_train_id("沟渠") > 0
    assert dir_name_to_train_id("养殖池塘") == dlmc_to_train_id("养殖坑塘")
    assert dir_name_to_train_id("裸地") == dlmc_to_train_id("裸岩石砾地")

    # Spec-only class has no train_id
    try:
        dlmc_to_train_id("果园")
        assert False, "Should raise"
    except KeyError:
        pass

    # Reverse lookup
    for tid in [1, 12, 24]:
        name = train_id_to_dlmc(tid)
        assert dlmc_to_train_id(name) == tid, f"Round-trip failed for {tid}: {name}"

    print("All label_map checks passed.")
