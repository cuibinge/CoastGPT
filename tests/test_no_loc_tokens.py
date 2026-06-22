from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = ["Models", "Dataset", "Tools", "Configs", "configs", "scripts"]
ALLOWLIST = {Path("tests/test_no_loc_tokens.py")}


def iter_text_files():
    for dirname in SEARCH_DIRS:
        root = ROOT / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() or path.suffix in {".pyc", ".pth", ".pt"}:
                continue
            rel = path.relative_to(ROOT)
            if rel in ALLOWLIST or "__pycache__" in rel.parts or ".ipynb_checkpoints" in rel.parts:
                continue
            try:
                yield rel, path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue


def test_loc_token_vocabulary_support_is_removed():
    offenders = []
    for rel, text in iter_text_files():
        if "<loc_" in text or "loc_tokens" in text or "Models.loc_tokens" in text:
            offenders.append(str(rel))
    assert offenders == []
