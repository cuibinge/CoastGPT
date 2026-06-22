import importlib.util
from pathlib import Path

import torch


def _load_history_buffer():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "Trainer" / "utils" / "history_buffer.py"
    spec = importlib.util.spec_from_file_location("history_buffer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HistoryBuffer


def test_tensor_metric_values_are_stored_as_python_floats():
    HistoryBuffer = _load_history_buffer()
    history = HistoryBuffer()

    history.update(torch.tensor(2.5))

    assert isinstance(history.latest, float)
    assert isinstance(history.global_sum, float)
    assert history.latest == 2.5
    assert history.global_sum == 2.5


if __name__ == "__main__":
    test_tensor_metric_values_are_stored_as_python_floats()
