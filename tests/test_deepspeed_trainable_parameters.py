import ast
from pathlib import Path


def _find_deepspeed_initialize(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "initialize"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "deepspeed"
        ):
            return node
    raise AssertionError("deepspeed.initialize call not found")


def test_deepspeed_initialize_receives_explicit_trainable_parameters():
    source = Path(__file__).resolve().parents[1] / "train_stage_two.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    call = _find_deepspeed_initialize(tree)

    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert "model_parameters" in kwargs, "DeepSpeed must receive trainable params explicitly"
    value = kwargs["model_parameters"]

    assert not (
        isinstance(value, ast.Constant) and value.value is None
    ), "model_parameters=None can omit PEFT LoRA params from the optimizer"

    if isinstance(value, ast.IfExp):
        branches_are_none = all(
            isinstance(branch, ast.Constant) and branch.value is None
            for branch in (value.body, value.orelse)
        )
        assert not branches_are_none, "model_parameters cannot resolve to None on every branch"


if __name__ == "__main__":
    test_deepspeed_initialize_receives_explicit_trainable_parameters()
