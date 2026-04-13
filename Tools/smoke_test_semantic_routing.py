import importlib.machinery
import os
import sys
import types
from types import SimpleNamespace

import ml_collections
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _mock_optional_modules() -> None:
    # Keep this script runnable on environments without training-only deps.
    for name in [
        "deepspeed",
        "deepspeed.ops",
        "deepspeed.ops.adam",
        "deepspeed.utils",
        "deepspeed.runtime",
        "torch_npu",
    ]:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = module


class DummyTokenizer:
    pad_token_id = 0
    model_max_length = 128

    def __call__(self, texts, padding="max_length", truncation=True, max_length=16, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        batch = len(texts)
        input_ids = torch.zeros(batch, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch, max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            n = min(max_length, max(1, min(8, len(text))))
            input_ids[i, :n] = torch.arange(1, n + 1)
            attention_mask[i, :n] = 1
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


def run() -> None:
    _mock_optional_modules()

    from Dataset.cap_dataset import DataCollatorForSupervisedDataset
    from Models.embedding_model_r1 import EmbeddingModel
    from Models.moe_seg import AquacultureSegMOE

    # 1) DataCollator
    tokenizer = DummyTokenizer()
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        physical_prompt_max_len=12,
        task_text_max_len=6,
        element_text_max_len=6,
    )
    instances = [
        {
            "text": {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            "rgb": torch.randn(3, 16, 16),
            "valid_image": True,
            "tsm": torch.randn(16, 16),
            "mask": torch.ones(16, 16),
            "task_text": "视觉定位",
            "element_text": "网箱养殖区",
            "physical_prompt": "[Sensor: JL-1] [GSD:0.5m]",
            "task_id": 2,
            "category_id": 1,
        },
        {
            "text": {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
            "rgb": torch.randn(3, 16, 16),
            "valid_image": True,
            "tsm": None,
            "mask": None,
            "task_text": "描述",
            "element_text": "无",
            "physical_prompt": "",
            "task_id": 3,
            "category_id": 0,
        },
    ]
    batch = collator(instances)
    assert batch["task_text_ids"].shape == (2, 6)
    assert batch["element_text_ids"].shape == (2, 6)
    assert batch["physical_prompt_ids"].shape == (2, 12)
    assert batch["tsm"].shape[0] == 2 and batch["mask"].shape[0] == 2
    assert batch["valid_physics"].dtype == torch.bool and batch["valid_physics"].shape[0] == 2

    # 2) EmbeddingModel
    cfg = ml_collections.ConfigDict()
    cfg.adjust_norm = False
    cfg.dtype = "float32"
    cfg.alignment_dim = 32
    cfg.use_checkpoint = False
    cfg.text = ml_collections.ConfigDict()
    cfg.text.hidden_size = 64
    cfg.rgb_vision = ml_collections.ConfigDict()
    cfg.rgb_vision.attn_pooler = ml_collections.ConfigDict()
    cfg.rgb_vision.attn_pooler.num_query = 144
    cfg.rgb_vision.attn_pooler.num_layers = 1
    cfg.rgb_vision.attn_pooler.num_attn_heads = 4
    cfg.moe_proj = ml_collections.ConfigDict()
    cfg.moe_proj.num_experts = 4
    cfg.moe_proj.task_dim = 16
    cfg.moe_proj.include_physical_prompt = True
    cfg.moe_proj.top_k = 2
    cfg.moe_proj.routing_strategy = "joint"
    cfg.physics = ml_collections.ConfigDict()
    cfg.physics.prompt_enabled = True
    cfg.physics.prompt_in_channels = 1
    cfg.physics.prompt_pool_sizes = [1, 2]

    model = EmbeddingModel(cfg)
    bsz = 2
    image_embedding = torch.randn(bsz, 20, 32)
    data = {
        "task_text_embs": torch.randn(bsz, 6, 64),
        "element_text_embs": torch.randn(bsz, 6, 64),
        "task_text_attention_mask": batch["task_text_attention_mask"],
        "element_text_attention_mask": batch["element_text_attention_mask"],
        "physical_prompt_embs": torch.randn(bsz, 12, 64),
        "physical_prompt_attention_mask": batch["physical_prompt_attention_mask"],
    }
    proj = model(data, image_embedding=image_embedding)
    assert proj.shape == (bsz, 144, 64)

    # 3) Seg branch
    seg = AquacultureSegMOE(
        in_channels=32,
        num_classes=2,
        include_context=True,
        adapter_cfg={"enable": False},
        top_k=2,
        text_embed_dim=64,
    )
    x = torch.randn(bsz, 32, 8, 8)
    seg_out = seg(
        x,
        input_size=(16, 16),
        task_text_embs=data["task_text_embs"],
        element_text_embs=data["element_text_embs"],
        task_text_mask=data["task_text_attention_mask"],
        element_text_mask=data["element_text_attention_mask"],
    )
    assert seg_out["logits"].shape == (bsz, 2, 16, 16)
    assert "task_bias_norm" in seg_out and "element_bias_norm" in seg_out

    print("SMOKE_TEST_PASS")
    print("proj shape:", tuple(proj.shape))
    print("seg logits shape:", tuple(seg_out["logits"].shape))


if __name__ == "__main__":
    run()
