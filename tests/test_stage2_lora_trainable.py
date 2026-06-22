import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.coastgpt import CoastGPT


class _FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 2))
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = _FakeTextEncoder()

    def get_text_encoder(self):
        return self.text_encoder


class _FakePeftWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class _FakeCoastGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(2, 2)
        self.language = _FakeLanguage()
        self.multimodal = nn.Linear(2, 2)
        self.physics_enabled = False
        self.wavelet_adapter_enabled = False
        self.wavelet_adapter = None


def test_stage2_prepare_training_reenables_loaded_lora_adapters():
    model = _FakeCoastGPT()

    CoastGPT.prepare_for_training(
        model,
        freeze_vision=True,
        freeze_text=False,
        tune_multimodal=True,
        compute_dtype=torch.bfloat16,
    )

    assert model.language.text_encoder.lora_A.requires_grad is True
    assert model.language.text_encoder.lora_B.requires_grad is True
    assert model.language.text_encoder.embed_tokens.weight.requires_grad is False
    assert model.language.text_encoder.lm_head.weight.requires_grad is False


def test_text_lora_checkpoint_load_unwraps_existing_peft_adapters(monkeypatch):
    import Models.coastgpt as coastgpt_module

    monkeypatch.setattr(coastgpt_module, "PeftModel", _FakePeftWrapper)
    base = _FakeTextEncoder()
    wrapped = _FakePeftWrapper(_FakePeftWrapper(base))

    assert CoastGPT._unwrap_text_lora_base_encoder(wrapped) is base


if __name__ == "__main__":
    test_stage2_prepare_training_reenables_loaded_lora_adapters()
