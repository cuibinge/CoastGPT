import torch

from Inference import _load_checkpoint
from Models.coastgpt import CoastGPT


class _FakeTextEncoder:
    def __init__(self, vocab_size=32000, hidden_size=4):
        self.input_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


class _FakeLanguage:
    def __init__(self, text_encoder):
        self.text_encoder = text_encoder

    def get_text_encoder(self):
        return self.text_encoder


def test_restore_old_checkpoint_text_weights_without_resizing_to_bin_vocab():
    model = CoastGPT.__new__(CoastGPT)
    text_encoder = _FakeTextEncoder()
    model.language = _FakeLanguage(text_encoder)

    embed_weight = torch.arange(33001 * 4, dtype=torch.float32).reshape(33001, 4)
    lm_head_weight = embed_weight + 7
    ckpt = {
        "other_ckpt": {
            "embed_tokens": {"weight": embed_weight},
            "lm_head": {"weight": lm_head_weight},
        }
    }

    model._restore_embed_tokens_from_ckpt(ckpt, lambda *_args: None)

    assert text_encoder.input_embeddings.weight.shape == (32000, 4)
    assert text_encoder.output_embeddings.weight.shape == (32000, 4)
    assert torch.equal(text_encoder.input_embeddings.weight, embed_weight[:32000])
    assert torch.equal(text_encoder.output_embeddings.weight, lm_head_weight[:32000])


class _FakeLoadable:
    def load_state_dict(self, *_args, **_kwargs):
        return None


class _FakeModelForCheckpointLoad:
    def __init__(self):
        self.vision = _FakeLoadable()
        self.multimodal = type("_FakeMultimodal", (), {"projection": _FakeLoadable()})()
        self.restore_called = False

    def _restore_embed_tokens_from_ckpt(self, *_args, **_kwargs):
        self.restore_called = True


def test_skip_text_lora_structured_loader_still_restores_text_weights(tmp_path):
    ckpt_path = tmp_path / "structured.pt"
    torch.save(
        {
            "vision_ckpt": {},
            "other_ckpt": {
                "multimodal_projection": {},
                "embed_tokens": {"weight": torch.zeros(33001, 4)},
                "lm_head": {"weight": torch.zeros(33001, 4)},
            },
        },
        ckpt_path,
    )
    model = _FakeModelForCheckpointLoad()

    _load_checkpoint(model, str(ckpt_path), skip_text_lora=True)

    assert model.restore_called
