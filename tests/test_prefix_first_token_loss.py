import sys
from pathlib import Path

import torch
from ml_collections import ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from Models.coastgpt import CoastGPT
from Models.language_model import LanguageModel


class _FakeBackbone(torch.nn.Module):
    def __init__(self, vocab_size=16, hidden_size=1):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            self.embed_tokens.weight[:, 0] = torch.arange(vocab_size).float()


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self, vocab_size=16, hidden_size=1):
        super().__init__()
        self.model = _FakeBackbone(vocab_size=vocab_size, hidden_size=hidden_size)
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.vocab_size = vocab_size
        self.last_inputs_embeds = None

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        self.last_inputs_embeds = inputs_embeds.detach().clone()
        token_hint = inputs_embeds[..., 0].round().long() + 1
        token_hint = token_hint.clamp(min=0, max=self.vocab_size - 1)
        logits = inputs_embeds.new_full(
            (*inputs_embeds.shape[:2], self.vocab_size),
            -20.0,
        )
        logits.scatter_(-1, token_hint.unsqueeze(-1), 20.0)
        return {"logits": logits}


def _make_language_model():
    language = object.__new__(LanguageModel)
    torch.nn.Module.__init__(language)
    language.max_position_embeddings = 128
    language.tune_pooler = False
    language.tune_im_start = False
    language.text_encoder = _FakeTextEncoder()
    object.__setattr__(language, "get_text_encoder", lambda: language.text_encoder)
    return language


def test_prefix_first_token_loss_predicts_first_answer_from_prefix_only():
    language = _make_language_model()
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 5, 6, 7, 2]])
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 2]])
    data = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }
    image_embedding = torch.tensor([[[50.0], [51.0]]])

    loss, stats = language.prefix_first_token_loss(data, image_embedding=image_embedding)

    assert float(loss) < 1e-4
    assert int(stats["prefix_first_token_count"].item()) == 1
    # Prefix should stop before target token 7:
    # [1] + two image embeddings + [5, 6] => expanded length 5.
    assert language.text_encoder.last_inputs_embeds.shape[1] == 5
    assert language.text_encoder.last_inputs_embeds[0, -1, 0].item() == 6.0


def test_prefix_first_token_loss_returns_none_when_no_supervised_tokens():
    language = _make_language_model()
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 5]])
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    data = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }

    loss, stats = language.prefix_first_token_loss(data, image_embedding=torch.tensor([[[50.0]]]))

    assert loss is None
    assert int(stats["prefix_first_token_count"].item()) == 0


class _PrefixLossLanguage:
    def __init__(self):
        self.called = False

    def prefix_first_token_loss(self, data, image_embedding=None):
        self.called = True
        return torch.tensor(2.0), {
            "prefix_first_token_count": torch.tensor(3),
            "prefix_first_token_exact": torch.tensor(0.5),
        }


def test_coastgpt_adds_weighted_prefix_first_token_loss_when_enabled():
    model = object.__new__(CoastGPT)
    torch.nn.Module.__init__(model)
    model.config = ConfigDict({"prefix_first_token_loss": {"enabled": True, "weight": 0.25}})
    model.language = _PrefixLossLanguage()
    out = {}

    total = model._add_prefix_first_token_loss(
        data={},
        multimodal_embedding=torch.zeros(1, 2, 1),
        total_loss=torch.tensor(10.0),
        out=out,
    )

    assert model.language.called is True
    assert total.item() == 10.5
    assert out["prefix_first_token_loss"].item() == 2.0
    assert out["prefix_first_token_loss_weighted"].item() == 0.5
    assert out["prefix_first_token_count"].item() == 3


def test_coastgpt_skips_prefix_first_token_loss_by_default():
    model = object.__new__(CoastGPT)
    torch.nn.Module.__init__(model)
    model.config = ConfigDict({})
    model.language = _PrefixLossLanguage()

    total = model._add_prefix_first_token_loss(
        data={},
        multimodal_embedding=torch.zeros(1, 2, 1),
        total_loss=torch.tensor(10.0),
        out={},
    )

    assert model.language.called is False
    assert total.item() == 10.0
