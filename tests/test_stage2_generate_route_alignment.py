import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.coastgpt import CoastGPT
from Models.embedding_model_r1 import EmbeddingModel
from Tools.run_geojson_batch_eval import _build_semantic_route_inputs, _ensure_image_token


class RecordingProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["image_embs"]


def test_encode_test_forwards_physical_prompts_and_route_masks():
    model = EmbeddingModel.__new__(EmbeddingModel)
    nn.Module.__init__(model)
    model.projection = RecordingProjection()

    image_embedding = torch.ones(1, 2, 4, dtype=torch.float32)
    physical_prompts = torch.full((1, 3, 4), 2.0, dtype=torch.float16)
    task_text_embs = torch.full((1, 2, 4), 3.0, dtype=torch.float16)
    element_text_embs = torch.full((1, 2, 4), 4.0, dtype=torch.float16)
    physical_mask = torch.tensor([[1, 1, 0]])
    task_mask = torch.tensor([[1, 0]])
    element_mask = torch.tensor([[1, 1]])

    out = model.encode_test(
        image_embedding,
        physical_prompts=physical_prompts,
        task_text_embs=task_text_embs,
        element_text_embs=element_text_embs,
        physical_prompt_mask=physical_mask,
        task_text_mask=task_mask,
        element_text_mask=element_mask,
    )

    assert out is image_embedding
    assert model.projection.kwargs["physical_prompts"].dtype == image_embedding.dtype
    assert torch.equal(model.projection.kwargs["physical_prompt_mask"], physical_mask)
    assert torch.equal(model.projection.kwargs["task_text_mask"], task_mask)
    assert torch.equal(model.projection.kwargs["element_text_mask"], element_mask)


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embeddings


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TinyTextEncoder()
        self.generate_kwargs = None

    def get_text_encoder(self):
        return self.text_encoder

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


def test_generate_embeds_and_passes_physical_prompt_to_image_encoder():
    model = CoastGPT.__new__(CoastGPT)
    nn.Module.__init__(model)
    model.language = TinyLanguage()

    captured = {}

    def fake_encode_image(image, pool, **kwargs):
        captured.update(kwargs)
        return torch.ones(1, 2, 4)

    model.encode_image = fake_encode_image

    input_ids = torch.tensor([[1, 2]])
    images = torch.ones(1, 3, 8, 8)
    physical_prompt_ids = torch.tensor([[3, 4, 0]])
    physical_prompt_mask = torch.tensor([[1, 1, 0]])
    task_text_ids = torch.tensor([[5, 0]])
    task_text_mask = torch.tensor([[1, 0]])
    element_text_ids = torch.tensor([[6, 7]])
    element_text_mask = torch.tensor([[1, 1]])

    model.generate(
        input_ids=input_ids,
        images=images,
        do_sample=False,
        physical_prompt_ids=physical_prompt_ids,
        physical_prompt_attention_mask=physical_prompt_mask,
        task_text_ids=task_text_ids,
        task_text_attention_mask=task_text_mask,
        element_text_ids=element_text_ids,
        element_text_attention_mask=element_text_mask,
    )

    assert captured["physical_prompt_embs"].shape == (1, 3, 4)
    assert torch.equal(captured["physical_prompt_attention_mask"], physical_prompt_mask)
    assert torch.equal(captured["task_text_attention_mask"], task_text_mask)
    assert torch.equal(captured["element_text_attention_mask"], element_text_mask)
    assert "physical_prompt_ids" not in model.language.generate_kwargs


class TinyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=None,
        **kwargs,
    ):
        assert return_tensors == "pt"
        self.calls.append(
            {
                "text": texts[0],
                "padding": padding,
                "truncation": truncation,
                "add_special_tokens": add_special_tokens,
                "max_length": max_length,
            }
        )
        text = texts[0]
        token_value = max(1, min(len(text), 15))
        length = int(max_length) if padding == "max_length" and max_length else 2
        ids = [1, token_value] + [2] * max(0, length - 2)
        mask = [1, 1] + [0] * max(0, length - 2)
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([mask]),
        }


def test_route_inputs_include_physical_prompt_when_available():
    tokenizer = TinyTokenizer()
    route_inputs = _build_semantic_route_inputs(
        tokenizer,
        "Is it a rural or an urban area?",
        torch.device("cpu"),
        physical_prompt_text="[Dataset: LR]",
    )

    assert "physical_prompt_ids" in route_inputs
    assert "physical_prompt_attention_mask" in route_inputs
    assert route_inputs["physical_prompt_ids"].shape == (1, 64)
    assert route_inputs["task_text_ids"].shape == (1, 16)
    assert route_inputs["element_text_ids"].shape == (1, 16)
    assert torch.equal(
        route_inputs["physical_prompt_attention_mask"][0, :4],
        torch.tensor([1, 1, 0, 0]),
    )
    assert tokenizer.calls[0]["padding"] == "max_length"
    assert tokenizer.calls[0]["max_length"] == 16
    assert tokenizer.calls[0]["add_special_tokens"] is True
    assert tokenizer.calls[2]["padding"] == "max_length"
    assert tokenizer.calls[2]["max_length"] == 64
    assert tokenizer.calls[2]["add_special_tokens"] is True


def test_ensure_image_token_normalizes_existing_image_token_with_newline():
    normalized = _ensure_image_token(
        "<image>Is it a rural or an urban area",
        tune_im_start=False,
        default_image_token="<image>",
        default_im_start_token="<im_start>",
        default_im_end_token="<im_end>",
    )

    assert normalized == "<image>\nIs it a rural or an urban area"
