import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset import conversation as conversation_lib
from Dataset.cap_dataset import IGNORE_INDEX, preprocess, preprocess_multimodal


def test_llama2_image_preprocess_keeps_answer_labels_when_pad_is_eos():
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llama_2"]
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    sources = preprocess_multimodal(
        [{"Question": "<image>Is it a rural or an urban area", "Answer": "rural"}]
    )

    output = preprocess(sources, tokenizer, has_image=True)
    label_ids = [
        int(token_id)
        for token_id in output["labels"][0].tolist()
        if int(token_id) != IGNORE_INDEX
    ]

    assert label_ids, "answer labels should not be fully masked"
    assert "rural" in tokenizer.decode(label_ids).lower()
