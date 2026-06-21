import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from transformers import CLIPImageProcessor

from . import conversation as conversation_lib
from .cap_dataset import preprocess, preprocess_multimodal
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


class Compose(T.Compose):
    """
    genericgeneric Compose genericㄤgenericㄧgeneric?
    genericョgenericфgeneric?torchvision.transforms.Composegeneric?
    genericgenericㄤgenericgeneric″genericㄦgeneric?
    """

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, x: Union[Any, Sequence]):
        if isinstance(x, Sequence):
            for t in self.transforms:
                x = [t(i) for i in x]
        else:
            for t in self.transforms:
                x = t(x)
        return x


class ToTensor(object):
    """
    genericgeneric ToTensor generic?- generic€genericуgeneric€generic€?
    genericgeneric?uint16 genericgeneric﹂『generic€?
    """

    def __init__(self, permute_dims: bool = True):
        self.permute_dims = permute_dims

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        if x.dtype == "uint16":
            x = x.astype("int32")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim == 2:
            if self.permute_dims:
                x = x[:, :, None]
            else:
                x = x[None, :, :]

        if x.ndim == 3 and x.shape[2] == 4:
            if self.permute_dims:
                x = x.permute((2, 0, 1)).contiguous()

        if self.permute_dims:
            if x.ndim == 4:
                x = x.permute((0, 3, 1, 2)).contiguous()
            elif x.ndim == 3 and x.shape[2] != 4:
                x = x.permute((2, 0, 1)).contiguous()

        return x


def sort(x):
    """
    generic?
    genericㄤgeneric?
    """
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return int(x)


class RSVQA(torch.utils.data.Dataset):
    """
    genericgeneric?RSVQA generic?
    genericㄤgenericユgenericgeneric°€?
    """

    splits = ["train", "val", "test"]
    prefix = ""

    def __init__(
        self,
        root: str = "",
        image_root: str = None,
        split: str = "train",
        image_transform: Compose = Compose([ToTensor()]),
        text_transform: Compose = Compose(
            [],
        ),
        token_prefix: str = "",
        tokenizer: Callable = None,
        **kwargs,
    ):
        assert split in self.splits
        prompt_type = kwargs.pop("prompt_type", "llava_llama_2")
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            prompt_type
        ]

        self.root = root
        self.split = split
        self.image_root = image_root
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.image_root = os.path.join(root, self.image_root)
        self.token_prefix = token_prefix
        self.tune_im_start = kwargs.pop("tune_im_start", False)
        self.tokenizer = tokenizer

        (
            self.ids,
            self.paths,
            self.images,
            self.questions,
            self.answers,
        ) = self.load_files(self.root, self.image_root, self.split, self.prefix)

        self.post_process()

    @staticmethod
    def load_files(
        root: str, image_root: str, split: str, prefix: str
    ) -> Tuple[List[int], List[str], List[Dict], List[Dict], List[Dict]]:
        """
        generic€genericㄤgeneric€generic℃generic?
        :param root: genericgeneric
        :param image_root: generic?
        :param split: generic?
        :param prefix: generic
        :return: generic ID generic〃genericㄣ€genericgenericㄣ€genericgenericㄣ€genericgeneric?
        """
        paths = glob(os.path.join(image_root, "*.tif"))
        paths = sorted(paths, key=sort)
        with open(os.path.join(root, f"{prefix}_split_{split}_questions.json")) as f:
            questions = json.load(f)["questions"]
        with open(os.path.join(root, f"{prefix}_split_{split}_answers.json")) as f:
            answers = json.load(f)["answers"]
        with open(os.path.join(root, f"{prefix}_split_{split}_images.json")) as f:
            images = json.load(f)["images"]
        ids = [x["id"] for x in images if x["active"]]
        return ids, paths, images, questions, answers

    def post_process(self):
        """
        genericgeneric "count" generic?"area" generic€?
        """
        neglect_question_type = ["count", "area"]

        new_questions = []
        new_ids = []
        for id in self.ids:
            questions_ids = self.images[id]["questions_ids"]
            valid_questions_ids = [
                i
                for i in questions_ids
                if self.questions[i]["type"].lower() not in neglect_question_type
            ]
            new_questions.extend(valid_questions_ids)
            new_ids.extend([id] * len(valid_questions_ids))

        self.questions_ids = new_questions
        self.ids = new_ids

    def __len__(self) -> int:
        """
        generic?
        """
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        generic㈠generic?
        :param idx: generic㈠generic
        :return: generic€generic€generic℃generic?
        """
        id = self.ids[idx]
        x = np.array(Image.open(os.path.join(self.image_root, f"{id}.tif")))
        if isinstance(self.image_transform, CLIPImageProcessor):
            x = self.image_transform(x, return_tensors="pt").pixel_values.squeeze()
        else:
            x = self.image_transform(x)
        questions = self.questions[self.questions_ids[idx]]
        answers = self.answers[questions["answers_ids"][0]]["answer"]
        types = questions["type"]
        questions = questions["question"]
        questions = self.text_transform(questions)
        answers = self.text_transform(answers)

        questions = self.token_prefix + questions

        item = dict(Question=questions, Answer=None)
        questions = preprocess_multimodal(item, tune_im_start=self.tune_im_start)
        questions = preprocess(questions, self.tokenizer, has_image=True)
        questions = questions["input_ids"][0]

        output = dict(
            x=x,
            question=questions,
            answer=answers,
            type=types,
            questions_idx=self.questions_ids[idx],
        )
        return output


class RSVQALR(RSVQA):
    """
    RSVQALR generic RSVQAgeneric?
    generic?LR genericgeneric?
    """
    prefix = "LR"

    def __init__(self, root: str = ".data/RSVQA_LR", *args, **kwargs):
        super().__init__(root, *args, **kwargs)


class RSVQAHR(RSVQA):
    """
    RSVQAHR generic RSVQAgeneric?
    generic?USGS genericgeneric?
    """
    prefix = "USGS"

    def __init__(self, root: str = ".data/RSVQA_HR", *args, **kwargs):
        super().__init__(root, *args, **kwargs)


class RSVQAxBEN(RSVQA):
    """
    RSVQAxBEN generic RSVQAgeneric?
    generic?RSVQAxBEN genericgeneric?
    """
    prefix = "RSVQAxBEN"

    def __init__(self, root: str = ".data/rsvqaxben", *args, **kwargs):
        super().__init__(root, *args, **kwargs)


@dataclass
class DataCollatorForVQASupervisedDataset(object):
    """
    genericㄤgenericgenericgeneric€?
    genericgenericgenericgeneric€genericgeneric°€?
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Tuple]) -> Tuple:
        """
        generic€genericgeneric°€?
        :param instances: generic
        :return: generic€generic╃generic€generic€generic
        """
        input_ids = tuple([instance["question"] for instance in instances])
        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)

        def left_pad_sequences(sequences, desired_length, padding_value):
            """
            genericgeneric﹁generic︺€?
            :param sequences: generic
            :param desired_length: generic?
            :param padding_value: genericgeneric?
            :return: genericgeneric
            """
            padded_sequences = tuple(
                [padding_value] * (desired_length - len(seq)) + list(seq)
                for seq in sequences
            )
            return padded_sequences

        input_ids = left_pad_sequences(
            input_ids, max_length, self.tokenizer.pad_token_id
        )
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        images = [instance["x"] for instance in instances]
        if not isinstance(images[0], Image.Image) and all(
            x is not None and x.shape == images[0].shape for x in images
        ):
            images = torch.stack(images)
        else:
            images = images

        targets = [instance["answer"] for instance in instances]
        type = [instance["type"] for instance in instances]
        questions_idx = [instance["questions_idx"] for instance in instances]

        out = dict(
            images=images,
            questions=input_ids,
            attn_mask=attention_mask,
            targets=targets,
            types=type,
            questions_idx=questions_idx,
        )

        return out
