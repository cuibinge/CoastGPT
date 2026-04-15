from io import BytesIO

import ml_collections
import requests
import torch
import transformers
from PIL import Image
from transformers import TextStreamer

from Trainer.utils import ConfigArgumentParser, str2bool
from Dataset.build_transform import build_vlp_transform
from Dataset.conversation import SeparatorStyle, default_conversation
from Models import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    tokenizer_image_token,
)
from Models.coastgpt import CoastGPT
from Models.utils import KeywordsStoppingCriteria, type_dict

# 加载图像的函数，可以从本地路径或URL加载图像，并将其转换为RGB格式。
def load_image(image_file)->Image.Image:
    """Load an image from a local path or a URL."""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def parse_option():
    parser = ConfigArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # basic
    parser.add_argument("--image-file", type=str, default="Images/00017.jpg", help="path to image")
    parser.add_argument(
        "--model-path",
        type=str,
        # default="Checkpoint/testlhrs/checkpoints/iter_999/mp_rank_00_model_states.pt",
        # default="Checkpoint/testlhrs/checkpoints/iter_9999/mp_rank_00_model_states.pt",
        # default="Checkpoint/test2/checkpoints/iter_1299/mp_rank_00_model_states.pt",
        # default="Checkpoint/LHRS/Stage2/FINAL.pt",
        default="/root/shared-nvme/CoastGPT/Checkpoint/test/checkpoints/FINAL.pt",
        help="pretrained checkpoint path for vision encoder",
    )
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")

    # HardWare
    parser.add_argument(
        "--accelerator",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )
    parser.add_argument("--use-checkpoint", default=False, type=str2bool)

    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main(config: ml_collections.ConfigDict):
    
    # 构建CoastGPT模型，传入配置对象作为参数。
    model = CoastGPT(config)
    
    # 如果是huggingface模型，使用模型自带的processor，否则使用默认的transform
    if getattr(config, "hf_model", False):             
        vision_processor = model.get_image_processor()
    else:
        vision_processor = build_vlp_transform(config, is_train=False)
    
    # 将模型参数转换为指定的数据类型，以节省内存和加速推理。数据类型由配置中的dtype字段指定，可以是float32、float16或bfloat16。
    dtype = type_dict[config.dtype]       
    model.to(dtype)
    
    # 创建一个对话对象，用于管理对话的状态和格式。对话对象是从默认对话模板复制而来的，包含了角色信息和消息列表。
    conv = default_conversation.copy()
    roles = conv.roles

    # 加载预训练模型权重和分词器。
    if config.model_path is not None:
        if getattr(config, "hf_model", False):
            msg = model.custom_load_state_dict(config.model_path, strict=False)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.path, use_fast=False
            )
        else:
            if hasattr(model, "custom_load_state_dict"):
                msg = model.custom_load_state_dict(config.model_path)
            else:
                ckpt = torch.load(config.model_path, map_location="cpu")
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                msg = model.load_state_dict(ckpt, strict=False)
                del ckpt
            tokenizer = model.language.tokenizer
        print(msg)

    # 将模型移动到设备上进行推理
    if config.accelerator == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)

    # 加载图像,并使用视觉处理器将图像转换为张量，以便输入到模型中进行推理。
    if config.image_file is not None:
        image = load_image(config.image_file)
        if config.rgb_vision.arch.startswith("vit"):
            image_tensor = (
                vision_processor(image, return_tensors="pt")
                .pixel_values.to(device)
                .to(dtype)
            )
        else:
            image_tensor = vision_processor(image).to(dtype).to(device).unsqueeze(0)
    else:
        image = None
        image_tensor = None

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if config.tune_im_start:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                max_new_tokens=512,
                temperature=0.4,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0]).strip()
        outputs = outputs.split("<s>")[-1].strip()  # remove <s>
        conv.messages[-1][-1] = outputs

        if config.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    config = parse_option()
    config.adjust_norm = False
    main(config)
