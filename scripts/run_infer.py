import argparse, json, os, sys
from pathlib import Path
import ml_collections, torch, yaml

os.environ.setdefault('HF_HOME', '/home/ma-user/work/CoastGPT/hf_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')

ROOT = Path('/home/ma-user/work/CoastGPT')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from Dataset.build_transform import build_vlp_transform
from Dataset.conversation import default_conversation
from Models import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, tokenizer_image_token
from Models.coastgpt import CoastGPT
from Models.utils import type_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--prompt', default='检测这张遥感影像中的海岸带地物，输出GeoJSON格式的标注结果。')
    p.add_argument('--output', default='./infer_results.json')
    p.add_argument('--sample-id', default=None)
    p.add_argument('--config', default='Configs/step3_dual.yaml')
    p.add_argument('--model-path', default='./output/stage3/mixed_v3/checkpoints/FINAL.pt')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--debug', action='store_true')
    return p.parse_args()


def load_model(config_path, model_path):
    with open(config_path) as f:
        cfg = ml_collections.ConfigDict(yaml.safe_load(f))

    # NPU safe mode
    if getattr(cfg, 'bits', 16) in (4, 8):
        print(f'[init] NPU safe: bits={cfg.bits} -> 16')
        cfg.bits = 16
    cfg.stage = 0

    dtype = type_dict.get(cfg.get('dtype', 'bfloat16'), torch.bfloat16)
    model = CoastGPT(cfg)
    model.to(dtype)

    ckpt = torch.load(model_path, map_location='cpu')
    if 'vision_ckpt' in ckpt or 'other_ckpt' in ckpt:
        msg = model.custom_load_state_dict(model_path)
    elif 'model' in ckpt:
        msg = model.load_state_dict(ckpt['model'], strict=False)
    else:
        msg = model.load_state_dict(ckpt, strict=False)
    print(f'[load] {msg}')

    model.to('npu')
    model.eval()
    return model, cfg


def run(model, cfg, image_path, prompt_text, max_tokens):
    from Dataset.cap_dataset import load_image_as_rgb

    image = load_image_as_rgb(image_path)
    transform = build_vlp_transform(cfg, is_train=False)
    image_tensor = transform(image).unsqueeze(0).to(torch.bfloat16).to('npu')

    conv = default_conversation.copy()
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    tokenizer = model.language.tokenizer
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('npu')

    print(f'[infer] prompt_tokens={input_ids.shape[1]}')

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f'[infer] generated_tokens={new_tokens.shape[0]}')
    print(text[:500])

    result = {'prompt': prompt_text, 'image': image_path, 'raw_output': text}
    try:
        result['geojson'] = json.loads(text)
    except json.JSONDecodeError:
        repair = text + ']' * max(0, text.count('[') - text.count(']'))
        repair += '}' * max(0, repair.count('{') - repair.count('}'))
        try:
            result['geojson'] = json.loads(repair)
            result['repaired'] = True
        except json.JSONDecodeError:
            result['geojson'] = None
    return result


def save(result, output_path, sample_id):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.load(open(output_path)) if output_path.exists() else {'predictions': []}
    if sample_id:
        result['sample_id'] = sample_id
    existing['predictions'].append(result)
    json.dump(existing, open(output_path, 'w'), ensure_ascii=False, indent=2)
    print(f'[save] {output_path}')


def main():
    args = parse_args()
    print(f'[init] model={args.model_path}')
    model, cfg = load_model(args.config, args.model_path)
    result = run(model, cfg, args.image, args.prompt, args.max_new_tokens)
    save(result, args.output, args.sample_id or Path(args.image).stem)
    print('[done]')


if __name__ == '__main__':
    main()
