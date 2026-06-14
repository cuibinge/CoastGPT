"""一键上传 landcover + aquaculture 模型到 HuggingFace.

在本地机器执行:
  git clone https://github.com/cuibinge/CoastGPT.git -b CoastGPT_dual --depth 1
  cd CoastGPT
  pip install huggingface_hub
  python upload_hf.py
"""

from huggingface_hub import HfApi, create_repo, upload_folder

TOKEN = "YOUR_HF_TOKEN_HERE"  # 替换为你的 Hugging Face token

MODELS = [
    {
        "repo": "cuibinge/coastgpt-landcover-semantic",
        "folder": "hf_upload/landcover",
        "msg": "Upload CoastGPT Landcover Semantic Head (FPN + 25-class decoder, mIoU 71.4%)",
    },
    {
        "repo": "cuibinge/coastgpt-aquaculture-det",
        "folder": "hf_upload/aquaculture",
        "msg": "Upload CoastGPT Aquaculture Detection Head (FPN + Mask R-CNN with RPN + ROI)",
    },
]

for m in MODELS:
    print(f"Uploading to {m['repo']}...")
    create_repo(m["repo"], token=TOKEN, exist_ok=True)
    upload_folder(
        folder_path=m["folder"],
        repo_id=m["repo"],
        repo_type="model",
        token=TOKEN,
        commit_message=m["msg"],
    )
    print(f"  ✓ https://huggingface.co/{m['repo']}")

print("\nAll done!")
