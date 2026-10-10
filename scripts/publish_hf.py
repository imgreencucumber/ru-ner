"""Publish the model (PyTorch weights, ONNX int8, model card) to the Hugging Face Hub.

Files are first assembled in outputs/hub/, the model card template hub/model_card.md gets the real
repo name. Authentication: `uv run hf auth login`, or HF_TOKEN (a write token) in the environment
or in .env.

Usage: uv run python scripts/publish_hf.py --dry-run    # assemble and list files only
       uv run python scripts/publish_hf.py
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

# HF_TOKEN from .env, if present; huggingface_hub reads it from the environment
load_dotenv()

MODEL_DIR = Path("models/rubert-ner")
STAGING = Path("outputs/hub/model")
CARD_TEMPLATE = Path("hub/model_card.md")
GITHUB_URL = "https://github.com/imgreencucumber/ru-ner"

FILES = {
    "config.json": MODEL_DIR / "config.json",
    "model.safetensors": MODEL_DIR / "model.safetensors",
    "tokenizer.json": MODEL_DIR / "tokenizer.json",
    "tokenizer_config.json": MODEL_DIR / "tokenizer_config.json",
    "onnx/model_int8.onnx": MODEL_DIR / "onnx" / "model_int8.onnx",
    "onnx/tokenizer.json": MODEL_DIR / "onnx" / "tokenizer.json",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", help="HF username, by default taken from the login")
    p.add_argument("--name", default="rubert-ner-collection3")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api = HfApi()
    repo_id = f"{args.user or api.whoami()['name']}/{args.name}"
    git_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    shutil.rmtree(STAGING, ignore_errors=True)
    for target, source in FILES.items():
        (STAGING / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, STAGING / target)
    card = CARD_TEMPLATE.read_text(encoding="utf-8")
    card = card.replace("{repo_id}", repo_id).replace("{github_url}", GITHUB_URL)
    (STAGING / "README.md").write_text(card, encoding="utf-8")

    print(f"model {repo_id}:")
    for f in sorted(f for f in STAGING.rglob("*") if f.is_file()):
        print(f"  {f.relative_to(STAGING).as_posix():24s} {f.stat().st_size / 2**20:8.1f} MB")
    if args.dry_run:
        return

    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=STAGING,
        repo_id=repo_id,
        commit_message=f"Upload from {GITHUB_URL}/commit/{git_ref}",
    )
    print(f"uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
