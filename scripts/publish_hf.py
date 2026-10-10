"""Publish the model (PyTorch weights, ONNX int8, model card) and the Gradio demo to Hugging Face.

Files are first assembled in outputs/hub/, templates in hub/ and space/ get the real repo names.
Authentication: `uv run hf auth login`, or HF_TOKEN (a write token) in the environment or in .env.

Usage: uv run python scripts/publish_hf.py --dry-run          # assemble and list files only
       uv run python scripts/publish_hf.py [--only model|space]
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
STAGING = Path("outputs/hub")
GITHUB_URL = "https://github.com/imgreencucumber/ru-ner"

MODEL_FILES = {
    "config.json": MODEL_DIR / "config.json",
    "model.safetensors": MODEL_DIR / "model.safetensors",
    "tokenizer.json": MODEL_DIR / "tokenizer.json",
    "tokenizer_config.json": MODEL_DIR / "tokenizer_config.json",
    "onnx/model_int8.onnx": MODEL_DIR / "onnx" / "model_int8.onnx",
    "onnx/tokenizer.json": MODEL_DIR / "onnx" / "tokenizer.json",
}
TEMPLATES = {
    "model": {"README.md": Path("hub/model_card.md")},
    "space": {
        "app.py": Path("space/app.py"),
        "README.md": Path("space/README.md"),
        "requirements.txt": Path("space/requirements.txt"),
    },
}


def render(path, values):
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def stage(kind, values):
    folder = STAGING / kind
    shutil.rmtree(folder, ignore_errors=True)
    if kind == "model":
        for target, source in MODEL_FILES.items():
            (folder / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, folder / target)
    for target, template in TEMPLATES[kind].items():
        (folder / target).parent.mkdir(parents=True, exist_ok=True)
        (folder / target).write_text(render(template, values), encoding="utf-8")
    return folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", help="HF username, by default taken from the login")
    p.add_argument("--model-name", default="rubert-ner-collection3")
    p.add_argument("--space-name", default="ru-ner-demo")
    p.add_argument("--only", choices=["model", "space"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api = HfApi()
    user = args.user or api.whoami()["name"]
    # the Space installs the package from GitHub at exactly this commit, so it must be pushed
    git_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    values = {
        "repo_id": f"{user}/{args.model_name}",
        "github_url": GITHUB_URL,
        "git_ref": git_ref,
    }
    repos = {"model": (values["repo_id"], "model"), "space": (f"{user}/{args.space_name}", "space")}

    for kind in [args.only] if args.only else ["model", "space"]:
        folder = stage(kind, values)
        repo_id, repo_type = repos[kind]
        files = sorted(f for f in folder.rglob("*") if f.is_file())
        print(f"{repo_type} {repo_id}:")
        for f in files:
            print(f"  {f.relative_to(folder).as_posix():28s} {f.stat().st_size / 2**20:8.1f} MB")
        if args.dry_run:
            continue
        api.create_repo(
            repo_id,
            repo_type=repo_type,
            space_sdk="gradio" if kind == "space" else None,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload from {GITHUB_URL}/commit/{git_ref}",
        )
        prefix = "spaces/" if kind == "space" else ""
        print(f"  uploaded: https://huggingface.co/{prefix}{repo_id}")


if __name__ == "__main__":
    main()
