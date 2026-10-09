"""Fine-tune ruBERT for NER on Collection3 and evaluate on Collection3 test and WikiNEuRal-ru test.

Usage: uv run python scripts/train_bert.py [--epochs 4] [--lr 3e-5] [--seed 42]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ru_ner.bert import MODEL_NAME, BertNER, compute_metrics, tokenize_and_align
from ru_ner.data import LABEL2ID, LABELS, load_collection3, load_wikineural_ru
from ru_ner.metrics import evaluate_on, results_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="models/rubert-ner")
    p.add_argument("--results", default="results/rubert.json")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training, evaluate the model already saved in --output",
    )
    return p.parse_args()


def train(args, c3):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = c3.map(
        tokenize_and_align,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["tokens", "ner_tags"],
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )

    checkpoints_dir = Path(args.output) / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_steps=0.1,  # float < 1 is a fraction of total steps
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        seed=args.seed,
        report_to="none",
        # GTX 1060 (Pascal) has no fast fp16, mixed precision would only slow it down
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - start) / 60

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    params = {
        "model": MODEL_NAME,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_epoch_val_f1": trainer.state.best_metric,
        "train_minutes": round(train_minutes, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    val_history = [
        {"epoch": h["epoch"], "f1": h["eval_f1"], "loss": h["eval_loss"]}
        for h in trainer.state.log_history
        if "eval_f1" in h
    ]
    return {"params": params, "val_history": val_history}


def main():
    args = parse_args()
    set_seed(args.seed)
    results_path = Path(args.results)

    c3 = load_collection3()
    if args.eval_only:
        # keep training info from the previous run, only metrics get recomputed
        old = json.loads(results_path.read_text(encoding="utf-8"))["rubert"]
        train_info = {"params": old["params"], "val_history": old["val_history"]}
    else:
        train_info = train(args, c3)

    ner = BertNER(args.output)
    test_sets = {"c3_test": c3["test"], "wikineural_test": load_wikineural_ru("test")}
    results = {}
    # the main model uses constrained decoding, plain argmax is kept for comparison
    for name, decoding in [("rubert", "viterbi"), ("rubert_argmax", "argmax")]:
        ner.decoding = decoding
        results[name] = evaluate_on(ner.predict, test_sets, model_name=name)
    results["rubert"].update(train_info)

    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(results_table(results).to_string())


if __name__ == "__main__":
    main()
