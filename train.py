"""Fine-tune MiniLM for 3-way NLI on Colab GPU (MNLI + SNLI).

Saves ./MODEL for CodaBench: zip model.py + MODEL/ then submit.
"""

import argparse
import os
import zipfile
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from model import NLI, NLIConfig, collate_fn

BACKBONE = "microsoft/MiniLM-L12-H384-uncased"
MAX_PARAMS = 40_000_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto by device")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output-dir", default="./NLIMODEL")
    p.add_argument("--model-dir", default="./MODEL")
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from a checkpoint; without a value, use the newest one",
    )
    p.add_argument("--no-snli", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    return p.parse_args()


def load_nli_frames(use_snli=True, max_train_samples=0):
    frames = []

    mnli = load_dataset("nyu-mll/multi_nli")
    df_train = mnli["train"].to_pandas().dropna()
    df_train = df_train[df_train["label"] != -1][["premise", "hypothesis", "label"]]
    frames.append(df_train)

    if use_snli:
        try:
            snli = load_dataset("stanfordnlp/snli")
            df_snli = snli["train"].to_pandas().dropna()
            df_snli = df_snli[df_snli["label"] != -1][
                ["premise", "hypothesis", "label"]
            ]
            frames.append(df_snli)
            print(f"SNLI train rows: {len(df_snli)}")
        except Exception as e:
            print(f"SNLI skipped: {e}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if max_train_samples > 0:
        df_all = df_all.head(max_train_samples)

    df_val = mnli["validation_matched"].to_pandas().dropna()
    df_val = df_val[df_val["label"] != -1][["premise", "hypothesis", "label"]]
    print(f"Train rows: {len(df_all)}, Val rows: {len(df_val)}")
    return df_all, df_val


def build_dataset(df, tokenizer):
    enc = tokenizer(
        df["premise"].tolist(),
        df["hypothesis"].tolist(),
        truncation=True,
        max_length=128,
        padding=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "labels": df["label"].astype(int).tolist(),
        }
    )


def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}
    for name in ("accuracy", "precision", "recall", "f1"):
        metric = evaluate.load(name)
        kwargs = {"predictions": pred, "references": targ}
        if name in ("precision", "recall", "f1"):
            kwargs["average"] = "macro"
        if name in ("precision", "recall"):
            kwargs["zero_division"] = 0
        res[name] = metric.compute(**kwargs)[name]
    return res


def zip_submission(model_dir, out_zip="submission.zip"):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("model.py", arcname="model.py")
        for root, _, files in os.walk(model_dir):
            for name in files:
                path = os.path.join(root, name)
                arc = os.path.relpath(path, start=".")
                zf.write(path, arcname=arc)
    print(f"Wrote {out_zip} ({os.path.getsize(out_zip) / 1e6:.1f} MB)")


def newest_checkpoint(output_dir):
    checkpoints = [
        path
        for path in Path(output_dir).glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* found in {output_dir}")
    return str(max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1])))


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    use_mps = (not use_cuda) and torch.backends.mps.is_available()

    if args.batch_size > 0:
        train_bs = args.batch_size
    elif use_cuda:
        train_bs = 64
    elif use_mps:
        train_bs = 32
    else:
        train_bs = 16
    eval_bs = min(128, train_bs * 2)

    print("device: cuda" if use_cuda else ("device: mps" if use_mps else "device: cpu"))
    print(f"train_bs={train_bs}, eval_bs={eval_bs}, epochs={args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    backbone = AutoModel.from_pretrained(BACKBONE)
    bc = backbone.config

    config = NLIConfig(
        vocab_size=bc.vocab_size,
        hidden_size=bc.hidden_size,
        nclass=3,
        num_hidden_layers=bc.num_hidden_layers,
        num_attention_heads=bc.num_attention_heads,
        intermediate_size=bc.intermediate_size,
        max_position_embeddings=bc.max_position_embeddings,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = NLI(config)
    missing, unexpected = model.encoder.load_state_dict(
        backbone.state_dict(), strict=False
    )
    print("encoder load missing:", len(missing), "unexpected:", len(unexpected))

    allparams = sum(p.numel() for p in model.parameters())
    print("All Param:", allparams)
    assert allparams < MAX_PARAMS, f"params {allparams} >= {MAX_PARAMS}"

    df_train, df_val = load_nli_frames(
        use_snli=not args.no_snli, max_train_samples=args.max_train_samples
    )
    print("Tokenizing train/val...")
    train_set = build_dataset(df_train, tokenizer)
    val_set = build_dataset(df_val, tokenizer)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=2 if use_cuda else 0,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=args.lr,
        warmup_steps=2000,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=200,
        report_to="none",
        fp16=use_cuda,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    resume_from = None
    if args.resume:
        resume_from = (
            newest_checkpoint(args.output_dir)
            if args.resume == "auto"
            else args.resume
        )
        print(f"Resuming from: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

    os.makedirs(args.model_dir, exist_ok=True)
    tokenizer.save_pretrained(args.model_dir)
    model.save_pretrained(args.model_dir)
    print("Saved", args.model_dir, "; All Param:", allparams)

    zip_submission(args.model_dir, out_zip="submission.zip")


if __name__ == "__main__":
    main()
