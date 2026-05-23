import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import build_datasets, collate_fn
from .model import Transformer, create_padding_mask, generate_square_subsequent_mask

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer from scratch for abstractive summarization")
    parser.add_argument("--train", type=str, default="data/train.tsv", help="Path to training data")
    parser.add_argument("--valid", type=str, default="data/valid.tsv", help="Path to validation data")
    parser.add_argument("--source-field", type=str, default="text")
    parser.add_argument("--target-field", type=str, default="summary")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-src-len", type=int, default=512)
    parser.add_argument("--max-tgt-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    return parser.parse_args()


def build_masks(src, tgt, pad_id):
    src_mask = create_padding_mask(src, pad_id)
    tgt_mask = create_padding_mask(tgt, pad_id) & generate_square_subsequent_mask(tgt.size(1), tgt.device)
    return src_mask, tgt_mask


class NoamScheduler:
    def __init__(self, optimizer, model_dim: int, warmup_steps: int):
        self.optimizer = optimizer
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.model_dim ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def run_epoch(model, dataloader, optimizer, scheduler, criterion, device, pad_id, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in dataloader:
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask = build_masks(src, tgt_input, pad_id)
            logits = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_rouge(model, dataset, device, max_tgt_len, eval_samples=None):
    if not ROUGE_AVAILABLE:
        return {}

    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    count = 0

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=partial(collate_fn, pad_id=dataset.vocab.pad_id))
    with torch.no_grad():
        for batch in loader:
            src = batch["source"].to(device)
            src_mask = create_padding_mask(src, dataset.vocab.pad_id)
            predictions = model.greedy_decode(src, src_mask, max_tgt_len, dataset.vocab.sos_id, dataset.vocab.eos_id)

            for index in range(src.size(0)):
                pred_text = dataset.vocab.decode(predictions[index].tolist())
                target_text = dataset.vocab.decode(batch["target"][index].tolist())
                rouge_scores = scorer.score(target_text, pred_text)
                results["rouge1"] += rouge_scores["rouge1"].fmeasure
                results["rouge2"] += rouge_scores["rouge2"].fmeasure
                results["rougeL"] += rouge_scores["rougeL"].fmeasure
                count += 1
                if eval_samples is not None and count >= eval_samples:
                    return {k: v / max(1, count) for k, v in results.items()}
    return {k: v / max(1, count) for k, v in results.items()}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, valid_dataset = build_datasets(
        args.train,
        args.valid,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
        max_source_length=args.max_src_len,
        max_target_length=args.max_tgt_len,
        source_field=args.source_field,
        target_field=args.target_field,
    )

    collate = partial(collate_fn, pad_id=train_dataset.vocab.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = Transformer(
        vocab_size=len(train_dataset.vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_len=max(args.max_src_len, args.max_tgt_len),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.pad_id, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, model_dim=args.embed_dim, warmup_steps=args.warmup_steps)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    train_dataset.vocab.save(str(save_path / "vocab.json"))

    model_config = {
        "vocab_size": len(train_dataset.vocab),
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_len": max(args.max_src_len, args.max_tgt_len),
    }
    with open(save_path / "config.json", "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, criterion, device, train_dataset.vocab.pad_id, train=True)
        valid_loss = run_epoch(model, valid_loader, optimizer, scheduler, criterion, device, train_dataset.vocab.pad_id, train=False)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")

        if ROUGE_AVAILABLE:
            rouge_scores = evaluate_rouge(model, valid_dataset, device, args.max_tgt_len, args.eval_samples)
            print(
                f"Validation ROUGE: 1={rouge_scores['rouge1']:.4f} 2={rouge_scores['rouge2']:.4f} L={rouge_scores['rougeL']:.4f}"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_step": scheduler.step_num,
                "config": model_config,
            },
            save_path / f"transformer_epoch_{epoch}.pt",
        )

    print("Training completed.")


if __name__ == "__main__":
    main()
