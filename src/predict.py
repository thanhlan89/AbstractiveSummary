import argparse
import json
from pathlib import Path

import torch

from .model import Transformer, create_padding_mask
from .vocab import Vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summaries with the custom Transformer checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json")
    parser.add_argument("--config", type=str, default="checkpoints/config.json")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-src-len", type=int, default=512)
    parser.add_argument("--max-summary-len", type=int, default=128)
    return parser.parse_args()


def truncate_ids(ids, max_length, eos_id):
    if len(ids) <= max_length:
        return ids
    ids = ids[:max_length]
    ids[-1] = eos_id
    return ids


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.load(args.vocab)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    model = Transformer(**config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    source_ids = truncate_ids(vocab.encode(args.text), args.max_src_len, vocab.eos_id)
    src = torch.tensor([source_ids], dtype=torch.long, device=device)
    src_mask = create_padding_mask(src, vocab.pad_id)

    with torch.no_grad():
        prediction = model.greedy_decode(src, src_mask, args.max_summary_len, vocab.sos_id, vocab.eos_id)

    print(vocab.decode(prediction[0].tolist()))


if __name__ == "__main__":
    main()
