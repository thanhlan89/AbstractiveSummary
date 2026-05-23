import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .vocab import Vocab


def read_summary_file(path: str, source_field: str = "text", target_field: str = "summary") -> List[Dict[str, str]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = []
    suffix = path_obj.suffix.lower()
    with open(path_obj, "r", encoding="utf-8") as handle:
        if suffix == ".jsonl":
            rows = (json.loads(line) for line in handle if line.strip())
            for row in rows:
                source_text = str(row.get(source_field, "")).strip()
                target_text = str(row.get(target_field, "")).strip()
                if source_text and target_text:
                    data.append({"text": source_text, "summary": target_text})
        else:
            delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                source_text = row.get(source_field, "").strip()
                target_text = row.get(target_field, "").strip()
                if source_text and target_text:
                    data.append({"text": source_text, "summary": target_text})
    return data


class SummaryDataset(Dataset):
    def __init__(
        self,
        path: str,
        vocab: Optional[Vocab] = None,
        min_freq: int = 2,
        max_vocab: int = 30000,
        max_source_length: int = 512,
        max_target_length: int = 128,
        source_field: str = "text",
        target_field: str = "summary",
    ):
        self.path = path
        self.raw_data = read_summary_file(path, source_field=source_field, target_field=target_field)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        if vocab is None:
            all_texts = [item["text"] for item in self.raw_data] + [item["summary"] for item in self.raw_data]
            self.vocab = Vocab(all_texts, min_freq=min_freq, max_size=max_vocab)
        else:
            self.vocab = vocab

        self.examples = []
        for item in self.raw_data:
            source_ids = truncate_ids(self.vocab.encode(item["text"]), max_source_length, self.vocab.eos_id)
            target_ids = truncate_ids(self.vocab.encode(item["summary"]), max_target_length, self.vocab.eos_id)
            self.examples.append({"source_ids": source_ids, "target_ids": target_ids})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.examples[index]


def truncate_ids(ids: List[int], max_length: int, eos_id: int) -> List[int]:
    if len(ids) <= max_length:
        return ids
    truncated = ids[:max_length]
    truncated[-1] = eos_id
    return truncated


def pad_sequence(sequences: List[List[int]], pad_value: int) -> torch.LongTensor:
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


def collate_fn(batch: List[Dict[str, List[int]]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    source_batch = [item["source_ids"] for item in batch]
    target_batch = [item["target_ids"] for item in batch]
    source_tensor = pad_sequence(source_batch, pad_id)
    target_tensor = pad_sequence(target_batch, pad_id)
    return {
        "source": source_tensor,
        "target": target_tensor,
    }


def build_datasets(
    train_path: str,
    valid_path: str,
    min_freq: int = 2,
    max_vocab: int = 30000,
    max_source_length: int = 512,
    max_target_length: int = 128,
    source_field: str = "text",
    target_field: str = "summary",
):
    train_dataset = SummaryDataset(
        train_path,
        vocab=None,
        min_freq=min_freq,
        max_vocab=max_vocab,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        source_field=source_field,
        target_field=target_field,
    )
    valid_dataset = SummaryDataset(
        valid_path,
        vocab=train_dataset.vocab,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        source_field=source_field,
        target_field=target_field,
    )
    return train_dataset, valid_dataset
