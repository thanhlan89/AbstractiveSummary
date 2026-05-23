import collections
import json
from typing import Iterable, List

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocab:
    def __init__(self, tokens: Iterable[str] = None, min_freq: int = 1, max_size: int = None):
        self.token2idx = {}
        self.idx2token = []
        self.min_freq = min_freq
        self.max_size = max_size
        self.add_tokens(SPECIAL_TOKENS)
        if tokens is not None:
            self.build(tokens)

    def add_token(self, token: str) -> int:
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)
        return self.token2idx[token]

    def add_tokens(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.add_token(token)

    def build(self, texts: Iterable[str]) -> None:
        counter = collections.Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        common_tokens = [token for token, freq in counter.most_common() if freq >= self.min_freq]
        if self.max_size is not None:
            common_tokens = common_tokens[: max(0, self.max_size - len(self.idx2token))]

        for token in common_tokens:
            self.add_token(token)

    def tokenize(self, text: str) -> List[str]:
        return text.strip().lower().split()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token2idx.get(token, self.token2idx[UNK_TOKEN]) for token in tokens]
        if add_special_tokens:
            ids = [self.sos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            if idx < 0 or idx >= len(self.idx2token):
                token = UNK_TOKEN
            else:
                token = self.idx2token[idx]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: str) -> None:
        payload = {
            "idx2token": self.idx2token,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        vocab = cls(tokens=None, min_freq=payload.get("min_freq", 1), max_size=payload.get("max_size"))
        vocab.token2idx = {}
        vocab.idx2token = []
        for token in payload["idx2token"]:
            vocab.add_token(token)
        return vocab

    def __len__(self) -> int:
        return len(self.idx2token)

    @property
    def pad_id(self) -> int:
        return self.token2idx[PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token2idx[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token2idx[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2idx[UNK_TOKEN]
