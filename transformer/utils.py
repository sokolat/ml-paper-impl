from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import pprint, pickle
import torch

class WMTDataset(Dataset):
    def __init__(self, csv_file: str) -> None:
        self.sentence_pairs = pd.read_csv(csv_file, lineterminator='\n', skiprows=1)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        de_sentence, en_sentence = self.sentence_pairs.iloc[idx]

        return de_sentence, en_sentence

    def __len__(self):
        return len(self.sentence_pairs)

class WMTTokenizedDataset(Dataset):
    def __init__(self, pickle_file: str) -> None:
        with open(pickle_file, 'rb') as f:
            self.sentence_pairs = pickle.load(f)

    def __getitem__(self, idx: int) ->tuple[list[int], list[int]]:
        de_sentence, en_sentence = self.sentence_pairs[idx]

        return de_sentence, en_sentence

    def __len__(self) -> None:
        return len(self.sentence_pairs)

class WMTSampler(Sampler):
    def __init__(self, max_tokens, meta_file, shuffle=True, seed=0) -> None:
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
        self.lengths = np.max(self.metadata, axis=1)
        self.indices = np.argsort(self.lengths, kind='mergesort')
        # Build length-bucketed batches once; reused every epoch.
        self.batches = self._batch_by_size(
            self.indices,
            num_tokens_fn=lambda i: int(self.lengths[i]),
            max_tokens=self.max_tokens,
            required_batch_size_multiple=1
            )

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch so batch order is reshuffled deterministically."""
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            for i in rng.permutation(len(self.batches)):
                yield self.batches[i]
        else:
            yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

    def _batch_by_size(self, indices, num_tokens_fn, max_tokens=25000,
        max_sentences=None, required_batch_size_multiple=1):
        """Self-contained token-budget batcher (no fairseq dependency).

        Args:
            indices: iterable of example indices, pre-sorted by length.
            num_tokens_fn: callable(index) -> int, token count for that example.
            max_tokens: max tokens per batch (after padding).
            max_sentences: optional cap on number of sentences per batch.
            required_batch_size_multiple: round batch size down to a multiple of this.

        Returns:
            list of batches, each a list of indices.
        """
        indices = list(indices)
        batches = []
        batch = []
        max_len_in_batch = 0

        def is_full(num_new, new_max):
            if max_sentences is not None and num_new > max_sentences:
                return True
            if num_new * new_max > max_tokens:
                return True
            return False

        for idx in indices:
            tok = num_tokens_fn(idx)
            new_max = max(max_len_in_batch, tok)
            if batch and is_full(len(batch) + 1, new_max):
                # round down to required multiple
                mult = required_batch_size_multiple
                if mult > 1 and len(batch) > mult:
                    cut = (len(batch) // mult) * mult
                    batches.append(batch[:cut])
                    batch = batch[cut:]
                    max_len_in_batch = max((num_tokens_fn(i) for i in batch), default=0)
                else:
                    batches.append(batch)
                    batch = []
                    max_len_in_batch = 0
                # start fresh accounting for current idx
                batch.append(idx)
                max_len_in_batch = max(max_len_in_batch, tok)
            else:
                batch.append(idx)
                max_len_in_batch = new_max

        if batch:
            batches.append(batch)
        return batches

def make_collate(pad_id: int):
    """Build a collate_fn that pads variable-length token lists to batch-max."""
    def collate(batch: list[tuple[list[int], list[int]]]):
        de = pad_sequence([torch.tensor(d) for d, _ in batch], batch_first=True, padding_value=pad_id)
        en = pad_sequence([torch.tensor(e) for _, e in batch], batch_first=True, padding_value=pad_id)
        return de, en
    return collate
