# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

"""
Evaluation dataset loader for genetic algorithm fitness scoring.

Loads validation splits from locally saved HuggingFace datasets and returns
n samples per dataset, each formatted as:
    {"input": str, "answer": str, "input_answer": str}

Usage:
    from elasticpruner.ga_dataset.eval_dataset import EvalDataset

    ds = EvalDataset(data_dir="/path/to/ga_eval_data", n=100)
    samples = ds.get_samples()          # dict[dataset_name -> list[dict]]
    flat    = ds.get_samples_flat()     # list[dict]  (all datasets merged)
"""

import os
import random
from typing import Dict, List, Optional

from datasets import load_from_disk

LABEL_MAP = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Per-dataset formatters
# Each returns (input_str, answer_str)
# ---------------------------------------------------------------------------

def _format_arc(example: dict):
    question = example["question"]
    choices  = example["choices"]          # {"label": [...], "text": [...]}
    answer_key = example["answerKey"]      # e.g. "B" or "2"

    choice_lines = "\n".join(
        f"{label}. {text}"
        for label, text in zip(choices["label"], choices["text"])
    )
    inp = f"Question: {question}\n{choice_lines}\nAnswer:"

    # answerKey can be "A"/"B"/... or "1"/"2"/...
    if answer_key.isdigit():
        idx = int(answer_key) - 1
        answer = choices["label"][idx] if idx < len(choices["label"]) else answer_key
    else:
        answer = answer_key

    inp_ans = f"{inp} {answer}"
    return inp, answer, inp_ans


def _format_hellaswag(example: dict):
    ctx = example["ctx"]
    endings = example["endings"]
    label = int(example["label"])          # index into endings

    choice_lines = "\n".join(
        f"{LABEL_MAP[i]}. {text}" for i, text in enumerate(endings)
    )
    inp = f"Context: {ctx}\n{choice_lines}\nAnswer:"
    answer = LABEL_MAP[label]
    inp_ans = f"{inp} {answer}"
    return inp, answer, inp_ans


def _format_boolq(example: dict):
    question = example["question"]
    passage  = example["passage"]
    answer   = "Yes" if example["answer"] else "No"

    inp = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
    inp_ans = f"{inp} {answer}"
    return inp, answer, inp_ans


def _format_winogrande(example: dict):
    sentence = example["sentence"]
    option1  = example["option1"]
    option2  = example["option2"]
    answer_idx = example["answer"]         # "1" or "2"

    inp = (
        f"Sentence: {sentence}\n"
        f"A. {option1}\n"
        f"B. {option2}\n"
        f"Answer:"
    )
    answer = "A" if answer_idx == "1" else "B"
    inp_ans = f"{inp} {answer}"
    return inp, answer, inp_ans


def _format_mmlu(example: dict):
    question = example["question"]
    choices  = example["choices"]          # list of 4 strings
    answer_idx = int(example["answer"])    # 0-3

    choice_lines = "\n".join(
        f"{LABEL_MAP[i]}. {text}" for i, text in enumerate(choices)
    )
    inp = f"Question: {question}\n{choice_lines}\nAnswer:"
    answer = LABEL_MAP[answer_idx]
    inp_ans = f"{inp} {answer}"
    return inp, answer, inp_ans


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "arc_challenge": {
        "split_candidates": ["validation", "test"],
        "formatter": _format_arc,
    },
    "arc_easy": {
        "split_candidates": ["validation", "test"],
        "formatter": _format_arc,
    },
    "hellaswag": {
        "split_candidates": ["validation"],
        "formatter": _format_hellaswag,
    },
    "boolq": {
        "split_candidates": ["validation"],
        "formatter": _format_boolq,
    },
    "winogrande": {
        "split_candidates": ["validation"],
        "formatter": _format_winogrande,
    },
    "mmlu": {
        "split_candidates": ["validation", "test"],
        "formatter": _format_mmlu,
    },
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GAEvalDataset:
    """
    Loads locally saved HuggingFace validation datasets and returns n samples
    per dataset, formatted as {"input", "answer", "input_answer"}.

    Args:
        data_dir: Root directory containing one sub-folder per dataset
                  (as saved by ga_eval_data/download_datasets.py).
        n:        Number of samples to return per dataset.
        seed:     Random seed for reproducible sampling (default 42).
        datasets: Optional list of dataset names to load. Loads all by default.
    """

    def __init__(
        self,
        data_dir: str,
        n: int,
        seed: int = 42,
        datasets: Optional[List[str]] = None,
    ):
        self.data_dir = data_dir
        self.n = n
        self.seed = seed
        self.dataset_names = datasets if datasets is not None else list(DATASET_CONFIG.keys())
        self._samples: Dict[str, List[dict]] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        rng = random.Random(self.seed)

        for name in self.dataset_names:
            cfg = DATASET_CONFIG[name]
            split_path = self._find_split(name, cfg["split_candidates"])
            if split_path is None:
                print(f"[EvalDataset] WARNING: no valid split found for '{name}' in {self.data_dir}")
                continue

            raw = load_from_disk(split_path)
            formatter = cfg["formatter"]

            all_samples = []
            for example in raw:
                try:
                    inp, answer, inp_ans = formatter(example)
                    all_samples.append({
                        "input":        inp,
                        "answer":       answer,
                        "input_answer": inp_ans,
                    })
                except Exception as e:
                    # skip malformed examples
                    continue

            if len(all_samples) <= self.n:
                chosen = all_samples
            else:
                chosen = rng.sample(all_samples, self.n)

            self._samples[name] = chosen
            print(f"[EvalDataset] Loaded '{name}': {len(chosen)} samples (pool={len(all_samples)})")

    def _find_split(self, name: str, candidates: List[str]) -> Optional[str]:
        dataset_dir = os.path.join(self.data_dir, name)
        for split in candidates:
            path = os.path.join(dataset_dir, split)
            if os.path.isdir(path):
                return path
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_samples(self) -> Dict[str, List[dict]]:
        """
        Returns a dict mapping dataset name -> list of n sample dicts.

        Each sample dict has keys:
            "input"        : prompt text (question + choices, no answer)
            "answer"       : correct answer label (e.g. "A", "Yes", ...)
            "input_answer" : prompt + correct answer concatenated
        """
        return self._samples

    def get_samples_flat(self) -> List[dict]:
        """
        Returns all samples from all datasets as a single flat list.
        Each dict also includes a "dataset" key with the dataset name.
        """
        flat = []
        for name, samples in self._samples.items():
            for s in samples:
                flat.append({**s, "dataset": name})
        return flat

    def __repr__(self):
        loaded = {k: len(v) for k, v in self._samples.items()}
        return f"EvalDataset(data_dir={self.data_dir!r}, n={self.n}, loaded={loaded})"
