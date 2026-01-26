"""Benchmark generative vs reranking preference models on JSONL datasets.

Dataset format (JSONL, flexible):
- "prompt": string prebuilt prompt to feed the model (recommended), OR
- "conversations"/"conversation": list of turns with keys {"from", "value"};
    will be serialized to alternating "User:"/"Assistant:" lines.
- One of:
    - "truth_policy": {"label": str, "description": str}
- Candidate policies (for rerank accuracy) can come from:
    - "truth_policy" + "negative_policies" (list of {label, description}).

Metrics:
- generative mode: accuracy on exact (trimmed) match between generated text and label.
- rerank mode: accuracy on whether the top-scoring policy (log P(label | prompt))
    equals the gold label.

The code keeps dependencies minimal so it can be reused with other datasets
like MMLU by preparing a JSONL file with prompt/label(+candidates). For rerank,
descriptions are used in the prompt to match the training setup in
preference_model_ft_reranking.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
from training.preference_model_fine_tuning.preference_model_ft_reranking import (
    CATCH_ALL_LABEL,
    build_prompt,
    build_training_examples,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from dataset_pipeline_sharegpt import RoutePolicy


@dataclass
class EvalExample:
    prompt_ids: List[int]
    label: str
    candidate_policies: Optional[List[RoutePolicy]] = None


def build_eval_examples(
    training_examples: Iterable,
    tokenizer: PreTrainedTokenizerBase,
    all_negative: bool,
    rng: np.random.Generator,
    max_length: int = 1024,
    max_samples: Optional[int] = None,
) -> List[EvalExample]:
    # customized for shareGPT for now
    examples: List[EvalExample] = []
    for example in training_examples:
        label_space = [policy for policy in example.negative_policies]
        label_space.append(
            RoutePolicy(label=CATCH_ALL_LABEL, description="None of the above")
        )
        if not all_negative:
            label_space.append(example.truth_policy)
        # TODO: vary label size
        randomized_label_space = rng.permutation(list(label_space)).tolist()
        prompt_encoding = build_prompt(
            conversation=example.conversation,
            label_space=randomized_label_space,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        examples.append(
            EvalExample(
                prompt_ids=prompt_encoding,
                label=(
                    example.truth_policy.label if not all_negative else CATCH_ALL_LABEL
                ),
                candidate_policies=randomized_label_space,
            )
        )
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def _normalize(text: str) -> str:
    return text.strip()


def evaluate_generative(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[EvalExample],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    all_negative: bool,
) -> float:
    model.eval()
    correct = 0
    for ex in examples:
        inputs = tokenizer(ex.prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = gen[0][inputs["input_ids"].shape[1] :]
        output_text = _normalize(
            tokenizer.decode(generated_tokens, skip_special_tokens=True)
        )
        expected_label = ex.label if not all_negative else CATCH_ALL_LABEL
        if expected_label == output_text:
            correct += 1
        else:
            # log the first 5 mismatches
            if correct < 5:
                logging.info(f"Mismatch: expected '{ex.label}', got '{output_text}'")
    return correct / len(examples) if examples else 0.0


def score_labels(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: List[int],
    candidate_policies: Sequence[RoutePolicy],
    device: torch.device,
    max_length: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Return [(label, logprob)] sorted desc by logprob."""
    model.eval()
    scores: List[Tuple[str, float]] = []
    for policy in candidate_policies:
        label_ids = tokenizer(
            f"{policy.label}<|im_end|>", add_special_tokens=False, truncation=False
        )["input_ids"]
        if max_length and len(prompt_ids) + len(label_ids) > max_length:
            continue
        input_ids = torch.tensor([prompt_ids + label_ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        start = len(prompt_ids) - 1
        token_positions = log_probs[0, start : start + len(label_ids), :]
        target = torch.tensor(label_ids, device=device)
        token_log_probs = token_positions.gather(1, target.unsqueeze(1)).squeeze(1)
        scores.append((policy.label, float(token_log_probs.sum().item())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def evaluate_rerank(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[EvalExample],
    device: torch.device,
    max_length: Optional[int] = None,
) -> float:
    correct = 0
    used = 0
    for ex in examples:
        if not ex.candidate_policies:
            continue
        scores = score_labels(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=ex.prompt_ids,
            candidate_policies=ex.candidate_policies,
            device=device,
            max_length=max_length,
        )
        if not scores:
            continue
        used += 1
        if scores[0][0] == ex.label:
            correct += 1
        else:
            # log the first 5 mismatches
            if correct < 5:
                logging.info(
                    f"Rerank mismatch: expected '{ex.label}', got '{scores[0][0]}'"
                )
    return correct / used if used else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark preference models (generative vs rerank)"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="JSONL dataset with prompt/label (+candidates for rerank)",
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="HF model name or local path"
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        required=True,
        default="sharegpt_preference_labeled_with_negative.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["generative", "rerank"],
        required=True,
        help="Evaluation mode",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit number of examples"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Max new tokens for generative mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional truncation length for rerank scoring",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g., 'cuda' or 'mps'",
    )
    parser.add_argument(
        "--all-negative",
        action="store_true",
        help="Use all-negative candidate policies (no positive) for rerank eval",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI helper
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True
    ).to(device)

    training_examples = build_training_examples(
        dataset_path=args.dataset_path,
        label_map_path=args.label_map_path,
        max_samples=None,
    )
    examples = build_eval_examples(
        training_examples,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        all_negative=args.all_negative,
    )
    if not examples:
        logger.error("No examples loaded from %s", args.dataset_path)
        return

    if args.mode == "generative":
        acc = evaluate_generative(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        logger.info("Generative accuracy: %.4f (%d examples)", acc, len(examples))
    else:
        acc = evaluate_rerank(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=device,
            max_length=args.max_length,
        )
        logger.info("Rerank accuracy: %.4f", acc)


if __name__ == "__main__":
    main()
