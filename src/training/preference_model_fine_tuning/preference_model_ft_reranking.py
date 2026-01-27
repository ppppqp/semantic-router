"""Reranking-based preference model fine-tuning on ShareGPT.

This follows the classification data pipeline in preference_model_ft.py but
exposes a reranking-style inference API where labels are scored via
log-probabilities conditioned on the prompt, i.e. score(label) = log P(label | prompt).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import string
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from torch.utils.data import Dataset as TorchDataset

from dataset_pipeline_sharegpt import ShareGPTConversation, Turn, RoutePolicy


CATCH_ALL_LABEL = "none_of_the_above"

DEFAULT_SYSTEM_PROMPT = (
    "You are a routing controller that reads a conversation and outputs "
    "the best preference label for downstream model routing.\n"
)


@dataclass
class PreferenceTrainingExample:
    conversation: ShareGPTConversation
    truth_policy: RoutePolicy
    negative_policies: List[RoutePolicy]


def _load_label_mapping(path: Path) -> Dict[str, Tuple[RoutePolicy, List[RoutePolicy]]]:
    if not path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {path}")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
        if isinstance(payload, list):
            mapping: Dict[str, str] = {}
            for item in payload:
                sample_id = str(item.get("sample_id"))
                label = item.get("label")
                if sample_id and label:
                    mapping[sample_id] = str(label)
            return mapping
        raise ValueError("Unsupported JSON label mapping format.")

    mapping: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", "").strip())
            truth_policy = RoutePolicy.from_dict(item.get("truth_policy", {}))
            negative_policies = [
                RoutePolicy.from_dict(p) for p in item.get("negative_policies", [])
            ]
            if sample_id and truth_policy:
                mapping[sample_id] = (truth_policy, negative_policies)
    if not mapping:
        raise ValueError(f"No label mappings found in {path}")
    return mapping


def load_shareGPT_conversation(
    raw_item: Dict[str, object],
) -> Optional[ShareGPTConversation]:
    convo = raw_item.get("conversations") or []
    sample_id = str(raw_item.get("id", "").strip())
    if not sample_id:
        return None
    normalized_messages: List[Turn] = []
    for turn in convo:
        speaker_raw = turn.get("from")
        speaker = str(speaker_raw).lower().strip()
        content_raw = turn.get("value")
        content = str(content_raw or "").strip()
        if not content:
            continue
        role = "assistant" if speaker == "gpt" else "user"
        normalized_messages.append(Turn(role=role, content=content))
    if not normalized_messages:
        logging.debug("Skipping sample %s with no valid messages", sample_id)
        return None
    return ShareGPTConversation(sample_id=sample_id, messages=normalized_messages)


def iter_sharegpt_conversations(dataset_path: Path) -> Iterable[ShareGPTConversation]:
    payload = json.loads(dataset_path.read_text())
    for raw_item in payload:
        conversation = load_shareGPT_conversation(raw_item)
        if conversation is not None:
            yield conversation


def build_training_examples(
    dataset_path: Path,
    label_map_path: Path,
    max_samples: Optional[int] = None,
    start_index: int = 0,
) -> List[PreferenceTrainingExample]:
    label_mapping = _load_label_mapping(label_map_path)
    examples: List[PreferenceTrainingExample] = []
    for idx, conversation in enumerate(iter_sharegpt_conversations(dataset_path)):
        if idx < start_index:
            continue
        policy_pairs = label_mapping.get(conversation.sample_id)

        if not policy_pairs:
            continue

        truth_policy, negative_policies = policy_pairs
        examples.append(
            PreferenceTrainingExample(
                conversation=conversation,
                truth_policy=truth_policy,
                negative_policies=negative_policies,
            )
        )
        if max_samples and len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError(
            "No overlapping samples found between ShareGPT dataset and label map."
        )
    logging.info("Loaded %s labeled ShareGPT samples", len(examples))
    return examples


def conversation_to_text(conversation: ShareGPTConversation) -> str:
    lines: List[str] = []
    for turn in conversation.messages:
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)


def build_prompt(
    conversation: ShareGPTConversation,
    label_space: Sequence[RoutePolicy],
    label_name_mapping: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[int]:
    label_clauses = []
    for policy in label_space:
        label_clauses.append(
            f"- {label_name_mapping.get(policy.label, policy.label)}: {policy.description}"
        )
    label_clause = "\n".join(label_clauses)

    convo_text = conversation_to_text(conversation)

    system_ids = tokenizer(
        system_prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    instruction_content = "\nAnswer with the single routing policy that best matches the conversation. Do not include thinking process.\n"
    instruction_ids = tokenizer(
        instruction_content,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    label_content = f"Each policy is represented by a label, followed by a short description. Valid policies:\n{label_clause}"
    label_ids = tokenizer(
        label_content,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    reserved = len(system_ids) + len(instruction_ids) + len(label_ids) + 15
    if reserved >= max_length:
        raise ValueError("Instruction + system + labels exceed max_length")
    remaining = max_length - reserved

    conversation_ids = tokenizer(
        convo_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remaining,
    )["input_ids"]

    user_content = (
        f"{label_content}\n"
        f"Conversation:\n{tokenizer.decode(conversation_ids, skip_special_tokens=True)}\n\n"
        f"{instruction_content}"
    )

    messages = [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_content},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(prompt_text, add_special_tokens=True, truncation=False)[
        "input_ids"
    ]


def sample_label_space(
    available_labels: List[RoutePolicy],
    max_labels_in_prompt: int,
    min_labels_in_prompt: int,
    rng: np.random.Generator,
) -> List[RoutePolicy]:
    max_candidates = len(available_labels)
    upper_bound = (
        max_candidates
        if max_labels_in_prompt is None
        else min(max_labels_in_prompt, max_candidates)
    )
    lower_bound = min(min_labels_in_prompt, upper_bound)
    sample_size = (
        upper_bound
        if lower_bound == upper_bound
        else int(rng.integers(lower_bound, upper_bound + 1))
    )
    sampled = rng.choice(available_labels, size=sample_size, replace=False)
    return sampled.tolist()


def generate_random_label_name(rng: np.random.Generator) -> str:
    # chose a random label name like "alpha1234"
    prefix = "".join(rng.choice(list(string.ascii_lowercase), size=5))
    return prefix + str(rng.integers(0, 9999))


class ChatAlignedDataset(TorchDataset):
    """Dataset that redraws label spaces per epoch for robustness."""

    def __init__(
        self,
        examples: Sequence[PreferenceTrainingExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        max_labels_in_prompt: int,
        min_labels_in_prompt: int,
        seed: int,
        pad_to_max_length: bool = False,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_labels_in_prompt = max_labels_in_prompt
        self.min_labels_in_prompt = min_labels_in_prompt
        self.pad_to_max_length = pad_to_max_length
        self.base_seed = seed
        self.epoch = 0

        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng_for_idx(self, idx: int) -> np.random.Generator:
        # Spread seeds so each epoch reshuffles independently.
        # 1_000_000 is the max number of samples for each epoch.
        return np.random.default_rng(self.base_seed + self.epoch * 1_000_000 + idx)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.examples[idx]
        rng = self._rng_for_idx(idx)

        all_negative = rng.random() < float(1 / self.max_labels_in_prompt)
        catch_all_label = RoutePolicy(
            label=CATCH_ALL_LABEL, description="None of the above matches"
        )
        sampled_label_space = sample_label_space(
            available_labels=example.negative_policies,
            max_labels_in_prompt=self.max_labels_in_prompt,
            min_labels_in_prompt=self.min_labels_in_prompt,
            rng=rng,
        )
        # always include the catch all label
        if all_negative:
            sampled_label_space.append(catch_all_label)
        else:
            # if not all negative, include truth label
            sampled_label_space.extend([example.truth_policy, catch_all_label])

        random_label_space: list[RoutePolicy] = rng.permutation(
            sampled_label_space
        ).tolist()

        # generate random label names to avoid ontology overfitting
        label_name_mapping = {}
        for policy in random_label_space:
            label_name_mapping[policy.label] = generate_random_label_name(rng)

        prompt_encoding = build_prompt(
            conversation=example.conversation,
            label_space=random_label_space,
            label_name_mapping=label_name_mapping,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        truth_label = (
            example.truth_policy.label if not all_negative else CATCH_ALL_LABEL
        )
        label_ids = self.tokenizer(
            f"{label_name_mapping[truth_label]}<|im_end|>",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        full_encoding: List[int] = prompt_encoding + label_ids

        # DEBUG;
        # print(self.tokenizer.decode(full_encoding, skip_special_tokens=True))
        labels = full_encoding.copy()
        for i in range(len(prompt_encoding)):
            labels[i] = -100
        for i, token_id in enumerate(full_encoding):
            if token_id == self.pad_id:
                labels[i] = -100

        attention_mask = [
            1 if token_id != self.pad_id else 0 for token_id in full_encoding
        ]

        if self.pad_to_max_length:
            pad_needed = self.max_length - len(full_encoding)
            if pad_needed > 0:
                full_encoding = full_encoding + [self.pad_id] * pad_needed
                attention_mask = attention_mask + [0] * pad_needed
                labels = labels + [-100] * pad_needed
            else:
                full_encoding = full_encoding[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                labels = labels[: self.max_length]

        return {
            "input_ids": full_encoding,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class EpochRandomizationCallback(TrainerCallback):
    """Reseeds the dataset each epoch to refresh label sampling."""

    def __init__(self, dataset: ChatAlignedDataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        current_epoch = int(state.epoch or 0)
        self.dataset.set_epoch(current_epoch)
        return control


class LabelPaddingCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]

        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_length - len(label)
            if pad_len < 0:
                label = label[:max_length]
                pad_len = 0
            padded_labels.append(label + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a reranking preference model on ShareGPT. Inference scores labels via log P(label | prompt)."
        )
    )
    parser.add_argument(
        "--sharegpt-path",
        type=Path,
        default=Path("ShareGPT_V3_unfiltered_cleaned_split.json"),
        help="Path to the raw ShareGPT JSON file.",
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=Path("sharegpt_preference_labeled_with_negative.jsonl"),
        help="Path to sample_id -> label mapping (JSON or JSONL).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model identifier (causal LM).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="preference_model_qwen3_rerank",
        help="Where to store checkpoints and tokenizer.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to this many matched samples (None = full dataset).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many samples from the start of the dataset.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Token truncation length for prompts + labels.",
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad each example to max_length (else dynamic padding).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for larger effective batch size.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to reserve for validation.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio for scheduler.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=1,
        help="How many checkpoints to keep.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training if supported.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training if supported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits and initialization.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation on the base model without training.",
    )
    parser.add_argument(
        "--candidate-labels",
        type=str,
        default=None,
        help="Comma-separated labels to rerank during demo inference.",
    )
    parser.add_argument(
        "--inference-sample-index",
        type=int,
        default=None,
        help="Optional dataset index to run a rerank demo after training (or in eval-only).",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples = build_training_examples(
        dataset_path=args.sharegpt_path,
        label_map_path=args.label_map_path,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    logger.info("Loaded %s labeled examples", len(examples))
    rng = np.random.default_rng(args.seed)
    if args.eval_ratio and args.eval_ratio > 0:
        eval_size = max(1, int(len(examples) * args.eval_ratio))
        permutation = rng.permutation(len(examples))
        eval_indices = permutation[:eval_size]
        train_indices = permutation[eval_size:]
        train_examples = [examples[i] for i in train_indices]
        eval_examples = [examples[i] for i in eval_indices]
    else:
        train_examples = examples
        eval_examples = None

    train_dataset = ChatAlignedDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
        max_labels_in_prompt=16,
        min_labels_in_prompt=4,
        seed=args.seed,
    )

    eval_dataset = (
        ChatAlignedDataset(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
            max_labels_in_prompt=16,
            min_labels_in_prompt=4,
            seed=args.seed + 13,
        )
        if eval_examples is not None
        else None
    )

    model_dtype = (
        torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    data_collator = LabelPaddingCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EpochRandomizationCallback(train_dataset)],
    )

    logger.info(
        "Starting training with %s train / %s eval examples",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )

    if args.eval_only:
        if eval_dataset is None:
            logger.error(
                "Eval-only requested but no eval split found. Set --eval-ratio > 0."
            )
            return
        metrics = trainer.evaluate()
        logger.info("Eval-only metrics: %s", metrics)
    else:
        trainer.train()
        if eval_dataset is not None:
            metrics = trainer.evaluate()
            logger.info("Eval metrics: %s", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model and tokenizer saved to %s", args.output_dir)


def main() -> None:  # pragma: no cover - CLI helper
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
