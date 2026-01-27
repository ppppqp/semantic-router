# convert the MMLU dataset into PreferenceTrainingExample format
from pathlib import Path
from typing import Iterable, List
from datasets import load_dataset
from dataset_pipeline_sharegpt import (
    Turn,
    ShareGPTConversation,
    RoutePolicy,
)
from preference_model_ft_reranking import (
    PreferenceTrainingExample,
)


def build_clinc150_training_examples(use_domain_only: bool = True):
    dataset = load_dataset("contemmcm/clinc150", split="complete")
    examples: List[PreferenceTrainingExample] = []

    if use_domain_only:
        all_labels = dataset.unique("domain")
    else:
        all_labels = dataset.unique("intent")
    for item in dataset:
        question = item["text"]
        # we only care about domain for routing purpose
        label = item["domain"] if use_domain_only else item["intent"]
        sample_id = item["id"]

        conversation = ShareGPTConversation(
            sample_id=sample_id,
            messages=[
                Turn(role="user", content=question),
            ],
        )

        truth_policy = RoutePolicy(
            label=label,
            description=label,
        )

        negative_policies = [
            RoutePolicy(label=l, description=l) for l in all_labels if l != label
        ]

        examples.append(
            PreferenceTrainingExample(
                conversation=conversation,
                truth_policy=truth_policy,
                negative_policies=negative_policies,
            )
        )
    return examples


def build_mmlu_training_examples():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    examples: List[PreferenceTrainingExample] = []

    all_categories = dataset.unique("category")
    for item in dataset:
        question = item["question"]
        # we only care about category for routing purpose
        category = item["category"]
        sample_id = item["question_id"]

        conversation = ShareGPTConversation(
            sample_id=sample_id,
            messages=[
                Turn(role="user", content=question),
            ],
        )

        truth_policy = RoutePolicy(
            label=category,
            description=category,
        )

        negative_policies = [
            RoutePolicy(label=cat, description=cat)
            for cat in all_categories
            if cat != category
        ]

        examples.append(
            PreferenceTrainingExample(
                conversation=conversation,
                truth_policy=truth_policy,
                negative_policies=negative_policies,
            )
        )
    return examples


if __name__ == "__main__":
    build_mmlu_training_examples()
