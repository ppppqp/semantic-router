# convert the MMLU dataset into PreferenceTrainingExample format
from pathlib import Path
from typing import Iterable, List
from datasets import load_dataset
from training.preference_model_fine_tuning.dataset_pipeline_sharegpt import (
    Turn,
    ShareGPTConversation,
    RoutePolicy,
)
from training.preference_model_fine_tuning.preference_model_ft_reranking import (
    PreferenceTrainingExample,
)


def build_mmlu_eval_examples():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    examples: List[PreferenceTrainingExample] = []

    all_categories = dataset.unique("category")
    for item in dataset:
        question = item["question"]
        # we only care about category for routing purpose
        category = item["category"]
        sample_id = item["id"]

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
    print(len(examples), "MMLU eval examples built.")
