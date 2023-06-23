import numpy as np
from transformers import EvalPrediction
from datasets import Dataset

INCORRECT = 0
CORRECT = 1
FORGOTTEN = -1

# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    tokenized_examples["label"] = examples["label"]
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        "accuracy": (np.argmax(eval_preds.predictions, axis=1) == eval_preds.label_ids)
        .astype(np.float32)
        .mean()
        .item()
    }


def prepare_forgotten_dataset(train_dataset_featurized, trainer):
    compute_forgotten_stats(trainer)
    forgotten = []
    for _, data in enumerate(train_dataset_featurized):
        sentence = np.array2string(np.array(data["input_ids"]), max_line_width=10000)
        if (
            trainer.classified_status[sentence] == INCORRECT
            or trainer.classified_status[sentence] == FORGOTTEN
        ):
            forgotten.append(data)

    return Dataset.from_list(forgotten)


def compute_forgotten_stats(trainer):
    never_count = 0
    forgotten_count = 0
    for _, value in trainer.classified_status.items():
        if value == INCORRECT:
            never_count += 1
        if value == FORGOTTEN:
            forgotten_count += 1
    print("Never learned", never_count)
    print("Forgotten", forgotten_count)
