import unittest
from types import SimpleNamespace

from fine_tune_whisper import (
    DEFAULT_TARGET_MODULES,
    MetricComputer,
    replace_ignored_label_ids,
    resolve_dataset_splits,
)


class FakeSplit:
    def __init__(self):
        self.calls = []

    def train_test_split(self, test_size, seed):
        self.calls.append((test_size, seed))
        return {"train": "train-split", "test": "eval-split"}


class FakeTokenizer:
    pad_token_id = 99

    def __init__(self):
        self.seen = []

    def batch_decode(self, values, skip_special_tokens=True):
        normalized = values.tolist() if hasattr(values, "tolist") else values
        self.seen.append((normalized, skip_special_tokens))
        return ["decoded"] * len(values)


class FakeMetric:
    def __init__(self):
        self.calls = []

    def compute(self, predictions, references):
        self.calls.append((predictions, references))
        return 0.25


class ResolveDatasetSplitsTests(unittest.TestCase):
    def test_uses_existing_eval_split_when_provided(self):
        dataset = {"train": "train-data", "validation": "eval-data"}

        result = resolve_dataset_splits(dataset, "train", "validation", 0.2, 11)

        self.assertEqual({"train": "train-data", "eval": "eval-data"}, result)

    def test_creates_validation_split_when_missing(self):
        train_split = FakeSplit()

        result = resolve_dataset_splits({"train": train_split}, "train", None, 0.15, 7)

        self.assertEqual({"train": "train-split", "eval": "eval-split"}, result)
        self.assertEqual([(0.15, 7)], train_split.calls)

    def test_raises_clear_error_for_missing_split(self):
        with self.assertRaisesRegex(ValueError, "Training split 'train' was not found"):
            resolve_dataset_splits({"validation": object()}, "train", None, 0.1, 42)


class MetricComputerTests(unittest.TestCase):
    def test_replaces_ignored_labels_before_decoding(self):
        tokenizer = FakeTokenizer()
        metric = FakeMetric()
        computer = MetricComputer(processor=SimpleNamespace(tokenizer=tokenizer), metric=metric)
        eval_prediction = SimpleNamespace(
            predictions=[[1, 2, 3]],
            label_ids=[[4, -100, 5]],
        )

        result = computer(eval_prediction)

        self.assertEqual({"wer": 0.25}, result)
        self.assertEqual([[1, 2, 3]], tokenizer.seen[0][0])
        self.assertEqual([[4, 99, 5]], tokenizer.seen[1][0])
        self.assertEqual([(["decoded"], ["decoded"])], metric.calls)

    def test_replace_ignored_label_ids_supports_plain_lists(self):
        self.assertEqual([[1, 7, 2]], replace_ignored_label_ids([[1, -100, 2]], 7))


class ConstantTests(unittest.TestCase):
    def test_lora_targets_query_and_value_projections(self):
        self.assertEqual(("q_proj", "v_proj"), DEFAULT_TARGET_MODULES)


if __name__ == "__main__":
    unittest.main()
