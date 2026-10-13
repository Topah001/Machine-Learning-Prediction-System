from dataclasses import dataclass
import csv
import math
import random
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "data" / "iris.csv"
_SPECIES_PREFIX = "Iris-"
_SPECIES = ("setosa", "versicolor", "virginica")


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    macro_precision: float
    test_samples: int


class NearestCentroid:
    """Nearest-centroid classifier with feature standardization fit on train data."""

    def fit(self, features: list[list[float]], labels: list[str]):
        if not features or len(features) != len(labels):
            raise ValueError("features and labels must be non-empty and equally sized")
        width = len(features[0])
        if any(len(row) != width for row in features):
            raise ValueError("all feature rows must have the same width")
        self.width = width
        self.means = [
            sum(row[column] for row in features) / len(features)
            for column in range(width)
        ]
        self.scales = []
        for column in range(width):
            variance = sum(
                (row[column] - self.means[column]) ** 2 for row in features
            ) / len(features)
            self.scales.append(math.sqrt(variance) or 1.0)

        self.labels = tuple(sorted(set(labels)))
        self.centroids = {}
        for label in self.labels:
            class_rows = [row for row, target in zip(features, labels) if target == label]
            self.centroids[label] = [
                sum(
                    (row[column] - self.means[column]) / self.scales[column]
                    for row in class_rows
                )
                / len(class_rows)
                for column in range(width)
            ]
        return self

    def _distances(self, features: list[float]) -> dict[str, float]:
        if len(features) != self.width:
            raise ValueError(f"expected exactly {self.width} measurements")
        scaled = [
            (value - self.means[column]) / self.scales[column]
            for column, value in enumerate(features)
        ]
        return {
            label: sum(
                (value - center) ** 2
                for value, center in zip(scaled, self.centroids[label])
            )
            for label in self.labels
        }

    def predict(self, features: list[float]) -> str:
        distances = self._distances(features)
        return min(distances, key=distances.get)

    def predict_confidence(self, features: list[float]) -> tuple[str, float]:
        distances = self._distances(features)
        minimum = min(distances.values())
        weights = {
            label: math.exp(-(distance - minimum))
            for label, distance in distances.items()
        }
        label = max(weights, key=weights.get)
        return label, weights[label] / sum(weights.values())


def load_iris() -> tuple[list[list[float]], list[str]]:
    features = []
    labels = []
    with _DATA_PATH.open(newline="", encoding="utf-8") as dataset:
        for row in csv.reader(dataset):
            if not row:
                continue
            if len(row) != 5 or not row[4].startswith(_SPECIES_PREFIX):
                raise ValueError(f"invalid Iris data row: {row!r}")
            features.append([float(value) for value in row[:4]])
            labels.append(row[4][len(_SPECIES_PREFIX) :])
    if not features:
        raise ValueError(f"Iris dataset is empty: {_DATA_PATH}")
    return features, labels


def stratified_split(
    features: list[list[float]], labels: list[str], test_size: float, seed: int
):
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    grouped = {label: [] for label in sorted(set(labels))}
    for index, label in enumerate(labels):
        grouped[label].append(index)

    target_test_count = math.ceil(len(labels) * test_size)
    quotas = {label: len(indices) * test_size for label, indices in grouped.items()}
    counts = {label: math.floor(quota) for label, quota in quotas.items()}
    extras = target_test_count - sum(counts.values())
    priority = sorted(grouped, key=lambda label: (-(quotas[label] % 1), label))
    for label in priority[:extras]:
        counts[label] += 1

    generator = random.Random(seed)
    test_indices = set()
    for label, indices in grouped.items():
        shuffled = indices[:]
        generator.shuffle(shuffled)
        test_indices.update(shuffled[: counts[label]])

    train_indices = [i for i in range(len(labels)) if i not in test_indices]
    test_indices = sorted(test_indices)
    return (
        [features[i] for i in train_indices],
        [features[i] for i in test_indices],
        [labels[i] for i in train_indices],
        [labels[i] for i in test_indices],
    )


def evaluate(seed: int = 42, test_size: float = 0.25) -> tuple[object, Evaluation]:
    """Train on a reproducible split and return the fitted model and held-out metrics."""
    features, labels = load_iris()
    x_train, x_test, y_train, y_test = stratified_split(
        features, labels, test_size, seed
    )
    model = NearestCentroid().fit(x_train, y_train)
    predictions = [model.predict(row) for row in x_test]
    correct = sum(expected == actual for expected, actual in zip(y_test, predictions))
    accuracy = correct / len(y_test)
    precisions = []
    for label in sorted(set(y_test)):
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(y_test, predictions)
        )
        predicted_positive = sum(actual == label for actual in predictions)
        precisions.append(true_positive / predicted_positive if predicted_positive else 0)
    return model, Evaluation(
        accuracy=float(accuracy),
        macro_precision=float(sum(precisions) / len(precisions)),
        test_samples=len(y_test),
    )


def predict(features: list[float], seed: int = 42) -> tuple[str, float]:
    """Fit on the built-in dataset and predict one flower's class and confidence."""
    if len(features) != 4:
        raise ValueError("exactly four measurements are required")
    if any(not isinstance(value, (int, float)) for value in features):
        raise ValueError("measurements must be numeric")
    if any(not math.isfinite(value) or value < 0 for value in features):
        raise ValueError("measurements must be finite and non-negative")

    model, _ = evaluate(seed=seed)
    return model.predict_confidence(features)
