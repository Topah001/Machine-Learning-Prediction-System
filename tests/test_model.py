import pytest

from ml_prediction.model import evaluate, predict


def test_evaluation_is_reproducible_and_reports_metrics():
    _, first = evaluate()
    _, second = evaluate()
    assert first == second
    assert 0 <= first.accuracy <= 1
    assert 0 <= first.macro_precision <= 1
    assert first.test_samples == 38


def test_prediction_returns_known_species_and_confidence():
    label, confidence = predict([5.1, 3.5, 1.4, 0.2])
    assert label == "setosa"
    assert 0 <= confidence <= 1


@pytest.mark.parametrize(
    "features",
    [[1, 2, 3], [1, 2, 3, -1]],
)
def test_prediction_rejects_invalid_measurements(features):
    with pytest.raises(ValueError):
        predict(features)


def test_evaluation_rejects_invalid_test_fraction():
    with pytest.raises(ValueError, match="test_size"):
        evaluate(test_size=1)
