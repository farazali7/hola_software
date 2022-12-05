from classification.utils.performance_metrics import metrics
import pytest


@pytest.mark.parametrize('labels, preds, result',
                         [([1, 1, 1], [0, 0, 0], 0),
                          ([0, 0, 0], [0, 0, 0],  1),
                          (['invalid'], ['invalid'], 'NA')])
def test_acc(labels, preds, result):
    assert metrics.get_accuracy(labels, preds) == result

