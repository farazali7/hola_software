from classification.utils.performance_metrics import metrics
import pytest


@pytest.mark.parametrize('labels, preds, result',
                         [([1, 1, 1], [1, 1, 0], 0.66),
                          ([0, 0, 0], [0, 0, 0],  0.0),
                          (['invalid'], ['invalid'], 'NA')])
def test_spec(labels, preds, result):
    assert metrics.get_recall(labels, preds, pos_label=1) == result

