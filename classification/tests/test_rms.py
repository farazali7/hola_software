from classification.utils.feature_extraction import features
import pytest


@pytest.mark.parametrize('input_array, result',
                         [([3, 4, 5], 2),
                          ([0, 0, 0], 0),
                          (['invalid'], 'NA')])
def test_rms(input_array, result):
    assert features.rms(input_array) == result

