from classification.utils.feature_extraction import features
import pytest


@pytest.mark.parametrize('input_array, result',
                         [([-3, -4, -5], 4),
                          ([0, 0, 0], 0),
                          (['invalid'], 'NA')])
def test_mav(input_array, result):
    assert features.mav(input_array) == result

