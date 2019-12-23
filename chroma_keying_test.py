import pytest
import chroma_keying
import numpy as np


def test_dist_rgb():
    a = np.array([123, 230, 5])
    b = np.array([13, 24, 255])
    dist = chroma_keying.dist_rgb(a, b)
    assert dist == pytest.approx(291.1, 0.1)
