import unittest
from ghost_unfairness.geomnb import GeomNB
import numpy as np


class Test_GeomNB(unittest.TestCase):
    x = np.array([[2, 3, 5], [4, 5, 3], [10, 15, 25], [0, 0, 0]])
    y = np.array([1, 1, 0, 1])

    def test__init__(self):
        g = GeomNB()

    def test_fit(self):
        g = GeomNB()
        g.fit(self.x, self.y)
        assert np.array_equal(g.class_count_, [1, 3])
        assert np.array_equal(g.classes_, [0, 1])
        assert np.array_equal(g.feature_means_, np.array(
            [[10, 15, 25], [2, 8/3, 8/3]]))
        assert np.array_equal(g.feature_prob_, np.array(
            [[1/10, 1/15, 1/25], [1/2, 3/8, 3/8]]))

    def test_joint_log_likelihood(self):
        g = GeomNB()
        g.fit(self.x, self.y)
        jll = g._joint_log_likelihood(self.x)
        assert np.allclose(jll, np.array(
            [[-10.2376151, -8.08881115],
             [-10.50467788, -9.47510552],
             [-12.72485357, -28.67410473],
             [-9.61580548, -2.94248776]]
        ))

    def test_predict(self):
        g = GeomNB()
        g.fit(self.x, self.y)
        pred = g.predict(self.x)
        assert all(np.equal(pred, np.array([1, 1, 0, 1])))
