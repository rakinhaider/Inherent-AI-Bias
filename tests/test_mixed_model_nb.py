import unittest
from ghost_unfairness.mixed_model_nb import MixedModelNB
import numpy as np
from sklearn.naive_bayes import BernoulliNB


class Test_MixedModelNB(unittest.TestCase):
    x = np.array([[2, 3, 5, 0, 0],
                  [4, 5, 3, 0, 1],
                  [10, 15, 25, 1, 0],
                  [0, 0, 0, 1, 1]])
    y = np.array([1, 1, 0, 1])

    def test_fit(self):
        g = MixedModelNB(multinom_indices=[3, 4], geom_indices=[0, 1, 2])
        g.fit(self.x, self.y)
        assert np.array_equal(g.class_count_, [1, 3])
        assert np.array_equal(g.classes_, [0, 1])
        assert np.array_equal(g.multinomial.classes_, g.geom.classes_)
        assert np.array_equal(g.multinomial.class_count_, g.geom.class_count_)

    def test_joint_log_likelihood(self):
        g = MixedModelNB(multinom_indices=[3, 4], geom_indices=[0, 1, 2])
        g.fit(self.x, self.y)
        jll = g._joint_log_likelihood(self.x)
        assert np.allclose(jll, np.array(
            [[-10.2376151, -8.08881115],
             [-31.22794372, -9.88057062],
             [-12.72485357, -29.77271701],
             [-30.33907132, -4.44656516]],
        ))

    def test_predict(self):
        g = MixedModelNB(multinom_indices=[3, 4], geom_indices=[0, 1, 2])
        g.fit(self.x, self.y)
        pred = g.predict(self.x)
        # print(pred)
        assert all(np.equal(pred, np.array([1, 1, 0, 1])))

    def test_bernouli(self):
        bern_x = np.array([[0], [0], [1], [1]])
        x = np.concatenate([self.x, bern_x], axis=1)
        g = MixedModelNB(multinom_indices=[3, 4], geom_indices=[0, 1, 2],
                         bern_indices=[5])
        g.fit(x, self.y)
        g_jll = g._joint_log_likelihood(x)

        g2 = MixedModelNB(multinom_indices=[3, 4], geom_indices=[0, 1, 2])
        g2.fit(self.x, self.y)
        mod = BernoulliNB(alpha=1e-9)
        mod.fit(bern_x, self.y)
        g2_jll = g2._joint_log_likelihood(self.x)
        bern_jll = mod._joint_log_likelihood(bern_x)

        assert np.allclose(g2_jll + bern_jll - g2.class_log_prior_, g_jll)

    def test_no_multinomial(self):
        bern_x = np.array([[0], [0], [1], [1]])
        x = np.concatenate([self.x, bern_x], axis=1)
        g = MixedModelNB(multinom_indices=[], geom_indices=[0, 1, 2],
                         bern_indices=[3, 4, 5])
        g.fit(x, self.y)
        g._joint_log_likelihood(x)
        g.predict(x)