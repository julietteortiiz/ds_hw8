"""Public examples: python3 -m unittest -v test_hw08."""
import unittest
import numpy as np
from PCA import PCA
from data_utils import load_digits_data, load_svd_image
from run_svd import low_rank_approximation
from run_pca import components_for_variance


class DataTests(unittest.TestCase):
    def test_digits(self):
        X, y = load_digits_data()
        self.assertEqual(X.shape, (1797, 64))
        self.assertEqual(y.shape, (1797,))
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(((0 <= X) & (X <= 16)).all())
        np.testing.assert_array_equal(np.unique(y), np.arange(10))
        np.testing.assert_array_equal(y[:10], np.arange(10))

    def test_image(self):
        axis = np.linspace(-1., 1., 128)
        xx, yy = np.meshgrid(axis, axis)
        expected = ((xx**2 + yy**2) <= .65**2).astype(float)
        np.testing.assert_array_equal(load_svd_image(), expected)


class PCATests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1., 2.], [2., 1.], [4., 5.], [5., 4.]])

    def test_hand_calculation(self):
        model = PCA(1)
        self.assertIs(model.fit(self.X), model)
        np.testing.assert_allclose(model.mean_, [3., 3.])
        self.assertEqual(model.components_.shape, (1, 2))
        np.testing.assert_allclose(model.singular_values_, [np.sqrt(18)])
        np.testing.assert_allclose(model.explained_variance_, [6.])
        np.testing.assert_allclose(model.explained_variance_ratio_, [.9])
        # Projection matrices avoid the arbitrary sign of singular vectors.
        np.testing.assert_allclose(model.components_.T @ model.components_,
                                   np.full((2, 2), .5))
        restored = model.reconstruct_data(model.transform(self.X))
        self.assertAlmostEqual(float(np.sum((self.X - restored)**2)), 2.)

    def test_full_reconstruction_and_no_mutation(self):
        original = self.X.copy()
        model = PCA(2).fit(self.X)
        Z = model.transform(self.X)
        copy = Z.copy()
        np.testing.assert_allclose(model.reconstruct_data(Z), self.X, atol=1e-12)
        np.testing.assert_allclose(model.components_ @ model.components_.T,
                                   np.eye(2), atol=1e-12)
        np.testing.assert_array_equal(self.X, original)
        np.testing.assert_array_equal(Z, copy)

    def test_stored_mean_and_refit(self):
        model = PCA(2).fit(self.X)
        shifted = self.X + 10
        np.testing.assert_allclose(model.reconstruct_data(model.transform(shifted)),
                                   shifted, atol=1e-12)
        old = model.transform(shifted)
        self.assertGreater(float(np.linalg.norm(old.mean(axis=0))), 1.)
        model.fit(shifted)
        np.testing.assert_allclose(model.mean_, [13., 13.])
        np.testing.assert_allclose(model.transform(shifted).mean(axis=0), 0, atol=1e-12)

    def test_before_fit(self):
        for method in ('transform', 'reconstruct_data'):
            with self.subTest(method=method):
                with self.assertRaises(RuntimeError) as caught:
                    getattr(PCA(1), method)(np.ones((2, 1)))
                # NotImplementedError is a RuntimeError subclass, but is a TODO.
                self.assertIs(type(caught.exception), RuntimeError)

    def test_invalid_fit(self):
        for X in (np.ones((2, 2)), np.ones((1, 2)), np.ones((2, 0)),
                  np.ones(3), np.array([[0., np.nan], [1., 2.]]),
                  np.array([[0., np.inf], [1., 2.]])):
            with self.subTest(X=X), self.assertRaises(ValueError):
                PCA(1).fit(X)
        for k in (0, 3, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                PCA(k).fit(self.X)

    def test_identical_decimal_rows(self):
        # A rounded mean must not manufacture variance from identical rows.
        for X in (np.full((3, 2), .1), np.tile([.1, .3], (3, 1))):
            with self.subTest(X=X), self.assertRaises(ValueError):
                PCA(1).fit(X)

    def test_small_genuine_variation(self):
        X = np.array([[0.], [1e-20], [2e-20]])
        model = PCA(1).fit(X)
        np.testing.assert_allclose(model.explained_variance_ratio_, [1.])
        np.testing.assert_allclose(model.reconstruct_data(model.transform(X)),
                                   X, rtol=1e-12, atol=1e-35)

    def test_transform_shapes_and_values(self):
        model = PCA(1).fit(self.X)
        self.assertEqual(model.transform(self.X[:1]).shape, (1, 1))
        for X in (np.ones(2), np.ones((2, 3)), np.array([[np.nan, 1.]])):
            with self.subTest(X=X), self.assertRaises(ValueError):
                model.transform(X)
        for Z in (np.ones(2), np.ones((2, 2)), np.array([[np.inf]])):
            with self.subTest(Z=Z), self.assertRaises(ValueError):
                model.reconstruct_data(Z)

    def test_wide_matrix_and_error_order(self):
        X = np.array([[0., 1., 2., 0.], [2., 0., 1., 3.], [4., 1., 0., 1.]])
        errors = []
        for k in (1, 2, 3):
            model = PCA(k).fit(X)
            errors.append(np.sum((X - model.reconstruct_data(model.transform(X)))**2))
        self.assertTrue(np.all(np.diff(errors) <= 1e-10))
        self.assertLess(errors[-1], 1e-20)


class ExperimentTests(unittest.TestCase):
    def test_component_counts(self):
        ratios = np.array([.5, .25, .15, .1])
        self.assertEqual(components_for_variance(ratios, .90), 3)
        self.assertEqual(components_for_variance(ratios, .95), 4)
        self.assertEqual(components_for_variance(np.array([.95, .05]), .95), 1)


class SVDTests(unittest.TestCase):
    def test_uncentered_approximation(self):
        A = np.diag([5., 3., 1.])
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        originals = [factor.copy() for factor in (U, s, Vt)]
        np.testing.assert_allclose(low_rank_approximation(U, s, Vt, 1),
                                   np.diag([5., 0., 0.]))
        np.testing.assert_allclose(low_rank_approximation(U, s, Vt, 3), A)
        for factor, original in zip((U, s, Vt), originals):
            np.testing.assert_array_equal(factor, original)


if __name__ == '__main__':
    unittest.main()
