"""Ponte Python do Katu. Rode com scripts/test_python.sh, que compila a extensão antes."""
import math
import tempfile
import unittest
from pathlib import Path

import katu


class ArrayTests(unittest.TestCase):
    def test_types_follow_numpy(self):
        self.assertEqual(katu.array([1, 2, 3]).dtype, "i64")
        self.assertEqual(katu.array([1, 2.5]).dtype, "f64")
        self.assertEqual(katu.array([True, False]).dtype, "bool")
        self.assertEqual((katu.array([1, 2]) / katu.array([2, 4])).tolist(), [0.5, 0.5])
        self.assertEqual((katu.array([True]) + katu.array([True])).tolist(), [True])
        with self.assertRaises(TypeError):
            katu.array([True]) - katu.array([False])

    def test_python_numbers_take_the_array_type(self):
        self.assertEqual((katu.array([1.5], dtype="f32") + 1).dtype, "f32")
        self.assertEqual((katu.array([1, 2]) + 0.5).tolist(), [1.5, 2.5])
        self.assertEqual((10 - katu.array([1, 2])).tolist(), [9, 8])
        self.assertEqual((-katu.array([1, -2])).tolist(), [-1, 2])

    def test_slices_and_masks(self):
        cube = katu.arange(24).reshape([2, 3, 4])
        self.assertEqual(cube.shape, (2, 3, 4))
        self.assertEqual(cube[1, ::-1, 0].tolist(), [20, 16, 12])
        self.assertEqual(cube[0, 1, -1], 7)
        self.assertEqual(cube.T.shape, (4, 3, 2))
        row = katu.array([5, 1, 7, 3])
        self.assertEqual(row[row > 2].tolist(), [5, 7, 3])
        self.assertEqual(row[[3, 0]].tolist(), [3, 5])

    def test_reductions(self):
        m = katu.array([[1.0, 2.0], [3.0, 5.0]])
        self.assertEqual(m.sum(), 11.0)
        self.assertEqual(m.sum(axis=0).tolist(), [4.0, 7.0])
        self.assertAlmostEqual(m.var(), 2.1875)
        self.assertAlmostEqual(m.std(ddof=1), math.sqrt(2.1875 * 4 / 3))
        self.assertEqual(m.argmax(), 3)
        self.assertEqual(katu.array([1.0, float("nan"), 3.0]).argmax(), 1)

    def test_matmul_uses_f32(self):
        a = katu.array([[1, 2], [3, 4]], dtype="f32")
        self.assertEqual((a @ a).tolist(), [[7.0, 10.0], [15.0, 22.0]])
        with self.assertRaises(TypeError):
            katu.array([[1.0]]) @ katu.array([[1.0]])

    def test_join_and_files(self):
        a, b = katu.array([[1, 2]]), katu.array([[3, 4]])
        self.assertEqual(katu.concatenate([a, b]).tolist(), [[1, 2], [3, 4]])
        self.assertEqual(katu.stack([a, b], axis=1).shape, (1, 2, 2))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "a.npy")
            katu.array([[1.5, 2.5]]).save(path)
            self.assertEqual(katu.load(path).tolist(), [[1.5, 2.5]])

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            katu.array([[1, 2], [3]])
        with self.assertRaises(TypeError):
            katu.array(["a"])
        with self.assertRaises(IndexError):
            katu.array([1, 2])[5]


class MathTests(unittest.TestCase):
    def test_calculus(self):
        self.assertEqual(katu.math.derivative("x^2 - 5*x"), "2 * x - 5")
        self.assertEqual(katu.math.antiderivative("x^2 - 5*x"), "x^3 / 3 - 5 * x^2 / 2")
        self.assertAlmostEqual(katu.math.integrate("x^2 - 5*x", 0, 3), -13.5)
        self.assertAlmostEqual(katu.math.root("x^2 - 2", 1.0), math.sqrt(2))
        self.assertAlmostEqual(katu.math.ode("y", 0, 1, 1), math.e, places=9)
        self.assertEqual(katu.math.evaluate("a*x + 1", x=2, a=3), 7.0)
        self.assertEqual(katu.math.latex("x^2/3"), r"\frac{x^{2}}{3}")

    def test_refuses_instead_of_guessing(self):
        for call in (lambda: katu.math.evaluate("1/0"), lambda: katu.math.evaluate("x + 1"),
                     lambda: katu.math.derivative("x +* 2")):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
