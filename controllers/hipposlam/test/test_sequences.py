import unittest
import numpy as np

from hipposlam.sequences import Sequences

# $ cd hipposlam
# $ python -m unittest test.test_sequences_propagation

class SequencesTestCase(unittest.TestCase):
    def test_sequences_propagation(self):
        """
        Test given the feature nodes (feature IDs, list of intergers), whether the Sequences Object returns
        the right X matrix and sigmas for each instance of the feature (dictionary of feature ID: list of sigmas).

        """
        seq = Sequences(R=5, L=3)

        fX1 = (
            [1, 49], np.array([[1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0]]),
            {1: [1], 49: [1]}

        )
        fX2 = (
            [1, 49], np.array([[1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 0]]),
            {1: [2, 1], 49: [2, 1]}
        )
        fX3 = (
            [1], np.array([[1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 1]]),
            {1: [3, 2, 1], 49: [3, 2]}
        )
        fX4 = (
            [1], np.array([[1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 1, 1, 1, 1, 1]]),
            {1: [3, 2, 1], 49: [3]}
        )
        fX5 = (
            [1], np.array([[1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0]]),
            {1: [3, 2, 1], 49: []}
        )
        fX6 = (
            [], np.array([[0, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0]]),
            {1: [3, 2], 49: []}
        )
        fX7 = (
            [], np.array([[0, 0, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0]]),
            {1: [3], 49: []}
        )
        fX8 = (
            [], np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]]),
            {1: [], 49: []}
        )
        fX9 = (
            [49], np.array([[0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]]),
            {1: [], 49: [1]}
        )
        fX10 = (
            [49], np.array([[0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0]]),
            {1: [], 49: [2, 1]}
        )

        all_fXs = [fX1, fX2, fX3, fX4, fX5, fX6, fX7, fX8, fX9, fX10]

        for i, fX in enumerate(all_fXs):
            print('fX %d' % (i+1))
            f, X, fsigma = fX
            seq.step(f)
            self.assertListEqual(X.tolist(), seq.X.tolist())
            self.assertDictEqual(fsigma, seq.f_sigma)


if __name__ == '__main__':
    unittest.main()
