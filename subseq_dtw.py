# Naive subsequence DFW -- implementation

# We implement a variation of the Dynamical Time-Series Wrapping (aka DTW) which replaces
# the best alignment problem with the best fit of a signal to another one on which one
# has more data. ## POORLY WRITTEN!

import numpy as np
import librosa

frnd_gaussian = np.random.randn(1, 5098)
srnd_gaussian = np.random.randn(1, 2023)

TESTS = [[np.array([2, 4, 0, 4, 0, 0, 5, 2]), np.array([3, 0, 6])],
         [np.array([1, 2, 3, 4, 5, 7]), np.array([1, 2, 3])],
         [np.array([1, 2, 3, 4, 5, 7]), np.array([5, 7])],
         [np.array([1, 2, 3, 4, 5, 7]), np.array([4])],
         [np.array([7, 2, 7, 4, 5, 7]), np.array([42])],
         [np.array([1, 1, 1, 1, 1, 1]), np.array([-90, 90, 0])],
         [frnd_gaussian, srnd_gaussian]]
N = len(TESTS)


def calculate_cost_matrix(rowvec: np.array, colvec: np.array, distance) -> np.ndarray:
    # Calculate the initial cost matrix at time and space complexity of |rowvec|*|colvec| through
    # a given distance function
    return np.array([[distance(colentry - rowentry) for rowentry in rowvec] for colentry in colvec])


def calculate_acc_cost_matrix(costs: np.ndarray) -> np.ndarray:
    # Compute the accumulated cost matrix obtained from modifying the DTW naive implementation to permit
    # beginning the pattern matching not from the 0th datum of the time series.
    # time and space complexities: O(|costs|)
    n, m = costs.shape
    accmat = np.zeros([n, m])
    accmat[0, :] = costs[0, :]
    accmat[:, 0] = np.cumsum(costs[:, 0])
    for idx in range(1, n):
        for jdx in range(1, m):
            accmat[idx][jdx] = costs[idx][jdx] + min(accmat[idx - 1][jdx], accmat[idx][jdx - 1],
                                                     accmat[idx - 1][jdx - 1])
    print(accmat)
    return accmat


def subseq_dtw(signal: np.array, subsignal: np.array, distance=np.abs) -> list:
    # the function gets a 1D signal and another 1D shorter subsignal both represented as a list of np.float
    # Using a variation on the classical DTW one checks which path wrapping (excluding the beginning and the
    # ending of the larger signal) is the optimal.
    # time complexity: O(nm) where n = |signal|, m = |subsignal|
    # space complexity: O(nm) for storing the cost matrix
    cost_matrix = calculate_cost_matrix(signal, subsignal, distance)
    acc_cost_matrix = calculate_acc_cost_matrix(cost_matrix)
    ending_pt = np.argmin(acc_cost_matrix[- 1, :])
    currpt = (len(subsignal) - 1, ending_pt)
    currn, currm = currpt
    path = [currpt]
    while currpt[0] > 0:
        if not currm:
            currpt = (currn - 1, 0)
        else:
            mincandidate = min(acc_cost_matrix[currn - 1][currm - 1], acc_cost_matrix[currn][currm - 1],
                               acc_cost_matrix[currn - 1][currm])
            if mincandidate == acc_cost_matrix[currn - 1][currm - 1]:
                currpt = (currn - 1, currm - 1)
            elif mincandidate == acc_cost_matrix[currn][currm - 1]:
                currpt = (currn, currm - 1)
            else:
                currpt = (currn - 1, currm)
        currn, currm = currpt
        path.append(currpt)
    # path.reverse()
    return path


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for idx in range(N):
        currsignal, currsub = TESTS[idx]
        expected_ouput = librosa.sequence.dtw(currsub, currsignal, subseq=True)[1]
        expected_ouput = [(line[0], line[1]) for line in expected_ouput]
        assert expected_ouput == subseq_dtw(currsignal, currsub), \
            f"Test no. {idx} failed.\n\t Test content:({currsignal, currsub}), Expected result {expected_ouput}"
