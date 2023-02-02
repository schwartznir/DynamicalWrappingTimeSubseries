# Implementation of SPRING algorithm

# 
# We implement the SPRING algorithm [1], for finding a reference (a.k.a "query") wavelet sampled at m points,
# inside a large 1D stream having a prefixes of increasing lengths n. The reference is found if the similairy 
# w.r.t a DTW metric is small enough. The algorithm possesses a significant imprvoment
# over a naive approach using the classical DTW solution. 
#
# Remark: the norm chosen in the DTW metric is L^2 for sake of conformality with the paper
# 
# [1] cf. <<Stream monitoring under the time warping distance>> by Faloutsos et al.

import numpy as np


def spring_tick(idx: int, stream: list, ref: list, topvec: list, botvec: list, epsilon: float) -> tuple:
	# Following the lines of formulae (7)-(8) suggested in the paper, we implement a SPRING session
	# (per new datum) verbatim. The precision is updated after each session to be the average
	# between previous epsilon and the average of the new DTW differences.
	# time complexity: O(m)
	# space complexity: O(m)
	# We assume that m > 0
	if not idx:
		topvec, botvec = calculate_vecs(stream, ref, idx, topvec, botvec)
	m = len(stream)
	eps = sum([pair[0] / m for pair in topvec]) + sum([pair[0] / m for pair in botvec]) # We choose epsilon to be a moving averaging
	dmin = min([pair[0] for pair in topvec])
	res = []
	tend = idx
	tstart = topvector[m][1]

	if dmin <= eps:
		toReport = True
		for jdx in range(1, m + 1):
			currd, currs = topvec[jdx]
			if currd < dmin and currs <= tend:
				toReport = False
		if toReport:
				res.append([dmin, tstart, tend])
		dmin = float('inf')
		for kdx in range(1, m + 1):
			if topvector[kdx][1] <= tend:
				topvector[kdx][0] = float('inf')

	if topvector[m][0] <= eps and topvector[m][0] < dmin:
		dmin = topvector[m][0]
		tstart = topvector[m][1]
		tend = idx


def init_vectors(datum: float, ref: list) -> tuple:
	# the function initalizes the two initial row vectors of length |ref| + 1. 
	# One is (0,\infty,...\infty) and the other is the vector of cummulative sums 
	# of the ref vector
	# time complexity: O(m)
	# space complexity: O(1)
	m = ref.size
	botvector = float('inf') * np.ones(m + 1)
	botvector[0] = 0
	topvector = np.zeros(m + 1)
	currsum = 0
	for idx in range(1, m + 1):
		currsum += (ref[idx] - datum) ^ 2
		topvector[idx] = (currsum, idx)
	return first_vector, second_vector


def calculate_vecs(stream: list, ref: list, idx: int, topvec: list, botvec: list) -> tuple:
	# calculate the vectors replacing the (D,S) table
	# time complexity: O(m)
	# space complexity: O(1)
	botvec = topvec # bottom vector is now the previosuly calculated "top" vector=
	topvec[0] = (0, 0)
	for jdx in range(1, m + 1):
		leftd, lefts = topvec[jdx- 1]
		botd, bots = botvec[jdx]
		diagd, diags = botvec[jdx - 1]
		bestd = min(diagd, botd, leftd)
		if bestd == diagd:
			currs = diags
		elif bestd == leftd:
			currs = lefts
		else:
			currs = bots
		currd = (stream[idx] - ref[jdx]) ^ 2 + bestd
		topvec[jdx] = (currd, currs)


def spring(ref: list, stream_pref: list) -> list:
	# The main function initating the SPRING algorithm on a given pair of (stream, query),
	# analysing the stream by ticks. If a subsequence "close" in the DTW metric is found append it to the 
	# output. 
	# time complexity: O(mn) with m, n being the length of the ref and stream_prefix correspondingly
	# space complexity: (m)
	subseqs = []
	m = len(ref)
	n = len(stream_pref)
	eps = 0
	first_vector, second_vector = init_vectors(ref)
	for idx in range(n):
		first_vector, second_vector, eps, result = spring_tick(stream_pref[idx], ref, first_vector, second_vector, eps)
		if result:
			subseqs.append(result)
	return seqs


if __name__ == '__main__':
        expected_ouput = [2, 5, [12, 6, 10, 6]]
        stream = [5, 12, 6, 10, 6, 5, 13]
        ref = [11, 6, 9, 4]
        assert expected_ouput == spring(stream, ref), \
            f"Test no. 0 failed.\n\t Test content:({stream, ref}), Expected result {expected_ouput}"
