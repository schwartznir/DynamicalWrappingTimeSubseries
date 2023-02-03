# Implementation of SPRING algorithm

# 
# We implement the SPRING algorithm [1], for finding a reference (a.k.a "query") wavelet sampled at m points,
# inside a large 1D stream having a prefixes of increasing lengths n. The reference is found if the similarity
# w.r.t a DTW metric is small enough. The algorithm possesses a significant improvement
# over a naive approach using the classical DTW solution. 
#
# Remark: the norm chosen in the DTW metric is L^2 for sake of conformity with the paper
# 
# [1] cf. <<Stream monitoring under the time warping distance>> by Faloutsos et al.


def spring_tick(idx: int, stream: list, ref: list, topvec: list,
				botvec: list, epsilon: float, tstart: int, tend: int, dmin: float) -> tuple:
	# Following the lines of formulae (7)-(8) suggested in the paper, we implement a SPRING session
	# (per new datum) verbatim. The precision is updated after each session to be the average
	# between previous epsilon and the average of the new DTW differences.
	# time complexity: O(m)
	# space complexity: O(m)
	# We assume that m > 0
	if idx:
		botvec, topvec, dmin = calculate_vecs(stream, ref, idx, topvec, dmin)
	m = len(ref)
	# We choose epsilon to be a moving averaging giving the "non-external" columns higher weight
	tmp = sum([pair[0] / m for pair in topvec])
	if botvec[-1][-1] != float('inf'):
		tmp += sum([pair[0] / m for pair in botvec])
	eps = min(epsilon, tmp) // 3
	res = []

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
				if topvec[kdx][1] <= tend:
					topvec[kdx] = (float('inf'), topvec[kdx][1])

	newstart = tstart
	newend = tend

	if topvec[m][0] <= eps and topvec[m][0] < dmin:
		dmin = topvec[m][0]
		newstart = topvec[m][1]
		newend = idx
	return topvec, botvec, eps, newstart, newend, dmin, res


def init_vectors(datum: float, ref: list) -> tuple:
	# the function initializes the two initial row vectors of length |ref| + 1.
	# One is (0,\infty,...\infty) and the other is the vector of cumulative sums
	# of the ref vector
	# time complexity: O(m)
	# space complexity: O(1)
	m = len(ref)
	botvect = [(float('inf'), float('inf'))] * (m + 1)
	botvect[0] = (0, 0)
	topvect = [(0, 1)] * (m + 1)
	currsum = 0
	dmin = float('inf')
	for idx in range(1, m + 1):
		currsum += (ref[idx - 1] - datum) ** 2
		topvect[idx] = (currsum, 1)
		dmin = min(dmin, currsum)
	return botvect, topvect, dmin


def calculate_vecs(stream: list, ref: list, idx: int, prevtop: list, dmin: int) -> tuple:
	# calculate the vectors replacing the (D,S) table
	# time complexity: O(m)
	# space complexity: O(1)
	tmp = prevtop.copy()  # bottom vector is now the previously calculated "top" vector=
	prevtop[0] = (0, idx + 1)
	m = len(ref)
	for jdx in range(1, m + 1):
		leftd, lefts = prevtop[jdx - 1]
		botd, bots = tmp[jdx]
		diagd, diags = tmp[jdx - 1]
		bestd = min(diagd, botd, leftd)
		if bestd == leftd:
			currs = lefts
		elif bestd == diagd:
			currs = diags
		else:
			currs = bots
		currd = (stream[idx] - ref[jdx - 1]) ** 2 + bestd
		dmin = min(dmin, currd)
		prevtop[jdx] = (currd, currs)
	botvec = tmp.copy()
	return botvec, prevtop, dmin


def spring(stream_pref: list, ref: list) -> list:
	# The main function initiating the SPRING algorithm on a given pair of (stream, query),
	# analysing the stream by ticks. If a subsequence "close" in the DTW metric is found append it to the 
	# output. 
	# time complexity: O(mn) with m, n being the length of the ref and stream_prefix correspondingly
	# space complexity: (m)
	subseqs = []
	n = len(stream_pref)
	eps = float('inf')
	bottom_vector, top_vector, dmin = init_vectors(stream_pref[0], ref)
	tstart = 1
	tend = len(stream_pref)
	for idx in range(n):
		top_vector, bottom_vector, eps, tstart, tend, dmin, result = spring_tick(idx, stream_pref, ref,
																		top_vector, bottom_vector,
																		eps, tstart, tend, dmin)
		if result:
			subseqs.append(result)
	return subseqs


if __name__ == '__main__':
	expected_output = [2, 5, [12, 6, 10, 6]]
	stream = [5, 12, 6, 10, 6, 5, 13]
	ref = [11, 6, 9, 4]
	# calculate_vecs(stream, ref, 2, [(0, 2), (1,2), (37, 2), (46, 2), (100,2)], [(0, 1),
	# assert expected_output == spring(stream, ref), \
	#	f"Test no. 0 failed.\n\t Test content:({stream, ref}), Expected result {expected_output}"
	print(spring(stream, ref))
