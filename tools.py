import numpy as np


def chose_central_rangebins(d_fft, DISCARD_BINS=[14, 15, 16], threshold=0.1, ARM_LEN=30):
	fft_discard = np.copy(d_fft)
	# discard velocities around 0
	for j in DISCARD_BINS:
		fft_discard[j, :] = np.zeros(188)
	# select range bins with the highest energy
	mmax = np.max(fft_discard)
	row, column = np.where(fft_discard == mmax)
	row, column = row[0], column[0]
	left = column
	right = column + 1
	if column > 0:
		for left in range(column - 1, 0, -1):
			if np.max(fft_discard[:, left]) < threshold * mmax:
				break
	if column < 187:
		for right in range(column + 1, 188, 1):
			if np.max(fft_discard[:, right]) < threshold * mmax:
				break
	left = max(left, column - ARM_LEN)
	right = min(right, column + ARM_LEN)
	return left, right