import numpy as np


def calc_dists(preds, target, normalize):
	preds  =  preds.astype(np.float32)
	target = target.astype(np.float32)
	dists  = np.zeros((preds.shape[1], preds.shape[0]))

	for n in range(preds.shape[0]):
		for c in range(preds.shape[1]):
			if target[n, c, 0] > 1 and target[n, c, 1] > 1:
				normed_preds   =  preds[n, c, :] / normalize[n]
				normed_targets = target[n, c, :] / normalize[n]
				dists[c, n]    = np.linalg.norm(normed_preds - normed_targets)
			else:
				dists[c, n]    = -1

	return dists


def dist_acc(dists, threshold = 0.5):
	dist_cal     = np.not_equal(dists, -1)
	num_dist_cal = dist_cal.sum()

	if num_dist_cal > 0:
		return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
	else:
		return -1


def get_max_preds(batch_heatmaps):
	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width      = batch_heatmaps.shape[3]

	# batch_heatmaps.shape = (1, 17, 46, 46)
	# 1 mert validation, 8 ha training
	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	# heatmaps_reshaped.shape = (1, 17, 2116)
	"""
	ez a sor veszi a batch-ben lévő összes képet (validation esetén ez csak 1)
	az összes keypointot (ami valamiért 17 a 16 helyett)
	a kép pixeleit átpakolja egyetlen sorba
	"""

	index_max_heat_of_joints = np.argmax(heatmaps_reshaped, 2)
	# Returns the indices of the maximum values along an axis.
	"""
	a 2. tengely, az ízületek mentén kiválasztja a maximumot
	tehát minden ízülethez azt a pixelt, ahol a maximum heat van
	"""

	max_heat_of_joints = np.amax(heatmaps_reshaped, 2)
	# Return the maximum of an array or maximum along an axis.

	max_heat_of_joints = max_heat_of_joints.reshape((batch_size, num_joints, 1))
	index_max_heat_of_joints = index_max_heat_of_joints.reshape((batch_size, num_joints, 1))

	# a predictions az indexből származik
	# a prediction_mask a hőértékből
	predictions = np.tile(index_max_heat_of_joints, (1,1,2)).astype(np.float32)
	"""np.tile(array, repetitions)
	constructs a new array by repeating 'array' 'repetitions' times
	np.tile([0, 1, 2], (1, 1, 2))
	output = [[[0 1 2 0 1 2]]]
	"""

	predictions[:,:,0] = (predictions[:,:,0]) % width
	"""Modulo Operator, %
	It returns the remainder of dividing the left operand by right operand.
	It's used to get the remainder of a division problem
	"""
	predictions[:,:,1] = np.floor((predictions[:,:,1]) / width)

	# masks for each detected instance
	predictions_mask = np.tile(
		np.greater(max_heat_of_joints, 0.0),
		(1,1,2)
	)
	"""numpy.greater(x1, x2)
	Return the truth value of (x1 > x2) element-wise.
	$ numpy.greater([1, -1, 0], 0.0)
	array([ True, False, False])
	"""

	predictions_mask = predictions_mask.astype(np.float32)
	"""
	$ numpy.array([True, False]).astype(numpy.float32)
	array([1., 0.], dtype=float32)
	"""

	predictions *= predictions_mask
	"""
	nullázza ki, azaz kvázi törölje azokat az index értékeket,
	ahol a max hőérték 0 volt
	"""
	return predictions


def accuracy(output, target, threshold_PCK, threshold_PCKh, dataset,
			 threshold=0.5):
	# output = heat = self.model(input) = Unipose.forward(input)
	idx  = list(range(output.shape[1]))  # output.shape[1]: 17
	# numClasses = output.shape[1]
	normalize = 1.0

	preds  = get_max_preds(output)
	target = get_max_preds(target)

	height    = output.shape[2]
	width     = output.shape[3]
	normalize = np.ones((preds.shape[0], 2)) * np.array([height, width]) / 10

	dists = calc_dists(preds, target, normalize)

	acc     = np.zeros((len(idx)))
	# acc     = np.zeros((numClasses))
	avg_acc = 0
	cnt     = 0
	visible = np.zeros((len(idx)))
	# visible = np.zeros((numClasses))

	for i in range(len(idx)):
		acc[i] = dist_acc(dists[idx[i]], threshold)
		if acc[i] >= 0:
			avg_acc = avg_acc + acc[i]
			cnt    += 1
			visible[i] = 1
		else:
			acc[i] = 0
	
	# for i in range(numClasses):
	# 	acc[i] = dist_acc(dists[i], threshold)
	# 	if acc[i] >= 0:
	# 		avg_acc = avg_acc + acc[i]
	# 		cnt    += 1
	# 		visible[i] = 1
	# 	else:
	# 		acc[i] = 0

	avg_acc = avg_acc / cnt if cnt != 0 else 0

	if cnt != 0:
		acc[0] = avg_acc

	# PCKh
	PCKh = np.zeros((len(idx)))
	# PCKh = np.zeros((numClasses))
	avg_PCKh = 0

	if dataset == "MPII":
		headLength = np.linalg.norm(target[0,9,:] - target[0,10,:])
	else:
		raise NotImplementedError

	for i in range(len(idx)):
		PCKh[i] = dist_acc(dists[idx[i]], threshold_PCKh*headLength)
		if PCKh[i] >= 0:
			avg_PCKh = avg_PCKh + PCKh[i]
		else:
			PCKh[i] = 0
	
	# for i in range(numClasses):
	# 	PCKh[i] = dist_acc(dists[i], threshold_PCKh*headLength)
	# 	if PCKh[i] >= 0:
	# 		avg_PCKh = avg_PCKh + PCKh[i]
	# 	else:
	# 		PCKh[i] = 0

	avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

	if cnt != 0:
		PCKh[0] = avg_PCKh

	# PCK
	PCK = np.zeros((len(idx)))
	avg_PCK = 0

	if dataset == "MPII":
		torso = np.linalg.norm(target[0, 7, 0] - target[0, 8, 0])
	else:
		raise NotImplementedError

	for i in range(len(idx)):
		PCK[i] = dist_acc(dists[idx[i]], threshold_PCK*torso)

		if PCK[i] >= 0:
			avg_PCK = avg_PCK + PCK[i]
		else:
			PCK[i] = 0
			
	# for i in range(numClasses):
	# 	PCK[i] = dist_acc(dists[i], threshold_PCK*torso)

	# 	if PCK[i] >= 0:
	# 		avg_PCK = avg_PCK + PCK[i]
	# 	else:
	# 		PCK[i] = 0

	avg_PCK = avg_PCK / cnt if cnt != 0 else 0

	if cnt != 0:
		PCK[0] = avg_PCK

	return acc, PCK, PCKh, cnt, preds, visible