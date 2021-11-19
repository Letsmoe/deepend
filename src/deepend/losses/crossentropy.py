from deepend.losses import *

class Categorical_Crossentropy(Loss):
	def forward(self, y_pred, y_true):
		# Number of samples in a batch
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		# Probabilities for target values - only if categorical labels
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[
				range(samples),
				y_true
			]
		# Mask values - only for one-hot encoded labels
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(
				y_pred_clipped * y_true,
				axis=1
			)
		# Losses
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		labels = len(dvalues[0])
		# If labels are sparse, turn them into one-hot vector
		if len(y_true.shape) == 1:
			y_true = np.eye(labels)[y_true]
		# Calculate gradient
		self.dinputs = -y_true / dvalues
		# Normalize gradient
		self.dinputs = self.dinputs / samples

class BinaryCrossentropy(Loss):
	def forward(self, y_pred, y_true):
		# Clip data to prevent division by 0
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		# Calculate sample-wise loss
		sample_losses = -(y_true * np.log(y_pred_clipped) +
						  (1 - y_true) * np.log(1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis=-1)
		return sample_losses

	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		outputs = len(dvalues[0])
		# Clip data to prevent division by 0
		clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
		# Calculate gradient
		self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
		# Normalize gradient
		self.dinputs = self.dinputs / samples

class Softmax_Categorical_Crossentropy():
	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		# If labels are one-hot encoded,
		# turn them into discrete values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)
		self.dinputs = dvalues.copy()
		# Calculate gradient
		self.dinputs[range(samples), y_true] -= 1
		# Normalize gradient
		self.dinputs = self.dinputs / samples