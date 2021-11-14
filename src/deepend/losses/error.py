from deepend.losses import *

class MeanSquaredError(Loss):
	def forward(self, y_pred, y_true):
		self.output = np.mean(np.square(y_true - y_pred), axis=-1)
		return self.output

	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		outputs = len(dvalues[0])
		# Gradient on values
		self.dinputs = -2 * (y_true - dvalues) / outputs
		# Normalize gradient
		self.dinputs = self.dinputs / samples


class MeanAbsoluteError(Loss):
	def forward(self, y_pred, y_true):
		sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
		return sample_losses

	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		outputs = len(dvalues[0])
		# Calculate gradient
		self.dinputs = np.sign(y_true - dvalues) / outputs
		# Normalize gradient
		self.dinputs = self.dinputs / samples