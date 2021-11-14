import numpy as np

class Loss:
	def regularization_loss(self):
		regularization_loss = 0
		for layer in self.trainable_layers:

			# L1 regularization - weights
			# calculate only when factor greater than 0
			if layer.weight_regularizer_l1 > 0:
				regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

			# L2 regularization - weights
			if layer.weight_regularizer_l2 > 0:
				regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

			# L1 regularization - biases
			# calculate only when factor greater than 0
			if layer.bias_regularizer_l1 > 0:
				regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

			# L2 regularization - biases
			if layer.bias_regularizer_l2 > 0:
				regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
		return regularization_loss

	def remember_trainable_layers(self, trainable_layers):
		self.trainable_layers = trainable_layers

	def calculate(self, output, y, *, include_regularization=False):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		# Add accumulated sum of losses and sample count
		self.accumulated_sum += np.sum(sample_losses)
		self.accumulated_count += len(sample_losses)
		# If just data loss - return it
		if not include_regularization:
			return data_loss
		# Return the data and regularization losses
		return data_loss, self.regularization_loss()

	def calculate_accumulated(self, *, include_regularization=False):
		# Calculate mean loss
		data_loss = self.accumulated_sum / self.accumulated_count
		# If just data loss - return it
		if not include_regularization:
			return data_loss
		# Return the data and regularization losses
		return data_loss, self.regularization_loss()

	def new_pass(self):
		self.accumulated_sum = 0
		self.accumulated_count = 0


from deepend.losses.error import MeanAbsoluteError, MeanSquaredError
from deepend.losses.crossentropy import Categorical_Crossentropy, BinaryCrossentropy, Softmax_Categorical_Crossentropy


LossFunctions = {
	"categorical_crossentropy": Categorical_Crossentropy,
	"sparse_categorical_crossentropy": Categorical_Crossentropy,
	"binary_crossentropy": BinaryCrossentropy,
	"mae": MeanAbsoluteError,
	"mse": MeanSquaredError
}