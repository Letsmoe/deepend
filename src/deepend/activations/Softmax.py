import numpy as np
from deepend.activations import *


class Softmax(Activation):
	def __init__(self):
		self.name = 'softmax'

	def forward(self, inputs, training):
		self.inputs = inputs
		# Get unnormalized probabilities
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

	def backward(self, dvalues):
		self.dinputs = np.empty_like(dvalues)
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			# Flatten output array
			single_output = single_output.reshape(-1, 1)
			# Calculate Jacobian matrix of the output
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
			# Calculate sample-wise gradient
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

	def predictions(self, outputs):
		return np.argmax(outputs, axis=1)