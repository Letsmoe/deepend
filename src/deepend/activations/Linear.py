import numpy as np
from deepend.activations import *

class Linear(Activation):
	"""Linear activation function, rectifying to a linearity with $T_{out} \in {\cal Z}(-\infty, \infty)$ where $T_{out} = T_{in}$"""
	def forward(self, inputs, training):
		"""Forward propagation function of linearity

		Args:
			inputs (np.ndarray): Values to be rectified
			training (bool): Whether to train the model by applying backpropagation to the layer
		"""
		self.inputs = inputs
		self.output = inputs

	def backward(self, dvalues):
		"""Backward propagation function of linearity

		Args:
			dvalues (np.ndarray): Output values of the preceding layers backpropagation routine
		"""
		# derivative is 1, 1 * dvalues = dvalues
		self.dinputs = dvalues.copy()

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs
