import numpy as np
from deepend.activations import *

class LeakyReLU(Activation):
	"""Applies the leaky rectified linear unit activation function to its inputs."""
	def __init__(self, gradient = 0.01):
		"""Initializes new instance of the leaky rectified linear unit activation function

		Args:
			gradient (float, optional): Multiplier of values in range ${\cal Z}(-\infty, 0)$. Defaults to 0.01.
		"""
		self.gradient = gradient
		self.name = 'leaky_relu'

	def forward(self, inputs, training):
		self.inputs = inputs
		# Calculate output values from inputs
		inputs[inputs < 0] = self.gradient * inputs[inputs < 0]
		self.output = inputs

	def backward(self, dvalues):
		"""Applies backward propagation to the outputs of a forward propagation operation

		Args:
			dvalues (np.ndarray): Previously outputted values
		"""
		self.dinputs = dvalues.copy()
		# Leaky zero gradient where input values were negative ()
		self.dinputs[self.inputs <= 0] = (1 / self.gradient) * self.dinputs[self.inputs < 0]

	def predictions(self, outputs):
		return outputs
