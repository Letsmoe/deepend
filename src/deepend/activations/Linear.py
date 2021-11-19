import numpy as np
from deepend.activations import *

class Linear(Activation):
	def forward(self, inputs, training):
		self.inputs = inputs
		self.output = inputs

	def backward(self, dvalues):
		# derivative is 1, 1 * dvalues = dvalues
		self.dinputs = dvalues.copy()

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs
