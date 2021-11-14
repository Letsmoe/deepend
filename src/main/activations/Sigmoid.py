import numpy as np
from deepend.activations import *

class Sigmoid(Activation):
	def forward(self, inputs, training):
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(-inputs))

	def backward(self, dvalues):
		self.dinputs = dvalues * (1 - self.output) * self.output

	def predictions(self, outputs):
		return (outputs > 0.5) * 1