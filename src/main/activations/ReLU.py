import numpy as np
from deepend.activations import *

class ReLU(Activation):
	def __init__(self):
		self.name = 'relu'

	def forward(self, inputs, training):
		self.inputs = inputs
		# Calculate output values from inputs
		self.output = np.maximum(0, inputs)

	def backward(self, dvalues):
		self.dinputs = dvalues.copy()
		# Zero gradient where input values were negative
		self.dinputs[self.inputs <= 0] = 0

	def predictions(self, outputs):
		return outputs
