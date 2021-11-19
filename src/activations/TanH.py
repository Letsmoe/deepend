import numpy as np
from deepend.activations import *

class TanH(Activation):
	def __init__(self):
		self.name = "tanh"
		
	def forward(self, inputs, training):
		self.inputs = inputs
		self.output = np.tanh(inputs)
	
	def backward(self, dvalues):
		self.dinputs = 1 - dvalues**2
	
	def predictions(self, outputs):
		return (outputs > 0) * 1