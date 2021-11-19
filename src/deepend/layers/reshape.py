from deepend.layers import *

class Reshape(Layer):
	def __init__(self, target_shape, **kwargs):
		super().__init__(**kwargs)
		self.output_shape = target_shape
	
	def forward(self, inputs):
		self.output = np.reshape(inputs, self.output_shape)

	def backward(self, dvalues):
		self.dinputs = np.reshape(dvalues, self.input_shape)
