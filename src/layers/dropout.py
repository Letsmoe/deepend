from deepend.layers import *

class Dropout(Layer):
	def __init__(self, rate : int, seed : int = 0, **kwargs):
		super().__init__(**kwargs)
		self.keep_prob = 1 - rate
		self.mask = None
		self.seed = seed

	def forward(self, inputs: np.array) -> np.array:
		if self.seed:
			np.random.seed(self.seed)
		self.mask = (np.random.rand(*inputs.shape) < self.keep_prob)
		self.output = inputs * self.mask / self.keep_prob

	def backward(self, dvalues: np.array) -> np.array:
		self.dinputs = dvalues * self.mask * self.keep_prob
