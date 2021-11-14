from deepend.layers import *

class Flatten(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._shape = ()

	def forward_pass(self, a_prev: np.array) -> np.array:
		self._shape = a_prev.shape
		self.output = np.ravel(a_prev)

	def backward_pass(self, dvalues: np.array) -> np.array:
		self.dinputs = dvalues.reshape(self._shape)
