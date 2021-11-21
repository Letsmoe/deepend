from deepend.initializers import *

class RandomNormal(Initializer):
	"""Initializer from a random normal distribution"""
	def __init__(self, **kwargs):
		self._config = kwargs

	def construct(self, shape):
		"""Random Tensor with values from a standard normal distribution

		Args:
			shape (tuple): Shape of the output tensor (N-Dimensional).
		"""
		if self._generator:
			self.weights = self._generator.normal(**self._config)
		else:
			self.weights = np.random.randn(*shape)

class GlorotNormal(Initializer):
	def __init__(self, **kwargs):
		self._config = kwargs
		

	def construct(self, shape):
		"""Xavier (Glorot) initialization from a random normal distribution filling the initialization tensor with values sampled from ${\cal N}(0,std^2)$ where
		$$
			std = gain * \sqrt{\dfrac{2}{fan_{in}+fan_{out}}}
		$$

		Args:
			shape (tuple): Shape of the output tensor (N-Dimensional).
		"""
		if self._config.get("seed"):
			np.random.seed(self._config["seed"])
		# --------------------- Adjust values to distribution --------------------- #
		sqrt_6 = np.sqrt(6)
		sqrt_in = np.sqrt(shape[0] + shape[-1])
		self.weights = (sqrt_6 / sqrt_in) * np.random.randn(*shape)

class GlorotUniform(Initializer):
	"""Xavier (Glorot) initialization from a random unfiform distribution over 
	$$
		\pm\dfrac{\sqrt{6}}{\sqrt{n\_i+n\_{i+1}}}
	$$

	Args:
		shape (tuple): Shape of the output tensor (N-Dimensional).
	"""
	def __init__(self, **kwargs):
		self._config = kwargs

	def construct(self, shape):
		if self._config.get("seed"):
			np.random.seed(self._config["seed"])
		# --------------------- Adjust values to distribution --------------------- #
		root = np.sqrt(6 / (shape[0] + shape[-1]))
		self.weights = (2 + root) * np.random.random(shape)