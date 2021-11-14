from deepend.initializers import *

class RandomNormal(Initializer):
	def __init__(self, **kwargs):
		self._config = kwargs

	def construct(self, shape):
		if self._generator:
			return self._generator.normal(**self._config)
		else:
			self.weights = np.random.randn(*shape)

class GlorotNormal(Initializer):
	def __init__(self, **kwargs):
		self._config = kwargs
		

	def construct(self, shape):
		if self._config.get("seed"):
			np.random.seed(self._config["seed"])
		# --------------------- Adjust values to distribution --------------------- #
		sqrt_6 = np.sqrt(6)
		sqrt_in = np.sqrt(shape[0] + shape[-1])
		self.weights = (sqrt_6 / sqrt_in) * np.random.randn(*shape)

class GlorotUniform(Initializer):
	def __init__(self, **kwargs):
		self._config = kwargs

	def construct(self, shape):
		if self._config.get("seed"):
			np.random.seed(self._config["seed"])
		# --------------------- Adjust values to distribution --------------------- #
		root = np.sqrt(6 / (shape[0] + shape[-1]))
		self.weights = (2 + root) * np.random.random(shape)