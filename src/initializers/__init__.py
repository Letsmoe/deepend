import numpy as np



class Initializer(object):
	def get_config(self):
		"""
		Returns the config of the initializer as a dict:
		{
			"mean": None,
			"stddev": 1,
			"seed": 42
		}
		"""
		return self._config

	def from_config(self, config):
		"""
		Configures the initializer as specified by the config.
		"""
		for (arg, val) in config:
			self[arg] = val

	def __call__(self, shape):
		return self.construct(shape)



import deepend.initializers.weight as weight
from deepend.initializers.weight import GlorotNormal, RandomNormal, GlorotUniform

"""weight_initializers = {
	"random_normal": weight.RandomNormal,
	"random_uniform": weight.RandomUniform,
	"truncated_normal": weight.TruncatedNormal,
	"zeros": weight.Zeros,
	"ones": weight.Ones,
	"glorot_normal": weight.GlorotNormal,
	"glorot_uniform": weight.GlorotUniform,
	"he_normal": weight.HeNormal,
	"he_uniform": weight.HeUniform,
	"identity": weight.Identity,
	"orthogonal": weight.Orthogonal,
	"constant": weight.Constant,
	"variance_scaling": weight.VarianceScaling
}"""