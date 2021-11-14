import numpy as np

class Optimizer(object):
	def __init__(self):
		pass
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
	def post_update_params(self):
		self.iterations += 1


from deepend.optimizers.adam import Adam
# from deepend.optimizers.adagrad import Adagrad
from deepend.optimizers.sgd import SGD
from deepend.optimizers.rmsprop import RMSProp