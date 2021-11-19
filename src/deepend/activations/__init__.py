class Activation(object):
	def __init__(self):
		pass

from deepend.activations.Softmax import Softmax
from deepend.activations.LeakyReLU import LeakyReLU
from deepend.activations.Linear import Linear
from deepend.activations.TanH import TanH
from deepend.activations.Sigmoid import Sigmoid
from deepend.activations.ReLU import ReLU



activation_mapping = {
	"softmax": Softmax,
	"leaky_relu": LeakyReLU,
	"linear": Linear,
	"tanh": TanH,
	"sigmoid": Sigmoid,
	"relu": ReLU,
}