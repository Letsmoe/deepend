import numpy as np
from deepend.activations import activation_mapping, Activation
from deepend.initializers import *

"""
	Define Main Layer Class
	name -> Used in summary or to reference layer
	dtype -> dtype of the layers weights
"""
class Layer(object):
	# Define a global layer count dict to check how many instances have been created to assign a name.
	layer_count = {}
	def __init__(self, name=None, dtype=None, weight_initializer=None):
		# ------------------ Check if a name was assigned by user ------------------ #
		if name:
			self.name = name
		else:
			class_name = self.__class__.__name__
			if not Layer.layer_count.get(class_name):
				Layer.layer_count[class_name] = 0
			# --------------- Assign the name from the layers classname --------------- #
			self.name = class_name + f"_{Layer.layer_count[class_name]}"
			Layer.layer_count[class_name] += 1

		# --------------------------- Initialize weights --------------------------- #
		if hasattr(self, "input_shape"):
			self.weights = weight_initializers[weight_initializer](self.input_shape)

		# -------------------- Set the dtype of internal weights ------------------- #
		if hasattr(self, "weights") and dtype:
			self.weights.dtype = dtype


	def construct_activation(self, activation):
		if type(activation) == str:
			self.activation = activation_mapping[activation.lower()]()
		elif type(activation) == Activation:
			self.activation = activation
		else:
			raise TypeError(f"Activation must be of kind str or Activation, given {type(activation)}")




"""
	Provide Modules
"""
from deepend.layers.dropout import Dropout
from deepend.layers.dense_layer import Dense
from deepend.layers.convolution import Conv2D
from deepend.layers.flatten import Flatten
from deepend.layers.reshape import Reshape
from deepend.layers.input import Input