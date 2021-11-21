import numpy as np
from deepend.activations import activation_mapping, Activation
from deepend.initializers import *

class Layer(object):
	# Define a global layer count dict to check how many instances have been created to assign a name.
	layer_count = {}
	def __init__(self, name=None, dtype="float32", weight_initializer=RandomNormal):
		"""Initializes a new layer object, can be inherited to provide additional support for user specified layers

		Args:
			name (str, optional): A name which will be used to identify the layer, auto generated if not specified. Defaults to None.
			dtype (str, optional): Datatype of the layers properties. Defaults to float32.
			weight_initializer (Initializer, optional): Custom weight initialization function. Defaults to RandomNormal.
		"""
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


	def _construct_activation(self, activation):
		if type(activation) == str:
			self.activation = activation_mapping[activation.lower()]()
		elif type(activation) == Activation:
			self.activation = activation
		else:
			raise TypeError(f"Activation must be of kind str or Activation, given {type(activation)}")



from deepend.layers.dropout import Dropout
from deepend.layers.dense_layer import Dense
from deepend.layers.convolution import Conv2D
from deepend.layers.flatten import Flatten
from deepend.layers.reshape import Reshape
from deepend.layers.input import Input