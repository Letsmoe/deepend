from deepend.layers import *

class Input(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def forward(self, inputs, training):
		self.output = inputs