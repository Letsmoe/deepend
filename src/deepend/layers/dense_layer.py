from deepend.layers import *

class Dense(Layer):
	def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0, **kwargs):
		super().__init__(**kwargs)
		#self.construct_activation(activation) # Create the activation function
		# Initialize weights and biases
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		self.shape = (n_inputs, n_neurons)
		self.params = {
			"trainable": (n_inputs + 1) * n_neurons,
			"non-trainable": 0
		}
		self.input_shape = (n_inputs,)
		self.output_shape = (n_neurons,)
		self.name = "dense_1"
		self.weight_regularizer_l1 = weight_regularizer_l1
		self.weight_regularizer_l2 = weight_regularizer_l2
		self.bias_regularizer_l1 = bias_regularizer_l1
		self.bias_regularizer_l2 = bias_regularizer_l2

	# Forward pass
	def forward(self, inputs, training):
		# Remember input values
		self.inputs = inputs
		# Calculate output values from inputs, weights and biases
		self.output = np.dot(inputs, self.weights) + self.biases

	# Backward pass
	def backward(self, dvalues):
		# Gradients on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
		# Gradients on regularization
		# L1 on weights
		if self.weight_regularizer_l1 > 0:
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dweights += self.weight_regularizer_l1 * dL1
		# L2 on weights
		if self.weight_regularizer_l2 > 0:
			self.dweights += 2 * self.weight_regularizer_l2 * self.weights
		# L1 on biases
		if self.bias_regularizer_l1 > 0:
			dL1 = np.ones_like(self.biases)
			dL1[self.biases < 0] = -1
			self.dbiases += self.bias_regularizer_l1 * dL1
		# L2 on biases
		if self.bias_regularizer_l2 > 0:
			self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
		# Gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)