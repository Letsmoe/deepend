from deepend.layers import *

class Conv2D(Layer):
	def __init__(self, filters : int, kernel_size : int or tuple, strides : int or tuple = (1, 1), padding : int or tuple = (1, 1), **kwargs):
		super().__init__(**kwargs)
		self.padding = padding if type(padding) == tuple else (padding, padding)
		self.strides = strides if type(strides) == tuple else (strides, strides)
		self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
		self.filters = filters
		self.kernel = np.random.randn(*self.kernel_size)
		
	def forward(self, inputs):
		# Pad the input matrix
		p = self.padding
		if len(inputs.shape) == 3 and inputs.shape[0] == 1:
			inputs = np.pad(inputs, ((0,0), (p, p), (p, p)), mode='constant', constant_values=0)
		else:
			inputs = np.pad(inputs, p, mode='constant', constant_values=0)
		# Split the input matrix into sizes of the kernel
		k_size = self.kernel_size
		output_height = (self.input_shape[1] + 2 * self.padding[0] - k_size[0]) / self.strides[0] + 1
		output_width = (self.input_shape[2] + 2 * self.padding[1] - k_size[1]) / self.strides[1] + 1
		# Fit size to kernel
		output = np.empty((int(output_height), int(output_width)), dtype=np.float32)
		for row in np.arange(len(inputs) - 1, step = self.strides[0]):
			for col in np.arange(len(inputs[0]) - 1, step = self.strides[1]):
				offset_kernel = inputs[row:row + k_size[0]][0:col + k_size[1], col:col + k_size[1]]
				output[int(np.floor(row / self.strides[0])), int(np.floor(col / self.strides[1]))] = np.sum(offset_kernel * self.kernel)
		return output

	def backward(self, dvalues):
		print(dvalues)
