from deepend.losses import *
from deepend.optimizers import *
from deepend.layers import *
from Metrics import *
from deepend.activations import *
import numpy as np
from pgrs import pgrs

class Model(object):
	def __init__(self, layers : list = []):
		self.layers = layers
		self.Optimizers = {
			"adam": Adam,
			"sgd": SGD
		}
		self.Losses = {
			"categorical_crossentropy": Categorical_Crossentropy,
		}
		self.softmax_classifier_output = None

	def predict(self, X, *, batch_size=None):
		# Default value if batch size is not being set
		prediction_steps = 1
		# Calculate number of steps
		if batch_size is not None:
			prediction_steps = len(X) // batch_size
			# Dividing rounds down.
			if prediction_steps * batch_size < len(X):
				prediction_steps += 1
		output = []
		for step in range(prediction_steps):
			# If batch size is not set - train using one step and full dataset
			if batch_size is None:
				batch_X = X
			else:
				batch_X = X[step*batch_size:(step+1)*batch_size]
			batch_output = self.forward(batch_X, training=False)
			# Append batch prediction to the list of predictions
			output.append(batch_output)
		# Stack and return results
		return np.vstack(output)

	def forward(self, X, training):
		self.input_layer.forward(X, training)
		for layer in self.layers:
			layer.forward(layer.prev.output, training)
		return layer.output

	def backward(self, output, y):
		# If softmax classifier
		if self.softmax_classifier_output is not None:
			# Set dinputs property
			self.softmax_classifier_output.backward(output, y)
			self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
			for layer in reversed(self.layers[:-1]):
				layer.backward(layer.next.dinputs)
		else:
			self.loss.backward(output, y)
			# Call backward method going through all the objects
			for layer in reversed(self.layers):
				layer.backward(layer.next.dinputs)

	def add(self, layer: Layer):
		self.layers.append(layer)

	def lookup(self, name: str or Optimizer, Lookup : dict):
		if type(name) == str:
			return Lookup[name.lower()]()
		else:
			return name

	def compile(self, optimizer : str or Optimizer, loss: str or Loss, metrics: list = ["accuracy"], callbacks : list = []):
		self.optimizer = self.lookup(optimizer, self.Optimizers)
		self.loss = self.lookup(loss, self.Losses)
		self.metrics = metrics
		self.callbacks = callbacks
		# SECTION Initialize Model
		self.input_layer = Input()
		layer_count = len(self.layers)
		# Initialize a list containing trainable layers:
		self.trainable_layers = []
		# Iterate the objects
		for i in range(layer_count):
			# If it's the first layer,
			# the previous layer object is the input layer
			if i == 0:
				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i+1]
			# All layers except for the first and the last
			elif i < layer_count - 1:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.layers[i+1]
			# The last layer - the next object is the loss
			else:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.loss
				self.output_layer_activation = self.layers[i]
			# If layer contains an attribute called "weights", it's a trainable layer
			if hasattr(self.layers[i], 'weights'):
				self.trainable_layers.append(self.layers[i])
		# Update loss object with trainable layers
		self.loss.remember_trainable_layers(
			self.trainable_layers
		)
		if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, Categorical_Crossentropy):
			self.softmax_classifier_output = Softmax_Categorical_Crossentropy()


	def fit(self, X, y, batch_size: int = None, epochs: int = 1, verbose: int = 1, shuffle=False, local_metrics: list = [], validation_data=None):
		# Default value if batch size is not being set
		train_steps = 1
		if validation_data is not None:
			validation_steps = 1
			X_val, y_val = validation_data

		# Calculate number of steps
		if batch_size is not None:
			train_steps = len(X) // batch_size
			if train_steps * batch_size < len(X):
				train_steps += 1

			if validation_data is not None:
				validation_steps = len(X_val) // batch_size
				if validation_steps * batch_size < len(X_val):
					validation_steps += 1

		# Main training loop
		for epoch in range(1, epochs+1):
			# Reset accumulated values in loss and accuracy objects
			self.loss.new_pass()
			# Iterate over steps
			if verbose == 1:
				print(f"Epoch {epoch}/{epochs}")
			prgsbar = pgrs(range(train_steps), auto_update=False)
			for step in prgsbar:
				# If batch size is not set -
				# train using one step and full dataset
				if batch_size is None:
					batch_X = X
					batch_y = y
				# Otherwise slice a batch
				else:
					batch_X = X[step*batch_size:(step+1)*batch_size]
					batch_y = y[step*batch_size:(step+1)*batch_size]
				# Perform the forward pass
				output = self.forward(batch_X, training=True)
				# Calculate loss
				data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
				loss = data_loss + regularization_loss
				# Get predictions and calculate an accuracy
				# predictions = self.output_layer_activation.predictions(output)
				# Perform backward pass
				self.backward(output, batch_y)
				output = np.argmax(output, axis=1)
				print(output)
				# Optimize (update parameters)
				self.optimizer.pre_update_params()
				for layer in self.trainable_layers:
					self.optimizer.update_params(layer)
				self.optimizer.post_update_params()
				if verbose == 1:
					output_string = " "
					for metric in local_metrics or self.metrics:
						output_string += f"- {metric}: {METRICS[metric](batch_y, output):.3f}"
					prgsbar.update(step, output_string + f" - loss: {loss:.3f}")
			# epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)

			if validation_data is not None:
				# Reset accumulated values in loss
				self.loss.new_pass()
				# Iterate over steps
				for step in range(validation_steps):
					if batch_size is None:
						batch_X = X_val
						batch_y = y_val
					else:
						batch_X = X_val[
							step*batch_size:(step+1)*batch_size
						]
						batch_y = y_val[
							step*batch_size:(step+1)*batch_size
						]
					output = self.forward(batch_X, training=False)
					# Calculate the loss
					self.loss.calculate(output, batch_y)
					# Get predictions and calculate an accuracy
					# predictions = self.output_layer_activation.predictions(output)
				# Get and print validation loss and accuracy
				validation_loss = self.loss.calculate_accumulated()
				# Print a summary
				# output_string += f'; Validation, ' + f'loss: {validation_loss:.4f}'

	def summary(self):
		inputs, outputs, names, param_counts = [],[],[],[]
		last_layer = None
		# ---------------------- Collect all layer information --------------------- #
		for layer in self.layers:
			if isinstance(layer, Layer):
				inputs.append(layer.input_shape)
				outputs.append(layer.output_shape)
				param_counts.append(layer.params)
				last_layer = layer
			elif isinstance(layer, Activation):
				inputs.append(last_layer.output_shape)
				outputs.append(last_layer.output_shape)
				param_counts.append({
					"trainable": 0,
					"non-trainable": 0
				})
			# Always append name
			names.append(layer.name + f" ({layer.__class__.__name__})")
		# --------------------------- Generate the table --------------------------- #
		params = np.sum(list(map(lambda x: [x["trainable"], x["non-trainable"]], param_counts)), axis=0)
		trainable = params[0]
		non_trainable = params[1]
		total_params = trainable + non_trainable
		end = f"Total Params: {total_params}"
		end += f"\nTrainable Params: {trainable}"
		end += f"\nNon-Trainable Params: {non_trainable}"
		str_table = table([[names[i], inputs[i], outputs[i], (param_counts[i]["trainable"] + param_counts[i]["non-trainable"])] for i in range(len(names))],
			['Layer', 'Input Shape', 'Output Shape', 'Param #'], end)
		print(str_table)


def table(rows, headers, add_end = "", separators = ["─", "=", "―"], space = "   "):
	max_widths = []
	for column in zip(*(rows + [headers])):
		# Set max widths for each column
		max_widths.append(max([len(str(text)) for text in column]))
	# Prepare Separators
	width = np.sum(max_widths) + len(max_widths) * len(space)
	separators = [i * width + "\n" for i in separators]
	output_string = '   '.join(['{{:<{}}}'.format(width) for width in max_widths])
	# Format Headers
	headers = output_string.format(*headers) + "\n" + separators[1]
	return separators[0] + headers +'\n'.join([(separators[0] if i > 0 else '') + output_string.format(*[str(x) for x in row]) for i, row in enumerate(rows)]) + "\n" + separators[1] + add_end + "\n" + separators[0]