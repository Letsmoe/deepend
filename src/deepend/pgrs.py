import time
from IPython.display import display
from IPython.core.display import HTML
import datetime

class pgrs(object):
	def __init__(self, range_object, notebook=True, steps=100, auto_update=True):
		self.auto_update = auto_update
		self.steps = steps
		self.notebook = notebook
		self.start = time.time()
		self.mean_exec_time = 0
		self.iterations = 1
		# Get Range length
		if isinstance(range_object, range):
			self.range = len(range_object)
		else:
			self.range = range_object

	def update(self, i, appendix = ""):
		i += 1
		percent_completed = (i / (self.range / 100))
		blocks = "█" * int(percent_completed / 4)
		elapsed = (time.time() - self.start) or 1e-7
		# Time each iteration takes - 1 to prevent zero division on first call
		step_time = elapsed / (percent_completed or 1)
		remaining = str(datetime.timedelta(seconds = int((100 - percent_completed) * step_time)))
		# Calculate iterations per second
		self.iterations = int(i * (1 / elapsed))
		elapsed_str = str(datetime.timedelta(seconds = int(elapsed)))
		time_per_step, time_unit = time_shift(1 / (i / elapsed), "s")
		# Construct the output string
		output = f"{percent_completed:.0f}%|{blocks:<25}| {i}/{self.range} [{elapsed_str} < {remaining}, {self.iterations: .2f}it/s, {time_per_step:.2f}{time_unit}/step]{appendix}"

		if self.notebook:
			output = f"<pre>{output}</pre>"
			self.output.update(output)
		else:
			self.output(output)
		return 1

	def __iter__(self):
		# Check if executed in a notebook
		if self.notebook:
			self.output = display("", display_id=True)
		else:
			self.output = self.stdout

		end_of_range = (self.range - 1)
		if self.auto_update:
			return (self.update(x) and x if (x/self.iterations).is_integer() or x == end_of_range else x for x in range(self.range))
		else:
			return (x for x in range(self.range))

	def stdout(self, output):
		print(output, end="\r")


def time_shift(time, unit="s"):
	units = {
		"s": "ms",
		"ms": "μs",
		"μs": "ns"
	}
	if time < 1:
		return time_shift(time * 1000, units[unit])
	else:
		return (time, unit)