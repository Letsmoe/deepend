from deepend.utils import *

def img_to_array(img, dtype=None):
	"""Takes an image path and returns the numpy array accordingly

	Args:
		img (str): Image path
		dtype (np.dtype, optional): Numpy datatype. Defaults to None.

	Returns:
		np.ndarray: Image Tensor
	"""
	if os.path.isfile(img):
		img = cv2.imread(img)

	return np.array(img, dtype=dtype)

def load_img(path, grayscale=False, color_mode=cv2.COLOR_BGR2RGB, target_size=None, interpolation=cv2.INTER_NEAREST):
	"""Loads an image from a path and returns it as a numpy array.

	Args:
		path (str): Image filepath
		grayscale (bool, optional): Whether to convert to grayscale. Defaults to False.
		color_mode (cv2.COLOR_MODE, optional): OpenCV color code to convert to if grayscale is false. Defaults to cv2.COLOR_BGR2RGB.
		target_size (list, optional): Size to reshape the image to. Defaults to None.
		interpolation (cv2.INTERPOLATION, optional): Interpolation to use for resizing. Defaults to cv2.CV_INTER_NN.

	Raises:
		TypeError: target_size is not of type list
		NameError: path not found or not of type file
		TypeError: path is not of type string

	Returns:
		np.ndarray: Image Tensor
	"""
	if isinstance(path, str):
		if os.path.isfile(path):
			img = cv2.imread(path)
			if grayscale:
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			else:
				img = cv2.cvtColor(img, color_mode)
			# Resize the image if target_size given
			if target_size:
				if isinstance(target_size, list):
					img = cv2.resize(img, target_size, interpolation=interpolation)
				else:
					raise TypeError("target_size must be of type list in deepend.utils.load_img")
			return img
		else:
			raise NameError("given path could not be found in deepend.utils.load_img")
	else:
		raise TypeError('Path must be of type string in deepend.utils.load_img')