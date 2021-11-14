from deepend.saving import *


def load_model_from_hd5f(path):
	File = h5py.File(path, 'r')