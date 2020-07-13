# this file contains functions required for loading the data for training or evaluation

import numpy as np
from flags import *
from os import listdir
import torch
import h5py

def load_dataset():
	# loading data
    if (DATASET == 'moving_mnist'):
        dataset = data_moving_mnist(DATA_PATH)
    elif (DATASET == 'dsprites_color'):
        dataset = data_dsprites_color(DATA_PATH)
    else:
        raise Exception('Invalid Dataset!')

    return dataset

class data_moving_mnist:
	def __init__(self, DATA_PATH):
		self.array = []
		for f in listdir(DATA_PATH):
			self.data = np.load(DATA_PATH + f)
			self.arr = np.reshape(self.data['arr_0'], [10000, 20, 64, 64])
			for i in range(10000):
				self.array.append(np.reshape(self.arr[i, :2*NUM_FRAMES:2, ], [NUM_FRAMES, NUM_INPUT_CHANNELS, H, W]))

	def __len__(self):
		return self.array.__len__()

	def __getitem__(self, index):
		return self.array[index]

class data_dsprites(torch.utils.data.Dataset):
	def __init__(self, file):
		super(data_dsprites, self).__init__()
		self.file = h5py.File(file, 'r')

		self.n_videos = np.asarray(self.file.get('data'))
		self.n_labels = np.asarray(self.file.get('labels'))

	def __getitem__(self, index):
		input, label = self.n_videos[index], self.n_labels[index]
		return input.astype('float32'), label

	def __len__(self):
		return len(list(self.n_labels))

class data_dsprites_color(torch.utils.data.Dataset):
	def __init__(self, file):
		super(data_dsprites_color, self).__init__()
		self.file = h5py.File(file, 'r')
		self.n_videos = np.asarray(self.file.get('data'))

	def __getitem__(self, index):
		input = self.n_videos[index]
		return input.astype('float32')

	def __len__(self):
		return self.n_videos.shape[0]