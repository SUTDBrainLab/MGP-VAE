# this file contains code that helps visualize the latent space encodings 
# NOTE: this is valid only when all features are kept 2-dimensional (i.e. FEA_DIM == 2)

import os
from os import listdir
import torch
import torch.nn as nn
import numpy as np
from itertools import cycle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from flags import *
from networks import Encoder, Decoder

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import weights_init
from dataloader import data_moving_mnist, data_dsprites, data_dsprites_color

def choose_color(i):

	# size of this list should be same as number of videos to be plotted (i.e. NUM_POINTS_VISUALIZATION)
	colors = ['r', 'g', 'b', 'y', 'm', 'c'] 
	# Add more colors here if you wish to plot more videos

	return colors[i]

if __name__ == "__main__":
	
	if not os.path.exists('./results/video_visualization/'):
		os.makedirs('./results/video_visualization/')

	dataset = load_dataset()
	loader = cycle(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))

	encoder = Encoder()
	encoder.apply(weights_init)

	decoder = Decoder()
	decoder.apply(weights_init)
			
	encoder.load_state_dict(torch.load(os.path.join('checkpoints/', ENCODER_SAVE)))
	decoder.load_state_dict(torch.load(os.path.join('checkpoints/', DECODER_SAVE)))

	encoder.eval().cuda()
	decoder.eval().cuda()

	for i in range(NUM_FEA):

		fea_x = []
		fea_y = []

		for j in range(NUM_POINTS_VISUALIZATION):

			X_in = next(loader)
			X_in = X_in.float().cuda()

			X1, KL1, muL1, det_q1 = encoder(X_in)
			X1 = X1.data.cpu().numpy()

			fea_x.append(X1[:, :, 2*i])
			fea_y.append(X1[:, :, 2*i + 1])

		fig, ax = plt.subplots(1)
		
		for j in range(NUM_POINTS_VISUALIZATION):
			
			# plottings video encodings with different color for each video
			plt.scatter(fea_x[j], fea_y[j], marker='o', c=choose_color(j), cmap=plt.cm.get_cmap("jet", 10), edgecolor='k')
		
		# may vary the limits of this figure depending on the spread of each Gaussian process
		plt.xlim(-3, 3)
		plt.ylim(-3, 3)
		plt.savefig("./results/video_visualization/fea{}.png".format(str(i)))
		
		plt.close()

		