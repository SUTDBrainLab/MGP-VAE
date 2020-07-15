import os
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from flags import *
from networks import Encoder, Decoder
from itertools import cycle
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import weights_init
from dataloader import data_moving_mnist, data_dsprites

if __name__ == "__main__":
	
	if not os.path.exists('./results/style_transfer_results'):
		os.makedirs('./results/style_transfer_results')

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
	
	video1 = next(loader).float().cuda()[0].unsqueeze(0)
	video2 = next(loader).float().cuda()[0].unsqueeze(0)
	
	X1, KL1, muL1, det_q1 = encoder(video1)
	X2, KL2, muL2, det_q2 = encoder(video2)

	# save reconstructed images
	dec_v1 = decoder(X1)
	save_image(dec_v1.squeeze(0).transpose(2, 3), './results/style_transfer_results/recon_v1.png', nrow=NUM_FRAMES, normalize=True)

	dec_v2 = decoder(X2)
	save_image(dec_v2.squeeze(0).transpose(2, 3), './results/style_transfer_results/recon_v2.png', nrow=NUM_FRAMES, normalize=True)

	v1_feature = []
	v2_feature = []

	for i in range(NUM_FEA):
		
		v1_feature.append(X1[:, :, i*FEA_DIM:(i+1)*FEA_DIM])
		v2_feature.append(X2[:, :, i*FEA_DIM:(i+1)*FEA_DIM])

	for i in range(NUM_FEA):

		for j in range(NUM_FEA):

			# style transfer on video1
			v1_feature_transferred = torch.zeros(NUM_INPUT_CHANNELS, NUM_FRAMES, NDIM).cuda()
			if (j == i):
				v1_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM] = v2_feature[j]
			else:
				v1_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM] = v1_feature[j]

			v1_feature_transferred_dec = decoder(v1_feature_transferred)
			save_image(v1_feature_transferred_dec.squeeze(0).transpose(2, 3), './results/style_transfer_results/v1_grid_feature{}_transferred.png'.format(j), nrow=NUM_FRAMES, normalize=True)

			# style transfer on video2
			v2_feature_transferred = torch.zeros(NUM_INPUT_CHANNELS, NUM_FRAMES, NDIM).cuda()
			if (j == i):
				v2_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM] = v1_feature[j]
			else:
				v2_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM] = v2_feature[j]
			
			v2_feature_transferred_dec = decoder(v2_feature_transferred)
			save_image(v2_feature_transferred_dec.squeeze(0).transpose(2, 3), './results/style_transfer_results/v2_grid_feature{}_transferred.png'.format(j), nrow=NUM_FRAMES, normalize=True)

	
