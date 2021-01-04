import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import os
from itertools import cycle

from flags import *
from networks import Encoder, Decoder, Prediction_Model
from utils import weights_init
from covariance_fns import *
from dataloader import load_dataset

# compute energy of the geodesic 
def find_energy(z0, z1, z2):
	z0 = z0.unsqueeze(1).data
	z0 = Variable(z0, requires_grad = True)
	
	z1 = z1.unsqueeze(1).data
	z1 = Variable(z1, requires_grad = True)

	z2 = z2.unsqueeze(1).data
	z2 = Variable(z2, requires_grad = True)

	dec = decoder(torch.cat([Z_remaining, z1], 1))
	dec = torch.transpose(dec, 1, 2)
	
	a2 = (decoder(torch.cat([Z_remaining, z2.view(1, NUM_SAMPLE_GEO_OUTPUT, NDIM)], 1)) - \
	 2 * decoder(torch.cat([Z_remaining, z1.view(1, NUM_SAMPLE_GEO_OUTPUT, NDIM)], 1)) + \
	 decoder(torch.cat([Z_remaining, z0.view(1, NUM_SAMPLE_GEO_OUTPUT, NDIM)], 1))).view(1, 1, NUM_FRAMES, H, W)
	
	dec.backward(a2, retain_graph=True)
	energy = -N * z1.grad

	return energy
	
# compute gradient of energy wrt latent points
def find_etta_i(z0, z1, z2):

	z0 = z0.view(NUM_SAMPLE_GEO_OUTPUT, -1).data
	z1 = Variable(z1, requires_grad = True)
	z2 = z2.view(NUM_SAMPLE_GEO_OUTPUT, -1)

	dec = decoder(torch.cat([Z_remaining, z1.view(1, NUM_SAMPLE_GEO_OUTPUT, -1)], 1))
	v = decoder(torch.cat([Z_remaining, z2.view(1, NUM_SAMPLE_GEO_OUTPUT, -1)], 1)) - \
	2*dec + decoder(torch.cat([Z_remaining, z0.view(1, NUM_SAMPLE_GEO_OUTPUT, -1)], 1))
	dec.backward(v, retain_graph=True)
	etta = -dt * z1.grad
	return etta

# computes L2 norm
def compute_norm(x):
	p = torch.zeros(NUM_SAMPLE_GEO_OUTPUT).float()
	x = x.data.cpu().view(NUM_SAMPLE_GEO_OUTPUT, NDIM)

	for i in range(NDIM):
		q = x[:, i].view(NUM_SAMPLE_GEO_OUTPUT)
		p += q*q
	out = (torch.sqrt(p)).view(NUM_SAMPLE_GEO_OUTPUT)
	return out

# compute total energy of the geodesic path
def sum_energy(z_collection):
	
	delta_e = torch.FloatTensor(1, NUM_SAMPLE_GEO_OUTPUT, NDIM).zero_().cuda()
	for i in range(1, N):
		delta_e += find_energy(z_collection[i-1].view(NUM_SAMPLE_GEO_OUTPUT, -1) ,z_collection[i].view(NUM_SAMPLE_GEO_OUTPUT, -1) ,z_collection[i+1].view(NUM_SAMPLE_GEO_OUTPUT, -1))

	# energy_arr: a float tensor of size = (num_frames) where each index corresponds to energy of each point
	energy_arr = compute_norm(Variable(delta_e))
	energy_sum = (torch.sum(energy_arr)).item()

	return energy_sum

#########################################################################################################################################################

def linear_interpolation(prev_sample, curr_sample):
	diff = curr_sample - prev_sample
		
	curPt = torch.zeros_like(prev_sample)
	allfeature_interpolation_z = []

	# adding the initial latent point
	allfeature_interpolation_z.append(prev_sample)
	
	# adding thr intermediate latent points
	for k in range(N - 1):
		curPt.copy_(prev_sample)
		curPt[:, :, :] += (((k + 1)/N) * diff[:, :, :])
		allfeature_interpolation_z.append(curPt)
	
	# adding the final latent point	
	allfeature_interpolation_z.append(curr_sample)
	return allfeature_interpolation_z

def geodesic_interpolation(z_collection):
	energy = sum_energy(z_collection)
	
	count = 0
	while True:

		for i in range(1, len(z_collection) - 1):
			etta_i = find_etta_i(z_collection[i - 1], z_collection[i], z_collection[i + 1])
			etta_i = torch.nn.functional.normalize(etta_i, p = 2, dim = 1)
			e1 = STEP_SIZE * etta_i

			# update latent points in the direction of decreasing gradient of energy
			z_collection[i] = z_collection[i] - e1.view(1, NUM_SAMPLE_GEO_OUTPUT, -1)
		
		energy = sum_energy(z_collection)
		count += 1
		
		if (energy < THRESHOLD or count >= MAX_GEO_ITER):
			break

	return z_collection

if (__name__ == '__main__'):

	# model definition
	BATCH_SIZE = 1

	dataset = load_dataset()
	loader = cycle(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))

	encoder = Encoder()
	encoder.apply(weights_init)

	decoder = Decoder()
	decoder.apply(weights_init)

	encoder.load_state_dict(torch.load(os.path.join('checkpoints', ENCODER_SAVE)))
	decoder.load_state_dict(torch.load(os.path.join('checkpoints', DECODER_SAVE)))

	encoder.eval()
	decoder.eval()

	prediction_model = Prediction_Model()
	prediction_model.apply(weights_init)
	
	if (CUDA):
		encoder.cuda()
		decoder.cuda()
		prediction_model.cuda()

	optimizer = torch.optim.Adam(list(prediction_model.parameters()), lr = LR, betas=(BETA1, BETA2))
	mse_loss = nn.MSELoss()
	
	# number of frames that the prediction network will take as input
	N = NUM_SAMPLE_GEO_INPUT
	dt = 1.0 / N
	
	# begin training
	for epoch in range(NUM_EPOCHS_GEO):

		for iteration in range(len(dataset) // BATCH_SIZE): 

				optimizer.zero_grad()						
				
				X_in = next(loader).float().cuda()
				
				X1, KL1, muL1, det_q1 = encoder(X_in)
				X1 = X1.view(BATCH_SIZE, NUM_FRAMES, NDIM)

				Z_remaining = X1[:, :-NUM_SAMPLE_GEO_OUTPUT, :]

				output = prediction_model(Z_remaining)
				output = torch.t(output).view(1, NUM_SAMPLE_GEO_OUTPUT, NDIM)
				output_video = torch.cat([Z_remaining, output], 1)
			
				# latent space loss
				z_allfeature = linear_interpolation(output, X1[:, -NUM_SAMPLE_GEO_OUTPUT:, :])
				geodesic = geodesic_interpolation(z_allfeature)
				z_t1 = geodesic[0]
				z_t = geodesic[1]
				loss_latent = mse_loss(z_t1, z_t)
				
				# image based loss
				decoded_orig = decoder(X1)
				decoded_pred = decoder(torch.cat([Z_remaining, output.view(1, NUM_SAMPLE_GEO_OUTPUT, NDIM)], dim=1))
				loss_image = mse_loss(decoded_pred, decoded_orig)

				# combined loss
				loss = LATENT_WEIGHT * loss_latent + loss_image			
				loss.backward()

				# updating weights
				optimizer.step()

				# printing loss
				if (iteration % 100 == 0):
					print ('Loss at Epoch {}, Iteration {} is {}'.format(str(epoch), str(iteration), str(loss.item())))
				
				# saving model
				torch.save(prediction_model.state_dict(), os.path.join('checkpoints', 'prediction_model'))

