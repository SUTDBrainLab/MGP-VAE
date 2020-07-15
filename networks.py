# this file contains networks for all individual components of MGP VAE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from flags import *

mask = np.ones(shape=(BATCH_SIZE, NUM_FRAMES, NDIM))
mask[:,0,:] = 0
mk = torch.from_numpy(mask).float().cuda()

def fill_triangular(vector):	
	if (len(vector.size()) == 1):
		l = vector.size()[0]
		n = int(np.sqrt(0.25 + 2. *l) - 0.5)
		rev_vec = torch.flip(vector, dims=[0])
		rev_vec = rev_vec[n:]
		vec = torch.cat([vector, rev_vec], 0)
		y = vec.view(n,n)
		y = torch.transpose(torch.triu(y), dim0=0, dim1=1)
		return y

	elif (len(vector.size()) == 2):
		l = vector.size()[1]
		n = int(np.sqrt(0.25 + 2. *l) - 0.5)
		rev_vec = torch.flip(vector, dims=[1])
		rev_vec = rev_vec[:, n:]
		vec = torch.cat([vector, rev_vec], 1)
		y = vec.view(vector.size()[0], n,n)
		y = torch.transpose(torch.triu(y), dim0=1, dim1=2)
		return y

	elif (len(vector.size()) == 3):
		vector = vector.view(vector.size()[0]*vector.size()[1], vector.size()[2])
		l = vector.size()[1]
		n = int(np.sqrt(0.25 + 2. *l) - 0.5)
		rev_vec = torch.flip(vector, dims=[1])
		rev_vec = rev_vec[:, n:]
		vec = torch.cat([vector, rev_vec], 1)
		y = vec.view(vector.size()[0], n,n)
		y = torch.transpose(torch.triu(y), dim0=1, dim1=2)
		y = y.view(BATCH_SIZE, (y.size()[0])//BATCH_SIZE, y.size()[1], y.size()[2])
		return y

	else:
		raise Exception("ERROR in fill_triangular function, size of input is : {}".format(str(len(vector.size()))))

raw_matSum = np.ones(NUM_FRAMES * (NUM_FRAMES + 1) // 2)
matSum = torch.from_numpy(raw_matSum).float().cuda()
matSum = fill_triangular(matSum)

class My_Tanh(nn.Module):
	def __init__(self):
		super(My_Tanh, self).__init__()
		self.tanh = nn.Tanh()

	def forward(self, x):
		return (0.5 * (self.tanh(x) + 1))

def matrix_diag_4d(diagonal):
	diagonal = diagonal.view(diagonal.size()[0]*diagonal.size()[1], diagonal.size()[2], diagonal.size()[3])
	result = torch.diagonal(diagonal, dim1 = -2, dim2 = -1)

	result = result.view(BATCH_SIZE, NDIM, NUM_FRAMES)
	return result

def matrix_diag_3d(diagonal):
	result = torch.diagonal(diagonal, dim1 = -2, dim2 = -1)
	return result

def create_path(K_L1, mu_L1, BATCH_SIZE=BATCH_SIZE):
	# this function samples random paths from given GP using lower triangular matrices K_L (obtained from covariance matrices) and mean mu_L
	
	inc_L1 = torch.randn(BATCH_SIZE, NUM_FRAMES, NDIM).cuda()
	X1 = torch.einsum('ikj,ijlk->ilj', inc_L1, K_L1) + mu_L1	# shape = (BATCH_SIZE, NUM_FRAMES, NDIM)
	return X1

def create_path_rho(K_L1, mu_L1, BATCH_SIZE=BATCH_SIZE, rho=0.5):
	# this function samples random paths from given GP using lower triangular matrices K_L (obtained from covariance matrices) and mean mu_L

	c11 = torch.randn(BATCH_SIZE, NUM_FRAMES).cuda()
	c12 = (rho * c11) + (np.sqrt(1.0 - (rho * rho)) * torch.randn(BATCH_SIZE, NUM_FRAMES).cuda())
	c21 = torch.randn(BATCH_SIZE, NUM_FRAMES).cuda()
	c22 = (rho * c21) + (np.sqrt(1.0 - (rho * rho)) * torch.randn(BATCH_SIZE, NUM_FRAMES).cuda())

	inc_L1 = torch.stack([c11, c12, c21, c22], dim=2)
	X1 = torch.einsum('ikj,ijlk->ilj', inc_L1, K_L1) + mu_L1  # shape = (BATCH_SIZE, NUM_FRAMES, NDIM)

	return X1

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=NUM_INPUT_CHANNELS, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(num_features=16)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=16)
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(num_features=32)
		self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(num_features=32)
		self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(num_features=64)
		self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
		self.bn6 = nn.BatchNorm2d(num_features=64)
		self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.bn7 = nn.BatchNorm2d(num_features=128)
		self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
		self.bn8 = nn.BatchNorm2d(num_features=128)

		# layers for MLP
		if (H == 64):
			self.dense1 = nn.Linear(in_features=NUM_FRAMES*128*4*4, out_features=128) 
		elif (H == 32):
			self.dense1 = nn.Linear(in_features=NUM_FRAMES*128*2*2, out_features=128) 
		self.bn1_mlp = nn.BatchNorm1d(num_features=128)
		
		self.raw_kl1_size = NDIM * NUM_FRAMES * NUM_FRAMES
		self.dense2_1 = nn.Linear(in_features=128, out_features = self.raw_kl1_size)
		self.bn2_1 = nn.BatchNorm1d(num_features=self.raw_kl1_size)

		self.dense2_2 = nn.Linear(in_features=128, out_features=(NDIM * NDIM) * ((NDIM * NDIM) + 1) // 2)
		self.bn2_2 = nn.BatchNorm1d(num_features=(NDIM * NDIM) * ((NDIM * NDIM) + 1) // 2)
		self.dense2_3 = nn.Linear(in_features=128, out_features=NDIM * NUM_FRAMES)
		self.bn2_3 = nn.BatchNorm1d(num_features=NDIM * NUM_FRAMES)
		self.dense2_4 = nn.Linear(in_features=128, out_features=NDIM * NDIM)
		self.bn2_4 = nn.BatchNorm1d(num_features=NDIM * NDIM)

		self.relu = nn.ReLU()
		self.elu = nn.ELU()
		self.tanh = nn.Tanh()

	def MLP_1(self, x):
		x = self.bn1_mlp(self.elu(self.dense1(x)))
		x = self.bn2_1(self.tanh(self.dense2_1(x)))
		return x

	def MLP_3(self, x):
		x = self.bn1_mlp(self.elu(self.dense1(x)))
		x = self.bn2_3(self.tanh(self.dense2_3(x)))
		return x

	def forward(self, x, BATCH_SIZE=BATCH_SIZE):
		x = x.view(BATCH_SIZE * NUM_FRAMES, NUM_INPUT_CHANNELS, H, W)
		x = self.bn1(self.elu(self.conv1(x)))
		x = self.bn2(self.elu(self.conv2(x)))
		x = self.bn3(self.elu(self.conv3(x)))
		x = self.bn4(self.elu(self.conv4(x)))
		x = self.bn5(self.elu(self.conv5(x)))
		x = self.bn6(self.elu(self.conv6(x)))
		x = self.bn7(self.elu(self.conv7(x)))
		x = self.bn8(self.elu(self.conv8(x)))
		
		x = x.view(BATCH_SIZE, NUM_FRAMES, 128, x.size()[2], x.size()[3])

		# flatten
		x = x.view(x.size()[0], -1)

		# create path sample 	
		raw_KL1 = self.MLP_1(x).view(-1, NDIM, NUM_FRAMES, NUM_FRAMES)
		KL1 = torch.tril(raw_KL1)
		
		if(BATCH_SIZE==1):
			KL1_diag = matrix_diag_3d(KL1)
		else:
			KL1_diag = matrix_diag_4d(KL1)
		det_q1 = torch.prod(KL1_diag*KL1_diag, dim=2)
		
		muL1 = self.MLP_3(x).view(-1, NUM_FRAMES, NDIM)
		
		if(KEEP_RHO):
			X1 = create_path_rho(KL1, muL1, BATCH_SIZE)
		else:
			X1 = create_path(KL1, muL1, BATCH_SIZE)

		return X1, KL1, muL1, det_q1
 

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		factor = NDIM

		self.dense1 = nn.Linear(in_features=NUM_FRAMES*factor, out_features=NUM_FRAMES*8*8*16, bias=True)
		self.bn_dense1 = nn.BatchNorm1d(num_features=NUM_FRAMES*8*8*16)
		self.dense2 = nn.Linear(in_features=NUM_FRAMES*8*8*16, out_features=NUM_FRAMES*8*8*64, bias=True)
		self.bn_dense2 = nn.BatchNorm1d(num_features=NUM_FRAMES*8*8*64)
		
		self.conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(num_features=64)
		self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=64)
		self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(num_features=32)
		self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(num_features=32)
		self.conv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(num_features=16)
		self.conv6 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
		self.bn6 = nn.BatchNorm2d(num_features=16)
		self.conv7 = nn.ConvTranspose2d(in_channels=16, out_channels=NUM_INPUT_CHANNELS, kernel_size=3, stride=1, padding=1)

		self.my_tanh = My_Tanh()

		self.relu = nn.ReLU()
		self.elu = nn.ELU()

	def forward(self, x1, BATCH_SIZE=BATCH_SIZE):
		x = x1

		# flatten 
		x = x.view(BATCH_SIZE, NUM_FRAMES * NDIM)
		x = self.bn_dense1(self.elu(self.dense1(x)))
		x = self.bn_dense2(self.elu(self.dense2(x)))
		
		x = x.view(BATCH_SIZE * NUM_FRAMES, 64, 8, 8)

		x = self.bn1(self.elu(self.conv1(x)))
		x = self.bn2(self.elu(self.conv2(x)))
		x = self.bn3(self.elu(self.conv3(x)))
		x = self.bn4(self.elu(self.conv4(x)))
		x = self.bn5(self.elu(self.conv5(x)))
		if (H == 64):	
			x = self.bn6(self.elu(self.conv6(x)))
		
		x = self.my_tanh(self.conv7(x))
		x = (255*x).view(BATCH_SIZE, NUM_FRAMES, NUM_INPUT_CHANNELS, H, W)
	
		return x

# model for prediction of future frames
class Prediction_Model(nn.Module):

	def __init__(self):
		super(Prediction_Model, self).__init__()

		self.fc1 = nn.Linear((NUM_FRAMES-1)*(NDIM), 15)
		self.fc2 = nn.Linear(15, NDIM)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = x.view(x.size()[0], -1)
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x