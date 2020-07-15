import os
import h5py
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
import cv2
import skimage.io as io
import copy
import random
import os

# saving images from npz in the folder

# data = np.load('./dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
# images = data['imgs']
# labels = data['latents_classes']

# for i in range(len(images)):
#     print(labels[i])
    # im = torch.from_numpy(images[i]).unsqueeze(0)
    # im = im.unsqueeze(0).float()
    # print(im.size())
    # im = F.upsample(im, size=(im.size(2)//2, im.size(3)//2), mode='bilinear')
    # save_image(im, './dsprites-dataset/images/'+(str)(labels[i][1])+'/'+str(labels[i][2])+'_'+str(labels[i][3])+'_'+str(labels[i][4])+'_'+str(labels[i][5])+'.png', normalize=True)

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, file):
        super(dataset_h5, self).__init__()
        self.file = h5py.File(file, 'r')

        self.n_videos = np.asarray(self.file.get('data'))
        self.n_labels = np.asarray(self.file.get('labels'))

    def __getitem__(self, index):
        input, label = self.n_videos[index], self.n_labels[index]
        return input.astype('float32'), label

    def __len__(self):
        return len(list(self.n_labels))

def save_list_to_hdf5(data, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('data', data=data)
    hf.close()

DATA_DIR = './dsprites-dataset/images/'

def generate_vid(shape,color,variation,motion):
    path = os.path.join(DATA_DIR,str(shape))
    pos_x = random.sample(range(0,32,8),1)[0]
    if variation == 0: #scale
        scale = random.randint(0,5)
        pos_y = 0
        orientation = random.randint(0,39)
        arr = np.zeros((8, 3, 32, 32))
        arr_transpose = np.zeros((8, 3, 32, 32))
        for frame_idx in range(8):
            varying_scale = scale + frame_idx
            if varying_scale > 5:
                if varying_scale > 10:
                    varying_scale = varying_scale - 10
                else:
                    varying_scale = 10 - varying_scale

            image_name = str(varying_scale) + '_' + str(orientation) + '_' + str(pos_x) + '_' + str(
                pos_y + 4 * frame_idx) + '.png'
            image_name = os.path.join(path, image_name)
            print(image_name)
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.transpose(image)
            image_transpose = np.transpose(image)
            if color == 3:
                for idx in range(3):
                    arr[frame_idx, idx, :, :] = image
                    arr_transpose[frame_idx, idx, :, :] = image_transpose
            else:
                arr[frame_idx, color, :, :] = image
                arr_transpose[frame_idx, color, :, :] = image_transpose
        if motion == 0:
            return arr
        elif motion == 1:
            arr_flip = np.flip(arr, 0)
            return arr_flip
        elif motion == 2:
            return arr_transpose
        elif motion == 3:
            arr_transpose_flip = np.flip(arr_transpose, 0)
            return arr_transpose_flip
    elif variation == 1: # orienation
        orientation = random.randint(0,39)
        pos_y = 0
        scale = random.randint(4,5)
        arr = np.zeros((8, 3, 32, 32))
        arr_transpose = np.zeros((8, 3, 32, 32))
        for frame_idx in range(8):
            varying_orientation = orientation + frame_idx * 2
            if varying_orientation > 39:
                varying_orientation = varying_orientation % 40
            image_name = str(scale) + '_' + str(varying_orientation) + '_' + str(pos_x) + '_' + str(
                pos_y + 4 * frame_idx) + '.png'
            image_name = os.path.join(path, image_name)
            print(image_name)
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.transpose(image)
            image_transpose = np.transpose(image)
            if color == 3:
                for idx in range(3):
                    arr[frame_idx, idx, :, :] = image
                    arr_transpose[frame_idx, idx, :, :] = image_transpose
            else:
                arr[frame_idx, color, :, :] = image
                arr_transpose[frame_idx, color, :, :] = image_transpose
        if motion == 0:
            return arr
        elif motion == 1:
            arr_flip = np.flip(arr, 0)
            return arr_flip
        elif motion == 2:
            return arr_transpose
        elif motion == 3:
            arr_transpose_flip = np.flip(arr_transpose, 0)
            return arr_transpose_flip

test_data = []

NUM_FRAMES = 8

################ test data
num_pairs = 100

for pair in range(num_pairs):
    # 2 videos with distinct shape, color and variation as a comparison pair
    shape =  random.sample(range(3),2)
    color = random.sample(range(4),2)
    variation = random.sample(range(2),2) # 0-scale 1-orientation
    motion = random.sample(range(4),2) # 0-left->right, 1-right->left, 2-up->down, 3-down->up

    v1 = generate_vid(shape[0],color[0],variation[0],motion[0])
    v2 = generate_vid(shape[1],color[1],variation[1],motion[1])

    test_data.append(v1)
    test_data.append(v2)

test_data = np.asarray(test_data, dtype='uint8')
print(test_data.shape)
save_list_to_hdf5(test_data, './trainset_dsprites_data_color_with_motion_test_data.h5')