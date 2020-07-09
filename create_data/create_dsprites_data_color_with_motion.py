import os

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data
from keras.preprocessing.image import save_img


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

DATA_DIR = ['./dsprites-dataset/images/0/', './dsprites-dataset/images/1/', './dsprites-dataset/images/2/']

train_scale_data = []
train_orientation_data = []

NUM_FRAMES = 8

############################ Scale
write_file = open('label.txt', 'w')

for d in DATA_DIR:
    path = d
    shape = (int)(d.split('/')[3])
    for color in range(4):
        for orientation in range(40):
            orientation_degree = orientation * 360/39
            for pos_x in range(0,32,8):
                for scale in range(6):
                    pos_y = 0
                    arr = np.zeros((8, 3, 32, 32))
                    arr_transpose = np.zeros((8, 3, 32, 32))
                    video_label_raw = []
                    for frame_idx in range(8):
                        varying_scale = scale + frame_idx
                        if varying_scale > 5:
                            if varying_scale > 10:
                                varying_scale = varying_scale - 10
                            else:
                                varying_scale = 10 - varying_scale

                        image_name = str(varying_scale) + '_' + str(orientation) + '_' + str(pos_x) + '_' + str(
                            pos_y + 4*frame_idx) + '.png'

                        video_label_raw.append([shape, color, varying_scale, orientation_degree, pos_x, pos_y+4*frame_idx])

                        image_name = os.path.join(path, image_name)
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

                    video_label_flip = []
                    for i in range(len(video_label_raw)):
                        video_label_flip.append(video_label_raw[-1-i])

                    video_label_transpose = []
                    for i in range(len(video_label_raw)):
                        frame_label_raw = video_label_raw[i].copy()

                        frame_label_raw[4], frame_label_raw[5] = frame_label_raw[5], frame_label_raw[4]
                        orientation_transpose = 90 - frame_label_raw[3]
                        frame_label_raw[3] = orientation_transpose if orientation_transpose>=0 else 360 + orientation_transpose

                        video_label_transpose.append(frame_label_raw)

                    video_label_transpose_flip = []
                    for i in range(len(video_label_transpose)):
                        video_label_transpose_flip.append(video_label_transpose[-1-i])

                    video_label = video_label_raw + video_label_flip + video_label_transpose + video_label_transpose_flip

                    for frame_label in video_label:
                        write_file.writelines('{} {} {} {} {} {}\n'.format(*frame_label))

                    for frm_idx in range(arr.shape[0]):
                        frame = arr[frm_idx,:,:,:]
                        save_img(os.path.join('./test',str(scale)+'_'+str(frm_idx)+'.png'), frame, data_format='channels_first')
                    train_scale_data.append(arr)

                    arr_flip = np.flip(arr, 0)
                    for frm_idx in range(arr_flip.shape[0]):
                        frame = arr_flip[frm_idx,:,:,:]
                        save_img(os.path.join('./test','flip_'+str(scale)+'_'+str(frm_idx)+'.png'), frame, data_format='channels_first')
                    train_scale_data.append(arr_flip)

                    for frm_idx in range(arr_transpose.shape[0]):
                        frame = arr_transpose[frm_idx,:,:,:]
                        save_img(os.path.join('./test','transpose_'+str(scale)+'_'+str(frm_idx)+'.png'), frame, data_format='channels_first')
                    train_scale_data.append(arr_transpose)

                    arr_transpose_flip = np.flip(arr_transpose, 0)
                    for frm_idx in range(arr_transpose_flip.shape[0]):
                        frame = arr_transpose_flip[frm_idx,:,:,:]
                        save_img(os.path.join('./test', 'transpose_flip_' + str(scale) + '_' + str(frm_idx) +'.png'), frame, data_format='channels_first')
                    train_scale_data.append(arr_transpose_flip)

train_scale_data = np.asarray(train_scale_data, dtype='uint8')
print(train_scale_data.shape)


#
# # ############################### Orientation

for d in DATA_DIR:
    path = d
    shape = (int)(d.split('/')[3])
    for color in range(4):
        for scale in range(4,6): # remove small scales as they are bad for training
            for pos_x in range(0,32,2):
                for orientation in range(40):
                    pos_y = 0
                    arr = np.zeros((8, 3, 32, 32))
                    arr_transpose = np.zeros((8, 3, 32, 32))
                    video_label_raw = []
                    for frame_idx in range(8):
                        varying_orientation = orientation + frame_idx*2
                        if varying_orientation > 39:
                            varying_orientation = varying_orientation%40

                        varying_orientation_degree = varying_orientation * 360 / 39

                        image_name = str(scale) + '_' + str(varying_orientation) + '_' + str(pos_x) + '_' + str(
                            pos_y + 4*frame_idx) + '.png'

                        video_label_raw.append([shape, color, scale, varying_orientation_degree, pos_x, pos_y + 4 * frame_idx])

                        image_name = os.path.join(path, image_name)
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

                    video_label_flip = []
                    for i in range(len(video_label_raw)):
                        video_label_flip.append(video_label_raw[-1-i])

                    video_label_transpose = []
                    for i in range(len(video_label_raw)):
                        frame_label_raw = video_label_raw[i].copy()

                        frame_label_raw[4], frame_label_raw[5] = frame_label_raw[5], frame_label_raw[4]
                        orientation_transpose = 90 - frame_label_raw[3]
                        frame_label_raw[3] = orientation_transpose if orientation_transpose>=0 else 360 + orientation_transpose

                        video_label_transpose.append(frame_label_raw)

                    video_label_transpose_flip = []
                    for i in range(len(video_label_transpose)):
                        video_label_transpose_flip.append(video_label_transpose[-1-i])

                    video_label = video_label_raw + video_label_flip + video_label_transpose + video_label_transpose_flip

                    for frame_label in video_label:
                        write_file.writelines('{} {} {} {} {} {}\n'.format(*frame_label))

                    for frm_idx in range(arr.shape[0]):
                        frame = arr[frm_idx,:,:,:]
                        save_img(os.path.join('./test',str(orientation)+'_'+str(frm_idx)+'.png'), frame, data_format='channels_first')
                    train_orientation_data.append(arr)

                    arr_flip = np.flip(arr, 0)
                    for frm_idx in range(arr_flip.shape[0]):
                        frame = arr_flip[frm_idx,:,:,:]
                        save_img(os.path.join('./test','flip_'+str(orientation)+'_'+str(frm_idx)+'.png'), frame, data_format='channels_first')
                    train_orientation_data.append(arr_flip)

                    for frm_idx in range(arr_transpose.shape[0]):
                        frame = arr_transpose[frm_idx,:,:,:]
                        save_img(os.path.join('./test','transpose_'+str(orientation)+'_'+str(frm_idx)+'.png'), frame, data_format='channels_first')
                    train_orientation_data.append(arr_transpose)

                    arr_transpose_flip = np.flip(arr_transpose, 0)
                    for frm_idx in range(arr_transpose_flip.shape[0]):
                        frame = arr_transpose_flip[frm_idx,:,:,:]
                        save_img(os.path.join('./test', 'transpose_flip_' + str(orientation) + '_' + str(frm_idx) +'.png'), frame, data_format='channels_first')
                    train_orientation_data.append(arr_transpose_flip)

train_orientation_data = np.asarray(train_orientation_data, dtype='uint8')
print(train_orientation_data.shape)
#
#
train_data = np.concatenate((train_scale_data, train_orientation_data), axis=0)
print(train_data.shape)
write_file.close()
save_list_to_hdf5(train_data, './trainset_dsprites_data_color_with_motion.h5')