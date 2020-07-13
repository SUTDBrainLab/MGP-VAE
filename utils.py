import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flags import *

def mse_loss(input, target):
    # mean square error

    return torch.sum((input - target).pow(2)) / input.data.nelement()

def weights_init(layer):
    # intialize weights

    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()

def plot_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # im = Image.fromarray(ndarr)
    return ndarr

def plot_training_images(video1, video2, dec_v1, dec_v2, X1_v1, X1_v2, epoch, decoder_test):

    # visualize test data feature transfer
    plt.close('all')
    plt.clf()
    fig, axs = plt.subplots(NUM_FEA+1,2) 

    axs[1,0].imshow(plot_image(dec_v1.squeeze(0).transpose(2, 3), nrow=NUM_FRAMES, normalize=True))
    axs[1,0].set_title('recon_v1', fontsize=6)
    axs[1,0].axis('off')

    axs[1,1].imshow(plot_image(dec_v2.squeeze(0).transpose(2, 3), nrow=NUM_FRAMES, normalize=True))
    axs[1,1].set_title('recon_v2', fontsize=6)
    axs[1,1].axis('off')

    axs[0,0].imshow(plot_image(video1.squeeze(0).transpose(2, 3), nrow=NUM_FRAMES, normalize=True))
    axs[0,0].set_title('v1', fontsize=6)
    axs[0,0].axis('off')

    axs[0,1].imshow(plot_image(video2.squeeze(0).transpose(2, 3), nrow=NUM_FRAMES, normalize=True))
    axs[0,1].set_title('v2', fontsize=6)
    axs[0,1].axis('off')

    for j in range(NUM_FEA):
    
        v1_feature = []
        v2_feature = []

        for i in range(NUM_FEA):
            v1_feature.append(X1_v1[:, :, i*FEA_DIM:(i+1)*FEA_DIM])
            v2_feature.append(X1_v2[:, :, i*FEA_DIM:(i+1)*FEA_DIM])

        # video 1
        v1_feature_transferred = torch.zeros(NUM_FRAMES, NDIM).unsqueeze(0).cuda()
    
        for i in range(NUM_FEA):
            if (i == j):
                v1_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM].copy_(v2_feature[i])
            else:    
                v1_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM].copy_(v1_feature[i])

        v1_decoded_feature_transferred = decoder_test(v1_feature_transferred, BATCH_SIZE=1)
        axs[j+1,0].imshow(plot_image(v1_decoded_feature_transferred.squeeze(0).transpose(2, 3), nrow=NUM_FRAMES, normalize=True))
        axs[j+1,0].set_title('v1_feature' + str(j) + '_transferred', fontsize=6)
        axs[j+1,0].axis('off')

        # video 2
        v2_feature_transferred = torch.zeros(NUM_FRAMES, NDIM).unsqueeze(0).cuda()
    
        for i in range(NUM_FEA):
            if (i == j):
                v2_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM].copy_(v1_feature[i])
            else:    
                v2_feature_transferred[:, :, i*FEA_DIM:(i+1)*FEA_DIM].copy_(v2_feature[i])

        v2_decoded_feature_transferred = decoder_test(v2_feature_transferred, BATCH_SIZE=1)
        axs[j+1,1].imshow(plot_image(v2_decoded_feature_transferred.squeeze(0).transpose(2, 3), nrow=NUM_FRAMES, normalize=True))
        axs[j+1,1].set_title('v2_feature' + str(j) + '_transferred', fontsize=6)
        axs[j+1,1].axis('off')

    plt.axis('off')
    plt.savefig('./style_transfer_training/{}.png'.format(str(epoch)), dpi=1000)
