# this file contains all arguments required for training and evaluation of MGP VAE

import argparse
import getpass

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")

parser.add_argument('--dataset', type=str, default='moving_mnist', help="dataset to be used for training/testing (moving_mnist/dsprites/dsprites_color)")

parser.add_argument('--batch_size', type=int, default=8, help="batch size for training")
parser.add_argument('--test_batch_size', type=int, default=1, help="batch size for inference")
parser.add_argument('--image_size', type=int, default=64, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels in the images")
parser.add_argument('--num_frames', type=int, default=8, help="number of frames in the video")

parser.add_argument('--num_dim', type=int, default=2, help="total dimension of latent space")
parser.add_argument('--num_fea', type=int, default=2, help="total number of features")

parser.add_argument('--fea', type=list, default=['frac_0.1', 'frac_0.9'], help="All Gaussian processes as a list (options = frac_0.1, frac_0.9, bb, bb2, ou)")
parser.add_argument('--zero_mean_fea', type=bool, default=False, help="Non-zero Mean for all GPs (False = zero mean)")
parser.add_argument('--mean_fea_s', type=list, default= [-2, -2, -2, -2, -2], help="Starting Means of all Gaussian processes")
parser.add_argument('--mean_fea_e', type=list, default= [2, 2, 2, 2, 2], help="Ending Means of all Gaussian processes")
parser.add_argument('--keep_rho', type=bool, default= False, help="use rho for create_path fuction")

parser.add_argument('--beta', type=float, default=2.0, help="coeff. of kl_loss in total_loss")

parser.add_argument('--lrate', type=float, default=0.0001, help="initial learning rate")
parser.add_argument('--beta_1', type=float, default=0.5, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")

parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")

parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=1, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")
parser.add_argument('--is_training', type=bool, default=True, help="flag to indicate if it is training or inference.")

# visualization
parser.add_argument('--num_points_visualization', type=int, default=6, help="number of videos to plot in visualization")

# geodesic prediction
parser.add_argument('--num_epochs_geodesic', type=int, default=10, help="number of epochs for geodesic prediction")
parser.add_argument('--max_geo_iter', type=int, default=5, help="number of maximum iterations for geodesic prediction")
parser.add_argument('--num_samples_input', type=int, default=5, help="number of frames that the prediction network takes as input")
parser.add_argument('--num_samples_output', type=int, default=1, help="number of frames that the prediction network gives as output")
parser.add_argument('--latent_weight', type=float, default=0.5, help="weight of geodesic latent space loss in total geodesic prediction loss function")
parser.add_argument('--step_size', type=float, default=0.1, help="step size for geodesic loss gradient update")
parser.add_argument('--threshold', type=float, default=0.1, help="geodesic energy threshold")

FLAGS = parser.parse_args()

DATASET = FLAGS.dataset

# data info
H = FLAGS.image_size
W = FLAGS.image_size
NUM_INPUT_CHANNELS = FLAGS.num_channels
NUM_FRAMES = FLAGS.num_frames

NDIM = FLAGS.num_dim
NUM_FEA = FLAGS.num_fea
FEA_DIM = int(NDIM / NUM_FEA)

NUM_EPOCHS = FLAGS.end_epoch

if(FLAGS.is_training):
    BATCH_SIZE = FLAGS.batch_size
else:
    BATCH_SIZE = FLAGS.test_batch_size

LOAD_SAVED = FLAGS.load_saved

# data paths
if (FLAGS.dataset == 'moving_mnist'):
    DATA_PATH = "./data/moving_mnist/"
elif (FLAGS.dataset == 'dsprites_color'):
    DATA_PATH = './data/dsprites/trainset_dsprites_data_color_with_motion.h5'
elif (FLAGS.dataset == 'dsprites_color_test'):
    DATA_PATH = './data/dsprites/trainset_dsprites_data_color_with_motion_test_data.h5'
else:
    raise Exception('Invalid Dataset!')

OUTPUT_PATH = "./reconstructed_images/"

CUDA = FLAGS.cuda
LR = FLAGS.lrate

BETA = FLAGS.beta

# file save for networks
ENCODER_SAVE = FLAGS.encoder_save
DECODER_SAVE = FLAGS.decoder_save
START_EPOCH = FLAGS.start_epoch
END_EPOCH = FLAGS.end_epoch

BETA1 = FLAGS.beta_1
BETA2 = FLAGS.beta_2

ZERO_MEAN_FEA = FLAGS.zero_mean_fea

# Gaussian processes to be used 
FEA = FLAGS.fea
FEA_MEAN_S = FLAGS.mean_fea_s
FEA_MEAN_E = FLAGS.mean_fea_e
KEEP_RHO = FLAGS.keep_rho

# visualization
NUM_POINTS_VISUALIZATION = FLAGS.num_points_visualization

# geodesic prediction
NUM_EPOCHS_GEO = FLAGS.num_epochs_geodesic
MAX_GEO_ITER = FLAGS.max_geo_iter
NUM_SAMPLE_GEO_INPUT = FLAGS.num_samples_input
NUM_SAMPLE_GEO_OUTPUT = FLAGS.num_samples_output
LATENT_WEIGHT = FLAGS.latent_weight
STEP_SIZE = FLAGS.step_size
THRESHOLD = FLAGS.threshold
