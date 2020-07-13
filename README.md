## Disentangling using Gaussian Processes and Rough Paths

#### Installing all required libraries and the dependencies
```
pip install -r requirements.txt

```

### Preparing Data

- Moving MNIST : Either download the prepared dataset from <a href="https://drive.google.com/file/d/1JAIpbRPqjbGyUltVbKnKYIxq8ig_aYfX/view?usp=sharing">this link</a>, or preparing dataset manually by running script create_data/moving_mnist/create_moving_mnist_data.py.
- Dsprites : Either download prepared dataset from <a href="https://drive.google.com/file/d/15pkq1NaU1HyfDP68zgI0Tk9V8OIgv3RS/view?usp=sharing">this link</a>, or prepare dataset manually by first downloading data from <a href="https://github.com/deepmind/dsprites-dataset">this link</a> then placing images in folders dsprites-dataset/0/, dsprites-dataset/1/, and dsprites-dataset/2/ according to their classes (shape in this case) and then by running the script create_data/dsprites/create_dsprites_data.py.

- - - - 
NOTE:
* For Moving MNIST, the number of frames is 5 and image size is 64x64, while for dsprites, the number of frames is 8 and image size is 32x32. 
* For testing non-zero mean for GPs, use all latent dim. as bb2.
- - - - 

### FLAGS

- cuda: Run the following code on a GPU
- dataset: Dataset to be used for training/testing (gray_shapes/moving_mnist)
- batch_size: Batch size for training
- test_batch_size: Batch size for inference
- image_size: Height and width of the image
- num_channels: Number of channels in the images
- num_frames: Number of frames in the video
- ndim: Total dimension of latent space
- fdim: Total number of features in latent space
- fea1: First choice of Gaussian Process (options = frac_0.1, frac_0.9, bb, bb2, ou)
- fea2: Second choice of Gaussian Process (options = frac_0.1, frac_0.9, bb, bb2, ou)
- fea3: Third choice of Gaussian Process (options = frac_0.1, frac_0.9, bb, bb2, ou)
- zero_mean_fea: Flag to indicate if GPs have zero mean (False=zero_mean)
- mean_fea1_s: Starting Mean for fea1
- mean_fea1_e: Ending Mean for fea1
- mean_fea2_s: Starting Mean for fea2
- mean_fea2_e: Ending Mean for fea2
- mean_fea3_s: Starting Mean for fea3
- mean_fea3_e: Ending Mean for fea3
- beta: Coeff. of kl_loss in total_loss
- lrate: Initial learning rate
- beta_1: Beta 1 value for Adam optimizer
- beta_2: Beta 2 value for Adam optimizer
- encoder_save: Save model for encoder
- decoder_save: Save model for decoder
- load_saved: Flag to indicate if a saved model will be loaded
- start_epoch: Flag to set the starting epoch for training
- end_epoch: Flag to indicate the final epoch of training
- is_training: Flag to indicate if it is training or inference

- - - - 
### Training 

- Place the dataset directory in the Path_to_Repo/data/
- Begin training using appropriate flags (NOTE: For training, use is_training=True).
```
python train.py [-h] [--cuda CUDA] [--dataset DATASET]
                [--batch_size BATCH_SIZE] [--is_training IS_TRAINING] 
                [--image_size IMAGE_SIZE] [--num_channels NUM_CHANNELS]
                [--num_frames NUM_FRAMES] [--buffer_size BUFFER_SIZE]
                [--ndim NDIM] [--fdim FDIM] [--fea1 FEA1] [--fea2 FEA2]
                [--fea3 FEA3] [--zero_mean_fea ZERO_MEAN_FEA]
                [--mean_fea1_s MEAN_FEA1_S] [--mean_fea1_e MEAN_FEA1_E]
                [--mean_fea2_s MEAN_FEA2_S] [--mean_fea2_e MEAN_FEA2_E]
                [--mean_fea3_s MEAN_FEA3_S] [--mean_fea3_e MEAN_FEA3_E]
                [--beta BETA] [--lrate LRATE] [--beta_1 BETA_1] [--beta_2 BETA_2]
                [--encoder_save ENCODER_SAVE] [--decoder_save DECODER_SAVE]
                [--log_file LOG_FILE] [--load_saved LOAD_SAVED]
                [--start_epoch START_EPOCH] [--end_epoch END_EPOCH]
```
- - - - 
### Testing

- Inference Grid
```
python inference_grid.py
```
- Style Transfer 
```
python style_transfer_across_videos.py
```
- Latent Visualization
```
python videowise_visualization.py [--num_points_visualization NUM_POINTS_VISUALIZATION]
```
- Interpolation (Linear)
```
python linear_interpolation.py [--threshold THRESHOLD] [--step_size STEP_SIZE] [--num_interpolation NUM_INTERPOLATION]
```
