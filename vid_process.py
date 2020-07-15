# this file contains code for basic preprocessing of videos before training 

import moviepy.editor as mp
from moviepy.video.fx.all import crop
import numpy as np
import matplotlib.pyplot as plt

from os import listdir

def resize_cropped(filename, side):
	"resize video denoted by filename to (side x side) video and output it as a numpy array of (frames x h x w x 3)"
	
	# Taking video file
	clip = mp.VideoFileClip(filename)
	(w, h) = clip.size

	# checking the type of video file
	landscape = (w >= h)

	# new dimensions
	h2 = w2 = side

	# rescaling
	clip_h = clip.resize(height = h2)       # height is constrained
	(w_h, h_h) = clip_h.size

	clip_w = clip.resize(width = w2)        # width is constrained
	(w_w, h_w) = clip_w.size

	# creating videos
	if landscape:
		cropped = crop(clip_h, width=w2, height=h2, x_center=w_h/2, y_center=h_h/2)
	else:
		cropped = crop(clip_w, width=w2, height=h2, x_center=w_w/2, y_center=h_w/2)
	
	# video to array
	cropped_array = np.asarray(list(cropped.iter_frames()))
	
	return cropped_array

def resize_keepAR(filename, h_0, write):
	"resize video denoted by filename to make height h_0 keeping aspect ratio, write it and output it as a numpy array of (frames x h x w x 3)"
	
	# Taking video file
	clip = mp.VideoFileClip(filename)

	# rescaling
	clip_h = clip.resize(height = h_0)       # height is constrained

	if write:
		clip_h.write_videofile("test.avi", codec='mpeg4')
	
	# video to array
	keepAR_array = np.asarray(list(clip_h.iter_frames()))

	return keepAR_array

def resize_mnist(filename, h_0):
	"resize video denoted by filename to make height h_0 keeping aspect ratio, write it and output it as a numpy array of (frames x h x w)"
	
	# Taking video file
	clip_h = mp.VideoFileClip(filename)

	# video to array
	vid_array = np.asarray(list(clip_h.iter_frames()))

	vid_array = np.transpose(vid_array, (3,0,1,2))			# 3 x FRAMES x H x W

	return vid_array[0]
