"""Utilities for visualising images."""

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def image_grid(x,
               size,
               channels,
               grid_size=None): # a tuple of integers
	# size = config.data.image_size
	# channels = config.data.num_channels
	img = x.reshape(-1, size, size, channels)
	print(img.shape)
	if grid_size is None:
		w = int(np.sqrt(img.shape[0]))
		img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
	else:
		h = grid_size[0]
		w = grid_size[1]
		img = img.reshape((h, w,
							size, size, channels)).\
							transpose((0, 2, 1, 3, 4)).\
							reshape((h * size, w * size, channels))
	return img


def show_samples(x,
                 size,
                 channels,
                 figsize=(6,6),
                 name="Samples",
                 grid_size=None):
	if channels>3:
		x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
		img = image_grid(x, size, channels, grid_size)
		plt.figure(figsize=figsize)
		plt.axis('off')
		plt.title(name)
		plt.imshow(img)
	else:
		if grid_size[0] != grid_size[1]:
			raise ValueError("Not square grid")

		img = make_grid(x, nrow=grid_size[1])
		plt.figure(figsize=figsize)
		plt.axis('off')
		plt.imshow(img.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
		plt.show()
