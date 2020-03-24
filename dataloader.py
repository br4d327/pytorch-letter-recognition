## -*- coding: utf-8 -*-
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from torch import Tensor
import cvfunction
from scipy.misc import *

class LettersDataset(Dataset):

	def __init__(self, root):

		Images, Y = [], []
		folders = os.listdir(root)

		for folder in folders:
			folder_path = os.path.join(root, folder)
			for ims in os.listdir(folder_path):
				try:
					img_path = os.path.join(folder_path, ims)
					img_transforming = cvfunction.image_processing(img_path)
					img_trimming = cvfunction.get_square(img_transforming, 28)
					Images.append(img_trimming)
					Y.append(ord(folder) - 65)

				except:
					print("File {}/{} is broken".format(folder, ims))

		data = [(x, y) for x, y in zip(Images, Y)]
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img = self.data[index][0]
		# 8 bit images. Scale between [0,1]. This helps speed up our training
		img = img.reshape(28, 28) / 255.0

		# Input for Conv2D should be Channels x Height x Width
		img_tensor = Tensor(img).view(1, 28, 28).float()
		label = self.data[index][1]
		return (img_tensor, label)

