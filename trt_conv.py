import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch2trt import torch2trt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable

#from PIL import Image
#import cv2

import time
import numpy as np

res = 512
s = 0.5

source = '/home/eehpc-nano/ARL_Collab/pretrained/UMBV2_models/'
dest = '/home/eehpc-nano/ARL_Collab/pretrained/TensorRT_Engines/'

print('Loading Model from: '+source+'Unet-umbv2_'+repr(res)+'-data-'+repr(s)+'s.pt')

model = torch.load(source+'Unet-umbv2_'+repr(res)+'-data-'+repr(s)+'s.pt').eval().cuda()
print('Model Loaded')

image = torch.ones((1, 3, res, res)).cuda()

print('\nStarting TensorRT Conversion')
model_trt = torch2trt(model, [image])
print('Finished TensorRT Conversion')

print('\nSaving TensorRT Engine to: '+dest+'Unet-umbv2_'+repr(res)+'-data-'+repr(s)+'s.pth')

torch.save(model_trt.state_dict(), dest+'Unet-umbv2_'+repr(res)+'-data-'+repr(s)+'s.pth')
print('Saved TensorRT Engine')

