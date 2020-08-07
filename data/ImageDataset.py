#LOADING THE IMAGE FILES IN A CUSTOM WAY
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections.abc import Iterable
from utils.Phase import Phase

from PIL import Image
import utils.im_manipulation as ImageManipulator
#IMAGES WILL BE pixel_res
pixel_res = 32

class ImageDataset(Dataset):

    # READS ALL IMAGE FILE IN THE TARGET DIRECTORY AND CREATES TWO LISTS - KEYS & FILEPATHS WHOSE INDICES CORRESPOND
    def constructImageDictionary(self, img_dir, ext):
        img_keys = []
        img_paths = []
        for roots, dir, files in os.walk(img_dir):
            for file in files:
                if ext in file:
                    file_abs_path = os.path.join(roots, file)
                    tokens = file_abs_path.split('/');
                    ln = len(tokens)
                    geohash = tokens[ln - 2]

                    f_tokens = file.split('$')
                    date = f_tokens[0]

                    img_keys.append(geohash + '$' + date)
                    img_paths.append(file_abs_path)
        return img_keys, img_paths

    # IMAGE UPSCALING
    # img: PIL image
    # INCREASE RESOLUTION OF IMAGE BY FACTOR OF 2^pow
    def upscaleImage(self, img, pow, method=Image.BICUBIC):
        ow, oh = img.size
        scale = 2**pow
        h = int(round(oh * scale))
        w = int(round(ow * scale))
        return img.resize((w, h), method)




    def imageResize(self, image):
        new_image = image.resize((pixel_res, pixel_res))
        return new_image


    # RETURNS A LOW RES AND HIGH RES VERSION OF THE IMAGE AT A GIVEN INDEX
    def __getitem__(self, index):
        ret_data = {}
        ret_data['scale'] = self.current_scale

        scale = self.current_scale

        # Load HIGH-RES image
        if len(self.image_fnames):
            # FULL HIGH RESOLUTION IMAGE
            target_img = ImageManipulator.image_loader(self.image_fnames[index])

            w, h = target_img.size
            # IF IMAGE PIXELS IS NOT DIVISIBLE BY SCALE, CROP THE BALANCE
            target_img = target_img.crop((0, 0, w - w % scale, h - h % scale))

            target_img = ImageManipulator.imageResize(target_img, self.highest_res)

            # THIS IS HOW MUCH TO DE-MAGNIFY THE MAIN IMAGE
            # eg. if target is x2, target has to be divided by 4 and source has to be divided by 8
            div_factor = self.max_scale / self.current_scale

            # LOW-RES IMAGE GENERATION

            hr_image = ImageManipulator.downscaleImage(target_img, div_factor)
            lr_image = ImageManipulator.downscaleImage(target_img, self.max_scale)

            if self.augmentON:
                # ROTATING & FLIPPING IMAGE PAIRS
                ret_data['lr'], ret_data['hr'] = ImageManipulator.augment_pairs(
                                                            lr_image, hr_image)

            else:
                ret_data['hr'] = hr_image
                ret_data['lr'] = lr_image

            # IMAGE & FILE_NAME
            ret_data['bicubic'] = ImageManipulator.downscaleImage(ret_data['lr'],1 / scale)
            ret_data['hr_fname'] = self.image_fnames[index]

            ret_data['hr'] = self.normalize_fn(ret_data['hr'])
            ret_data['lr'] = self.normalize_fn(ret_data['lr'])
            ret_data['bicubic'] = self.normalize_fn(ret_data['bicubic'])


        return ret_data

    def __len__(self):
        return len(self.image_fnames)

    # DISPLAY AN IMAGE AT A GIVEN PATH
    def displayImage(self, image_path):
        img = Image.open(image_path)
        plt.figure()
        plt.imshow(img)
        plt.show()

    # DISPLAY AN IMAGE OBJECT
    def displayImageMem(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def getTransforms(self):
        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    # CALLED DURING SWITCHING OF GEARS
    def set_scales(self, s_ind):
        if s_ind >= 0 and s_ind< len(self.scales):
            self.current_scale_id = s_ind
            self.current_scale = self.scales[s_ind]
            print("RESET TO "+str(self.current_scale))
        else:
            print("RIKI: SCALE ERROR")

    def copy_scales(self, scal):
        if self.scales.index(scal) >= 0 :
            self.current_scale = scal
            #print("RESET TEST SCALE TO "+str(self.current_scale))
        else:
            print("RIKI: SCALE ERROR")

    def __init__(self, phase, img_dir, img_type, mean, stddev, scales, high_res):
        self.phase = phase
        self.mean = mean
        self.stddev = stddev

        # THE RESOLUTION OF THE x8 ACTUAL IMAGE
        self.highest_res = high_res

        # INITIALIZING SCALES
        self.scales = scales if isinstance(scales, Iterable) else [scales]
        self.current_scale = self.scales[0]
        self.current_scale_id = self.scales.index(self.current_scale)+1
        self.max_scale = np.max(self.scales)

        # IF PHASE IS TRAINING PHASE, WE AUGMENT THE INPUT IMAGE RANDOMLY
        #self.augmentON = self.phase == Phase.TRAIN
        self.augmentON = False
        # THE DIRECTIRY WHERE IMAGE ARE
        self.img_dir = img_dir

        # ALL IMAGE FILES IN AN ARRAY
        self.image_fnames = ImageManipulator.get_filenames(img_dir, img_type)

        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])
