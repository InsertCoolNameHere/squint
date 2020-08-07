import os
import os.path as osp
import glob
import torch.nn as nn
from math import floor
from numpy import random
from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from skimage import img_as_float
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr, compare_ssim


def get_filenames(source, ext):
    img_paths = []
    for roots, dir, files in os.walk(source):
        for file in files:
            if ext in file:
                file_abs_path = os.path.join(roots, file)
                tokens = file_abs_path.split('/');
                ln = len(tokens)
                geohash = tokens[ln - 2]

                f_tokens = file.split('$')
                date = f_tokens[0]

                img_paths.append(file_abs_path)
    return img_paths


# RETURNS ALL FILENAMES IN THE GIVEN DIRECTORY IN AN ARRAY
def get_filenames1(source, image_format):
    # If image_format is a list
    if source is None:
        return []
    # Seamlessy load single file, list of files and files from directories.
    source_fns = []
    if isinstance(source, str):
        if os.path.isdir(source) or source[-1] == '*':
            if isinstance(image_format, list):
                for fmt in image_format:
                    source_fns += get_filenames(source, fmt)
            else:
                source_fns = sorted(
                    glob.glob("{}/*.{}".format(source, image_format)))
        elif os.path.isfile(source):
            source_fns = [source]
        assert (all([is_image_file(f) for f in source_fns
                     ])), "Given files contain files with unsupported format"
    elif len(source) and isinstance(source[0], str):
        for s in source:
            source_fns.extend(get_filenames(s, image_format=image_format))
    return source_fns


def load_dataset(args, img_format):
    files = {'train':{},'test':{}}

    for phase in ['train','test']:
        for ft in ['source','target']:
            if args[phase].dataset.path[ft]:
                files[phase][ft] = get_filenames(args[phase].dataset.path[ft], image_format=img_format)
            else:
                files[phase][ft] = []

    return files['train'],files['test']

def is_image_file(filename, ext):
    return any(filename.lower().endswith(ext))



def image_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def imageResize(image, pixel_res):
    new_image = image.resize((pixel_res, pixel_res))
    return new_image

# IMAGE DOWNGRADING
    # img: PIL image
    # DECREASE RESOLUTION OF IMAGE BY FACTOR OF 2^pow
def downscaleImage(img, scale, method=Image.BICUBIC):
    ow, oh = img.size

    h = int(round(oh / scale))
    w = int(round(ow / scale))
    return img.resize((w, h), method)


def random_rot90(img, r=None):
    if r is None:
        r = random.random() * 4  # TODO Check and rewrite func
    if r < 1:
        return img.transpose(Image.ROTATE_90)
    elif r < 2:
        return img.transpose(Image.ROTATE_270)
    elif r < 3:
        return img.transpose(Image.ROTATE_180)
    else:
        return img


# RANDOM FLIPPING AND ROTATION OF THE INPUT AND TARGET IMAGES
def augment_pairs(img1, img2):
    vflip = random.random() > 0.5
    hflip = random.random() > 0.5
    rot = random.random() * 4
    if hflip:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
        img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

    img1 = random_rot90(img1, rot)
    img2 = random_rot90(img2, rot)
    return img1, img2

# COMBINE 3 IMAGES INTO A SINGLE IMAGE-GRID
'''
def combine_images1(lr, hr, gen, scale):
    img_hr = make_grid(hr, nrow=1, normalize=False, padding=10, pad_value=0)

    img_lr = nn.functional.interpolate(lr, scale_factor=scale)

    img_lr = make_grid(img_lr, nrow=1, normalize=False, padding=10, pad_value=0)
    gen_img = make_grid(gen, nrow=1, normalize=False, padding=10, pad_value=0)

    img_grid = torch.cat((img_lr.cpu(), img_hr.cpu(), gen_img.cpu(),), -1)
    return img_grid
'''

# COMBINE 3 IMAGES INTO A SINGLE IMAGE-GRID
def combine_images(lr, hr, gen, scale):

    img_lr = nn.functional.interpolate(lr, scale_factor=scale)
    img_grid = torch.cat((img_lr.detach().cpu(), hr.detach().cpu(), gen.detach().cpu()), 0)
    # nrow= NUMBER OF IMAGES IN A SINGLE ROW
    img = make_grid(img_grid, nrow=3, normalize=False, padding=3, pad_value=255)

    return img

# COMBINE 4 IMAGES INTO A SINGLE IMAGE-GRID...DURING TESTING
def combine_images_test(lr, hr, bic, gen, scale):

    img_lr = nn.functional.interpolate(lr, scale_factor=scale)
    img_grid = torch.cat((img_lr.detach().cpu(), hr.detach().cpu(), bic.detach().cpu(), gen.detach().cpu()), 0)
    # nrow= NUMBER OF IMAGES IN A SINGLE ROW
    img = make_grid(img_grid, nrow=4, normalize=False, padding=3, pad_value=255)

    return img

# Converts a Tensor into a Numpy array
def tensor2im(image_tensor, mean=(0.5, 0.5, 0.5), stddev=2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy,
                                (1, 2, 0)) * stddev + np.array(mean)) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    return np.around(image_numpy).astype(np.uint8)



def crop_boundaries(im, cs):
    if cs > 1:
        return im[cs:-cs, cs:-cs, ...]
    else:
        return im

def mod_crop(im, scale):
    h, w = im.shape[:2]
    # return im[(h % scale):, (w % scale):, ...]
    return im[:h - (h % scale), :w - (w % scale), ...]


def eval_psnr_and_ssim(im1, im2, scale):
    im1_t = np.atleast_3d(img_as_float(im1))
    im2_t = np.atleast_3d(img_as_float(im2))

    if im1_t.shape[2] == 1 or im2_t.shape[2] == 1:
        im1_t = im1_t[..., 0]
        im2_t = im2_t[..., 0]

    else:
        im1_t = rgb2ycbcr(im1_t)[:, :, 0:1] / 255.0
        im2_t = rgb2ycbcr(im2_t)[:, :, 0:1] / 255.0

    if scale > 1:
        im1_t = mod_crop(im1_t, scale)
        im2_t = mod_crop(im2_t, scale)

        # NOTE conventionally, crop scale+6 pixels (EDSR, VDSR etc)
        im1_t = crop_boundaries(im1_t, int(scale) + 6)
        im2_t = crop_boundaries(im2_t, int(scale) + 6)

    psnr_val = compare_psnr(im1_t, im2_t)
    ssim_val = compare_ssim(
        im1_t,
        im2_t,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        data_range=1.0,
        K1=0.01,
        K2=0.03,
        sigma=1.5)

    return psnr_val, ssim_val

