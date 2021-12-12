from torchvision import transforms
import random
import torch
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F

import scipy.misc
import SimpleITK as sitk
import os
import time
import shutil

# from  A Fourier-based Framework for Domain Generalization
def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    # img21 = np.uint8(np.clip(img21, 0, 255))
    # img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12

# from FedDG
def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_target, axes=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # deep copy
    a_local_copy = a_local.copy()
    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    a_trg[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2] * ratio + a_local_copy[:,h1:h2,w1:w2] * (1- ratio)

    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1))
    return a_local, a_trg

def freq_space_interpolation(local_img, trg_img, L=0 , ratio=0):

    local_img_np = local_img
    tar_img_np = trg_img

    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( tar_img_np, axes=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_, amp_trg_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    fft_trg_ = amp_trg_ * np.exp( 1j * pha_trg )
    trg_in_local = np.fft.ifft2( fft_trg_, axes=(-2, -1) )
    trg_in_local = np.real(trg_in_local)

    return local_in_trg, trg_in_local

# i is the lambda of target
def fourier_transform(im_local, im_trg, L=0.01, i=1):
    im_local = im_local.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))
    local_in_trg, trg_in_local = freq_space_interpolation(im_local, im_trg, L=L, ratio=1-i)
    local_in_trg = local_in_trg.transpose((1, 2, 0))
    trg_in_local = trg_in_local.transpose((1, 2, 0))
    return local_in_trg, trg_in_local

def save_image(image, path):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path)
    plt.show()
    return 0

if __name__ == '__main__':
    img1 = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorA/000003.npz'
    img2 = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorB/center2/000002.npz'
    mask1 = '/home/listu/code/semi_medical/mnms_split_2D/mask/Labeled/vendorA/000003.png'
    mask2 = '/home/listu/code/semi_medical/mnms_split_2D/mask/Labeled/vendorB/center2/000002.png'

    image_size=224
    img_transform = transforms.Compose([
                transforms.Resize((image_size, image_size))
            ])

    mask1 = Image.open(mask1)
    mask2 = Image.open(mask2)
    mask1 = img_transform(mask1)
    mask2 = img_transform(mask2)
    mask1 = np.asarray(mask1)
    mask2 = np.asarray(mask2)

    img1 = np.load(img1)['arr_0']
    img2 = np.load(img2)['arr_0']
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    img1 = img_transform(img1)
    img2 = img_transform(img2)

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    img1 = img1[:,:,np.newaxis]
    img2 = img2[:,:,np.newaxis]
    print(img1.shape)
    print(img2.shape)

    im_local = img1
    im_trg = img2
    aug_local, aug_traget = fourier_transform(im_local, im_trg)
    save_image((im_local / 255),'./fpic/im_local.png')
    save_image((im_trg / 255),'./fpic/im_trg.png')
    save_image((aug_local / 255),'./fpic/aug_local.png')
    save_image((aug_traget / 255),'./fpic/aug_traget.png')
