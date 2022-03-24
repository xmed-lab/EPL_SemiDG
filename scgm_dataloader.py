from PIL import Image
import torchfile
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import os
import sys
import torchvision.utils as vutils
import numpy as np
import torch.nn.init as init
import torch.utils.data as data
import random
import xlrd
import math
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import im_convert
from utils.data_utils import colorful_spectrum_mix, fourier_transform, save_image
from config import default_config

LabeledVendorA_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Labeled/vendorA/'
LabeledVendorA_mask_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/mask/Labeled/vendorA/'

LabeledVendorB_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Labeled/vendorB/'
LabeledVendorB_mask_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/mask/Labeled/vendorB/'

LabeledVendorC_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Labeled/vendorC/'
LabeledVendorC_mask_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/mask/Labeled/vendorC/'

LabeledVendorD_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Labeled/vendorD/'
LabeledVendorD_mask_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/mask/Labeled/vendorD/'

UnlabeledVendorA_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Unlabeled/vendorA/'
UnlabeledVendorB_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Unlabeled/vendorB/'
UnlabeledVendorC_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Unlabeled/vendorC/'
UnlabeledVendorD_data_dir = '/home/hyaoad/remote/semi_medical/scgm_split_2D/data/Unlabeled/vendorD/'

Labeled_data_dir = [LabeledVendorA_data_dir, LabeledVendorB_data_dir, LabeledVendorC_data_dir, LabeledVendorD_data_dir]
Labeled_mask_dir = [LabeledVendorA_mask_dir, LabeledVendorB_mask_dir, LabeledVendorC_mask_dir, LabeledVendorD_mask_dir]
Unlabeled_data_dir = [UnlabeledVendorA_data_dir, UnlabeledVendorB_data_dir, UnlabeledVendorC_data_dir, UnlabeledVendorD_data_dir]

def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

def default_loader(path):
    return np.load(path)['arr_0']

def fourier_augmentation(img, tar_img, mode, alpha):
    # transfer image from PIL to numpy
    img = np.array(img)
    tar_img = np.array(tar_img)
    img = img[:,:,np.newaxis]
    tar_img = tar_img[:,:,np.newaxis]

    # the mode comes from the paper "A Fourier-based Framework for Domain Generalization"
    if mode == 'AS':
        # print("using AS mode")
        aug_img, aug_tar_img = fourier_transform(img, tar_img, L=0.01, i=1)
    elif mode == 'AM':
        # print("using AM mode")
        aug_img, aug_tar_img = colorful_spectrum_mix(img, tar_img, alpha=alpha)
    else:
        print("mode name error")

    aug_img = np.squeeze(aug_img)
    aug_img = Image.fromarray(aug_img)

    aug_tar_img = np.squeeze(aug_tar_img)
    aug_tar_img = Image.fromarray(aug_tar_img)

    return aug_img, aug_tar_img

def get_meta_split_data_loaders(test_vendor='D'):
    random.seed(14)

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
    domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
    test_dataset = \
        get_data_loader_folder(Labeled_data_dir, Labeled_mask_dir, test_num=test_vendor)

    return  domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
            domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
            test_dataset 

def get_data_loader_folder(data_folders, mask_folders, test_num='D'):
    if test_num=='A':
        domain_1_img_dirs = data_folders[1]
        domain_1_mask_dirs = mask_folders[1]
        domain_2_img_dirs = data_folders[2]
        domain_2_mask_dirs = mask_folders[2]
        domain_3_img_dirs = data_folders[3]
        domain_3_mask_dirs = mask_folders[3]

        fourier_dirs = [data_folders[1], data_folders[2], data_folders[3]]

        test_data_dirs = data_folders[0]
        test_mask_dirs = mask_folders[0]

    elif test_num=='B':
        domain_1_img_dirs = data_folders[0]
        domain_1_mask_dirs = mask_folders[0]
        domain_2_img_dirs = data_folders[2]
        domain_2_mask_dirs = mask_folders[2]
        domain_3_img_dirs = data_folders[3]
        domain_3_mask_dirs = mask_folders[3]

        fourier_dirs = [data_folders[0], data_folders[2], data_folders[3]]

        test_data_dirs = data_folders[1]
        test_mask_dirs = mask_folders[1]

    elif test_num=='C':
        domain_1_img_dirs = data_folders[1]
        domain_1_mask_dirs = mask_folders[1]
        domain_2_img_dirs = data_folders[0]
        domain_2_mask_dirs = mask_folders[0]
        domain_3_img_dirs = data_folders[3]
        domain_3_mask_dirs = mask_folders[3]

        fourier_dirs = [data_folders[1], data_folders[0], data_folders[3]]

        test_data_dirs = data_folders[2]
        test_mask_dirs = mask_folders[2]

    elif test_num=='D':
        domain_1_img_dirs = data_folders[1]
        domain_1_mask_dirs = mask_folders[1]
        domain_2_img_dirs = data_folders[2]
        domain_2_mask_dirs = mask_folders[2]
        domain_3_img_dirs = data_folders[0]
        domain_3_mask_dirs = mask_folders[0]

        fourier_dirs = [data_folders[1], data_folders[2], data_folders[0]]

        test_data_dirs = data_folders[3]
        test_mask_dirs = mask_folders[3]

    else:
        print('Wrong test vendor!')

    print("loading labeled dateset")
    domain_1_labeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, fourier_dir=fourier_dirs , label=True, train=True)
    domain_2_labeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, fourier_dir=fourier_dirs , label=True, train=True)
    domain_3_labeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, fourier_dir=fourier_dirs , label=True, train=True)

    print("loading unlabeled dateset")
    domain_1_unlabeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, fourier_dir=fourier_dirs, label=False, train=True)
    domain_2_unlabeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, fourier_dir=fourier_dirs, label=False, train=True)
    domain_3_unlabeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, fourier_dir=fourier_dirs, label=False, train=True)

    print("loading test dateset")
    test_dataset = ImageFolder(test_data_dirs, test_mask_dirs, fourier_dir=fourier_dirs, label=True, train=False)

    return domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
           domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
           test_dataset

class ImageFolder(data.Dataset):
    def __init__(self, data_dir, mask_dir, fourier_dir=None, train=True, label=True, loader=default_loader):

        print("data_dirs", data_dir)
        # print("mask_dirs", mask_dir)
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.fourier_dir = fourier_dir
        self.loader = loader
        self.train = train
        self.label = label
        self.newsize = 288

        if self.train and self.label:
            ratio = 0.2       #  20%
        else:
            ratio = 1

        data_roots = sorted(make_dataset(self.data_dir))
        data_len = len(data_roots)
        new_data_dir = []
        new_len = int(data_len * ratio)
        for i in range(new_len):
            new_data_dir.append(data_roots[i])

        mask_roots = sorted(make_dataset(self.mask_dir))
        mask_len = len(mask_roots)
        new_mask_dir = []
        new_len = int(mask_len * ratio)
        for i in range(new_len):
            new_mask_dir.append(mask_roots[i])

        fourier_imgs = []
        # for Fourier dirs
        if self.train == True :
            for num_set in range(len(fourier_dir)):
                data_roots = sorted(make_dataset(fourier_dir[num_set]))
                for num_data in range(len(data_roots)):
                    fourier_imgs.append(data_roots[num_data])
        
        self.imgs = new_data_dir
        self.masks = new_mask_dir
        self.fourier = fourier_imgs

        print("length of imgs",len(self.imgs))

        self.Fourier_aug = default_config['Fourier_aug']
        self.fourier_mode = default_config['fourier_mode']
        self.alpha = 0.3

    def __getitem__(self, index):

        path_img = self.imgs[index]
        img = self.loader(path_img) 
        img = Image.fromarray(img)
        h, w = img.size
        # print(h, w)

        path_mask = self.masks[index]
        mask = self.loader(path_mask)
        mask = mask[:,:,1]
        mask = Image.fromarray(mask)

        # for Fourier dirs
        if self.train == True :
            fourier_paths = random.sample(self.fourier, 1)
            fourier_img = self.loader(fourier_paths[0])
            fourier_img = Image.fromarray(fourier_img)

        # label
        if self.label:
            # train labeled data
            if self.train:
                # rotate, random angle between 0 - 90
                angle = random.randint(0, 90)
                img = F.rotate(img, angle, InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, InterpolationMode.NEAREST)
                if h > 110 and w > 110:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.newsize, self.newsize))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)

                img = transform(img)
                mask = transform(mask)

                fourier_img = F.rotate(fourier_img, angle, InterpolationMode.BILINEAR)
                fourier_img = transform(fourier_img)
                aug_img, _ = fourier_augmentation(img, fourier_img, self.fourier_mode, self.alpha)

                img = F.to_tensor(np.array(img))
                aug_img = F.to_tensor(np.array(aug_img))
                mask = F.to_tensor(np.array(mask))
                mask = (mask > 0.1).float()

                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0)
            # test data
            else:
                if h > 110 and w > 110:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.newsize, self.newsize))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)
                img = transform(img)
                mask = transform(mask)

                img = F.to_tensor(np.array(img))
                mask = F.to_tensor(np.array(mask))
                mask = (mask > 0.1).float()
                aug_img = torch.tensor([0])

        # train unlabel data
        else:
            # rotate, random angle between 0 - 90
            angle = random.randint(0, 90)
            img = F.rotate(img, angle, InterpolationMode.BILINEAR)

            if h > 110 and w > 110:
                size = (100, 100)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = [transforms.Resize((self.newsize, self.newsize))] + transform_list
                transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                transform = transforms.Compose(transform_list)
            else:
                size = (100, 100)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                transform = transforms.Compose(transform_list)

            img = transform(img)

            fourier_img = F.rotate(fourier_img, angle, InterpolationMode.BILINEAR)
            fourier_img = transform(fourier_img)
            aug_img, _ = fourier_augmentation(img, fourier_img, self.fourier_mode, self.alpha)

            img = F.to_tensor(np.array(img))
            aug_img = F.to_tensor(np.array(aug_img))
            mask = torch.tensor([0])

        ouput_dict = dict(
            img = img,
            aug_img = aug_img,
            mask = mask,
            path_img = path_img
        )
        return ouput_dict # pytorch: N,C,H,W

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    test_vendor = 'D'

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
    domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
    test_dataset  = get_meta_split_data_loaders(test_vendor=test_vendor)

    label_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True, pin_memory=True)

    dataiter = iter(label_loader)
    output = dataiter.next()
    img = output['img']
    mask = output['mask']
    aug_img = output['aug_img']

    print(img.shape)
    print(mask.shape)
    print(aug_img.shape)

    # torch.set_printoptions(threshold=np.inf)
    # with open('./mask.txt', 'wt') as f:
    #     print(mask, file=f)
    mask = mask[:,0,:,:]
    img = im_convert(img, False)
    # aug_img = im_convert(aug_img, False)
    mask = im_convert(mask, False)
    save_image(img, './fpic/label_'+str(default_config['fourier_mode'])+'_img.png')
    # save_image(aug_img, './fpic/label_'+str(default_config['fourier_mode'])+'_aug_img.png')
    save_image(mask, './fpic/label_'+str(default_config['fourier_mode'])+'_mask.png')
