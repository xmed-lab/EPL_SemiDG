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

from utils.utils import im_convert
from utils.data_utils import colorful_spectrum_mix, fourier_transform, save_image
from config import default_config

# torch.cuda.set_device(6)

# Data directories
LabeledVendorA_data_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/data/Labeled/vendorA/'
LabeledVendorA_mask_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/mask/Labeled/vendorA/'
ReA_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D_re/Labeled/vendorA/'

LabeledVendorB2_data_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/data/Labeled/vendorB/center2/'
LabeledVendorB2_mask_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/mask/Labeled/vendorB/center2/'
ReB2_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D_re/Labeled/vendorB/center2/'

LabeledVendorB3_data_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/data/Labeled/vendorB/center3/'
LabeledVendorB3_mask_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/mask/Labeled/vendorB/center3/'
ReB3_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D_re/Labeled/vendorB/center3/'

LabeledVendorC_data_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/data/Labeled/vendorC/'
LabeledVendorC_mask_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/mask/Labeled/vendorC/'
ReC_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D_re/Labeled/vendorC/'

LabeledVendorD_data_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/data/Labeled/vendorD/'
LabeledVendorD_mask_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/mask/Labeled/vendorD/'
ReD_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D_re/Labeled/vendorD/'

UnlabeledVendorC_data_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D/data/Unlabeled/vendorC/'
UnReC_dir = '/home/hyaoad/remote/semi_medical/mnms_split_2D_re/Unlabeled/vendorC/'

Re_dir = [ReA_dir, ReB2_dir, ReB3_dir, ReC_dir, ReD_dir]
Labeled_data_dir = [LabeledVendorA_data_dir, LabeledVendorB2_data_dir, LabeledVendorB3_data_dir, LabeledVendorC_data_dir, LabeledVendorD_data_dir]
Labeled_mask_dir = [LabeledVendorA_mask_dir, LabeledVendorB2_mask_dir, LabeledVendorB3_mask_dir, LabeledVendorC_mask_dir, LabeledVendorD_mask_dir]

def get_meta_split_data_loaders(test_vendor='D', image_size=224, batch_size=1):
    random.seed(14)

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
    domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
    test_dataset = \
        get_data_loader_folder(Labeled_data_dir, Labeled_mask_dir, batch_size, image_size, test_num=test_vendor)

    return  domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
            domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
            test_dataset 

def get_data_loader_folder(data_folders, mask_folders, batch_size, new_size=288, test_num='D', num_workers=0):
    if test_num=='A':
        domain_1_img_dirs = [data_folders[1], data_folders[2]]
        domain_1_mask_dirs = [mask_folders[1], mask_folders[2]]
        domain_2_img_dirs = [data_folders[3]]
        domain_2_mask_dirs = [mask_folders[3]]
        domain_3_img_dirs = [data_folders[4]]
        domain_3_mask_dirs = [mask_folders[4]]

        fourier_dirs = [data_folders[1], data_folders[2], data_folders[3], data_folders[4]]
        fourier_masks = [mask_folders[1], mask_folders[2], mask_folders[3], mask_folders[4]]

        test_data_dirs = [data_folders[0]]
        test_mask_dirs = [mask_folders[0]]

        domain_1_re = [Re_dir[1], Re_dir[2]]
        domain_2_re = [Re_dir[3]]
        domain_3_re = [Re_dir[4]]

        test_re = [Re_dir[0]]

        domain_1_num = [74, 51]
        domain_2_num = [50]
        domain_3_num = [50]
        test_num = [95]

    elif test_num=='B':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[3]]
        domain_2_mask_dirs = [mask_folders[3]]
        domain_3_img_dirs = [data_folders[4]]
        domain_3_mask_dirs = [mask_folders[4]]

        fourier_dirs = [data_folders[0], data_folders[3], data_folders[4]]
        fourier_masks = [mask_folders[0], mask_folders[3], mask_folders[4]]

        test_data_dirs = [data_folders[1], data_folders[2]]
        test_mask_dirs = [mask_folders[1], mask_folders[2]]

        domain_1_re = [Re_dir[0]]
        domain_2_re = [Re_dir[3]]
        domain_3_re = [Re_dir[4]]
        test_re = [Re_dir[1], Re_dir[2]]

        domain_1_num = [95]
        domain_2_num = [50]
        domain_3_num = [50]
        test_num = [74, 51]

    elif test_num=='C':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1], data_folders[2]]
        domain_2_mask_dirs = [mask_folders[1], mask_folders[2]]
        domain_3_img_dirs = [data_folders[4]]
        domain_3_mask_dirs = [mask_folders[4]]

        fourier_dirs = [data_folders[1], data_folders[2], data_folders[0], data_folders[4]]
        fourier_masks = [mask_folders[1], mask_folders[2], mask_folders[0], mask_folders[4]]

        test_data_dirs = [data_folders[3]]
        test_mask_dirs = [mask_folders[3]]

        domain_1_re = [Re_dir[0]]
        domain_2_re = [Re_dir[1], Re_dir[2]]
        domain_3_re = [Re_dir[4]]
        test_re = [Re_dir[3]]

        domain_1_num = [95]
        domain_2_num = [74, 51]
        domain_3_num = [50]
        test_num = [50]

    elif test_num=='D':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1], data_folders[2]]
        domain_2_mask_dirs = [mask_folders[1], mask_folders[2]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        fourier_dirs = [data_folders[0], data_folders[1], data_folders[2], data_folders[3]]
        fourier_masks = [mask_folders[0], mask_folders[1], mask_folders[2], mask_folders[3]]

        test_data_dirs = [data_folders[4]]
        test_mask_dirs = [mask_folders[4]]

        domain_1_re = [Re_dir[0]]
        domain_2_re = [Re_dir[1], Re_dir[2]]
        domain_3_re = [Re_dir[3]]
        test_re = [Re_dir[4]]

        domain_1_num = [95]
        domain_2_num = [74, 51]
        domain_3_num = [50]
        test_num = [50]

    else:
        print('Wrong test vendor!')

    print("loading labeled dateset")
    domain_1_labeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, domain_1_img_dirs, domain_1_re, fourier_dir=fourier_dirs, fourier_mask=fourier_masks, label=0, num_label=domain_1_num, train=True, labeled=True)
    domain_2_labeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, domain_1_img_dirs, domain_2_re, fourier_dir=fourier_dirs, fourier_mask=fourier_masks, label=1, num_label=domain_2_num, train=True, labeled=True)
    domain_3_labeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, domain_1_img_dirs, domain_3_re, fourier_dir=fourier_dirs, fourier_mask=fourier_masks, label=2, num_label=domain_3_num, train=True, labeled=True)

    # domain_1_labeled_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    # domain_2_labeled_loader = DataLoader(dataset=domain_2_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    # domain_3_labeled_loader = DataLoader(dataset=domain_3_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    print("loading unlabeled dateset")
    domain_1_unlabeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, domain_1_img_dirs, domain_1_re, fourier_dir=fourier_dirs, fourier_mask=fourier_masks, label=0, train=True, labeled=False)
    domain_2_unlabeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, domain_1_img_dirs, domain_2_re, fourier_dir=fourier_dirs, fourier_mask=fourier_masks, label=1, train=True, labeled=False)
    domain_3_unlabeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, domain_1_img_dirs, domain_3_re, fourier_dir=fourier_dirs, fourier_mask=fourier_masks, label=2, train=True, labeled=False)

    # domain_1_unlabeled_loader = DataLoader(dataset=domain_1_unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    # domain_2_unlabeled_loader = DataLoader(dataset=domain_2_unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    # domain_3_unlabeled_loader = DataLoader(dataset=domain_3_unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)

    print("loading test dateset")
    test_dataset = ImageFolder(test_data_dirs, test_mask_dirs, domain_1_img_dirs, test_re, train=False, labeled=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)

    return domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
           domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
           test_dataset

def default_loader(path):
    return np.load(path)['arr_0']

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

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

class ImageFolder(data.Dataset):
    def __init__(self, data_dirs, mask_dirs, ref_dir, re, fourier_dir=None, fourier_mask=None, train=True, label=None, num_label=None, labeled=True, loader=default_loader):

        print("data_dirs", data_dirs)
        print("mask_dirs", mask_dirs)

        reso_dir = re
        temp_imgs = []
        temp_masks = []
        temp_re = []
        domain_labels = []

        tem_ref_imgs = []

        fourier_imgs = []
        labeled_fourier_imgs = []

        if train:
            #100%
            # k = 1
            # 50%
            # k = 0.5
            # 5%
            k = default_config['ratio']
            print("The ratio is ", k)
            #2%
            # k = 0.02
        else:
            k = 1

        for num_set in range(len(data_dirs)):
            re_roots = sorted(make_dataset(reso_dir[num_set]))
            data_roots = sorted(make_dataset(data_dirs[num_set]))
            mask_roots = sorted(make_dataset(mask_dirs[num_set]))
            num_label_data = 0
            ratio = 1
            for num_data in range(len(data_roots)):
                if labeled:
                    if train:
                        n_label = str(math.ceil(num_label[num_set] * k + 1))
                        if '00'+n_label==data_roots[num_data][-10:-7] or '0'+n_label==data_roots[num_data][-10:-7]:
                            # print("n_label",n_label)
                            # print(data_roots[num_data][-10:-7])
                            break

                    for num_mask in range(len(mask_roots)):
                        if data_roots[num_data][-10:-4] == mask_roots[num_mask][-10:-4]:
                            temp_re.append(re_roots[num_data])
                            temp_imgs.append(data_roots[num_data])
                            temp_masks.append(mask_roots[num_mask])
                            domain_labels.append(label)
                            num_label_data += 1
                        else:
                            pass
                else:
                    if default_config['ifFast']:
                        if ratio % 10 == 0:
                            temp_re.append(re_roots[num_data])
                            temp_imgs.append(data_roots[num_data])
                            domain_labels.append(label)
                            ratio += 1
                        else:
                            ratio += 1
                    else:
                        temp_re.append(re_roots[num_data])
                        temp_imgs.append(data_roots[num_data])
                        domain_labels.append(label)

        for num_set in range(len(ref_dir)):
            data_roots = sorted(make_dataset(ref_dir[num_set]))
            for num_data in range(len(data_roots)):
                    tem_ref_imgs.append(data_roots[num_data])

        # for Fourier dirs
        if train == True :
            for num_set in range(len(fourier_dir)):
                data_roots = sorted(make_dataset(fourier_dir[num_set]))
                for num_data in range(len(data_roots)):
                    fourier_imgs.append(data_roots[num_data])
        # for labeled Fourier dirs
            # for num_set in range(len(fourier_dir)):
            #     data_roots = sorted(make_dataset(fourier_dir[num_set]))
            #     mask_roots = sorted(make_dataset(fourier_mask[num_set]))
            #     for num_data in range(len(data_roots)):

            #         for num_mask in range(len(mask_roots)):
            #             if data_roots[num_data][-10:-4] == mask_roots[num_mask][-10:-4]:
            #                 labeled_fourier_imgs.append(data_roots[num_data])

        reso = temp_re
        imgs = temp_imgs
        masks = temp_masks
        labels = domain_labels

        print("length of imgs",len(imgs))
        print("length of masks",len(masks))
        # print("length of fourier imgs",len(fourier_imgs))
        # print("length of labeled fourier imgs",len(labeled_fourier_imgs))

        ref_imgs = tem_ref_imgs

        # add something here to index, such that can split the data
        # index = random.sample(range(len(temp_img)), len(temp_mask))

        self.reso = reso
        self.imgs = imgs
        self.masks = masks
        self.labels = labels
        self.new_size = 288
        self.loader = loader
        self.labeled = labeled
        self.train = train
        self.ref = ref_imgs
        self.Fourier_aug = default_config['Fourier_aug']
        self.fourier = fourier_imgs
        self.fourier_mode = default_config['fourier_mode']
        self.alpha = 0.3
        self.aug_p = 0.1

    def __getitem__(self, index):
        if self.train:
            index = random.randrange(len(self.imgs))
        else:
            pass

        path_re = self.reso[index]
        re = self.loader(path_re)
        re = re[0]

        path_img = self.imgs[index]
        img = self.loader(path_img) # numpy, HxW, numpy.Float64

        ref_paths = random.sample(self.ref, 1)
        ref_img = self.loader(ref_paths[0])

        img = match_histograms(img, ref_img)

        label = self.labels[index]

        if label==0:
            one_hot_label = torch.tensor([[1], [0], [0]])
        elif label==1:
            one_hot_label = torch.tensor([[0], [1], [0]])
        elif label==2:
            one_hot_label = torch.tensor([[0], [0], [1]])
        else:
            one_hot_label = torch.tensor([[0], [0], [0]])

        # Intensity cropping:
        p5 = np.percentile(img.flatten(), 0.5)
        p95 = np.percentile(img.flatten(), 99.5)
        img = np.clip(img, p5, p95)

        img -= img.min()
        img /= img.max()
        img = img.astype('float32')

        # for Fourier augmentation
        if self.Fourier_aug == True and self.train == True:
            fourier_paths = random.sample(self.fourier, 1)
            fourier_img = self.loader(fourier_paths[0])

            fourier_p5 = np.percentile(fourier_img.flatten(), 0.5)
            fourier_p95 = np.percentile(fourier_img.flatten(), 99.5)
            fourier_img = np.clip(fourier_img, fourier_p5, fourier_p95)

            fourier_img -= fourier_img.min()
            fourier_img /= fourier_img.max()
            fourier_img = fourier_img.astype('float32')

        crop_size = 300

        # Augmentations:
        # 1. random rotation
        # 2. random scaling 0.8 - 1.2
        # 3. random crop from 280x280
        # 4. random hflip
        # 5. random vflip
        # 6. color jitter
        # 7. Gaussian filtering

        img_tensor = F.to_tensor(np.array(img))
        img_size = img_tensor.size()

        # labeled
        if self.labeled:
            if self.train:
                img = Image.fromarray(img)
                # rotate, random angle between 0 - 90
                angle = random.randint(0, 90)
                img = F.rotate(img, angle, InterpolationMode.BILINEAR)

                path_mask = self.masks[index]
                mask = Image.open(path_mask)  # numpy, HxWx3
                # rotate, random angle between 0 - 90
                mask = F.rotate(mask, angle, InterpolationMode.NEAREST)

                ## Find the region of mask
                norm_mask = F.to_tensor(np.array(mask))
                region = norm_mask[0] + norm_mask[1] + norm_mask[2]
                non_zero_index = torch.nonzero(region == 1, as_tuple=False)
                if region.sum() > 0:
                    len_m = len(non_zero_index[0])
                    x_region = non_zero_index[len_m//2][0]
                    y_region = non_zero_index[len_m//2][1]
                    x_region = int(x_region.item())
                    y_region = int(y_region.item())
                else:
                    x_region = norm_mask.size(-2) // 2
                    y_region = norm_mask.size(-1) // 2

                # resize and center-crop to 280x280
                resize_order = re / 1.1
                resize_size_h = int(img_size[-2] * resize_order)
                resize_size_w = int(img_size[-1] * resize_order)

                left_size = 0
                top_size = 0
                right_size = 0
                bot_size = 0
                if resize_size_h < self.new_size:
                    top_size = (self.new_size - resize_size_h) // 2
                    bot_size = (self.new_size - resize_size_h) - top_size
                if resize_size_w < self.new_size:
                    left_size = (self.new_size - resize_size_w) // 2
                    right_size = (self.new_size - resize_size_w) - left_size

                transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))]
                transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
                transform = transforms.Compose(transform_list)

                img = transform(img)
                if self.Fourier_aug:
                    # aug_p = random.random()
                    # if aug_p > self.aug_p:
                    fourier_img = Image.fromarray(fourier_img)
                    fourier_img = F.rotate(fourier_img, angle, InterpolationMode.BILINEAR)
                    fourier_img = transform(fourier_img)
                    aug_img, _ = fourier_augmentation(img, fourier_img, self.fourier_mode, self.alpha)
                else:
                    fourier_img = torch.tensor([0])
                    aug_tar_img = torch.tensor([0])
                    aug_img = torch.tensor([0])

                ## Define the crop index
                if top_size >= 0:
                    top_crop = 0
                else:
                    if x_region > self.new_size//2:
                        if x_region - self.new_size//2 + self.new_size <= norm_mask.size(-2):
                            top_crop = x_region - self.new_size//2
                        else:
                            top_crop = norm_mask.size(-2) - self.new_size
                    else:
                        top_crop = 0

                if left_size >= 0:
                    left_crop = 0
                else:
                    if y_region > self.new_size//2:
                        if y_region - self.new_size//2 + self.new_size <= norm_mask.size(-1):
                            left_crop = y_region - self.new_size//2
                        else:
                            left_crop = norm_mask.size(-1) - self.new_size
                    else:
                        left_crop = 0

                # random crop to 224x224
                img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)
                # random flip
                hflip_p = random.random()
                img = F.hflip(img) if hflip_p >= 0.5 else img
                vflip_p = random.random()
                img = F.vflip(img) if vflip_p >= 0.5 else img
                img = F.to_tensor(np.array(img))
                # Gaussian bluring:
                transform_list = [transforms.GaussianBlur(5, sigma=(0.25, 1.25))]
                transform = transforms.Compose(transform_list)
                img = transform(img)
                
                if self.Fourier_aug:
                    aug_img = F.crop(aug_img, top_crop, left_crop, self.new_size, self.new_size)
                    aug_img = F.hflip(aug_img) if hflip_p >= 0.5 else aug_img
                    aug_img = F.vflip(aug_img) if vflip_p >= 0.5 else aug_img
                    aug_img = F.to_tensor(np.array(aug_img))
                    aug_img = transform(aug_img)

                # resize and center-crop to 280x280
                transform_mask_list = [transforms.Pad(
                    (left_size, top_size, right_size, bot_size))]
                transform_mask_list = [transforms.Resize((resize_size_h, resize_size_w),
                                                         interpolation=InterpolationMode.NEAREST)] + transform_mask_list
                transform_mask = transforms.Compose(transform_mask_list)

                mask = transform_mask(mask)  # C,H,W

                # random crop to 224x224
                mask = F.crop(mask, top_crop, left_crop, self.new_size, self.new_size)

                # random flip
                mask = F.hflip(mask) if hflip_p >= 0.5 else mask
                mask = F.vflip(mask) if vflip_p >= 0.5 else mask

                mask = F.to_tensor(np.array(mask))

                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0)

            else:
                path_mask = self.masks[index]
                mask = Image.open(path_mask)  # numpy, HxWx3
                # resize and center-crop to 280x280

                ## Find the region of mask
                norm_mask = F.to_tensor(np.array(mask))
                region = norm_mask[0] + norm_mask[1] + norm_mask[2]
                non_zero_index = torch.nonzero(region == 1, as_tuple=False)
                if region.sum() > 0:
                    len_m = len(non_zero_index[0])
                    x_region = non_zero_index[len_m//2][0]
                    y_region = non_zero_index[len_m//2][1]
                    x_region = int(x_region.item())
                    y_region = int(y_region.item())
                else:
                    x_region = norm_mask.size(-2) // 2
                    y_region = norm_mask.size(-1) // 2

                resize_order = re / 1.1
                resize_size_h = int(img_size[-2] * resize_order)
                resize_size_w = int(img_size[-1] * resize_order)

                left_size = 0
                top_size = 0
                right_size = 0
                bot_size = 0
                if resize_size_h < self.new_size:
                    top_size = (self.new_size - resize_size_h) // 2
                    bot_size = (self.new_size - resize_size_h) - top_size
                if resize_size_w < self.new_size:
                    left_size = (self.new_size - resize_size_w) // 2
                    right_size = (self.new_size - resize_size_w) - left_size

                # transform_list = [transforms.CenterCrop((crop_size, crop_size))]
                transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))]
                transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
                transform_list = [transforms.ToPILImage()] + transform_list
                transform = transforms.Compose(transform_list)
                img = transform(img)
                img = F.to_tensor(np.array(img))

                ## Define the crop index
                if top_size >= 0:
                    top_crop = 0
                else:
                    if x_region > self.new_size//2:
                        if x_region - self.new_size//2 + self.new_size <= norm_mask.size(-2):
                            top_crop = x_region - self.new_size//2
                        else:
                            top_crop = norm_mask.size(-2) - self.new_size
                    else:
                        top_crop = 0

                if left_size >= 0:
                    left_crop = 0
                else:
                    if y_region > self.new_size//2:
                        if y_region - self.new_size//2 + self.new_size <= norm_mask.size(-1):
                            left_crop = y_region - self.new_size//2
                        else:
                            left_crop = norm_mask.size(-1) - self.new_size
                    else:
                        left_crop = 0

                # random crop to 224x224
                img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)

                # resize and center-crop to 280x280
                # transform_mask_list = [transforms.CenterCrop((crop_size, crop_size))]
                transform_mask_list = [transforms.Pad(
                    (left_size, top_size, right_size, bot_size))]
                transform_mask_list = [transforms.Resize((resize_size_h, resize_size_w),
                                                         interpolation=InterpolationMode.NEAREST)] + transform_mask_list
                transform_mask = transforms.Compose(transform_mask_list)

                mask = transform_mask(mask)  # C,H,W
                mask = F.crop(mask, top_crop, left_crop, self.new_size, self.new_size)
                mask = F.to_tensor(np.array(mask))

                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0)

                fourier_img = torch.tensor([0])
                aug_img = torch.tensor([0])
                aug_tar_img = torch.tensor([0])

                # mask_0 = mask[0,:,:]
                # mask_1 = mask[1,:,:]
                # mask_2 = mask[2,:,:]
                # mask_3 = mask[3,:,:]
                # mask = mask_0*1 + mask_1*2 + mask_2*3 + mask_3*0

        # unlabel
        else:
            mask = torch.tensor([0])
            img = Image.fromarray(img)
            # rotate, random angle between 0 - 90
            angle = random.randint(0, 90)
            img = F.rotate(img, angle, InterpolationMode.BILINEAR)

            # resize and center-crop to 280x280
            resize_order = re / 1.1
            resize_size_h = int(img_size[-2] * resize_order)
            resize_size_w = int(img_size[-1] * resize_order)

            left_size = 0
            top_size = 0
            right_size = 0
            bot_size = 0
            if resize_size_h < crop_size:
                top_size = (crop_size - resize_size_h) // 2
                bot_size = (crop_size - resize_size_h) - top_size
            if resize_size_w < crop_size:
                left_size = (crop_size - resize_size_w) // 2
                right_size = (crop_size - resize_size_w) - left_size

            transform_list = [transforms.CenterCrop((crop_size, crop_size))]
            transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))] + transform_list
            transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
            transform = transforms.Compose(transform_list)

            img = transform(img)

            if self.Fourier_aug:
                # aug_p = random.random()
                # if aug_p > self.aug_p:
                fourier_img = Image.fromarray(fourier_img)
                fourier_img = F.rotate(fourier_img, angle, InterpolationMode.BILINEAR)
                fourier_img = transform(fourier_img)
                aug_img, aug_tar_img = fourier_augmentation(img, fourier_img, self.fourier_mode, self.alpha)
            else:
                aug_img = torch.tensor([0])
                aug_tar_img = torch.tensor([0])
                fourier_img = torch.tensor([0])

            # random crop to 224x224
            top_crop = random.randint(0, crop_size - self.new_size)
            left_crop = random.randint(0, crop_size - self.new_size)
            img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)
            # random flip
            hflip_p = random.random()
            vflip_p = random.random()
            img = F.hflip(img) if hflip_p >= 0.5 else img
            img = F.vflip(img) if vflip_p >= 0.5 else img

            img = F.to_tensor(np.array(img))
            # Gaussian bluring:
            transform_list = [transforms.GaussianBlur(5, sigma=(0.25, 1.25))]
            transform = transforms.Compose(transform_list)
            img = transform(img)

            if self.Fourier_aug:
                aug_img = F.crop(aug_img, top_crop, left_crop, self.new_size, self.new_size)
                aug_img = F.hflip(aug_img) if hflip_p >= 0.5 else aug_img
                aug_img = F.vflip(aug_img) if vflip_p >= 0.5 else aug_img
                aug_img = F.to_tensor(np.array(aug_img))
                aug_img = transform(aug_img)

        ouput_dict = dict(
            img = img,
            aug_img = aug_img,
            mask = mask,
            path_img = path_img,
            domain_label = one_hot_label.squeeze()
        )
        return ouput_dict # pytorch: N,C,H,W

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    test_vendor = 'A'

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
    domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
    test_dataset  = get_meta_split_data_loaders(test_vendor=test_vendor, image_size=224)

    label_dataset = ConcatDataset([domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])
    label_loader = DataLoader(dataset=label_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    unlabel_dataset = ConcatDataset([domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset])
    unlabel_loader = DataLoader(dataset=unlabel_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    dataiter = iter(unlabel_loader)
    output = dataiter.next()
    img = output['img']
    aug_img = output['aug_img']
    mask = output['mask']
    # domain_label = output['domain_label']
    print("img shape",img.shape)
    print("aug_img shape",aug_img.shape)
    print("mask shape",mask.shape)
    # mask = mask[:, 0:3, :, :]
    # img = im_convert(img, False)
    # aug_img = im_convert(aug_img, False)
    # save_image(img, './fpic/label_'+str(default_config['fourier_mode'])+'_img.png')
    # save_image(aug_img, './fpic/label_'+str(default_config['fourier_mode'])+'_aug_img.png')

    # mask = im_convert(mask, True)
    # save_image(mask, './fpic/label_'+str(default_config['fourier_mode'])+'_mask.png')
