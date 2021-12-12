import nibabel as nib
import argparse
import os
import numpy as np
from PIL import Image
import configparser

import re
import SimpleITK as stik
from collections import OrderedDict
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import cv2
import torch

def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def save_np(img, path, name):
    np.savez_compressed(os.path.join(path, name), img)

def save_mask_np(mask1, mask2, path, name):
    h = mask1.shape[0]
    w = mask1.shape[1]
    mask = np.concatenate((mask1.reshape(h,w,1), mask2.reshape(h,w,1)), axis=2)
    np.savez_compressed(os.path.join(path, name), mask)

path_train = '/home/listu/code/semi_medical/SCGMdata/train_set/'
past_test = '/home/listu/code/semi_medical/SCGMdata/test/'

######################################################################################################

# Save data dirs
LabeledVendorA_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Labeled/vendorA/'
LabeledVendorA_mask_dir = '/home/listu/code/semi_medical/scgm_split_2D/mask/Labeled/vendorA/'

LabeledVendorB_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Labeled/vendorB/'
LabeledVendorB_mask_dir = '/home/listu/code/semi_medical/scgm_split_2D/mask/Labeled/vendorB/'

LabeledVendorC_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Labeled/vendorC/'
LabeledVendorC_mask_dir = '/home/listu/code/semi_medical/scgm_split_2D/mask/Labeled/vendorC/'

LabeledVendorD_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Labeled/vendorD/'
LabeledVendorD_mask_dir = '/home/listu/code/semi_medical/scgm_split_2D/mask/Labeled/vendorD/'

UnlabeledVendorA_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Unlabeled/vendorA/'
UnlabeledVendorB_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Unlabeled/vendorB/'
UnlabeledVendorC_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Unlabeled/vendorC/'
UnlabeledVendorD_data_dir = '/home/listu/code/semi_medical/scgm_split_2D/data/Unlabeled/vendorD/'

labeled_data_dir = [LabeledVendorA_data_dir, LabeledVendorB_data_dir, LabeledVendorC_data_dir, LabeledVendorD_data_dir]
labeled_mask_dir = [LabeledVendorA_mask_dir, LabeledVendorB_mask_dir, LabeledVendorC_mask_dir, LabeledVendorD_mask_dir]
un_labeled_data_dir = [UnlabeledVendorA_data_dir, UnlabeledVendorB_data_dir, UnlabeledVendorC_data_dir, UnlabeledVendorD_data_dir]

safe_mkdir(LabeledVendorA_data_dir)
safe_mkdir(LabeledVendorA_mask_dir)
safe_mkdir(LabeledVendorB_data_dir)
safe_mkdir(LabeledVendorB_mask_dir)
safe_mkdir(LabeledVendorC_data_dir)
safe_mkdir(LabeledVendorC_mask_dir)
safe_mkdir(LabeledVendorD_data_dir)
safe_mkdir(LabeledVendorD_mask_dir)
safe_mkdir(UnlabeledVendorA_data_dir)
safe_mkdir(UnlabeledVendorB_data_dir)
safe_mkdir(UnlabeledVendorC_data_dir)
safe_mkdir(UnlabeledVendorD_data_dir)


def read_numpy(file_name):
    reader = stik.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file_name)
    data = reader.Execute()
    return stik.GetArrayFromImage(data)

def read_dataset_into_memory(data_list, labeled=True):
    for val in data_list.values():
        val['input'] = read_numpy(val['input'])
        if labeled:
            for idx, gt in enumerate(val['gt']):
                val['gt'][idx] = read_numpy(gt)
        else:
            pass
    return data_list

def get_index_map(data_list, labeled=True):
    map_list = []
    num_sbj = 0
    for data in data_list.values():
        slice_num = data['input'].shape[0]
        print(data['input'].shape)
        for i in range(slice_num):
            if labeled:
                map_list.append([data['input'][i], np.stack([data['gt'][idx][i] for idx in range(4)], axis=0), num_sbj])
            else:
                map_list.append([data['input'][i], num_sbj])
        num_sbj += 1
    return map_list

# data_list: data_list['input'] path, data_list['gt'] path
def Save_vendor_data(data_list, labeled=True, data_dir='', mask_dir=''):
    data_list = read_dataset_into_memory(data_list, labeled)
    # data_list: data_list['input'] images, data_list['gt'] ground-truth labels
    map_list = get_index_map(data_list,labeled)
    # map_list: map_list[i]: [images, gt]
    if labeled:
        num_sbj = 0
        num_slice = 0
        for idx in range(len(map_list)):
            img, gt_list, flag = map_list[idx]
            if flag != num_sbj:
                num_sbj = flag
                num_slice = 0
            img = img / (img.max() if img.max() > 0 else 1)
            gt_list = torch.tensor(gt_list, dtype=torch.uint8)
            spinal_cord_mask = (torch.mean(((gt_list > 0)).float(), dim=0) > 0.5).float()
            spinal_cord_mask = spinal_cord_mask.numpy()
            gm_mask = (torch.mean((gt_list == 1).float(), dim=0) > 0.5).float()
            gm_mask = gm_mask.numpy()
            save_np(img, data_dir, '%03d%03d' % (num_sbj, num_slice))
            save_mask_np(spinal_cord_mask, gm_mask, mask_dir, '%03d%03d' % (num_sbj, num_slice))
            num_slice += 1
    else:
        num_sbj = 0
        num_slice = 0
        for idx in range(len(map_list)):
            img, flag = map_list[idx]
            if flag != num_sbj:
                num_sbj = flag
                num_slice = 0
            img = img / (img.max() if img.max() > 0 else 1)
            save_np(img, data_dir, '%03d%03d' % (num_sbj, num_slice))
            num_slice += 1

resolution = {
    'site1': [5, 0.5, 0.5],
    'site2': [5, 0.5, 0.5],
    'site3': [2.5, 0.5, 0.5],
    'site4': [5, 0.29, 0.29],
}

labeled_imageFileList = [os.path.join(path_train, f) for f in os.listdir(path_train) if 'site' in f and '.txt' not in f]
labeled_data_dict = {'site1': OrderedDict(), 'site2': OrderedDict(), 'site3': OrderedDict(), 'site4': OrderedDict()}
for file in sorted(labeled_imageFileList):
    res = re.search('site(\d)-sc(\d*)-(image|mask)', file).groups()
    if res[1] not in labeled_data_dict['site' + res[0]].keys():
        labeled_data_dict['site' + res[0]][res[1]] = {'input': None, 'gt': []}
    if res[2] == 'image':
        labeled_data_dict['site' + res[0]][res[1]]['input'] = file
    if res[2] == 'mask':
        labeled_data_dict['site' + res[0]][res[1]]['gt'].append(file)
i = 0
for domain, data_list in labeled_data_dict.items():
    print(domain)
    Save_vendor_data(data_list, labeled=True, data_dir=labeled_data_dir[i], mask_dir=labeled_mask_dir[i])
    i += 1

unlabeled_imageFileList = [os.path.join(past_test, f) for f in os.listdir(past_test) if 'site' in f and '.txt' not in f]
unlabeled_data_dict = {'site1': OrderedDict(), 'site2': OrderedDict(), 'site3': OrderedDict(), 'site4': OrderedDict()}
for file in sorted(unlabeled_imageFileList):
    res = re.search('site(\d)-sc(\d*)-(image|mask)', file).groups()
    if res[1] not in unlabeled_data_dict['site' + res[0]].keys():
        unlabeled_data_dict['site' + res[0]][res[1]] = {'input': None, 'gt': []}
    if res[2] == 'image':
        unlabeled_data_dict['site' + res[0]][res[1]]['input'] = file
    if res[2] == 'mask':
        unlabeled_data_dict['site' + res[0]][res[1]]['gt'].append(file)
i = 0
for domain, data_list in unlabeled_data_dict.items():
    print(domain)
    Save_vendor_data(data_list, labeled=False, data_dir=un_labeled_data_dir[i], mask_dir=None)
    i += 1


# two items: 1. domain name, 2. data['input'], data[gt]