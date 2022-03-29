import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import math
import statistics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from network.network import my_net
from utils.utils import get_device, check_accuracy, dice_loss, im_convert, label_to_onehot
from mms_dataloader import get_meta_split_data_loaders
from config import default_config
from utils.data_utils import save_image
from utils.dice_loss import dice_coeff
from draw_dataloader import OneImageFolder

device = 'cuda'

def pre_data(batch_size, num_workers, test_vendor):
    test_vendor = test_vendor

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(
            test_vendor=test_vendor, image_size=224)

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    # unlabel_dataset = ConcatDataset(
    #     [domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset])
    unlabel_dataset = domain_2_unlabeled_dataset

    print("before length of label_dataset", len(label_dataset))

    # new_labeldata_num = len(unlabel_dataset) // len(label_dataset) + 1
    # new_label_dataset = label_dataset
    # for i in range(new_labeldata_num):
    #     new_label_dataset = ConcatDataset([new_label_dataset, label_dataset])
    # label_dataset = new_label_dataset

    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False)

    unlabel_loader = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True, drop_last=True, pin_memory=False)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=True, drop_last=True, pin_memory=False)

    print("after length of label_dataset", len(label_dataset))
    print("length of unlabel_dataset", len(unlabel_dataset))
    print("length of val_dataset", len(val_dataset))
    print("length of test_dataset", len(test_dataset))

    return label_loader, unlabel_loader, test_loader, val_loader, len(label_dataset), len(unlabel_dataset)

def inference(model_path, test_loader):

    model_l = torch.load(model_path)
    model_l = model_l.to(device)
    model_l.eval()

    test_loss = []
    loss = 0
    dice_loss_lv_l = 0
    dice_loss_myo_l = 0
    dice_loss_rv_l = 0
    dice_loss_bg_l = 0

    for batch in tqdm(test_loader):
        imgs, mask, _ = batch
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits = model_l(imgs)

        sof_l = F.softmax(logits, dim=1)
        dice_loss_lv_l = dice_loss(sof_l[:, 0, :, :], mask[:, 0, :, :])
        dice_loss_myo_l = dice_loss(sof_l[:, 1, :, :], mask[:, 1, :, :])
        dice_loss_rv_l = dice_loss(sof_l[:, 2, :, :], mask[:, 2, :, :])
        dice_loss_bg_l = dice_loss(sof_l[:, 3, :, :], mask[:, 3, :, :])
        loss = dice_loss_lv_l + dice_loss_myo_l + dice_loss_rv_l + dice_loss_bg_l

        test_loss.append(loss.item())

    test_dice, test_dice_lv, test_dice_myo, test_dice_rv  = check_accuracy(test_loader, model_l)

    test_loss = sum(test_loss) / len(test_loss)
    print(
        f"[ Test | test_loss = {test_loss:.5f} test_dice = {test_dice:.5f}")

def draw_img(model_path_l, model_path_r, test_loader, domain):
    model_l = torch.load(model_path_l, map_location=device)
    model_r = torch.load(model_path_r, map_location=device)
    model_l = model_l.to(device)
    model_r = model_r.to(device)
    model_l.eval()
    model_r.eval()

    dataiter = iter(test_loader)
    minibatch = dataiter.next()
    imgs = minibatch['img']
    aug_img = minibatch['aug_img']
    mask = minibatch['mask']
    img_path = minibatch['path_img']
    imgs = imgs.to(device)
    aug_img = aug_img.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits_l, _ = model_l(imgs)
        aug_logits_l, _ = model_l(aug_img)
        logits_r, _ = model_r(imgs)
        aug_logits_r, _ = model_r(aug_img)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)
        aug_sof_l = F.softmax(aug_logits_l, dim=1)
        aug_sof_r = F.softmax(aug_logits_r, dim=1)

    ensemble_l = (sof_l + aug_sof_l) / 2
    ensemble_r = (sof_r + aug_sof_r) / 2

    pred = sof_l
    aug_pred = aug_sof_l

    pred_l = (ensemble_l > 0.5).float()
    pred_r = (ensemble_r > 0.5).float()
    pred_l = pred_l[:,0:3,:,:]
    
    sof_l = sof_l[:,0:3,:,:]
    aug_sof_l = aug_sof_l[:,0:3,:,:]
    sof_r = sof_r[:,0:3,:,:]
    aug_sof_r = aug_sof_r[:,0:3,:,:]
    ensemble_l = ensemble_l[:,0:3,:,:]
    ensemble_r = ensemble_r[:,0:3,:,:]
    
    # sof_l = im_convert(sof_l, True)
    # save_image(sof_l,'./fpic/sof_l'+str(domain)+'.png')

    # sof_r = im_convert(sof_r, True)
    # save_image(sof_r,'./fpic/sof_r'+str(domain)+'.png')

    # aug_sof_l = im_convert(aug_sof_l, True)
    # save_image(aug_sof_l,'./fpic/aug_sof_l'+str(domain)+'.png')

    # aug_sof_r = im_convert(aug_sof_r, True)
    # save_image(aug_sof_r,'./fpic/aug_sof_r'+str(domain)+'.png')

    # ensemble_l = im_convert(ensemble_l, True)
    # save_image(ensemble_l,'./fpic/ensemble_l'+str(domain)+'.png')

    # ensemble_r = im_convert(ensemble_r, True)
    # save_image(ensemble_r,'./fpic/ensemble_r'+str(domain)+'.png')

    # pred_l = im_convert(pred_l, True)
    # save_image(pred_l,'./usepic/pred_l'+str(domain)+'.png')

    # pred_r = im_convert(pred_r, True)
    # save_image(pred_r,'./fpic/pred_r'+str(domain)+'.png')

    pred = (pred > 0.5).float()
    aug_pred = (aug_pred > 0.5).float()
    # dice score
    tot = dice_coeff(pred[:, 0:3, :, :], mask[:, 0:3, :, :], device).item()
    aug_tot = dice_coeff(aug_pred[:, 0:3, :, :], mask[:, 0:3, :, :], device).item()

    image = imgs
    # torch.set_printoptions(threshold=np.inf)
    # with open('./test.txt', 'wt') as f:
    #     print(onehot_predmax==mask, file=f)
    pred = pred[:,0:3,:,:]
    aug_pred = aug_pred[:,0:3,:,:]
    real_mask = mask[:,0:3,:,:]

    print(img_path[0])
    # image slice
    # print(img_path[0][-7:-4])
    # patient num
    # print(img_path[0][-10: -7])
    print("dice score: ", tot)
    print("aug dice score: ", aug_tot)
    real_mask = im_convert(real_mask, True)
    image = im_convert(image, False)
    aug_img = im_convert(aug_img, False)
    pred = im_convert(pred, True)
    aug_pred = im_convert(aug_pred, True)
    save_image(real_mask,'./usepic/gt'+str(domain)+'.png')
    save_image(image,'./usepic/image'+str(domain)+'.png')
    save_image(aug_img,'./usepic/aug_image'+str(domain)+'.png')
    save_image(pred,'./usepic/pred'+str(domain)+'.png')
    save_image(aug_pred,'./usepic/aug_pred'+str(domain)+'.png')

def save_once(image, pred, mask, flag, image_slice):
    pred = pred[:,0:3,:,:]
    real_mask = mask[:,0:3,:,:]
    mask = im_convert(real_mask, True)
    image = im_convert(image, False)
    pred = im_convert(pred, True)
    
    save_image(mask,'./pic/'+str(flag)+'/real_mask'+str(image_slice)+'.png')
    save_image(image,'./pic/'+str(flag)+'/image'+str(image_slice)+'.png')
    save_image(pred,'./pic/'+str(flag)+'/pred'+str(image_slice)+'.png')

def draw_many_img(model_path_l, model_path_r, test_loader):
    model_l = torch.load(model_path_l, map_location=device)
    model_r = torch.load(model_path_r, map_location=device)
    model_l = model_l.to(device)
    model_r = model_r.to(device)
    model_l.eval()
    model_r.eval()

    flag = '047'
    tot = 0
    tot_sub = []
    for minibatch in tqdm(test_loader):
        imgs = minibatch['img']
        mask = minibatch['mask']
        path_img = minibatch['path_img']
        imgs = imgs.to(device)
        mask = mask.to(device)
        if path_img[0][-10: -7] == flag:
            image_slice = path_img[0][-7:-4]
            with torch.no_grad():
                logits_l = model_l(imgs)
                logits_r = model_r(imgs)

            sof_l = F.softmax(logits_l, dim=1)
            sof_r = F.softmax(logits_r, dim=1)

            pred = (sof_l + sof_r) / 2
            pred = (pred > 0.5).float()

            save_once(imgs, pred, mask, flag, image_slice)

            # dice score
            tot = dice_coeff(pred[:, 0:3, :, :], mask[:, 0:3, :, :], device).item()

            tot_sub.append(tot)
        else:
            pass

    print('dice is ', sum(tot_sub)/len(tot_sub))

def inference_dual(model_path_l, model_path_r, test_loader):

    model_l = torch.load(model_path_l, map_location=device)
    model_l = model_l.to(device)
    model_l.eval()

    model_r = torch.load(model_path_r, map_location=device)
    model_r = model_r.to(device)
    model_r.eval()

    tot = []
    tot_sub = []
    flag = '000'
    record_flag = {}

    for minibatch in tqdm(test_loader):
        imgs = minibatch['img']
        mask = minibatch['mask']
        path_img = minibatch['path_img']
        imgs = imgs.to(device)
        mask = mask.to(device)
        # print(flag)
        # print(path_img[0][-10: -7])
        if path_img[0][-10: -7] != flag:
            score = sum(tot_sub)/len(tot_sub)
            tot.append(score)
            tot_sub = []

            if score <= 0.7:
                record_flag[flag] = score
            flag = path_img[0][-10: -7]

        with torch.no_grad():
            logits_l, _ = model_l(imgs)
            logits_r, _ = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)
        pred = (sof_l + sof_r) / 2
        pred = (pred > 0.5).float()
        dice = dice_coeff(pred[:, 0:3, :, :], mask[:, 0:3, :, :], device).item()
        tot_sub.append(dice)
    tot.append(sum(tot_sub)/len(tot_sub))

    for i in range(len(tot)):
        tot[i] = tot[i] * 100

    print(tot)
    print(len(tot))
    print(sum(tot)/len(tot))
    print(statistics.stdev(tot))
    print(record_flag)

def main():
    batch_size = 1
    num_workers = 4
    test_vendor = 'A'

    model_path_l = './tmodel/l_deeplab_2%_'+str(test_vendor)+'.pt'
    model_path_r = './tmodel/r_deeplab_2%_'+str(test_vendor)+'.pt'

    label_loader, unlabel_loader, test_loader, val_loader, num_label_imgs, num_unsup_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)
    # id = '047258'
    # # id = '002126'
    # img_path = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorC/'+ id +'.npz'
    # mask_path = '/home/listu/code/semi_medical/mnms_split_2D/mask/Labeled/vendorC/'+ id +'.png'
    # re_path = '/home/listu/code/semi_medical/mnms_split_2D_re/Labeled/vendorC/'+ id +'.npz'
    # fourier_path = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorB/center2/000005.npz'
    # one_image_data = OneImageFolder(img_path, mask_path, re_path, fourier_path)
    # one_image_loader = DataLoader(dataset=one_image_data, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    # draw_img(model_path_l, model_path_r, one_image_loader, test_vendor)
    # draw_many_img(model_path_l, model_path_r, test_loader)
    inference_dual(model_path_l, model_path_r, test_loader)

if __name__ == '__main__':
    main()
