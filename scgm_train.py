import os
import sys
import math
import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from network.scgm_network import my_net
from utils.utils import get_device, check_accuracy, check_accuracy_dual, label_to_onehot
from scgm_dataloader import get_meta_split_data_loaders
from config import default_config
from utils.dice_loss import dice_coeff
# from losses import SupConLoss
import utils.mask_gen as mask_gen
from utils.custom_collate import SegCollate

gpus = default_config['gpus']
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

wandb.init(project='SCGM_seg', entity='nekokiku',
           config=default_config, name=default_config['train_name'])
config = wandb.config

device = get_device()

def pre_data(batch_size, num_workers, test_vendor):
    test_vendor = test_vendor

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(
            test_vendor=test_vendor)

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    unlabel_dataset = ConcatDataset(
        [domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset])
    # unlabel_dataset = domain_2_unlabeled_dataset

    print("before length of label_dataset", len(label_dataset))

    new_labeldata_num = len(unlabel_dataset) // len(label_dataset) + 1
    new_label_dataset = label_dataset
    for i in range(new_labeldata_num):
        new_label_dataset = ConcatDataset([new_label_dataset, label_dataset])
    label_dataset = new_label_dataset

    # For CutMix
    mask_generator = mask_gen.BoxMaskGenerator(prop_range=config['cutmix_mask_prop_range'], n_boxes=config['cutmix_boxmask_n_boxes'],
                                               random_aspect_ratio=config['cutmix_boxmask_fixed_aspect_ratio'],
                                               prop_by_area=config['cutmix_boxmask_by_size'], within_bounds=config[
                                                   'cutmix_boxmask_outside_bounds'],
                                               invert=config['cutmix_boxmask_no_invert'])

    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    unlabel_loader_0 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=mask_collate_fn)

    unlabel_loader_1 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    print("after length of label_dataset", len(label_dataset))
    print("length of unlabel_dataset", len(unlabel_dataset))
    print("length of val_dataset", len(val_dataset))
    print("length of test_dataset", len(test_dataset))

    return label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, len(label_dataset), len(unlabel_dataset)

# Dice loss

def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.1  # 1e-12

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    #A_sum = torch.sum(tflat * iflat)
    #B_sum = torch.sum(tflat * tflat)
    loss = ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss

def ini_model(restore=False, restore_from=None):
    if restore:
        model_path_l = './tmodel/' + 'l_' + str(restore_from)
        model_path_r = './tmodel/' + 'r_' + str(restore_from)
        model_l = torch.load(model_path_l)
        model_r = torch.load(model_path_r)
        print("restore from ", model_path_l)
        print("restore from ", model_path_r)
    else:
        # two models with different init
        model_l = my_net(modelname='mydeeplabV3P')
        model_r = my_net(modelname='mydeeplabV3P')

    model_l = model_l.to(device)
    model_l.device = device

    model_r = model_r.to(device)
    model_r.device = device

    model_r = nn.DataParallel(model_r, device_ids=gpus, output_device=gpus[0])
    model_l = nn.DataParallel(model_l, device_ids=gpus, output_device=gpus[0])
    return model_l, model_r

def ini_optimizer(model_l, model_r, learning_rate, weight_decay):
    # Initialize two optimizer.
    optimizer_l = torch.optim.AdamW(
        model_l.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_r = torch.optim.AdamW(
        model_r.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer_l, optimizer_r

def cal_variance(pred, aug_pred):
    kl_distance = nn.KLDivLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)
    variance = torch.sum(kl_distance(
        log_sm(pred), sm(aug_pred)), dim=1)
    exp_variance = torch.exp(-variance)

    return variance, exp_variance

def train_one_epoch(model_l, model_r, niters_per_epoch, label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1, optimizer_r, optimizer_l, cross_criterion, epoch):
    # loss data
    total_loss = []
    total_loss_l = []
    total_loss_r = []
    total_cps_loss = []
    total_con_loss = []
    # tqdm
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(niters_per_epoch),
                file=sys.stdout, bar_format=bar_format)
    kl_distance = nn.KLDivLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)

    for idx in pbar:
        minibatch = label_dataloader.next()
        unsup_minibatch_0 = unlabel_dataloader_0.next()
        unsup_minibatch_1 = unlabel_dataloader_1.next()

        imgs = minibatch['img']
        aug_imgs = minibatch['aug_img']
        mask = minibatch['mask']

        unsup_imgs_0 = unsup_minibatch_0['img']
        unsup_imgs_1 = unsup_minibatch_1['img']

        aug_unsup_imgs_0 = unsup_minibatch_0['aug_img']
        aug_unsup_imgs_1 = unsup_minibatch_1['aug_img']
        mask_params = unsup_minibatch_0['mask_params']

        imgs = imgs.to(device)
        aug_imgs = aug_imgs.to(device)
        mask_type = torch.long
        mask = mask.to(device=device, dtype=mask_type)

        unsup_imgs_0 = unsup_imgs_0.to(device)
        unsup_imgs_1 = unsup_imgs_1.to(device)
        aug_unsup_imgs_0 = aug_unsup_imgs_0.to(device)
        aug_unsup_imgs_1 = aug_unsup_imgs_1.to(device)
        mask_params = mask_params.to(device)

        batch_mix_masks = mask_params
        # unlabeled mixed images
        unsup_imgs_mixed = unsup_imgs_0 * \
            (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
        # unlabeled r mixed images
        aug_unsup_imgs_mixed = aug_unsup_imgs_0 * \
            (1 - batch_mix_masks) + aug_unsup_imgs_1 * batch_mix_masks

        # add uncertainty
        with torch.no_grad():
            # Estimate the pseudo-label with model_l using original data
            logits_u0_tea_1, _ = model_l(unsup_imgs_0)
            logits_u1_tea_1, _ = model_l(unsup_imgs_1)
            logits_u0_tea_1 = logits_u0_tea_1.detach()
            logits_u1_tea_1 = logits_u1_tea_1.detach()
            aug_logits_u0_tea_1, _ = model_l(aug_unsup_imgs_0)
            aug_logits_u1_tea_1, _ = model_l(aug_unsup_imgs_1)
            aug_logits_u0_tea_1 = aug_logits_u0_tea_1.detach()
            aug_logits_u1_tea_1 = aug_logits_u1_tea_1.detach()
            # Estimate the pseudo-label with model_r using augmentated data
            logits_u0_tea_2, _ = model_r(unsup_imgs_0)
            logits_u1_tea_2, _ = model_r(unsup_imgs_1)
            logits_u0_tea_2 = logits_u0_tea_2.detach()
            logits_u1_tea_2 = logits_u1_tea_2.detach()
            aug_logits_u0_tea_2, _ = model_r(aug_unsup_imgs_0)
            aug_logits_u1_tea_2, _ = model_r(aug_unsup_imgs_1)
            aug_logits_u0_tea_2 = aug_logits_u0_tea_2.detach()
            aug_logits_u1_tea_2 = aug_logits_u1_tea_2.detach()

        logits_u0_tea_1 = (logits_u0_tea_1 + aug_logits_u0_tea_1) / 2
        logits_u1_tea_1 = (logits_u1_tea_1 + aug_logits_u1_tea_1) / 2
        logits_u0_tea_2 = (logits_u0_tea_2 + aug_logits_u0_tea_2) / 2
        logits_u1_tea_2 = (logits_u1_tea_2 + aug_logits_u1_tea_2) / 2

        # Mix teacher predictions using same mask
        # It makes no difference whether we do this with logits or probabilities as
        # the mask pixels are either 1 or 0
        logits_cons_tea_1 = logits_u0_tea_1 * \
            (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
        _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
        ps_label_1 = ps_label_1.long()

        logits_cons_tea_2 = logits_u0_tea_2 * \
            (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
        _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
        ps_label_2 = ps_label_2.long()

        # Get student_l prediction for mixed image
        logits_cons_stu_1, _ = model_l(unsup_imgs_mixed)
        aug_logits_cons_stu_1,_ = model_l(aug_unsup_imgs_mixed)
        # Get student_r prediction for mixed image
        logits_cons_stu_2, _ = model_r(unsup_imgs_mixed)
        aug_logits_cons_stu_2, _ = model_r(aug_unsup_imgs_mixed)

        # add uncertainty
        var_l, exp_var_l = cal_variance(logits_cons_stu_1, aug_logits_cons_stu_1)
        var_r, exp_var_r = cal_variance(logits_cons_stu_2, aug_logits_cons_stu_2)

        # cps loss
        cps_loss = torch.mean(exp_var_r * cross_criterion(logits_cons_stu_1, ps_label_2)) + torch.mean(
                        exp_var_l * cross_criterion(logits_cons_stu_2, ps_label_1)) + torch.mean(var_l) + torch.mean(var_r)

        # cps weight
        cps_loss = cps_loss * config['CPS_weight']

        # supervised loss on both models
        pre_sup_l, feature_l = model_l(imgs)
        pre_sup_r, feature_r = model_r(imgs)
        # dice loss
        sof_l = F.softmax(pre_sup_l, dim=1)
        sof_r = F.softmax(pre_sup_r, dim=1)

        loss_r = dice_loss(sof_r[:, 0, :, :], mask[:, 0, :, :])
        loss_l = dice_loss(sof_l[:, 0, :, :], mask[:, 0, :, :])

        # contrastive loss SupConLoss
        # features means different views
        # feature_l = feature_l.unsqueeze(1)
        # feature_r = feature_r.unsqueeze(1)
        # features = torch.cat((feature_l, feature_r),dim=1)

        # supconloss = SupConLoss()
        # con_loss = supconloss(features)
        con_loss = 1
        optimizer_l.zero_grad()
        optimizer_r.zero_grad()

        if epoch <= 3:
            loss = loss_l + loss_r
        else:
            loss = loss_l + loss_r + cps_loss
        # loss = loss_l + loss_r + cps_loss

        loss.backward()
        optimizer_l.step()
        optimizer_r.step()

        total_loss.append(loss.item())
        total_loss_l.append(loss_l.item())
        total_loss_r.append(loss_r.item())
        total_cps_loss.append(cps_loss.item())
        total_con_loss.append(con_loss)

    total_loss = sum(total_loss) / len(total_loss)
    total_loss_l = sum(total_loss_l) / len(total_loss_l)
    total_loss_r = sum(total_loss_r) / len(total_loss_r)
    total_cps_loss = sum(total_cps_loss) / len(total_cps_loss)
    total_con_loss = sum(total_con_loss) / len(total_con_loss)

    return model_l, model_r, total_loss, total_loss_l, total_loss_r, total_cps_loss, total_con_loss

# use the function to calculate the valid loss or test loss
def test_dual(model_l, model_r, loader):
    model_l.eval()
    model_r.eval()

    loss = []
    t_loss = 0
    r_loss = 0

    tot = 0

    for batch in tqdm(loader):
        imgs = batch['img']
        mask = batch['mask']
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits_l,_ = model_l(imgs)
            logits_r,_ = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)

        pred = (sof_l + sof_r) / 2
        pred = (pred > 0.5).float()

        # loss
        t_loss = dice_loss(pred[:, 0, :, :], mask[:, 0, :, :])
        loss.append(t_loss.item())

        # dice score
        tot += dice_coeff(pred[:, 0, :, :],
                          mask[:, 0, :, :], device).item()

    r_loss = sum(loss) / len(loss)

    dice = tot/len(loader)

    model_l.train()
    model_r.train()
    return r_loss, dice

def train(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, learning_rate, weight_decay, num_epoch, model_path, niters_per_epoch):

    # Initialize model
    model_l, model_r = ini_model(config['restore'], config['restore_from'])

    # loss
    cross_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    # Initialize optimizer.
    optimizer_l, optimizer_r = ini_optimizer(
        model_l, model_r, learning_rate, weight_decay)

    best_dice = 0

    for epoch in range(num_epoch):
        # ---------- Training ----------
        model_l.train()
        model_r.train()

        label_dataloader = iter(label_loader)
        unlabel_dataloader_0 = iter(unlabel_loader_0)
        unlabel_dataloader_1 = iter(unlabel_loader_1)

        # normal images
        model_l, model_r, total_loss, total_loss_l, total_loss_r, total_cps_loss, total_con_loss = train_one_epoch(
            model_l, model_r, niters_per_epoch, label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1, optimizer_r, optimizer_l, cross_criterion, epoch)

        # Print the information.
        print(
            f"[ Normal image Train | {epoch + 1:03d}/{num_epoch:03d} ] total_loss = {total_loss:.5f} total_loss_l = {total_loss_l:.5f} total_loss_r = {total_loss_r:.5f} total_cps_loss = {total_cps_loss:.5f}")

        # ---------- Validation ----------
        val_loss, val_dice = test_dual(
            model_l, model_r, val_loader)
        print(
            f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] val_loss = {val_loss:.5f} val_dice = {val_dice:.5f}")

        # ---------- Testing (using ensemble)----------
        test_loss, test_dice= test_dual(
            model_l, model_r, test_loader)
        print(
            f"[ Test | {epoch + 1:03d}/{num_epoch:03d} ] test_loss = {test_loss:.5f} test_dice = {test_dice:.5f}")

        # val
        wandb.log({'val/val_dice': val_dice})
        # test
        wandb.log({'test/test_dice': test_dice})
        # loss
        wandb.log({'epoch': epoch + 1, 'loss/total_loss': total_loss, 'loss/total_loss_l': total_loss_l,
                  'loss/total_loss_r': total_loss_r, 'loss/total_cps_loss': total_cps_loss,
                   'loss/test_loss': test_loss, 'loss/val_loss': val_loss, 'loss/con_loss': total_con_loss })

        # if the model improves, save a checkpoint at this epoch
        if val_dice > best_dice:
            best_dice = val_dice
            # 使用了多GPU需要加上module
            print('saving model with best_dice {:.5f}'.format(best_dice))
            model_name_l = './tmodel/' + 'l_' + model_path
            model_name_r = './tmodel/' + 'r_' + model_path
            torch.save(model_l.module, model_name_l)
            torch.save(model_r.module, model_name_r)

def main():
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epoch = config['num_epoch']
    model_path = config['model_path']
    test_vendor = config['test_vendor']

    label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, num_label_imgs, num_unsup_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)

    max_samples = num_unsup_imgs
    niters_per_epoch = int(math.ceil(max_samples * 1.0 // batch_size))
    print("max_samples", max_samples)
    print("niters_per_epoch", niters_per_epoch)

    if config['Fourier_aug']:
        print("Fourier mode")
    else:
        print("Normal mode")

    train(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, learning_rate,
          weight_decay, num_epoch, model_path, niters_per_epoch)

if __name__ == '__main__':
    main()
