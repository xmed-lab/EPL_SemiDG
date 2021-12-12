import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.dice_loss import dice_coeff

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if ifimg:
        image = image.transpose(1,2,0)
    return image

def get_one_hot(label, N, device='cuda'):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N).to(device)
    ones = ones.index_select(0, label)   # 转换为one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def label_to_onehot(label, num_classes = 4):
    # one_hot = torch.nn.functional.one_hot(label, num_classes).float() # size=(b,h,w,n)
    one_hot = get_one_hot(label, num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2) # size(b,n,h,w)
    return one_hot

def find_max_region(mask_sel):
    __, contours,hierarchy = cv2.findContours(mask_sel,cv2.RETR_TREE, v2.CHAIN_APPROX_NONE)
 
    #找到最大区域并填充 
    area = []
 
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
 
    max_idx = np.argmax(area)
 
    max_area = cv2.contourArea(contours[max_idx])
 
    for k in range(len(contours)):
    
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

# def flatten(tensor):
#     """Flattens a given tensor such that the channel axis is first.
#     The shapes are transformed as follows:
#        (N, C, D, H, W) -> (C, N * D * H * W)
#     """
#     C = tensor.size(1)
#     # new axis order
#     axis_order = (1, 0) + tuple(range(2, tensor.dim()))
#     # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#     transposed = tensor.permute(axis_order)
#     # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#     return transposed.contiguous().view(C, -1)
 
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.epsilon = 1e-5

    # def forward(self, output, target):
    #     assert output.size() == target.size(), "'input' and 'target' must have the same shape"
    #     output = F.softmax(output, dim=1)
    #     output = flatten(output)
    #     target = flatten(target)
    #     # intersect = (output * target).sum(-1).sum() + self.epsilon
    #     # denominator = ((output + target).sum(-1)).sum() + self.epsilon
 
    #     intersect = (output * target).sum(-1) + self.epsilon
    #     denominator = (output + target).sum(-1) + self.epsilon
    #     dice = intersect / denominator
    #     dice = torch.mean(dice)
    #     return 1 - dice
    #     # return 1 - 2. * intersect / denominator

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

def check_accuracy(loader, model, device="cuda"):
    model.eval()

    tot = 0
    tot_lv = 0
    tot_myo = 0
    tot_rv = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch['img']
            y = batch['mask']

            x = x.to(device)
            y = y.to(device)
            preds = model(x)

            true_masks = y
            pred = F.softmax(preds, dim=1)
            pred = (pred > 0.5).float()

            tot += dice_coeff(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :], device).item()
            tot_lv += dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
            tot_myo += dice_coeff(pred[:, 1, :, :], true_masks[:, 1, :, :], device).item()
            tot_rv += dice_coeff(pred[:, 2, :, :], true_masks[:, 2, :, :], device).item()

    dice_lv = tot_lv/len(loader)
    dice_myo = tot_myo/len(loader)
    dice_rv = tot_rv/len(loader)
    dice = tot/len(loader)
    model.train()
    return dice, dice_lv, dice_myo, dice_rv

def check_accuracy_dual(loader, model_r, model_l, device="cuda"):
    model_r.eval()
    model_l.eval()

    tot = 0
    tot_lv = 0
    tot_myo = 0
    tot_rv = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch['img']
            y = batch['mask']

            x = x.to(device)
            y = y.to(device)
            true_masks = y

            preds_r = model_r(x)
            preds_l = model_l(x)

            pred_r = F.softmax(preds_r, dim=1)
            pred_l = F.softmax(preds_l, dim=1)
            pred = (pred_r + pred_l) / 2
            pred = (pred > 0.5).float()

            tot += dice_coeff(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :], device).item()
            tot_lv += dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
            tot_myo += dice_coeff(pred[:, 1, :, :], true_masks[:, 1, :, :], device).item()
            tot_rv += dice_coeff(pred[:, 2, :, :], true_masks[:, 2, :, :], device).item()

    dice_lv = tot_lv/len(loader)
    dice_myo = tot_myo/len(loader)
    dice_rv = tot_rv/len(loader)
    dice = tot/len(loader)
    model_r.train()
    model_l.train()
    return dice, dice_lv, dice_myo, dice_rv