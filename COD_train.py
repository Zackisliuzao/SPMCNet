import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.SPMCNet_models import SPMCNet_VGG
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou
import pytorch_fm
import pytorch_ssim
from pytorch_par.pamr import BinaryPamr

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./SPMC-logs/COD')

from data import test_dataset

torch.cuda.set_device(-1)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = SPMCNet_VGG()

model.cuda()
params = model.parameters()
optimizer = torch.optim.NAdam(params, opt.lr)
# 改
image_root = './dataset/COD_SC/train/image/'
gt_root = './dataset/COD_SC/train/gt/'
depth_root = './dataset/COD_SC/train/depth/'


train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)


def only_ssim_loss(pred, target):
    pred = torch.sigmoid(pred)
    ssim_out = 1 - ssim_loss(pred, target)

    loss = ssim_out

    return loss


def bce_iou_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    bce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return (bce + iou).mean()


def run_pamr(img, sal):
    lbl_self = BinaryPamr(img, sal.clone().detach(), binary=0.4)
    return lbl_self


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


def smooth_normal_loss(disp):
    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = Sobel().cuda()
    grad = get_gradient(disp)
    grad_x, grad_y = grad[:, 0].unsqueeze(1), grad[:, 1].unsqueeze(1)
    ones = torch.ones(grad.size(0), 1, grad.size(2), grad.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)
    depth_normal = torch.cat((-grad_x, -grad_y, ones), 1)
    l1 = torch.abs(1 - cos(depth_normal[:, :, :, :-1], depth_normal[:, :, :, 1:]))
    l2 = torch.abs(1 - cos(depth_normal[:, :, :-1, :], depth_normal[:, :, 1:, :]))
    l3 = torch.abs(1 - cos(depth_normal[:, :, 1:, :-1], depth_normal[:, :, :-1, 1:]))
    l4 = torch.abs(1 - cos(depth_normal[:, :, :-1, :-1], depth_normal[:, :, 1:, 1:]))
    return (l1.mean() + l2.mean() + l3.mean() + l4.mean()) / 4


def total_variation_loss(img, weight):
    edge_h = 1 - torch.abs(weight[:, :, 1:, :] - weight[:, :, :-1, :])
    edge_h += edge_h.mean()
    edge_w = 1 - torch.abs(weight[:, :, :, 1:] - weight[:, :, :, :-1])
    edge_w += edge_w.mean()
    tv_h = edge_h * torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2)
    tv_w = edge_w * torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2)
    return (tv_h.sum() + tv_w.sum()) / 2


def weighted_total_variant(disp, gt):
    get_gradient = Sobel().cuda()
    grad = get_gradient(gt)
    grad_x, grad_y = grad[:, 0].unsqueeze(1), grad[:, 1].unsqueeze(1)
    temp_edge = torch.mul(grad_x, grad_x) + torch.mul(grad_y, grad_y)
    temp_edge[temp_edge != 0] = 1
    temp_edge[temp_edge == 0] = 0.5
    to_var = total_variation_loss(disp, temp_edge)
    return to_var


def train(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, depths, label_gt, label_gt_depth = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        label_gt = Variable(label_gt)
        label_gt_depth = Variable(label_gt_depth)
        images = images.cuda()
        gts = gts.cuda()
        depths = depths.cuda()
        label_gt = label_gt.cuda()
        label_gt_depth = label_gt_depth.cuda()

        rgbd_pred1, rgbd_pred2, rgbd_pred3, rgbd_pred4, depth_pred = model(images, depths)

        lbl_tea = run_pamr(depths, (label_gt + label_gt_depth) / 2)
        depth_ce_loss = CE(depth_pred, label_gt_depth)
        depth_sm_loss = smooth_normal_loss(depth_pred * lbl_tea)




        loss1 = bce_iou_loss(rgbd_pred1, lbl_tea) + only_ssim_loss(rgbd_pred1, lbl_tea)
        loss2 = bce_iou_loss(rgbd_pred2, lbl_tea) + only_ssim_loss(rgbd_pred2, lbl_tea)
        loss3 = bce_iou_loss(rgbd_pred3, lbl_tea) + only_ssim_loss(rgbd_pred3, lbl_tea)
        loss4 = bce_iou_loss(rgbd_pred4, lbl_tea) + only_ssim_loss(rgbd_pred4, lbl_tea)

        loss = 1 * loss1 + 0.8 * loss2 + 0.6 * loss3 + 0.4 * loss4 + depth_ce_loss + depth_sm_loss

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                       loss2.data))
        total_loss += loss.item()

    save_path = 'models/SPMCNet_VGG/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + 'SPMCNet_VGG.pth' + '.%d' % epoch)

print("Let's go!")

if __name__ == '__main__':
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
    writer.close()
