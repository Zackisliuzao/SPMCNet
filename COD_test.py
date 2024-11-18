import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import time

from model.SPMCNet_models import SPMCNet_VGG
from data import test_dataset
# 改
import imageio

torch.cuda.set_device(-1)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

# 改
dataset_path = './dataset/COD_CAMO/test/'

model = SPMCNet_VGG()
model.load_state_dict(torch.load('./models/SPMCNet_VGG/SPMCNet_VGG.pth.40'))

model.cuda()
model.eval()

test_datasets = ['COD_CAMO']

for dataset in test_datasets:
    save_path = './results/VGG/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 改
    image_root = dataset_path + '/image/'
    print(dataset)
    gt_root = dataset_path + '/gt/'
    depth_root = dataset_path + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        time_start = time.time()
        res, rgbd_pred2, rgbd_pred3, rgbd_pred4, depth_pred, = model(image, depth)
        time_end = time.time()
        time_sum = time_sum + (time_end - time_start)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # 改
        res = (res * 255).astype(np.uint8)
        imageio.imsave(save_path + name, res)
        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))
