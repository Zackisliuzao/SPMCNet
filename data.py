import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from fast_slic import Slic
import cv2
import torch


# several data augumentation strategies
class RandomScale(object):
    def __init__(self, scale_list=[0.75, 1.0, 1.25], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask, depth):
        oh, ow = img.size
        scale_amt = 1.0
        if self.mode == 'value':
            scale_amt = np.random.choice(self.scale_list, 1)
        elif self.mode == 'range':
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])
        h = int(scale_amt * oh)
        w = int(scale_amt * ow)
        return img.resize((h, w), Image.BICUBIC), mask.resize((h, w), Image.NEAREST), depth.resize((h, w),
                                                                                                   Image.BICUBIC)


def cv_random_flip_1(img, label, depth):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth


def cv_random_flip_2(img, label, depth):
    actions = ['none', 'horizontal', 'vertical']
    flip_action = random.choice(actions)

    if flip_action == 'horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_action == 'vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)

    return img, label, depth


def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)


def randomRotation(image, label, depth):
    angles = [0, 90, 180, 270]
    random_angle = random.choice(angles)

    if random_angle != 0:
        image = image.rotate(random_angle, Image.BICUBIC, expand=True)
        label = label.rotate(random_angle, Image.NEAREST, expand=True)
        depth = depth.rotate(random_angle, Image.BICUBIC, expand=True)

    return image, label, depth


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.scale = RandomScale(scale_list=[0.75, 1.0, 1.25], mode='value')

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])
        image, gt, depth = self.scale(image, gt, depth)
        image, gt, depth = cv_random_flip_2(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)

        # Convert to numpy array and resize
        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)

        np_depth = np.array(depth)
        np_depth = cv2.resize(np_depth, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)

        np_gt = np.array(gt)
        np_gt = cv2.resize(np_gt, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)

        slic = Slic(num_components=100, compactness=10)
        SS_map = slic.iterate(np_img)
        SS_map_depth = slic.iterate(np_depth)

        SS_map = SS_map + 1
        SS_map_depth = SS_map_depth + 1

        label_gt = np.zeros((1, self.trainsize, self.trainsize))
        label_gt_depth = np.zeros((1, self.trainsize, self.trainsize))

        for i in range(1, 100 + 1):
            buffer = np.copy(SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                # Calculate overlap with foreground
                foreground_overlap = np.sum(buffer * (np_gt == 1))
                # Calculate overlap with the background
                background_overlap = np.sum(buffer * (np_gt == 2))

                # If it is mainly foreground overlap, mark it as foreground
                if foreground_overlap > background_overlap and foreground_overlap > 1:
                    label_gt = np.maximum(label_gt, buffer)
                # Otherwise, keep it as background or unmarked

        for i in range(1, 100 + 1):
            buffer = np.copy(SS_map_depth)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                foreground_overlap = np.sum(buffer * (np_gt == 1))
                background_overlap = np.sum(buffer * (np_gt == 2))

                if foreground_overlap > background_overlap and foreground_overlap > 1:
                    label_gt_depth = np.maximum(label_gt_depth, buffer)

        label_gt = torch.tensor(label_gt).to(torch.float32)
        label_gt_depth = torch.tensor(label_gt_depth).to(torch.float32)

        # label_gt_ar = label_gt.squeeze(0)
        # label_gt_ar = np.array(label_gt_ar)
        #
        # label_gt_depth_ar = label_gt_depth.squeeze(0)
        # label_gt_depth_ar = np.array(label_gt_depth_ar)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        return image, gt, depth, label_gt, label_gt_depth

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.depths)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                      Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=16, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if
                       f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        # self.gt_transform = transforms.ToTensor()
        self.depth_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.rgb_loader(self.depths[self.index]) 
        depth = self.depth_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
