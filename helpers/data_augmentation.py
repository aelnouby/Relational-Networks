"""
Implementing Data augmentation transform classes
Author: Alaaeldin El-Nouby
"""
import numpy as np
from imgaug import augmenters as iaa


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.augmenter_flip = iaa.Sequential([iaa.Fliplr(1)])

    def __call__(self, sample):
        rgb = sample

        if np.random.random() > self.p:
            rgb = self.augmenter_flip.augment_images(rgb)

        sample = rgb
        return sample


class VerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.augmenter_flip = iaa.Sequential([iaa.Flipud(1)])

    def __call__(self, sample):
        rgb = sample

        if np.random.random() > self.p:
            rgb = self.augmenter_flip.augment_images(rgb)

        sample = rgb
        return sample


class ColorDistortion(object):
    def __init__(self, bounds=30):
        self.bounds = bounds
        self.augmenter_hsv = iaa.Sequential([iaa.AddToHueAndSaturation((-bounds, bounds))])

    def __call__(self, sample):

        rgb = sample

        rgb = self.augmenter_hsv.augment_images(rgb)

        sample = rgb

        return sample


class RandomCrop(object):
    def __init__(self, p=1, percent=(0.06, 0.06)):
        self.p = p
        self.augmenter_crop = iaa.Sequential([iaa.Crop(percent=percent)])

    def __call__(self, sample):
        rgb = sample

        if np.random.random() < self.p:
            rgb = self.augmenter_crop.augment_images(rgb)

        sample = rgb

        return sample


class Rotation(object):
    def __init__(self, rad=(-2.86, 2.86)):
        self.augmenter_rotate = iaa.Sequential([iaa.Affine(rotate=rad)])

    def __call__(self, sample):
        sample = self.augmenter_rotate.augment_images(sample)
        return sample
