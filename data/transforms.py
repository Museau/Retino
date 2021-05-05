import random

import cv2
import torch
import torchvision.transforms.functional as F
import numpy as np


class MaskedStandardization:
    """
    Standardize image and then set padding values back to zero.
    """

    def __init__(self, mean, std):
        """
        Parameters
        ----------
        mean : tuple
            The mean of each dimensions.
        std : tuple
            The standard deviation of each dimensions.
        """
        self._mean = torch.as_tensor(mean, dtype=torch.float32)
        self._std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError(f"img should be a torch tensor. Got {type(img)}.")

        if img.ndimension() != 3:
            raise ValueError(f"Expected img to be a tensor image of size (C, H, W). Got tensor.size() = {img.size()}.")

        padding_mask = img != 0

        img.transpose(0, -1).sub_(self._mean).div_(self._std)

        img *= padding_mask
        return img


class KaggleWinner:
    """
    Apply 'Feature Scaling' / run time pre-processing of the winner of the Kaggle competition.

    https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py
    """

    def __init__(self, image_size):
        """
        Parameters
        ----------
        image_size : int
            The size of one side of a square image.
        """
        self._scale = image_size

    def __call__(self, img):
        original_img = img
        mask = np.zeros(original_img.shape)

        # The original parameter was assuming diameter and not radius
        # It was also assuming that the eyes were not totally centered `*0.9` (we do it in offline pre-processing)
        # Original formula was : int(scale*0.9)
        radius = self._scale // 2
        circle_center = (original_img.shape[1] // 2, original_img.shape[0] // 2)
        color = (1, 1, 1)
        cv2.circle(mask, circle_center, radius, color, -1, 8, 0)

        # Make a blurred copy
        kernel_size = (0, 0)
        sigma_x = self._scale / 30
        blurred_img = cv2.GaussianBlur(original_img, kernel_size, sigma_x)

        # Blending original and blurred image and bias toward grey (128)
        # dst = α ⋅ src1 + β ⋅ src2 + γ | (4 * img) + (-4 * blurred_img) + 128
        new_img = cv2.addWeighted(original_img, 4, blurred_img, -4, 128)

        # Bias toward grey and apply mask
        masked_image = new_img * mask + 128 * (1 - mask)
        return masked_image


class GreenChannelOnly:
    """
    Keep only the green channel

    Farida's group has promising preliminarily results on fundus data.
    """

    def __call__(self, img):
        return img[1].unsqueeze(0).expand(3, -1, -1)


class Rotate:
    """
    Apply a rotation.
    """

    def __init__(self, angle):
        """
        Parameters
        ----------
        angle : float
            Angle for the rotation.
        """
        self.angle = angle

    def __call__(self, img):
        # img = F.rotate(img, self.angle)  # Uncomment at pytorch release > 7.0
        img = torch.rot90(img, self.angle // 90, (2, 1))  # Remove at pytorch release > 7.0
        return img


class RandomHorizontalFlipRotate:
    """
    Apply random horizontal flip followed by a rotation.
    """

    def __init__(self, angle):
        """
        Parameters
        ----------
        angle : float
            Angle for the rotation.
        """
        self.angle = angle

    def __call__(self, img):
        img = F.hflip(img)
        # img = F.rotate(img, self.angle)  # Uncomment at pytorch release > 7.0
        img = torch.rot90(img, self.angle // 90, (2, 1))  # Remove at pytorch release > 7.0
        return img


class RandomApplyChoice:
    """
    Apply single transformation randomly picked from a list.
    """

    def __init__(self, transforms):
        """
        Parameters
        ----------
        transforms : list or tuple
            List of transformations.
        """
        self.transforms = transforms

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)
