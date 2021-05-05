import os
import glob
import argparse

import numpy as np

from multiprocessing import Pool
from PIL import Image, ImageFilter


def parse_args():
    """
    Parser for the command line arguments.
    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Crop and resize kaggle 2015 image data.")

    parser.add_argument("images_folder",
                        type=str,
                        help="Path to images to extract stats from.")

    args = parser.parse_args()
    return args


def resize_and_pad(im, desired_size):
    """
    Resize to desired size while keeping aspect ratio and pad the image so that
    it's a square.

    Parameters
    ----------
    im : numpy array
        Image to resize.
    desired_size : int
        Desired size.
    """
    # argwhere gives the coordinates of every non-zero point
    true_points = np.argwhere(im)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    box = (top_left[1], top_left[0], bottom_right[1], bottom_right[0])
    w = bottom_right[1] - top_left[1]
    h = bottom_right[0] - top_left[0]
    ratio = float(desired_size) / max((w, h))
    new_size = tuple([int(x * ratio) for x in (w, h)])
    # Convert numpy array to PIL image
    im = Image.fromarray(im)
    resized_512 = im.resize(new_size, Image.NEAREST, box)
    new_im = Image.new('RGB', (desired_size, desired_size))
    new_im.paste(resized_512, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def preprocess_image(fname):
    """
    Crop and resize an image.

    Parameters
    ----------
    fname : str
        Path to file.
    """
    try:
        with Image.open(fname) as im_pil:
            # build mask
            mask = im_pil.convert('L').filter(ImageFilter.BoxBlur(20)).point(lambda x: 1 if x >= 9 else 0)
            # apply mask to the image
            im = np.array(im_pil) * np.array(mask)[..., None]

        # 512 resize, zero padding and save
        im_512 = resize_and_pad(im, 512)
        im_512.save(os.path.join(f"{os.path.dirname(fname)}_512", fname.split('/')[-1]))

        # Warn if the mask seems to small
        croppedmask = np.array(im_512).sum(2)
        ratio = np.round(croppedmask.flatten().nonzero()[0].size / croppedmask.size, 2)
        if ratio <= 0.66:
            print(f"Warning: ratio ({ratio}) <= 0.66 threshold")
            print(fname, "\n")

        # 1024 resize, zero padding and save
        im_1024 = resize_and_pad(im, 1024)
        im_1024.save(os.path.join(f"{os.path.dirname(fname)}_1024", fname.split('/')[-1]))

    except ValueError:
        print(fname)
        pass


if __name__ == "__main__":
    args = parse_args()
    fnames = glob.glob(f"{args.images_folder}/*.jpeg")
    with Pool() as p:
        p.map(preprocess_image, fnames)
