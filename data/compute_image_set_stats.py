#!/usr/bin/env python3

import glob
import time
import argparse

import numpy as np

from matplotlib.pyplot import imread


def parse_args():
    """
    Parser for the command line arguments.

    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Compute mean and std of group of images by ignoring the masks.")

    parser.add_argument("images_folder",
                        type=str,
                        help="Path to image to folder containing the images to extract the stats from.")

    parser.add_argument("-f", "--format",
                        type=str,
                        default="jpeg",
                        help="Image format to look for. Default: 'jpeg'")

    parser.add_argument("--low-ram",
                        dest='low_ram',
                        action='store_true',
                        default=False,
                        help="Use slower but low ram method to compute the stats.")

    parser.add_argument("--subset",
                        type=int,
                        default=None,
                        help="Use first N element of the dataset to compute the stats. Default: all")

    args = parser.parse_args()
    return args


def mean_std_quick_high_ram(images_path):
    """
    Using a very high ram high speed technique.
    Compute the standard deviation and mean per channel of pixel values
    of all the images contained in the dataset.

    Parameters
    ----------
    images_path : list
        List of path to all images of the dataset.
    """

    dims = [0, 1, 2]
    pix = {0: [], 1: [], 2: []}

    s_time = time.time()
    print("Extracting pixel values ...", end=' ')
    for img_path in images_path:
        img = imread(img_path)
        for d in dims:
            channel = img[:, :, d].flatten()
            pix[d] += [channel[channel.nonzero()]]
            del channel
        del img
    print(f"Done in {time.time()-s_time:.2f}s")

    for d in dims:
        s_time = time.time()
        print(f"Concatenating pixels from dim {d} ...", end=' ')
        pix[d] = np.concatenate(pix[d])
        print(f"Done in {time.time()-s_time:.2f}s")

        s_time = time.time()
        print(f"Dim {d} - mean: {np.round(np.mean(pix[d]), 4)} std: {np.round(np.std(pix[d]), 4)}. Computed in {time.time()-s_time:.2f}s")
        del pix[d]


def _get_mean_slow(images_path):
    """
    Compute the mean per channel of pixel values
    of all the images contained in the dataset.

    Parameters
    ----------
    images_path : list
        List of path to all images of the dataset.

    Returns
    -------
    mean : array
        RGB mean channel pixel value.
    """
    print("Computing mean ...", end=' ')
    s_time = time.time()
    per_channel_mean = 0
    for fname in images_path:
        img = imread(fname)
        mask = img != 0
        per_channel_mean += (img.sum(axis=(0, 1)) / mask.sum(axis=(0, 1)))

    per_channel_mean /= len(images_path)
    print(f"Done in {time.time()-s_time:.2f}s")
    print(f"RGB mean: {np.round(per_channel_mean, 4)}")
    return per_channel_mean


def _get_std_slow(images_path, mean):
    """
    Compute the standard deviation per channel of pixel values
    of all the images contained in the dataset.

    Parameters
    ----------
    images_path : list
        List of path to all images of the dataset.
    mean : arary
        Data set mean per channel (R, G, B).
    """
    print("Computing std ...", end=' ')
    s_time = time.time()
    per_channel_std = 0
    total_num_pixel_per_channel = 0
    for fname in images_path:
        img = imread(fname)
        mask = img != 0
        total_num_pixel_per_channel += mask.sum((0, 1))
        per_channel_std += (((img - mean) * mask)**2).sum(axis=(0, 1))

    per_channel_std /= total_num_pixel_per_channel
    per_channel_std = np.sqrt(per_channel_std)

    print(f"Done in {time.time()-s_time:.2f}s")
    print(f"RGB std: {np.round(per_channel_std, 4)}")


def mean_std_slow_low_ram(images_path):
    """
    Using a low ram low speed technique.
    Compute the standard deviation and mean per channel of pixel values
    of all the images contained in the dataset.

    Parameters
    ----------
    images_path : list
        List of path to all images of the dataset.
    """

    mean = _get_mean_slow(images_path)
    _get_std_slow(images_path, mean)


if __name__ == '__main__':
    args = parse_args()

    s_time = time.time()
    print("Gathering images ...", end=' ')
    images_path = sorted(glob.glob(f"{args.images_folder}/*.{args.format}"))
    print(f"Done in {time.time()-s_time:.2f}s")

    if len(images_path) == 0:
        raise ValueError("Invalid path, no images found.")

    print(f"Dataset contains {len(images_path)} images.")

    if args.subset is not None:
        images_path = images_path[:args.subset]
        print(f"Using a subset of {args.subset} to compute the stats.")

    if args.low_ram:
        mean_std_slow_low_ram(images_path)
    else:
        mean_std_quick_high_ram(images_path)
