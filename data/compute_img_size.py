import glob
import time
import argparse

import matplotlib.pyplot as plt

from collections import Counter, OrderedDict
from PIL import Image


def parse_args():
    """
    Parser for the command line arguments.

    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Compute mean and std of group of images by ignoring the masks.")

    parser.add_argument("dataset_name",
                        type=str,
                        help="Dataset name")

    parser.add_argument("images_folder",
                        type=str,
                        help="Path to images to extract stats from.")

    parser.add_argument("-f", "--format",
                        type=str,
                        default="jpeg",
                        help="Image format to look for. Default: 'jpeg'")

    args = parser.parse_args()
    return args


def plot_image_sizes(im_size, dataset_name):
    """
    Plot and save image sizes.

    Parameters
    ----------
    im_size : list
        List of image sizes.
    dataset_name : str
        Dataset name.
    """
    D = OrderedDict(sorted(Counter(im_size).items()))
    fig, ax = plt.subplots()
    bar_plot = plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()), rotation=90)

    def autolabel_vertical(bar_plot, bar_label):
        for idx, rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.01 * sum(bar_label),
                    f'n = {bar_label[idx]} ({(bar_label[idx] / sum(bar_label)) * 100:.2f}%)',
                    ha='center', va='bottom', rotation=90)

    autolabel_vertical(bar_plot, list(D.values()))
    plt.ylim(0, max(list(D.values())) + 0.8 * max(list(D.values())))
    plt.ylabel("# of images")
    plt.xlabel("image sizes")
    plt.title(f"{' '.join(dataset_name.split('_'))} (n={len(im_size)})")
    plt.tight_layout()
    plt.savefig(dataset_name + '.png', transparent=True, dpi=1000)


if __name__ == '__main__':
    args = parse_args()
    s_time = time.time()
    print("Extracting image sizes ...", end=' ')
    im_size = []
    for fname in glob.glob(f"{args.images_folder}/*.{args.format}"):
        with Image.open(fname) as im:
            im_size.append(im.size)
    print(f"Done in {time.time() - s_time:.2f}s")

    plot_image_sizes(im_size, args.dataset_name)
