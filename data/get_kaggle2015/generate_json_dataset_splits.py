import os
import json
import argparse

from collections import defaultdict

import pandas as pd


def parse_args():
    """
    Parser for the command line arguments.
    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Generate json dataset.")

    parser.add_argument('--data_folder',
                        type=str,
                        help="Data folder.")

    parser.add_argument('--split',
                        type=str,
                        help="Split.")

    args = parser.parse_args()
    return args


def generate_split_json(data_folder, split):
    """
    Get preprocess label for Kaggle retinopathy 2015 data set. Merge
    information about labels and quality.

    Parameters
    ----------
    data_folder: str
        Absolute path to the data folder.
    split: str
        Split to preprocess. In ["train", "valid", "test"].
    """
    print(f"Split: {split}")
    if split == "train":
        data = pd.read_csv(os.path.join(data_folder, "trainLabels.csv"))
        data_quality = pd.read_csv(os.path.join(data_folder, "Label_EyeQ_train.csv"))
    elif split in ["valid", "test"]:
        data = pd.read_csv(os.path.join(data_folder, "retinopathy_solution.csv"))
        data_quality = pd.read_csv(os.path.join(data_folder, "Label_EyeQ_test.csv"))
        if split == "valid":
            data = data[data["Usage"] == "Public"]
        else:
            data = data[data["Usage"] == "Private"]

    mapping_screening = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}

    data_json = defaultdict(dict)
    patients = sorted(list(set([int(i.split("_")[0]) for i in
                                data["image"].tolist()])))

    # Remove unwanted patients due to image quality
    if split == "train":
        unwanted_patients = [492, 1986, 32253, 34689, 43457]
    else:
        unwanted_patients = []
    patients = [patient for patient in patients if
                patient not in unwanted_patients]

    for patient in patients:
        # For each patient there is data for both eye (right/left)
        for side in ["right", "left"]:
            image_path = f"{patient}_{side}"
            level = int(data[data["image"] == image_path]["level"])
            screening = mapping_screening[level]
            data_json[patient][f"{side}_image_path"] = image_path
            data_json[patient][f"{side}_DR_level"] = level
            data_json[patient][f"{side}_screening_level"] = screening
            if not data_quality[data_quality["image"] == f"{image_path}.jpeg"].empty:
                quality = int(data_quality[data_quality["image"] == f"{image_path}.jpeg"]["quality"])
                data_json[patient][f"{side}_quality"] = quality
            else:
                quality = -1
                data_json[patient][f"{side}_quality"] = quality

        data_json[patient]["patient_DR_level"] = max(data_json[patient]["right_DR_level"], data_json[patient]["left_DR_level"])
        data_json[patient]["screening_RD"] = max(data_json[patient]["right_screening_level"], data_json[patient]["left_screening_level"])
        data_json[patient]["overall_quality"] = max(data_json[patient]["right_quality"], data_json[patient]["left_quality"])

    label_file_name = f"labels_{split}.json"
    print(f"Saving: {label_file_name}")
    with open(os.path.join(data_folder, label_file_name), "w") as f:
        json.dump(data_json, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    args = parse_args()
    generate_split_json(args.data_folder, args.split)
