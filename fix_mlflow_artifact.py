#!/usr/bin/env python3

import os
import re
import glob
import argparse


def parse_args():
    """
    Parser for the command line arguments.

    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Update artifact path of mlruns.")

    parser.add_argument('mlruns_path',
                        type=str,
                        help="Path to mlruns folder to update.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    base_path = os.path.realpath(args.mlruns_path)
    path_search = f"{base_path}/**/meta.yaml"

    updated_files = 0
    # Retreive and update the artifact path in all meta.yaml files
    for path in glob.glob(path_search, recursive=True):

        dir_name = os.path.basename(os.path.dirname(path))
        # If the folder containing the meta.yaml is a digit ex: 0/, 1/ ...
        # The key to update in the file is artifact_location
        # Otherwise if the folder is a 32 char hash (a run id)
        # The key to upate is artifact_uri
        if dir_name.isdigit():
            artifact_path = os.path.dirname(path)
            var_name = "artifact_location"
        elif len(dir_name) == 32:
            artifact_path = os.path.join(os.path.dirname(path), "artifacts")
            var_name = "artifact_uri"
        else:
            raise Exception("Unexpected parent folder for meta.yaml.\n{path}")

        # Replace the appropriate key in the yaml
        with open(path, 'r+') as f:
            text = f.read()
            text = re.sub(f"{var_name}: .+", f"{var_name}: {artifact_path}", text)
            f.seek(0)
            f.write(text)
            f.truncate()
            updated_files += 1
    print(f"Updated {updated_files} meta.yaml")
