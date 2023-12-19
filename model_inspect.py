import argparse
import numpy as np
import torch
from safetensors.torch import safe_open, save_file

import os

def model_inspect(file_path1):
    merged_tensors = {}

    with safe_open(file_path1, framework="pt", device="cpu") as f1:
        keys1 = set(f1.keys())

        for key in keys1:
            tensor1 = f1.get_tensor(key)
            print("Found", key, tensor1.shape)

    return merged_tensors

def map_tensors_to_files(directory_path, extension=".safetensors"):
    tensor_file_map = {}

    for filename in os.listdir(directory_path):
        # Check if the file has the specified extension
        if filename.endswith(extension):
            file_path = os.path.join(directory_path, filename)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                print("-metadata", f.metadata())
                keys = set(f.keys())
                for key in keys:
                    tensor = f.get_tensor(key)
                    # Map the tensor key to its filename
                    tensor_file_map[key] = {'filename':filename, 'shape':tensor.shape, 'tensor': tensor}
    return tensor_file_map


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge two safetensor model files.')
    parser.add_argument('base_model', type=str, help='The base model safetensor file')
    args = parser.parse_args()

    if os.path.isdir(args.base_model):
        tmap = map_tensors_to_files(args.base_model)
        for key in sorted(tmap.keys()):
            print(key, tmap[key]['shape'])
    else:
        model_inspect(args.base_model)

if __name__ == '__main__':
    main()
