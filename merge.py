import argparse
import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

def merge_tensors(tensor1, tensor2, p):
    # Calculate the delta of the weights
    delta = tensor2 - tensor1
    # Generate the mask m^t from Bernoulli distribution
    m = torch.from_numpy(np.random.binomial(1, p, delta.shape)).to(tensor1.dtype)
    # Apply the mask to the delta to get δ̃^t
    delta_tilde = m * delta
    # Scale the masked delta by the dropout rate to get δ̂^t
    delta_hat = delta_tilde / (1 - p)
    return delta_hat

def merge_safetensors(file_path1, file_path2, p, lambda_val):
    merged_tensors = {}

    with safe_open(file_path1, framework="pt", device="cpu") as f1, safe_open(file_path2, framework="pt", device="cpu") as f2:
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())
        common_keys = keys1.intersection(keys2)

        for key in common_keys:
            tensor1 = f1.get_tensor(key)
            tensor2 = f2.get_tensor(key)
            tensor1, tensor2 = resize_tensors(tensor1, tensor2)
            merged_tensors[key] = tensor1 + lambda_val * merge_tensors(tensor1, tensor2, p)
            print("merging", key)

    return merged_tensors

class BinDataHandler():
    def __init__(self, data):
        self.data = data

    def get_tensor(self, key):
        return self.data[key]

def read_tensors(file_path, ext):
    if ext == ".safetensors" and file_path.endswith(".safetensors"):
        f = safe_open(file_path, framework="pt", device="cpu")
        return f, set(f.keys())
    if ext == ".bin" and file_path.endswith(".bin"):
        data = torch.load(file_path, map_location=torch.device('cpu'))
        f = BinDataHandler(data)
        return f, set(data.keys())
    return None, None

def resize_tensors(tensor1, tensor2):
    if len(tensor1.shape) not in [1, 2]:
        return tensor1, tensor2

    # Pad along the last dimension (width)
    if tensor1.shape[-1] < tensor2.shape[-1]:
        padding_size = tensor2.shape[-1] - tensor1.shape[-1]
        tensor1 = F.pad(tensor1, (0, padding_size, 0, 0))
    elif tensor2.shape[-1] < tensor1.shape[-1]:
        padding_size = tensor1.shape[-1] - tensor2.shape[-1]
        tensor2 = F.pad(tensor2, (0, padding_size, 0, 0))

    # Pad along the first dimension (height)
    if tensor1.shape[0] < tensor2.shape[0]:
        padding_size = tensor2.shape[0] - tensor1.shape[0]
        tensor1 = F.pad(tensor1, (0, 0, 0, padding_size))
    elif tensor2.shape[0] < tensor1.shape[0]:
        padding_size = tensor1.shape[0] - tensor2.shape[0]
        tensor2 = F.pad(tensor2, (0, 0, 0, padding_size))

    return tensor1, tensor2

def merge_folder(tensor_map, directory_path, p, lambda_val):
    keys1 = set(tensor_map.keys())
    # Some repos have both bin and safetensors, choose safetensors if so
    ext = None
    for filename in os.listdir(directory_path):
        # Default to safetensors
        if filename.endswith(".safetensors"):
            ext = ".safetensors"
        if filename.endswith(".bin") and ext is None:
            ext = ".bin"
    if ext is None:
        raise "Could not find model files"

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        f, keys2 = read_tensors(file_path, ext)
        if keys2:
            common_keys = keys1.intersection(keys2)
            for key in common_keys:
                if "block_sparse_moe.gate" in key:
                    tensor1 = tensor_map[key]['tensor']
                    tensor2 = f.get_tensor(key)
                    tensor_map[key]['tensor'] = (tensor1 + tensor2) /2.0
                    continue
                tensor1 = tensor_map[key]['tensor']
                tensor2 = f.get_tensor(key)
                tensor1, tensor2 = resize_tensors(tensor1, tensor2)
                tensor_map[key]['tensor'] = tensor1 + lambda_val * merge_tensors(tensor1, tensor2, p)
    return tensor_map

def map_tensors_to_files(directory_path):
    tensor_map = {}

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        f, keys = read_tensors(file_path, '.safetensors')
        if keys:
            for key in keys:
                tensor = f.get_tensor(key)
                tensor_map[key] = {'filename':filename, 'shape':tensor.shape, 'tensor': tensor}

    return tensor_map

def copy_nontensor_files(from_path, to_path):
    for filename in os.listdir(from_path):
        file_path = os.path.join(from_path, filename)
        if from_path != to_path and not filename.startswith(".") and not filename.startswith("README") and not filename.endswith(".bin") and not filename.endswith(".safetensors") and not filename.endswith(".pt") and not os.path.isdir(file_path):
            print(f"Copying {file_path} to {to_path}")
            shutil.copyfile(file_path, to_path+'/'+filename)

def save_tensor_map(tensor_map, output_folder):
    metadata = {'format': 'pt'}
    by_filename = {}

    for key, value in tensor_map.items():
        filename = value["filename"]
        tensor = value["tensor"]
        if filename not in by_filename:
            by_filename[filename] = {}
        by_filename[filename][key] = tensor

    for filename in sorted(by_filename.keys()):
        output_file = output_folder+'/'+filename
        print("Saving:", output_file)
        save_file(by_filename[filename], output_file, metadata=metadata)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge two safetensor model files.')
    parser.add_argument('base_model', type=str, help='The base model safetensor file')
    parser.add_argument('second_model', type=str, help='The second model safetensor file')
    parser.add_argument('output_model', type=str, help='The output merged model safetensor file')
    parser.add_argument('-p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('-lambda', dest='lambda_val', type=float, default=1.0, help='Scaling factor for the weight delta')
    args = parser.parse_args()

    if os.path.isdir(args.base_model):
        if not os.path.exists(args.output_model):
            os.makedirs(args.output_model)

        tensor_map = map_tensors_to_files(args.base_model)
        tensor_map = merge_folder(tensor_map, args.second_model, args.p, args.lambda_val)
        copy_nontensor_files(args.base_model, args.output_model)
        save_tensor_map(tensor_map, args.output_model)
    else:
        merged = merge_safetensors(args.base_model, args.second_model, args.p, args.lambda_val)
        save_file(merged, args.output_model)

if __name__ == '__main__':
    main()
