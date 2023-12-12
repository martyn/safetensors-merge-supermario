import argparse
import numpy as np
import os
import shutil
import torch
from safetensors.torch import safe_open, save_file

def merge_tensors(tensor1, tensor2, p):
    # Calculate the delta of the weights
    delta = tensor2 - tensor1
    # Generate the mask m^t from Bernoulli distribution
    m = torch.from_numpy(np.random.binomial(1, p, delta.shape)).to(tensor1.dtype).to(tensor1.device)
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
            merged_tensors[key] = tensor1 + lambda_val * merge_tensors(tensor1, tensor2, p)
            print("merging", key)

    return merged_tensors

def merge_folder(tensor_map, directory_path, p, lambda_val, extension=".safetensors"):
    keys1 = set(tensor_map.keys())
    for filename in os.listdir(directory_path):
        if filename.endswith(extension):
            file_path = os.path.join(directory_path, filename)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys2 = set(f.keys())
                common_keys = keys1.intersection(keys2)
                for key in common_keys:
                    tensor1 = tensor_map[key]['tensor']
                    tensor2 = f.get_tensor(key)
                    print("merging", key)
                    tensor_map[key]['tensor'] = tensor1 + lambda_val * merge_tensors(tensor1, tensor2, p)
    return tensor_map

def map_tensors_to_files(directory_path, output_path, extension=".safetensors"):
    tensor_map = {}

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(extension):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for key in keys:
                    tensor = f.get_tensor(key)
                    tensor_map[key] = {'filename':filename, 'shape':tensor.shape, 'tensor': tensor}
        else:
            shutil.copyfile(file_path, output_path+'/'+filename)

    return tensor_map

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

        tensor_map = map_tensors_to_files(args.base_model, args.output_model)
        tensor_map = merge_folder(tensor_map, args.second_model, args.p, args.lambda_val)
        save_tensor_map(tensor_map, args.output_model)
    else:
        merged = merge_safetensors(args.base_model, args.second_model, args.p, args.lambda_val)
        save_file(merged, args.output_model)

if __name__ == '__main__':
    main()
