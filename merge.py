import torch
import numpy as np
from safetensors.torch import safe_open, save_file

def merge_tensors(tensor1, tensor2, p):
    # Calculate the delta of the weights (weights0 - weights1)
    delta = tensor2 - tensor1
    # Generate the mask m^t from Bernoulli distribution
    m = torch.from_numpy(np.random.binomial(1, p, delta.shape)).to(tensor1.dtype).to(tensor1.device)
    # Apply the mask to the delta to get δ̃^t
    delta_tilde = m * delta
    # Scale the masked delta by the dropout rate to get δ̂^t
    delta_hat = delta_tilde / (1 - p)
    return delta_hat

def merge_safetensors(file_path1, file_path2, p):
    merged_tensors = {}
    lam = 3.0

    with safe_open(file_path1, framework="pt", device="cpu") as f1, safe_open(file_path2, framework="pt", device="cpu") as f2:
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())
        common_keys = keys1.intersection(keys2)

        for key in common_keys:
            tensor1 = f1.get_tensor(key)
            tensor2 = f2.get_tensor(key)
            merged_tensors[key] = tensor1 + lam * merge_tensors(tensor1, tensor2, p)
            print("merging", key)

    return merged_tensors

# Usage
file_path1 = 'sd_xl_turbo_1.0_fp16.safetensors'
file_path2 = 'sdxl.safetensors'
p = 0.1  # Dropout probability
merged = merge_safetensors(file_path1, file_path2, p)

# Save the merged tensors to a new safetensors file
output_file_path = 'merged_output.safetensors'
print(f"saving to {output_file_path}")
save_file(merged, output_file_path)

