from time import sleep
from tqdm import tqdm
import argparse
import requests
import git
import merge
import os
import shutil
import sys
import torch
import yaml

from normalize import normalize_tensor_map

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge HuggingFace models")
    parser.add_argument('model_list', type=str, help='File containing list of models to merge, supports yaml or txt')
    parser.add_argument('output_dir', type=str, help='Directory for the merged models')
    parser.add_argument('-base_model_path', type=str, default='staging/base_model', help='Base model directory')
    parser.add_argument('-staging_model_path', type=str, default='staging/merge_model', help='Staging model directory')
    parser.add_argument('-p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('-lambda', dest='lambda_val', type=float, default=1.0, help='Scaling factor for the weight delta')
    parser.add_argument('-seed', dest='seed', type=int, default=None, help='Random seed')
    parser.add_argument('-norm', default='none', type=str, choices=["none", "spectral"], help='Type of normalization to use')
    return parser.parse_args()

def load_models_list(file_path, default_lambda, default_norm, default_p, default_density = 1):
    models = []
    _, file_extension = os.path.splitext(file_path)

    # Branching based on file extension
    with open(file_path, 'r') as file:
        if file_extension.lower() == '.yaml' or file_extension.lower() == ".yml":
            data = yaml.safe_load(file)
        else:
            data = {"models": [{"model":model} for model in file.readlines()]}
    default_p = data.get("p", default_p)
    default_lambda = data.get("lambda", default_lambda)
    default_norm = data.get("norm", default_norm)

    for model_info in data['models']:
        model_name = model_info['model']
        p = model_info.get('parameters', {}).get('weight', default_p)
        density = model_info.get('parameters', {}).get('density', default_density)
        models.append((model_name.strip(), p, density))

    return models, default_lambda, default_norm

def reset_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Directory {directory} deleted successfully.")

def do_merge(source_map, tensor_map, staging_path, p, lambda_val):
    print(f"Merge operation for {staging_path}")
    tensor_map = merge.merge_folder(source_map, tensor_map, staging_path, p, lambda_val, diff_only=True)
    print("Merge operation completed successfully.")
    return tensor_map

def download_repo(repo_name, path):
    print(f"Repository {repo_name} cloning.")
    git.Repo.clone_from(f"https://huggingface.co/{repo_name}", path, depth=1)
    print(f"Repository {repo_name} cloned successfully.")

def should_create_symlink(repo_name):
    if os.path.exists(repo_name):
        return True, os.path.isfile(repo_name)
    return False, False

def download_or_link_repo(repo_name, path):
    symlink, is_file = should_create_symlink(repo_name)

    if symlink and is_file:
        os.makedirs(path, exist_ok=True)
        symlink_path = os.path.join(path, os.path.basename(repo_name))
        os.symlink(repo_name, symlink_path)
    elif symlink:
        os.symlink(repo_name, path)
    else:
        download_repo(repo_name, path)

def delete_repo(path):
    try:
        shutil.rmtree(path)
        print(f"Repository at {path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting repository at {path}: {e}")

def get_max_vocab_size(models):
    max_vocab_size = 0
    repo_with_max_vocab = None

    for repo in models:
        #TODO local
        repo_name = repo[0]
        url = f"https://huggingface.co/{repo_name}/raw/main/config.json"

        try:
            response = requests.get(url)
            response.raise_for_status()
            config = response.json()
            vocab_size = config.get("vocab_size", 0)

            if vocab_size > max_vocab_size:
                max_vocab_size = vocab_size
                repo_with_max_vocab = repo_name

        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")

    return max_vocab_size, repo_with_max_vocab

def download_json_files(repo_name, file_paths, output_dir):
    base_url = f"https://huggingface.co/{repo_name}/raw/main/"

    for file_path in file_paths:
        url = base_url + file_path
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(output_dir, os.path.basename(file_path)), 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download {file_path}")

def process_repos(output_dir, base_model_path, staging_model_path, models, lambda_val, norm):
    # Check if output_dir exists
    if os.path.exists(output_dir):
        sys.exit(f"Output directory '{output_dir}' already exists. Exiting to prevent data loss.")

    # Reset base and staging directories
    reset_directories([base_model_path, staging_model_path])

    # Make sure staging and output directories exist
    os.makedirs(base_model_path, exist_ok=True)
    os.makedirs(staging_model_path, exist_ok=True)

    # Initial download for 'base_model_path'
    download_or_link_repo(models[0][0], base_model_path)
    base_tensor_map = merge.map_tensors_to_files(base_model_path)
    tensor_map = {}
    for key in base_tensor_map.keys():
        tensor_map[key] = {"tensor": torch.zeros_like(base_tensor_map[key]['tensor']), "filename": base_tensor_map[key]['filename']}

    # Merge models
    for i, repo in enumerate(tqdm(models[1:], desc='Merging Repos')):
        repo_name, repo_p, repo_density = repo
        delete_repo(staging_model_path)
        download_or_link_repo(repo_name, staging_model_path)
        tensor_map = do_merge(base_tensor_map, tensor_map, staging_model_path, repo_p, 1/repo_density)

    os.makedirs(output_dir, exist_ok=True)
    merge.copy_nontensor_files(base_model_path, output_dir)

    # Handle LLMs that add tokens by taking the largest
    if os.path.exists(os.path.join(output_dir, 'config.json')):
        max_vocab_size, repo_name = get_max_vocab_size(models)
        if max_vocab_size > 0:
            file_paths = ['config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']
            download_json_files(repo_name, file_paths, output_dir)

    print("Normalizing...")
    normalize_tensor_map(tensor_map, norm)
    for key in tensor_map:
        tensor1, tensor2 = merge.resize_tensors(base_tensor_map[key]['tensor'], tensor_map[key]['tensor'])
        tensor_map[key]['tensor'] = tensor1 + lambda_val * tensor2
    print("Normalize complete")

    #reset_directories([base_model, staging_model_path])
    merge.save_tensor_map(tensor_map, output_dir)

if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    models, default_lambda, default_norm = load_models_list(args.model_list, args.lambda_val, args.norm, args.p)
    process_repos(args.output_dir, args.base_model_path, args.staging_model_path, models, default_lambda, default_norm)
