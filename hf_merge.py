from pathlib import Path
from time import sleep
from tqdm import tqdm
import argparse
import requests
import git
import merge
import os
import shutil
import sys
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge HuggingFace models")
    parser.add_argument('repo_list', type=str, help='File containing list of repositories to merge, supports mergekit yaml or txt')
    parser.add_argument('output_dir', type=str, help='Directory for the merged models')
    parser.add_argument('-staging', type=str, default='./staging', help='Path to staging folder')
    parser.add_argument('-p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('-lambda', dest='lambda_val', type=float, default=1.0, help='Scaling factor for the weight delta')
    parser.add_argument('--dry', action='store_true', help='Run in dry mode without making any changes')
    return parser.parse_args()

def repo_list_generator(file_path, default_p, default_lambda_val):
    _, file_extension = os.path.splitext(file_path)

    # Branching based on file extension
    if file_extension.lower() == '.yaml' or file_extension.lower() == ".yml":
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        for model_info in data['models']:
            model_name = model_info['model']
            p = model_info.get('parameters', {}).get('weight', default_p)
            lambda_val = 1 / model_info.get('parameters', {}).get('density', default_lambda_val)
            yield model_name, p, lambda_val

    else:  # Defaulting to txt file processing
        with open(file_path, "r") as file:
            repos_to_process = file.readlines()

        for repo in repos_to_process:
            yield repo.strip(), default_p, default_lambda_val

def reset_directories(directories, dry_run):
    for directory in directories:
        if os.path.exists(directory):
            if dry_run:
                print(f"[DRY RUN] Would delete directory {directory}")
            else:
                # Check if the directory is a symlink
                if os.path.islink(directory):
                    os.unlink(directory)  # Remove the symlink
                else:
                    shutil.rmtree(directory, ignore_errors=False)
                print(f"Directory {directory} deleted successfully.")

def do_merge(tensor_map, staging_path, p, lambda_val, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would merge with {staging_path}")
    else:
        try:
            print(f"Merge operation for {staging_path}")
            tensor_map = merge.merge_folder(tensor_map, staging_path, p, lambda_val)
            print("Merge operation completed successfully.")
        except Exception as e:
            print(f"Error during merge operation: {e}")
    return tensor_map

def download_repo(repo_name, path, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would download repository {repo_name} to {path}")
    else:
        print(f"Repository {repo_name} cloning.")
        git.Repo.clone_from(f"https://huggingface.co/{repo_name}", path, depth=1)
        print(f"Repository {repo_name} cloned successfully.")

def should_create_symlink(repo_name):
    if os.path.exists(repo_name):
        return True, os.path.isfile(repo_name)
    return False, False

def download_or_link_repo(repo_name, path, dry_run=False):
    symlink, is_file = should_create_symlink(repo_name)

    if symlink and is_file:
        os.makedirs(path, exist_ok=True)
        symlink_path = os.path.join(path, os.path.basename(repo_name))
        os.symlink(repo_name, symlink_path)
    elif symlink:
        os.symlink(repo_name, path)
    else:
        download_repo(repo_name, path, dry_run)

def delete_repo(path, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would delete repository at {path}")
    else:
        try:
            shutil.rmtree(path)
            print(f"Repository at {path} deleted successfully.")
        except Exception as e:
            print(f"Error deleting repository at {path}: {e}")

def get_max_vocab_size(repo_list):
    max_vocab_size = 0
    repo_with_max_vocab = None

    for repo in repo_list:
        repo_name = repo[0].strip()
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

def process_repos(output_dir, base_model, staging_model, repo_list_file, p, lambda_val, dry_run=False):
    # Check if output_dir exists
    if os.path.exists(output_dir):
        sys.exit(f"Output directory '{output_dir}' already exists. Exiting to prevent data loss.")

    # Reset base and staging directories
    reset_directories([base_model, staging_model], dry_run)

    repo_list_gen = repo_list_generator(repo_list_file, p, lambda_val)

    repos_to_process = list(repo_list_gen)

    # Initial download for 'base_model'
    download_or_link_repo(repos_to_process[0][0].strip(), base_model, dry_run)
    tensor_map = merge.map_tensors_to_files(base_model)

    for i, repo in enumerate(tqdm(repos_to_process[1:], desc='Merging Repos')):
        repo_name = repo[0].strip()
        repo_p = repo[1]
        repo_lambda = repo[2]
        delete_repo(staging_model, dry_run)
        download_or_link_repo(repo_name, staging_model, dry_run)
        tensor_map = do_merge(tensor_map, staging_model, repo_p, repo_lambda, dry_run)

    os.makedirs(output_dir, exist_ok=True)
    merge.copy_nontensor_files(base_model, output_dir)

    # Handle LLMs that add tokens by taking the largest
    if os.path.exists(os.path.join(output_dir, 'config.json')):
        max_vocab_size, repo_name = get_max_vocab_size(repos_to_process)
        if max_vocab_size > 0:
            file_paths = ['config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']
            download_json_files(repo_name, file_paths, output_dir)

    reset_directories([base_model, staging_model], dry_run)
    merge.save_tensor_map(tensor_map, output_dir)

if __name__ == "__main__":
    args = parse_arguments()
    staging_path = Path(args.staging)
    os.makedirs(args.staging, exist_ok=True)
    process_repos(args.output_dir, staging_path / 'base_model', staging_path / 'staging_model', args.repo_list, args.p, args.lambda_val, args.dry)

