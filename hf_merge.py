from time import sleep
import argparse
import git
import merge
import os
import shutil
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge HuggingFace models")
    parser.add_argument('repo_list', type=str, help='File containing list of repositories to merge')
    parser.add_argument('output_dir', type=str, help='Directory for the merged models')
    parser.add_argument('-base_model', type=str, default='staging/base_model', help='Base model directory')
    parser.add_argument('-staging_model', type=str, default='staging/merge_model', help='Staging model directory')
    parser.add_argument('-p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('-lambda', dest='lambda_val', type=float, default=1.0, help='Scaling factor for the weight delta')
    parser.add_argument('--dry', action='store_true', help='Run in dry mode without making any changes')
    return parser.parse_args()

def reset_directories(directories, dry_run):
    for directory in directories:
        if os.path.exists(directory):
            if dry_run:
                print(f"[DRY RUN] Would delete directory {directory}")
            else:
                shutil.rmtree(directory)
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


def download_repo(repo_name, path, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would download repository {repo_name} to {path}")
    else:
        print(f"Repository {repo_name} cloning.")
        git.Repo.clone_from(f"https://huggingface.co/{repo_name}", path, depth=1)
        print(f"Repository {repo_name} cloned successfully.")

def delete_repo(path, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would delete repository at {path}")
    else:
        try:
            shutil.rmtree(path)
            print(f"Repository at {path} deleted successfully.")
        except Exception as e:
            print(f"Error deleting repository at {path}: {e}")


def process_repos(output_dir, base_model, staging_model, repo_list_file, p, lambda_val, dry_run=False):
    # Check if output_dir exists
    if os.path.exists(output_dir):
        sys.exit(f"Output directory '{output_dir}' already exists. Exiting to prevent data loss.")

    # Reset base and staging directories
    reset_directories([base_model, staging_model], dry_run)

    # Make sure staging and output directories exist
    os.makedirs(base_model, exist_ok=True)
    os.makedirs(staging_model, exist_ok=True)

    repos_to_process = []
    with open(repo_list_file, "r") as file:
        repos_to_process = file.readlines()

    # Initial download for 'base_model'
    download_repo(repos_to_process[0].strip(), base_model, dry_run)
    tensor_map = merge.map_tensors_to_files(base_model)

    for i, repo in enumerate(repos_to_process[1:]):
        repo_name = repo.strip()
        delete_repo(staging_model, dry_run)
        download_repo(repo_name, staging_model, dry_run)
        do_merge(tensor_map, staging_model, p, lambda_val, dry_run)


    os.makedirs(output_dir, exist_ok=True)
    merge.copy_nontensor_files(base_model, output_dir)
    reset_directories([base_model, staging_model], dry_run)
    merge.save_tensor_map(tensor_map, output_dir)

if __name__ == "__main__":
    args = parse_arguments()
    process_repos(args.output_dir, args.base_model, args.staging_model, args.repo_list, args.p, args.lambda_val, args.dry)

