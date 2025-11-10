import os
import json
from dataset.load_dataset import load_dataset, load_dataset_split
import random


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_hookfree_completions(cfg, folder, model_base, dataset_name, aux_name, dataset=None, return_completions=False):
    """Always generate and save completions, appending (N) if the file already exists."""

    folder_path = os.path.join(cfg.artifact_path(), folder)
    base_file_name = f'{aux_name}_{dataset_name}_completions.json'
    file_path = os.path.join(folder_path, base_file_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('Created folder:', folder_path)

    # Find next available file name
    counter = 1
    final_file_path = file_path
    while os.path.exists(final_file_path):
        name_part = f'{aux_name}_{dataset_name}_completions'
        final_file_path = os.path.join(folder_path, f'{name_part}({counter}).json')
        counter += 1

    # Load dataset if not provided
    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_hookfree_completions(dataset, max_new_tokens=cfg.max_new_tokens)

    with open(final_file_path, "w") as f:
        json.dump(completions, f, indent=4)

    print(f"Completion saved at {final_file_path}")

    if return_completions:
        return completions

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, aux_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
    with open(f'{cfg.artifact_path()}/completions/{aux_name}_{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)
